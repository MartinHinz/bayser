from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from scipy.special import gammaln, logsumexp


# -----------------------------------------------------------------------------
# Basic numerical helpers
# -----------------------------------------------------------------------------


def spacing_repulsion(
    t: pt.TensorVariable,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    strength: float = 0.35,
    min_dist: float = 0.10,
) -> pt.TensorVariable:
    """Softly discourage graves from occupying identical latent positions.

    The pair indices are precomputed outside the PyMC graph. This avoids building
    a full n_graves x n_graves distance matrix and triangular mask inside the
    symbolic graph.
    """

    if strength <= 0:
        return pt.as_tensor_variable(0.0)

    d2 = (t[pair_i] - t[pair_j]) ** 2 + 1e-6

    return -float(strength) * pt.sum(
        pt.exp(-d2 / (2.0 * float(min_dist) ** 2))
    )


def zscore_vector(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")

    sd = float(np.std(x))
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError(f"{name} must have non-zero variance.")

    return (x - float(np.mean(x))) / sd


def normal_logpdf_pt(x, mu, sigma):
    return (
        -0.5 * pt.log(2.0 * np.pi)
        - pt.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )


def student_t_logpdf_pt(x, mu, sigma, nu: float = 5.0):
    nu_t = pt.as_tensor_variable(float(nu))
    z = (x - mu) / sigma

    return (
        pt.gammaln((nu_t + 1.0) / 2.0)
        - pt.gammaln(nu_t / 2.0)
        - 0.5 * pt.log(nu_t * np.pi)
        - pt.log(sigma)
        - ((nu_t + 1.0) / 2.0) * pt.log1p((z**2) / nu_t)
    )


def normal_logpdf_np(x, mu, sigma):
    return (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )


def student_t_logpdf_np(x, mu, sigma, nu: float = 5.0):
    z = (x - mu) / sigma

    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
        - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu)
    )


def softmax_np(log_w: np.ndarray, axis: int = -1) -> np.ndarray:
    w = np.exp(log_w - np.max(log_w, axis=axis, keepdims=True))
    return w / np.sum(w, axis=axis, keepdims=True)


# -----------------------------------------------------------------------------
# C14 preparation and prior-scale helper
# -----------------------------------------------------------------------------


def infer_cal_span_prior_from_reference_calibration(
    row_scores: np.ndarray,
    unmodelled_cal_bp: np.ndarray,
    min_n: int = 8,
    min_span: float = 20.0,
    max_span: float = 160.0,
    min_sigma: float = 20.0,
    max_sigma: float = 80.0,
) -> tuple[float, float]:
    x = np.asarray(row_scores, dtype=float)
    y = np.asarray(unmodelled_cal_bp, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) < min_n or np.std(x) <= 0:
        return 60.0, 40.0

    x = zscore_vector(x, "row_scores")

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    keep = (
        np.triu(np.ones(dx.shape, dtype=bool), 1)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & (np.abs(dx) > 1e-8)
    )

    slopes = dy[keep] / dx[keep]
    slopes = slopes[np.isfinite(slopes)]

    if len(slopes) == 0:
        return 60.0, 40.0

    span_mu = float(np.clip(abs(np.median(slopes)), min_span, max_span))

    q25, q75 = np.quantile(np.abs(slopes), [0.25, 0.75])
    span_sigma = max(min_sigma, 0.5 * span_mu, abs(q75 - q25))
    span_sigma = float(np.clip(span_sigma, min_sigma, max_sigma))

    return span_mu, span_sigma


def empty_c14_inputs():
    return (
        np.array([], dtype=int),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        None,
        None,
    )


def validate_optional_vector(x, n: int, name: str):
    if x is None:
        return None

    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or len(x) != n or np.any(np.isinf(x)):
        raise ValueError(
            f"{name} must be a one-dimensional vector of length {n}, "
            "with no infinite values."
        )

    return x


def calendar_grid_weights(cal_grid: np.ndarray) -> np.ndarray:
    cal_grid = np.asarray(cal_grid, dtype=float)

    if len(cal_grid) < 3 or np.any(np.diff(cal_grid) <= 0):
        raise ValueError(
            "Calendar grid must be strictly increasing and contain at least three points."
        )

    w = np.empty_like(cal_grid, dtype=float)
    w[1:-1] = 0.5 * (cal_grid[2:] - cal_grid[:-2])
    w[0] = cal_grid[1] - cal_grid[0]
    w[-1] = cal_grid[-1] - cal_grid[-2]

    return w


def thin_calendar_grid(x: np.ndarray, *arrays: np.ndarray, step: int = 1):
    if step < 1:
        raise ValueError("calendar_grid_step must be >= 1.")

    if step == 1:
        return (x, *arrays)

    idx = np.arange(0, len(x), step, dtype=int)

    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)

    return (x[idx], *(a[idx] for a in arrays))


def prepare_c14_inputs(
    c14_bp,
    c14_error,
    curve: pd.DataFrame | None,
    n_graves: int,
    padding: float,
    grid_step: int,
):
    bp = validate_optional_vector(c14_bp, n_graves, "c14_bp")
    err = validate_optional_vector(c14_error, n_graves, "c14_error")

    if bp is None and err is None:
        return empty_c14_inputs()

    if bp is None or err is None:
        raise ValueError("c14_bp and c14_error must either both be supplied or both be None.")

    finite = np.isfinite(bp) & np.isfinite(err)

    if np.any(np.isfinite(bp) != np.isfinite(err)):
        raise ValueError("c14_bp and c14_error must be missing in the same rows.")

    if np.any(err[finite] <= 0):
        raise ValueError("Finite c14_error values must be positive.")

    c14_index = np.where(finite)[0].astype(int)
    bp = bp[c14_index]
    err = err[c14_index]

    if len(c14_index) == 0:
        return c14_index, bp, err, *empty_c14_inputs()[3:]

    if curve is None:
        raise ValueError("At least one finite C14 date requires intcal20_curve.")

    missing = {"cal_bp", "c14_age", "c14_sigma"}.difference(curve.columns)
    if missing:
        raise ValueError("IntCal20 curve is missing columns: " + ", ".join(sorted(missing)))

    curve = curve.sort_values("cal_bp")

    cal_full = curve["cal_bp"].to_numpy(float)
    age_full = curve["c14_age"].to_numpy(float)
    sig_full = curve["c14_sigma"].to_numpy(float)

    if np.any(np.diff(cal_full) <= 0) or not all(
        np.all(np.isfinite(a)) for a in [cal_full, age_full, sig_full]
    ):
        raise ValueError("Invalid IntCal20 curve arrays.")

    rough = bp + 300.0

    cal_lo = max(float(cal_full.min()), float(np.nanmin(rough) - padding))
    cal_hi = min(float(cal_full.max()), float(np.nanmax(rough) + padding))

    keep = (cal_full >= cal_lo) & (cal_full <= cal_hi)

    cal_grid, curve_age, curve_sigma = thin_calendar_grid(
        cal_full[keep],
        age_full[keep],
        sig_full[keep],
        step=grid_step,
    )

    if len(cal_grid) < 3:
        raise ValueError("Local IntCal20 grid is too small.")

    return (
        c14_index,
        bp,
        err,
        cal_grid,
        curve_age,
        curve_sigma,
        calendar_grid_weights(cal_grid),
        cal_lo,
        cal_hi,
    )


# -----------------------------------------------------------------------------
# Marginalised IntCal20 likelihood
# -----------------------------------------------------------------------------


def precompute_c14_loglikelihood_np(
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    c14_extra_sigma: float = 0.0,
) -> np.ndarray:
    """Precompute log p(C14 observation | calendar grid).

    This matrix is constant with respect to the MCMC parameters. It has shape
    (n_dated, n_calendar_grid) and can be passed into the PyTensor graph as a
    fixed tensor.
    """

    observed_c14_bp = np.asarray(observed_c14_bp, dtype=float)
    observed_c14_error = np.asarray(observed_c14_error, dtype=float)
    c14_curve_age = np.asarray(c14_curve_age, dtype=float)
    c14_curve_sigma = np.asarray(c14_curve_sigma, dtype=float)

    if observed_c14_bp.ndim != 1:
        raise ValueError("observed_c14_bp must be one-dimensional.")

    if observed_c14_error.ndim != 1:
        raise ValueError("observed_c14_error must be one-dimensional.")

    if len(observed_c14_bp) != len(observed_c14_error):
        raise ValueError("observed_c14_bp and observed_c14_error must have equal length.")

    if c14_curve_age.ndim != 1 or c14_curve_sigma.ndim != 1:
        raise ValueError("c14_curve_age and c14_curve_sigma must be one-dimensional.")

    if len(c14_curve_age) != len(c14_curve_sigma):
        raise ValueError("c14_curve_age and c14_curve_sigma must have equal length.")

    if np.any(observed_c14_error <= 0):
        raise ValueError("observed_c14_error must be positive.")

    total_sigma = np.sqrt(
        observed_c14_error[:, None] ** 2
        + c14_curve_sigma[None, :] ** 2
        + float(c14_extra_sigma) ** 2
    )

    return normal_logpdf_np(
        observed_c14_bp[:, None],
        c14_curve_age[None, :],
        total_sigma,
    )


def marginal_c14_loglike(
    bp,
    err,
    calendar_logprior,
    curve_age,
    curve_sigma,
    weights,
    extra_sigma: float = 0.0,
    log_c14_given_cal: np.ndarray | None = None,
):
    if log_c14_given_cal is None:
        bp = pt.as_tensor_variable(np.asarray(bp, dtype=float))
        err = pt.as_tensor_variable(np.asarray(err, dtype=float))
        age = pt.as_tensor_variable(np.asarray(curve_age, dtype=float))
        sig = pt.as_tensor_variable(np.asarray(curve_sigma, dtype=float))

        total_sigma = pt.sqrt(
            err[:, None] ** 2
            + sig[None, :] ** 2
            + float(extra_sigma) ** 2
        )

        log_c14 = normal_logpdf_pt(bp[:, None], age[None, :], total_sigma)
    else:
        log_c14 = pt.as_tensor_variable(np.asarray(log_c14_given_cal, dtype=float))

    log_w = pt.log(pt.as_tensor_variable(np.asarray(weights, dtype=float)))

    log_num = pt.logsumexp(
        log_c14 + calendar_logprior + log_w[None, :],
        axis=1,
    )
    log_den = pt.logsumexp(
        calendar_logprior + log_w[None, :],
        axis=1,
    )

    return log_num - log_den


def marginalised_intcal20_logp(
    observed_c14_bp,
    observed_c14_error,
    expected_cal_bp,
    sigma_cal_link,
    cal_grid,
    c14_curve_age,
    c14_curve_sigma,
    grid_weights,
    c14_extra_sigma: float = 0.0,
    outlier_prior: np.ndarray | None = None,
    outlier_scale: float = 5.0,
    outlier_nu: float = 5.0,
    log_c14_given_cal: np.ndarray | None = None,
):
    """Marginalised IntCal20 likelihood with optional per-date outlier mixture.

    Default behaviour is the regular linked chronology model.

    If outlier_prior is supplied and contains values > 0, only those dated
    observations are evaluated as a regular/outlier mixture. Dates with prior == 0
    remain pure regular observations.

    This follows an OxCal-like logic: the outlier model is not a global feature
    that is always sampled, but an explicit check for observations that have been
    flagged as potentially problematic.
    """

    observed_c14_bp_np = np.asarray(observed_c14_bp, dtype=float)
    observed_c14_error_np = np.asarray(observed_c14_error, dtype=float)
    cal = pt.as_tensor_variable(np.asarray(cal_grid, dtype=float))

    n_dated = len(observed_c14_bp_np)

    log_cal_regular = normal_logpdf_pt(
        cal[None, :],
        expected_cal_bp[:, None],
        sigma_cal_link,
    )

    log_like_regular = marginal_c14_loglike(
        observed_c14_bp_np,
        observed_c14_error_np,
        log_cal_regular,
        c14_curve_age,
        c14_curve_sigma,
        grid_weights,
        c14_extra_sigma,
        log_c14_given_cal=log_c14_given_cal,
    )

    if outlier_prior is None:
        return pt.sum(log_like_regular), None, log_like_regular, None

    outlier_prior_np = np.asarray(outlier_prior, dtype=float)

    if outlier_prior_np.ndim != 1:
        raise ValueError("outlier_prior must be one-dimensional.")

    if len(outlier_prior_np) != n_dated:
        raise ValueError(
            "outlier_prior must have one value per dated C14 observation."
        )

    if np.any(~np.isfinite(outlier_prior_np)):
        raise ValueError("outlier_prior must contain only finite values.")

    if np.any((outlier_prior_np < 0.0) | (outlier_prior_np > 1.0)):
        raise ValueError("outlier_prior values must lie between 0 and 1.")

    active_idx = np.where(outlier_prior_np > 0.0)[0].astype(int)

    if len(active_idx) == 0:
        return pt.sum(log_like_regular), None, log_like_regular, None

    if outlier_scale <= 1.0:
        raise ValueError("outlier_scale should be > 1.")

    if outlier_nu <= 0.0:
        raise ValueError("outlier_nu must be positive.")

    p_active_np = np.clip(outlier_prior_np[active_idx], 1e-12, 1.0 - 1e-12)
    p_active = pt.as_tensor_variable(p_active_np)

    expected_active = expected_cal_bp[active_idx]
    log_cal_outlier_active = student_t_logpdf_pt(
        cal[None, :],
        expected_active[:, None],
        sigma_cal_link * float(outlier_scale),
        nu=float(outlier_nu),
    )

    if log_c14_given_cal is None:
        log_c14_active = None
    else:
        log_c14_active = np.asarray(log_c14_given_cal, dtype=float)[active_idx, :]

    log_like_outlier_active = marginal_c14_loglike(
        observed_c14_bp_np[active_idx],
        observed_c14_error_np[active_idx],
        log_cal_outlier_active,
        c14_curve_age,
        c14_curve_sigma,
        grid_weights,
        c14_extra_sigma,
        log_c14_given_cal=log_c14_active,
    )

    log_like_regular_active = log_like_regular[active_idx]

    log_regular = pt.log1p(-p_active) + log_like_regular_active
    log_outlier = pt.log(p_active) + log_like_outlier_active
    log_mix_active = pt.logaddexp(log_regular, log_outlier)

    # Start with the regular likelihood for all dated observations, then replace
    # only the explicitly flagged observations by their mixture contribution.
    logp = (
        pt.sum(log_like_regular)
        + pt.sum(log_mix_active - log_like_regular_active)
    )

    p_outlier_active = pt.exp(log_outlier - log_mix_active)

    p_outlier = pt.zeros_like(log_like_regular)
    p_outlier = pt.set_subtensor(p_outlier[active_idx], p_outlier_active)

    log_like_outlier = pt.zeros_like(log_like_regular)
    log_like_outlier = pt.set_subtensor(
        log_like_outlier[active_idx],
        log_like_outlier_active,
    )

    return logp, p_outlier, log_like_regular, log_like_outlier


# -----------------------------------------------------------------------------
# Post-hoc calendar reconstruction
# -----------------------------------------------------------------------------


def reconstruct_marginal_calendar_draws(
    idata: az.InferenceData,
    c14_index: np.ndarray,
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    cal_grid: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    grid_weights: np.ndarray,
    c14_extra_sigma: float = 0.0,
    random_seed: int = 123,
    chunk_size: int = 250,
    outlier_prior: np.ndarray | None = None,
    outlier_scale: float = 5.0,
    outlier_nu: float = 5.0,
    log_c14_given_cal: np.ndarray | None = None,
) -> az.InferenceData:
    """Post-hoc reconstruction of marginal calendar-age draws.

    If outlier_prior contains values > 0, the reconstructed calendar posterior
    uses the same regular/outlier mixture as the fitted model. The posterior
    outlier probability is stored as p_outlier_reconstructed.

    The outlier branch is evaluated only for actively flagged observations.
    """

    c14_index = np.asarray(c14_index, dtype=int)

    if len(c14_index) == 0:
        return idata

    expected = idata.posterior["expected_cal_bp"].values
    sigma_link = idata.posterior["sigma_cal_link"].values

    n_chain, n_draw, n_grave = expected.shape

    expected_flat = expected.reshape(-1, n_grave)
    sigma_flat = sigma_link.reshape(-1)

    n_sample = expected_flat.shape[0]
    n_dated = len(c14_index)

    observed_c14_bp_np = np.asarray(observed_c14_bp, dtype=float)
    observed_c14_error_np = np.asarray(observed_c14_error, dtype=float)

    if outlier_prior is None:
        outlier_prior_np = np.zeros(n_dated, dtype=float)
    else:
        outlier_prior_np = np.asarray(outlier_prior, dtype=float)

        if outlier_prior_np.ndim != 1 or len(outlier_prior_np) != n_dated:
            raise ValueError(
                "outlier_prior must be a one-dimensional vector with one value "
                "per dated C14 observation."
            )

        if np.any(~np.isfinite(outlier_prior_np)):
            raise ValueError("outlier_prior must contain only finite values.")

        if np.any((outlier_prior_np < 0.0) | (outlier_prior_np > 1.0)):
            raise ValueError("outlier_prior values must lie between 0 and 1.")

    active_idx = np.where(outlier_prior_np > 0.0)[0].astype(int)
    use_outlier_model = bool(len(active_idx) > 0)

    if use_outlier_model:
        if outlier_scale <= 1.0:
            raise ValueError("outlier_scale should be > 1.")

        if outlier_nu <= 0.0:
            raise ValueError("outlier_nu must be positive.")

    rng = np.random.default_rng(random_seed)

    reconstructed = rng.normal(
        expected_flat,
        sigma_flat[:, None],
        size=(n_sample, n_grave),
    )
    cond_mean = expected_flat.copy()
    cond_sd = np.broadcast_to(sigma_flat[:, None], (n_sample, n_grave)).copy()
    cond_p = np.full((n_sample, n_grave), np.nan, dtype=float)

    if log_c14_given_cal is None:
        log_c14 = precompute_c14_loglikelihood_np(
            observed_c14_bp_np,
            observed_c14_error_np,
            c14_curve_age,
            c14_curve_sigma,
            c14_extra_sigma,
        )
    else:
        log_c14 = np.asarray(log_c14_given_cal, dtype=float)

    log_grid_w = np.log(grid_weights)

    rec_d = np.empty((n_sample, n_dated), dtype=float)
    mean_d = np.empty((n_sample, n_dated), dtype=float)
    sd_d = np.empty((n_sample, n_dated), dtype=float)
    p_d = np.zeros((n_sample, n_dated), dtype=float) if use_outlier_model else None

    p_active = np.clip(outlier_prior_np[active_idx], 1e-12, 1.0 - 1e-12)

    for start in range(0, n_sample, chunk_size):
        stop = min(start + chunk_size, n_sample)

        exp_chunk = expected_flat[start:stop, :][:, c14_index]
        sig_chunk = sigma_flat[start:stop]

        log_cal_reg = normal_logpdf_np(
            cal_grid[None, None, :],
            exp_chunk[:, :, None],
            sig_chunk[:, None, None],
        )

        log_w_reg = (
            log_c14[None, :, :]
            + log_cal_reg
            + log_grid_w[None, None, :]
        )

        log_den_reg = logsumexp(
            log_cal_reg + log_grid_w[None, None, :],
            axis=2,
        )

        log_grid_reg_norm = log_w_reg - log_den_reg[:, :, None]

        log_w = log_grid_reg_norm.copy()

        if use_outlier_model:
            assert p_d is not None

            exp_active = exp_chunk[:, active_idx]
            sig_out = sig_chunk * float(outlier_scale)

            log_cal_out_active = student_t_logpdf_np(
                cal_grid[None, None, :],
                exp_active[:, :, None],
                sig_out[:, None, None],
                nu=float(outlier_nu),
            )

            log_w_out_active = (
                log_c14[None, active_idx, :]
                + log_cal_out_active
                + log_grid_w[None, None, :]
            )

            log_den_out_active = logsumexp(
                log_cal_out_active + log_grid_w[None, None, :],
                axis=2,
            )

            log_grid_out_norm_active = (
                log_w_out_active - log_den_out_active[:, :, None]
            )

            log_grid_reg_norm_active = log_grid_reg_norm[:, active_idx, :]

            log_marg_reg_active = logsumexp(
                log_grid_reg_norm_active,
                axis=2,
            )
            log_marg_out_active = logsumexp(
                log_grid_out_norm_active,
                axis=2,
            )

            log_regular_comp = (
                np.log1p(-p_active)[None, :]
                + log_marg_reg_active
            )
            log_outlier_comp = (
                np.log(p_active)[None, :]
                + log_marg_out_active
            )
            log_mix_marg = np.logaddexp(
                log_regular_comp,
                log_outlier_comp,
            )

            p_d[start:stop, active_idx] = np.exp(
                log_outlier_comp - log_mix_marg
            )

            log_grid_mix_active = np.logaddexp(
                np.log1p(-p_active)[None, :, None]
                + log_grid_reg_norm_active,
                np.log(p_active)[None, :, None]
                + log_grid_out_norm_active,
            )

            log_w[:, active_idx, :] = log_grid_mix_active

        probs = softmax_np(log_w, axis=2)

        mean_d[start:stop, :] = np.sum(
            probs * cal_grid[None, None, :],
            axis=2,
        )

        sd_d[start:stop, :] = np.sqrt(
            np.maximum(
                np.sum(probs * cal_grid[None, None, :] ** 2, axis=2)
                - mean_d[start:stop, :] ** 2,
                0.0,
            )
        )

        cdf = np.cumsum(probs, axis=2)

        idx = np.sum(
            cdf < rng.random(size=(stop - start, n_dated, 1)),
            axis=2,
        )

        rec_d[start:stop, :] = cal_grid[np.clip(idx, 0, len(cal_grid) - 1)]

    reconstructed[:, c14_index] = rec_d
    cond_mean[:, c14_index] = mean_d
    cond_sd[:, c14_index] = sd_d

    if use_outlier_model and p_d is not None:
        cond_p[:, c14_index] = p_d

    coords = {k: idata.posterior.coords[k] for k in ["chain", "draw", "grave"]}
    shape = (n_chain, n_draw, n_grave)

    idata.posterior["latent_cal_bp"] = xr.DataArray(
        reconstructed.reshape(shape),
        dims=("chain", "draw", "grave"),
        coords=coords,
    )
    idata.posterior["latent_cal_bp_cond_mean"] = xr.DataArray(
        cond_mean.reshape(shape),
        dims=("chain", "draw", "grave"),
        coords=coords,
    )
    idata.posterior["latent_cal_bp_cond_sd"] = xr.DataArray(
        cond_sd.reshape(shape),
        dims=("chain", "draw", "grave"),
        coords=coords,
    )

    if use_outlier_model:
        idata.posterior["p_outlier_reconstructed"] = xr.DataArray(
            cond_p.reshape(shape),
            dims=("chain", "draw", "grave"),
            coords=coords,
        )

    return idata


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def store_model_attrs(idata: az.InferenceData, **settings) -> az.InferenceData:
    n_c14 = int(settings.get("n_c14_dated", 0) or 0)
    outlier = bool(settings.get("use_outlier_model", False))

    idata.attrs["model_name"] = "bayser"
    idata.attrs["model_version"] = (
        "hierarchical_typology_only"
        if n_c14 == 0
        else "hierarchical_typology_intcal20_outlier"
        if outlier
        else "hierarchical_typology_intcal20"
    )

    for key, value in settings.items():
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, (str, int, float, bool)):
            idata.attrs[key] = value

    return idata


# -----------------------------------------------------------------------------
# Backwards-compatible private aliases for the current model.py transition
# -----------------------------------------------------------------------------


_zscore = zscore_vector
_empty_c14 = empty_c14_inputs
_prepare_c14 = prepare_c14_inputs
_store_attrs = store_model_attrs