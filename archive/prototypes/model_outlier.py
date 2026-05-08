from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import logsumexp

# -----------------------------------------------------------------------------
# General helper functions
# -----------------------------------------------------------------------------


def spacing_repulsion(
    t: pt.TensorVariable,
    strength: float = 0.35,
    min_dist: float = 0.10,
) -> pt.TensorVariable:
    """Softly encourage grave coordinates to spread along the latent axis."""

    diffs = t[:, None] - t[None, :]
    dist2 = diffs**2 + 1e-6
    n = t.shape[0]
    mask = pt.triu(pt.ones((n, n)), k=1)

    return -strength * pt.sum(
        pt.exp(-dist2 / (2.0 * min_dist**2)) * mask
    )


def _standardise_reference(
    x: np.ndarray,
    name: str = "orientation_reference",
) -> np.ndarray:
    """Return a finite, centred, unit-scale reference vector."""

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")

    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{name} must have non-zero variance.")

    return (x - float(np.mean(x))) / sd


def _validate_vector(
    x: np.ndarray | None,
    n: int,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    """Validate a one-dimensional numeric vector of length n."""

    if x is None:
        raise ValueError(f"{name} is required.")

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if len(x) != n:
        raise ValueError(f"{name} must have length {n}.")

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")

    if positive and np.any(x <= 0):
        raise ValueError(f"{name} must contain only positive values.")

    return x


def _normal_logpdf(
    x: pt.TensorVariable,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
) -> pt.TensorVariable:
    """Elementwise Normal log-density for PyTensor variables."""

    return (
        -0.5 * pt.log(2.0 * np.pi)
        - pt.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )


def _student_t_logpdf(
    x: pt.TensorVariable,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
    nu: float = 5.0,
) -> pt.TensorVariable:
    """Elementwise Student-t log-density for PyTensor variables."""

    nu_t = pt.as_tensor_variable(float(nu))
    z = (x - mu) / sigma

    return (
        pt.gammaln((nu_t + 1.0) / 2.0)
        - pt.gammaln(nu_t / 2.0)
        - 0.5 * pt.log(nu_t * np.pi)
        - pt.log(sigma)
        - ((nu_t + 1.0) / 2.0) * pt.log1p((z**2) / nu_t)
    )


def _normal_logpdf_np(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Elementwise Normal log-density for NumPy arrays."""

    return (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )


def _student_t_logpdf_np(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float = 5.0,
) -> np.ndarray:
    """Elementwise Student-t log-density for NumPy arrays."""

    from scipy.special import gammaln

    nu = float(nu)
    z = (x - mu) / sigma

    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
        - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu)
    )


def _softmax_np(log_w: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""

    m = np.max(log_w, axis=axis, keepdims=True)
    w = np.exp(log_w - m)
    return w / np.sum(w, axis=axis, keepdims=True)


# -----------------------------------------------------------------------------
# Prior-scale helpers
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
    """Infer a weakly data-informed prior for calendar years per t-unit."""

    x = np.asarray(row_scores, dtype=float)
    y = np.asarray(unmodelled_cal_bp, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < min_n:
        return 60.0, 40.0

    x_sd = float(np.std(x))
    if not np.isfinite(x_sd) or x_sd <= 0:
        return 60.0, 40.0

    xz = (x - float(np.mean(x))) / x_sd

    dx = xz[:, None] - xz[None, :]
    dy = y[:, None] - y[None, :]

    upper = np.triu(np.ones(dx.shape, dtype=bool), k=1)
    valid = upper & np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-8)

    slopes = dy[valid] / dx[valid]
    slopes = slopes[np.isfinite(slopes)]

    if len(slopes) == 0:
        return 60.0, 40.0

    span_mu = float(np.clip(abs(np.median(slopes)), min_span, max_span))

    q25, q75 = np.quantile(np.abs(slopes), [0.25, 0.75])
    robust_width = float(abs(q75 - q25))

    span_sigma = max(min_sigma, 0.5 * span_mu, robust_width)
    span_sigma = float(np.clip(span_sigma, min_sigma, max_sigma))

    return span_mu, span_sigma


# -----------------------------------------------------------------------------
# IntCal20 preparation and marginalised C14 likelihood
# -----------------------------------------------------------------------------


def _trapezoid_grid_weights(x: np.ndarray) -> np.ndarray:
    """Return positive trapezoid-style integration weights for an ordered grid."""

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Grid must be one-dimensional.")

    if len(x) < 3:
        raise ValueError("Grid must contain at least three points.")

    if np.any(np.diff(x) <= 0):
        raise ValueError("Grid must be strictly increasing.")

    w = np.empty_like(x, dtype=float)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = x[1] - x[0]
    w[-1] = x[-1] - x[-2]

    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("Invalid integration weights.")

    return w


def _thin_ordered_grid(
    x: np.ndarray,
    *arrays: np.ndarray,
    step: int = 1,
) -> tuple[np.ndarray, ...]:
    """Thin an ordered grid and aligned arrays by keeping every `step`th point."""

    if step < 1:
        raise ValueError("calendar_grid_step must be >= 1.")

    x = np.asarray(x, dtype=float)
    arrays = tuple(np.asarray(a, dtype=float) for a in arrays)

    if step == 1:
        return (x, *arrays)

    if len(x) < 3:
        raise ValueError("Grid must contain at least three points before thinning.")

    for a in arrays:
        if len(a) != len(x):
            raise ValueError("All arrays must have the same length as the grid.")

    keep_idx = np.arange(0, len(x), step, dtype=int)

    if keep_idx[-1] != len(x) - 1:
        keep_idx = np.append(keep_idx, len(x) - 1)

    x_thin = x[keep_idx]
    arrays_thin = tuple(a[keep_idx] for a in arrays)

    if len(x_thin) < 3:
        raise ValueError(
            "Thinned calendar grid is too small. Use a smaller calendar_grid_step."
        )

    if np.any(np.diff(x_thin) <= 0):
        raise ValueError("Thinned calendar grid must be strictly increasing.")

    return (x_thin, *arrays_thin)


def _prepare_c14_inputs(
    c14_bp: np.ndarray | None,
    c14_error: np.ndarray | None,
    intcal20_curve: pd.DataFrame | None,
    n_graves: int,
    local_window_padding: float = 500.0,
    calendar_grid_step: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Validate and prepare C14 data and local IntCal20 curve arrays."""

    c14_bp = _validate_vector(c14_bp, n_graves, "c14_bp")
    c14_error = _validate_vector(c14_error, n_graves, "c14_error", positive=True)

    if intcal20_curve is None:
        raise ValueError(
            "chronology_likelihood='intcal20' requires intcal20_curve."
        )

    required_cols = {"cal_bp", "c14_age", "c14_sigma"}
    missing_cols = required_cols.difference(intcal20_curve.columns)
    if missing_cols:
        raise ValueError(
            "IntCal20 curve is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    curve = intcal20_curve.sort_values("cal_bp").reset_index(drop=True)

    cal_grid_full = curve["cal_bp"].to_numpy(dtype=float)
    c14_curve_age_full = curve["c14_age"].to_numpy(dtype=float)
    c14_curve_sigma_full = curve["c14_sigma"].to_numpy(dtype=float)

    if np.any(np.diff(cal_grid_full) <= 0):
        raise ValueError("IntCal20 cal_bp grid must be strictly increasing.")

    if not np.all(np.isfinite(cal_grid_full)):
        raise ValueError("IntCal20 cal_bp grid must contain only finite values.")

    if not np.all(np.isfinite(c14_curve_age_full)):
        raise ValueError("IntCal20 c14_age must contain only finite values.")

    if not np.all(np.isfinite(c14_curve_sigma_full)):
        raise ValueError("IntCal20 c14_sigma must contain only finite values.")

    rough_center = c14_bp + 300.0

    local_cal_lower = max(
        float(cal_grid_full.min()),
        float(np.nanmin(rough_center) - local_window_padding),
    )
    local_cal_upper = min(
        float(cal_grid_full.max()),
        float(np.nanmax(rough_center) + local_window_padding),
    )

    if local_cal_upper <= local_cal_lower:
        raise ValueError("Invalid local calendar window for C14 calibration.")

    local_keep = (
        (cal_grid_full >= local_cal_lower)
        & (cal_grid_full <= local_cal_upper)
    )

    cal_grid = cal_grid_full[local_keep]
    c14_curve_age = c14_curve_age_full[local_keep]
    c14_curve_sigma = c14_curve_sigma_full[local_keep]

    if len(cal_grid) < 3:
        raise ValueError("Local IntCal20 grid is too small.")

    cal_grid, c14_curve_age, c14_curve_sigma = _thin_ordered_grid(
        cal_grid,
        c14_curve_age,
        c14_curve_sigma,
        step=calendar_grid_step,
    )

    grid_weights = _trapezoid_grid_weights(cal_grid)

    return (
        c14_bp,
        c14_error,
        cal_grid,
        c14_curve_age,
        c14_curve_sigma,
        grid_weights,
        local_cal_lower,
        local_cal_upper,
    )


def _prepare_calibrated_gaussian_inputs(
    calibrated_cal_bp_mean: np.ndarray | None,
    calibrated_cal_bp_sd: np.ndarray | None,
    n_graves: int,
    local_window_padding: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Validate Gaussian approximation to calibrated single-date posteriors."""

    cal_mean = _validate_vector(
        calibrated_cal_bp_mean,
        n_graves,
        "calibrated_cal_bp_mean",
    )
    cal_sd = _validate_vector(
        calibrated_cal_bp_sd,
        n_graves,
        "calibrated_cal_bp_sd",
        positive=True,
    )

    local_cal_lower = float(np.nanmin(cal_mean - 4.0 * cal_sd) - local_window_padding)
    local_cal_upper = float(np.nanmax(cal_mean + 4.0 * cal_sd) + local_window_padding)

    if local_cal_upper <= local_cal_lower:
        raise ValueError("Invalid local calendar window for calibrated Gaussian data.")

    return cal_mean, cal_sd, local_cal_lower, local_cal_upper


def _marginalised_intcal20_loglike_with_calendar_logprior(
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    expected_cal_bp: pt.TensorVariable,
    calendar_logprior: pt.TensorVariable,
    cal_grid: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    grid_weights: np.ndarray,
    c14_extra_sigma: float = 0.0,
) -> pt.TensorVariable:
    """Per-grave marginal IntCal20 log-likelihood for an arbitrary calendar prior."""

    observed_c14_bp = np.asarray(observed_c14_bp, dtype=float)
    observed_c14_error = np.asarray(observed_c14_error, dtype=float)
    cal_grid = np.asarray(cal_grid, dtype=float)
    c14_curve_age = np.asarray(c14_curve_age, dtype=float)
    c14_curve_sigma = np.asarray(c14_curve_sigma, dtype=float)
    grid_weights = np.asarray(grid_weights, dtype=float)

    if len(cal_grid) != len(c14_curve_age):
        raise ValueError("cal_grid and c14_curve_age must have equal length.")

    if len(cal_grid) != len(c14_curve_sigma):
        raise ValueError("cal_grid and c14_curve_sigma must have equal length.")

    if len(cal_grid) != len(grid_weights):
        raise ValueError("cal_grid and grid_weights must have equal length.")

    if np.any(grid_weights <= 0):
        raise ValueError("grid_weights must be positive.")

    obs_bp_t = pt.as_tensor_variable(observed_c14_bp)
    obs_err_t = pt.as_tensor_variable(observed_c14_error)
    curve_age_t = pt.as_tensor_variable(c14_curve_age)
    curve_sigma_t = pt.as_tensor_variable(c14_curve_sigma)
    log_weights_t = pt.log(pt.as_tensor_variable(grid_weights))

    total_c14_sigma = pt.sqrt(
        obs_err_t[:, None] ** 2
        + curve_sigma_t[None, :] ** 2
        + float(c14_extra_sigma) ** 2
    )

    log_c14_given_cal = _normal_logpdf(
        obs_bp_t[:, None],
        curve_age_t[None, :],
        total_c14_sigma,
    )

    log_num = pt.logsumexp(
        log_c14_given_cal
        + calendar_logprior
        + log_weights_t[None, :],
        axis=1,
    )

    log_den = pt.logsumexp(
        calendar_logprior
        + log_weights_t[None, :],
        axis=1,
    )

    return log_num - log_den


def marginalised_intcal20_logp_per_grave(
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    expected_cal_bp: pt.TensorVariable,
    sigma_cal_link: pt.TensorVariable,
    cal_grid: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    grid_weights: np.ndarray,
    c14_extra_sigma: float = 0.0,
) -> pt.TensorVariable:
    """Per-grave marginalised IntCal20 radiocarbon log-likelihood."""

    cal_grid_t = pt.as_tensor_variable(np.asarray(cal_grid, dtype=float))
    calendar_logprior = _normal_logpdf(
        cal_grid_t[None, :],
        expected_cal_bp[:, None],
        sigma_cal_link,
    )

    return _marginalised_intcal20_loglike_with_calendar_logprior(
        observed_c14_bp=observed_c14_bp,
        observed_c14_error=observed_c14_error,
        expected_cal_bp=expected_cal_bp,
        calendar_logprior=calendar_logprior,
        cal_grid=cal_grid,
        c14_curve_age=c14_curve_age,
        c14_curve_sigma=c14_curve_sigma,
        grid_weights=grid_weights,
        c14_extra_sigma=c14_extra_sigma,
    )


def marginalised_intcal20_logp(
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    expected_cal_bp: pt.TensorVariable,
    sigma_cal_link: pt.TensorVariable,
    cal_grid: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    grid_weights: np.ndarray,
    c14_extra_sigma: float = 0.0,
    use_calendar_mismatch: bool = False,
    p_calendar_mismatch_prior: pt.TensorVariable | None = None,
    calendar_mismatch_scale: pt.TensorVariable | None = None,
    calendar_mismatch_nu: float = 5.0,
) -> tuple[pt.TensorVariable, pt.TensorVariable | None, pt.TensorVariable, pt.TensorVariable | None]:
    """Marginalised radiocarbon likelihood over calendar age.

    If use_calendar_mismatch=True, each date is modelled as a mixture:

    regular:
        cal_i ~ Normal(expected_cal_i, sigma_cal_link)

    mismatch/outlier:
        cal_i ~ StudentT(nu, expected_cal_i, sigma_cal_link * scale)

    Returns total logp, posterior mismatch responsibility per grave, regular
    component logp per grave, and outlier component logp per grave if available.
    """

    cal_grid_t = pt.as_tensor_variable(np.asarray(cal_grid, dtype=float))

    log_cal_regular = _normal_logpdf(
        cal_grid_t[None, :],
        expected_cal_bp[:, None],
        sigma_cal_link,
    )

    log_like_regular = _marginalised_intcal20_loglike_with_calendar_logprior(
        observed_c14_bp=observed_c14_bp,
        observed_c14_error=observed_c14_error,
        expected_cal_bp=expected_cal_bp,
        calendar_logprior=log_cal_regular,
        cal_grid=cal_grid,
        c14_curve_age=c14_curve_age,
        c14_curve_sigma=c14_curve_sigma,
        grid_weights=grid_weights,
        c14_extra_sigma=c14_extra_sigma,
    )

    if not use_calendar_mismatch:
        return pt.sum(log_like_regular), None, log_like_regular, None

    if p_calendar_mismatch_prior is None:
        raise ValueError(
            "p_calendar_mismatch_prior is required when use_calendar_mismatch=True."
        )

    if calendar_mismatch_scale is None:
        raise ValueError(
            "calendar_mismatch_scale is required when use_calendar_mismatch=True."
        )

    sigma_outlier = sigma_cal_link * calendar_mismatch_scale

    log_cal_outlier = _student_t_logpdf(
        cal_grid_t[None, :],
        expected_cal_bp[:, None],
        sigma_outlier,
        nu=calendar_mismatch_nu,
    )

    log_like_outlier = _marginalised_intcal20_loglike_with_calendar_logprior(
        observed_c14_bp=observed_c14_bp,
        observed_c14_error=observed_c14_error,
        expected_cal_bp=expected_cal_bp,
        calendar_logprior=log_cal_outlier,
        cal_grid=cal_grid,
        c14_curve_age=c14_curve_age,
        c14_curve_sigma=c14_curve_sigma,
        grid_weights=grid_weights,
        c14_extra_sigma=c14_extra_sigma,
    )

    p = p_calendar_mismatch_prior

    log_regular_weighted = pt.log1p(-p) + log_like_regular
    log_outlier_weighted = pt.log(p) + log_like_outlier

    log_mix = pt.logaddexp(log_regular_weighted, log_outlier_weighted)
    p_mismatch = pt.exp(log_outlier_weighted - log_mix)

    return pt.sum(log_mix), p_mismatch, log_like_regular, log_like_outlier


# -----------------------------------------------------------------------------
# Post-hoc reconstruction of individual calendar draws
# -----------------------------------------------------------------------------


def reconstruct_marginal_calendar_draws(
    idata: az.InferenceData,
    observed_c14_bp: np.ndarray,
    observed_c14_error: np.ndarray,
    cal_grid: np.ndarray,
    c14_curve_age: np.ndarray,
    c14_curve_sigma: np.ndarray,
    grid_weights: np.ndarray,
    c14_extra_sigma: float = 0.0,
    random_seed: int = 123,
    chunk_size: int = 250,
    replace_latent_cal_bp: bool = True,
    use_calendar_mismatch: bool = False,
    calendar_mismatch_nu: float = 5.0,
) -> az.InferenceData:
    """Add reconstructed individual calendar-age draws to InferenceData."""

    if "expected_cal_bp" not in idata.posterior:
        raise ValueError("idata.posterior must contain expected_cal_bp.")

    if "sigma_cal_link" not in idata.posterior:
        raise ValueError("idata.posterior must contain sigma_cal_link.")

    observed_c14_bp = np.asarray(observed_c14_bp, dtype=float)
    observed_c14_error = np.asarray(observed_c14_error, dtype=float)
    cal_grid = np.asarray(cal_grid, dtype=float)
    c14_curve_age = np.asarray(c14_curve_age, dtype=float)
    c14_curve_sigma = np.asarray(c14_curve_sigma, dtype=float)
    grid_weights = np.asarray(grid_weights, dtype=float)

    if np.any(grid_weights <= 0):
        raise ValueError("grid_weights must be positive.")

    expected = idata.posterior["expected_cal_bp"].values
    sigma_link = idata.posterior["sigma_cal_link"].values

    n_chain, n_draw, n_grave = expected.shape
    n_grid = len(cal_grid)

    if len(observed_c14_bp) != n_grave:
        raise ValueError("observed_c14_bp length must match number of graves.")

    if len(observed_c14_error) != n_grave:
        raise ValueError("observed_c14_error length must match number of graves.")

    expected_flat = expected.reshape(-1, n_grave)
    sigma_flat = sigma_link.reshape(-1)
    n_sample = expected_flat.shape[0]

    if use_calendar_mismatch:
        if "p_calendar_mismatch_prior" not in idata.posterior:
            raise ValueError(
                "idata.posterior must contain p_calendar_mismatch_prior "
                "when use_calendar_mismatch=True."
            )

        if "calendar_mismatch_scale" not in idata.posterior:
            raise ValueError(
                "idata.posterior must contain calendar_mismatch_scale "
                "when use_calendar_mismatch=True."
            )

        p_flat = idata.posterior["p_calendar_mismatch_prior"].values.reshape(-1)
        scale_flat = idata.posterior["calendar_mismatch_scale"].values.reshape(-1)
    else:
        p_flat = None
        scale_flat = None

    total_c14_sigma = np.sqrt(
        observed_c14_error[:, None] ** 2
        + c14_curve_sigma[None, :] ** 2
        + float(c14_extra_sigma) ** 2
    )

    log_c14_given_cal = _normal_logpdf_np(
        observed_c14_bp[:, None],
        c14_curve_age[None, :],
        total_c14_sigma,
    )

    log_grid_weights = np.log(grid_weights)

    reconstructed = np.empty((n_sample, n_grave), dtype=float)
    cond_mean = np.empty((n_sample, n_grave), dtype=float)
    cond_sd = np.empty((n_sample, n_grave), dtype=float)

    if use_calendar_mismatch:
        cond_p_mismatch = np.empty((n_sample, n_grave), dtype=float)
    else:
        cond_p_mismatch = None

    rng = np.random.default_rng(random_seed)

    for start in range(0, n_sample, chunk_size):
        stop = min(start + chunk_size, n_sample)

        expected_chunk = expected_flat[start:stop, :]
        sigma_chunk = sigma_flat[start:stop]

        log_cal_regular = _normal_logpdf_np(
            cal_grid[None, None, :],
            expected_chunk[:, :, None],
            sigma_chunk[:, None, None],
        )

        log_w_regular = (
            log_c14_given_cal[None, :, :]
            + log_cal_regular
            + log_grid_weights[None, None, :]
        )

        log_den_regular = logsumexp(
            log_cal_regular + log_grid_weights[None, None, :],
            axis=2,
        )

        log_like_regular_grid = log_w_regular - log_den_regular[:, :, None]

        if use_calendar_mismatch:
            assert p_flat is not None
            assert scale_flat is not None

            p_chunk = p_flat[start:stop]
            scale_chunk = scale_flat[start:stop]
            sigma_out_chunk = sigma_chunk * scale_chunk

            log_cal_outlier = _student_t_logpdf_np(
                cal_grid[None, None, :],
                expected_chunk[:, :, None],
                sigma_out_chunk[:, None, None],
                nu=calendar_mismatch_nu,
            )

            log_w_outlier = (
                log_c14_given_cal[None, :, :]
                + log_cal_outlier
                + log_grid_weights[None, None, :]
            )

            log_den_outlier = logsumexp(
                log_cal_outlier + log_grid_weights[None, None, :],
                axis=2,
            )

            log_like_outlier_grid = log_w_outlier - log_den_outlier[:, :, None]

            log_regular = (
                np.log1p(-p_chunk)[:, None, None]
                + log_like_regular_grid
            )

            log_outlier = (
                np.log(p_chunk)[:, None, None]
                + log_like_outlier_grid
            )

            log_w = np.logaddexp(log_regular, log_outlier)
        else:
            log_w = log_like_regular_grid

        probs = _softmax_np(log_w, axis=2)

        mean_chunk = np.sum(probs * cal_grid[None, None, :], axis=2)
        second_chunk = np.sum(probs * cal_grid[None, None, :] ** 2, axis=2)
        sd_chunk = np.sqrt(np.maximum(second_chunk - mean_chunk**2, 0.0))

        cond_mean[start:stop, :] = mean_chunk
        cond_sd[start:stop, :] = sd_chunk

        if use_calendar_mismatch:
            log_outlier_marginal = logsumexp(log_outlier, axis=2)
            log_total_marginal = logsumexp(log_w, axis=2)
            cond_p_mismatch[start:stop, :] = np.exp(
                log_outlier_marginal - log_total_marginal
            )

        cdf = np.cumsum(probs, axis=2)
        u = rng.random(size=(stop - start, n_grave, 1))
        idx = np.sum(cdf < u, axis=2)
        idx = np.clip(idx, 0, n_grid - 1)

        reconstructed[start:stop, :] = cal_grid[idx]

    reconstructed = reconstructed.reshape(n_chain, n_draw, n_grave)
    cond_mean = cond_mean.reshape(n_chain, n_draw, n_grave)
    cond_sd = cond_sd.reshape(n_chain, n_draw, n_grave)

    coords = {
        "chain": idata.posterior.coords["chain"],
        "draw": idata.posterior.coords["draw"],
        "grave": idata.posterior.coords["grave"],
    }

    latent_da = xr.DataArray(
        reconstructed,
        dims=("chain", "draw", "grave"),
        coords=coords,
        name="latent_cal_bp",
        attrs={
            "source": (
                "posthoc_reconstruction_from_marginalised_intcal20_likelihood"
            ),
            "description": (
                "Posterior draws from p(cal_i | c14_i, expected_cal_bp_i, "
                "calendar-link model), reconstructed after marginalised sampling."
            ),
        },
    )

    cond_mean_da = xr.DataArray(
        cond_mean,
        dims=("chain", "draw", "grave"),
        coords=coords,
        name="latent_cal_bp_cond_mean",
    )

    cond_sd_da = xr.DataArray(
        cond_sd,
        dims=("chain", "draw", "grave"),
        coords=coords,
        name="latent_cal_bp_cond_sd",
    )

    if replace_latent_cal_bp or "latent_cal_bp" not in idata.posterior:
        idata.posterior["latent_cal_bp"] = latent_da
    else:
        idata.posterior["latent_cal_bp_reconstructed"] = latent_da

    idata.posterior["latent_cal_bp_cond_mean"] = cond_mean_da
    idata.posterior["latent_cal_bp_cond_sd"] = cond_sd_da

    if use_calendar_mismatch and cond_p_mismatch is not None:
        cond_p_mismatch = cond_p_mismatch.reshape(n_chain, n_draw, n_grave)

        idata.posterior["p_calendar_mismatch_reconstructed"] = xr.DataArray(
            cond_p_mismatch,
            dims=("chain", "draw", "grave"),
            coords=coords,
            name="p_calendar_mismatch_reconstructed",
            attrs={
                "description": (
                    "Post-hoc posterior responsibility of the Student-t "
                    "calendar mismatch component."
                )
            },
        )

    return idata


def _store_model_configuration(
    idata: az.InferenceData,
    *,
    n_graves: int,
    n_types: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int,
    include_richness: bool,
    repulsion_strength: float,
    chronology_mode: str,
    chronology_likelihood: str,
    calendar_grid_step: int,
    local_window_padding: float,
    local_cal_lower: float | None,
    local_cal_upper: float | None,
    cal_alpha_mu: float | None,
    cal_alpha_sigma: float,
    cal_span_mu: float | None,
    cal_span_sigma: float,
    cal_span_lower: float,
    cal_span_upper: float,
    sigma_cal_link_mu: float,
    sigma_cal_link_sigma: float,
    sigma_cal_link_lower: float,
    sigma_cal_link_upper: float,
    c14_extra_sigma: float,
    use_calendar_mismatch: bool,
    calendar_mismatch_p_alpha: float,
    calendar_mismatch_p_beta: float,
    calendar_mismatch_scale_mu: float,
    calendar_mismatch_scale_sigma: float,
    calendar_mismatch_scale_lower: float,
    calendar_mismatch_scale_upper: float,
    calendar_mismatch_nu: float,
    intercept_mu: float,
    intercept_sigma: float,
    mu_sigma: float,
    sigma_mu_prior: float,
    log_sigma_hyper_sigma: float,
    log_sigma_hyper_sd_sigma: float,
    a_mu_prior: float,
    a_mu_sigma: float,
    a_hyper_sd_sigma: float,
    richness_sigma: float,
    max_treedepth: int,
) -> az.InferenceData:
    """Store model configuration and prior settings in idata.attrs."""

    idata.attrs["model_name"] = "parametric_pymc_seriation"
    idata.attrs["model_version"] = (
        "hierarchical_typology_marginalised_intcal20_student_t_calendar_mismatch"
        if use_calendar_mismatch
        else "hierarchical_typology_marginalised_intcal20"
    )
    idata.attrs["n_graves"] = int(n_graves)
    idata.attrs["n_types"] = int(n_types)

    idata.attrs["draws"] = int(draws)
    idata.attrs["tune"] = int(tune)
    idata.attrs["chains"] = int(chains)
    idata.attrs["target_accept"] = float(target_accept)
    idata.attrs["random_seed"] = int(random_seed)
    idata.attrs["max_treedepth"] = int(max_treedepth)

    idata.attrs["include_richness"] = bool(include_richness)
    idata.attrs["repulsion_strength"] = float(repulsion_strength)
    idata.attrs["chronology_mode"] = chronology_mode
    idata.attrs["chronology_likelihood"] = chronology_likelihood
    idata.attrs["calendar_grid_step"] = int(calendar_grid_step)
    idata.attrs["local_window_padding"] = float(local_window_padding)

    if local_cal_lower is not None:
        idata.attrs["local_cal_lower"] = float(local_cal_lower)

    if local_cal_upper is not None:
        idata.attrs["local_cal_upper"] = float(local_cal_upper)

    if cal_alpha_mu is not None:
        idata.attrs["cal_alpha_mu"] = float(cal_alpha_mu)

    idata.attrs["cal_alpha_sigma"] = float(cal_alpha_sigma)

    if cal_span_mu is not None:
        idata.attrs["cal_span_mu"] = float(cal_span_mu)

    idata.attrs["cal_span_sigma"] = float(cal_span_sigma)
    idata.attrs["cal_span_lower"] = float(cal_span_lower)
    idata.attrs["cal_span_upper"] = float(cal_span_upper)

    idata.attrs["sigma_cal_link_mu"] = float(sigma_cal_link_mu)
    idata.attrs["sigma_cal_link_sigma"] = float(sigma_cal_link_sigma)
    idata.attrs["sigma_cal_link_lower"] = float(sigma_cal_link_lower)
    idata.attrs["sigma_cal_link_upper"] = float(sigma_cal_link_upper)
    idata.attrs["c14_extra_sigma"] = float(c14_extra_sigma)

    idata.attrs["use_calendar_mismatch"] = bool(use_calendar_mismatch)
    idata.attrs["calendar_mismatch_p_alpha"] = float(calendar_mismatch_p_alpha)
    idata.attrs["calendar_mismatch_p_beta"] = float(calendar_mismatch_p_beta)
    idata.attrs["calendar_mismatch_scale_mu"] = float(calendar_mismatch_scale_mu)
    idata.attrs["calendar_mismatch_scale_sigma"] = float(
        calendar_mismatch_scale_sigma
    )
    idata.attrs["calendar_mismatch_scale_lower"] = float(
        calendar_mismatch_scale_lower
    )
    idata.attrs["calendar_mismatch_scale_upper"] = float(
        calendar_mismatch_scale_upper
    )
    idata.attrs["calendar_mismatch_nu"] = float(calendar_mismatch_nu)

    idata.attrs["intercept_mu"] = float(intercept_mu)
    idata.attrs["intercept_sigma"] = float(intercept_sigma)
    idata.attrs["mu_sigma"] = float(mu_sigma)

    idata.attrs["sigma_mu_prior"] = float(sigma_mu_prior)
    idata.attrs["log_sigma_hyper_sigma"] = float(log_sigma_hyper_sigma)
    idata.attrs["log_sigma_hyper_sd_sigma"] = float(log_sigma_hyper_sd_sigma)

    idata.attrs["a_mu_prior"] = float(a_mu_prior)
    idata.attrs["a_mu_sigma"] = float(a_mu_sigma)
    idata.attrs["a_hyper_sd_sigma"] = float(a_hyper_sd_sigma)

    idata.attrs["richness_sigma"] = float(richness_sigma)

    idata.attrs["calendar_prior_summary"] = (
        f"cal_alpha ~ Normal({cal_alpha_mu}, {cal_alpha_sigma}); "
        f"cal_span ~ TruncatedNormal({cal_span_mu}, {cal_span_sigma}, "
        f"{cal_span_lower}, {cal_span_upper}); "
        f"sigma_cal_link ~ TruncatedNormal({sigma_cal_link_mu}, "
        f"{sigma_cal_link_sigma}, {sigma_cal_link_lower}, "
        f"{sigma_cal_link_upper}); "
        f"use_calendar_mismatch={use_calendar_mismatch}; "
        f"p_calendar_mismatch_prior ~ Beta("
        f"{calendar_mismatch_p_alpha}, {calendar_mismatch_p_beta}); "
        f"calendar_mismatch_scale ~ TruncatedNormal("
        f"{calendar_mismatch_scale_mu}, {calendar_mismatch_scale_sigma}, "
        f"{calendar_mismatch_scale_lower}, {calendar_mismatch_scale_upper}); "
        f"calendar_mismatch_nu={calendar_mismatch_nu}"
    )

    idata.attrs["typology_prior_summary"] = (
        f"intercept ~ Normal({intercept_mu}, {intercept_sigma}); "
        f"mu[type] ~ Normal(0, {mu_sigma}); "
        f"mu_log_sigma ~ Normal(log({sigma_mu_prior}), "
        f"{log_sigma_hyper_sigma}); "
        f"sd_log_sigma ~ HalfNormal({log_sigma_hyper_sd_sigma}); "
        f"mu_a ~ Normal({a_mu_prior}, {a_mu_sigma}); "
        f"sd_a ~ HalfNormal({a_hyper_sd_sigma}); "
        f"sigma_g ~ HalfNormal({richness_sigma})"
    )

    return idata


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------


def fit_parametric_pymc_seriation(
    Y: np.ndarray,
    draws: int = 800,
    tune: int = 1_200,
    chains: int = 4,
    target_accept: float = 0.96,
    random_seed: int = 123,
    include_richness: bool = True,
    repulsion_strength: float = 0.35,
    chronology_mode: str = "none",
    chronology_likelihood: str = "intcal20",
    orientation_reference: np.ndarray | None = None,
    c14_bp: np.ndarray | None = None,
    c14_error: np.ndarray | None = None,
    intcal20_curve: pd.DataFrame | None = None,
    calibrated_cal_bp_mean: np.ndarray | None = None,
    calibrated_cal_bp_sd: np.ndarray | None = None,
    cal_alpha_mu: float | None = None,
    cal_alpha_sigma: float = 120.0,
    cal_span_mu: float | None = None,
    cal_span_sigma: float = 80.0,
    cal_span_lower: float = 5.0,
    cal_span_upper: float = 220.0,
    sigma_cal_link_mu: float = 60.0,
    sigma_cal_link_sigma: float = 35.0,
    sigma_cal_link_lower: float = 15.0,
    sigma_cal_link_upper: float = 150.0,
    c14_extra_sigma: float = 0.0,
    use_calendar_mismatch: bool = True,
    calendar_mismatch_p_alpha: float = 1.0,
    calendar_mismatch_p_beta: float = 9.0,
    calendar_mismatch_scale_mu: float = 5.0,
    calendar_mismatch_scale_sigma: float = 2.0,
    calendar_mismatch_scale_lower: float = 2.0,
    calendar_mismatch_scale_upper: float = 15.0,
    calendar_mismatch_nu: float = 5.0,
    local_window_padding: float = 500.0,
    calendar_grid_step: int = 10,
    max_treedepth: int = 12,
    intercept_mu: float = -0.5,
    intercept_sigma: float = 1.2,
    mu_sigma: float = 2.0,
    sigma_mu_prior: float = 0.70,
    log_sigma_hyper_sigma: float = 0.70,
    log_sigma_hyper_sd_sigma: float = 0.60,
    a_mu_prior: float = 1.2,
    a_mu_sigma: float = 1.0,
    a_hyper_sd_sigma: float = 1.0,
    richness_sigma: float = 0.5,
) -> az.InferenceData:
    """Fit a parametric PyMC seriation model."""

    Y = np.asarray(Y)

    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional matrix.")

    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be a binary matrix containing only 0 and 1.")

    if calendar_grid_step < 1:
        raise ValueError("calendar_grid_step must be >= 1.")

    if cal_span_upper <= cal_span_lower:
        raise ValueError("cal_span_upper must be greater than cal_span_lower.")

    if sigma_cal_link_upper <= sigma_cal_link_lower:
        raise ValueError(
            "sigma_cal_link_upper must be greater than sigma_cal_link_lower."
        )

    if sigma_mu_prior <= 0:
        raise ValueError("sigma_mu_prior must be > 0.")

    if mu_sigma <= 0:
        raise ValueError("mu_sigma must be > 0.")

    if richness_sigma <= 0:
        raise ValueError("richness_sigma must be > 0.")

    if calendar_mismatch_p_alpha <= 0:
        raise ValueError("calendar_mismatch_p_alpha must be > 0.")

    if calendar_mismatch_p_beta <= 0:
        raise ValueError("calendar_mismatch_p_beta must be > 0.")

    if calendar_mismatch_scale_lower <= 1.0:
        raise ValueError("calendar_mismatch_scale_lower must be > 1.")

    if calendar_mismatch_scale_upper <= calendar_mismatch_scale_lower:
        raise ValueError(
            "calendar_mismatch_scale_upper must be greater than "
            "calendar_mismatch_scale_lower."
        )

    if calendar_mismatch_nu <= 0:
        raise ValueError("calendar_mismatch_nu must be > 0.")

    n_graves, n_types = Y.shape
    coords = {"grave": np.arange(n_graves), "type": np.arange(n_types)}

    allowed_chronology_modes = {"none", "c14_typology_linked"}
    if chronology_mode not in allowed_chronology_modes:
        raise ValueError(
            "chronology_mode must be one of: "
            + ", ".join(sorted(allowed_chronology_modes))
        )

    allowed_chronology_likelihoods = {"intcal20", "calibrated_gaussian"}
    if chronology_likelihood not in allowed_chronology_likelihoods:
        raise ValueError(
            "chronology_likelihood must be one of: "
            + ", ".join(sorted(allowed_chronology_likelihoods))
        )

    use_chronology = chronology_mode == "c14_typology_linked"
    initvals: dict[str, object] = {}

    # -------------------------------------------------------------------------
    # Optional orientation reference
    # -------------------------------------------------------------------------

    orientation_ref_z = None
    if orientation_reference is not None:
        orientation_ref_z = _standardise_reference(orientation_reference)

        if len(orientation_ref_z) != n_graves:
            raise ValueError(
                "orientation_reference must have the same length as the number of graves."
            )

        initvals["t_raw"] = orientation_ref_z

    # -------------------------------------------------------------------------
    # Optional chronological data preparation
    # -------------------------------------------------------------------------

    local_cal_lower = None
    local_cal_upper = None

    if use_chronology:
        if chronology_likelihood == "intcal20":
            (
                c14_bp,
                c14_error,
                cal_grid,
                c14_curve_age,
                c14_curve_sigma,
                grid_weights,
                local_cal_lower,
                local_cal_upper,
            ) = _prepare_c14_inputs(
                c14_bp=c14_bp,
                c14_error=c14_error,
                intcal20_curve=intcal20_curve,
                n_graves=n_graves,
                local_window_padding=local_window_padding,
                calendar_grid_step=calendar_grid_step,
            )

            rough_cal_bp = c14_bp + 300.0

            if cal_alpha_mu is None:
                cal_alpha_mu = float(np.nanmedian(rough_cal_bp))

            if cal_span_mu is None:
                q10, q90 = np.nanquantile(rough_cal_bp, [0.10, 0.90])
                cal_span_mu = float(np.clip((q90 - q10) / 4.0, 20.0, 120.0))

        elif chronology_likelihood == "calibrated_gaussian":
            (
                calibrated_cal_bp_mean,
                calibrated_cal_bp_sd,
                local_cal_lower,
                local_cal_upper,
            ) = _prepare_calibrated_gaussian_inputs(
                calibrated_cal_bp_mean=calibrated_cal_bp_mean,
                calibrated_cal_bp_sd=calibrated_cal_bp_sd,
                n_graves=n_graves,
                local_window_padding=local_window_padding,
            )

            if cal_alpha_mu is None:
                cal_alpha_mu = float(np.nanmedian(calibrated_cal_bp_mean))

            if cal_span_mu is None:
                q10, q90 = np.nanquantile(calibrated_cal_bp_mean, [0.10, 0.90])
                cal_span_mu = float(np.clip((q90 - q10) / 4.0, 20.0, 120.0))

        if cal_alpha_mu is None:
            raise ValueError("cal_alpha_mu could not be inferred.")

        if cal_span_mu is None:
            raise ValueError("cal_span_mu could not be inferred.")

        print("\nC14/calendar model used:")
        print("chronology_mode:", chronology_mode)
        print("chronology_likelihood:", chronology_likelihood)
        print(
            "calendar treatment:",
            "marginalised + posthoc reconstructed"
            if chronology_likelihood == "intcal20"
            else "latent",
        )
        print("local_cal_lower:", round(float(local_cal_lower), 3))
        print("local_cal_upper:", round(float(local_cal_upper), 3))

        if chronology_likelihood == "intcal20":
            print("local IntCal20 grid size:", int(len(cal_grid)))
            print("calendar_grid_step:", int(calendar_grid_step))

        print("cal_alpha prior:", f"Normal({cal_alpha_mu:.3f}, {cal_alpha_sigma:.3f})")
        print(
            "cal_span prior:",
            (
                "TruncatedNormal("
                f"mu={cal_span_mu:.3f}, "
                f"sigma={cal_span_sigma:.3f}, "
                f"lower={cal_span_lower:.3f}, "
                f"upper={cal_span_upper:.3f})"
            ),
        )
        print(
            "sigma_cal_link prior:",
            (
                "TruncatedNormal("
                f"mu={sigma_cal_link_mu:.3f}, "
                f"sigma={sigma_cal_link_sigma:.3f}, "
                f"lower={sigma_cal_link_lower:.3f}, "
                f"upper={sigma_cal_link_upper:.3f})"
            ),
        )

        if chronology_likelihood == "intcal20":
            print("calendar mismatch mixture:", bool(use_calendar_mismatch))

            if use_calendar_mismatch:
                print(
                    "calendar mismatch prior:",
                    (
                        "p_calendar_mismatch_prior ~ "
                        f"Beta({calendar_mismatch_p_alpha:.3f}, "
                        f"{calendar_mismatch_p_beta:.3f})"
                    ),
                )
                print(
                    "calendar mismatch scale prior:",
                    (
                        "TruncatedNormal("
                        f"mu={calendar_mismatch_scale_mu:.3f}, "
                        f"sigma={calendar_mismatch_scale_sigma:.3f}, "
                        f"lower={calendar_mismatch_scale_lower:.3f}, "
                        f"upper={calendar_mismatch_scale_upper:.3f})"
                    ),
                )
                print("calendar mismatch likelihood:", f"StudentT(nu={calendar_mismatch_nu:.3f})")

        initvals.update(
            {
                "cal_alpha": float(cal_alpha_mu),
                "cal_span": float(
                    np.clip(
                        cal_span_mu,
                        cal_span_lower + 1e-3,
                        cal_span_upper - 1e-3,
                    )
                ),
                "sigma_cal_link": float(
                    np.clip(
                        sigma_cal_link_mu,
                        sigma_cal_link_lower + 1e-3,
                        sigma_cal_link_upper - 1e-3,
                    )
                ),
            }
        )

        if use_calendar_mismatch and chronology_likelihood == "intcal20":
            initvals.update(
                {
                    "p_calendar_mismatch_prior": float(
                        calendar_mismatch_p_alpha
                        / (calendar_mismatch_p_alpha + calendar_mismatch_p_beta)
                    ),
                    "calendar_mismatch_scale": float(
                        np.clip(
                            calendar_mismatch_scale_mu,
                            calendar_mismatch_scale_lower + 1e-3,
                            calendar_mismatch_scale_upper - 1e-3,
                        )
                    ),
                }
            )

        if chronology_likelihood == "calibrated_gaussian":
            initvals["latent_cal_bp_raw"] = np.zeros(n_graves)

    print("typological model: hierarchical type widths and amplitudes")
    print(
        "type width hyperprior:",
        f"mu_log_sigma ~ Normal(log({sigma_mu_prior:.3f}), {log_sigma_hyper_sigma:.3f}); "
        f"sd_log_sigma ~ HalfNormal({log_sigma_hyper_sd_sigma:.3f})",
    )
    print(
        "type amplitude hyperprior:",
        f"mu_a ~ Normal({a_mu_prior:.3f}, {a_mu_sigma:.3f}); "
        f"sd_a ~ HalfNormal({a_hyper_sd_sigma:.3f})",
    )

    initvals.update(
        {
            "mu_log_sigma": float(np.log(sigma_mu_prior)),
            "sd_log_sigma": 0.35,
            "mu_a": float(a_mu_prior),
            "sd_a": 0.6,
        }
    )

    # -------------------------------------------------------------------------
    # PyMC model
    # -------------------------------------------------------------------------

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------------
        # Latent typological seriation axis
        # ------------------------------------------------------------------

        t_raw = pm.Normal("t_raw", 0.0, 1.0, dims="grave")
        t_centered = t_raw - pt.mean(t_raw)

        t = pm.Deterministic(
            "t",
            t_centered / pt.sqrt(pt.var(t_centered) + 1e-6),
            dims="grave",
        )

        if orientation_ref_z is not None:
            ref_t = pt.as_tensor_variable(orientation_ref_z)

            orientation_score = pm.Deterministic(
                "orientation_score",
                pt.sum(t * ref_t),
            )

            pm.Potential(
                "axis_orientation",
                pt.switch(orientation_score >= 0.0, 0.0, -np.inf),
            )

        repulsion_lp = pm.Deterministic(
            "repulsion_lp",
            spacing_repulsion(
                t,
                strength=repulsion_strength,
                min_dist=0.10,
            ),
        )
        pm.Potential("spacing_repulsion", repulsion_lp)

        # ------------------------------------------------------------------
        # Optional linked chronology model
        # ------------------------------------------------------------------

        if use_chronology:
            cal_alpha = pm.Normal(
                "cal_alpha",
                mu=cal_alpha_mu,
                sigma=cal_alpha_sigma,
            )

            cal_span = pm.TruncatedNormal(
                "cal_span",
                mu=cal_span_mu,
                sigma=cal_span_sigma,
                lower=cal_span_lower,
                upper=cal_span_upper,
            )

            cal_beta = pm.Deterministic("cal_beta", -cal_span)

            expected_cal_bp = pm.Deterministic(
                "expected_cal_bp",
                cal_alpha + cal_beta * t,
                dims="grave",
            )

            typological_cal_bp = pm.Deterministic(
                "typological_cal_bp",
                expected_cal_bp,
                dims="grave",
            )

            sigma_cal_link = pm.TruncatedNormal(
                "sigma_cal_link",
                mu=sigma_cal_link_mu,
                sigma=sigma_cal_link_sigma,
                lower=sigma_cal_link_lower,
                upper=sigma_cal_link_upper,
            )

            sigma_cal_regular = pm.Deterministic(
                "sigma_cal_regular",
                sigma_cal_link,
            )

            p_calendar_mismatch_prior = None
            calendar_mismatch_scale = None

            if use_calendar_mismatch and chronology_likelihood == "intcal20":
                p_calendar_mismatch_prior = pm.Beta(
                    "p_calendar_mismatch_prior",
                    alpha=calendar_mismatch_p_alpha,
                    beta=calendar_mismatch_p_beta,
                )

                calendar_mismatch_scale = pm.TruncatedNormal(
                    "calendar_mismatch_scale",
                    mu=calendar_mismatch_scale_mu,
                    sigma=calendar_mismatch_scale_sigma,
                    lower=calendar_mismatch_scale_lower,
                    upper=calendar_mismatch_scale_upper,
                )

                sigma_cal_outlier = pm.Deterministic(
                    "sigma_cal_outlier",
                    sigma_cal_link * calendar_mismatch_scale,
                )

                pm.Deterministic(
                    "calendar_mismatch_nu",
                    pt.as_tensor_variable(float(calendar_mismatch_nu)),
                )

            c14_extra_sigma_det = pm.Deterministic(
                "c14_extra_sigma",
                pt.as_tensor_variable(float(c14_extra_sigma)),
            )

            if chronology_likelihood == "intcal20":
                (
                    marginal_logp,
                    p_mismatch,
                    c14_logp_regular,
                    c14_logp_outlier,
                ) = marginalised_intcal20_logp(
                    observed_c14_bp=c14_bp,
                    observed_c14_error=c14_error,
                    expected_cal_bp=expected_cal_bp,
                    sigma_cal_link=sigma_cal_link,
                    cal_grid=cal_grid,
                    c14_curve_age=c14_curve_age,
                    c14_curve_sigma=c14_curve_sigma,
                    grid_weights=grid_weights,
                    c14_extra_sigma=float(c14_extra_sigma),
                    use_calendar_mismatch=use_calendar_mismatch,
                    p_calendar_mismatch_prior=p_calendar_mismatch_prior,
                    calendar_mismatch_scale=calendar_mismatch_scale,
                    calendar_mismatch_nu=calendar_mismatch_nu,
                )

                pm.Potential(
                    "c14_marginalised_likelihood",
                    marginal_logp,
                )

                pm.Deterministic(
                    "c14_logp_regular",
                    c14_logp_regular,
                    dims="grave",
                )

                if use_calendar_mismatch and p_mismatch is not None:
                    pm.Deterministic(
                        "p_calendar_mismatch",
                        p_mismatch,
                        dims="grave",
                    )

                if use_calendar_mismatch and c14_logp_outlier is not None:
                    pm.Deterministic(
                        "c14_logp_outlier",
                        c14_logp_outlier,
                        dims="grave",
                    )

            elif chronology_likelihood == "calibrated_gaussian":
                latent_cal_bp_raw = pm.Normal(
                    "latent_cal_bp_raw",
                    0.0,
                    1.0,
                    dims="grave",
                )

                latent_cal_bp_unbounded = (
                    expected_cal_bp + sigma_cal_link * latent_cal_bp_raw
                )

                latent_cal_bp = pm.Deterministic(
                    "latent_cal_bp",
                    pt.clip(
                        latent_cal_bp_unbounded,
                        float(local_cal_lower) + 1e-3,
                        float(local_cal_upper) - 1e-3,
                    ),
                    dims="grave",
                )

                pm.Normal(
                    "cal_bp_obs_approx",
                    mu=latent_cal_bp,
                    sigma=pt.as_tensor_variable(calibrated_cal_bp_sd),
                    observed=calibrated_cal_bp_mean,
                    dims="grave",
                )

        # ------------------------------------------------------------------
        # Typological observation model
        # ------------------------------------------------------------------

        intercept = pm.Normal(
            "intercept",
            mu=intercept_mu,
            sigma=intercept_sigma,
        )

        mu = pm.Normal(
            "mu",
            mu=0.0,
            sigma=mu_sigma,
            dims="type",
        )

        mu_log_sigma = pm.Normal(
            "mu_log_sigma",
            mu=np.log(float(sigma_mu_prior)),
            sigma=log_sigma_hyper_sigma,
        )

        sd_log_sigma = pm.HalfNormal(
            "sd_log_sigma",
            sigma=log_sigma_hyper_sd_sigma,
        )

        log_sigma_raw = pm.Normal(
            "log_sigma_raw",
            0.0,
            1.0,
            dims="type",
        )

        log_sigma = pm.Deterministic(
            "log_sigma",
            mu_log_sigma + sd_log_sigma * log_sigma_raw,
            dims="type",
        )

        sigma = pm.Deterministic(
            "sigma",
            pt.exp(log_sigma),
            dims="type",
        )

        mu_a = pm.Normal(
            "mu_a",
            mu=a_mu_prior,
            sigma=a_mu_sigma,
        )

        sd_a = pm.HalfNormal(
            "sd_a",
            sigma=a_hyper_sd_sigma,
        )

        a_raw = pm.Normal(
            "a_raw",
            0.0,
            1.0,
            dims="type",
        )

        a = pm.Deterministic(
            "a",
            mu_a + sd_a * a_raw,
            dims="type",
        )

        if include_richness:
            sigma_g = pm.HalfNormal("sigma_g", sigma=richness_sigma)

            g_raw = pm.Normal(
                "g_raw",
                0.0,
                1.0,
                dims="grave",
            )

            g = pm.Deterministic(
                "g",
                sigma_g * (g_raw - pt.mean(g_raw)),
                dims="grave",
            )

            grave_effect = g[:, None]
        else:
            grave_effect = 0.0

        dist_penalty = ((t[:, None] - mu[None, :]) ** 2) / (
            2.0 * sigma[None, :] ** 2
        )

        eta = intercept + a[None, :] + grave_effect - dist_penalty

        pm.Bernoulli(
            "Y_obs",
            logit_p=eta,
            observed=Y,
            dims=("grave", "type"),
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            init="jitter+adapt_diag",
            initvals=initvals if initvals else None,
            nuts_sampler_kwargs={"max_treedepth": max_treedepth},
        )

        if use_chronology and chronology_likelihood == "intcal20":
            idata = reconstruct_marginal_calendar_draws(
                idata=idata,
                observed_c14_bp=c14_bp,
                observed_c14_error=c14_error,
                cal_grid=cal_grid,
                c14_curve_age=c14_curve_age,
                c14_curve_sigma=c14_curve_sigma,
                grid_weights=grid_weights,
                c14_extra_sigma=float(c14_extra_sigma),
                random_seed=random_seed,
                replace_latent_cal_bp=True,
                use_calendar_mismatch=use_calendar_mismatch,
                calendar_mismatch_nu=calendar_mismatch_nu,
            )

        idata = _store_model_configuration(
            idata,
            n_graves=n_graves,
            n_types=n_types,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            include_richness=include_richness,
            repulsion_strength=repulsion_strength,
            chronology_mode=chronology_mode,
            chronology_likelihood=chronology_likelihood,
            calendar_grid_step=calendar_grid_step,
            local_window_padding=local_window_padding,
            local_cal_lower=local_cal_lower,
            local_cal_upper=local_cal_upper,
            cal_alpha_mu=cal_alpha_mu,
            cal_alpha_sigma=cal_alpha_sigma,
            cal_span_mu=cal_span_mu,
            cal_span_sigma=cal_span_sigma,
            cal_span_lower=cal_span_lower,
            cal_span_upper=cal_span_upper,
            sigma_cal_link_mu=sigma_cal_link_mu,
            sigma_cal_link_sigma=sigma_cal_link_sigma,
            sigma_cal_link_lower=sigma_cal_link_lower,
            sigma_cal_link_upper=sigma_cal_link_upper,
            c14_extra_sigma=c14_extra_sigma,
            use_calendar_mismatch=use_calendar_mismatch,
            calendar_mismatch_p_alpha=calendar_mismatch_p_alpha,
            calendar_mismatch_p_beta=calendar_mismatch_p_beta,
            calendar_mismatch_scale_mu=calendar_mismatch_scale_mu,
            calendar_mismatch_scale_sigma=calendar_mismatch_scale_sigma,
            calendar_mismatch_scale_lower=calendar_mismatch_scale_lower,
            calendar_mismatch_scale_upper=calendar_mismatch_scale_upper,
            calendar_mismatch_nu=calendar_mismatch_nu,
            intercept_mu=intercept_mu,
            intercept_sigma=intercept_sigma,
            mu_sigma=mu_sigma,
            sigma_mu_prior=sigma_mu_prior,
            log_sigma_hyper_sigma=log_sigma_hyper_sigma,
            log_sigma_hyper_sd_sigma=log_sigma_hyper_sd_sigma,
            a_mu_prior=a_mu_prior,
            a_mu_sigma=a_mu_sigma,
            a_hyper_sd_sigma=a_hyper_sd_sigma,
            richness_sigma=richness_sigma,
            max_treedepth=max_treedepth,
        )

    return idata
