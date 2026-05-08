from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def spacing_repulsion(
    t: pt.TensorVariable,
    strength: float = 0.35,
    min_dist: float = 0.10,
) -> pt.TensorVariable:
    """Softly encourage grave coordinates to spread along the latent axis.

    This is only a weak regularising potential. It prevents the latent grave
    coordinates from collapsing too strongly into identical positions, but it
    should not determine the seriation by itself.
    """

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


def _softmax_np(log_w: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""

    m = np.max(log_w, axis=axis, keepdims=True)
    w = np.exp(log_w - m)
    return w / np.sum(w, axis=axis, keepdims=True)


def infer_cal_span_prior_from_reference_calibration(
    row_scores: np.ndarray,
    unmodelled_cal_bp: np.ndarray,
    min_n: int = 8,
    min_span: float = 20.0,
    max_span: float = 160.0,
    min_sigma: float = 20.0,
    max_sigma: float = 80.0,
) -> tuple[float, float]:
    """Infer a weakly data-informed prior for calendar years per t-unit.

    The model uses a standardised latent typological axis t and

        expected_cal_bp = cal_alpha - cal_span * t

    Therefore, cal_span is approximately the number of calendar years per
    one standardised unit of the typological axis.

    This function estimates that scale from the robust relationship between
    an already oriented reference axis and unmodelled calibrated C14 means.

    It deliberately returns a weak prior, not a tight empirical constraint.
    """

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
    """Thin an ordered grid and aligned arrays by keeping every `step`th point.

    The first and last grid points are always retained, so that the local
    integration window remains unchanged.
    """

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
    """Validate and prepare C14 data and local IntCal20 curve arrays.

    The returned IntCal20 arrays are restricted to a local calendar window,
    optionally thinned, and include grid weights for numerical marginalisation
    over calendar age.
    """

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
    """Validate Gaussian approximation to calibrated single-date posteriors.

    calibrated_cal_bp_mean and calibrated_cal_bp_sd should normally come from
    an unmodelled single-date calibration step.

    This mode is diagnostic only. It is not intended as the final chronology
    likelihood.
    """

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
) -> pt.TensorVariable:
    """Marginalised radiocarbon likelihood over calendar age.

    Computes, for each grave i:

        log p(c14_i | expected_cal_i)
        =
        log ∫ p(c14_i | cal, IntCal20)
              p(cal | expected_cal_i, sigma_cal_link) dcal

    using a discrete calendar grid and logsumexp.

    This removes individual latent calendar ages from NUTS while retaining an
    explicit IntCal20-based radiocarbon likelihood.
    """

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

    cal_grid_t = pt.as_tensor_variable(cal_grid)
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

    log_cal_given_expected = _normal_logpdf(
        cal_grid_t[None, :],
        expected_cal_bp[:, None],
        sigma_cal_link,
    )

    log_num = pt.logsumexp(
        log_c14_given_cal
        + log_cal_given_expected
        + log_weights_t[None, :],
        axis=1,
    )

    log_den = pt.logsumexp(
        log_cal_given_expected
        + log_weights_t[None, :],
        axis=1,
    )

    return pt.sum(log_num - log_den)


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
) -> az.InferenceData:
    """Add reconstructed individual calendar-age draws to an InferenceData object.

    The marginalised IntCal20 model integrates out individual calendar ages
    during NUTS sampling. This function reconstructs posterior draws from

        p(cal_i | c14_i, expected_cal_bp_i, sigma_cal_link)

    for each posterior draw and each grave.

    If replace_latent_cal_bp=True, the reconstructed draws are written to
    idata.posterior["latent_cal_bp"], preserving compatibility with downstream
    summary and plotting functions that expect that variable.
    """

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

    rng = np.random.default_rng(random_seed)

    for start in range(0, n_sample, chunk_size):
        stop = min(start + chunk_size, n_sample)

        expected_chunk = expected_flat[start:stop, :]
        sigma_chunk = sigma_flat[start:stop]

        log_cal_given_expected = _normal_logpdf_np(
            cal_grid[None, None, :],
            expected_chunk[:, :, None],
            sigma_chunk[:, None, None],
        )

        log_w = (
            log_c14_given_cal[None, :, :]
            + log_cal_given_expected
            + log_grid_weights[None, None, :]
        )

        probs = _softmax_np(log_w, axis=2)

        mean_chunk = np.sum(probs * cal_grid[None, None, :], axis=2)
        second_chunk = np.sum(probs * cal_grid[None, None, :] ** 2, axis=2)
        sd_chunk = np.sqrt(np.maximum(second_chunk - mean_chunk**2, 0.0))

        cond_mean[start:stop, :] = mean_chunk
        cond_sd[start:stop, :] = sd_chunk

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
            "source": "posthoc_reconstruction_from_marginalised_intcal20_likelihood",
            "description": (
                "Posterior draws from p(cal_i | c14_i, expected_cal_bp_i, "
                "sigma_cal_link), reconstructed after marginalised sampling."
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
    sigma_mu_prior: float = 0.70,
    sigma_sigma_prior: float = 0.35,
    sigma_lower: float = 0.20,
    sigma_upper: float = 1.80,
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
    cal_span_upper: float = 180.0,
    sigma_cal_link_mu: float = 50.0,
    sigma_cal_link_sigma: float = 30.0,
    sigma_cal_link_lower: float = 15.0,
    sigma_cal_link_upper: float = 120.0,
    c14_extra_sigma: float = 0.0,
    local_window_padding: float = 500.0,
    calendar_grid_step: int = 10,
    max_treedepth: int = 12,
) -> az.InferenceData:
    """Fit a parametric PyMC seriation model.

    Parameters
    ----------
    chronology_mode:
        "none"
            Pure typological seriation.

        "c14_typology_linked"
            Typological position defines an expected calendar position.

    chronology_likelihood:
        "intcal20"
            Uses an explicit IntCal20 radiocarbon likelihood, marginalised over
            individual calendar ages. Individual calendar-age draws are
            reconstructed after sampling and written to posterior["latent_cal_bp"].

        "calibrated_gaussian"
            Uses a Gaussian approximation to already calibrated single-date
            posterior summaries. This is mainly a diagnostic mode.

    calendar_grid_step:
        Thinning factor for the local IntCal20 calendar grid. A value of 1 keeps
        every grid point. A value of 10 keeps roughly every tenth point plus
        both endpoints. Integration weights are recomputed after thinning.

    Model convention
    ----------------
    t is a standardised typological axis.

    increasing t = typologically later / younger
    larger cal BP = older

    Therefore:

        expected_cal_bp = cal_alpha - cal_span * t
    """

    Y = np.asarray(Y)

    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional matrix.")

    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be a binary matrix containing only 0 and 1.")

    if calendar_grid_step < 1:
        raise ValueError("calendar_grid_step must be >= 1.")

    if sigma_lower <= 0:
        raise ValueError("sigma_lower must be > 0.")

    if sigma_upper <= sigma_lower:
        raise ValueError("sigma_upper must be greater than sigma_lower.")

    if cal_span_upper <= cal_span_lower:
        raise ValueError("cal_span_upper must be greater than cal_span_lower.")

    if sigma_cal_link_upper <= sigma_cal_link_lower:
        raise ValueError(
            "sigma_cal_link_upper must be greater than sigma_cal_link_lower."
        )

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
        print(
            "type sigma prior:",
            (
                "TruncatedNormal("
                f"mu={sigma_mu_prior:.3f}, "
                f"sigma={sigma_sigma_prior:.3f}, "
                f"lower={sigma_lower:.3f}, "
                f"upper={sigma_upper:.3f})"
            ),
        )

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

        if chronology_likelihood == "calibrated_gaussian":
            initvals["latent_cal_bp_raw"] = np.zeros(n_graves)

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

            c14_extra_sigma_det = pm.Deterministic(
                "c14_extra_sigma",
                pt.as_tensor_variable(float(c14_extra_sigma)),
            )

            if chronology_likelihood == "intcal20":
                marginal_logp = marginalised_intcal20_logp(
                    observed_c14_bp=c14_bp,
                    observed_c14_error=c14_error,
                    expected_cal_bp=expected_cal_bp,
                    sigma_cal_link=sigma_cal_link,
                    cal_grid=cal_grid,
                    c14_curve_age=c14_curve_age,
                    c14_curve_sigma=c14_curve_sigma,
                    grid_weights=grid_weights,
                    c14_extra_sigma=float(c14_extra_sigma),
                )

                pm.Potential(
                    "c14_marginalised_likelihood",
                    marginal_logp,
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

        intercept = pm.Normal("intercept", mu=-0.5, sigma=1.0)

        mu = pm.Normal(
            "mu",
            mu=0.0,
            sigma=1.6,
            dims="type",
        )

        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu_prior,
            sigma=sigma_sigma_prior,
            lower=sigma_lower,
            upper=sigma_upper,
            dims="type",
        )

        a = pm.Normal(
            "a",
            mu=1.5,
            sigma=1.0,
            dims="type",
        )

        if include_richness:
            sigma_g = pm.HalfNormal("sigma_g", sigma=0.5)

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
            )

    return idata