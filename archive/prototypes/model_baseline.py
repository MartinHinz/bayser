from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from bayser.calibration import pytensor_linear_interp


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

    # Robust Theil-Sen-like slope without adding another dependency:
    # median of all pairwise slopes.
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


def _prepare_c14_inputs(
    c14_bp: np.ndarray | None,
    c14_error: np.ndarray | None,
    intcal20_curve: pd.DataFrame | None,
    n_graves: int,
    local_window_padding: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Validate and prepare C14 data and IntCal20 curve arrays."""

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

    cal_grid = curve["cal_bp"].to_numpy(dtype=float)
    c14_curve_age = curve["c14_age"].to_numpy(dtype=float)
    c14_curve_sigma = curve["c14_sigma"].to_numpy(dtype=float)

    if np.any(np.diff(cal_grid) <= 0):
        raise ValueError("IntCal20 cal_bp grid must be strictly increasing.")

    if not np.all(np.isfinite(cal_grid)):
        raise ValueError("IntCal20 cal_bp grid must contain only finite values.")

    if not np.all(np.isfinite(c14_curve_age)):
        raise ValueError("IntCal20 c14_age must contain only finite values.")

    if not np.all(np.isfinite(c14_curve_sigma)):
        raise ValueError("IntCal20 c14_sigma must contain only finite values.")

    # Broad but local enough for this dataset. This is not a calibration prior;
    # it only prevents the sampler from wandering through irrelevant parts of
    # the full 0–55 ka calibration curve.
    rough_center = c14_bp + 300.0

    local_cal_lower = max(
        float(cal_grid.min()),
        float(np.nanmin(rough_center) - local_window_padding),
    )
    local_cal_upper = min(
        float(cal_grid.max()),
        float(np.nanmax(rough_center) + local_window_padding),
    )

    if local_cal_upper <= local_cal_lower:
        raise ValueError("Invalid local calendar window for C14 calibration.")

    return (
        c14_bp,
        c14_error,
        cal_grid,
        c14_curve_age,
        c14_curve_sigma,
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
    sigma_mu_prior: float = 0.60,
    sigma_sigma_prior: float = 0.25,
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
    cal_span_sigma: float = 40.0,
    cal_span_lower: float = 15.0,
    cal_span_upper: float = 160.0,
    sigma_cal_link_mu: float = 35.0,
    sigma_cal_link_sigma: float = 15.0,
    sigma_cal_link_lower: float = 10.0,
    sigma_cal_link_upper: float = 80.0,
    c14_extra_sigma: float = 0.0,
    local_window_padding: float = 500.0,
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
            Individual calendar ages may deviate from this expectation.

    chronology_likelihood:
        "intcal20"
            Uses an explicit radiocarbon likelihood through the IntCal20 curve.

        "calibrated_gaussian"
            Uses a Gaussian approximation to already calibrated single-date
            posterior summaries. This is mainly a diagnostic / proof-of-concept
            mode to test whether the difficult sampling comes from the explicit
            calibration likelihood.

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

    c14_data_prepared = None
    gaussian_data_prepared = None

    local_cal_lower = None
    local_cal_upper = None

    if use_chronology:
        if chronology_likelihood == "intcal20":
            c14_data_prepared = _prepare_c14_inputs(
                c14_bp=c14_bp,
                c14_error=c14_error,
                intcal20_curve=intcal20_curve,
                n_graves=n_graves,
                local_window_padding=local_window_padding,
            )

            (
                c14_bp,
                c14_error,
                cal_grid,
                c14_curve_age,
                c14_curve_sigma,
                local_cal_lower,
                local_cal_upper,
            ) = c14_data_prepared

            rough_cal_bp = c14_bp + 300.0

            if cal_alpha_mu is None:
                cal_alpha_mu = float(np.nanmedian(rough_cal_bp))

            if cal_span_mu is None:
                q10, q90 = np.nanquantile(rough_cal_bp, [0.10, 0.90])
                cal_span_mu = float(np.clip((q90 - q10) / 4.0, 20.0, 120.0))

        elif chronology_likelihood == "calibrated_gaussian":
            gaussian_data_prepared = _prepare_calibrated_gaussian_inputs(
                calibrated_cal_bp_mean=calibrated_cal_bp_mean,
                calibrated_cal_bp_sd=calibrated_cal_bp_sd,
                n_graves=n_graves,
                local_window_padding=local_window_padding,
            )

            (
                calibrated_cal_bp_mean,
                calibrated_cal_bp_sd,
                local_cal_lower,
                local_cal_upper,
            ) = gaussian_data_prepared

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
        print("local_cal_lower:", round(float(local_cal_lower), 3))
        print("local_cal_upper:", round(float(local_cal_upper), 3))
        print("cal_alpha_mu:", round(float(cal_alpha_mu), 3))
        print("cal_alpha_sigma:", round(float(cal_alpha_sigma), 3))
        print("cal_span_mu:", round(float(cal_span_mu), 3))
        print("cal_span_sigma:", round(float(cal_span_sigma), 3))
        print("cal_span_lower:", round(float(cal_span_lower), 3))
        print("cal_span_upper:", round(float(cal_span_upper), 3))
        print(
            "sigma_cal_link prior:",
            (
                "TruncatedNormal("
                f"mu={sigma_cal_link_mu}, "
                f"sigma={sigma_cal_link_sigma}, "
                f"lower={sigma_cal_link_lower}, "
                f"upper={sigma_cal_link_upper})"
            ),
        )

        initvals.update(
            {
                "cal_alpha": float(cal_alpha_mu),
                "cal_span": float(cal_span_mu),
                "sigma_cal_link": float(
                    np.clip(
                        sigma_cal_link_mu,
                        sigma_cal_link_lower + 1e-3,
                        sigma_cal_link_upper - 1e-3,
                    )
                ),
                "latent_cal_bp_raw": np.zeros(n_graves),
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

            sigma_cal_link = pm.TruncatedNormal(
                "sigma_cal_link",
                mu=sigma_cal_link_mu,
                sigma=sigma_cal_link_sigma,
                lower=sigma_cal_link_lower,
                upper=sigma_cal_link_upper,
            )

            latent_cal_bp_raw = pm.Normal(
                "latent_cal_bp_raw",
                0.0,
                1.0,
                dims="grave",
            )

            latent_cal_bp_unbounded = (
                expected_cal_bp + sigma_cal_link * latent_cal_bp_raw
            )

            # Clipping is a pragmatic local-window guard. It is not ideal as a
            # long-term final parameterisation, but it keeps this diagnostic
            # model close to the previous working version.
            latent_cal_bp = pm.Deterministic(
                "latent_cal_bp",
                pt.clip(
                    latent_cal_bp_unbounded,
                    float(local_cal_lower) + 1e-3,
                    float(local_cal_upper) - 1e-3,
                ),
                dims="grave",
            )

            if chronology_likelihood == "intcal20":
                expected_c14_bp = pytensor_linear_interp(
                    latent_cal_bp,
                    cal_grid,
                    c14_curve_age,
                )

                curve_sigma = pytensor_linear_interp(
                    latent_cal_bp,
                    cal_grid,
                    c14_curve_sigma,
                )

                c14_extra_sigma_det = pm.Deterministic(
                    "c14_extra_sigma",
                    pt.as_tensor_variable(float(c14_extra_sigma)),
                )

                total_c14_sigma = pt.sqrt(
                    pt.as_tensor_variable(c14_error) ** 2
                    + curve_sigma**2
                    + c14_extra_sigma_det**2
                )

                pm.Normal(
                    "c14_bp_obs",
                    mu=expected_c14_bp,
                    sigma=total_c14_sigma,
                    observed=c14_bp,
                    dims="grave",
                )

            elif chronology_likelihood == "calibrated_gaussian":
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

        intercept = pm.Normal("intercept", mu=-0.5, sigma=0.8)

        mu = pm.Normal(
            "mu",
            mu=0.0,
            sigma=1.4,
            dims="type",
        )

        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu_prior,
            sigma=sigma_sigma_prior,
            lower=0.20,
            upper=1.40,
            dims="type",
        )

        a = pm.Normal(
            "a",
            mu=1.5,
            sigma=0.8,
            dims="type",
        )

        if include_richness:
            sigma_g = pm.HalfNormal("sigma_g", sigma=0.4)

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

    return idata