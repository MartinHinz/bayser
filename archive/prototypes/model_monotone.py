from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from bayser.calibration import pytensor_linear_interp


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


def infer_cal_span_prior_from_reference_calibration(
    row_scores: np.ndarray,
    unmodelled_cal_bp: np.ndarray,
    min_n: int = 8,
    min_span: float = 20.0,
    max_span: float = 160.0,
    min_sigma: float = 20.0,
    max_sigma: float = 80.0,
) -> tuple[float, float]:
    """Infer a weakly data-informed cal_span prior.

    The model uses a standardised latent typological axis t and, in the
    linear case,

        expected_cal_bp = cal_alpha - cal_span * t

    Therefore cal_span is approximately the number of calendar years per
    one standardised unit of the typological axis.

    In the monotone model below, cal_span is retained as the average calendar
    slope per one standardised t-unit.
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
    # median of all pairwise slopes. For n ~ 50 this is trivial.
    dx = xz[:, None] - xz[None, :]
    dy = y[:, None] - y[None, :]

    upper = np.triu(np.ones(dx.shape, dtype=bool), k=1)
    valid = upper & np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-8)

    slopes = dy[valid] / dx[valid]
    slopes = slopes[np.isfinite(slopes)]

    if len(slopes) == 0:
        return 60.0, 40.0

    span_mu = float(np.clip(abs(np.median(slopes)), min_span, max_span))

    # Use a robust slope spread, but deliberately keep it broad.
    q25, q75 = np.quantile(np.abs(slopes), [0.25, 0.75])
    robust_width = float(abs(q75 - q25))

    span_sigma = max(min_sigma, 0.5 * span_mu, robust_width)
    span_sigma = float(np.clip(span_sigma, min_sigma, max_sigma))

    return span_mu, span_sigma


def pytensor_piecewise_linear_fixed_x(
    x: pt.TensorVariable,
    x_knots: np.ndarray,
    y_knots: pt.TensorVariable,
) -> pt.TensorVariable:
    """Piecewise-linear interpolation with fixed x-knots and tensor y-knots.

    Parameters
    ----------
    x:
        PyTensor vector of values to interpolate.
    x_knots:
        Fixed strictly increasing NumPy array of knot positions.
    y_knots:
        PyTensor vector of knot values.

    Notes
    -----
    This is intentionally simple and explicit for the proof of concept.
    Values outside the knot range are extrapolated by the first/last interval.
    Since most standardised t values should fall roughly within [-2.5, 2.5],
    this is acceptable for a first test.
    """

    x_knots = np.asarray(x_knots, dtype=float)

    if x_knots.ndim != 1:
        raise ValueError("x_knots must be one-dimensional.")

    if len(x_knots) < 2:
        raise ValueError("At least two x-knots are required.")

    if np.any(np.diff(x_knots) <= 0):
        raise ValueError("x_knots must be strictly increasing.")

    out = pt.zeros_like(x)

    for k in range(len(x_knots) - 1):
        x0 = float(x_knots[k])
        x1 = float(x_knots[k + 1])

        y0 = y_knots[k]
        y1 = y_knots[k + 1]

        w = (x - x0) / (x1 - x0)
        y = y0 + w * (y1 - y0)

        if k == 0:
            mask = x <= x1
        elif k == len(x_knots) - 2:
            mask = x > x0
        else:
            mask = (x > x0) & (x <= x1)

        out = pt.switch(mask, y, out)

    return out


def _prepare_c14_inputs(
    c14_bp: np.ndarray | None,
    c14_error: np.ndarray | None,
    intcal20_curve: pd.DataFrame | None,
    n_graves: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Validate and prepare C14 data and IntCal20 curve arrays."""

    if c14_bp is None or c14_error is None:
        raise ValueError(
            "chronology_mode='c14_typology_linked' requires c14_bp and c14_error."
        )

    if intcal20_curve is None:
        raise ValueError(
            "chronology_mode='c14_typology_linked' requires intcal20_curve."
        )

    c14_bp = np.asarray(c14_bp, dtype=float)
    c14_error = np.asarray(c14_error, dtype=float)

    if len(c14_bp) != n_graves or len(c14_error) != n_graves:
        raise ValueError(
            "C14 arrays must have the same length as the number of graves."
        )

    if not np.all(np.isfinite(c14_bp)) or not np.all(np.isfinite(c14_error)):
        raise ValueError("C14 BP and error arrays must contain only finite values.")

    if np.any(c14_error <= 0):
        raise ValueError("C14 errors must be positive.")

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

    # Broad but local enough for this dataset. This is not a calibration prior;
    # it only prevents the sampler from wandering through irrelevant parts of
    # the full 0–55 ka calibration curve.
    rough_center = c14_bp + 300.0

    local_cal_lower = max(
        float(cal_grid.min()),
        float(np.nanmin(rough_center) - 1_200.0),
    )
    local_cal_upper = min(
        float(cal_grid.max()),
        float(np.nanmax(rough_center) + 1_200.0),
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
    c14_bp: np.ndarray | None = None,
    c14_error: np.ndarray | None = None,
    intcal20_curve: pd.DataFrame | None = None,
    chronology_mode: str = "none",
    orientation_reference: np.ndarray | None = None,
    cal_alpha_mu: float | None = None,
    cal_alpha_sigma: float = 120.0,
    cal_span_mu: float | None = None,
    cal_span_sigma: float = 40.0,
    cal_span_lower: float = 15.0,
    cal_span_upper: float = 160.0,
    sigma_cal_link_prior: float = 30.0,
) -> az.InferenceData:
    """Fit a parametric PyMC seriation model.

    chronology_mode:
        "none"
            Pure typological seriation.

        "c14_typology_linked"
            Typological position defines an expected calendar position.
            Individual calendar ages may deviate from this expectation, but are
            jointly constrained by the typological sequence and the explicit
            IntCal20 radiocarbon likelihood.

    This version is a proof-of-concept monotone calendar-depth model.

    Instead of

        expected_cal_bp = cal_alpha - cal_span * t

    it uses a monotone, piecewise-linear mapping

        expected_cal_bp = f(t)

    where f is constrained to decrease with increasing t, because increasing t
    means typologically later / younger, while larger cal BP means older.
    """

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

    use_c14_model = chronology_mode == "c14_typology_linked"

    # ------------------------------------------------------------------
    # Hard-coded proof-of-concept calendar-map settings
    # ------------------------------------------------------------------
    calendar_map = "monotone"
    n_calendar_knots = 5
    calendar_warp_alpha = 50.0

    initvals: dict[str, object] = {}

    orientation_ref_z = None
    if orientation_reference is not None:
        orientation_ref_z = _standardise_reference(orientation_reference)

        if len(orientation_ref_z) != n_graves:
            raise ValueError(
                "orientation_reference must have the same length as the number of graves."
            )

        initvals["t_raw"] = orientation_ref_z

    if use_c14_model:
        (
            c14_bp,
            c14_error,
            cal_grid,
            c14_curve_age,
            c14_curve_sigma,
            local_cal_lower,
            local_cal_upper,
        ) = _prepare_c14_inputs(
            c14_bp=c14_bp,
            c14_error=c14_error,
            intcal20_curve=intcal20_curve,
            n_graves=n_graves,
        )

        rough_cal_bp = c14_bp + 300.0

        if cal_alpha_mu is None:
            cal_alpha_mu = float(np.nanmedian(rough_cal_bp))

        if cal_span_mu is None:
            # t is standardised, so one t-unit is not the entire sequence.
            # This gives a weak prior centre for calendar years per t-unit.
            q10, q90 = np.nanquantile(rough_cal_bp, [0.10, 0.90])
            cal_span_mu = float(np.clip((q90 - q10) / 4.0, 20.0, 120.0))

        print("\nC14 calendar-depth model used:")
        print("calendar_map:", calendar_map)
        print("n_calendar_knots:", n_calendar_knots)
        print("calendar_warp_alpha:", calendar_warp_alpha)
        print("cal_span_mu:", round(float(cal_span_mu), 3))
        print("cal_span_sigma:", round(float(cal_span_sigma), 3))
        print("cal_span_lower:", round(float(cal_span_lower), 3))
        print("cal_span_upper:", round(float(cal_span_upper), 3))
        print("sigma_cal_link prior: TruncatedNormal(mu=35, sigma=15, lower=10, upper=80)")

        initvals.update(
            {
                "cal_alpha": float(cal_alpha_mu),
                "cal_total_span": float(cal_span_mu) * 5.0,
                "sigma_cal_link": min(float(sigma_cal_link_prior), 60.0),
                "latent_cal_bp_raw": np.zeros(n_graves),
            }
        )

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
        # Optional linked C14 chronology model
        # ------------------------------------------------------------------

        if use_c14_model:
            #cal_alpha = pm.Normal(
            #    "cal_alpha",
            #    mu=cal_alpha_mu,
            #    sigma=cal_alpha_sigma,
            #)

            cal_alpha = pm.Deterministic(
                "cal_alpha",
                pt.as_tensor_variable(float(cal_alpha_mu)),
            )

            if calendar_map == "linear":
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

            elif calendar_map == "monotone":
                # Fixed knot positions on the standardised typological axis.
                #
                # t is standardised, so most graves should lie roughly between
                # -2.5 and 2.5. The interval is intentionally broad enough for
                # a first proof of concept without requiring workflow changes.
                t_knot_values = np.linspace(-2.5, 2.5, n_calendar_knots)
                t_knot_range = float(t_knot_values[-1] - t_knot_values[0])

                # cal_span_mu still means average calendar years per one
                # standardised t-unit. The monotone model converts this into a
                # total calendar depth over the full knot range.
                total_span_mu = float(cal_span_mu) * t_knot_range
                total_span_sigma = float(cal_span_sigma) * t_knot_range
                total_span_lower = float(cal_span_lower) * t_knot_range
                total_span_upper = float(cal_span_upper) * t_knot_range

                #cal_total_span = pm.TruncatedNormal(
                #    "cal_total_span",
                #    mu=total_span_mu,
                #    sigma=total_span_sigma,
                #    lower=total_span_lower,
                #    upper=total_span_upper,
                #)

                cal_total_span = pm.Deterministic(
                    "cal_total_span",
                    pt.as_tensor_variable(float(total_span_mu)),
                )

                # Average years per one standardised t-unit.
                # This keeps the old cal_span diagnostic interpretable.
                cal_span = pm.Deterministic(
                    "cal_span",
                    cal_total_span / t_knot_range,
                )

                cal_beta = pm.Deterministic("cal_beta", -cal_span)

                # Positive interval fractions. A relatively high alpha keeps
                # the curve close to linear at first, but still permits local
                # acceleration/deceleration of typological change.
                cal_interval_frac = pm.Dirichlet(
                    "cal_interval_frac",
                    a=np.full(n_calendar_knots - 1, calendar_warp_alpha),
                    shape=n_calendar_knots - 1,
                )

                cal_interval_lengths = pm.Deterministic(
                    "cal_interval_lengths",
                    cal_total_span * cal_interval_frac,
                )

                # Larger cal BP means older.
                # Increasing t means younger.
                # Therefore knot calendar values must decrease from left to right.
                oldest_knot = cal_alpha + 0.5 * cal_total_span

                cumulative_drop = pt.concatenate(
                    [
                        pt.zeros(1),
                        pt.cumsum(cal_interval_lengths),
                    ]
                )

                cal_knot_bp = pm.Deterministic(
                    "cal_knot_bp",
                    oldest_knot - cumulative_drop,
                )

                expected_cal_bp = pm.Deterministic(
                    "expected_cal_bp",
                    pytensor_piecewise_linear_fixed_x(
                        x=t,
                        x_knots=t_knot_values,
                        y_knots=cal_knot_bp,
                    ),
                    dims="grave",
                )

            else:
                raise ValueError("calendar_map must be either 'linear' or 'monotone'.")

            #sigma_cal_link = pm.TruncatedNormal(
            #    "sigma_cal_link",
            #    mu=35.0,
            #    sigma=15.0,
            #    lower=10.0,
            #    upper=80.0,
            #)

            sigma_cal_link = pm.Deterministic(
                "sigma_cal_link",
                pt.as_tensor_variable(40.0),
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

            latent_cal_bp = pm.Deterministic(
                "latent_cal_bp",
                pt.clip(
                    latent_cal_bp_unbounded,
                    local_cal_lower + 1e-3,
                    local_cal_upper - 1e-3,
                ),
                dims="grave",
            )

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

            c14_extra_sigma = pm.Deterministic(
                "c14_extra_sigma",
                pt.as_tensor_variable(0.0),
            )

            total_c14_sigma = pt.sqrt(
                pt.as_tensor_variable(c14_error) ** 2
                + curve_sigma**2
                + c14_extra_sigma**2
            )

            pm.Normal(
                "c14_bp_obs",
                mu=expected_c14_bp,
                sigma=total_c14_sigma,
                observed=c14_bp,
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
            nuts_sampler_kwargs={"max_treedepth": 12},
        )

    return idata