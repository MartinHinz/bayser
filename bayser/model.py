from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from bayser.model_helpers import (
    empty_c14_inputs,
    marginalised_intcal20_logp,
    prepare_c14_inputs,
    precompute_c14_loglikelihood_np,
    reconstruct_marginal_calendar_draws,
    spacing_repulsion,
    store_model_attrs,
    zscore_vector,
)


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
    outlier_prior_by_grave: np.ndarray | None = None,
    local_window_padding: float = 500.0,
    calendar_grid_step: int = 10,
    max_treedepth: int = 12,
    intercept_mu: float = -0.5,
    intercept_sigma: float = 1.2,
    mu_sigma: float = 2.0,
    sigma_mu_prior: float = 0.70,
    log_sigma_hyper_sigma: float = 0.70,
    log_sigma_hyper_sd_sigma: float = 0.35,
    a_mu_prior: float = 1.2,
    a_mu_sigma: float = 1.0,
    a_hyper_sd_sigma: float = 1.0,
    richness_sigma: float = 0.5,
) -> az.InferenceData:
    Y = np.asarray(Y)

    if Y.ndim != 2 or not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be a binary two-dimensional matrix.")

    if chronology_likelihood != "intcal20":
        raise ValueError("Only chronology_likelihood='intcal20' is supported.")

    if chronology_mode not in {"none", "c14_typology_linked"}:
        raise ValueError("chronology_mode must be 'none' or 'c14_typology_linked'.")

    if sigma_cal_link_upper <= sigma_cal_link_lower:
        raise ValueError("sigma_cal_link_upper must be larger than sigma_cal_link_lower.")

    n_graves, n_types = Y.shape

    # Fixed OxCal-style outlier component parameters.
    # These are deliberately not exposed through the public API/CLI for now.
    outlier_scale = 5.0
    outlier_nu = 5.0

    initvals = {
        "mu_log_sigma": float(np.log(sigma_mu_prior)),
        "sd_log_sigma": 0.35,
        "mu_a": float(a_mu_prior),
        "sd_a": 0.6,
    }

    # -------------------------------------------------------------------------
    # Orientation reference
    # -------------------------------------------------------------------------

    ref_z = None

    if orientation_reference is not None:
        ref_z = zscore_vector(orientation_reference, "orientation_reference")

        if len(ref_z) != n_graves:
            raise ValueError("orientation_reference must have one value per grave.")

        initvals["t_raw"] = ref_z

    # -------------------------------------------------------------------------
    # Optional outlier-prior preparation
    # -------------------------------------------------------------------------

    if outlier_prior_by_grave is None:
        outlier_prior_by_grave_arr = np.zeros(n_graves, dtype=float)
    else:
        outlier_prior_by_grave_arr = np.asarray(outlier_prior_by_grave, dtype=float)

        if outlier_prior_by_grave_arr.ndim != 1:
            raise ValueError("outlier_prior_by_grave must be one-dimensional.")

        if len(outlier_prior_by_grave_arr) != n_graves:
            raise ValueError(
                "outlier_prior_by_grave must have one value per grave "
                f"({n_graves} expected, got {len(outlier_prior_by_grave_arr)})."
            )

        if not np.all(np.isfinite(outlier_prior_by_grave_arr)):
            raise ValueError("outlier_prior_by_grave must contain only finite values.")

        if np.any((outlier_prior_by_grave_arr < 0.0) | (outlier_prior_by_grave_arr > 1.0)):
            raise ValueError("outlier_prior_by_grave values must lie between 0 and 1.")

    # -------------------------------------------------------------------------
    # Optional C14 preparation
    # -------------------------------------------------------------------------

    c14_requested = chronology_mode == "c14_typology_linked"

    if c14_requested:
        (
            c14_index,
            bp_dated,
            err_dated,
            cal_grid,
            curve_age,
            curve_sigma,
            weights,
            cal_lo,
            cal_hi,
        ) = prepare_c14_inputs(
            c14_bp,
            c14_error,
            intcal20_curve,
            n_graves,
            local_window_padding,
            calendar_grid_step,
        )
    else:
        (
            c14_index,
            bp_dated,
            err_dated,
            cal_grid,
            curve_age,
            curve_sigma,
            weights,
            cal_lo,
            cal_hi,
        ) = empty_c14_inputs()

    n_c14_dated = int(len(c14_index))
    use_chronology = c14_requested and n_c14_dated > 0

    log_c14_given_cal = None
    sigma_cal_link_logit_mu = None
    outlier_prior_dated = np.array([], dtype=float)
    outlier_prior_for_model = None
    use_outlier_model = False

    if use_chronology:
        outlier_prior_dated = outlier_prior_by_grave_arr[c14_index]
        use_outlier_model = bool(np.any(outlier_prior_dated > 0.0))
        outlier_prior_for_model = outlier_prior_dated if use_outlier_model else None

        log_c14_given_cal = precompute_c14_loglikelihood_np(
            bp_dated,
            err_dated,
            curve_age,
            curve_sigma,
            c14_extra_sigma=c14_extra_sigma,
        )

        rough_cal_bp = bp_dated + 300.0

        if cal_alpha_mu is None:
            cal_alpha_mu = float(np.nanmedian(rough_cal_bp))

        if cal_span_mu is None:
            q10, q90 = np.nanquantile(rough_cal_bp, [0.10, 0.90])
            cal_span_mu = float(np.clip((q90 - q10) / 4.0, 20.0, 120.0))

        # Smooth bounded parametrisation for sigma_cal_link:
        # unit value -> logit -> Normal on unconstrained scale -> sigmoid back.
        sigma_cal_link_mu_unit = (
            (sigma_cal_link_mu - sigma_cal_link_lower)
            / (sigma_cal_link_upper - sigma_cal_link_lower)
        )
        sigma_cal_link_mu_unit = float(
            np.clip(sigma_cal_link_mu_unit, 1e-4, 1.0 - 1e-4)
        )
        sigma_cal_link_logit_mu = float(
            np.log(sigma_cal_link_mu_unit / (1.0 - sigma_cal_link_mu_unit))
        )

        initvals.update(
            cal_alpha=float(cal_alpha_mu),
            cal_span=float(
                np.clip(
                    cal_span_mu,
                    cal_span_lower + 1e-3,
                    cal_span_upper - 1e-3,
                )
            ),
            logit_sigma_cal_link=sigma_cal_link_logit_mu,
        )

    coords = {
        "grave": np.arange(n_graves),
        "type": np.arange(n_types),
    }

    if use_chronology:
        coords["c14_grave"] = c14_index

    # Precompute upper-triangle pairs outside the PyMC graph.
    # This keeps the spacing-repulsion potential equivalent, but avoids building
    # a full symbolic n_graves x n_graves distance matrix and triangular mask.
    repulsion_pair_i, repulsion_pair_j = np.triu_indices(n_graves, k=1)

    if use_chronology:
        print(
            f"\nC14/calendar model: linked IntCal20, "
            f"dated graves {n_c14_dated}/{n_graves}, grid {len(cal_grid)}"
        )

        if use_outlier_model:
            print(
                "outlier model: enabled for "
                f"{int(np.sum(outlier_prior_dated > 0.0))}/{n_c14_dated} dated graves"
            )
        else:
            print("outlier model: disabled")
    elif c14_requested:
        print(
            "\nC14/calendar model requested, but no finite C14 dates were supplied. "
            "Using typology only."
        )

    print("typological model: hierarchical type widths and amplitudes")

    # -------------------------------------------------------------------------
    # PyMC model
    # -------------------------------------------------------------------------

    with pm.Model(coords=coords):
        # ------------------------------------------------------------------
        # Latent grave axis
        # ------------------------------------------------------------------

        t_raw = pm.Normal("t_raw", 0.0, 1.0, dims="grave")
        t0 = t_raw - pt.mean(t_raw)

        t = pm.Deterministic(
            "t",
            t0 / pt.sqrt(pt.var(t0) + 1e-6),
            dims="grave",
        )

        if ref_z is not None:
            score = pm.Deterministic(
                "orientation_score",
                pt.sum(t * pt.as_tensor_variable(ref_z)),
            )

            pm.Potential(
                "axis_orientation",
                pt.switch(score >= 0.0, 0.0, -np.inf),
            )

        repulsion_lp = pm.Deterministic(
            "repulsion_lp",
            spacing_repulsion(
                t,
                pair_i=repulsion_pair_i,
                pair_j=repulsion_pair_j,
                strength=repulsion_strength,
            ),
        )

        pm.Potential("spacing_repulsion", repulsion_lp)

        # ------------------------------------------------------------------
        # Optional linked chronology
        # ------------------------------------------------------------------

        if use_chronology:
            cal_alpha = pm.Normal(
                "cal_alpha",
                cal_alpha_mu,
                cal_alpha_sigma,
            )

            cal_span = pm.TruncatedNormal(
                "cal_span",
                cal_span_mu,
                cal_span_sigma,
                lower=cal_span_lower,
                upper=cal_span_upper,
            )

            cal_beta = pm.Deterministic("cal_beta", -cal_span)

            expected_cal_bp = pm.Deterministic(
                "expected_cal_bp",
                cal_alpha + cal_beta * t,
                dims="grave",
            )

            logit_sigma_cal_link = pm.Normal(
                "logit_sigma_cal_link",
                sigma_cal_link_logit_mu,
                1.0,
            )

            sigma_cal_link = pm.Deterministic(
                "sigma_cal_link",
                sigma_cal_link_lower
                + (sigma_cal_link_upper - sigma_cal_link_lower)
                * pm.math.sigmoid(logit_sigma_cal_link),
            )

            if use_outlier_model:
                pm.Deterministic(
                    "sigma_cal_outlier",
                    sigma_cal_link * outlier_scale,
                )

            logp, p_outlier, _, _ = marginalised_intcal20_logp(
                bp_dated,
                err_dated,
                expected_cal_bp[c14_index],
                sigma_cal_link,
                cal_grid,
                curve_age,
                curve_sigma,
                weights,
                c14_extra_sigma,
                outlier_prior=outlier_prior_for_model,
                outlier_scale=outlier_scale,
                outlier_nu=outlier_nu,
                log_c14_given_cal=log_c14_given_cal,
            )

            pm.Potential("c14_marginalised_likelihood", logp)

            if use_outlier_model and p_outlier is not None:
                pm.Deterministic(
                    "p_outlier",
                    p_outlier,
                    dims="c14_grave",
                )

        # ------------------------------------------------------------------
        # Typological observation model
        # ------------------------------------------------------------------

        intercept = pm.Normal(
            "intercept",
            intercept_mu,
            intercept_sigma,
        )

        mu = pm.Normal(
            "mu",
            0.0,
            mu_sigma,
            dims="type",
        )

        mu_log_sigma = pm.Normal(
            "mu_log_sigma",
            np.log(float(sigma_mu_prior)),
            log_sigma_hyper_sigma,
        )

        sd_log_sigma = pm.HalfNormal(
            "sd_log_sigma",
            log_sigma_hyper_sd_sigma,
        )

        log_sigma_raw = pm.Normal(
            "log_sigma_raw",
            0.0,
            1.0,
            dims="type",
        )

        sigma = pm.Deterministic(
            "sigma",
            pt.exp(mu_log_sigma + sd_log_sigma * log_sigma_raw),
            dims="type",
        )

        mu_a = pm.Normal(
            "mu_a",
            a_mu_prior,
            a_mu_sigma,
        )

        sd_a = pm.HalfNormal(
            "sd_a",
            a_hyper_sd_sigma,
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
            sigma_g = pm.HalfNormal("sigma_g", richness_sigma)

            g_raw = pm.Normal(
                "g_raw",
                0.0,
                1.0,
                dims="grave",
            )

            grave_effect = pm.Deterministic(
                "g",
                sigma_g * (g_raw - pt.mean(g_raw)),
                dims="grave",
            )[:, None]
        else:
            grave_effect = 0.0

        eta = (
            intercept
            + a[None, :]
            + grave_effect
            - ((t[:, None] - mu[None, :]) ** 2) / (2.0 * sigma[None, :] ** 2)
        )

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
            initvals=initvals,
            nuts_sampler_kwargs={"max_treedepth": max_treedepth},
        )

    # -------------------------------------------------------------------------
    # Post-hoc reconstruction of individual calendar draws
    # -------------------------------------------------------------------------

    if use_chronology:
        idata = reconstruct_marginal_calendar_draws(
            idata,
            c14_index,
            bp_dated,
            err_dated,
            cal_grid,
            curve_age,
            curve_sigma,
            weights,
            c14_extra_sigma=c14_extra_sigma,
            random_seed=random_seed,
            outlier_prior=outlier_prior_for_model,
            outlier_scale=outlier_scale,
            outlier_nu=outlier_nu,
            log_c14_given_cal=log_c14_given_cal,
        )

    return store_model_attrs(
        idata,
        n_graves=n_graves,
        n_types=n_types,
        n_c14_dated=n_c14_dated,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        max_treedepth=max_treedepth,
        include_richness=include_richness,
        repulsion_strength=repulsion_strength,
        chronology_mode=chronology_mode,
        chronology_likelihood=chronology_likelihood,
        use_chronology=use_chronology,
        calendar_grid_step=calendar_grid_step,
        local_window_padding=local_window_padding,
        local_cal_lower=cal_lo,
        local_cal_upper=cal_hi,
        use_outlier_model=use_outlier_model,
        n_outlier_candidates=int(np.sum(outlier_prior_dated > 0.0)),
        outlier_prior_sum=float(np.sum(outlier_prior_dated)),
        outlier_scale=outlier_scale,
        outlier_nu=outlier_nu,
    )