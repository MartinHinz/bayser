from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
# Model build context
# -----------------------------------------------------------------------------


@dataclass
class SeriationModelContext:
    """Objects needed outside the PyMC graph for sampling and post-processing."""

    initvals: dict[str, Any]

    n_graves: int
    n_types: int
    n_c14_dated: int

    draws: int
    tune: int
    chains: int
    target_accept: float
    random_seed: int
    max_treedepth: int

    include_richness: bool
    repulsion_strength: float

    chronology_mode: str
    chronology_likelihood: str
    use_chronology: bool

    calendar_grid_step: int
    local_window_padding: float
    local_cal_lower: float | None
    local_cal_upper: float | None

    c14_index: np.ndarray
    bp_dated: np.ndarray
    err_dated: np.ndarray
    cal_grid: np.ndarray
    curve_age: np.ndarray
    curve_sigma: np.ndarray
    weights: np.ndarray
    c14_extra_sigma: float

    outlier_prior_for_model: np.ndarray | None
    outlier_prior_dated: np.ndarray
    use_outlier_model: bool
    outlier_scale: float
    outlier_nu: float

    log_c14_given_cal: np.ndarray | None


# -----------------------------------------------------------------------------
# Progress callback
# -----------------------------------------------------------------------------


class JsonProgressCallback:
    """Write PyMC sampling progress to a JSON file.

    This is intended for lightweight frontends such as Streamlit. It does not
    depend on PyMC's textual progress bar and can therefore be used while stdout
    is suppressed by the CLI workflow.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        draws: int,
        tune: int,
        chains: int,
        throttle_seconds: float = 0.25,
    ) -> None:
        self.path = Path(path)
        self.draws = int(draws)
        self.tune = int(tune)
        self.chains = int(chains)
        self.throttle_seconds = float(throttle_seconds)

        self.total = max(1, self.chains * (self.draws + self.tune))
        self.done = 0
        self.last_write = 0.0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(status="initialising", phase="initialising")

    def __call__(self, trace, draw) -> None:
        """PyMC callback signature: callback(trace, draw)."""

        self.done += 1

        now = time.time()
        if now - self.last_write < self.throttle_seconds and self.done < self.total:
            return

        tuning = bool(getattr(draw, "tuning", False))
        phase = "tuning" if tuning else "sampling"

        self._write(
            status="running",
            phase=phase,
            chain=getattr(draw, "chain", None),
            draw=getattr(draw, "draw_idx", getattr(draw, "draw", None)),
            tuning=tuning,
        )

    def finish(self) -> None:
        self.done = self.total
        self._write(status="finished", phase="finished")

    def fail(self, message: str) -> None:
        self._write(status="failed", phase="failed", message=message)

    def _write(
        self,
        *,
        status: str,
        phase: str | None = None,
        chain: int | None = None,
        draw: int | None = None,
        tuning: bool | None = None,
        message: str | None = None,
    ) -> None:
        self.last_write = time.time()

        progress = min(1.0, max(0.0, self.done / self.total))

        payload: dict[str, Any] = {
            "status": status,
            "phase": phase,
            "done": int(self.done),
            "total": int(self.total),
            "progress": float(progress),
            "percent": float(progress * 100.0),
            "chains": int(self.chains),
            "draws": int(self.draws),
            "tune": int(self.tune),
            "chain": chain,
            "draw": draw,
            "tuning": tuning,
            "updated_at": self.last_write,
        }

        if message is not None:
            payload["message"] = message

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _validate_binary_matrix(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y)

    if Y.ndim != 2 or not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be a binary two-dimensional matrix.")

    return Y


def _prepare_outlier_prior(
    outlier_prior_by_grave: np.ndarray | None,
    n_graves: int,
) -> np.ndarray:
    if outlier_prior_by_grave is None:
        return np.zeros(n_graves, dtype=float)

    out = np.asarray(outlier_prior_by_grave, dtype=float)

    if out.ndim != 1:
        raise ValueError("outlier_prior_by_grave must be one-dimensional.")

    if len(out) != n_graves:
        raise ValueError(
            "outlier_prior_by_grave must have one value per assemblage "
            f"({n_graves} expected, got {len(out)})."
        )

    if not np.all(np.isfinite(out)):
        raise ValueError("outlier_prior_by_grave must contain only finite values.")

    if np.any((out < 0.0) | (out > 1.0)):
        raise ValueError("outlier_prior_by_grave values must lie between 0 and 1.")

    return out


# -----------------------------------------------------------------------------
# Build PyMC model
# -----------------------------------------------------------------------------


def build_parametric_pymc_seriation_model(
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
    print_model_summary: bool = True,
) -> tuple[pm.Model, SeriationModelContext]:
    """Build the Bayser PyMC model without sampling it.

    This function is useful for generating model graphs with
    `pm.model_to_graphviz(model)` and for separating model construction from
    inference.
    """

    Y = _validate_binary_matrix(Y)

    if chronology_likelihood != "intcal20":
        raise ValueError("Only chronology_likelihood='intcal20' is supported.")

    if chronology_mode not in {"none", "c14_typology_linked"}:
        raise ValueError("chronology_mode must be 'none' or 'c14_typology_linked'.")

    if sigma_cal_link_upper <= sigma_cal_link_lower:
        raise ValueError(
            "sigma_cal_link_upper must be larger than sigma_cal_link_lower."
        )

    n_graves, n_types = Y.shape

    # Fixed outlier-mixture parameters. These are deliberately not exposed through
    # the public API/CLI for now.
    outlier_scale = 5.0
    outlier_nu = 5.0

    initvals: dict[str, Any] = {
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
            raise ValueError("orientation_reference must have one value per assemblage.")

        initvals["t_raw"] = ref_z

    # -------------------------------------------------------------------------
    # Optional outlier-prior preparation
    # -------------------------------------------------------------------------

    outlier_prior_by_grave_arr = _prepare_outlier_prior(
        outlier_prior_by_grave,
        n_graves,
    )

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
        sigma_cal_link_mu_unit = (sigma_cal_link_mu - sigma_cal_link_lower) / (
            sigma_cal_link_upper - sigma_cal_link_lower
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

    coords: dict[str, Any] = {
        "grave": np.arange(n_graves),
        "type": np.arange(n_types),
    }

    if use_chronology:
        coords["c14_grave"] = c14_index

    # Precompute upper-triangle pairs outside the PyMC graph.
    # This keeps the spacing-repulsion potential equivalent, but avoids building
    # a full symbolic n_graves x n_graves distance matrix and triangular mask.
    repulsion_pair_i, repulsion_pair_j = np.triu_indices(n_graves, k=1)

    if print_model_summary:
        if use_chronology:
            print(
                f"\nC14/calendar model: linked IntCal20, "
                f"dated assemblages {n_c14_dated}/{n_graves}, grid {len(cal_grid)}"
            )

            if use_outlier_model:
                print(
                    "outlier model: enabled for "
                    f"{int(np.sum(outlier_prior_dated > 0.0))}/{n_c14_dated} dated assemblages"
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

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------------
        # Latent assemblage axis
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

            orientation_strength = 20.0

            pm.Potential(
                "axis_orientation",
                pm.math.log(pm.math.sigmoid(orientation_strength * score)),
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

    context = SeriationModelContext(
        initvals=initvals,
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
        c14_index=c14_index,
        bp_dated=bp_dated,
        err_dated=err_dated,
        cal_grid=cal_grid,
        curve_age=curve_age,
        curve_sigma=curve_sigma,
        weights=weights,
        c14_extra_sigma=c14_extra_sigma,
        outlier_prior_for_model=outlier_prior_for_model,
        outlier_prior_dated=outlier_prior_dated,
        use_outlier_model=use_outlier_model,
        outlier_scale=outlier_scale,
        outlier_nu=outlier_nu,
        log_c14_given_cal=log_c14_given_cal,
    )

    return model, context


# -----------------------------------------------------------------------------
# Main fitting wrapper
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
    return_model: bool = False,
    progress_file: str | Path | None = None,
) -> az.InferenceData | tuple[az.InferenceData, pm.Model]:
    """Fit the Bayser PyMC model.

    If `return_model=True`, return `(idata, model)`. This is useful when the
    caller wants to inspect or visualise the PyMC model after fitting. Existing
    callers that expect only `idata` remain unaffected.

    If `progress_file` is supplied, sampling progress is written to this JSON
    file via a PyMC callback. This is intended for web frontends and avoids
    relying on PyMC's terminal progress bar.
    """

    model, context = build_parametric_pymc_seriation_model(
        Y=Y,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        include_richness=include_richness,
        repulsion_strength=repulsion_strength,
        chronology_mode=chronology_mode,
        chronology_likelihood=chronology_likelihood,
        orientation_reference=orientation_reference,
        c14_bp=c14_bp,
        c14_error=c14_error,
        intcal20_curve=intcal20_curve,
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
        outlier_prior_by_grave=outlier_prior_by_grave,
        local_window_padding=local_window_padding,
        calendar_grid_step=calendar_grid_step,
        max_treedepth=max_treedepth,
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
        print_model_summary=True,
    )

    progress_callback = None
    if progress_file is not None:
        progress_callback = JsonProgressCallback(
            progress_file,
            draws=context.draws,
            tune=context.tune,
            chains=context.chains,
        )

    try:
        with model:
            idata = pm.sample(
                draws=context.draws,
                tune=context.tune,
                chains=context.chains,
                target_accept=context.target_accept,
                random_seed=context.random_seed,
                return_inferencedata=True,
                init="jitter+adapt_diag",
                initvals=context.initvals,
                progressbar=progress_callback is None,
                callback=progress_callback,
                nuts={"max_treedepth": context.max_treedepth},
            )

        if progress_callback is not None:
            progress_callback.finish()

    except Exception as exc:
        if progress_callback is not None:
            progress_callback.fail(str(exc))
        raise

    # -------------------------------------------------------------------------
    # Post-hoc reconstruction of individual calendar draws
    # -------------------------------------------------------------------------

    if context.use_chronology:
        idata = reconstruct_marginal_calendar_draws(
            idata,
            context.c14_index,
            context.bp_dated,
            context.err_dated,
            context.cal_grid,
            context.curve_age,
            context.curve_sigma,
            context.weights,
            c14_extra_sigma=context.c14_extra_sigma,
            random_seed=context.random_seed,
            outlier_prior=context.outlier_prior_for_model,
            outlier_scale=context.outlier_scale,
            outlier_nu=context.outlier_nu,
            log_c14_given_cal=context.log_c14_given_cal,
        )

    idata = store_model_attrs(
        idata,
        n_graves=context.n_graves,
        n_types=context.n_types,
        n_c14_dated=context.n_c14_dated,
        draws=context.draws,
        tune=context.tune,
        chains=context.chains,
        target_accept=context.target_accept,
        random_seed=context.random_seed,
        max_treedepth=context.max_treedepth,
        include_richness=context.include_richness,
        repulsion_strength=context.repulsion_strength,
        chronology_mode=context.chronology_mode,
        chronology_likelihood=context.chronology_likelihood,
        use_chronology=context.use_chronology,
        calendar_grid_step=context.calendar_grid_step,
        local_window_padding=context.local_window_padding,
        local_cal_lower=context.local_cal_lower,
        local_cal_upper=context.local_cal_upper,
        use_outlier_model=context.use_outlier_model,
        n_outlier_candidates=int(np.sum(context.outlier_prior_dated > 0.0)),
        outlier_prior_sum=float(np.sum(context.outlier_prior_dated)),
        outlier_scale=context.outlier_scale,
        outlier_nu=context.outlier_nu,
    )

    if return_model:
        return idata, model

    return idata
