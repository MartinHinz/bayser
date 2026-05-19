from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Input diagnostics
# -----------------------------------------------------------------------------


def print_matrix_diagnostics(Y: np.ndarray, label: str = "Matrix") -> None:
    Y = np.asarray(Y)
    type_counts = Y.sum(axis=0)
    grave_counts = Y.sum(axis=1)

    print(f"{label} shape:", Y.shape)
    print("Overall presence rate:", round(float(Y.mean()), 3))
    print(
        "Type frequencies, min/median/max:",
        int(type_counts.min()),
        float(np.median(type_counts)),
        int(type_counts.max()),
    )
    print(
        "Assemblage richness, min/median/max:",
        int(grave_counts.min()),
        float(np.median(grave_counts)),
        int(grave_counts.max()),
    )


def print_type_diagnostics(
    Y: np.ndarray,
    type_ids: list[str],
    n: int = 25,
) -> None:
    counts = np.asarray(Y).sum(axis=0)
    table = (
        pd.DataFrame({"type_id": type_ids, "frequency": counts})
        .sort_values(["frequency", "type_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("\nType diagnostics:")
    print("Retained type columns:", len(type_ids))

    print("\nMost frequent retained types:")
    print(table.head(n).to_string(index=False))

    print("\nLeast frequent retained types:")
    print(table.tail(min(n, len(table))).to_string(index=False))


def print_c14_diagnostics(data) -> None:
    if data.c14_bp is None or data.c14_error is None:
        print("C14 dates: none supplied")
        return

    c14_bp = np.asarray(data.c14_bp, dtype=float)
    c14_error = np.asarray(data.c14_error, dtype=float)
    finite = np.isfinite(c14_bp) & np.isfinite(c14_error)

    print(
        "C14 columns:", {"id": data.id_col, "bp": data.bp_col, "error": data.error_col}
    )
    print("C14 dates, finite/total:", int(finite.sum()), "/", len(c14_bp))

    if not finite.any():
        return

    print(
        "BP min/median/max:",
        round(float(np.nanmin(c14_bp[finite])), 1),
        round(float(np.nanmedian(c14_bp[finite])), 1),
        round(float(np.nanmax(c14_bp[finite])), 1),
    )
    print(
        "C14 error min/median/max:",
        round(float(np.nanmin(c14_error[finite])), 1),
        round(float(np.nanmedian(c14_error[finite])), 1),
        round(float(np.nanmax(c14_error[finite])), 1),
    )

    table = pd.DataFrame(
        {
            "grave_id": data.grave_ids,
            "c14_bp": c14_bp,
            "c14_error": c14_error,
        }
    ).loc[finite]

    print("\nOldest retained C14 dates:")
    print(table.sort_values("c14_bp", ascending=False).head(10).to_string(index=False))

    print("\nYoungest retained C14 dates:")
    print(table.sort_values("c14_bp", ascending=True).head(10).to_string(index=False))

    print("Removed assemblages during preparation:", len(data.removed_grave_ids))
    if data.removed_grave_ids:
        print("Removed assemblages, first 20:", ", ".join(data.removed_grave_ids[:20]))

    print("Removed types during preparation:", len(data.removed_type_ids))
    if data.removed_type_ids:
        print("Removed types, first 20:", ", ".join(data.removed_type_ids[:20]))

    if getattr(data, "unmatched_c14_ids", None):
        print("Unmatched C14 IDs:", len(data.unmatched_c14_ids))
        print("Unmatched C14 IDs, first 20:", ", ".join(data.unmatched_c14_ids[:20]))

    print_type_diagnostics(data.Y, data.type_ids)


# -----------------------------------------------------------------------------
# Posterior helpers
# -----------------------------------------------------------------------------


def _has(idata: az.InferenceData, group: str, var: str) -> bool:
    return hasattr(idata, group) and var in getattr(idata, group)


def _posterior_has(idata: az.InferenceData, var: str) -> bool:
    return _has(idata, "posterior", var)


def _sample_stats_has(idata: az.InferenceData, var: str) -> bool:
    return _has(idata, "sample_stats", var)


def _draws(idata: az.InferenceData, var: str) -> np.ndarray:
    x = idata.posterior[var].values
    if x.ndim == 2:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0] * x.shape[1], -1)


def _nonconstant_vars(idata: az.InferenceData, vars_: list[str]) -> list[str]:
    out = []

    for var in vars_:
        if not _posterior_has(idata, var):
            continue

        x = _draws(idata, var)
        if x.size and np.nanmax(np.nanstd(x, axis=0)) > 1e-12:
            out.append(var)

    return out


def _summary(idata: az.InferenceData, vars_: list[str]) -> pd.DataFrame | None:
    vars_ = _nonconstant_vars(idata, vars_)

    if not vars_:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return az.summary(idata, var_names=vars_)


def _mean(idata: az.InferenceData, var: str) -> np.ndarray | None:
    return _draws(idata, var).mean(axis=0) if _posterior_has(idata, var) else None


def _sd(idata: az.InferenceData, var: str) -> np.ndarray | None:
    return _draws(idata, var).std(axis=0) if _posterior_has(idata, var) else None


def _optional_vector(x, n: int, name: str) -> np.ndarray | None:
    if x is None:
        return None

    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(arr) != n:
        raise ValueError(f"{name} must have length {n}, got {len(arr)}.")

    return arr


def _posterior_rank(idata: az.InferenceData) -> np.ndarray | None:
    t = _mean(idata, "t")
    if t is None:
        return None

    order = np.argsort(t)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def _c14_indices_from_da(
    idata: az.InferenceData, var: str, n_c14: int, n_graves: int
) -> np.ndarray:
    da = idata.posterior[var]

    if "c14_grave" in da.coords:
        idx = np.asarray(da.coords["c14_grave"].values, dtype=int)
        if len(idx) == n_c14:
            return idx

    if n_c14 == n_graves:
        return np.arange(n_graves, dtype=int)

    return np.arange(n_c14, dtype=int)


def _print_range(idata: az.InferenceData, var: str, label: str, unit: str = "") -> None:
    if not _posterior_has(idata, var):
        return

    x = _draws(idata, var)
    m = x.mean(axis=0)
    s = x.std(axis=0)

    print(f"\n{label}:")
    print(
        "Mean range:",
        round(float(np.nanmin(m)), 1),
        "to",
        round(float(np.nanmax(m)), 1),
        unit,
    )
    print(
        "SD min/median/max:",
        round(float(np.nanmin(s)), 1),
        round(float(np.nanmedian(s)), 1),
        round(float(np.nanmax(s)), 1),
        unit,
    )


# -----------------------------------------------------------------------------
# Core posterior diagnostics
# -----------------------------------------------------------------------------


def print_prior_metadata(idata: az.InferenceData) -> None:
    if not getattr(idata, "attrs", None):
        return

    keys = [
        "model_name",
        "model_version",
        "n_graves",
        "n_types",
        "n_c14_dated",
        "chronology_mode",
        "chronology_likelihood",
        "calendar_grid_step",
        "local_window_padding",
        "local_cal_lower",
        "local_cal_upper",
        "cal_alpha_mu",
        "cal_alpha_sigma",
        "cal_span_mu",
        "cal_span_sigma",
        "sigma_cal_link_mu",
        "sigma_cal_link_sigma",
        "use_outlier_model",
        "outlier_scale",
        "outlier_nu",
        "repulsion_strength",
        "include_richness",
        "target_accept",
        "max_treedepth",
    ]

    rows = [{"setting": k, "value": idata.attrs[k]} for k in keys if k in idata.attrs]

    if rows:
        print("\nModel settings:")
        print(pd.DataFrame(rows).to_string(index=False))


def print_scalar_diagnostics(
    idata: az.InferenceData, include_richness: bool = True
) -> None:
    vars_ = ["intercept"]

    if include_richness:
        vars_.append("sigma_g")

    vars_.extend(
        [
            "cal_alpha",
            "cal_span",
            "cal_beta",
            "sigma_cal_link",
            "sigma_cal_outlier",
            "mu_log_sigma",
            "sd_log_sigma",
            "mu_a",
            "sd_a",
        ]
    )

    s = _summary(idata, vars_)
    if s is not None:
        print("\nMain scalar / hyperparameter diagnostics:")
        print(s)


def print_vector_diagnostics(idata: az.InferenceData) -> None:
    vars_ = _nonconstant_vars(idata, ["t", "mu", "sigma", "a", "g"])

    if vars_:
        s = _summary(idata, vars_)
        rhat = s["r_hat"].replace([np.inf, -np.inf], np.nan)
        ess = s["ess_bulk"].replace([np.inf, -np.inf], np.nan)

        print("\nVector parameter diagnostics:")
        print("Worst R-hat:", round(float(rhat.max()), 3))
        print("Lowest bulk ESS:", round(float(ess.min()), 1))

    for var, label in [
        ("mu", "Type positions mu"),
        ("sigma", "Type career widths sigma"),
        ("a", "Type amplitudes a"),
    ]:
        _print_range(idata, var, label)


def print_calendar_ranges(idata: az.InferenceData) -> None:
    _print_range(
        idata,
        "expected_cal_bp",
        "Expected typological calendar positions",
        "cal BP",
    )
    _print_range(
        idata,
        "latent_cal_bp",
        "Modelled individual calendar ages",
        "cal BP",
    )


# -----------------------------------------------------------------------------
# Active outlier diagnostics
# -----------------------------------------------------------------------------


def build_outlier_table(
    idata: az.InferenceData,
    *,
    grave_ids=None,
    c14_bp=None,
    c14_error=None,
    unmodelled_cal_bp_mean=None,
    unmodelled_cal_bp_sd=None,
    posterior_rank=None,
    min_probability: float = 1e-9,
) -> pd.DataFrame | None:
    p_model = _draws(idata, "p_outlier") if _posterior_has(idata, "p_outlier") else None
    p_reconstructed = (
        _draws(idata, "p_outlier_reconstructed")
        if _posterior_has(idata, "p_outlier_reconstructed")
        else None
    )

    if p_model is None and p_reconstructed is None:
        return None

    n_graves = int(idata.attrs.get("n_graves", 0) or 0)
    n_c14 = int(idata.attrs.get("n_c14_dated", 0) or 0)

    if n_graves <= 0:
        if p_reconstructed is not None:
            n_graves = p_reconstructed.shape[1]
        elif p_model is not None:
            n_graves = p_model.shape[1]

    if n_c14 <= 0:
        n_c14 = p_model.shape[1] if p_model is not None else n_graves

    if p_model is not None:
        c14_index = _c14_indices_from_da(idata, "p_outlier", p_model.shape[1], n_graves)
    elif p_reconstructed is not None:
        vals = p_reconstructed
        finite_cols = np.where(np.isfinite(vals).any(axis=0))[0]
        c14_index = finite_cols.astype(int)
        n_c14 = len(c14_index)
    else:
        return None

    grave_ids_full = _optional_vector(grave_ids, n_graves, "grave_ids")
    c14_bp_full = _optional_vector(c14_bp, n_graves, "c14_bp")
    c14_error_full = _optional_vector(c14_error, n_graves, "c14_error")
    unmodelled_mean_full = _optional_vector(
        unmodelled_cal_bp_mean,
        n_graves,
        "unmodelled_cal_bp_mean",
    )
    unmodelled_sd_full = _optional_vector(
        unmodelled_cal_bp_sd,
        n_graves,
        "unmodelled_cal_bp_sd",
    )

    if posterior_rank is None:
        posterior_rank_full = _posterior_rank(idata)
    else:
        posterior_rank_full = _optional_vector(
            posterior_rank, n_graves, "posterior_rank"
        )

    rows = pd.DataFrame({"grave_index": c14_index})

    if grave_ids_full is not None:
        rows.insert(0, "grave_id", grave_ids_full[c14_index].astype(str))

    if posterior_rank_full is not None:
        rows["posterior_rank"] = posterior_rank_full[c14_index].astype(int)

    if c14_bp_full is not None:
        rows["c14_bp"] = c14_bp_full[c14_index].astype(float)

    if c14_error_full is not None:
        rows["c14_error"] = c14_error_full[c14_index].astype(float)

    if p_model is not None:
        rows["p_outlier_mean"] = p_model.mean(axis=0)
        rows["p_outlier_sd"] = p_model.std(axis=0)
        rows["p_outlier_hdi_3"] = np.quantile(p_model, 0.03, axis=0)
        rows["p_outlier_hdi_97"] = np.quantile(p_model, 0.97, axis=0)

    if p_reconstructed is not None:
        pr = p_reconstructed[:, c14_index]
        rows["p_outlier_reconstructed_mean"] = np.nanmean(pr, axis=0)
        rows["p_outlier_reconstructed_sd"] = np.nanstd(pr, axis=0)
        rows["p_outlier_reconstructed_hdi_3"] = np.nanquantile(pr, 0.03, axis=0)
        rows["p_outlier_reconstructed_hdi_97"] = np.nanquantile(pr, 0.97, axis=0)

    p_col = (
        "p_outlier_mean"
        if "p_outlier_mean" in rows.columns
        else "p_outlier_reconstructed_mean"
    )

    rows["outlier_interpretation"] = pd.cut(
        rows[p_col],
        bins=[-np.inf, 0.10, 0.33, 0.66, np.inf],
        labels=["unlikely", "possible", "probable", "strong"],
    ).astype(str)

    for var, prefix in [
        ("expected_cal_bp", "expected_cal_bp"),
        ("latent_cal_bp", "posterior_cal_bp"),
    ]:
        m = _mean(idata, var)
        s = _sd(idata, var)

        if m is not None and len(m) == n_graves:
            rows[f"{prefix}_mean"] = m[c14_index]

        if s is not None and len(s) == n_graves:
            rows[f"{prefix}_sd"] = s[c14_index]

    if unmodelled_mean_full is not None:
        rows["unmodelled_cal_bp_mean"] = unmodelled_mean_full[c14_index].astype(float)

    if unmodelled_sd_full is not None:
        rows["unmodelled_cal_bp_sd"] = unmodelled_sd_full[c14_index].astype(float)

    if _posterior_has(idata, "expected_cal_bp") and _posterior_has(
        idata, "latent_cal_bp"
    ):
        expected = _draws(idata, "expected_cal_bp")
        latent = _draws(idata, "latent_cal_bp")

        if expected.shape == latent.shape and expected.shape[1] == n_graves:
            shift = latent[:, c14_index] - expected[:, c14_index]
            rows["shift_from_typological_expectation"] = shift.mean(axis=0)
            rows["abs_shift_from_typological_expectation"] = rows[
                "shift_from_typological_expectation"
            ].abs()

    if {"posterior_cal_bp_mean", "unmodelled_cal_bp_mean"}.issubset(rows.columns):
        rows["shift_from_unmodelled_calibration"] = (
            rows["posterior_cal_bp_mean"] - rows["unmodelled_cal_bp_mean"]
        )
        rows["abs_shift_from_unmodelled_calibration"] = rows[
            "shift_from_unmodelled_calibration"
        ].abs()

    # Keep only rows where the outlier component was actually active/relevant.
    # This prevents large tables full of p=0 rows.
    active_cols = [
        c
        for c in [
            "p_outlier_mean",
            "p_outlier_reconstructed_mean",
            "p_outlier_hdi_97",
            "p_outlier_reconstructed_hdi_97",
        ]
        if c in rows.columns
    ]

    if active_cols:
        active = np.zeros(len(rows), dtype=bool)
        for col in active_cols:
            active |= np.asarray(rows[col], dtype=float) > min_probability
        rows = rows[active].copy()

    if rows.empty:
        return None

    return rows.sort_values(p_col, ascending=False).reset_index(drop=True)


def print_outlier_diagnostics(
    idata: az.InferenceData,
    *,
    grave_ids=None,
    c14_bp=None,
    c14_error=None,
    unmodelled_cal_bp_mean=None,
    unmodelled_cal_bp_sd=None,
    posterior_rank=None,
    n: int = 20,
) -> None:
    if not (
        _posterior_has(idata, "p_outlier")
        or _posterior_has(idata, "p_outlier_reconstructed")
    ):
        return

    rows = build_outlier_table(
        idata,
        grave_ids=grave_ids,
        c14_bp=c14_bp,
        c14_error=c14_error,
        unmodelled_cal_bp_mean=unmodelled_cal_bp_mean,
        unmodelled_cal_bp_sd=unmodelled_cal_bp_sd,
        posterior_rank=posterior_rank,
    )

    if rows is None or rows.empty:
        return

    print("\nActive outlier-model diagnostics:")

    scalar = _summary(idata, ["sigma_cal_link", "sigma_cal_outlier"])
    if scalar is not None:
        print("\nOutlier scale parameters:")
        print(scalar)

    show = [
        "grave_id",
        "grave_index",
        "posterior_rank",
        "c14_bp",
        "c14_error",
        "p_outlier_mean",
        "p_outlier_sd",
        "p_outlier_hdi_3",
        "p_outlier_hdi_97",
        "p_outlier_reconstructed_mean",
        "p_outlier_reconstructed_sd",
        "p_outlier_reconstructed_hdi_3",
        "p_outlier_reconstructed_hdi_97",
        "outlier_interpretation",
        "unmodelled_cal_bp_mean",
        "expected_cal_bp_mean",
        "posterior_cal_bp_mean",
        "shift_from_typological_expectation",
        "shift_from_unmodelled_calibration",
        "posterior_cal_bp_sd",
    ]
    show = [c for c in show if c in rows.columns]

    print("\nExplicitly modelled outlier candidates:")
    print(rows.head(n)[show].round(3).to_string(index=False))

    p_col = (
        "p_outlier_mean"
        if "p_outlier_mean" in rows.columns
        else "p_outlier_reconstructed_mean"
    )

    print("\nOutlier probability summary for active candidates:")
    print(
        "mean/median/max:",
        round(float(rows[p_col].mean()), 3),
        round(float(rows[p_col].median()), 3),
        round(float(rows[p_col].max()), 3),
    )


def build_parameter_diagnostics(
    idata: az.InferenceData,
    vars_: list[str] | None = None,
) -> pd.DataFrame | None:
    """Return ArviZ parameter diagnostics as a machine-readable table.

    This mirrors the diagnostic quantities printed in debug mode, but makes
    them available for downstream workflows and paper-level reporting.

    The table includes scalar and vector parameters where available, excluding
    constant variables. For vector variables, ArviZ returns one row per indexed
    element, e.g. t[0], t[1], ...
    """

    if not hasattr(idata, "posterior"):
        return None

    if vars_ is None:
        vars_ = [
            "t",
            "mu",
            "sigma",
            "a",
            "g",
            "intercept",
            "cal_alpha",
            "cal_span",
            "cal_beta",
            "sigma_cal_link",
            "sigma_cal_outlier",
            "mu_log_sigma",
            "sd_log_sigma",
            "mu_a",
            "sd_a",
        ]

        if _posterior_has(idata, "p_outlier"):
            p = _draws(idata, "p_outlier")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier")

        if _posterior_has(idata, "p_outlier_reconstructed"):
            p = _draws(idata, "p_outlier_reconstructed")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier_reconstructed")

    vars_ = _nonconstant_vars(idata, vars_)

    if not vars_:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        diag = az.summary(
            idata,
            var_names=vars_,
            round_to=None,
        )

    if diag is None or diag.empty:
        return None

    diag = diag.reset_index(names="parameter").replace([np.inf, -np.inf], np.nan)

    preferred = [
        "parameter",
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "mcse_mean",
        "mcse_sd",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ]

    cols = [c for c in preferred if c in diag.columns]
    rest = [c for c in diag.columns if c not in cols]

    return diag[cols + rest]


# -----------------------------------------------------------------------------
# Divergences and worst parameters
# -----------------------------------------------------------------------------


def print_worst_parameter_diagnostics(
    idata: az.InferenceData,
    vars_: list[str] | None = None,
    n: int = 15,
) -> None:
    if vars_ is None:
        vars_ = [
            "t",
            "mu",
            "sigma",
            "a",
            "g",
            "cal_alpha",
            "cal_span",
            "sigma_cal_link",
            "sigma_cal_outlier",
            "mu_log_sigma",
            "sd_log_sigma",
            "mu_a",
            "sd_a",
        ]

        # Only include explicit outlier probabilities if they are not all zero.
        if _posterior_has(idata, "p_outlier"):
            p = _draws(idata, "p_outlier")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier")

        if _posterior_has(idata, "p_outlier_reconstructed"):
            p = _draws(idata, "p_outlier_reconstructed")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier_reconstructed")

    tables = []
    for var in _nonconstant_vars(idata, vars_):
        try:
            s = _summary(idata, [var])
        except Exception:
            # Some transformed or reconstructed variables can fail ArviZ summary
            # generation in edge cases. Skip them in printed debug diagnostics.
            continue

        if s is None or s.empty:
            continue

        out = s.reset_index(names="parameter")
        keep = [
            c
            for c in ["parameter", "mean", "sd", "ess_bulk", "ess_tail", "r_hat"]
            if c in out.columns
        ]
        tables.append(out[keep])

    if not tables:
        return

    diag = pd.concat(tables, ignore_index=True).replace([np.inf, -np.inf], np.nan)

    if "r_hat" in diag.columns:
        worst = (
            diag.dropna(subset=["r_hat"]).sort_values("r_hat", ascending=False).head(n)
        )
        if not worst.empty:
            print("\nWorst individual parameters by R-hat:")
            print(worst.round(3).to_string(index=False))

    if "ess_bulk" in diag.columns:
        worst = diag.dropna(subset=["ess_bulk"]).sort_values("ess_bulk").head(n)
        if not worst.empty:
            print("\nLowest individual parameters by bulk ESS:")
            print(worst.round(3).to_string(index=False))


def print_divergence_diagnostics(
    idata: az.InferenceData,
    vars_: list[str] | None = None,
    n: int = 12,
) -> None:
    if not _sample_stats_has(idata, "diverging"):
        return

    diverging = idata.sample_stats["diverging"].values.reshape(-1).astype(bool)
    n_div = int(diverging.sum())

    print("\nDivergence diagnostics:")
    print("Divergences per chain:")
    for i, value in enumerate(idata.sample_stats["diverging"].sum(dim="draw").values):
        print(f"  chain {i}: {int(value)}")

    if n_div == 0:
        print("No divergent transitions detected.")
        return

    if vars_ is None:
        vars_ = [
            "cal_alpha",
            "cal_span",
            "sigma_cal_link",
            "sigma_cal_outlier",
            "intercept",
            "sigma_g",
            "mu_log_sigma",
            "sd_log_sigma",
            "mu_a",
            "sd_a",
        ]

        if _posterior_has(idata, "p_outlier"):
            p = _draws(idata, "p_outlier")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier")

        if _posterior_has(idata, "p_outlier_reconstructed"):
            p = _draws(idata, "p_outlier_reconstructed")
            if np.nanmax(p) > 1e-9:
                vars_.append("p_outlier_reconstructed")

    rows = []

    for var in _nonconstant_vars(idata, vars_):
        x = _draws(idata, var)

        if x.shape[0] != diverging.size:
            continue

        div = x[diverging]
        non = x[~diverging]

        if div.size == 0 or non.size == 0:
            continue

        non_mean = non.mean(axis=0)
        div_mean = div.mean(axis=0)
        non_sd = non.std(axis=0)
        std_diff = (div_mean - non_mean) / np.where(non_sd > 0, non_sd, np.nan)

        for i in range(x.shape[1]):
            rows.append(
                {
                    "parameter": var if x.shape[1] == 1 else f"{var}[{i}]",
                    "nondiv_mean": non_mean[i],
                    "div_mean": div_mean[i],
                    "raw_diff": div_mean[i] - non_mean[i],
                    "std_diff": std_diff[i],
                    "abs_std_diff": abs(std_diff[i]),
                }
            )

    if not rows:
        return

    out = (
        pd.DataFrame(rows)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["abs_std_diff"])
        .sort_values("abs_std_diff", ascending=False)
        .head(n)
    )

    if not out.empty:
        print("\nLargest divergent vs non-divergent posterior differences:")
        print(
            out[
                [
                    "parameter",
                    "nondiv_mean",
                    "div_mean",
                    "raw_diff",
                    "std_diff",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def print_sampling_diagnostics(
    idata: az.InferenceData,
    include_richness: bool = True,
    *,
    grave_ids=None,
    c14_bp=None,
    c14_error=None,
    unmodelled_cal_bp_mean=None,
    unmodelled_cal_bp_sd=None,
    unmodelled_cal_bp_hdi_3=None,
    unmodelled_cal_bp_hdi_97=None,
    posterior_rank=None,
) -> None:
    if not hasattr(idata, "posterior"):
        raise ValueError("idata must contain a posterior group.")

    div = (
        int(idata.sample_stats["diverging"].sum().values)
        if _sample_stats_has(idata, "diverging")
        else 0
    )
    print("Divergences:", div)

    print_prior_metadata(idata)
    print_scalar_diagnostics(idata, include_richness=include_richness)
    print_vector_diagnostics(idata)
    print_calendar_ranges(idata)

    # The cheap post-hoc outlier screen is now handled in workflow.py for the
    # normal/verbose output. Here we only report the active modelled outlier
    # component, i.e. when --outlier or --outlier-all was actually used.
    print_outlier_diagnostics(
        idata,
        grave_ids=grave_ids,
        c14_bp=c14_bp,
        c14_error=c14_error,
        unmodelled_cal_bp_mean=unmodelled_cal_bp_mean,
        unmodelled_cal_bp_sd=unmodelled_cal_bp_sd,
        posterior_rank=posterior_rank,
    )

    print_worst_parameter_diagnostics(idata)
    print_divergence_diagnostics(idata)
