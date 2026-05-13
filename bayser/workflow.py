from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bayser.calibration import calibrate_c14_dates_unmodelled, load_intcal20_curve
from bayser.classical import classical_order, order_correlation
from bayser.data import load_seriation_input
from bayser.diagnostics import (
    build_outlier_table,
    build_parameter_diagnostics,
    print_c14_diagnostics,
    print_matrix_diagnostics,
    print_sampling_diagnostics,
)
from bayser.model import fit_parametric_pymc_seriation
from bayser.model_helpers import infer_cal_span_prior_from_reference_calibration
from bayser.plots import (
    configure_plots,
    plot_c14_against_order,
    plot_matrix,
    plot_model_shift_against_rank,
    plot_observed_bp_vs_posterior_cal_bp,
    plot_posterior_cal_bp_against_order,
    plot_posterior_rank_distributions,
    plot_pymc_vs_ra_scores,
    plot_unmodelled_vs_modelled_cal_bp,
)
from bayser.posterior import (
    chain_order_diagnostics,
    compare_pymc_to_ra_scores,
    order_from_summary,
    pairwise_uncertainty_summary,
    posterior_pairwise_order_probabilities,
    summarise_graves,
    summarise_types,
)
from bayser.results import write_results


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _is_quiet(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "quiet", False))


def _is_verbose(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "verbose", False)) and not _is_quiet(args)


def _is_debug(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "debug", False)) and not _is_quiet(args)


def _print(args: argparse.Namespace, *values, **kwargs) -> None:
    if not _is_quiet(args):
        print(*values, **kwargs)


def _print_verbose(args: argparse.Namespace, *values, **kwargs) -> None:
    if _is_verbose(args) or _is_debug(args):
        print(*values, **kwargs)


def _print_debug(args: argparse.Namespace, *values, **kwargs) -> None:
    if _is_debug(args):
        print(*values, **kwargs)


def _finite_c14(c14_bp, c14_error) -> np.ndarray:
    if c14_bp is None or c14_error is None:
        return np.array([], dtype=bool)

    return np.isfinite(np.asarray(c14_bp, float)) & np.isfinite(
        np.asarray(c14_error, float)
    )


def _by_grave(table: pd.DataFrame | None, grave_ids: list[str], col: str) -> np.ndarray | None:
    if table is None or col not in table.columns:
        return None

    return table.set_index("grave_id").reindex(grave_ids)[col].to_numpy(float)


def _keep_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _validate_probability(p: float, name: str = "probability") -> float:
    p = float(p)

    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1.")

    return p


def _count_divergences(idata) -> int:
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        return 0

    return int(idata.sample_stats["diverging"].sum().values)


def _compact_sampling_summary(idata) -> pd.DataFrame:
    rows = []

    def add(var: str, label: str | None = None):
        if not hasattr(idata, "posterior") or var not in idata.posterior:
            return

        x = idata.posterior[var].values.reshape(-1)

        if not np.isfinite(x).any():
            return

        rows.append(
            {
                "parameter": label or var,
                "mean": float(np.nanmean(x)),
                "sd": float(np.nanstd(x)),
                "q03": float(np.nanquantile(x, 0.03)),
                "q97": float(np.nanquantile(x, 0.97)),
            }
        )

    for var in [
        "cal_alpha",
        "cal_span",
        "sigma_cal_link",
        "sigma_cal_outlier",
        "sigma_g",
    ]:
        add(var)

    return pd.DataFrame(rows)


def _c14_input_table(
    grave_ids: list[str],
    c14_bp,
    c14_error,
) -> pd.DataFrame:
    n = len(grave_ids)

    if c14_bp is None:
        bp = np.full(n, np.nan)
    else:
        bp = np.asarray(c14_bp, dtype=float)

    if c14_error is None:
        err = np.full(n, np.nan)
    else:
        err = np.asarray(c14_error, dtype=float)

    finite = np.isfinite(bp) & np.isfinite(err)

    return pd.DataFrame(
        {
            "grave_id": grave_ids,
            "grave_index": np.arange(n),
            "c14_bp": bp,
            "c14_error": err,
            "finite_c14": finite,
        }
    )


def _metadata_table(metadata: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in metadata.items()]
    )


# -----------------------------------------------------------------------------
# Data and calibration
# -----------------------------------------------------------------------------


def _load_data(args: argparse.Namespace):
    data = load_seriation_input(
        feature_path=args.features,
        c14_path=args.c14,
        feature_sep=args.feature_sep,
        c14_sep=args.c14_sep,
        feature_id_col=args.feature_id_col,
        c14_id_col=args.c14_id_col,
        bp_col=args.bp_col,
        error_col=args.error_col,
        min_type_count=args.min_type_count,
        min_grave_count=args.min_grave_count,
        filter_matrix=args.filter,
        exclude_cols=args.exclude_col,
        exclude_regex=args.exclude_regex,
    )

    _print(args, "Source:", data.source)
    _print(args, "Filtering:", "enabled" if args.filter else "disabled")

    if _is_verbose(args) or _is_debug(args):
        if args.c14 and _finite_c14(data.c14_bp, data.c14_error).any():
            print_c14_diagnostics(data)
        elif args.c14:
            print("C14 dates: none finite after preparation")
    else:
        finite = _finite_c14(data.c14_bp, data.c14_error)

        _print(
            args,
            "Input:",
            f"{data.Y.shape[0]} graves × {data.Y.shape[1]} types;",
            f"{int(finite.sum())} finite C14 dates",
        )

        if getattr(data, "removed_grave_ids", None):
            _print(args, f"Removed graves during preparation: {len(data.removed_grave_ids)}")

        if getattr(data, "removed_type_ids", None):
            _print(args, f"Removed types during preparation: {len(data.removed_type_ids)}")

        if getattr(data, "unmatched_c14_ids", None):
            _print(args, f"Unmatched C14 IDs: {len(data.unmatched_c14_ids)}")

    return data


def _load_curve(
    args: argparse.Namespace,
    path: str | None,
    has_c14: bool,
) -> pd.DataFrame | None:
    if not has_c14:
        return None

    if path is None:
        raise ValueError("Finite C14 dates require --intcal20.")

    curve = load_intcal20_curve(path)

    _print_verbose(args, "IntCal20 curve:", path)
    _print_verbose(
        args,
        "IntCal20 cal BP range:",
        round(float(curve["cal_bp"].min()), 1),
        "to",
        round(float(curve["cal_bp"].max()), 1),
    )

    return curve


def _calibrate_unmodelled(
    args: argparse.Namespace,
    grave_ids: list[str],
    c14_bp,
    c14_error,
    curve: pd.DataFrame | None,
) -> pd.DataFrame | None:
    keep = _finite_c14(c14_bp, c14_error)

    if not keep.any():
        return None

    out = calibrate_c14_dates_unmodelled(
        grave_ids=[g for g, k in zip(grave_ids, keep) if k],
        c14_bp=np.asarray(c14_bp, float)[keep],
        c14_error=np.asarray(c14_error, float)[keep],
        curve=curve,
    )

    if _is_verbose(args) or _is_debug(args):
        print("\nUnmodelled single-date calibration summary:")
        print(out.sort_values("unmodelled_cal_bp_mean").head(20).to_string(index=False))

    return out


# -----------------------------------------------------------------------------
# Outlier CLI bridge
# -----------------------------------------------------------------------------


def _parse_outlier_spec(spec: str, default_probability: float = 0.5) -> tuple[str, float]:
    """Parse one CLI outlier specification.

    Accepted forms:
    - ASO_6
    - ASO_6:0.5
    - ASO_6=0.5
    - ASO_6,0.5

    If no probability is given, default_probability is used.
    """

    spec = str(spec).strip()

    if not spec:
        raise ValueError("Empty outlier specification.")

    parts = re.split(r"[:=,]", spec, maxsplit=1)
    grave_id = parts[0].strip()

    if not grave_id:
        raise ValueError(f"Invalid outlier specification: {spec!r}")

    if len(parts) == 1:
        return grave_id, _validate_probability(default_probability, "default outlier prior")

    return grave_id, _validate_probability(parts[1].strip(), f"outlier prior for {grave_id}")


def _build_outlier_prior_by_grave(
    args: argparse.Namespace,
    grave_ids: list[str],
    c14_bp,
    c14_error,
) -> np.ndarray:
    """Construct the model-level outlier prior vector.

    Default is all zero, i.e. no outlier model. The outlier component is only
    activated in model.py if at least one dated grave has prior > 0.
    """

    n_graves = len(grave_ids)
    out = np.zeros(n_graves, dtype=float)

    finite_c14 = _finite_c14(c14_bp, c14_error)
    id_to_index = {g: i for i, g in enumerate(grave_ids)}

    outlier_all = getattr(args, "outlier_all", None)
    outlier_specs = getattr(args, "outlier", None) or []

    if outlier_all is not None:
        p = _validate_probability(outlier_all, "--outlier-all")

        if not finite_c14.any():
            raise ValueError("--outlier-all was supplied, but there are no finite C14 dates.")

        out[finite_c14] = p

    for spec in outlier_specs:
        grave_id, p = _parse_outlier_spec(spec)

        if grave_id not in id_to_index:
            raise ValueError(
                f"Outlier candidate {grave_id!r} is not present in the retained grave IDs."
            )

        idx = id_to_index[grave_id]

        if len(finite_c14) != n_graves or not finite_c14[idx]:
            raise ValueError(
                f"Outlier candidate {grave_id!r} has no finite C14 date after preparation."
            )

        out[idx] = p

    if np.any(out > 0.0):
        table = pd.DataFrame(
            {
                "grave_id": grave_ids,
                "outlier_prior": out,
            }
        )
        table = table[table["outlier_prior"] > 0.0].sort_values(
            ["outlier_prior", "grave_id"],
            ascending=[False, True],
        )

        _print(args, "\nOutlier model requested for:")
        _print(args, table.to_string(index=False))
    else:
        _print(args, "\nOutlier model: not requested; running default non-outlier chronology.")

    return out


# -----------------------------------------------------------------------------
# Orientation and summaries
# -----------------------------------------------------------------------------


def _orient_by_c14(
    row_scores: np.ndarray,
    col_scores: np.ndarray,
    grave_ids: list[str],
    unmodelled: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    cal = _by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_mean")

    if cal is None:
        return row_scores, col_scores, "RA/CA axis retained; no calibrated C14 reference."

    keep = np.isfinite(row_scores) & np.isfinite(cal)

    if keep.sum() < 5:
        return row_scores, col_scores, f"RA/CA axis retained; too few C14 dates n={keep.sum()}."

    rho = spearmanr(row_scores[keep], cal[keep]).statistic

    if not np.isfinite(rho) or abs(float(rho)) < 0.25:
        return row_scores, col_scores, f"RA/CA axis retained; weak C14 trend rho={rho:.3f}."

    if rho > 0:
        return -row_scores, -col_scores, f"RA/CA axis flipped; rho={rho:.3f}."

    return row_scores, col_scores, f"RA/CA axis retained; rho={rho:.3f}."


def _merge_unmodelled(summary: pd.DataFrame, unmodelled: pd.DataFrame | None) -> pd.DataFrame:
    if unmodelled is None:
        return summary

    attrs = dict(summary.attrs)

    cols = _keep_cols(
        unmodelled,
        [
            "grave_id",
            "unmodelled_cal_bp_mean",
            "unmodelled_cal_bp_sd",
            "unmodelled_cal_bp_hdi_3",
            "unmodelled_cal_bp_hdi_97",
        ],
    )

    out = summary.merge(unmodelled[cols], on="grave_id", how="left")
    out.attrs.update(attrs)

    return out


def _print_main_tables(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    type_summary: pd.DataFrame,
) -> None:
    if not (_is_verbose(args) or _is_debug(args)):
        return

    grave_cols = _keep_cols(
        summary,
        [
            "grave_id",
            "posterior_t_mean",
            "posterior_t_sd",
            "posterior_rank",
            "unmodelled_cal_bp_mean",
            "expected_cal_bp_mean",
            "posterior_cal_bp_mean",
            "posterior_cal_bp_sd",
            "posterior_cal_bp_source",
            "ra_rank",
            "rank_difference_pymc_minus_ra",
            "p_outlier_mean",
            "p_outlier_sd",
            "p_outlier_hdi_3",
            "p_outlier_hdi_97",
        ],
    )

    type_cols = _keep_cols(
        type_summary,
        [
            "type_id",
            "posterior_mu_mean",
            "posterior_mu_sd",
            "posterior_sigma_mean",
            "posterior_a_mean",
            "posterior_type_rank",
            "ra_type_rank",
            "type_rank_difference_pymc_minus_ra",
        ],
    )

    print("\nPosterior grave order summary:")
    print(summary[grave_cols].head(25).to_string(index=False))

    print("\nPosterior type summary:")
    print(type_summary[type_cols].head(25).to_string(index=False))


def _print_compact_result_summary(
    args: argparse.Namespace,
    idata,
    summary: pd.DataFrame,
    score_comparison: pd.DataFrame,
    pymc_order: np.ndarray,
    ra_order: np.ndarray,
) -> None:
    if _is_quiet(args):
        return

    div = _count_divergences(idata)

    print("\nRun summary:")
    print("Divergences:", div)
    print(
        "PyMC vs RA order correlation:",
        round(order_correlation(pymc_order, ra_order), 3),
    )
    print(
        "PyMC posterior t vs RA:",
        "Pearson",
        round(score_comparison.attrs["pearson_abs"], 3),
        "| Spearman",
        round(score_comparison.attrs["spearman_abs"], 3),
    )

    compact = _compact_sampling_summary(idata)
    if not compact.empty:
        print("\nKey posterior scales:")
        print(compact.round(3).to_string(index=False))

    cols = _keep_cols(
        summary,
        [
            "grave_id",
            "posterior_rank",
            "unmodelled_cal_bp_mean",
            "expected_cal_bp_mean",
            "posterior_cal_bp_mean",
            "posterior_cal_bp_sd",
        ],
    )

    if cols:
        print("\nPosterior grave order, first 12:")
        print(summary[cols].head(12).round(3).to_string(index=False))


def _print_shift_summary(
    args: argparse.Namespace,
    summary: pd.DataFrame,
) -> None:
    if not (_is_verbose(args) or _is_debug(args)):
        return

    if "posterior_cal_bp_mean" not in summary.columns:
        return

    s = summary.copy()

    for base, label in [
        ("expected_cal_bp_mean", "typological expectation"),
        ("unmodelled_cal_bp_mean", "unmodelled calibration"),
    ]:
        if base not in s.columns:
            continue

        shift_col = f"shift_from_{base}"
        s[shift_col] = s["posterior_cal_bp_mean"] - s[base]

        cols = _keep_cols(
            s,
            ["grave_id", "posterior_rank", base, "posterior_cal_bp_mean", shift_col],
        )

        print(f"\nLargest shifts from {label}:")
        print(
            s.assign(abs_shift=s[shift_col].abs())
            .sort_values("abs_shift", ascending=False)
            .head(15)[cols]
            .to_string(index=False)
        )


def _build_posthoc_outlier_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    """Cheap post-hoc screen for candidates worth rerunning with --outlier.

    This does not change the model. It only flags cases where the typological
    expectation and the single-date calibration pull strongly in different
    directions.
    """

    needed = {
        "grave_id",
        "posterior_rank",
        "expected_cal_bp_mean",
        "posterior_cal_bp_mean",
        "unmodelled_cal_bp_mean",
        "unmodelled_cal_bp_hdi_3",
        "unmodelled_cal_bp_hdi_97",
    }

    if not needed.issubset(summary.columns):
        return pd.DataFrame()

    s = summary.copy()

    s["shift_model_vs_expected"] = (
        s["posterior_cal_bp_mean"] - s["expected_cal_bp_mean"]
    )
    s["shift_unmodelled_vs_expected"] = (
        s["unmodelled_cal_bp_mean"] - s["expected_cal_bp_mean"]
    )
    s["shift_model_vs_unmodelled"] = (
        s["posterior_cal_bp_mean"] - s["unmodelled_cal_bp_mean"]
    )

    s["expected_outside_unmodelled_hdi"] = (
        (s["expected_cal_bp_mean"] < s["unmodelled_cal_bp_hdi_3"])
        | (s["expected_cal_bp_mean"] > s["unmodelled_cal_bp_hdi_97"])
    )

    s["posterior_outside_unmodelled_hdi"] = (
        (s["posterior_cal_bp_mean"] < s["unmodelled_cal_bp_hdi_3"])
        | (s["posterior_cal_bp_mean"] > s["unmodelled_cal_bp_hdi_97"])
    )

    s["max_abs_conflict"] = np.maximum(
        s["shift_model_vs_expected"].abs(),
        s["shift_unmodelled_vs_expected"].abs(),
    )

    strong = (
        (s["shift_unmodelled_vs_expected"].abs() >= 150.0)
        & (s["shift_model_vs_expected"].abs() >= 75.0)
        & s["expected_outside_unmodelled_hdi"]
    )

    possible = (
        s["expected_outside_unmodelled_hdi"]
        & (s["shift_unmodelled_vs_expected"].abs() >= 90.0)
    ) | (
        s["posterior_outside_unmodelled_hdi"]
        & (s["shift_model_vs_unmodelled"].abs() >= 75.0)
    )

    s["posthoc_outlier_suggestion"] = np.select(
        [strong, possible],
        ["strong", "possible"],
        default="",
    )

    s["suggested_outlier_prior"] = np.select(
        [strong, possible],
        [0.5, 0.25],
        default=np.nan,
    )

    suggested_prior = pd.Series(s["suggested_outlier_prior"], index=s.index).map(
        lambda x: "" if not np.isfinite(x) else f"{x:g}"
    )

    s["suggested_cli_arg"] = np.where(
        s["posthoc_outlier_suggestion"] != "",
        "--outlier " + s["grave_id"].astype(str) + ":" + suggested_prior,
        "",
    )

    priority = pd.Series(
        s["posthoc_outlier_suggestion"],
        index=s.index,
    ).map({"strong": 0, "possible": 1}).fillna(9)

    return (
        s[s["posthoc_outlier_suggestion"] != ""]
        .assign(priority=priority)
        .sort_values(
            ["priority", "max_abs_conflict"],
            ascending=[True, False],
        )
        .drop(columns=["priority"])
        .reset_index(drop=True)
    )


def _print_posthoc_outlier_screening(
    args: argparse.Namespace,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    candidates = _build_posthoc_outlier_candidates(summary)

    if _is_quiet(args):
        return candidates

    print("\nPost-hoc outlier screen:")

    if candidates.empty:
        print("No strong candidates flagged by the current heuristic.")
        return candidates

    compact_cols = _keep_cols(
        candidates,
        [
            "grave_id",
            "posterior_rank",
            "posthoc_outlier_suggestion",
            "suggested_cli_arg",
            "unmodelled_cal_bp_mean",
            "expected_cal_bp_mean",
            "posterior_cal_bp_mean",
            "shift_unmodelled_vs_expected",
            "shift_model_vs_expected",
        ],
    )

    print("Suggested candidates:")
    print(candidates[compact_cols].head(8).round(3).to_string(index=False))

    strong = candidates[candidates["posthoc_outlier_suggestion"] == "strong"]
    first = strong.iloc[0] if not strong.empty else candidates.iloc[0]

    print("\nRecommended first rerun:")
    print(f"  add: {first['suggested_cli_arg']}")

    if _is_verbose(args) or _is_debug(args):
        detail_cols = _keep_cols(
            candidates,
            [
                "grave_id",
                "posterior_rank",
                "unmodelled_cal_bp_mean",
                "unmodelled_cal_bp_hdi_3",
                "unmodelled_cal_bp_hdi_97",
                "expected_cal_bp_mean",
                "posterior_cal_bp_mean",
                "shift_model_vs_expected",
                "shift_unmodelled_vs_expected",
                "expected_outside_unmodelled_hdi",
                "posterior_outside_unmodelled_hdi",
            ],
        )

        print("\nPost-hoc outlier screen detail:")
        print(candidates.head(20)[detail_cols].round(3).to_string(index=False))

    return candidates


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------


def _build_run_metadata(
    args: argparse.Namespace,
    data,
    idata,
    *,
    has_c14: bool,
    chronology_mode: str,
    orientation_message: str,
    cal_span_mu: float | None,
    cal_span_sigma: float | None,
    score_comparison: pd.DataFrame,
    pymc_order: np.ndarray,
    ra_order: np.ndarray,
) -> dict:
    finite = _finite_c14(data.c14_bp, data.c14_error)

    metadata = {
        "features": getattr(args, "features", None),
        "c14": getattr(args, "c14", None),
        "intcal20": getattr(args, "intcal20", None),
        "filter": bool(getattr(args, "filter", True)),
        "min_type_count": getattr(args, "min_type_count", None),
        "min_grave_count": getattr(args, "min_grave_count", None),
        "n_graves": int(data.Y.shape[0]),
        "n_types": int(data.Y.shape[1]),
        "n_c14_finite": int(finite.sum()),
        "has_c14": bool(has_c14),
        "chronology_mode": chronology_mode,
        "orientation": orientation_message,
        "ra_method": getattr(args, "ra_method", None),
        "draws": getattr(args, "draws", None),
        "tune": getattr(args, "tune", None),
        "chains": getattr(args, "chains", None),
        "target_accept": getattr(args, "target_accept", None),
        "max_treedepth": getattr(args, "max_treedepth", None),
        "random_seed": getattr(args, "random_seed", None),
        "include_richness": bool(getattr(args, "include_richness", True)),
        "repulsion_strength": getattr(args, "repulsion_strength", None),
        "calendar_grid_step": getattr(args, "calendar_grid_step", None),
        "local_window_padding": getattr(args, "local_window_padding", None),
        "cal_span_mu": cal_span_mu,
        "cal_span_sigma": cal_span_sigma,
        "outlier": getattr(args, "outlier", []),
        "outlier_all": getattr(args, "outlier_all", None),
        "divergences": _count_divergences(idata),
        "pymc_ra_order_correlation": order_correlation(pymc_order, ra_order),
        "pymc_ra_pearson_abs": score_comparison.attrs.get("pearson_abs"),
        "pymc_ra_spearman_abs": score_comparison.attrs.get("spearman_abs"),
        "ra_flipped_to_pymc": score_comparison.attrs.get("ra_flipped_to_pymc"),
        "plot_dir": getattr(args, "plot_dir", None),
        "results_dir": getattr(args, "results_dir", None),
    }

    if getattr(idata, "attrs", None):
        for key, value in idata.attrs.items():
            metadata[f"idata_{key}"] = value

    return metadata


def _write_results_with_summary(
    args: argparse.Namespace,
    *,
    tables: dict[str, pd.DataFrame | None],
    metadata: dict,
) -> None:
    if bool(getattr(args, "no_results", False)):
        return

    results_dir = Path(getattr(args, "results_dir", "results"))

    written = write_results(
        results_dir=results_dir,
        tables=tables,
        metadata=metadata,
    )

    if _is_quiet(args):
        return

    if isinstance(written, dict):
        n_files = len(written)
    elif isinstance(written, (list, tuple, set)):
        n_files = len(written)
    elif isinstance(written, int):
        n_files = written
    else:
        n_files = None

    if n_files is None:
        print(f"\nSaved result files in: {results_dir}")
    else:
        print(f"\nSaved {n_files} result files in: {results_dir}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def _make_plots(
    Y: np.ndarray,
    idata,
    grave_ids: list[str],
    c14_bp,
    c14_error,
    row_scores: np.ndarray,
    col_scores: np.ndarray,
    ra_order: np.ndarray,
    pymc_order: np.ndarray,
    score_comparison: pd.DataFrame,
    summary: pd.DataFrame,
    ra_method: str,
) -> None:
    plot_matrix(Y, np.arange(Y.shape[0]), "Dataset: input order")
    plot_matrix(Y, ra_order, f"Dataset: {ra_method} order", type_order=np.argsort(col_scores))
    plot_matrix(Y, pymc_order, "Dataset: PyMC posterior order")

    plot_pymc_vs_ra_scores(score_comparison, "Dataset: PyMC posterior axis vs RA/CA axis")

    plot_posterior_rank_distributions(
        idata,
        grave_ids=grave_ids,
        reference=row_scores,
        ra_row_scores=row_scores,
        posterior_order=pymc_order,
        title="Dataset: posterior rank distributions",
    )

    if not _finite_c14(c14_bp, c14_error).any():
        return

    plot_c14_against_order(
        grave_ids,
        c14_bp,
        c14_error,
        ra_order,
        f"Dataset: uncalibrated 14C BP along {ra_method} order",
    )

    plot_c14_against_order(
        grave_ids,
        c14_bp,
        c14_error,
        pymc_order,
        "Dataset: uncalibrated 14C BP along PyMC order",
    )

    if any(
        c in summary.columns
        for c in ["posterior_cal_bp_mean", "expected_cal_bp_mean", "unmodelled_cal_bp_mean"]
    ):
        plot_posterior_cal_bp_against_order(
            summary=summary,
            order=pymc_order,
            grave_ids=grave_ids,
            title="Dataset: calendar ages along PyMC order",
        )

        plot_observed_bp_vs_posterior_cal_bp(
            summary=summary,
            c14_bp=c14_bp,
            c14_error=c14_error,
            grave_ids=grave_ids,
            title="Dataset: observed 14C BP vs calendar-age estimates",
        )

    if {"unmodelled_cal_bp_mean", "posterior_cal_bp_mean"}.issubset(summary.columns):
        plot_unmodelled_vs_modelled_cal_bp(
            summary=summary,
            title="Dataset: unmodelled vs modelled calendar ages",
        )

    if "posterior_cal_bp_mean" in summary.columns and (
        "expected_cal_bp_mean" in summary.columns
        or "unmodelled_cal_bp_mean" in summary.columns
    ):
        plot_model_shift_against_rank(
            summary=summary,
            order=pymc_order,
            grave_ids=grave_ids,
            title="Dataset: calendar-age shifts",
        )


def _make_plots_with_summary(
    args: argparse.Namespace,
    *plot_args,
    **plot_kwargs,
) -> None:
    plot_dir = Path(getattr(args, "plot_dir", "plots"))

    if _is_debug(args):
        _make_plots(*plot_args, **plot_kwargs)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            _make_plots(*plot_args, **plot_kwargs)

    if _is_quiet(args):
        return

    pngs = sorted(plot_dir.glob("*.png")) if plot_dir.exists() else []
    if pngs:
        print(f"\nSaved/updated {len(pngs)} plots in: {plot_dir}")
    else:
        print(f"\nPlot directory: {plot_dir}")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> None:
    data = _load_data(args)

    configure_plots(
        plot_dir=getattr(args, "plot_dir", "plots"),
        show_plots=getattr(args, "show_plots", False),
        dpi=getattr(args, "plot_dpi", 200),
    )

    Y = data.Y
    grave_ids = data.grave_ids
    type_ids = data.type_ids
    c14_bp = data.c14_bp
    c14_error = data.c14_error

    has_c14 = _finite_c14(c14_bp, c14_error).any()
    curve = _load_curve(args, args.intcal20, has_c14)

    if _is_verbose(args) or _is_debug(args):
        print_matrix_diagnostics(Y, label="Dataset")

    _, _, row_scores_raw, col_scores_raw = classical_order(Y, method=args.ra_method)

    _print_verbose(
        args,
        "\nClassical comparison:",
        "reciprocal averaging / CA-SVD"
        if args.ra_method == "ca"
        else "iterative reciprocal averaging",
    )

    unmodelled = _calibrate_unmodelled(args, grave_ids, c14_bp, c14_error, curve)

    row_scores, col_scores, orientation_message = _orient_by_c14(
        row_scores_raw,
        col_scores_raw,
        grave_ids,
        unmodelled,
    )

    _print(args, "\nAxis orientation:", orientation_message)

    cal_span_mu = None
    cal_span_sigma = None

    if unmodelled is not None:
        cal_span_mu, cal_span_sigma = infer_cal_span_prior_from_reference_calibration(
            row_scores=row_scores,
            unmodelled_cal_bp=_by_grave(
                unmodelled,
                grave_ids,
                "unmodelled_cal_bp_mean",
            ),
        )

        _print_verbose(args, "\nData-informed cal_span prior:")
        _print_verbose(args, "cal_span_mu:", round(float(cal_span_mu), 3))
        _print_verbose(args, "cal_span_sigma:", round(float(cal_span_sigma), 3))

    ra_order = np.argsort(row_scores)
    chronology_mode = "c14_typology_linked" if has_c14 else "none"

    _print_verbose(args, "RA order computed.")
    _print(args, "\nC14 model:", "linked to typology" if has_c14 else "disabled")

    outlier_prior_by_grave = _build_outlier_prior_by_grave(
        args=args,
        grave_ids=grave_ids,
        c14_bp=c14_bp,
        c14_error=c14_error,
    )

    fit_kwargs = dict(
        Y=Y,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        include_richness=args.include_richness,
        repulsion_strength=args.repulsion_strength,
        chronology_mode=chronology_mode,
        orientation_reference=row_scores,
        c14_bp=c14_bp,
        c14_error=c14_error,
        intcal20_curve=curve,
        cal_span_mu=cal_span_mu,
        cal_span_sigma=cal_span_sigma if cal_span_sigma is not None else 40.0,
        calendar_grid_step=args.calendar_grid_step,
        local_window_padding=args.local_window_padding,
        max_treedepth=args.max_treedepth,
        outlier_prior_by_grave=outlier_prior_by_grave,
    )

    if _is_verbose(args) or _is_debug(args):
        idata = fit_parametric_pymc_seriation(**fit_kwargs)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            idata = fit_parametric_pymc_seriation(**fit_kwargs)

    if _is_debug(args):
        print_sampling_diagnostics(
            idata,
            include_richness=args.include_richness,
            grave_ids=grave_ids,
            c14_bp=c14_bp,
            c14_error=c14_error,
            unmodelled_cal_bp_mean=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_mean"),
            unmodelled_cal_bp_sd=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_sd"),
            unmodelled_cal_bp_hdi_3=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_hdi_3"),
            unmodelled_cal_bp_hdi_97=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_hdi_97"),
        )

    summary = summarise_graves(idata, grave_ids, ra_row_scores=row_scores)
    summary = _merge_unmodelled(summary, unmodelled)

    pymc_order = order_from_summary(summary, grave_ids)

    score_comparison = compare_pymc_to_ra_scores(
        posterior_t_mean=summary.set_index("grave_id")
        .loc[grave_ids, "posterior_t_mean"]
        .to_numpy(),
        ra_row_scores=row_scores,
        grave_ids=grave_ids,
    )

    type_summary = summarise_types(
        idata,
        type_ids,
        chain_flips=summary.attrs.get("pymc_chain_flips", []),
        ra_col_scores=col_scores,
    )

    # These are useful both for verbose printing and for CSV output, so compute
    # them once outside the verbosity branch.
    pairwise_probs = posterior_pairwise_order_probabilities(idata, row_scores)
    chain_diagnostics = chain_order_diagnostics(idata, ra_row_scores=row_scores)
    pairwise_uncertainty = pairwise_uncertainty_summary(
        pairwise_probs,
        grave_ids=grave_ids,
        order=pymc_order,
        n_pairs=20,
    )

    compact_sampling = _compact_sampling_summary(idata)
    parameter_diagnostics = build_parameter_diagnostics(idata)
    
    posthoc_outlier_candidates = _print_posthoc_outlier_screening(args, summary)

    active_outliers = build_outlier_table(
        idata,
        grave_ids=grave_ids,
        c14_bp=c14_bp,
        c14_error=c14_error,
        unmodelled_cal_bp_mean=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_mean"),
        unmodelled_cal_bp_sd=_by_grave(unmodelled, grave_ids, "unmodelled_cal_bp_sd"),
    )

    _print_compact_result_summary(
        args=args,
        idata=idata,
        summary=summary,
        score_comparison=score_comparison,
        pymc_order=pymc_order,
        ra_order=ra_order,
    )

    if _is_verbose(args) or _is_debug(args):
        print("\nChain-wise PyMC order diagnostics:")
        print(chain_diagnostics.to_string(index=False))

        print("\nLargest PyMC–RA rank differences:")
        print(
            score_comparison
            .assign(abs_diff=lambda d: d["rank_difference_pymc_minus_ra"].abs())
            .sort_values("abs_diff", ascending=False)
            .head(15)
            [
                [
                    "grave_id",
                    "pymc_rank",
                    "ra_rank",
                    "rank_difference_pymc_minus_ra",
                    "pymc_t_z",
                    "ra_score_z",
                ]
            ]
            .to_string(index=False)
        )

        print("\nMost uncertain adjacent pairs:")
        print(pairwise_uncertainty.to_string(index=False))

    _print_main_tables(args, summary, type_summary)
    _print_shift_summary(args, summary)

    metadata = _build_run_metadata(
        args=args,
        data=data,
        idata=idata,
        has_c14=has_c14,
        chronology_mode=chronology_mode,
        orientation_message=orientation_message,
        cal_span_mu=cal_span_mu,
        cal_span_sigma=cal_span_sigma,
        score_comparison=score_comparison,
        pymc_order=pymc_order,
        ra_order=ra_order,
    )

    result_tables = {
        "metadata": _metadata_table(metadata),
        "grave_summary": summary,
        "type_summary": type_summary,
        "score_comparison": score_comparison,
        "chain_diagnostics": chain_diagnostics,
        "pairwise_uncertainty": pairwise_uncertainty,
        "posthoc_outlier_candidates": posthoc_outlier_candidates,
        "active_outliers": active_outliers,
        "compact_sampling_summary": compact_sampling,
        "parameter_diagnostics": parameter_diagnostics,
        "unmodelled_calibration": unmodelled,
        "c14_input": _c14_input_table(grave_ids, c14_bp, c14_error),
    }

    _write_results_with_summary(
        args,
        tables=result_tables,
        metadata=metadata,
    )

    _make_plots_with_summary(
        args,
        Y,
        idata,
        grave_ids,
        c14_bp,
        c14_error,
        row_scores,
        col_scores,
        ra_order,
        pymc_order,
        score_comparison,
        summary,
        args.ra_method,
    )