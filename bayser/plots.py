from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayser.posterior import orient_scores_to_reference, posterior_rank_samples


# -----------------------------------------------------------------------------
# Plot output configuration
# -----------------------------------------------------------------------------

_PLOT_DIR: Path | None = Path("plots")
_SHOW_PLOTS = False
_DPI = 200


def configure_plots(
    plot_dir: str | Path | None = "plots",
    show_plots: bool = False,
    dpi: int = 200,
) -> None:
    """Configure plot output.

    By default, plots are saved to files and closed. Set show_plots=True for
    interactive display instead.
    """

    global _PLOT_DIR, _SHOW_PLOTS, _DPI

    _SHOW_PLOTS = bool(show_plots)
    _DPI = int(dpi)

    if _SHOW_PLOTS:
        _PLOT_DIR = None
        return

    _PLOT_DIR = Path(plot_dir or "plots")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _slug(title: str) -> str:
    s = title.lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "plot"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _has(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(df.columns)


def _finite(*arrays) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)

    for a in arrays:
        a = np.asarray(a, dtype=float)
        if len(a) != len(mask):
            raise ValueError("All arrays must have equal length.")
        mask &= np.isfinite(a)

    return mask


def _err(mean, lower, upper) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    return np.vstack(
        [
            np.maximum(mean - lower, 0.0),
            np.maximum(upper - mean, 0.0),
        ]
    )


def _ordered(
    summary: pd.DataFrame, order: np.ndarray, grave_ids: list[str]
) -> pd.DataFrame:
    ids = [grave_ids[i] for i in order]
    s = summary.set_index("grave_id")

    missing = [g for g in ids if g not in s.index]
    if missing:
        raise ValueError("Summary is missing assemblage IDs: " + ", ".join(missing[:10]))

    return s.loc[ids].copy()


def _width(n: int, min_width: float = 12, max_width: float = 32) -> float:
    return max(min_width, min(max_width, n * 0.28))


def _xticks(
    ax: plt.Axes, x: np.ndarray, labels: list[str], max_labels: int = 80
) -> None:
    if len(labels) <= max_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])


def _cal_cols(prefix: str) -> set[str]:
    return {f"{prefix}_mean", f"{prefix}_hdi_3", f"{prefix}_hdi_97"}


def _cal_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    df: pd.DataFrame,
    prefix: str,
    label: str,
    x_offset: float = 0.0,
    alpha: float = 1.0,
    marker: str = "o",
) -> bool:
    if not _has(df, _cal_cols(prefix)):
        return False

    y = df[f"{prefix}_mean"].to_numpy(float)
    lo = df[f"{prefix}_hdi_3"].to_numpy(float)
    hi = df[f"{prefix}_hdi_97"].to_numpy(float)

    keep = _finite(y, lo, hi)
    if not np.any(keep):
        return False

    ax.errorbar(
        x[keep] + x_offset,
        y[keep],
        yerr=_err(y[keep], lo[keep], hi[keep]),
        fmt=marker,
        capsize=2,
        alpha=alpha,
        label=label,
    )

    return True


def _finish(fig: plt.Figure, title: str) -> None:
    fig.tight_layout()

    if _SHOW_PLOTS:
        plt.show()
        return

    if _PLOT_DIR is None:
        plt.close(fig)
        return

    path = _PLOT_DIR / f"{_slug(title)}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", path)


# -----------------------------------------------------------------------------
# Matrix and order plots
# -----------------------------------------------------------------------------


def plot_matrix(
    Y: np.ndarray,
    order: np.ndarray,
    title: str,
    type_order: Optional[np.ndarray] = None,
) -> None:
    Y_ord = Y[order, :]

    if type_order is None:
        weights = np.arange(Y_ord.shape[0])[:, None] + 1
        denom = np.maximum(Y_ord.sum(axis=0), 1)
        type_order = np.argsort((Y_ord * weights).sum(axis=0) / denom)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(Y_ord[:, type_order], aspect="auto", interpolation="nearest")
    ax.set_xlabel("Types")
    ax.set_ylabel("Assemblages")
    ax.set_title(title)

    _finish(fig, title)


def plot_pymc_vs_ra_scores(comparison: pd.DataFrame, title: str) -> None:
    if not _has(comparison, {"ra_score_z", "pymc_t_z"}):
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(comparison["ra_score_z"], comparison["pymc_t_z"])

    ax.axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.0, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("RA/CA row score, oriented and standardised")
    ax.set_ylabel("PyMC posterior t, standardised")
    ax.set_title(title)

    _finish(fig, title)


def plot_posterior_rank_distributions(
    idata: az.InferenceData,
    grave_ids: list[str],
    reference: np.ndarray,
    ra_row_scores: np.ndarray,
    posterior_order: np.ndarray,
    title: str,
    max_labels: int = 80,
) -> None:
    ranks = posterior_rank_samples(idata, reference=reference)

    ra_oriented, _, _ = orient_scores_to_reference(ra_row_scores, reference)
    ra_rank = np.argsort(np.argsort(ra_oriented)) + 1

    ordered_ranks = [ranks[:, i] for i in posterior_order]
    ordered_ra_ranks = ra_rank[posterior_order]
    ordered_labels = [grave_ids[i] for i in posterior_order]

    x = np.arange(1, len(grave_ids) + 1)

    fig, ax = plt.subplots(figsize=(_width(len(grave_ids)), 7))

    ax.boxplot(
        ordered_ranks,
        positions=x,
        widths=0.65,
        showfliers=False,
        manage_ticks=False,
    )

    ax.scatter(
        x,
        ordered_ra_ranks,
        marker="x",
        s=35,
        label="RA/CA rank",
        zorder=3,
    )

    ax.set_xlabel("Assemblages ordered by PyMC posterior mean position")
    ax.set_ylabel("Posterior rank")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()

    _xticks(ax, x, ordered_labels, max_labels=max_labels)
    _finish(fig, title)


# -----------------------------------------------------------------------------
# C14 and calendar plots
# -----------------------------------------------------------------------------


def plot_c14_against_order(
    grave_ids: list[str],
    c14_bp: np.ndarray,
    c14_error: np.ndarray,
    order: np.ndarray,
    title: str,
) -> None:
    bp = np.asarray(c14_bp, dtype=float)[order]
    err = np.asarray(c14_error, dtype=float)[order]
    labels = [grave_ids[i] for i in order]
    x = np.arange(1, len(order) + 1)

    keep = _finite(bp, err)
    if not np.any(keep):
        return

    fig, ax = plt.subplots(figsize=(_width(len(order)), 6))

    ax.errorbar(
        x[keep],
        bp[keep],
        yerr=err[keep],
        fmt="o",
        capsize=2,
    )

    ax.set_xlabel("Seriation rank")
    ax.set_ylabel("Observed uncalibrated radiocarbon age BP")
    ax.set_title(title)
    ax.invert_yaxis()

    _xticks(ax, x, labels)
    _finish(fig, title)


def plot_posterior_cal_bp_against_order(
    summary: pd.DataFrame,
    order: np.ndarray,
    grave_ids: list[str],
    title: str,
) -> None:
    prefixes = [
        "unmodelled_cal_bp",
        "expected_cal_bp",
        "posterior_cal_bp",
    ]

    if not any(_has(summary, _cal_cols(p)) for p in prefixes):
        return

    ordered = _ordered(summary, order, grave_ids)
    labels = [grave_ids[i] for i in order]
    x = np.arange(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(_width(len(order)), 6))

    plotted = False
    plotted |= _cal_errorbar(
        ax,
        x,
        ordered,
        "unmodelled_cal_bp",
        "Unmodelled single-date calibration",
        x_offset=-0.18,
        alpha=0.45,
    )
    plotted |= _cal_errorbar(
        ax,
        x,
        ordered,
        "expected_cal_bp",
        "Typological calendar expectation",
        marker="s",
        alpha=0.75,
    )
    plotted |= _cal_errorbar(
        ax,
        x,
        ordered,
        "posterior_cal_bp",
        "Modelled individual posterior calendar age",
        x_offset=0.18,
    )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Seriation rank")
    ax.set_ylabel("Calendar age cal BP")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()

    _xticks(ax, x, labels)
    _finish(fig, title)


def plot_observed_bp_vs_posterior_cal_bp(
    summary: pd.DataFrame,
    c14_bp: np.ndarray,
    c14_error: np.ndarray,
    grave_ids: list[str],
    title: str,
) -> None:
    prefixes = [
        "unmodelled_cal_bp",
        "expected_cal_bp",
        "posterior_cal_bp",
    ]

    if not any(_has(summary, _cal_cols(p)) for p in prefixes):
        return

    s = summary.set_index("grave_id").loc[grave_ids]
    x = np.asarray(c14_bp, dtype=float)
    xerr = np.asarray(c14_error, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    plotted = False

    for prefix, label, alpha, marker in [
        ("unmodelled_cal_bp", "Unmodelled single-date calibration", 0.45, "o"),
        ("expected_cal_bp", "Typological calendar expectation", 0.75, "s"),
        ("posterior_cal_bp", "Modelled individual posterior calendar age", 1.0, "o"),
    ]:
        if not _has(s, _cal_cols(prefix)):
            continue

        y = s[f"{prefix}_mean"].to_numpy(float)
        lo = s[f"{prefix}_hdi_3"].to_numpy(float)
        hi = s[f"{prefix}_hdi_97"].to_numpy(float)

        keep = _finite(x, xerr, y, lo, hi)
        if not np.any(keep):
            continue

        ax.errorbar(
            x[keep],
            y[keep],
            xerr=xerr[keep],
            yerr=_err(y[keep], lo[keep], hi[keep]),
            fmt=marker,
            capsize=2,
            alpha=alpha,
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Observed uncalibrated radiocarbon age BP")
    ax.set_ylabel("Calendar age cal BP")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()

    _finish(fig, title)


def plot_unmodelled_vs_modelled_cal_bp(
    summary: pd.DataFrame,
    title: str,
    label_n: int = 8,
) -> None:
    required = {
        "grave_id",
        "unmodelled_cal_bp_mean",
        "unmodelled_cal_bp_hdi_3",
        "unmodelled_cal_bp_hdi_97",
        "posterior_cal_bp_mean",
        "posterior_cal_bp_hdi_3",
        "posterior_cal_bp_hdi_97",
    }

    if not _has(summary, required):
        return

    df = summary.copy()

    x = df["unmodelled_cal_bp_mean"].to_numpy(float)
    xlo = df["unmodelled_cal_bp_hdi_3"].to_numpy(float)
    xhi = df["unmodelled_cal_bp_hdi_97"].to_numpy(float)

    y = df["posterior_cal_bp_mean"].to_numpy(float)
    ylo = df["posterior_cal_bp_hdi_3"].to_numpy(float)
    yhi = df["posterior_cal_bp_hdi_97"].to_numpy(float)

    keep = _finite(x, xlo, xhi, y, ylo, yhi)
    if not np.any(keep):
        return

    df = df.loc[keep].copy()
    x, xlo, xhi = x[keep], xlo[keep], xhi[keep]
    y, ylo, yhi = y[keep], ylo[keep], yhi[keep]

    min_age = float(np.nanmin([xlo, ylo]))
    max_age = float(np.nanmax([xhi, yhi]))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.errorbar(
        x,
        y,
        xerr=_err(x, xlo, xhi),
        yerr=_err(y, ylo, yhi),
        fmt="o",
        capsize=2,
    )

    ax.plot(
        [min_age, max_age],
        [min_age, max_age],
        linestyle="--",
        linewidth=1,
        label="No shift relative to unmodelled calibration",
    )

    df["shift"] = y - x

    label_rows = df.assign(abs_shift=lambda d: d["shift"].abs()).nlargest(
        label_n, "abs_shift"
    )

    for _, row in label_rows.iterrows():
        ax.annotate(
            str(row["grave_id"]),
            (row["unmodelled_cal_bp_mean"], row["posterior_cal_bp_mean"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Unmodelled single-date calibration cal BP")
    ax.set_ylabel("Modelled individual posterior calendar age cal BP")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()

    _finish(fig, title)


def plot_model_shift_against_rank(
    summary: pd.DataFrame,
    order: np.ndarray,
    grave_ids: list[str],
    title: str,
) -> None:
    if "posterior_cal_bp_mean" not in summary.columns:
        return

    has_expected = "expected_cal_bp_mean" in summary.columns
    has_unmodelled = "unmodelled_cal_bp_mean" in summary.columns

    if not has_expected and not has_unmodelled:
        return

    ordered = _ordered(summary, order, grave_ids)
    labels = [grave_ids[i] for i in order]
    x = np.arange(1, len(ordered) + 1)

    fig, ax = plt.subplots(figsize=(_width(len(order)), 5))
    plotted = False

    if has_expected:
        y = ordered["posterior_cal_bp_mean"].to_numpy(float) - ordered[
            "expected_cal_bp_mean"
        ].to_numpy(float)
        keep = np.isfinite(y)

        if np.any(keep):
            ax.scatter(
                x[keep],
                y[keep],
                label="Posterior individual age minus typological expectation",
            )
            plotted = True

    if has_unmodelled:
        y = ordered["posterior_cal_bp_mean"].to_numpy(float) - ordered[
            "unmodelled_cal_bp_mean"
        ].to_numpy(float)
        keep = np.isfinite(y)

        if np.any(keep):
            ax.scatter(
                x[keep],
                y[keep],
                marker="x",
                alpha=0.7,
                label="Posterior individual age minus unmodelled calibration",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Seriation rank")
    ax.set_ylabel("Calendar-age shift in years")
    ax.set_title(title)
    ax.legend()

    _xticks(ax, x, labels)
    _finish(fig, title)
