from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bayser.classical import order_correlation


# -----------------------------------------------------------------------------
# Basic posterior access
# -----------------------------------------------------------------------------


def _samples(idata: az.InferenceData, var: str) -> np.ndarray:
    if var not in idata.posterior:
        raise KeyError(f"No '{var}' variable found in posterior.")
    return idata.posterior[var].values


def _mean(idata: az.InferenceData, var: str) -> np.ndarray:
    return idata.posterior[var].mean(dim=("chain", "draw")).values


def _sd(idata: az.InferenceData, var: str) -> np.ndarray:
    return idata.posterior[var].std(dim=("chain", "draw")).values


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def standardise_vector(x: np.ndarray, name: str = "vector") -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")

    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError(f"{name} must have non-zero variance.")

    return (x - float(np.mean(x))) / sd


def rank_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    return np.argsort(np.argsort(scores)) + 1


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of equal shape.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    rho = spearmanr(x, y).statistic
    return abs(float(rho)) if np.isfinite(rho) else float("nan")


def orient_scores_to_reference(
    scores: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if scores.shape != reference.shape:
        raise ValueError("scores and reference must have the same shape.")

    corr = _safe_pearson(scores, reference)

    if np.isfinite(corr) and corr < 0:
        return -scores, True, -corr

    return scores, False, corr


def orient_t_samples_to_reference(
    t_samples: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, list[bool]]:
    ref = standardise_vector(reference, "reference")
    t = np.asarray(t_samples, dtype=float).copy()

    if t.ndim != 3:
        raise ValueError("t_samples must have shape (chain, draw, grave).")
    if t.shape[-1] != len(ref):
        raise ValueError("reference length must match number of graves.")

    flips = []

    for c in range(t.shape[0]):
        corr = _safe_pearson(t[c].mean(axis=0), ref)

        if np.isfinite(corr) and corr < 0:
            t[c] = -t[c]
            flips.append(True)
        else:
            flips.append(False)

    return t, flips


def apply_chain_flips(samples: np.ndarray, chain_flips: list[bool]) -> np.ndarray:
    out = np.asarray(samples, dtype=float).copy()

    if out.shape[0] != len(chain_flips):
        raise ValueError("chain_flips must match first sample dimension.")

    for c, flip in enumerate(chain_flips):
        if flip:
            out[c] = -out[c]

    return out


# -----------------------------------------------------------------------------
# Calendar and probability summaries
# -----------------------------------------------------------------------------


def _posterior_var_summary(
    idata: az.InferenceData,
    var: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = _samples(idata, var)

    if values.ndim < 3:
        raise ValueError(f"'{var}' must have item/grave as trailing dimension.")

    s = values.reshape(-1, values.shape[-1])

    return (
        s.mean(axis=0),
        s.std(axis=0),
        np.quantile(s, 0.03, axis=0),
        np.quantile(s, 0.97, axis=0),
    )


def _add_var_summary(
    out: pd.DataFrame,
    idata: az.InferenceData,
    var: str,
    prefix: str,
    source_col: str | None = None,
) -> pd.DataFrame:
    if var not in idata.posterior:
        return out

    mean, sd, hdi_3, hdi_97 = _posterior_var_summary(idata, var)

    if len(mean) != len(out):
        raise ValueError(
            f"'{var}' has trailing dimension {len(mean)}, but grave summary has "
            f"{len(out)} rows. Use _add_probability_summary_by_grave for "
            "variables stored on c14_grave."
        )

    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_sd"] = sd
    out[f"{prefix}_hdi_3"] = hdi_3
    out[f"{prefix}_hdi_97"] = hdi_97

    if source_col is not None:
        out[source_col] = var

    return out


def _c14_index_for_var(
    idata: az.InferenceData,
    var: str,
    n_values: int,
    n_graves: int,
) -> np.ndarray:
    """Return grave indices for a posterior variable defined on c14_grave.

    Falls back to 0..n_values-1 if no c14_grave coordinate is present.
    """

    da = idata.posterior[var]

    if "c14_grave" in da.coords:
        idx = np.asarray(da.coords["c14_grave"].values, dtype=int)

        if len(idx) == n_values:
            return idx

    if n_values == n_graves:
        return np.arange(n_graves, dtype=int)

    return np.arange(n_values, dtype=int)


def _add_probability_summary_by_grave(
    out: pd.DataFrame,
    idata: az.InferenceData,
    var: str,
    prefix: str,
    fill_missing: float | None = np.nan,
) -> pd.DataFrame:
    """Add a posterior probability variable to the grave-level summary.

    Supports variables stored either on the full grave dimension or on the
    c14_grave dimension. Values are mapped back to grave_index.
    """

    if var not in idata.posterior:
        return out

    values = _samples(idata, var)

    if values.ndim < 3:
        raise ValueError(f"'{var}' must have a trailing item dimension.")

    flat = values.reshape(-1, values.shape[-1])

    n_items = flat.shape[1]
    n_graves = len(out)

    idx = _c14_index_for_var(
        idata=idata,
        var=var,
        n_values=n_items,
        n_graves=n_graves,
    )

    if len(idx) != n_items:
        raise ValueError(
            f"Could not map '{var}' to grave summary: got {len(idx)} indices "
            f"for {n_items} posterior columns."
        )

    if np.any(idx < 0) or np.any(idx >= n_graves):
        raise ValueError(f"'{var}' contains invalid grave indices.")

    mean = np.full(n_graves, fill_missing, dtype=float)
    sd = np.full(n_graves, fill_missing, dtype=float)
    hdi_3 = np.full(n_graves, fill_missing, dtype=float)
    hdi_97 = np.full(n_graves, fill_missing, dtype=float)

    mean[idx] = np.nanmean(flat, axis=0)
    sd[idx] = np.nanstd(flat, axis=0)
    hdi_3[idx] = np.nanquantile(flat, 0.03, axis=0)
    hdi_97[idx] = np.nanquantile(flat, 0.97, axis=0)

    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_sd"] = sd
    out[f"{prefix}_hdi_3"] = hdi_3
    out[f"{prefix}_hdi_97"] = hdi_97

    return out


# -----------------------------------------------------------------------------
# Oriented posterior summaries
# -----------------------------------------------------------------------------


def oriented_posterior_t_summary(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    t, flips = orient_t_samples_to_reference(_samples(idata, "t"), reference)
    s = t.reshape(-1, t.shape[-1])
    return s.mean(axis=0), s.std(axis=0), flips


def oriented_posterior_mu_summary(
    idata: az.InferenceData,
    chain_flips: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    mu = apply_chain_flips(_samples(idata, "mu"), chain_flips)
    s = mu.reshape(-1, mu.shape[-1])
    return s.mean(axis=0), s.std(axis=0)


# -----------------------------------------------------------------------------
# Rank and pairwise summaries
# -----------------------------------------------------------------------------


def posterior_rank_samples(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> np.ndarray:
    t, _ = orient_t_samples_to_reference(_samples(idata, "t"), reference)
    s = t.reshape(-1, t.shape[-1])

    ranks = np.empty_like(s, dtype=int)

    for i in range(s.shape[0]):
        order = np.argsort(s[i])
        ranks[i, order] = np.arange(1, s.shape[1] + 1)

    return ranks


def posterior_rank_summary(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> pd.DataFrame:
    ranks = posterior_rank_samples(idata, reference)

    return pd.DataFrame(
        {
            "grave_index": np.arange(ranks.shape[1]),
            "posterior_rank_mean": ranks.mean(axis=0),
            "posterior_rank_sd": ranks.std(axis=0),
            "posterior_rank_hdi_3": np.quantile(ranks, 0.03, axis=0),
            "posterior_rank_hdi_97": np.quantile(ranks, 0.97, axis=0),
            "posterior_rank_p10": np.quantile(ranks, 0.10, axis=0),
            "posterior_rank_p90": np.quantile(ranks, 0.90, axis=0),
        }
    )


def posterior_pairwise_order_probabilities(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> np.ndarray:
    t, _ = orient_t_samples_to_reference(_samples(idata, "t"), reference)
    s = t.reshape(-1, t.shape[-1])

    n = s.shape[1]
    P = np.zeros((n, n), dtype=float)

    for i in range(n):
        P[i, :] = (s[:, i, None] < s).mean(axis=0)

    np.fill_diagonal(P, 0.5)
    return P


def pairwise_uncertainty_summary(
    pairwise_probs: np.ndarray,
    grave_ids: list[str],
    order: np.ndarray,
    n_pairs: int = 20,
) -> pd.DataFrame:
    rows = []

    for left, right in zip(order[:-1], order[1:]):
        p = float(pairwise_probs[left, right])
        u = abs(p - 0.5)

        rows.append(
            {
                "left_grave": grave_ids[left],
                "right_grave": grave_ids[right],
                "p_left_before_right": p,
                "uncertainty_distance_from_0_5": u,
                "interpretation": (
                    "highly_uncertain"
                    if u < 0.05
                    else "moderately_uncertain"
                    if u < 0.15
                    else "comparatively_stable"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("uncertainty_distance_from_0_5")
        .head(n_pairs)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Chain and RA comparison
# -----------------------------------------------------------------------------


def chain_order_diagnostics(
    idata: az.InferenceData,
    ra_row_scores: np.ndarray,
) -> pd.DataFrame:
    t, flips = orient_t_samples_to_reference(_samples(idata, "t"), ra_row_scores)
    ra_order = np.argsort(ra_row_scores)

    rows = []

    for c in range(t.shape[0]):
        chain_mean = t[c].mean(axis=0)
        chain_sd = t[c].std(axis=0)
        chain_order = np.argsort(chain_mean)

        rows.append(
            {
                "chain": c,
                "chain_flipped": flips[c],
                "chain_t_mean_range": float(chain_mean.max() - chain_mean.min()),
                "chain_t_sd_median": float(np.median(chain_sd)),
                "chain_vs_ra_spearman_abs": order_correlation(
                    chain_order,
                    ra_order,
                    orientation_free=True,
                ),
                "chain_vs_ra_spearman_signed": order_correlation(
                    chain_order,
                    ra_order,
                    orientation_free=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_pymc_to_ra_scores(
    posterior_t_mean: np.ndarray,
    ra_row_scores: np.ndarray,
    grave_ids: list[str],
) -> pd.DataFrame:
    posterior_t_mean = np.asarray(posterior_t_mean, dtype=float)
    ra_row_scores = np.asarray(ra_row_scores, dtype=float)

    if len(posterior_t_mean) != len(ra_row_scores):
        raise ValueError("posterior_t_mean and ra_row_scores must have same length.")
    if len(grave_ids) != len(posterior_t_mean):
        raise ValueError("grave_ids length must match score vectors.")

    ra_oriented, ra_flipped, pearson_abs = orient_scores_to_reference(
        ra_row_scores,
        posterior_t_mean,
    )

    pymc_z = standardise_vector(posterior_t_mean, "posterior_t_mean")
    ra_z = standardise_vector(ra_oriented, "ra_oriented")

    pymc_rank = rank_from_scores(pymc_z)
    ra_rank = rank_from_scores(ra_z)

    out = pd.DataFrame(
        {
            "grave_id": grave_ids,
            "pymc_t_mean": posterior_t_mean,
            "pymc_t_z": pymc_z,
            "ra_score_oriented": ra_oriented,
            "ra_score_z": ra_z,
            "pymc_rank": pymc_rank,
            "ra_rank": ra_rank,
            "rank_difference_pymc_minus_ra": pymc_rank - ra_rank,
            "abs_rank_difference_pymc_ra": np.abs(pymc_rank - ra_rank),
            "ra_flipped_to_pymc": ra_flipped,
        }
    )

    out.attrs["pearson_abs"] = pearson_abs
    out.attrs["spearman_abs"] = _safe_spearman_abs(pymc_rank, ra_rank)
    out.attrs["ra_flipped_to_pymc"] = ra_flipped

    return out.sort_values("pymc_t_mean").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main summaries
# -----------------------------------------------------------------------------


def summarise_graves(
    idata: az.InferenceData,
    grave_ids: list[str],
    ra_row_scores: np.ndarray,
) -> pd.DataFrame:
    n_graves = idata.posterior["t"].shape[-1]

    if len(grave_ids) != n_graves:
        raise ValueError("grave_ids length must match posterior t dimension.")

    t_mean, t_sd, chain_flips = oriented_posterior_t_summary(
        idata,
        ra_row_scores,
    )

    posterior_rank = rank_from_scores(t_mean)

    out = pd.DataFrame(
        {
            "grave_id": grave_ids,
            "grave_index": np.arange(n_graves),
            "posterior_t_mean": t_mean,
            "posterior_t_sd": t_sd,
            "posterior_rank": posterior_rank,
            "pymc_orientation_target": "ra",
            "pymc_chain_flips": ",".join(str(x) for x in chain_flips),
        }
    )

    out = out.merge(
        posterior_rank_summary(idata, ra_row_scores),
        on="grave_index",
        how="left",
    )

    out.attrs["pymc_chain_flips"] = chain_flips
    out.attrs["pymc_orientation_target"] = "ra"

    out = _add_var_summary(
        out,
        idata,
        var="expected_cal_bp",
        prefix="expected_cal_bp",
    )

    out = _add_var_summary(
        out,
        idata,
        var="latent_cal_bp",
        prefix="posterior_cal_bp",
        source_col="posterior_cal_bp_source",
    )

    out = _add_var_summary(
        out,
        idata,
        var="latent_cal_bp_cond_mean",
        prefix="posterior_cal_bp_cond_mean",
    )

    out = _add_var_summary(
        out,
        idata,
        var="latent_cal_bp_cond_sd",
        prefix="posterior_cal_bp_cond_sd",
    )

    # Optional OxCal-style outlier probabilities.
    #
    # p_outlier is usually stored on c14_grave and is mapped back to grave_index.
    # p_outlier_reconstructed is usually stored on the full grave dimension, with
    # NaN for non-dated or inactive graves depending on the reconstruction logic.
    out = _add_probability_summary_by_grave(
        out,
        idata,
        var="p_outlier",
        prefix="p_outlier",
    )

    out = _add_probability_summary_by_grave(
        out,
        idata,
        var="p_outlier_reconstructed",
        prefix="p_outlier_reconstructed",
    )

    if {"posterior_cal_bp_mean", "expected_cal_bp_mean"}.issubset(out.columns):
        out["model_shift_cal_bp"] = (
            out["posterior_cal_bp_mean"] - out["expected_cal_bp_mean"]
        )
        out["abs_model_shift_cal_bp"] = out["model_shift_cal_bp"].abs()

    if {"posterior_cal_bp_mean", "unmodelled_cal_bp_mean"}.issubset(out.columns):
        out["shift_from_unmodelled_calibration"] = (
            out["posterior_cal_bp_mean"] - out["unmodelled_cal_bp_mean"]
        )
        out["abs_shift_from_unmodelled_calibration"] = out[
            "shift_from_unmodelled_calibration"
        ].abs()

    ra_oriented, ra_flipped, ra_pearson = orient_scores_to_reference(
        ra_row_scores,
        t_mean,
    )

    ra_rank = rank_from_scores(ra_oriented)

    out["ra_score_oriented"] = ra_oriented
    out["ra_rank"] = ra_rank
    out["rank_difference_pymc_minus_ra"] = posterior_rank - ra_rank
    out["abs_rank_difference_pymc_ra"] = out["rank_difference_pymc_minus_ra"].abs()
    out["ra_flipped"] = ra_flipped
    out["ra_orientation_target"] = "pymc_t"
    out["ra_orientation_pearson_abs"] = ra_pearson

    return out.sort_values("posterior_t_mean").reset_index(drop=True)


def summarise_types(
    idata: az.InferenceData,
    type_ids: list[str],
    chain_flips: list[bool],
    ra_col_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    n_types = idata.posterior["mu"].shape[-1]

    if len(type_ids) != n_types:
        raise ValueError("type_ids length must match posterior mu dimension.")

    mu, mu_sd = oriented_posterior_mu_summary(idata, chain_flips)
    posterior_type_rank = rank_from_scores(mu)

    out = pd.DataFrame(
        {
            "type_id": type_ids,
            "type_index": np.arange(n_types),
            "posterior_mu_mean": mu,
            "posterior_mu_sd": mu_sd,
            "posterior_sigma_mean": _mean(idata, "sigma"),
            "posterior_sigma_sd": _sd(idata, "sigma"),
            "posterior_a_mean": _mean(idata, "a"),
            "posterior_a_sd": _sd(idata, "a"),
            "posterior_type_rank": posterior_type_rank,
        }
    )

    if ra_col_scores is not None:
        ra_oriented, ra_flipped, ra_pearson = orient_scores_to_reference(
            ra_col_scores,
            mu,
        )

        ra_type_rank = rank_from_scores(ra_oriented)

        out["ra_type_score_oriented"] = ra_oriented
        out["ra_type_rank"] = ra_type_rank
        out["type_rank_difference_pymc_minus_ra"] = (
            posterior_type_rank - ra_type_rank
        )
        out["abs_type_rank_difference_pymc_ra"] = (
            out["type_rank_difference_pymc_minus_ra"].abs()
        )
        out["ra_type_flipped"] = ra_flipped
        out["ra_type_orientation_target"] = "pymc_mu"
        out["ra_type_orientation_pearson_abs"] = ra_pearson

    return out.sort_values("posterior_mu_mean").reset_index(drop=True)


def order_from_summary(
    summary: pd.DataFrame,
    grave_ids: list[str],
) -> np.ndarray:
    ordered_ids = summary.sort_values("posterior_t_mean")["grave_id"].tolist()
    index = {g: i for i, g in enumerate(grave_ids)}

    missing = [g for g in ordered_ids if g not in index]
    if missing:
        raise ValueError(
            "Summary contains grave IDs not present in grave_ids: "
            + ", ".join(missing[:10])
        )

    return np.array([index[g] for g in ordered_ids], dtype=int)