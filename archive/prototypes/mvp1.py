"""
Pure PyMC seriation prototype with simulation, Münsingen import, and RA comparison
================================================================================

This script is a simplified test bed for a pure PyMC approach to archaeological
seriation from a grave × type presence/absence matrix.

It can run in two modes:

1. Simulated data
   A validated synthetic grave × type matrix is generated from unimodal type
   careers. The true order is known, so recovery can be measured directly.

2. Münsingen data
   A CSV matrix such as the Rdatasets/folio `munsingen.csv` file is loaded. Rows
   are graves, columns are artefact types, and cells are 0/1. The first column
   `rownames` is treated as grave IDs.

The PyMC model is deliberately parametric:

    logit(p_ij) = intercept + a_j + g_i - (t_i - mu_j)^2 / (2 * sigma_j^2)

where:

    t_i       latent chronological coordinate of grave i
    mu_j      centre of artefact type j
    sigma_j   temporal spread of artefact type j
    a_j       type-specific peak tendency
    g_i       optional grave-specific richness / inclusion tendency

The model does not use correspondence analysis and does not use an external
permutation optimiser. All model unknowns are inferred inside PyMC.

For comparison only, the script also computes a classical Reciprocal Averaging /
Correspondence Analysis order from the same matrix. This order is not passed to
PyMC.

Important post-processing note
------------------------------
The latent PyMC axis is arbitrary up to sign. Therefore, posterior t samples are
oriented chain-wise before they are averaged. For simulations, chains are
oriented to the known true axis. For real data such as Münsingen, chains are
oriented to the RA/CA axis only as a post-hoc reporting convention; RA is never
used in the PyMC model.

Examples
--------
Run validated simulation:

    uv run python mvp1.py simulate

Run Münsingen:

    uv run python mvp1.py munsingen --csv munsingen.csv

Use fewer draws for quick tests:

    uv run python mvp1.py munsingen --csv munsingen.csv --draws 400 --tune 600

Dependencies:

    uv add numpy pandas matplotlib pymc arviz scipy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import spearmanr


@dataclass
class SimulatedData:
    """Container for synthetic seriation data."""

    Y: np.ndarray
    grave_ids: list[str]
    type_ids: list[str]
    true_t_observed: np.ndarray
    true_order: np.ndarray
    true_mu: np.ndarray
    true_sigma: np.ndarray
    true_a: np.ndarray
    true_intercept: float
    true_richness_observed: np.ndarray
    seed_used: int
    simulation_attempt: int


@dataclass
class RealData:
    """Container for real seriation data."""

    Y: np.ndarray
    grave_ids: list[str]
    type_ids: list[str]
    source: str


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically simple logistic transform for NumPy arrays."""

    return 1.0 / (1.0 + np.exp(-x))


def simulate_moderate_data(
    n_graves: int = 50,
    n_types: int = 22,
    seed: int = 42,
    shuffle: bool = True,
) -> SimulatedData:
    """Simulate a moderately difficult synthetic seriation dataset."""

    rng = np.random.default_rng(seed)

    t_sorted = np.linspace(-2.2, 2.2, n_graves)
    mu = np.sort(rng.uniform(-1.9, 1.9, size=n_types))
    sigma = rng.uniform(0.40, 0.80, size=n_types)
    a = rng.normal(1.40, 0.35, size=n_types)
    richness_sorted = rng.normal(0.0, 0.25, size=n_graves)
    intercept = -0.65

    eta = (
        intercept
        + richness_sorted[:, None]
        + a[None, :]
        - ((t_sorted[:, None] - mu[None, :]) ** 2) / (2.0 * sigma[None, :] ** 2)
    )

    p = logistic(eta)
    Y_sorted = rng.binomial(1, p).astype(int)

    if shuffle:
        perm = rng.permutation(n_graves)
        Y = Y_sorted[perm, :]
        true_t_observed = t_sorted[perm]
        richness_observed = richness_sorted[perm]
        true_order = np.argsort(true_t_observed)
    else:
        Y = Y_sorted
        true_t_observed = t_sorted
        richness_observed = richness_sorted
        true_order = np.arange(n_graves)

    grave_ids = [f"G{i + 1:03d}" for i in range(n_graves)]
    type_ids = [f"T{j + 1:03d}" for j in range(n_types)]

    return SimulatedData(
        Y=Y,
        grave_ids=grave_ids,
        type_ids=type_ids,
        true_t_observed=true_t_observed,
        true_order=true_order,
        true_mu=mu,
        true_sigma=sigma,
        true_a=a,
        true_intercept=intercept,
        true_richness_observed=richness_observed,
        seed_used=seed,
        simulation_attempt=1,
    )


def matrix_is_informative(
    Y: np.ndarray,
    min_type_count: int = 2,
    min_grave_count: int = 2,
    max_type_frequency: Optional[int] = None,
) -> bool:
    """Check whether a matrix meets simple information criteria."""

    type_counts = Y.sum(axis=0)
    grave_counts = Y.sum(axis=1)

    if np.any(type_counts < min_type_count):
        return False
    if np.any(grave_counts < min_grave_count):
        return False
    if max_type_frequency is not None and np.any(type_counts > max_type_frequency):
        return False

    return True


def simulate_valid_moderate_data(
    n_graves: int = 50,
    n_types: int = 22,
    seed: int = 42,
    shuffle: bool = True,
    min_type_count: int = 2,
    min_grave_count: int = 2,
    max_attempts: int = 500,
) -> SimulatedData:
    """Repeat moderate simulation until the matrix is sufficiently informative."""

    for attempt in range(1, max_attempts + 1):
        current_seed = seed + attempt - 1
        data = simulate_moderate_data(
            n_graves=n_graves,
            n_types=n_types,
            seed=current_seed,
            shuffle=shuffle,
        )

        if matrix_is_informative(
            data.Y,
            min_type_count=min_type_count,
            min_grave_count=min_grave_count,
        ):
            data.seed_used = current_seed
            data.simulation_attempt = attempt
            return data

    raise RuntimeError(
        "Could not simulate a valid matrix after "
        f"{max_attempts} attempts. Consider increasing n_types, broadening "
        "type careers, or lowering the minimum count thresholds."
    )


def load_presence_absence_csv(
    path: str,
    id_col: Optional[str] = None,
    min_type_count: int = 2,
    min_grave_count: int = 2,
    filter_matrix: bool = True,
) -> RealData:
    """Load a grave × type matrix from CSV.

    Expected format for Münsingen/Rdatasets:

        rownames,A1,A2,...,A70
        G42,0,0,...
        G07,0,0,...

    Counts greater than 0 are converted to presence = 1.
    """

    df = pd.read_csv(path)

    if id_col is None:
        for candidate in ["rownames", "grave_id", "grave", "Grave", "id", "ID", "Unnamed: 0"]:
            if candidate in df.columns:
                id_col = candidate
                break

    if id_col is not None and id_col in df.columns:
        grave_ids = df[id_col].astype(str).tolist()
        df = df.drop(columns=[id_col])
    else:
        grave_ids = [f"G{i + 1:03d}" for i in range(df.shape[0])]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    Y = (df.to_numpy() > 0).astype(int)
    type_ids = df.columns.astype(str).tolist()

    if filter_matrix:
        Y, grave_ids, type_ids = filter_informative_matrix(
            Y,
            grave_ids,
            type_ids,
            min_type_count=min_type_count,
            min_grave_count=min_grave_count,
        )

    return RealData(Y=Y, grave_ids=grave_ids, type_ids=type_ids, source=path)


def filter_informative_matrix(
    Y: np.ndarray,
    grave_ids: list[str],
    type_ids: list[str],
    min_type_count: int = 2,
    min_grave_count: int = 2,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Iteratively remove rare types and poorly furnished graves."""

    Y_f = Y.copy()
    grave_ids_f = list(grave_ids)
    type_ids_f = list(type_ids)

    changed = True
    while changed:
        changed = False

        type_keep = Y_f.sum(axis=0) >= min_type_count
        if not np.all(type_keep):
            Y_f = Y_f[:, type_keep]
            type_ids_f = [x for x, keep in zip(type_ids_f, type_keep) if keep]
            changed = True

        grave_keep = Y_f.sum(axis=1) >= min_grave_count
        if not np.all(grave_keep):
            Y_f = Y_f[grave_keep, :]
            grave_ids_f = [x for x, keep in zip(grave_ids_f, grave_keep) if keep]
            changed = True

    if Y_f.shape[0] < 3 or Y_f.shape[1] < 3:
        raise ValueError("Filtered matrix is too small for seriation.")

    return Y_f, grave_ids_f, type_ids_f


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
    return -strength * pt.sum(pt.exp(-dist2 / (2.0 * min_dist**2)) * mask)


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
) -> az.InferenceData:
    """Fit a pure PyMC parametric seriation model."""

    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be a binary matrix containing only 0 and 1.")

    n_graves, n_types = Y.shape
    coords = {"grave": np.arange(n_graves), "type": np.arange(n_types)}

    with pm.Model(coords=coords) as model:
        t_raw = pm.Normal("t_raw", 0.0, 1.0, dims="grave")
        t_centered = t_raw - pt.mean(t_raw)
        t = pm.Deterministic(
            "t",
            t_centered / pt.sqrt(pt.var(t_centered) + 1e-6),
            dims="grave",
        )

        pm.Potential(
            "spacing_repulsion",
            spacing_repulsion(t, strength=repulsion_strength, min_dist=0.10),
        )

        intercept = pm.Normal("intercept", mu=-0.5, sigma=0.8)
        mu = pm.Normal("mu", mu=0.0, sigma=1.4, dims="type")
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu_prior,
            sigma=sigma_sigma_prior,
            lower=0.20,
            upper=1.40,
            dims="type",
        )
        a = pm.Normal("a", mu=1.5, sigma=0.8, dims="type")

        if include_richness:
            sigma_g = pm.HalfNormal("sigma_g", sigma=0.4)
            g_raw = pm.Normal("g_raw", 0.0, 1.0, dims="grave")
            g = pm.Deterministic("g", sigma_g * (g_raw - pt.mean(g_raw)), dims="grave")
            grave_effect = g[:, None]
        else:
            grave_effect = 0.0

        dist_penalty = ((t[:, None] - mu[None, :]) ** 2) / (2.0 * sigma[None, :] ** 2)
        eta = intercept + a[None, :] + grave_effect - dist_penalty

        pm.Bernoulli("Y_obs", logit_p=eta, observed=Y, dims=("grave", "type"))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            init="jitter+adapt_diag",
        )

    return idata


def reciprocal_averaging_order(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a simple reciprocal averaging / correspondence-analysis order."""

    X = np.asarray(Y, dtype=float)
    grand_total = X.sum()
    if grand_total <= 0:
        raise ValueError("Cannot run reciprocal averaging on an empty matrix.")

    P = X / grand_total
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    if np.any(r <= 0) or np.any(c <= 0):
        raise ValueError("RA/CA requires no empty rows or columns.")

    expected = np.outer(r, c)
    S = (P - expected) / np.sqrt(expected)

    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)

    row_scores = U[:, 0] / np.sqrt(r) * singular_values[0]
    col_scores = Vt[0, :] / np.sqrt(c) * singular_values[0]

    grave_order = np.argsort(row_scores)
    type_order = np.argsort(col_scores)

    return grave_order, type_order, row_scores, col_scores


def posterior_mean(idata: az.InferenceData, var: str) -> np.ndarray:
    """Return posterior mean for a named variable."""

    return idata.posterior[var].mean(dim=("chain", "draw")).values


def posterior_t_samples(idata: az.InferenceData) -> np.ndarray:
    """Return posterior t samples as array with shape (chain, draw, grave)."""

    return idata.posterior["t"].values


def posterior_mu_samples(idata: az.InferenceData) -> np.ndarray:
    """Return posterior mu samples as array with shape (chain, draw, type)."""

    return idata.posterior["mu"].values


def orient_scores_to_reference(
    scores: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    """Flip a score vector if necessary to align it with a reference vector."""

    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(reference, dtype=float)

    corr = float(np.corrcoef(scores, reference)[0, 1])
    if corr < 0:
        return -scores, True, -corr
    return scores, False, corr


def orient_t_samples_to_reference(
    t_samples: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, list[bool]]:
    """Orient each chain of t samples to a reference axis."""

    ref = np.asarray(reference, dtype=float)
    ref = (ref - ref.mean()) / ref.std()

    oriented = t_samples.copy()
    flipped: list[bool] = []

    for c in range(oriented.shape[0]):
        chain_mean = oriented[c].mean(axis=0)
        corr = np.corrcoef(chain_mean, ref)[0, 1]
        if corr < 0:
            oriented[c] = -oriented[c]
            flipped.append(True)
        else:
            flipped.append(False)

    return oriented, flipped


def oriented_posterior_t_summary(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Summarise t after orienting each chain to a reference axis."""

    t = posterior_t_samples(idata)
    t_oriented, chain_flips = orient_t_samples_to_reference(t, reference)
    samples = t_oriented.reshape(-1, t_oriented.shape[-1])

    return samples.mean(axis=0), samples.std(axis=0), chain_flips


def oriented_posterior_mu_summary(
    idata: az.InferenceData,
    chain_flips: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Summarise mu after applying the same chain flips as for t.

    If a chain's t axis is flipped, the type centres mu from that chain must be
    flipped as well, otherwise grave and type axes become inconsistent.
    """

    mu = posterior_mu_samples(idata).copy()
    for c, flip in enumerate(chain_flips):
        if flip:
            mu[c] = -mu[c]

    samples = mu.reshape(-1, mu.shape[-1])
    return samples.mean(axis=0), samples.std(axis=0)


def rank_from_order(order: np.ndarray) -> np.ndarray:
    """Convert order vector into rank per original row index."""

    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank


def order_correlation(order_a: np.ndarray, order_b: np.ndarray, orientation_free: bool = True) -> float:
    """Spearman correlation between two orders."""

    ra = rank_from_order(order_a)
    rb = rank_from_order(order_b)
    rho = float(spearmanr(ra, rb).statistic)
    return abs(rho) if orientation_free else rho


def orient_order_to_reference(order: np.ndarray, reference_order: np.ndarray) -> np.ndarray:
    """Reverse an order if that better matches a reference order."""

    rank_a = rank_from_order(order)
    rank_b = rank_from_order(reference_order)
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    if corr < 0:
        return order[::-1].copy()
    return order.copy()


def chain_order_diagnostics(
    idata: az.InferenceData,
    ra_row_scores: np.ndarray,
    true_t: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare each PyMC chain order against RA and, if available, true order."""

    t = posterior_t_samples(idata)

    if true_t is not None:
        ref = (true_t - np.mean(true_t)) / np.std(true_t)
        ra_oriented, ra_flipped, _ = orient_scores_to_reference(ra_row_scores, ref)
        true_order = np.argsort(ref)
        orientation_target = "true_t"
    else:
        ra_oriented = np.asarray(ra_row_scores, dtype=float)
        ra_flipped = False
        true_order = None
        orientation_target = "ra"

    reference_for_chains = ref if true_t is not None else ra_oriented
    t_oriented, chain_flips = orient_t_samples_to_reference(t, reference_for_chains)
    ra_order = np.argsort(ra_oriented)

    rows = []
    for c in range(t_oriented.shape[0]):
        chain_mean = t_oriented[c].mean(axis=0)
        chain_order = np.argsort(chain_mean)

        row = {
            "chain": c,
            "chain_flipped": chain_flips[c],
            "orientation_target": orientation_target,
            "ra_flipped_to_true": ra_flipped if true_t is not None else False,
            "chain_vs_ra_spearman_abs": order_correlation(chain_order, ra_order, orientation_free=True),
            "chain_vs_ra_spearman_signed": order_correlation(chain_order, ra_order, orientation_free=False),
        }

        if true_order is not None:
            row["chain_vs_true_spearman"] = order_correlation(
                chain_order,
                true_order,
                orientation_free=False,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def posterior_pairwise_order_probabilities(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> np.ndarray:
    """Compute P(t_i < t_k) after orienting chains to a reference axis."""

    t = posterior_t_samples(idata)
    t_oriented, _ = orient_t_samples_to_reference(t, reference)
    samples = t_oriented.reshape(-1, t_oriented.shape[-1])

    n = samples.shape[1]
    P = np.zeros((n, n), dtype=float)

    for i in range(n):
        P[i, :] = (samples[:, i, None] < samples).mean(axis=0)

    return P


def posterior_rank_samples(
    idata: az.InferenceData,
    reference: np.ndarray,
) -> np.ndarray:
    """Compute posterior rank samples after orienting chains to a reference axis.

    Returns
    -------
    ranks:
        Array with shape (samples, graves). Each value is a 1-based rank position
        of the corresponding grave in one posterior draw.
    """

    t = posterior_t_samples(idata)
    t_oriented, _ = orient_t_samples_to_reference(t, reference)
    samples = t_oriented.reshape(-1, t_oriented.shape[-1])

    ranks = np.empty_like(samples, dtype=int)
    for s in range(samples.shape[0]):
        order = np.argsort(samples[s])
        ranks[s, order] = np.arange(1, samples.shape[1] + 1)

    return ranks


def plot_posterior_rank_distributions(
    idata: az.InferenceData,
    grave_ids: list[str],
    reference: np.ndarray,
    ra_row_scores: np.ndarray,
    posterior_order: np.ndarray,
    title: str,
    max_labels: int = 80,
) -> None:
    """Plot posterior rank distributions with RA ranks overlaid.

    The posterior samples are first oriented chain-wise to `reference`. For
    simulations this should be true_t; for Münsingen we currently use RA row
    scores as a post-hoc orientation reference.

    The x-axis follows the PyMC posterior mean order. Each box shows the
    posterior distribution of the rank position of one grave. The overlaid point
    shows the corresponding RA rank, oriented to the same reference.
    """

    ranks = posterior_rank_samples(idata, reference=reference)

    # Orient RA to the same reference used for posterior ranks.
    ra_oriented, _, _ = orient_scores_to_reference(ra_row_scores, reference)
    ra_rank = np.argsort(np.argsort(ra_oriented)) + 1

    ordered_ranks = [ranks[:, idx] for idx in posterior_order]
    ordered_ra_ranks = ra_rank[posterior_order]
    ordered_labels = [grave_ids[idx] for idx in posterior_order]

    fig_width = max(12, min(28, len(grave_ids) * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    positions = np.arange(1, len(grave_ids) + 1)
    ax.boxplot(
        ordered_ranks,
        positions=positions,
        widths=0.65,
        showfliers=False,
        manage_ticks=False,
    )
    ax.scatter(
        positions,
        ordered_ra_ranks,
        marker="x",
        s=35,
        label="RA rank",
        zorder=3,
    )

    ax.set_xlabel("Graves ordered by PyMC posterior mean position")
    ax.set_ylabel("Posterior rank position")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()

    if len(grave_ids) <= max_labels:
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    plt.tight_layout()
    plt.show()


def pairwise_uncertainty_summary(
    pairwise_probs: np.ndarray,
    grave_ids: list[str],
    order: np.ndarray,
    n_pairs: int = 20,
) -> pd.DataFrame:
    """List the most uncertain adjacent pairs in a proposed order."""

    rows = []
    for left, right in zip(order[:-1], order[1:]):
        p_left_before_right = pairwise_probs[left, right]
        uncertainty = abs(p_left_before_right - 0.5)
        rows.append(
            {
                "left_grave": grave_ids[left],
                "right_grave": grave_ids[right],
                "p_left_before_right": p_left_before_right,
                "uncertainty_distance_from_0_5": uncertainty,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("uncertainty_distance_from_0_5", ascending=True)
        .head(n_pairs)
        .reset_index(drop=True)
    )


def compare_pymc_to_ra_scores(
    posterior_t_mean: np.ndarray,
    ra_row_scores: np.ndarray,
    grave_ids: list[str],
) -> pd.DataFrame:
    """Compare continuous PyMC posterior positions with RA/CA row scores."""

    posterior_t_mean = np.asarray(posterior_t_mean, dtype=float)
    ra_row_scores = np.asarray(ra_row_scores, dtype=float)

    ra_oriented, ra_flipped, pearson_abs = orient_scores_to_reference(
        ra_row_scores,
        posterior_t_mean,
    )

    pymc_z = (posterior_t_mean - posterior_t_mean.mean()) / posterior_t_mean.std()
    ra_z = (ra_oriented - ra_oriented.mean()) / ra_oriented.std()

    pymc_rank = np.argsort(np.argsort(pymc_z)) + 1
    ra_rank = np.argsort(np.argsort(ra_z)) + 1

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
            "ra_flipped_to_pymc": ra_flipped,
        }
    )

    out.attrs["pearson_abs"] = pearson_abs
    out.attrs["spearman_abs"] = abs(float(spearmanr(pymc_rank, ra_rank).statistic))
    out.attrs["ra_flipped_to_pymc"] = ra_flipped

    return out.sort_values("pymc_t_mean").reset_index(drop=True)


def plot_pymc_vs_ra_scores(comparison: pd.DataFrame, title: str) -> None:
    """Plot continuous PyMC posterior positions against oriented RA scores."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(comparison["ra_score_z"], comparison["pymc_t_z"])
    ax.set_xlabel("RA/CA row score, oriented and standardised")
    ax.set_ylabel("PyMC posterior t, standardised")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def summarise_graves(
    idata: az.InferenceData,
    grave_ids: list[str],
    true_t: Optional[np.ndarray] = None,
    ra_row_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Summarise posterior grave coordinates and ranks after chain alignment."""

    if true_t is not None:
        ref = (true_t - np.mean(true_t)) / np.std(true_t)
        t_mean, t_sd, chain_flips = oriented_posterior_t_summary(idata, ref)
        pymc_orientation_target = "true_t"
        ra_orientation_reference = ref
        ra_orientation_target = "true_t"
    elif ra_row_scores is not None:
        ref = np.asarray(ra_row_scores, dtype=float)
        t_mean, t_sd, chain_flips = oriented_posterior_t_summary(idata, ref)
        pymc_orientation_target = "ra"
        ra_orientation_reference = t_mean
        ra_orientation_target = "pymc_t"
    else:
        ref = None
        t_mean = posterior_mean(idata, "t")
        t_sd = idata.posterior["t"].std(dim=("chain", "draw")).values
        chain_flips = []
        pymc_orientation_target = "raw"
        ra_orientation_reference = t_mean
        ra_orientation_target = "pymc_t"

    posterior_rank = np.argsort(np.argsort(t_mean)) + 1

    out = pd.DataFrame(
        {
            "grave_id": grave_ids,
            "posterior_t_mean": t_mean,
            "posterior_t_sd": t_sd,
            "posterior_rank": posterior_rank,
            "pymc_orientation_target": pymc_orientation_target,
            "pymc_chain_flips": ",".join(str(x) for x in chain_flips),
        }
    )
    out.attrs["pymc_chain_flips"] = chain_flips
    out.attrs["pymc_orientation_target"] = pymc_orientation_target

    if true_t is not None:
        true_rank = np.argsort(np.argsort(ref)) + 1
        out["true_t_scaled"] = ref
        out["true_rank"] = true_rank
        out["rank_difference_pymc_minus_true"] = posterior_rank - true_rank

    if ra_row_scores is not None:
        ra_oriented, ra_flipped, ra_pearson = orient_scores_to_reference(
            ra_row_scores,
            ra_orientation_reference,
        )
        ra_rank = np.argsort(np.argsort(ra_oriented)) + 1

        out["ra_score_oriented"] = ra_oriented
        out["ra_rank"] = ra_rank
        out["rank_difference_pymc_minus_ra"] = posterior_rank - ra_rank
        out["ra_flipped"] = ra_flipped
        out["ra_orientation_target"] = ra_orientation_target
        out["ra_orientation_pearson_abs"] = ra_pearson

        if true_t is not None:
            out["rank_difference_ra_minus_true"] = ra_rank - out["true_rank"]

    return out.sort_values("posterior_t_mean").reset_index(drop=True)


def summarise_types(
    idata: az.InferenceData,
    type_ids: list[str],
    chain_flips: list[bool],
    ra_col_scores: Optional[np.ndarray] = None,
    true_mu: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Summarise posterior type-career parameters after matching chain flips."""

    mu, mu_sd = oriented_posterior_mu_summary(idata, chain_flips)
    sigma = posterior_mean(idata, "sigma")
    a = posterior_mean(idata, "a")
    posterior_type_rank = np.argsort(np.argsort(mu)) + 1

    out = pd.DataFrame(
        {
            "type_id": type_ids,
            "posterior_mu_mean": mu,
            "posterior_mu_sd": mu_sd,
            "posterior_sigma_mean": sigma,
            "posterior_a_mean": a,
            "posterior_type_rank": posterior_type_rank,
        }
    )

    if true_mu is not None:
        true_mu_scaled = (true_mu - np.mean(true_mu)) / np.std(true_mu)
        true_type_rank = np.argsort(np.argsort(true_mu_scaled)) + 1
        out["true_mu_scaled"] = true_mu_scaled
        out["true_type_rank"] = true_type_rank
        out["type_rank_difference_pymc_minus_true"] = posterior_type_rank - true_type_rank
        ra_type_orientation_reference = true_mu_scaled
        ra_type_orientation_target = "true_mu"
    else:
        ra_type_orientation_reference = mu
        ra_type_orientation_target = "pymc_mu"

    if ra_col_scores is not None:
        ra_oriented, ra_flipped, ra_pearson = orient_scores_to_reference(
            ra_col_scores,
            ra_type_orientation_reference,
        )
        ra_type_rank = np.argsort(np.argsort(ra_oriented)) + 1
        out["ra_type_score_oriented"] = ra_oriented
        out["ra_type_rank"] = ra_type_rank
        out["type_rank_difference_pymc_minus_ra"] = posterior_type_rank - ra_type_rank
        out["ra_type_flipped"] = ra_flipped
        out["ra_type_orientation_target"] = ra_type_orientation_target
        out["ra_type_orientation_pearson_abs"] = ra_pearson

        if true_mu is not None:
            out["type_rank_difference_ra_minus_true"] = ra_type_rank - out["true_type_rank"]

    return out.sort_values("posterior_mu_mean").reset_index(drop=True)


def order_from_summary(summary: pd.DataFrame, grave_ids: list[str]) -> np.ndarray:
    """Return row indices ordered by posterior chronology."""

    ordered_ids = summary.sort_values("posterior_t_mean")["grave_id"].tolist()
    return np.array([grave_ids.index(g) for g in ordered_ids], dtype=int)


def plot_matrix(
    Y: np.ndarray,
    order: np.ndarray,
    title: str,
    type_order: Optional[np.ndarray] = None,
) -> None:
    """Plot a presence/absence matrix ordered by graves and types."""

    Y_ord = Y[order, :]

    if type_order is None:
        weights = np.arange(Y_ord.shape[0])[:, None] + 1
        denom = np.maximum(Y_ord.sum(axis=0), 1)
        type_scores = (Y_ord * weights).sum(axis=0) / denom
        type_order = np.argsort(type_scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(Y_ord[:, type_order], aspect="auto", interpolation="nearest")
    ax.set_xlabel("Types")
    ax.set_ylabel("Graves")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_true_vs_estimated(summary: pd.DataFrame) -> None:
    """Plot true against estimated grave positions for synthetic data."""

    if "true_t_scaled" not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(summary["true_t_scaled"], summary["posterior_t_mean"])
    ax.set_xlabel("True latent chronology, scaled")
    ax.set_ylabel("Posterior mean t")
    ax.set_title("Pure PyMC seriation: true vs estimated grave positions")
    plt.tight_layout()
    plt.show()


def print_matrix_diagnostics(Y: np.ndarray, label: str = "Matrix") -> None:
    """Print basic information about the matrix."""

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
        "Grave richness, min/median/max:",
        int(grave_counts.min()),
        float(np.median(grave_counts)),
        int(grave_counts.max()),
    )


def print_sampling_diagnostics(idata: az.InferenceData, include_richness: bool = True) -> None:
    """Print compact sampler diagnostics."""

    div = int(idata.sample_stats["diverging"].sum().values)
    print("Divergences:", div)

    vars_to_show = ["intercept"]
    if include_richness and "sigma_g" in idata.posterior:
        vars_to_show.append("sigma_g")

    print(az.summary(idata, var_names=vars_to_show))

    full_summary = az.summary(idata, var_names=["t", "mu", "sigma", "a"])
    print("\nWorst raw R-hat among t/mu/sigma/a:", round(float(full_summary["r_hat"].max()), 3))
    print("Lowest raw bulk ESS among t/mu/sigma/a:", round(float(full_summary["ess_bulk"].min()), 1))


def evaluate_recovery(summary: pd.DataFrame) -> float:
    """Compute Spearman recovery of posterior ranks against true ranks."""

    if "true_rank" not in summary.columns:
        return float("nan")

    return float(spearmanr(summary["true_rank"], summary["posterior_rank"]).statistic)


def run_pymc_and_report(
    Y: np.ndarray,
    grave_ids: list[str],
    type_ids: list[str],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int,
    include_richness: bool,
    repulsion_strength: float,
    true_t: Optional[np.ndarray] = None,
    true_order: Optional[np.ndarray] = None,
    true_mu: Optional[np.ndarray] = None,
    title_prefix: str = "Dataset",
) -> None:
    """Fit PyMC model and print/plot comparison outputs."""

    print_matrix_diagnostics(Y, label=title_prefix)

    ra_order, ra_type_order, row_scores, col_scores = reciprocal_averaging_order(Y)
    print("\nClassical comparison: reciprocal averaging / CA")

    if true_order is not None:
        ra_oriented = orient_order_to_reference(ra_order, true_order)
        ra_rho = order_correlation(ra_oriented, true_order, orientation_free=False)
        print("RA Spearman recovery:", round(ra_rho, 3))
    else:
        ra_oriented = ra_order
        print("RA order computed. No true order available for recovery score.")

    plot_matrix(Y, np.arange(Y.shape[0]), f"{title_prefix}: input order")
    if true_order is not None:
        plot_matrix(Y, true_order, f"{title_prefix}: true order")
    plot_matrix(Y, ra_oriented, f"{title_prefix}: reciprocal averaging order", type_order=ra_type_order)

    idata = fit_parametric_pymc_seriation(
        Y,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        include_richness=include_richness,
        repulsion_strength=repulsion_strength,
    )

    print_sampling_diagnostics(idata, include_richness=include_richness)

    print("\nChain-wise PyMC order diagnostics:")
    chain_diag = chain_order_diagnostics(
        idata,
        ra_row_scores=row_scores,
        true_t=true_t,
    )
    print(chain_diag.to_string(index=False))

    pairwise_reference = true_t if true_t is not None else row_scores
    pairwise_probs = posterior_pairwise_order_probabilities(idata, pairwise_reference)

    summary = summarise_graves(idata, grave_ids, true_t=true_t, ra_row_scores=row_scores)
    pymc_order = order_from_summary(summary, grave_ids)

    if true_order is not None:
        rho = evaluate_recovery(summary)
        print("\nPyMC Spearman recovery:", round(rho, 3))
        print("PyMC vs RA order correlation:", round(order_correlation(pymc_order, ra_oriented), 3))
    else:
        print("\nPyMC vs RA order correlation, orientation-free:", round(order_correlation(pymc_order, ra_oriented), 3))

    score_comparison = compare_pymc_to_ra_scores(
        posterior_t_mean=summary.set_index("grave_id").loc[grave_ids, "posterior_t_mean"].to_numpy(),
        ra_row_scores=row_scores,
        grave_ids=grave_ids,
    )

    print("\nPyMC posterior t vs RA/CA row scores:")
    print("RA flipped to PyMC axis:", score_comparison.attrs["ra_flipped_to_pymc"])
    print("Pearson correlation, abs/oriented:", round(score_comparison.attrs["pearson_abs"], 3))
    print("Spearman rank correlation, abs/oriented:", round(score_comparison.attrs["spearman_abs"], 3))
    print("\nLargest PyMC–RA rank differences:")
    print(
        score_comparison
        .assign(abs_rank_difference=lambda d: d["rank_difference_pymc_minus_ra"].abs())
        .sort_values("abs_rank_difference", ascending=False)
        .head(15)
        [["grave_id", "pymc_rank", "ra_rank", "rank_difference_pymc_minus_ra", "pymc_t_z", "ra_score_z"]]
        .to_string(index=False)
    )

    print("\nMost uncertain adjacent pairs in PyMC posterior order:")
    uncertain_pairs = pairwise_uncertainty_summary(
        pairwise_probs,
        grave_ids=grave_ids,
        order=pymc_order,
        n_pairs=20,
    )
    print(uncertain_pairs.to_string(index=False))

    print("\nPosterior grave order summary with RA comparison:")
    grave_cols = [
        "grave_id",
        "posterior_t_mean",
        "posterior_t_sd",
        "posterior_rank",
        "ra_score_oriented",
        "ra_rank",
        "rank_difference_pymc_minus_ra",
        "ra_flipped",
        "ra_orientation_target",
        "pymc_orientation_target",
        "pymc_chain_flips",
    ]
    if "true_rank" in summary.columns:
        grave_cols.insert(4, "true_rank")
        grave_cols.insert(5, "rank_difference_pymc_minus_true")
        grave_cols.append("rank_difference_ra_minus_true")
    print(summary[grave_cols].head(25).to_string(index=False))

    chain_flips = summary.attrs.get("pymc_chain_flips", [])
    type_summary = summarise_types(
        idata,
        type_ids,
        chain_flips=chain_flips,
        ra_col_scores=col_scores,
        true_mu=true_mu,
    )
    print("\nPosterior type summary with RA comparison:")
    type_cols = [
        "type_id",
        "posterior_mu_mean",
        "posterior_mu_sd",
        "posterior_sigma_mean",
        "posterior_a_mean",
        "posterior_type_rank",
        "ra_type_score_oriented",
        "ra_type_rank",
        "type_rank_difference_pymc_minus_ra",
        "ra_type_flipped",
        "ra_type_orientation_target",
    ]
    if "true_type_rank" in type_summary.columns:
        type_cols.insert(6, "true_type_rank")
        type_cols.insert(7, "type_rank_difference_pymc_minus_true")
        type_cols.append("type_rank_difference_ra_minus_true")
    print(type_summary[type_cols].head(25).to_string(index=False))

    plot_matrix(Y, pymc_order, f"{title_prefix}: PyMC posterior order")
    plot_pymc_vs_ra_scores(score_comparison, f"{title_prefix}: PyMC posterior axis vs RA/CA axis")
    plot_posterior_rank_distributions(
        idata,
        grave_ids=grave_ids,
        reference=pairwise_reference,
        ra_row_scores=row_scores,
        posterior_order=pymc_order,
        title=f"{title_prefix}: posterior rank distributions with RA ranks",
    )
    plot_true_vs_estimated(summary)


def run_simulation(args: argparse.Namespace) -> None:
    """Run validated moderate simulation and fit the PyMC model."""

    data = simulate_valid_moderate_data(
        n_graves=args.n_graves,
        n_types=args.n_types,
        seed=args.seed,
        shuffle=True,
        min_type_count=args.min_type_count,
        min_grave_count=args.min_grave_count,
        max_attempts=args.max_attempts,
    )

    print("Mode: simulation")
    print("Simulation seed used:", data.seed_used)
    print("Simulation attempt:", data.simulation_attempt)

    run_pymc_and_report(
        data.Y,
        data.grave_ids,
        data.type_ids,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        include_richness=args.include_richness,
        repulsion_strength=args.repulsion_strength,
        true_t=data.true_t_observed,
        true_order=data.true_order,
        true_mu=data.true_mu,
        title_prefix="Simulation",
    )


def run_munsingen(args: argparse.Namespace) -> None:
    """Load Münsingen CSV and fit the PyMC model."""

    data = load_presence_absence_csv(
        args.csv,
        id_col=args.id_col,
        min_type_count=args.min_type_count,
        min_grave_count=args.min_grave_count,
        filter_matrix=not args.no_filter,
    )

    print("Mode: Münsingen / CSV")
    print("Source:", data.source)
    if args.no_filter:
        print("Filtering: disabled")
    else:
        print(
            "Filtering: enabled "
            f"(min_type_count={args.min_type_count}, min_grave_count={args.min_grave_count})"
        )

    run_pymc_and_report(
        data.Y,
        data.grave_ids,
        data.type_ids,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        include_richness=args.include_richness,
        repulsion_strength=args.repulsion_strength,
        true_t=None,
        true_order=None,
        true_mu=None,
        title_prefix="Münsingen",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        description="Pure PyMC seriation prototype with simulation, Münsingen import, and RA comparison."
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--draws", type=int, default=800, help="Posterior draws per chain.")
        p.add_argument("--tune", type=int, default=1200, help="Tuning draws per chain.")
        p.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
        p.add_argument("--target-accept", type=float, default=0.96, help="NUTS target_accept.")
        p.add_argument("--random-seed", type=int, default=123, help="PyMC random seed.")
        p.add_argument("--repulsion-strength", type=float, default=0.35, help="Strength of spacing repulsion prior.")
        p.add_argument(
            "--include-richness",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include grave-specific richness effect g_i.",
        )
        p.add_argument("--min-type-count", type=int, default=2, help="Minimum number of occurrences per type.")
        p.add_argument("--min-grave-count", type=int, default=2, help="Minimum number of types per grave.")

    p_sim = sub.add_parser("simulate", help="Run validated moderate simulation.")
    add_common_options(p_sim)
    p_sim.add_argument("--n-graves", type=int, default=50, help="Number of simulated graves.")
    p_sim.add_argument("--n-types", type=int, default=22, help="Number of simulated types.")
    p_sim.add_argument("--seed", type=int, default=42, help="Simulation seed.")
    p_sim.add_argument("--max-attempts", type=int, default=500, help="Maximum simulation attempts.")
    p_sim.set_defaults(func=run_simulation)

    p_mun = sub.add_parser("munsingen", help="Run on Münsingen or compatible CSV matrix.")
    add_common_options(p_mun)
    p_mun.add_argument("--csv", type=str, default="munsingen.csv", help="Path to CSV matrix.")
    p_mun.add_argument("--id-col", type=str, default=None, help="Optional grave ID column name.")
    p_mun.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable iterative filtering of rare types and poorly furnished graves.",
    )
    p_mun.set_defaults(func=run_munsingen)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
