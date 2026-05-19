from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulatedData:
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


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_moderate_data(
    n_graves: int = 50,
    n_types: int = 22,
    seed: int = 42,
    shuffle: bool = True,
) -> SimulatedData:
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
        f"Could not simulate a valid matrix after {max_attempts} attempts."
    )


def simulated_feature_matrix(data: SimulatedData) -> pd.DataFrame:
    """Return simulated binary feature matrix in Bayser-compatible format."""

    return pd.DataFrame(
        data.Y,
        index=data.grave_ids,
        columns=data.type_ids,
    ).reset_index(names="grave_id")


def simulated_grave_truth(data: SimulatedData) -> pd.DataFrame:
    """Return true assemblage-level quantities for recovery evaluation."""

    true_rank = np.empty_like(data.true_order)
    true_rank[data.true_order] = np.arange(1, len(data.true_order) + 1)

    return pd.DataFrame(
        {
            "grave_id": data.grave_ids,
            "true_t": data.true_t_observed,
            "true_rank": true_rank,
            "true_richness": data.true_richness_observed,
            "observed_richness": data.Y.sum(axis=1),
            "seed_used": data.seed_used,
            "simulation_attempt": data.simulation_attempt,
        }
    ).sort_values("true_rank")


def simulated_type_truth(data: SimulatedData) -> pd.DataFrame:
    """Return true type-level quantities for recovery evaluation."""

    true_type_rank = np.argsort(np.argsort(data.true_mu)) + 1

    return pd.DataFrame(
        {
            "type_id": data.type_ids,
            "true_mu": data.true_mu,
            "true_sigma": data.true_sigma,
            "true_a": data.true_a,
            "true_type_rank": true_type_rank,
        }
    ).sort_values("true_type_rank")


def write_simulated_dataset(
    data: SimulatedData,
    out_dir: str | Path,
    *,
    feature_filename: str = "features.csv",
    grave_truth_filename: str = "grave_truth.csv",
    type_truth_filename: str = "type_truth.csv",
) -> dict[str, Path]:
    """Write simulated dataset and truth tables to disk."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "features": out / feature_filename,
        "grave_truth": out / grave_truth_filename,
        "type_truth": out / type_truth_filename,
    }

    simulated_feature_matrix(data).to_csv(paths["features"], index=False)
    simulated_grave_truth(data).to_csv(paths["grave_truth"], index=False)
    simulated_type_truth(data).to_csv(paths["type_truth"], index=False)

    return paths
