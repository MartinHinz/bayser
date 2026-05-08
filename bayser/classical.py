from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def reciprocal_averaging_order(
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CA/SVD-style reciprocal averaging order and scores."""

    X = np.asarray(Y, dtype=float)

    if X.sum() <= 0:
        raise ValueError("Cannot run reciprocal averaging on an empty matrix.")

    P = X / X.sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    if np.any(r <= 0) or np.any(c <= 0):
        raise ValueError("RA/CA requires no empty rows or columns.")

    expected = np.outer(r, c)
    S = (P - expected) / np.sqrt(expected)

    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)

    row_scores = U[:, 0] / np.sqrt(r) * singular_values[0]
    col_scores = Vt[0, :] / np.sqrt(c) * singular_values[0]

    return (
        np.argsort(row_scores),
        np.argsort(col_scores),
        row_scores,
        col_scores,
    )


def _normalise_score(x: np.ndarray, offset: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if np.isclose(xmax, xmin):
        return np.zeros_like(x)

    return (x - xmin + offset) / (xmax - xmin + offset)


def iterative_reciprocal_averaging_order(
    Y: np.ndarray,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return classical iterative reciprocal averaging order and scores."""

    X = np.asarray(Y, dtype=float).copy()

    if X.sum() <= 0:
        raise ValueError("Cannot run iterative RA on an empty matrix.")

    if np.any(X.sum(axis=1) <= 0) or np.any(X.sum(axis=0) <= 0):
        raise ValueError("Iterative RA requires no empty rows or columns.")

    n_rows, n_cols = X.shape
    row_order = np.arange(n_rows)
    col_order = np.arange(n_cols)

    previous = None
    previous_previous = None

    for _ in range(max_iter):
        previous_previous = previous
        previous = X.copy()

        col_positions = np.arange(1, n_cols + 1, dtype=float)
        row_scores_ordered = _normalise_score((X @ col_positions) / X.sum(axis=1))

        row_sort = np.argsort(row_scores_ordered, kind="stable")
        X = X[row_sort, :]
        row_order = row_order[row_sort]

        row_positions = np.arange(1, n_rows + 1, dtype=float)
        col_scores_ordered = _normalise_score((row_positions @ X) / X.sum(axis=0))

        col_sort = np.argsort(col_scores_ordered, kind="stable")
        X = X[:, col_sort]
        col_order = col_order[col_sort]

        if np.array_equal(X, previous):
            break

        if previous_previous is not None and np.array_equal(X, previous_previous):
            print("Warning: iterative RA caught in a two-step loop; accepting current order.")
            break
    else:
        print("Warning: iterative RA did not converge within max_iter.")

    col_positions = np.arange(1, n_cols + 1, dtype=float)
    row_scores_ordered = _normalise_score((X @ col_positions) / X.sum(axis=1))

    row_scores = np.empty(n_rows, dtype=float)
    row_scores[row_order] = row_scores_ordered

    row_positions = np.arange(1, n_rows + 1, dtype=float)
    col_scores_ordered = _normalise_score((row_positions @ X) / X.sum(axis=0))

    col_scores = np.empty(n_cols, dtype=float)
    col_scores[col_order] = col_scores_ordered

    return row_order, col_order, row_scores, col_scores


def classical_order(
    Y: np.ndarray,
    method: str = "ca",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == "ca":
        return reciprocal_averaging_order(Y)

    if method == "iterative":
        return iterative_reciprocal_averaging_order(Y)

    raise ValueError(f"Unknown RA method: {method}")


def _rank_from_order(order: np.ndarray) -> np.ndarray:
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank


def order_correlation(
    order_a: np.ndarray,
    order_b: np.ndarray,
    orientation_free: bool = True,
) -> float:
    rank_a = _rank_from_order(order_a)
    rank_b = _rank_from_order(order_b)

    rho = float(spearmanr(rank_a, rank_b).statistic)

    if not np.isfinite(rho):
        return float("nan")

    return abs(rho) if orientation_free else rho