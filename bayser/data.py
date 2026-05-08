from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SeriationData:
    Y: np.ndarray
    grave_ids: list[str]
    type_ids: list[str]
    source: str
    c14_bp: np.ndarray | None = None
    c14_error: np.ndarray | None = None
    original_counts: np.ndarray | None = None
    removed_grave_ids: list[str] = field(default_factory=list)
    removed_type_ids: list[str] = field(default_factory=list)
    unmatched_c14_ids: list[str] = field(default_factory=list)
    id_col: str | None = None
    bp_col: str | None = None
    error_col: str | None = None


def normalise_column_names(columns: list[str]) -> list[str]:
    return [
        str(c).strip().replace("+/-", "±").replace("+-", "±")
        for c in columns
    ]


def read_id_indexed_csv(
    path: str,
    sep: str = ",",
    id_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, sep=sep)
    df.columns = normalise_column_names(list(df.columns))

    id_col = id_col or df.columns[0]

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in {path}.")

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=True)
    df.index = df.index.astype(str)

    return df, id_col


def infer_c14_date_columns(
    df: pd.DataFrame,
    bp_col: str | None = None,
    error_col: str | None = None,
) -> tuple[str, str]:
    columns = list(df.columns)

    bp_candidates = [
        "BP",
        "bp",
        "C14",
        "c14",
        "C14_BP",
        "c14_bp",
        "Radiocarbon",
        "radiocarbon",
    ]

    error_candidates = [
        "STD",
        "std",
        "Std",
        "±",
        "sd",
        "SD",
        "sigma",
        "Sigma",
        "error",
        "Error",
        "err",
        "Err",
    ]

    bp_col = bp_col or next((c for c in bp_candidates if c in columns), None)
    error_col = error_col or next((c for c in error_candidates if c in columns), None)

    if bp_col is None or bp_col not in columns:
        raise ValueError("Could not infer BP column. Use --bp-col.")

    if error_col is None or error_col not in columns:
        raise ValueError("Could not infer C14 error column. Use --error-col.")

    return bp_col, error_col


def filter_informative_matrix(
    Y: np.ndarray,
    grave_ids: list[str],
    type_ids: list[str],
    min_type_count: int = 2,
    min_grave_count: int = 2,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray]:
    Y_f = Y.copy()
    grave_ids_f = list(grave_ids)
    type_ids_f = list(type_ids)

    grave_keep = np.ones(Y.shape[0], dtype=bool)
    type_keep = np.ones(Y.shape[1], dtype=bool)

    while True:
        changed = False

        keep_types_now = Y_f.sum(axis=0) >= min_type_count
        if not np.all(keep_types_now):
            current = np.where(type_keep)[0]
            type_keep[current[~keep_types_now]] = False
            Y_f = Y_f[:, keep_types_now]
            type_ids_f = [t for t, keep in zip(type_ids_f, keep_types_now) if keep]
            changed = True

        keep_graves_now = Y_f.sum(axis=1) >= min_grave_count
        if not np.all(keep_graves_now):
            current = np.where(grave_keep)[0]
            grave_keep[current[~keep_graves_now]] = False
            Y_f = Y_f[keep_graves_now, :]
            grave_ids_f = [g for g, keep in zip(grave_ids_f, keep_graves_now) if keep]
            changed = True

        if not changed:
            break

    if Y_f.shape[0] < 3 or Y_f.shape[1] < 3:
        raise ValueError("Filtered matrix is too small for seriation.")

    return Y_f, grave_ids_f, type_ids_f, grave_keep, type_keep


def _feature_columns(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    exclude_regex: str | None = None,
) -> list[str]:
    excluded = {c.strip() for c in (exclude_cols or [])}
    cols = [c for c in df.columns if c not in excluded]

    if exclude_regex:
        pattern = re.compile(exclude_regex)
        cols = [c for c in cols if not pattern.search(c)]

    if not cols:
        raise ValueError("No feature columns remain after exclusions.")

    return cols


def _load_c14_aligned(
    c14_path: str,
    grave_ids: list[str],
    c14_sep: str = ",",
    c14_id_col: str | None = None,
    bp_col: str | None = None,
    error_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str, str, list[str]]:
    c14, c14_id_name = read_id_indexed_csv(
        c14_path,
        sep=c14_sep,
        id_col=c14_id_col,
    )

    bp_col, error_col = infer_c14_date_columns(
        c14,
        bp_col=bp_col,
        error_col=error_col,
    )

    c14_aligned = c14.reindex(grave_ids)

    c14_bp = pd.to_numeric(c14_aligned[bp_col], errors="coerce").to_numpy(dtype=float)
    c14_error = pd.to_numeric(c14_aligned[error_col], errors="coerce").to_numpy(dtype=float)

    finite_bp = np.isfinite(c14_bp)
    finite_error = np.isfinite(c14_error)

    if np.any(finite_bp != finite_error):
        raise ValueError("C14 BP and error columns must be missing in the same rows.")

    if np.any(c14_error[finite_error] <= 0):
        raise ValueError("Finite C14 error values must be positive.")

    grave_id_set = set(grave_ids)
    unmatched = [idx for idx in c14.index.astype(str) if idx not in grave_id_set]

    return c14_bp, c14_error, c14_id_name, bp_col, error_col, unmatched


def load_seriation_input(
    feature_path: str,
    c14_path: str | None = None,
    feature_sep: str = ",",
    c14_sep: str = ",",
    feature_id_col: str | None = None,
    c14_id_col: str | None = None,
    bp_col: str | None = None,
    error_col: str | None = None,
    min_type_count: int = 2,
    min_grave_count: int = 2,
    filter_matrix: bool = True,
    exclude_cols: list[str] | None = None,
    exclude_regex: str | None = None,
) -> SeriationData:
    features, feature_id_name = read_id_indexed_csv(
        feature_path,
        sep=feature_sep,
        id_col=feature_id_col,
    )

    feature_cols = _feature_columns(
        features,
        exclude_cols=exclude_cols,
        exclude_regex=exclude_regex,
    )

    feature_numeric = (
        features[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    counts_all = feature_numeric.to_numpy(dtype=float)
    Y_all = (counts_all > 0).astype(int)

    grave_ids_all = list(features.index.astype(str))
    type_ids_all = [str(c) for c in feature_cols]

    c14_bp_all = None
    c14_error_all = None
    c14_id_name = None
    unmatched_c14_ids: list[str] = []

    if c14_path is not None:
        (
            c14_bp_all,
            c14_error_all,
            c14_id_name,
            bp_col,
            error_col,
            unmatched_c14_ids,
        ) = _load_c14_aligned(
            c14_path=c14_path,
            grave_ids=grave_ids_all,
            c14_sep=c14_sep,
            c14_id_col=c14_id_col,
            bp_col=bp_col,
            error_col=error_col,
        )

    if filter_matrix:
        Y, grave_ids, type_ids, grave_keep, type_keep = filter_informative_matrix(
            Y_all,
            grave_ids_all,
            type_ids_all,
            min_type_count=min_type_count,
            min_grave_count=min_grave_count,
        )

        original_counts = counts_all[np.ix_(grave_keep, type_keep)]
        removed_grave_ids = [g for g, keep in zip(grave_ids_all, grave_keep) if not keep]
        removed_type_ids = [t for t, keep in zip(type_ids_all, type_keep) if not keep]

        c14_bp = c14_bp_all[grave_keep] if c14_bp_all is not None else None
        c14_error = c14_error_all[grave_keep] if c14_error_all is not None else None
    else:
        Y = Y_all
        grave_ids = grave_ids_all
        type_ids = type_ids_all
        original_counts = counts_all
        removed_grave_ids = []
        removed_type_ids = []
        c14_bp = c14_bp_all
        c14_error = c14_error_all

    if Y.shape[0] < 3 or Y.shape[1] < 3:
        raise ValueError("Prepared matrix is too small for seriation.")

    source = f"features={feature_path}"
    id_col = feature_id_name

    if c14_path is not None:
        source = f"{source}; c14={c14_path}"
        id_col = f"feature:{feature_id_name}; c14:{c14_id_name}"

    return SeriationData(
        Y=Y,
        grave_ids=grave_ids,
        type_ids=type_ids,
        source=source,
        c14_bp=c14_bp,
        c14_error=c14_error,
        original_counts=original_counts,
        removed_grave_ids=removed_grave_ids,
        removed_type_ids=removed_type_ids,
        unmatched_c14_ids=unmatched_c14_ids,
        id_col=id_col,
        bp_col=bp_col,
        error_col=error_col,
    )