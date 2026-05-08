from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import arviz as az
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _is_quiet(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "quiet", False)) if args is not None else False


def _print(args: argparse.Namespace | None, *values, **kwargs) -> None:
    if not _is_quiet(args):
        print(*values, **kwargs)


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_stem(name: str) -> str:
    """Convert a logical table name into a safe filename stem."""
    stem = str(name).strip().replace(" ", "_").replace("-", "_")

    if stem.endswith(".csv"):
        stem = stem[:-4]

    stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"_", "."})
    return stem or "result"


def _clean_for_json(value: Any) -> Any:
    """Convert common NumPy/Pandas/Path objects to JSON-safe values."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _clean_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_clean_for_json(v) for v in value]

    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _write_csv_if_table(
    table: pd.DataFrame | Any | None,
    path: Path,
    *,
    index: bool = False,
) -> bool:
    if table is None:
        return False

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)

    if table.empty:
        return False

    table.to_csv(path, index=index)
    return True


def _idata_attrs(idata: az.InferenceData | None) -> dict[str, Any]:
    if idata is None or not getattr(idata, "attrs", None):
        return {}

    return {str(k): _clean_for_json(v) for k, v in idata.attrs.items()}


def _sampling_metadata(idata: az.InferenceData | None) -> dict[str, Any]:
    if idata is None or not hasattr(idata, "sample_stats"):
        return {}

    out: dict[str, Any] = {}

    if "diverging" in idata.sample_stats:
        div = idata.sample_stats["diverging"]
        out["divergences_total"] = int(div.sum().values)

        if "chain" in div.dims:
            out["divergences_by_chain"] = [
                int(x) for x in div.sum(dim="draw").values.tolist()
            ]

    if "tree_depth" in idata.sample_stats:
        td = idata.sample_stats["tree_depth"].values
        out["tree_depth_max"] = int(np.nanmax(td))
        out["tree_depth_median"] = float(np.nanmedian(td))

    if "n_steps" in idata.sample_stats:
        ns = idata.sample_stats["n_steps"].values
        out["n_steps_median"] = float(np.nanmedian(ns))
        out["n_steps_max"] = int(np.nanmax(ns))

    if "step_size_bar" in idata.sample_stats:
        ss = idata.sample_stats["step_size_bar"].values
        out["step_size_bar_median"] = float(np.nanmedian(ss))

    return _clean_for_json(out)


def _args_metadata(args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return {}

    return {
        key: _clean_for_json(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _metadata_to_frame(metadata: Mapping[str, Any] | None) -> pd.DataFrame:
    """Flat CSV-friendly metadata representation."""

    if not metadata:
        return pd.DataFrame(columns=["setting", "value"])

    rows = []

    def flatten(prefix: str, value: Any) -> None:
        value = _clean_for_json(value)

        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                flatten(key, v)
        else:
            rows.append(
                {
                    "setting": prefix,
                    "value": json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple))
                    else value,
                }
            )

    for key, value in metadata.items():
        flatten(str(key), value)

    return pd.DataFrame(rows)


def _write_json(payload: dict[str, Any], path: Path) -> bool:
    path.write_text(
        json.dumps(_clean_for_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True


def _build_metadata(
    *,
    args: argparse.Namespace | None = None,
    idata: az.InferenceData | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit metadata with args/idata-derived metadata."""

    out: dict[str, Any] = {}

    if metadata:
        out.update(_clean_for_json(dict(metadata)))

    args_meta = _args_metadata(args)
    if args_meta:
        out.setdefault("args", args_meta)

    model_attrs = _idata_attrs(idata)
    if model_attrs:
        out.setdefault("model_attrs", model_attrs)

    sampling = _sampling_metadata(idata)
    if sampling:
        out.setdefault("sampling", sampling)

    if extra_metadata:
        out.setdefault("extra", _clean_for_json(dict(extra_metadata)))

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def write_results(
    *,
    results_dir: str | Path,
    tables: Mapping[str, pd.DataFrame | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    args: argparse.Namespace | None = None,
    idata: az.InferenceData | None = None,
    grave_summary: pd.DataFrame | None = None,
    type_summary: pd.DataFrame | None = None,
    rank_comparison: pd.DataFrame | None = None,
    adjacent_uncertainty: pd.DataFrame | None = None,
    chain_diagnostics: pd.DataFrame | None = None,
    outlier_screen: pd.DataFrame | None = None,
    active_outliers: pd.DataFrame | None = None,
    extra_tables: Mapping[str, pd.DataFrame | None] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write tabular and metadata results for one seriation run.

    Preferred generic API
    ---------------------
    write_results(
        results_dir=...,
        tables={"grave_summary": grave_summary, ...},
        metadata={...},
    )

    Backwards-compatible explicit API
    ---------------------------------
    write_results(
        results_dir=...,
        args=args,
        idata=idata,
        grave_summary=grave_summary,
        type_summary=type_summary,
        ...
    )

    Empty or missing tables are skipped. Metadata is always written as both
    ``run_metadata.json`` and ``metadata.csv``.
    """

    out_dir = _ensure_dir(results_dir)
    written: list[Path] = []

    all_tables: dict[str, pd.DataFrame | None] = {}

    # New generic table API.
    if tables:
        for name, table in tables.items():
            all_tables[_safe_stem(name)] = table

    # Backwards-compatible explicit table API.
    explicit_tables: dict[str, pd.DataFrame | None] = {
        "grave_summary": grave_summary,
        "type_summary": type_summary,
        "rank_comparison": rank_comparison,
        "adjacent_uncertainty": adjacent_uncertainty,
        "chain_diagnostics": chain_diagnostics,
        "outlier_screen": outlier_screen,
        "active_outliers": active_outliers,
    }

    for name, table in explicit_tables.items():
        if table is not None and name not in all_tables:
            all_tables[name] = table

    if extra_tables:
        for name, table in extra_tables.items():
            all_tables[_safe_stem(name)] = table

    for stem, table in all_tables.items():
        path = out_dir / f"{_safe_stem(stem)}.csv"
        if _write_csv_if_table(table, path):
            written.append(path)

    merged_metadata = _build_metadata(
        args=args,
        idata=idata,
        metadata=metadata,
        extra_metadata=extra_metadata,
    )

    metadata_json_path = out_dir / "run_metadata.json"
    _write_json(merged_metadata, metadata_json_path)
    written.append(metadata_json_path)

    metadata_csv_path = out_dir / "metadata.csv"
    metadata_frame = _metadata_to_frame(merged_metadata)
    metadata_frame.to_csv(metadata_csv_path, index=False)
    written.append(metadata_csv_path)

    _print(args, f"\nSaved {len(written)} result files to: {out_dir}")

    return written