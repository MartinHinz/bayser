from __future__ import annotations

import numpy as np
import pandas as pd


def load_intcal20_curve(path: str) -> pd.DataFrame:
    curve = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=[
            "cal_bp",
            "c14_age",
            "c14_sigma",
            "delta14c",
            "delta14c_sigma",
        ],
    )

    curve = (
        curve[["cal_bp", "c14_age", "c14_sigma"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("cal_bp")
        .reset_index(drop=True)
    )

    if curve.shape[0] < 10:
        raise ValueError(f"Calibration curve in {path} looks too small.")

    if np.any(np.diff(curve["cal_bp"].to_numpy()) <= 0):
        raise ValueError("Calibration curve cal_bp values must be strictly increasing.")

    return curve


def calibrate_single_c14_date(
    bp: float,
    error: float,
    curve: pd.DataFrame,
) -> dict[str, float]:
    cal_bp = curve["cal_bp"].to_numpy(dtype=float)
    c14_age = curve["c14_age"].to_numpy(dtype=float)
    c14_sigma = curve["c14_sigma"].to_numpy(dtype=float)

    sigma = np.sqrt(float(error) ** 2 + c14_sigma**2)

    log_lik = -0.5 * ((float(bp) - c14_age) / sigma) ** 2 - np.log(sigma)
    log_lik -= np.max(log_lik)

    density = np.exp(log_lik)
    mass = density * np.gradient(cal_bp)
    mass /= mass.sum()

    cdf = np.cumsum(mass)

    mean = float(np.sum(cal_bp * mass))
    sd = float(np.sqrt(np.sum((cal_bp - mean) ** 2 * mass)))

    return {
        "unmodelled_cal_bp_mean": mean,
        "unmodelled_cal_bp_sd": sd,
        "unmodelled_cal_bp_hdi_3": float(np.interp(0.03, cdf, cal_bp)),
        "unmodelled_cal_bp_median": float(np.interp(0.50, cdf, cal_bp)),
        "unmodelled_cal_bp_hdi_97": float(np.interp(0.97, cdf, cal_bp)),
    }


def calibrate_c14_dates_unmodelled(
    grave_ids: list[str],
    c14_bp: np.ndarray,
    c14_error: np.ndarray,
    curve: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for grave_id, bp, error in zip(grave_ids, c14_bp, c14_error):
        row = {
            "grave_id": grave_id,
            "c14_bp": float(bp),
            "c14_error": float(error),
        }
        row.update(calibrate_single_c14_date(bp, error, curve))
        rows.append(row)

    return pd.DataFrame(rows)
