"""Continuous cross-channel analyses on simultaneously recorded fibers (L vs R NAc DA, lat vs med NAc, DA vs PL NE):
whole-session correlation, multi-timescale sliding-window correlation, circular-shift nulls, cross-correlation lags."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from analysis import config
from analysis.dataio import foraging as F

WINDOWS_S = (1, 2, 5, 10, 20, 30)
MAX_LAG_S = 2.0
N_SHIFT = 100
MIN_SHIFT_S = 60.0


def pair_kind(a: dict, b: dict) -> str | None:
    if a["signal"] == "DA" and b["signal"] == "DA":
        if a["region"] == b["region"] == "latNAcc":
            return "latL-latR"
        if {a["region"], b["region"]} == {"latNAcc", "medNAcc"}:
            return "lat-med"
        return None
    if {a["signal"], b["signal"]} == {"DA", "NE"}:
        return "DA-NE"
    return None


def sliding_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Pearson r in non-overlapping windows of w samples."""
    n = len(x) // w
    X = x[: n * w].reshape(n, w)
    Y = y[: n * w].reshape(n, w)
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    with np.errstate(all="ignore"):
        return (X * Y).sum(axis=1) / den


def xcorr(x, y, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    return lags, np.array([np.mean(x[max(0, -L):n - max(0, L)] * y[max(0, L):n - max(0, -L)]) for L in lags])


def session_worker(args):
    subject_id, ses_idx, d = args
    fip = F.load_fip(d)
    if len(fip) < 2:
        return []
    trials = F.load_trials(d, subject_id, ses_idx)
    t_lo = max(max(r["t"][0] for r in fip.values()), trials["goCue_start_time_in_session"].min())
    t_hi = min(min(r["t"][-1] for r in fip.values()), trials["bonsai_stop_time_in_session"].max())
    grid = np.arange(t_lo, t_hi, config.DT)
    sig = {ch: np.interp(grid, r["t"], r["data_z"]) for ch, r in fip.items()}
    rng = np.random.default_rng(len(grid))
    rows = []
    for a, b in itertools.combinations(sorted(fip), 2):
        kind = pair_kind(fip[a], fip[b])
        if kind is None:
            continue
        if kind == "DA-NE" and fip[a]["signal"] == "NE":
            a, b = b, a
        x, y = sig[a], sig[b]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        row = {"subject_id": subject_id, "ses_idx": ses_idx, "pair": kind, "ch_a": a, "ch_b": b,
               "duration_s": len(x) * config.DT, "r_full": float(np.corrcoef(x, y)[0, 1])}
        shifts = rng.integers(int(MIN_SHIFT_S / config.DT), len(x) - int(MIN_SHIFT_S / config.DT), N_SHIFT)
        for w_s in WINDOWS_S:
            w = int(w_s / config.DT)
            row[f"r_win{w_s}"] = float(np.nanmean(sliding_corr(x, y, w)))
            null = [np.nanmean(sliding_corr(x, np.roll(y, s), w)) for s in shifts[:30]]
            row[f"r_win{w_s}_null"] = float(np.mean(null))
            row[f"r_win{w_s}_null_sd"] = float(np.std(null))
        null_full = np.array([np.corrcoef(x, np.roll(y, s))[0, 1] for s in shifts])
        row["r_full_null"] = float(null_full.mean())
        row["r_full_null_sd"] = float(null_full.std())
        row["r_full_p_shift"] = float((np.sum(np.abs(null_full) >= abs(row["r_full"])) + 1) / (len(null_full) + 1))
        lags, cc = xcorr(x, y, int(MAX_LAG_S / config.DT))
        row["xcorr_peak_lag_s"] = float(lags[np.argmax(cc)] * config.DT)
        row["xcorr_peak"] = float(cc.max())
        row["xcorr"] = cc.astype(np.float32).tolist()
        rows.append(row)
    return rows


def build_all(workers: int = 24):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(session_worker, F.discover_sessions()))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(config.CACHE / "continuous_pairs.parquet")
    return df


if __name__ == "__main__":
    import time
    t = time.time()
    d = build_all()
    print(len(d), round(time.time() - t, 1))
    print(d.groupby("pair")[["r_full", "r_full_null", "r_win1", "r_win1_null", "r_win30", "xcorr_peak_lag_s"]].mean().round(3))
