"""Event-aligned trace extraction, window means, and animal-level trace averaging."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import config


def align_events(t: np.ndarray, x: np.ndarray, t0: np.ndarray, grid: np.ndarray,
                 max_nan_frac: float = config.MAX_NAN_FRAC) -> np.ndarray:
    """Interpolate x(t) onto grid+t0 for every event; rows outside the recording or too NaN-heavy are NaN."""
    t0 = np.asarray(t0, float)
    out = np.full((len(t0), len(grid)), np.nan, dtype=np.float32)
    ok_t = np.isfinite(t0)
    if len(t) < 5 or not ok_t.any():
        return out
    good = np.isfinite(x)
    tt, xx = t[good], x[good]
    abs_t = grid[None, :] + t0[ok_t, None]
    vals = np.interp(abs_t.ravel(), tt, xx).reshape(abs_t.shape)
    inside = (abs_t >= t[0] - config.DT) & (abs_t <= t[-1] + config.DT)
    # mark samples that fall into NaN stretches of the raw signal
    if not good.all():
        nan_mask = np.interp(abs_t.ravel(), t, (~good).astype(float)).reshape(abs_t.shape) > 0.5
        inside &= ~nan_mask
    vals[~inside] = np.nan
    bad = np.mean(~inside, axis=1) > max_nan_frac
    vals[bad] = np.nan
    out[ok_t] = vals.astype(np.float32)
    return out


def window_mean(traces: np.ndarray, grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(traces[..., m], axis=-1)


def baseline_subtract(traces: np.ndarray, grid: np.ndarray, window=config.BASELINE) -> np.ndarray:
    return traces - window_mean(traces, grid, *window)[..., None]


def grouped_animal_traces(meta: pd.DataFrame, traces: np.ndarray, mask: np.ndarray | None = None,
                          levels=("ses_idx", "subject_id")) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean over trials within session, then sessions within animal; returns across-animal mean, SEM, n."""
    if mask is None:
        mask = np.ones(len(meta), bool)
    sub = meta.loc[mask, list(levels)].copy()
    tr = traces[mask]
    if len(sub) == 0:
        return np.full(traces.shape[1], np.nan), np.full(traces.shape[1], np.nan), 0
    sub["_row"] = np.arange(len(sub))
    ses_means = {}
    for key, g in sub.groupby(list(levels)):
        with np.errstate(all="ignore"):
            ses_means[key] = np.nanmean(tr[g["_row"].to_numpy()], axis=0)
    by_animal: dict = {}
    for (ses, animal), v in ses_means.items():
        by_animal.setdefault(animal, []).append(v)
    with np.errstate(all="ignore"):
        animal_means = np.vstack([np.nanmean(np.vstack(v), axis=0) for v in by_animal.values()])
    n = animal_means.shape[0]
    mean = np.nanmean(animal_means, axis=0)
    sem = np.nanstd(animal_means, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem, n


def trial_mean_sem(traces: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    tr = traces if mask is None else traces[mask]
    n = int(np.isfinite(tr).any(axis=1).sum())
    with np.errstate(all="ignore"):
        mean = np.nanmean(tr, axis=0)
        sem = np.nanstd(tr, axis=0, ddof=1) / np.sqrt(max(n, 1)) if n > 1 else np.zeros(tr.shape[1])
    return mean, sem, n
