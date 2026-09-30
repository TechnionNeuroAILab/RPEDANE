"""Inter-trial-interval licking: ITIs from the raw trial table, isolated bout onsets, matched no-lick pseudo-events,
bout-aligned photometry, and per-trial lick features used as covariates elsewhere."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core.align import align_events, window_mean
from analysis.dataio import foraging as F

T_PRE, T_POST = 1.0, 2.0
GRID = config.time_grid(-T_PRE, T_POST)
GAPS = (0.5, 1.0, 2.0)
MAX_BOUTS = 150


def build_itis(trials: pd.DataFrame) -> pd.DataFrame:
    """ITI_i = [bonsai_stop_i, goCue_{i+1}] over consecutive rows of the raw table (ignored trials included),
    so an ITI never contains another trial's go cue."""
    d = trials.sort_values("trial").reset_index(drop=True)
    start = d["bonsai_stop_time_in_session"].to_numpy(float)[:-1]
    end = d["goCue_start_time_in_session"].to_numpy(float)[1:]
    out = pd.DataFrame({"trial": d["trial"].to_numpy()[:-1], "iti_start": start, "iti_end": end,
                        "prev_responded": d["responded"].to_numpy()[:-1],
                        "prev_rewarded": d["rewarded"].to_numpy()[:-1].astype(float),
                        "prev_choice_right": d["choice_right"].to_numpy()[:-1],
                        "next_responded": d["responded"].to_numpy()[1:]})
    out["iti_duration"] = out["iti_end"] - out["iti_start"]
    return out[np.isfinite(out["iti_duration"]) & (out["iti_duration"] > 0)].reset_index(drop=True)


def detect_bouts(licks: pd.DataFrame, itis: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Onsets preceded by >= gap s without any lick; [t0-1, t0+2] must lie inside a single ITI."""
    t = licks["timestamps"].to_numpy(float)
    side = licks["side"].to_numpy()
    order = np.argsort(t)
    t, side = t[order], side[order]
    if len(t) == 0 or itis.empty:
        return pd.DataFrame()
    idx = np.flatnonzero(np.r_[True, np.diff(t) >= gap])
    t0 = t[idx]
    s = itis["iti_start"].to_numpy()
    e = itis["iti_end"].to_numpy()
    j = np.clip(np.searchsorted(s, t0, side="right") - 1, 0, len(s) - 1)
    ok = (t0 - T_PRE >= s[j]) & (t0 + T_POST <= e[j])
    bout_n = np.r_[idx[1:], len(t)] - idx
    out = pd.DataFrame({"t0": t0, "side": side[idx], "bout_n": bout_n, "iti_row": j})[ok].reset_index(drop=True)
    it = itis.iloc[out["iti_row"].to_numpy()].reset_index(drop=True)
    out = pd.concat([out, it[["trial", "iti_start", "iti_end", "iti_duration", "prev_responded", "prev_rewarded",
                              "prev_choice_right"]]], axis=1)
    out["iti_frac"] = (out["t0"] - out["iti_start"]) / out["iti_duration"]
    out["gap"] = gap
    return out


def pseudo_events(bouts: pd.DataFrame, licks: pd.DataFrame, itis: pd.DataFrame, rng: np.random.Generator,
                  exclusion: float = 1.0, n_try: int = 30) -> pd.DataFrame:
    """One no-lick time per bout, at the same fractional ITI position in a random usable ITI."""
    lt = np.sort(licks["timestamps"].to_numpy(float))
    usable = itis[itis["iti_duration"] > T_PRE + T_POST + 0.1].reset_index(drop=True)
    if bouts.empty or usable.empty or len(lt) == 0:
        return pd.DataFrame()
    st_, du = usable["iti_start"].to_numpy(), usable["iti_duration"].to_numpy()
    lo, hi = st_ + T_PRE, usable["iti_end"].to_numpy() - T_POST
    rows = []
    for frac in bouts["iti_frac"].to_numpy():
        cj = rng.integers(0, len(usable), size=n_try)
        tt = np.clip(st_[cj] + frac * du[cj], lo[cj], hi[cj])
        k = np.searchsorted(lt, tt)
        near = np.minimum(np.abs(lt[np.clip(k - 1, 0, len(lt) - 1)] - tt), np.abs(lt[np.clip(k, 0, len(lt) - 1)] - tt))
        free = np.flatnonzero(near >= exclusion)
        if len(free) == 0:
            continue
        r = usable.iloc[cj[free[0]]]
        t_pick = tt[free[0]]
        rows.append({"t0": t_pick, "side": "none", "bout_n": 0, "trial": r["trial"], "iti_duration": r["iti_duration"],
                     "prev_responded": r["prev_responded"], "prev_rewarded": r["prev_rewarded"],
                     "prev_choice_right": r["prev_choice_right"], "iti_frac": (t_pick - r["iti_start"]) / r["iti_duration"]})
    return pd.DataFrame(rows)


def _count(sorted_t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b)
    res = np.full(len(a), np.nan)
    res[ok] = np.searchsorted(sorted_t, b[ok]) - np.searchsorted(sorted_t, a[ok])
    return res


def trial_lick_features(trials: pd.DataFrame, licks: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    lt = np.sort(licks["timestamps"].to_numpy(float))
    d = trials.sort_values("trial")
    ch = d["choice_time_in_session"].to_numpy(float)
    go = d["goCue_start_time_in_session"].to_numpy(float)
    prev_stop = np.r_[np.nan, d["bonsai_stop_time_in_session"].to_numpy(float)[:-1]]
    out = pd.DataFrame({"trial": d["trial"].to_numpy(), "licks_0_2": _count(lt, ch + 1e-3, ch + 2.0),
                        "licks_0_05": _count(lt, ch + 1e-3, ch + 0.5), "licks_pre_choice": _count(lt, go, ch),
                        "licks_prev_iti": _count(lt, prev_stop, go - 1e-3)})
    bt = np.sort(bouts["t0"].to_numpy()) if len(bouts) else np.array([])
    out["bouts_prev_iti"] = _count(bt, prev_stop, go)
    return out


def session_worker(args):
    subject_id, ses_idx, d = args
    trials = F.load_trials(d, subject_id, ses_idx)
    licks = F.load_licks(d)
    fip = F.load_fip(d)
    itis = build_itis(trials)
    rng = np.random.default_rng(int(ses_idx.split("_")[0]) + int(ses_idx.split("_")[1].replace("-", "")))
    ev_rows, tr_list = [], []
    bouts_main = pd.DataFrame()
    for gap in GAPS:
        b = detect_bouts(licks, itis, gap)
        if gap == 1.0:
            bouts_main = b
        if len(b) > MAX_BOUTS:
            b = b.sample(MAX_BOUTS, random_state=0).sort_values("t0")
        p = pseudo_events(b, licks, itis, rng)
        for kind, ev in (("bout", b), ("pseudo", p)):
            if ev.empty:
                continue
            for ch, r in fip.items():
                tr = align_events(r["t"], r["data_z"], ev["t0"].to_numpy(), GRID)
                bs = tr - window_mean(tr, GRID, -T_PRE, 0.0)[:, None]
                e = ev[["t0", "side", "bout_n", "trial", "iti_duration", "prev_responded", "prev_rewarded",
                        "iti_frac"]].copy()
                e["kind"], e["gap"], e["channel"] = kind, gap, ch
                e["signal"], e["region"], e["hemi"] = r["signal"], r["region"], r["hemi"]
                e["subject_id"], e["ses_idx"] = subject_id, ses_idx
                e["auc_0_05"] = window_mean(bs, GRID, 0, 0.5)
                e["auc_0_1"] = window_mean(bs, GRID, 0, 1.0)
                e["auc_0_2"] = window_mean(bs, GRID, 0, 2.0)
                e["absmod_0_1"] = window_mean(np.abs(bs), GRID, 0, 1.0)
                e["valid"] = np.isfinite(tr).all(axis=1)
                ev_rows.append(e)
                tr_list.append(bs.astype(np.float32))
    feats = trial_lick_features(trials, licks, bouts_main)
    feats["ses_idx"], feats["subject_id"] = ses_idx, subject_id
    itis["ses_idx"], itis["subject_id"] = ses_idx, subject_id
    lt = np.sort(licks["timestamps"].to_numpy(float))
    in_iti = int(np.nansum(_count(lt, itis["iti_start"].to_numpy(), itis["iti_end"].to_numpy()))) if len(itis) else 0
    summary = {"ses_idx": ses_idx, "subject_id": subject_id, "n_licks": len(licks), "n_itis": len(itis),
               "iti_total_s": float(itis["iti_duration"].sum()), "n_bouts_gap1": int(len(bouts_main)),
               "licks_in_iti": in_iti}
    events = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame()
    traces = np.vstack(tr_list) if tr_list else np.zeros((0, len(GRID)), np.float32)
    return events, traces, feats, itis, summary


def build_all(workers: int = 24):
    from concurrent.futures import ProcessPoolExecutor
    sessions = F.discover_sessions()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(session_worker, sessions))
    events = pd.concat([r[0] for r in res], ignore_index=True)
    np.save(config.CACHE / "iti_traces.npy", np.vstack([r[1] for r in res]))
    events.to_parquet(config.CACHE / "iti_events.parquet")
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(config.CACHE / "trial_lick_features.parquet")
    pd.concat([r[3] for r in res], ignore_index=True).to_parquet(config.CACHE / "itis.parquet")
    pd.DataFrame([r[4] for r in res]).to_parquet(config.CACHE / "iti_session_summary.parquet")
    return events


if __name__ == "__main__":
    import time
    t = time.time()
    ev = build_all()
    print(len(ev), round(time.time() - t, 1))
    print(ev.groupby(["gap", "kind", "signal"]).size())
