"""Stage 1: build trial tables, aligned trace matrices, lick tables and a session inventory.

Outputs (paper_figures_v2/cache):
  for_trials.parquet        all foraging trials (incl. ignored) with derived history/model columns
  for_meta.parquet          one row per responded trial x channel, with window features
  for_choice.npy/for_gocue.npy  float32 traces (rows match for_meta), grids in for_grids.npz
  for_licks.parquet, for_chan_stats.parquet, for_sess.parquet
  pav_trials.parquet, pav_meta.parquet, pav_cs_dff.npy, pav_cs_z.npy, pav_licks.parquet, pav_chan_stats.parquet
  session_inventory.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis import config
from analysis.core.align import align_events, window_mean
from analysis.dataio import foraging as F
from analysis.dataio import pavlovian as P

FOR_GRID = config.time_grid(config.FOR_T_PRE, config.FOR_T_POST)
PAV_GRID = config.time_grid(config.PAV_T_PRE, config.PAV_T_POST)

META_TRIAL_COLS = ["trial", "rewarded", "choice_right", "Q_chosen", "Q_Delta", "Q_RL", "Q_sum", "RPE_all",
                   "abs_RPE", "rpe_bin", "q_bin", "streak_past", "num_reward_past", "delta_L", "P_change",
                   "response_time", "choice_time_in_session", "goCue_start_time_in_session"]


def _chan_stats(fip: dict, ses_idx: str, subject_id: str) -> list[dict]:
    return [{"ses_idx": ses_idx, "subject_id": subject_id, "channel": ch, "signal": r["signal"],
             "region": r["region"], "hemi": r["hemi"], "dff_mean": float(np.nanmean(r["data"])),
             "dff_sd": float(np.nanstd(r["data"])), "nan_frac": float(np.mean(~np.isfinite(r["data"]))),
             "duration_s": float(r["t"][-1] - r["t"][0])} for ch, r in fip.items()]


def foraging_session(args):
    subject_id, ses_idx, d = args
    trials = F.load_trials(d, subject_id, ses_idx)
    fip = F.load_fip(d)
    licks = F.load_licks(d)
    licks["ses_idx"], licks["subject_id"] = ses_idx, subject_id
    resp = trials[trials["responded"] & trials["choice_time_in_session"].notna()].reset_index(drop=True)
    metas, choice, gocue = [], [], []
    for ch, r in fip.items():
        tc = align_events(r["t"], r["data_z"], resp["choice_time_in_session"].to_numpy(), FOR_GRID)
        tg = align_events(r["t"], r["data_z"], resp["goCue_start_time_in_session"].to_numpy(), FOR_GRID)
        m = resp[[c for c in META_TRIAL_COLS if c in resp.columns]].copy()
        m["ses_idx"], m["subject_id"] = ses_idx, subject_id
        m["channel"], m["signal"], m["region"], m["hemi"] = ch, r["signal"], r["region"], r["hemi"]
        m["valid"] = np.isfinite(tc).all(axis=1)
        base = window_mean(tc, FOR_GRID, *config.BASELINE)
        m["baseline"] = base
        m["baseline_alt"] = window_mean(tc, FOR_GRID, *config.BASELINE_ALT)
        m["outcome_raw"] = window_mean(tc, FOR_GRID, *config.OUTCOME)
        m["outcome_bs"] = m["outcome_raw"] - base
        for name, (lo, hi) in config.OUTCOME_ALTS.items():
            m[f"outcome_bs_{name}"] = window_mean(tc, FOR_GRID, lo, hi) - base
        gbase = window_mean(tg, FOR_GRID, -1.0, 0.0)
        m["gocue_base"] = gbase
        m["gocue_resp"] = window_mean(tg, FOR_GRID, *config.GOCUE_WINDOW) - gbase
        metas.append(m)
        choice.append(tc)
        gocue.append(tg)
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    shape = (0, len(FOR_GRID))
    return (trials, meta, np.vstack(choice) if choice else np.zeros(shape, np.float32),
            np.vstack(gocue) if gocue else np.zeros(shape, np.float32), licks, _chan_stats(fip, ses_idx, subject_id))


def pavlovian_session(args):
    subject_id, ses_idx, d = args
    trials = P.load_trials(d, subject_id, ses_idx)
    fip = P.load_fip(d)
    lk = pd.DataFrame({"timestamps": P.load_licks(d)})
    lk["ses_idx"], lk["subject_id"] = ses_idx, subject_id
    metas, dff, zz = [], [], []
    cs = trials["CS_start_time_in_session"].to_numpy()
    for ch, r in fip.items():
        a = align_events(r["t"], r["data"], cs, PAV_GRID)
        z = align_events(r["t"], r["data_z"], cs, PAV_GRID)
        m = trials[["trial", "CS_type", "p_reward", "water", "puff", "is_airpuff_cs", "lick_anticip",
                    "antilick", "lick_pre", "lick_post_us", "CS_start_time_in_session"]].copy()
        m["ses_idx"], m["subject_id"] = ses_idx, subject_id
        m["channel"], m["signal"], m["region"], m["hemi"] = ch, r["signal"], r["region"], r["hemi"]
        m["valid"] = np.isfinite(a).all(axis=1)
        for tag, tr in (("dff", a * 100.0), ("z", z)):
            base = window_mean(tr, PAV_GRID, *config.PAV_BASELINE)
            m[f"{tag}_base"] = base
            m[f"{tag}_cs"] = window_mean(tr, PAV_GRID, *config.PAV_CS_WINDOW) - base
            m[f"{tag}_delay"] = window_mean(tr, PAV_GRID, *config.PAV_DELAY_WINDOW) - base
            m[f"{tag}_us"] = window_mean(tr, PAV_GRID, *config.PAV_US_WINDOW) - base
            # outcome relative to the late-delay level, isolating the US-evoked change
            m[f"{tag}_us_rel"] = window_mean(tr, PAV_GRID, *config.PAV_US_WINDOW) - window_mean(tr, PAV_GRID, 1.5, 2.0)
        metas.append(m)
        dff.append(a * 100.0)
        zz.append(z)
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    shape = (0, len(PAV_GRID))
    return (trials, meta, np.vstack(dff).astype(np.float32) if dff else np.zeros(shape, np.float32),
            np.vstack(zz).astype(np.float32) if zz else np.zeros(shape, np.float32), lk,
            _chan_stats(fip, ses_idx, subject_id))


def _run(fn, sessions, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, sessions))


def build_foraging(workers: int) -> pd.DataFrame:
    sessions = F.discover_sessions()
    res = _run(foraging_session, sessions, workers)
    trials = pd.concat([r[0] for r in res], ignore_index=True)
    meta = pd.concat([r[1] for r in res], ignore_index=True)
    np.save(config.CACHE / "for_choice.npy", np.vstack([r[2] for r in res]))
    np.save(config.CACHE / "for_gocue.npy", np.vstack([r[3] for r in res]))
    trials.to_parquet(config.CACHE / "for_trials.parquet")
    meta.to_parquet(config.CACHE / "for_meta.parquet")
    pd.concat([r[4] for r in res], ignore_index=True).to_parquet(config.CACHE / "for_licks.parquet")
    stats = pd.DataFrame([s for r in res for s in r[5]])
    stats.to_parquet(config.CACHE / "for_chan_stats.parquet")
    F.load_session_meta().to_parquet(config.CACHE / "for_sess.parquet")

    inv = []
    for (subject_id, ses_idx, _), r in zip(sessions, res):
        t = r[0]
        n_resp = int(t["responded"].sum())
        chans = sorted(r[1]["channel"].unique()) if len(r[1]) else []
        q_ok = bool(t.loc[t["responded"], "Q_chosen"].notna().mean() > 0.95)
        reason = []
        if n_resp < config.MIN_RESPONDED_TRIALS:
            reason.append(f"responded<{config.MIN_RESPONDED_TRIALS}")
        if not q_ok:
            reason.append("missing_Q")
        if not chans:
            reason.append("no_channels")
        inv.append({"task": "foraging", "subject_id": subject_id, "ses_idx": ses_idx, "n_trials": len(t),
                    "n_responded": n_resp, "channels": ";".join(chans), "included": not reason,
                    "reason": ";".join(reason)})
    return pd.DataFrame(inv)


def build_pavlovian(workers: int) -> pd.DataFrame:
    sessions = P.discover_sessions()
    res = _run(pavlovian_session, sessions, workers)
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(config.CACHE / "pav_trials.parquet")
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(config.CACHE / "pav_meta.parquet")
    np.save(config.CACHE / "pav_cs_dff.npy", np.vstack([r[2] for r in res]))
    np.save(config.CACHE / "pav_cs_z.npy", np.vstack([r[3] for r in res]))
    pd.concat([r[4] for r in res], ignore_index=True).to_parquet(config.CACHE / "pav_licks.parquet")
    pd.DataFrame([s for r in res for s in r[5]]).to_parquet(config.CACHE / "pav_chan_stats.parquet")
    inv = []
    for (subject_id, ses_idx, _), r in zip(sessions, res):
        t = r[0]
        counts = t["CS_type"].value_counts()
        few = [c for c in config.CS_PROB if counts.get(c, 0) < 20]
        chans = sorted(r[1]["channel"].unique()) if len(r[1]) else []
        reason = ([f"few_{'_'.join(few)}"] if few else []) + ([] if chans else ["no_channels"])
        inv.append({"task": "pavlovian", "subject_id": subject_id, "ses_idx": ses_idx, "n_trials": len(t),
                    "n_responded": len(t), "channels": ";".join(chans), "included": not reason,
                    "reason": ";".join(reason)})
    return pd.DataFrame(inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=["foraging", "pavlovian"], default=None)
    a = ap.parse_args()
    config.ensure_dirs()
    np.savez(config.CACHE / "grids.npz", foraging=FOR_GRID, pavlovian=PAV_GRID)
    parts = []
    if a.only in (None, "foraging"):
        parts.append(build_foraging(a.workers))
    if a.only in (None, "pavlovian"):
        parts.append(build_pavlovian(a.workers))
    inv_path = config.CACHE / "session_inventory.csv"
    inv = pd.concat(parts, ignore_index=True)
    if a.only and inv_path.exists():
        old = pd.read_csv(inv_path)
        inv = pd.concat([old[old["task"] != a.only], inv], ignore_index=True)
    inv.to_csv(inv_path, index=False)
    print(inv.groupby(["task", "included"]).size())


if __name__ == "__main__":
    main()
