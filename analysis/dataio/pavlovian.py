"""Loading for the Pavlovian dataset (DA_NE_pavlovian/nwbs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config
from analysis.dataio.channels import parse_channel


def discover_sessions(root: Path = config.PAV_ROOT) -> list[tuple[str, str, Path]]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "df_trials.parquet").exists() and (d / "df_fip.parquet").exists():
            out.append((d.name.split("_")[0], d.name, d))
    return out


def load_licks(session_dir: Path) -> np.ndarray:
    ev = pd.read_parquet(session_dir / "df_events.parquet", columns=["timestamps", "event"])
    return np.sort(ev.loc[ev["event"].eq("lick"), "timestamps"].to_numpy(float))


def load_trials(session_dir: Path, subject_id: str, ses_idx: str) -> pd.DataFrame:
    """Trials with cue probability; reward comes from US_type, never from df_events (which pools airpuffs)."""
    t = pd.read_parquet(session_dir / "df_trials.parquet")
    t = t[["trial", "CS_type", "rewarded", "US_type", "airpuff", "antilick",
           "CS_start_time_in_session", "US_start_time_in_session",
           "start_time_in_session", "stop_time_in_session"]].copy()
    t = t.sort_values("trial").reset_index(drop=True)
    t["subject_id"] = subject_id
    t["ses_idx"] = ses_idx
    t["is_airpuff_cs"] = t["CS_type"].eq(config.CS_AIRPUFF)
    t["water"] = t["US_type"].eq("EarnedReward")
    t["puff"] = t["US_type"].eq("Airpuff")
    t["p_reward"] = t["CS_type"].map(config.CS_PROB)
    t["session_has_airpuff"] = bool(t["is_airpuff_cs"].any())
    t["outcome_time"] = t["CS_start_time_in_session"] + config.US_DELAY

    licks = load_licks(session_dir)
    cs = t["CS_start_time_in_session"].to_numpy(float)
    t["lick_anticip"] = np.searchsorted(licks, cs + config.US_DELAY) - np.searchsorted(licks, cs)
    t["lick_pre"] = np.searchsorted(licks, cs) - np.searchsorted(licks, cs - config.US_DELAY)
    t["lick_post_us"] = np.searchsorted(licks, cs + config.US_DELAY + 2.0) - np.searchsorted(licks, cs + config.US_DELAY)
    return t


def load_fip(session_dir: Path) -> dict[str, dict]:
    """{channel: {'t', 'data', 'data_z', ...}}; data_z is a session-wide z-score computed here."""
    df = pd.read_parquet(session_dir / "df_fip.parquet", columns=["timestamps", "data", "event"])
    out = {}
    for ch, g in df.groupby("event"):
        info = parse_channel(ch)
        if info is None:
            continue
        g = g.sort_values("timestamps")
        x = g["data"].to_numpy(float)
        mu, sd = np.nanmean(x), np.nanstd(x)
        out[ch] = {"t": g["timestamps"].to_numpy(float), "data": x,
                   "data_z": (x - mu) / sd if sd > 0 else x * np.nan, **info}
    return out
