"""Loading and trial-level enrichment for the dynamic-foraging dataset (DA_NE_cleaned_data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config
from analysis.dataio.channels import parse_channel

TRIAL_COLS = [
    "trial", "animal_response", "earned_reward", "extra_reward", "reward_all",
    "reward_probabilityL", "reward_probabilityR",
    "Q_left", "Q_right", "Q_chosen", "Q_unchosen", "Q_Delta", "Q_sum", "Q_change",
    "RPE_all", "RPE_earned", "P_chosen", "P_Delta", "P_change", "L_prob", "R_prob",
    "K_chosen", "response_time", "side_bias",
    "side_bias_confidence_interval_low", "side_bias_confidence_interval_high",
    "bonsai_start_time_in_session", "bonsai_stop_time_in_session",
    "goCue_start_time_in_session", "choice_time_in_session",
    "reward_outcome_time_in_session", "delay_duration", "ITI_duration", "num_reward_past",
]


def discover_sessions(root: Path = config.FORAGING_ROOT) -> list[tuple[str, str, Path]]:
    out = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if (session_dir / "df_trials.parquet").exists() and (session_dir / "df_fip.parquet").exists():
                out.append((subject_dir.name, session_dir.name, session_dir))
    return out


def load_session_meta(root: Path = config.FORAGING_ROOT) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(root.glob("df_sess_*.csv"))]
    meta = pd.concat(frames, ignore_index=True).drop(columns=["Unnamed: 0"], errors="ignore")
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["ses_idx"] = meta["ses_idx"].astype(str)
    meta["bias_R_model"] = -meta["biasL"]
    return meta.drop_duplicates("ses_idx")


def _signed_streak(outcomes: np.ndarray) -> np.ndarray:
    """Signed run length of the outcomes preceding each entry (+k: k rewards in a row, -k: k omissions)."""
    out = np.zeros(len(outcomes))
    run = 0
    for i, r in enumerate(outcomes):
        out[i] = run
        if r:
            run = run + 1 if run > 0 else 1
        else:
            run = run - 1 if run < 0 else -1
    return out


def enrich_trials(df: pd.DataFrame, subject_id: str, ses_idx: str) -> pd.DataFrame:
    df = df.sort_values("trial").reset_index(drop=True).copy()
    df["subject_id"] = subject_id
    df["ses_idx"] = ses_idx
    df["responded"] = df["animal_response"].isin([0, 1])
    df["ignored"] = df["animal_response"].eq(2)
    df["choice_right"] = np.where(df["responded"], df["animal_response"].eq(1).astype(float), np.nan)
    df["rewarded"] = df["reward_all"].fillna(False).astype(bool) & df["responded"]
    df["session_n_trials"] = len(df)

    resp = df[df["responded"]].copy()
    ridx = resp.index.to_numpy()
    rew = resp["rewarded"].to_numpy().astype(float)
    ch = resp["choice_right"].to_numpy()
    # number of ignored trials between consecutive responded trials
    gap = np.r_[0, np.diff(ridx) - 1]
    for k in range(1, config.HISTORY_MAX + 1):
        lag = np.full(len(resp), np.nan)
        lag[k:] = rew[:-k]
        df.loc[ridx, f"reward_lag{k}"] = lag
        clag = np.full(len(resp), np.nan)
        clag[k:] = ch[:-k]
        df.loc[ridx, f"choice_lag{k}"] = clag
        spans = np.full(len(resp), np.nan)
        cum = np.cumsum(gap)
        spans[k:] = (cum[k:] - cum[:-k]) > 0
        df.loc[ridx, f"ignored_within_lag{k}"] = spans
    df.loc[ridx, "prev_choice_right"] = np.r_[np.nan, ch[:-1]]
    df.loc[ridx, "prev_rewarded"] = np.r_[np.nan, rew[:-1]]
    df.loc[ridx, "streak_past"] = _signed_streak(rew.astype(bool))
    df.loc[ridx, "stay"] = np.r_[np.nan, (ch[1:] == ch[:-1]).astype(float)]
    df["prev_choice_sign"] = np.where(df["prev_choice_right"].isna(), 0.0,
                                      np.where(df["prev_choice_right"] > 0.5, 1.0, -1.0))
    df["abs_RPE"] = df["RPE_all"].abs()
    # upstream Q_Delta is Q_chosen - Q_unchosen; Q_RL is the side-referenced value difference
    df["Q_RL"] = df["Q_right"] - df["Q_left"]
    df["abs_Q_Delta"] = df["Q_Delta"].abs()
    df["rpe_bin"] = pd.cut(df["RPE_all"], bins=config.RPE_EDGES, labels=config.RPE_LABELS,
                           right=True, include_lowest=True).astype(object)
    df["q_bin"] = pd.cut(df["Q_chosen"], bins=config.Q_EDGES, labels=config.Q_LABELS,
                         right=True, include_lowest=True).astype(object)
    df["history_sum"] = df[[f"reward_lag{k}" for k in range(1, config.HISTORY_MAX + 1)]].sum(axis=1, min_count=1)
    df = add_delta_L(df)
    return df


def add_delta_L(df: pd.DataFrame) -> pd.DataFrame:
    """Su & Cohen policy update: change in log-odds of the chosen action from trial t to t+1.

    Uses the model choice probabilities L_prob / R_prob (P of left/right on each trial).
    """
    out = df.copy()
    pr = out["R_prob"].clip(1e-6, 1 - 1e-6)
    logit_r = np.log(pr / (1 - pr))
    out["logit_policy_R"] = logit_r
    out["delta_L"] = np.nan
    resp = out.index[out["responded"]]
    lr = logit_r.loc[resp].to_numpy()
    sign = np.where(out.loc[resp, "choice_right"].to_numpy() == 1, 1.0, -1.0)
    out.loc[resp, "delta_L"] = sign * (np.r_[lr[1:], np.nan] - lr)
    return out


def load_trials(session_dir: Path, subject_id: str, ses_idx: str) -> pd.DataFrame:
    df = pd.read_parquet(session_dir / "df_trials.parquet")
    df = df[[c for c in TRIAL_COLS if c in df.columns]]
    return enrich_trials(df, subject_id, ses_idx)


def load_events(session_dir: Path) -> pd.DataFrame:
    path = session_dir / "df_events.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamps", "event", "trial"])
    df = pd.read_parquet(path, columns=["timestamps", "event", "trial"])
    return df.sort_values("timestamps").reset_index(drop=True)


def load_licks(session_dir: Path) -> pd.DataFrame:
    ev = load_events(session_dir)
    lk = ev[ev["event"].isin(["left_lick_time", "right_lick_time"])].copy()
    lk["side"] = np.where(lk["event"].eq("right_lick_time"), "R", "L")
    return lk[["timestamps", "side", "trial"]].reset_index(drop=True)


def load_fip(session_dir: Path, columns=("timestamps", "data", "data_z", "event")) -> dict[str, dict]:
    """Return {channel: {'t', 'data', 'data_z', **channel_info}} for DA/NE channels, sorted by time."""
    df = pd.read_parquet(session_dir / "df_fip.parquet", columns=list(columns))
    out = {}
    for ch, g in df.groupby("event"):
        info = parse_channel(ch)
        if info is None:
            continue
        g = g.sort_values("timestamps")
        rec = {"t": g["timestamps"].to_numpy(float)}
        for c in columns:
            if c not in ("timestamps", "event"):
                rec[c] = g[c].to_numpy(float)
        rec.update(info)
        out[ch] = rec
    return out
