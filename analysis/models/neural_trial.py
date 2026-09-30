"""Trial-level nested regressions of window responses with lick covariates (blocked cross-validation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S

LICK_COLS = ["licks_0_2", "licks_0_05", "licks_pre_choice", "bouts_prev_iti"]
STEPS_OUTCOME = [
    ("nuisance", ["rewarded", "side", "prev_choice", "trial_frac"]),
    ("+licks", LICK_COLS),
    ("+RPE", ["RPE_all"]),
    ("+|RPE|", ["abs_RPE"]),
    ("+Q_chosen", ["Q_chosen"]),
    ("+delta_L", ["delta_L"]),
]
STEPS_BASELINE = [
    ("nuisance", ["side", "prev_choice", "trial_frac", "licks_pre_choice", "bouts_prev_iti"]),
    ("+Q_chosen", ["Q_chosen"]),
    ("+Q_sum", ["Q_sum"]),
    ("+history", ["history_sum"]),
]


def trial_table() -> pd.DataFrame:
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]].copy()
    lf = pd.read_parquet(config.CACHE / "trial_lick_features.parquet")
    t = cache.for_trials()[["ses_idx", "trial", "prev_choice_sign", "history_sum", "session_n_trials"]]
    m = m.merge(lf, on=["ses_idx", "subject_id", "trial"], how="left").merge(t, on=["ses_idx", "trial"], how="left")
    m["rewarded"] = m["rewarded"].astype(float)
    m["side"] = np.where(m["choice_right"] == 1, 1.0, -1.0)
    m["prev_choice"] = m["prev_choice_sign"]
    m["trial_frac"] = m["trial"] / m["session_n_trials"]
    m["rpe_pos"] = m["RPE_all"].clip(lower=0)
    m["rpe_neg"] = m["RPE_all"].clip(upper=0)
    return m


def nested(g: pd.DataFrame, y: str, steps) -> dict:
    cols, out, prev = [], {}, None
    for name, add in steps:
        cols = cols + add
        r2 = S.blocked_cv_r2(g[y], {c: g[c] for c in cols})
        out[f"cv_{name}"] = r2
        if prev is not None:
            out[f"d_{name}"] = r2 - prev
        prev = r2
    return out


def split_with_licks(g: pd.DataFrame, y: str = "outcome_bs") -> dict:
    out = {}
    for tag, extra in (("nolick", []), ("lick", LICK_COLS)):
        fit = S.ols(g[y], {c: g[c] for c in ["rpe_pos", "rpe_neg", "rewarded", *extra]})
        if fit:
            out[f"slope_pos_{tag}"], out[f"slope_neg_{tag}"] = fit["rpe_pos"], fit["rpe_neg"]
            out[f"reward_jump_{tag}"] = fit["rewarded"]
    return out


def fit_all() -> pd.DataFrame:
    m = trial_table()
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        g = g.sort_values("trial")
        row = {"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
               "region": g["region"].iloc[0], "n": len(g)}
        row.update({f"out_{k}": v for k, v in nested(g, "outcome_bs", STEPS_OUTCOME).items()})
        row.update({f"base_{k}": v for k, v in nested(g, "baseline", STEPS_BASELINE).items()})
        row.update(split_with_licks(g))
        rows.append(row)
    return pd.DataFrame(rows)
