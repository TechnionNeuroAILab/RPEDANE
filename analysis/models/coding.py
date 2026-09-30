"""Main-figure coding analyses: foraging value/RPE regimes and Pavlovian RPE signatures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.classify import session_summary


def partial_r(x, y, z) -> float:
    """Pearson r between x and y after regressing out z (and an intercept) from both."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 20:
        return np.nan
    Z = np.column_stack([np.ones(m.sum()), z[m]])
    rx = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]
    ry = y[m] - Z @ np.linalg.lstsq(Z, y[m], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def foraging_summary(outcome_col: str = "outcome_bs", history_col: str = "streak_past",
                     baseline_col: str = "baseline") -> pd.DataFrame:
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    s = session_summary(m, outcome_col=outcome_col, history_col=history_col, baseline_col=baseline_col)
    extra = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        r_rpe = S.pearson(g["RPE_all"], g[outcome_col])[0]
        r_q = S.pearson(g["Q_chosen"], g[baseline_col])[0]
        fit = S.ols(g[baseline_col], {"Q_chosen": g["Q_chosen"]})
        extra.append({"ses_idx": ses, "channel": ch, "r_rpe": r_rpe, "r_qchosen_baseline": r_q,
                      "rpe_graded": partial_r(g[outcome_col], g["RPE_all"], g["rewarded"].astype(float)),
                      "slope_baseline_q": fit["Q_chosen"] if fit else np.nan})
    out = s.merge(pd.DataFrame(extra), on=["ses_idx", "channel"])
    out["rpe_split"] = out[["slope_pos", "slope_neg"]].mean(axis=1)
    return out


def pav_session_summary() -> pd.DataFrame:
    """Per fiber-session Pavlovian indices on CS1-3 trials (dF/F %)."""
    m = cache.pav_meta()
    m = m[m["valid"] & m["p_reward"].notna()]
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        g = g.assign(rpe=g["water"].astype(float) - g["p_reward"])
        rew, om = g[g["water"]], g[~g["water"]]
        row = {"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
               "hemi": g["hemi"].iloc[0], "n_trials": len(g),
               "has_airpuff": bool(cache.pav_trials().query("ses_idx == @ses")["session_has_airpuff"].iloc[0])}
        for tag in ("dff", "z"):
            row[f"{tag}_cs_slope"] = stats.linregress(g["p_reward"], g[f"{tag}_cs"]).slope
            row[f"{tag}_rew_slope"] = stats.linregress(rew["p_reward"], rew[f"{tag}_us_rel"]).slope \
                if rew["p_reward"].nunique() > 1 else np.nan
            row[f"{tag}_om_slope"] = stats.linregress(om["p_reward"], om[f"{tag}_us_rel"]).slope \
                if om["p_reward"].nunique() > 1 else np.nan
            for cs, p in config.CS_PROB.items():
                row[f"{tag}_cs_{cs}"] = g.loc[g["CS_type"] == cs, f"{tag}_cs"].mean()
                row[f"{tag}_rew_{cs}"] = rew.loc[rew["CS_type"] == cs, f"{tag}_us_rel"].mean()
                row[f"{tag}_om_{cs}"] = om.loc[om["CS_type"] == cs, f"{tag}_us_rel"].mean()
        row["r_cue_value"] = S.pearson(g["p_reward"], g["z_cs"])[0]
        row["r_rpe"] = S.pearson(g["rpe"], g["z_us_rel"])[0]
        row["rpe_graded"] = partial_r(g["z_us_rel"], g["rpe"], g["water"].astype(float))
        row["canonical"] = bool(row["dff_cs_slope"] > 0 and row["dff_rew_slope"] < 0 and row["dff_om_slope"] < 0)
        # z-scored outcome response per unit RPE (RPE = 1-p on rewarded, -p on omitted trials)
        row["rpe_pos"] = -row["z_rew_slope"]
        row["rpe_neg"] = -row["z_om_slope"]
        row["rpe_split"] = np.nanmean([row["rpe_pos"], row["rpe_neg"]])
        rows.append(row)
    return pd.DataFrame(rows)


def pav_lick_summary() -> pd.DataFrame:
    t = cache.pav_trials()
    t = t[t["p_reward"].notna()]
    rows = []
    for ses, g in t.groupby("ses_idx"):
        a, b = g.loc[g["CS_type"] == "CS3", "lick_anticip"], g.loc[g["CS_type"] == "CS1", "lick_anticip"]
        rows.append({"ses_idx": ses, "subject_id": g["subject_id"].iloc[0],
                     "has_airpuff": bool(g["session_has_airpuff"].iloc[0]),
                     **{f"lick_{c}": g.loc[g["CS_type"] == c, "lick_anticip"].mean() for c in config.CS_PROB},
                     "lick_slope": stats.linregress(g["p_reward"], g["lick_anticip"]).slope,
                     "p_acq": stats.mannwhitneyu(a, b, alternative="greater").pvalue})
    out = pd.DataFrame(rows)
    out["acquired"] = out["p_acq"] < 0.05
    return out


def animal_test(df: pd.DataFrame, col: str, null: float = 0.0) -> dict:
    return S.summarize_animals(df, col, null)
