"""Session-level RPE split regressions, tonic value index, and coding-regime classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis import config


def split_rpe_slopes(rpe, auc, min_n: int = 8) -> dict:
    """Separate linear fits of outcome response on RPE for RPE<0 and RPE>=0."""
    rpe, auc = np.asarray(rpe, float), np.asarray(auc, float)
    m = np.isfinite(rpe) & np.isfinite(auc)
    rpe, auc = rpe[m], auc[m]
    out = {"n_all": int(len(rpe))}
    for name, sel in (("all", np.ones(len(rpe), bool)), ("neg", rpe < 0), ("pos", rpe >= 0)):
        keys = {f"slope_{name}": np.nan, f"intercept_{name}": np.nan, f"r_{name}": np.nan,
                f"p_{name}": np.nan, f"n_{name}": int(sel.sum())}
        if sel.sum() >= min_n and np.ptp(rpe[sel]) > 0:
            lr = stats.linregress(rpe[sel], auc[sel])
            keys.update({f"slope_{name}": float(lr.slope), f"intercept_{name}": float(lr.intercept),
                         f"r_{name}": float(lr.rvalue), f"p_{name}": float(lr.pvalue)})
        out.update(keys)
    return out


def baseline_history_corr(history, baseline) -> dict:
    h, b = np.asarray(history, float), np.asarray(baseline, float)
    m = np.isfinite(h) & np.isfinite(b)
    out = {"r_history": np.nan, "p_history": np.nan, "n_history": int(m.sum())}
    if m.sum() >= 20 and np.ptp(h[m]) > 0:
        r, p = stats.pearsonr(h[m], b[m])
        out.update({"r_history": float(r), "p_history": float(p)})
    return out


def classify_regime(row, alpha: float = config.REGIME_ALPHA, r_thresh: float = config.REGIME_R_HISTORY) -> str:
    """mixed / rpe_dominant / value_dominant / weak from split slopes and the tonic history correlation."""
    pos_ok = bool(np.isfinite(row.get("slope_pos", np.nan)) and row["slope_pos"] > 0
                  and np.isfinite(row.get("p_pos", np.nan)) and row["p_pos"] < alpha)
    neg_ok = bool(np.isfinite(row.get("slope_neg", np.nan)) and row["slope_neg"] > 0
                  and np.isfinite(row.get("p_neg", np.nan)) and row["p_neg"] < alpha)
    r = row.get("r_history", np.nan)
    value = bool(np.isfinite(r) and r >= r_thresh)
    if pos_ok or neg_ok:
        return "mixed" if value else "rpe_dominant"
    return "value_dominant" if value else "weak"


def session_summary(feat: pd.DataFrame, outcome_col: str = "outcome_bs", history_col: str = "streak_past",
                    baseline_col: str = "baseline") -> pd.DataFrame:
    """One row per (ses_idx, channel): split RPE slopes, tonic history r, regime."""
    rows = []
    for (ses, ch), g in feat.groupby(["ses_idx", "channel"]):
        row = {"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0],
               "signal": g["signal"].iloc[0], "region": g["region"].iloc[0], "hemi": g["hemi"].iloc[0],
               "n_trials": int(g[outcome_col].notna().sum())}
        row.update(split_rpe_slopes(g["RPE_all"], g[outcome_col]))
        row.update(baseline_history_corr(g[history_col], g[baseline_col]))
        row["regime"] = classify_regime(row)
        rows.append(row)
    return pd.DataFrame(rows)


def pick_regime_examples(summary: pd.DataFrame, feat: pd.DataFrame, signal: str = "DA",
                         region: str | None = config.DA_MAIN_REGION) -> dict[str, tuple[str, str]]:
    """Clearest (ses_idx, channel) per regime among sessions with good coverage of the plotted strata."""
    s = summary[summary["signal"] == signal].copy()
    if region:
        s = s[s["region"] == region]
    cov = []
    for (ses, ch), g in feat[feat["signal"] == signal].groupby(["ses_idx", "channel"]):
        q_cells = sum(((g["q_bin"] == q) & (g["rewarded"] == r)).sum() >= 10
                      for q in ("low", "high") for r in (True, False))
        rpe_bins = int((g.groupby("rpe_bin").size() >= 5).sum())
        hist_bins = int((g["streak_past"].clip(-6, 6).value_counts() >= 3).sum())
        cov.append({"ses_idx": ses, "channel": ch, "q_cells": q_cells, "rpe_bins": rpe_bins, "hist_bins": hist_bins})
    s = s.merge(pd.DataFrame(cov), on=["ses_idx", "channel"], how="left")
    s = s[(s["n_trials"] >= 200) & (s["q_cells"] >= 3) & (s["rpe_bins"] >= 5) & (s["hist_bins"] >= 8)]
    best_slope = s[["slope_pos", "slope_neg"]].clip(lower=0).sum(axis=1)
    sig = (s["p_pos"].fillna(1) < 0.05).astype(int) + (s["p_neg"].fillna(1) < 0.05).astype(int)
    s = s.assign(
        score_value_dominant=s["r_history"] - best_slope - sig,
        score_mixed=s["r_history"] + np.minimum(s["slope_pos"].clip(lower=0), s["slope_neg"].clip(lower=0)) + sig,
        score_rpe_dominant=best_slope + sig - 3 * s["r_history"].abs(),
    )
    picks, used = {}, set()
    for regime in ("value_dominant", "mixed", "rpe_dominant"):
        pool = s[s["regime"] == regime]
        if pool.empty:
            pool = s
        for _, r in pool.sort_values(f"score_{regime}", ascending=False).iterrows():
            if r["ses_idx"] not in used:
                picks[regime] = (r["ses_idx"], r["channel"])
                used.add(r["ses_idx"])
                break
    return picks


def icc_oneway(values: pd.Series, groups: pd.Series) -> float:
    """ICC(1): share of variance between groups (e.g. animals) for a session-level index."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    k = d.groupby("g").size()
    if len(k) < 2:
        return np.nan
    grand = d["v"].mean()
    msb = (k * (d.groupby("g")["v"].mean() - grand) ** 2).sum() / (len(k) - 1)
    msw = ((d["v"] - d.groupby("g")["v"].transform("mean")) ** 2).sum() / (len(d) - len(k))
    k0 = (len(d) - (k ** 2).sum() / len(d)) / (len(k) - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))
