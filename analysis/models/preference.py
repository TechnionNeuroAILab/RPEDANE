"""Left/right side bias. Every bias estimate uses the convention positive = rightward.

  side_bias             upstream running bias in df_trials (already +R)
  bias_R_model          -biasL from the per-session Q-learning fit (df_sess biasL is a left bias)
  b_side                intercept of logit P(R) = b_side + beta_Q * (Q_R - Q_L) + rho * prev_choice_sign
Note: the upstream Q_Delta column is Q_chosen - Q_unchosen, so it cannot be used as a side-referenced regressor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.core import stats as S

MIN_TRIALS, MIN_MINORITY, MIN_SWITCHES = 40, 8, 5
N_BOOT = 120


def fit_session(g: pd.DataFrame, seed: int = 0) -> dict:
    d = g[g["responded"]].dropna(subset=["choice_right", "Q_RL"])
    y = d["choice_right"].to_numpy(float)
    cols = {"beta_Q": d["Q_RL"].to_numpy(float), "rho": d["prev_choice_sign"].to_numpy(float)}
    n, k = len(y), int(y.sum())
    stay = d["stay"].dropna()
    out = {"n_trials": n, "p_right": k / n if n else np.nan, "minority_n": min(k, n - k),
           "n_switch": int((stay == 0).sum()), "p_stay": float(stay.mean()) if len(stay) else np.nan,
           "binom_p": float(stats.binomtest(k, n, 0.5).pvalue) if n else np.nan,
           "side_bias_upstream": float(d["side_bias"].mean()) if "side_bias" in d else np.nan}
    fit = S.logistic_mle(y, cols, min_n=MIN_TRIALS)
    if fit is None:
        return {**out, "identifiable": False, "flag": "fit_failed"}
    out.update({"b_side": fit["intercept"], "beta_Q": fit["beta_Q"], "rho": fit["rho"], "converged": fit["converged"],
                "condition_number": fit["condition_number"]})
    rng = np.random.default_rng(seed)
    bs = max(10, min(25, n // 8))
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.arange(s, min(s + bs, n)) for s in rng.integers(0, max(1, n - bs + 1), int(np.ceil(n / bs)))])[:n]
        f = S.logistic_mle(y[idx], {c: v[idx] for c, v in cols.items()}, min_n=MIN_TRIALS)
        if f is not None and f["converged"]:
            boots.append((f["intercept"], f["rho"]))
    if len(boots) >= 30:
        b = np.array(boots)
        out.update({"b_side_lo": np.quantile(b[:, 0], 0.025), "b_side_hi": np.quantile(b[:, 0], 0.975),
                    "rho_lo": np.quantile(b[:, 1], 0.025), "rho_hi": np.quantile(b[:, 1], 0.975)})
    flags = [f for f, bad in (("few_minority", out["minority_n"] < MIN_MINORITY), ("few_switches", out["n_switch"] < MIN_SWITCHES),
                              ("not_converged", not fit["converged"]), ("ill_conditioned", fit["condition_number"] > 1e8)) if bad]
    out["identifiable"] = not flags
    out["flag"] = ";".join(flags)
    out["stubborn"] = bool((out["p_stay"] >= 0.9 and out["minority_n"] < max(15, 0.15 * n))
                           or (abs(out["b_side"]) >= 1.0 and (out["p_right"] > 0.75 or out["p_right"] < 0.25)))
    return out


def fit_all(trials: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (ses, g) in enumerate(trials.groupby("ses_idx", sort=True)):
        r = fit_session(g, seed=i)
        r.update({"ses_idx": ses, "subject_id": g["subject_id"].iloc[0]})
        rows.append(r)
    out = pd.DataFrame(rows).merge(sess[["ses_idx", "bias_R_model", "session_date"]], on="ses_idx", how="left")
    out["day"] = out.groupby("subject_id")["session_date"].rank(method="first")
    return out


def psychometric(trials: pd.DataFrame, edges=np.linspace(-1, 1, 9)) -> pd.DataFrame:
    d = trials[trials["responded"]].dropna(subset=["Q_RL", "choice_right"]).copy()
    d["qbin"] = pd.cut(d["Q_RL"], edges, include_lowest=True)
    d["qc"] = d["qbin"].apply(lambda b: b.mid).astype(float)
    return d.groupby(["subject_id", "ses_idx", "qc"], observed=True)["choice_right"].mean().groupby(
        ["subject_id", "qc"]).mean().reset_index()


def hemisphere_side_interaction(meta: pd.DataFrame) -> pd.DataFrame:
    """Per fiber-session: outcome and baseline responses for choices ipsilateral vs contralateral to the fiber."""
    rows = []
    for (ses, ch), g in meta.groupby(["ses_idx", "channel"]):
        hemi = g["hemi"].iloc[0]
        ipsi = (g["choice_right"] == 1) if hemi == "R" else (g["choice_right"] == 0)
        row = {"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
               "region": g["region"].iloc[0], "hemi": hemi}
        for col in ("outcome_bs", "baseline", "gocue_resp"):
            row[f"{col}_contra_minus_ipsi"] = g.loc[~ipsi, col].mean() - g.loc[ipsi, col].mean()
        for rew in (True, False):
            sel = g["rewarded"] == rew
            row[f"outcome_bs_contra_minus_ipsi_{'R' if rew else 'U'}"] = (
                g.loc[sel & ~ipsi, "outcome_bs"].mean() - g.loc[sel & ipsi, "outcome_bs"].mean())
        fit = S.ols(g["outcome_bs"], {"rpe": g["RPE_all"], "contra": (~ipsi).astype(float),
                                      "rpe_x_contra": g["RPE_all"] * (~ipsi)})
        row["rpe_x_contra"] = fit["rpe_x_contra"] if fit else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
