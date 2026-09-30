"""Bayer-Glimcher reward-history regressions of phasic and tonic signals.

Model A (legacy):  outcome_bs  ~ R_t + R_{t-1..6}
Model B (primary): outcome_raw ~ R_t + R_{t-1..6} + baseline     (avoids the baseline-subtraction artefact)
Model C (tonic):   baseline    ~ R_{t-1..6}

For a scalar Q-learning RPE, beta_0 = g and beta_k = -g * alpha * (1-alpha)^(k-1), so the kernel cancels
(sum over lags + beta_0 -> 0) and the decay rate is the learning rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analysis import config
from analysis.core import stats as S

LAGS = list(range(1, config.HISTORY_MAX + 1))
LAG_COLS = [f"reward_lag{k}" for k in LAGS]


def _cols(g, with_current=True, with_baseline=False):
    c = {}
    if with_current:
        c["b0"] = g["rewarded"].astype(float).to_numpy()
    for k in LAGS:
        c[f"b{k}"] = g[f"reward_lag{k}"].to_numpy(float)
    if with_baseline:
        c["b_base"] = g["baseline"].to_numpy(float)
    return c


def fit_session(g: pd.DataFrame, drop_ignored_gaps: bool = False) -> dict | None:
    if drop_ignored_gaps:
        g = g[g[[f"ignored_within_lag{k}" for k in LAGS]].fillna(1).sum(axis=1) == 0]
    g = g.dropna(subset=LAG_COLS)
    if len(g) < config.MIN_BG_TRIALS:
        return None
    out = {"n": len(g)}
    for name, y, kw in (("A", g["outcome_bs"], {}), ("B", g["outcome_raw"], {"with_baseline": True}),
                        ("C", g["baseline"], {"with_current": False})):
        fit = S.ols(y.to_numpy(float), _cols(g, **kw), min_n=config.MIN_BG_TRIALS)
        if fit is None:
            return None
        for k in ([0] if kw.get("with_current", True) else []) + LAGS:
            out[f"{name}_b{k}"] = fit[f"b{k}"]
            out[f"{name}_se{k}"] = fit[f"se_b{k}"]
        if "with_baseline" in kw:
            out[f"{name}_b_base"] = fit["b_base"]
        out[f"{name}_r2"] = fit["r2"]
        if kw.get("with_current", True):
            out[f"{name}_hist_sum"] = sum(fit[f"b{k}"] for k in LAGS)
            out[f"{name}_total"] = fit["b0"] + out[f"{name}_hist_sum"]
            out[f"{name}_cancel_frac"] = -out[f"{name}_hist_sum"] / fit["b0"] if abs(fit["b0"]) > 1e-6 else np.nan
        else:
            out[f"{name}_hist_sum"] = sum(fit[f"b{k}"] for k in LAGS)
    return out


def fit_all_sessions(meta: pd.DataFrame, trials: pd.DataFrame, drop_ignored_gaps: bool = False) -> pd.DataFrame:
    lag_info = trials[["ses_idx", "trial", *LAG_COLS, *[f"ignored_within_lag{k}" for k in LAGS]]]
    d = meta.merge(lag_info, on=["ses_idx", "trial"], how="left")
    rows = []
    for (ses, ch), g in d.groupby(["ses_idx", "channel"]):
        r = fit_session(g, drop_ignored_gaps)
        if r is None:
            continue
        r.update({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                  "region": g["region"].iloc[0]})
        rows.append(r)
    return pd.DataFrame(rows)


def exp_kernel_fit(betas: np.ndarray) -> dict:
    """Fit beta_0 = g, beta_k = -h * a * (1-a)^(k-1): returns gain g, history scale h, decay a and h/g."""
    betas = np.asarray(betas, float)
    k = np.arange(1, len(betas))

    def resid(p):
        g, h, a = p
        return np.r_[betas[0] - g, betas[1:] + h * a * (1 - a) ** (k - 1)]

    best = None
    for a0 in (0.2, 0.5, 0.8):
        r = least_squares(resid, [betas[0], max(betas[0], 1e-3), a0], bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 0.99]))
        if best is None or r.cost < best.cost:
            best = r
    g, h, a = best.x
    return {"gain": float(g), "hist_scale": float(h), "alpha": float(a), "hist_over_gain": float(h / g) if abs(g) > 1e-6 else np.nan,
            "cost": float(best.cost)}


SCENARIOS = {
    # name: (tonic value in baseline, tonic value carried into the outcome window, phasic term)
    "Scalar RPE": (0.0, 0.0, "rpe"),
    "Tonic value + phasic RPE": (1.0, 1.0, "rpe"),
    "Tonic resets pre-outcome, reward-only phasic": (1.0, 0.0, "reward"),
    "Tonic value + reward (no RPE)": (1.0, 1.0, "reward"),
}


def simulate_scenarios(trials: pd.DataFrame, seed: int = 0, noise: float = 0.6, n_sessions: int = 40) -> pd.DataFrame:
    """Synthetic baseline/outcome signals built on real reward sequences and model Q_chosen/RPE.

    'Tonic value + phasic RPE' and 'Tonic value, reward-only phasic' produce the same outcome window
    (Q + (R - Q) = R), so no window-based regression can separate them; Model A reads both as RPE.
    """
    rng = np.random.default_rng(seed)
    ses = rng.choice(trials.loc[trials["responded"], "ses_idx"].unique(), size=n_sessions, replace=False)
    rows = []
    for s in ses:
        g = trials[(trials["ses_idx"] == s) & trials["responded"]].dropna(subset=LAG_COLS + ["Q_chosen", "RPE_all"])
        if len(g) < 100:
            continue
        q, r, d = g["Q_chosen"].to_numpy(), g["rewarded"].astype(float).to_numpy(), g["RPE_all"].to_numpy()
        for name, (k_base, k_out, ph) in SCENARIOS.items():
            base = k_base * q + rng.normal(0, noise, len(g))
            raw = k_out * q + (d if ph == "rpe" else r) + rng.normal(0, noise, len(g))
            fit = fit_session(g.assign(baseline=base, outcome_raw=raw, outcome_bs=raw - base))
            if fit:
                rows.append({"scenario": name, "ses_idx": s, **fit})
    return pd.DataFrame(rows)


def simulate_session_example(trials: pd.DataFrame, ses_idx: str, seed: int = 1, noise: float = 0.6) -> pd.DataFrame:
    """Per-trial simulated baseline/outcome values for one session, for each SCENARIOS entry (illustrative:
    shows what the four ground truths in `simulate_scenarios` actually look like trial by trial)."""
    rng = np.random.default_rng(seed)
    g = (trials[(trials["ses_idx"] == ses_idx) & trials["responded"]]
         .dropna(subset=LAG_COLS + ["Q_chosen", "RPE_all"]).sort_values("trial"))
    q, r, d = g["Q_chosen"].to_numpy(), g["rewarded"].astype(float).to_numpy(), g["RPE_all"].to_numpy()
    out = {"trial": np.arange(len(g)), "Q_chosen": q, "reward": r, "RPE": d}
    for name, (k_base, k_out, ph) in SCENARIOS.items():
        out[f"{name}__baseline"] = k_base * q + rng.normal(0, noise, len(g))
        out[f"{name}__outcome_raw"] = k_out * q + (d if ph == "rpe" else r) + rng.normal(0, noise, len(g))
    return pd.DataFrame(out)
