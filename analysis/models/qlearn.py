"""Q-learning refit (per animal, pooled over sessions) with forgetting of the unchosen value and a choice kernel.

Model (same family as the upstream QLearning_L1F1_CK1_softmax):
  delta = r - Q(c);  Q(c) += alpha * delta;  Q(u) *= (1 - forget)
  K(c) += k_step * (1 - K(c)); K(u) *= (1 - k_step)
  logit P(right) = beta * (Q_R - Q_L) + w_k * (K_R - K_L) + bias_R
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

PARAMS = ["alpha", "forget", "beta", "w_kernel", "k_step", "bias_R"]
BOUNDS = [(0.001, 0.999), (0.0, 1.0), (0.01, 30.0), (-10.0, 10.0), (0.01, 1.0), (-5.0, 5.0)]
STARTS = [[0.3, 0.1, 5.0, 0.5, 0.3, 0.0], [0.6, 0.3, 3.0, 1.0, 0.6, 0.0], [0.15, 0.0, 8.0, 0.0, 0.2, 0.0]]


def simulate_latents(params, choices: np.ndarray, rewards: np.ndarray) -> dict:
    """Run the model forward on observed choices/rewards; return per-trial latents and P(right)."""
    alpha, forget, beta, w_k, k_step, bias = params
    n = len(choices)
    q = np.zeros(2)
    k = np.zeros(2)
    out = {name: np.empty(n) for name in ("p_right", "Q_L", "Q_R", "Q_chosen", "RPE", "K_L", "K_R")}
    for i in range(n):
        out["Q_L"][i], out["Q_R"][i] = q
        out["K_L"][i], out["K_R"][i] = k
        out["p_right"][i] = expit(beta * (q[1] - q[0]) + w_k * (k[1] - k[0]) + bias)
        c = int(choices[i])
        d = rewards[i] - q[c]
        out["Q_chosen"][i] = q[c]
        out["RPE"][i] = d
        q[c] += alpha * d
        q[1 - c] *= 1 - forget
        k[c] += k_step * (1 - k[c])
        k[1 - c] *= 1 - k_step
    return out


def nll(params, sessions) -> float:
    total = 0.0
    for choices, rewards in sessions:
        p = simulate_latents(params, choices, rewards)["p_right"]
        p = np.clip(np.where(choices == 1, p, 1 - p), 1e-9, 1.0)
        total -= float(np.log(p).sum())
    return total


def fit_animal(sessions: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    fits = [minimize(nll, np.asarray(s), args=(sessions,), method="L-BFGS-B", bounds=BOUNDS) for s in STARTS]
    best = min(fits, key=lambda r: r.fun)
    n = sum(len(c) for c, _ in sessions)
    allc = np.concatenate([c for c, _ in sessions])
    p0 = np.clip(allc.mean(), 1e-6, 1 - 1e-6)
    nll0 = -float(np.sum(allc * np.log(p0) + (1 - allc) * np.log(1 - p0)))
    return {**dict(zip(PARAMS, best.x)), "nll": float(best.fun), "nll_null": nll0,
            "pseudo_r2": 1 - best.fun / nll0, "n_trials": n, "n_sessions": len(sessions),
            "converged": bool(best.success), "bic": 2 * best.fun + len(PARAMS) * np.log(n)}


def responded_sessions(trials: pd.DataFrame, subject_id: str):
    out = []
    for ses, g in trials[(trials["subject_id"] == subject_id) & trials["responded"]].groupby("ses_idx", sort=True):
        out.append((ses, g["choice_right"].to_numpy(int), g["rewarded"].to_numpy(float)))
    return out


def _fit_one(args):
    subject_id, sessions = args
    fit = fit_animal([(c, r) for _, c, r in sessions])
    fit["subject_id"] = subject_id
    return fit


def fit_all(trials: pd.DataFrame, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    from concurrent.futures import ProcessPoolExecutor
    subjects = sorted(trials["subject_id"].unique())
    jobs = [(s, responded_sessions(trials, s)) for s in subjects]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fits = list(ex.map(_fit_one, jobs))
    fits = pd.DataFrame(fits)
    lat = []
    for (s, sessions), fit in zip(jobs, fits.to_dict("records")):
        params = [fit[p] for p in PARAMS]
        for ses, c, r in sessions:
            z = simulate_latents(params, c, r)
            trial_ids = trials.loc[(trials["ses_idx"] == ses) & trials["responded"], "trial"].to_numpy()
            lat.append(pd.DataFrame({"ses_idx": ses, "trial": trial_ids, **{f"refit_{k}": v for k, v in z.items()}}))
    return fits, pd.concat(lat, ignore_index=True)


def simulate_agent(params, p_left: np.ndarray, p_right: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generative simulation on given block reward probabilities (no baiting); returns choices, rewards."""
    rng = np.random.default_rng(seed)
    alpha, forget, beta, w_k, k_step, bias = params
    q, k = np.zeros(2), np.zeros(2)
    ch, rw = np.empty(len(p_left), int), np.empty(len(p_left))
    for i in range(len(p_left)):
        pr = expit(beta * (q[1] - q[0]) + w_k * (k[1] - k[0]) + bias)
        c = int(rng.random() < pr)
        r = float(rng.random() < (p_right[i] if c else p_left[i]))
        ch[i], rw[i] = c, r
        d = r - q[c]
        q[c] += alpha * d
        q[1 - c] *= 1 - forget
        k[c] += k_step * (1 - k[c])
        k[1 - c] *= 1 - k_step
    return ch, rw
