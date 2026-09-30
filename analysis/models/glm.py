"""Continuous 20 Hz ridge GLM of DA/NE photometry with event kernels and trial-level modulators.

Design per fiber-session (all regressors on the photometry grid):
  event kernels (FIR, 0.05 s bins): go cue [-0.5, 2], choice [-0.5, 3], outcome contrast (+1 reward / -1 omission at
    choice, [0, 3]; choice + contrast is full rank, unlike choice + reward + omission), ITI lick-bout onset [-0.5, 1.5]
  pre-choice boxcar [-1, 0] s: Q_chosen, Q_sum, |Q_Delta|, reward-history sum (tonic value family)
  post-choice weighted kernels (0.25 s bins, [0, 3] s): RPE, |RPE|, delta_L, side, previous choice, licks 0-2 s
Families for unique (drop-one) cross-validated Delta R^2 are listed in FAMILIES.
The ridge penalty is chosen by nested blocked CV; everything is computed from per-block sufficient statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.dataio import foraging as F
from analysis.models import iti_licks as IL

DT = config.DT
KERNELS = {"gocue": (-0.5, 2.0), "choice": (-0.5, 3.0), "outcome": (0.0, 3.0), "lick_bout": (-0.5, 1.5)}
PRE_MODS = ["Q_chosen", "Q_sum", "abs_Q_Delta", "history_sum"]
POST_MODS: list[str] = []
MOD_KERNELS = ["RPE_all", "abs_RPE", "delta_L", "side", "prev_choice", "licks_0_2"]
MOD_RANGE, MOD_BIN = (0.0, 3.0), 0.25
FAMILIES = {
    "events": ["gocue", "choice"],
    "outcome": ["outcome"],
    "lick": ["lick_bout", "licks_0_2"],
    "value": PRE_MODS,
    "rpe": ["RPE_all", "abs_RPE"],
    "policy": ["delta_L"],
    "side": ["side", "prev_choice"],
    "latents": PRE_MODS + ["RPE_all", "abs_RPE", "delta_L"],
}
LAMBDAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
N_BLOCKS = 5


def _impulses(n, t0, times, lo, hi, weights=None):
    lags = np.arange(int(round(lo / DT)), int(round(hi / DT)) + 1)
    cols = np.zeros((n, len(lags)), np.float32)
    idx0 = np.round((np.asarray(times) - t0) / DT).astype(int)
    w = np.ones(len(idx0)) if weights is None else np.asarray(weights, float)
    for j, L in enumerate(lags):
        ii = idx0 + L
        ok = (ii >= 0) & (ii < n) & np.isfinite(w)
        np.add.at(cols[:, j], ii[ok], w[ok])
    return cols, lags * DT


def _boxcar(n, t0, times, values, lo, hi):
    col = np.zeros(n, np.float32)
    a = np.round((np.asarray(times) + lo - t0) / DT).astype(int)
    b = np.round((np.asarray(times) + hi - t0) / DT).astype(int)
    for i0, i1, v in zip(a, b, values):
        if np.isfinite(v) and i1 > 0 and i0 < n:
            col[max(i0, 0):min(i1, n)] = v
    return col


def build_design(t: np.ndarray, trials: pd.DataFrame, bouts: np.ndarray, lick_feat: pd.DataFrame):
    n = len(t)
    t0 = t[0]
    r = trials[trials["responded"] & trials["choice_time_in_session"].notna()].merge(
        lick_feat[["trial", "licks_0_2"]], on="trial", how="left")
    ch = r["choice_time_in_session"].to_numpy()
    cols, names, groups, lag_of = [], [], [], []

    def add(mat, base, lags=None):
        mat = mat if mat.ndim == 2 else mat[:, None]
        cols.append(mat)
        for j in range(mat.shape[1]):
            names.append(f"{base}_{j}" if mat.shape[1] > 1 else base)
            groups.append(base)
            lag_of.append(lags[j] if lags is not None else np.nan)

    go = trials["goCue_start_time_in_session"].dropna().to_numpy()
    m, lags = _impulses(n, t0, go, *KERNELS["gocue"])
    add(m, "gocue", lags)
    m, lags = _impulses(n, t0, ch, *KERNELS["choice"])
    add(m, "choice", lags)
    m, lags = _impulses(n, t0, ch, *KERNELS["outcome"], weights=np.where(r["rewarded"], 1.0, -1.0))
    add(m, "outcome", lags)
    m, lags = _impulses(n, t0, bouts, *KERNELS["lick_bout"])
    add(m, "lick_bout", lags)
    r = r.assign(side=np.where(r["choice_right"] == 1, 1.0, -1.0), prev_choice=r["prev_choice_sign"])
    for c in PRE_MODS:
        v = r[c].to_numpy(float)
        add(_boxcar(n, t0, ch, v - np.nanmean(v), -1.0, 0.0), c)
    for c in POST_MODS:
        v = r[c].to_numpy(float)
        add(_boxcar(n, t0, ch, v - np.nanmean(v), 0.0, 2.0), c)
    step = int(round(MOD_BIN / DT))
    for c in MOD_KERNELS:
        v = r[c].to_numpy(float)
        m, lags = _impulses(n, t0, ch, MOD_RANGE[0], MOD_RANGE[1] - DT, weights=np.nan_to_num(v - np.nanmean(v)))
        nb = m.shape[1] // step
        mb = m[:, :nb * step].reshape(n, nb, step).sum(axis=2)
        add(mb, c, lags[:nb * step:step] + MOD_BIN / 2)
    X = np.hstack(cols).astype(np.float64)
    return X, np.array(names), np.array(groups), np.array(lag_of)


def _block_stats(X, y, n_blocks=N_BLOCKS):
    Z = np.column_stack([np.ones(len(y)), X])
    out = []
    for idx in np.array_split(np.arange(len(y)), n_blocks):
        Zb, yb = Z[idx], y[idx]
        out.append({"ZtZ": Zb.T @ Zb, "Zty": Zb.T @ yb, "yty": float(yb @ yb), "n": len(idx), "sy": float(yb.sum())})
    return out


def _solve(ZtZ, Zty, lam, scale):
    A = ZtZ.copy()
    pen = lam * scale ** 2
    pen[0] = 0.0
    A[np.diag_indices_from(A)] += pen
    return np.linalg.solve(A + 1e-8 * np.eye(len(A)), Zty)


def _sse(st, c):
    return st["yty"] - 2 * c @ st["Zty"] + c @ st["ZtZ"] @ c


def _sub(st, keep):
    return {"ZtZ": st["ZtZ"][np.ix_(keep, keep)], "Zty": st["Zty"][keep], "yty": st["yty"], "n": st["n"], "sy": st["sy"]}


def _sum(stats):
    return {k: sum(s[k] for s in stats) for k in ("ZtZ", "Zty", "yty", "n", "sy")}


def cv_r2(stats, scale, lam=None) -> tuple[float, float]:
    """Outer blocked CV R^2; if lam is None choose it per fold by inner blocked CV on the training blocks."""
    sse = sst = 0.0
    chosen = []
    for i in range(len(stats)):
        train = [s for j, s in enumerate(stats) if j != i]
        if lam is None:
            inner = []
            for L in LAMBDAS:
                e = 0.0
                for k in range(len(train)):
                    tr = _sum([s for j, s in enumerate(train) if j != k])
                    e += _sse(train[k], _solve(tr["ZtZ"], tr["Zty"], L, scale))
                inner.append(e)
            L = LAMBDAS[int(np.argmin(inner))]
        else:
            L = lam
        chosen.append(L)
        tr = _sum(train)
        c = _solve(tr["ZtZ"], tr["Zty"], L, scale)
        te = stats[i]
        sse += _sse(te, c)
        mu = tr["sy"] / tr["n"]
        sst += te["yty"] - 2 * mu * te["sy"] + te["n"] * mu ** 2
    return 1 - sse / sst, float(np.median(chosen))


def fit_fiber(t, y, X, names, groups, lag_of):
    good = np.isfinite(y)
    t, y, X = t[good], y[good], X[good]
    y = (y - y.mean()) / y.std()
    scale = np.r_[1.0, X.std(axis=0) + 1e-9]
    stats = _block_stats(X, y)
    full_r2, lam = cv_r2(stats, scale)
    out = {"cv_r2_full": full_r2, "lambda": lam, "n_samples": int(len(y))}
    col_group = np.concatenate([["intercept"], groups])
    for fam, members in FAMILIES.items():
        keep = np.flatnonzero(~np.isin(col_group, members))
        r2, _ = cv_r2([_sub(s, keep) for s in stats], scale[keep], lam)
        out[f"dR2_{fam}"] = full_r2 - r2
    tot = _sum(stats)
    coef = _solve(tot["ZtZ"], tot["Zty"], lam, scale)
    beta = coef[1:]
    out["pred_example"] = None
    kern = [{"kernel": g, "lag_s": float(l), "coef": float(b)} for g, l, b in zip(groups, lag_of, beta)
            if g in KERNELS or g in MOD_KERNELS]
    for g in PRE_MODS + POST_MODS:
        j = np.flatnonzero(groups == g)[0]
        out[f"beta_{g}"] = float(beta[j] * X[:, j].std())
    for g in MOD_KERNELS:
        j = np.flatnonzero((groups == g) & (lag_of <= 1.0))
        # response per 1 SD of the trial variable, averaged over the first second after choice
        out[f"beta_{g}"] = float(np.mean(beta[j]) * np.nanstd(X[:, j][X[:, j] != 0]) if len(j) else np.nan)
    return out, pd.DataFrame(kern), coef


def session_worker(args):
    subject_id, ses_idx, d = args
    trials = F.load_trials(d, subject_id, ses_idx)
    licks = F.load_licks(d)
    fip = F.load_fip(d)
    itis = IL.build_itis(trials)
    bouts = IL.detect_bouts(licks, itis, 1.0)
    bt = bouts["t0"].to_numpy() if len(bouts) else np.array([])
    lick_feat = IL.trial_lick_features(trials, licks, bouts)
    rows, kerns, examples = [], [], []
    resp = trials[trials["responded"]]
    if resp["Q_chosen"].isna().mean() > 0.05 or len(resp) < config.MIN_RESPONDED_TRIALS:
        return pd.DataFrame(), pd.DataFrame(), None
    for ch, r in fip.items():
        t_lo = trials["goCue_start_time_in_session"].min() - 2
        t_hi = trials["bonsai_stop_time_in_session"].max() + 2
        grid = np.arange(max(t_lo, r["t"][0]), min(t_hi, r["t"][-1]), DT)
        y = np.interp(grid, r["t"], r["data_z"])
        X, names, groups, lag_of = build_design(grid, trials, bt, lick_feat)
        res, kern, coef = fit_fiber(grid, y, X, names, groups, lag_of)
        meta = {"subject_id": subject_id, "ses_idx": ses_idx, "channel": ch, "signal": r["signal"],
                "region": r["region"], "hemi": r["hemi"]}
        res.pop("pred_example")
        rows.append({**meta, **res})
        kerns.append(kern.assign(**meta))
        if examples == []:
            seg = slice(len(grid) // 2, len(grid) // 2 + int(60 / DT))
            yz = (y - np.nanmean(y)) / np.nanstd(y)
            pred = np.column_stack([np.ones(len(grid)), X])[seg] @ coef
            examples = [{"channel": ch, "t": grid[seg] - grid[seg][0], "y": yz[seg], "pred": pred}]
    return pd.DataFrame(rows), pd.concat(kerns, ignore_index=True) if kerns else pd.DataFrame(), \
        (ses_idx, examples[0]) if examples else None


def vif_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation of the trial-level regressors (responded trials pooled within session, then averaged)."""
    cols = ["rewarded", *PRE_MODS, "RPE_all", "abs_RPE", "delta_L", "choice_right", "prev_choice_sign"]
    out = []
    for ses, g in trials[trials["responded"]].groupby("ses_idx"):
        d = g[cols].astype(float).dropna()
        if len(d) < 100:
            continue
        c = np.corrcoef(d.to_numpy().T)
        try:
            v = np.diag(np.linalg.inv(c))
        except np.linalg.LinAlgError:
            continue
        out.append(dict(zip(cols, v)))
    return pd.DataFrame(out)


def build_all(workers: int = 24):
    from concurrent.futures import ProcessPoolExecutor
    sessions = F.discover_sessions()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(session_worker, sessions))
    fits = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    kern = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    fits.to_parquet(config.CACHE / "glm_fits.parquet")
    kern.to_parquet(config.CACHE / "glm_kernels.parquet")
    ex = {s: e for s, e in (r[2] for r in res if r[2] is not None)}
    np.save(config.CACHE / "glm_examples.npy", ex, allow_pickle=True)
    return fits


if __name__ == "__main__":
    import time
    t = time.time()
    f = build_all()
    print(len(f), round(time.time() - t, 1))
    print(f.groupby("signal")[[c for c in f.columns if c.startswith("dR2") or c == "cv_r2_full"]].mean().round(4).T)
