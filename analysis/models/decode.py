"""Single-trial decoding of upcoming stay/switch and of left/right choice from DA/NE features.

Within each animal: leave-one-session-out logistic regression (balanced class weights), balanced accuracy.
Chance comes from label permutation within sessions (features untouched). Behavior-only models (current reward and
choice) are the control; the neural contribution is neural+behavior minus behavior-only.
Features never use information from after the predicted choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis import config
from analysis.core import cache
from analysis.core.align import baseline_subtract, window_mean

N_PERM = 100
BINS = np.arange(-1.0, 3.0, 0.25)


def trial_features() -> pd.DataFrame:
    """One row per (session, trial) with DA (lat NAcc, hemispheres averaged) and NE window features, plus labels."""
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    grid = cache.grids()["foraging"]
    m = m[(m["signal"] == "NE") | (m["region"] == "latNAcc")].copy()
    tr = baseline_subtract(cache.traces_for(m), grid)
    for b in BINS:
        m[f"bin_{b:.2f}"] = window_mean(tr, grid, b, b + 0.25)
    feats = ["baseline", "gocue_resp", "outcome_bs"] + [f"bin_{b:.2f}" for b in BINS]
    wide = m.groupby(["subject_id", "ses_idx", "trial", "signal"])[feats].mean().unstack("signal")
    wide.columns = [f"{s}_{f}" for f, s in wide.columns]
    t = cache.for_trials()
    t = t[t["responded"]].sort_values(["ses_idx", "trial"]).copy()
    t["next_switch"] = t.groupby("ses_idx")["stay"].shift(-1).rsub(1)
    lab = t[["ses_idx", "trial", "rewarded", "choice_right", "next_switch", "prev_choice_sign", "prev_rewarded"]]
    return wide.reset_index().merge(lab, on=["ses_idx", "trial"], how="inner")


def loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    ok = np.isfinite(pred)
    return balanced_accuracy_score(y[ok], pred[ok]) if ok.sum() > 20 else np.nan


def permuted(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    yp = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        yp[idx] = y[rng.permutation(idx)]
    return yp


TASKS = {
    # name: (label, feature columns (templated by signal), behavior controls)
    "stay_switch": ("next_switch", ["{s}_baseline", "{s}_gocue_resp", "{s}_outcome_bs"], ["rewarded", "choice_right"]),
    "left_right": ("choice_right", ["{s}_baseline", "{s}_gocue_resp"], ["prev_choice_sign", "prev_rewarded"]),
}


def animal_job(args):
    subject, d, signals_list = args
    rng = np.random.default_rng(int(subject))
    rows = []
    for task, (label, tmpl, beh) in TASKS.items():
        for sigs in signals_list:
            cols = [c.format(s=s) for s in sigs for c in tmpl]
            dd = d.dropna(subset=cols + beh + [label])
            if len(dd) < 200 or dd["ses_idx"].nunique() < 3:
                continue
            y = dd[label].to_numpy().astype(int)
            g = dd["ses_idx"].to_numpy()
            Xn, Xb = dd[cols].to_numpy(float), dd[beh].to_numpy(float)
            res = {"subject_id": subject, "task": task, "features": "+".join(sigs), "n_trials": len(dd),
                   "n_sessions": int(dd["ses_idx"].nunique()),
                   "ba_neural": loso(Xn, y, g), "ba_behavior": loso(Xb, y, g),
                   "ba_neural_behavior": loso(np.column_stack([Xn, Xb]), y, g)}
            null = [loso(Xn, permuted(y, g, rng), g) for _ in range(N_PERM)]
            res["ba_neural_null_mean"] = float(np.nanmean(null))
            res["p_perm"] = float((np.sum(np.array(null) >= res["ba_neural"]) + 1) / (N_PERM + 1))
            rows.append(res)
            if task == "stay_switch":
                for b in BINS:
                    c = [f"{s}_bin_{b:.2f}" for s in sigs]
                    de = dd.dropna(subset=c)
                    rows.append({"subject_id": subject, "task": "stay_switch_time", "features": "+".join(sigs),
                                 "t": float(b + 0.125), "ba_neural": loso(de[c].to_numpy(float), de[label].to_numpy().astype(int),
                                                                            de["ses_idx"].to_numpy())})
    return rows


def build_all(workers: int = 16) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    f = trial_features()
    jobs = []
    for subject, d in f.groupby("subject_id"):
        has_ne = d["NE_baseline"].notna().any() if "NE_baseline" in d else False
        sl = [["DA"]] + ([["NE"], ["DA", "NE"]] if has_ne else [])
        jobs.append((subject, d, sl))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(animal_job, jobs))
    out = pd.DataFrame([r for rr in res for r in rr])
    out.to_parquet(config.CACHE / "decoding.parquet")
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    o = build_all()
    print(round(time.time() - t, 1))
    print(o[o["task"] != "stay_switch_time"].groupby(["task", "features"])[
        ["ba_neural", "ba_neural_null_mean", "ba_behavior", "ba_neural_behavior"]].mean().round(3))
