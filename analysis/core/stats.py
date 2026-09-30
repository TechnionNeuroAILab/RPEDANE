"""Inference helpers. The animal is the unit of inference: fiber -> session -> animal, then test across animals."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from analysis import config


def ols(y, columns: dict[str, np.ndarray], min_n: int = 20, standardize: bool = False) -> dict | None:
    names = list(columns)
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(columns[n], float) for n in names])
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[good], X[good]
    if len(y) < max(min_n, X.shape[1] + 5):
        return None
    if standardize:
        sd = X.std(axis=0)
        X = np.where(sd > 1e-12, (X - X.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)
        y = (y - y.mean()) / (y.std() + 1e-12)
    Z = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    pred = Z @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    dof = len(y) - Z.shape[1]
    sigma2 = ss_res / dof if dof > 0 else np.nan
    try:
        se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * sigma2)
    except np.linalg.LinAlgError:
        se = np.full(Z.shape[1], np.nan)
    out = {"intercept": float(coef[0]), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
           "n": int(len(y)), "ss_res": ss_res, "ss_tot": ss_tot}
    for i, name in enumerate(names):
        out[name] = float(coef[i + 1])
        out[f"se_{name}"] = float(se[i + 1])
        out[f"p_{name}"] = float(2 * stats.t.sf(abs(coef[i + 1] / se[i + 1]), dof)) if se[i + 1] > 0 else np.nan
    return out


def blocked_folds(n: int, n_blocks: int = 5) -> list[np.ndarray]:
    return np.array_split(np.arange(n), n_blocks)


def blocked_cv_r2(y, columns: dict[str, np.ndarray], n_blocks: int = 5, min_n: int = 40) -> float:
    """Contiguous-block cross-validated R^2 (OLS); NaN when too few trials."""
    names = list(columns)
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(columns[n], float) for n in names]) if names else np.zeros((len(y), 0))
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[good], X[good]
    n = len(y)
    if n < max(min_n, (X.shape[1] + 2) * n_blocks):
        return np.nan
    preds = np.full(n, np.nan)
    folds = blocked_folds(n, n_blocks)
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_blocks) if j != i])
        Ztr = np.column_stack([np.ones(len(tr)), X[tr]])
        Zte = np.column_stack([np.ones(len(te)), X[te]])
        coef, *_ = np.linalg.lstsq(Ztr, y[tr], rcond=None)
        preds[te] = Zte @ coef
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def logistic_mle(y, columns: dict[str, np.ndarray], min_n: int = 40) -> dict | None:
    from scipy.optimize import minimize

    names = list(columns)
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(columns[n], float) for n in names])
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[good], X[good]
    if len(y) < max(min_n, X.shape[1] + 10) or y.min() == y.max():
        return None
    Z = np.column_stack([np.ones(len(y)), X])

    def nll(theta):
        p = np.clip(expit(Z @ theta), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def grad(theta):
        return -Z.T @ (y - expit(Z @ theta))

    starts = [np.zeros(Z.shape[1]), np.r_[0.0, [0.5] * len(names)], np.r_[0.0, [-0.5] * len(names)]]
    best = min((minimize(nll, s, jac=grad, method="L-BFGS-B") for s in starts), key=lambda r: r.fun)
    p = expit(Z @ best.x)
    H = Z.T @ (Z * (p * (1 - p))[:, None])
    cond = float(np.linalg.cond(H))
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    out = {"intercept": float(best.x[0]), "se_intercept": float(se[0]), "nll": float(best.fun),
           "n": int(len(y)), "converged": bool(best.success), "condition_number": cond}
    for i, name in enumerate(names):
        out[name] = float(best.x[i + 1])
        out[f"se_{name}"] = float(se[i + 1])
    return out


def exact_sign_flip(values, null: float = 0.0, n_mc: int = 20000) -> float:
    """Two-sided sign-flip test of the mean; exact enumeration for n<=14."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)] - null
    n = len(v)
    if n < 2:
        return np.nan
    obs = abs(v.mean())
    if n <= 14:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    else:
        signs = np.random.default_rng(config.SEED).choice([-1.0, 1.0], size=(n_mc, n))
    return float(np.mean(np.abs((signs * v).mean(axis=1)) >= obs - 1e-12))


def min_sign_flip_p(n: int) -> float:
    return 2.0 / 2 ** n if n > 0 else np.nan


def bootstrap_mean(values, n_boot: int = config.N_BOOT, seed: int = config.SEED):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        m = float(v.mean()) if len(v) else np.nan
        return m, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def block_bootstrap_ci(values, block_size: int = 20, n_boot: int = 2000, seed: int = config.SEED):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        m = float(v.mean()) if len(v) else np.nan
        return m, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = max(1, min(block_size, len(v)))
    nb = int(np.ceil(len(v) / bs))
    boots = [np.concatenate([v[s:s + bs] for s in rng.integers(0, len(v) - bs + 1, size=nb)])[:len(v)].mean()
             for _ in range(n_boot)]
    return float(v.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def bh_fdr(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty_like(adj)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def tost_mean(values, bound: float) -> float:
    """Two one-sided t-tests: p that |mean| < bound (equivalence)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    se = v.std(ddof=1) / np.sqrt(len(v))
    t1 = (v.mean() + bound) / se
    t2 = (v.mean() - bound) / se
    return float(max(stats.t.sf(t1, len(v) - 1), stats.t.cdf(t2, len(v) - 1)))


def hierarchical_mean(df: pd.DataFrame, value: str, levels=("ses_idx", "subject_id")) -> pd.DataFrame:
    """Average `value` within each level successively (e.g. fiber rows -> session -> animal)."""
    cur = df[[*levels, value]].dropna(subset=[value])
    for i in range(len(levels)):
        keep = list(levels[i:])
        cur = cur.groupby(keep, as_index=False)[value].mean()
    return cur


def animal_values(df: pd.DataFrame, value: str) -> pd.Series:
    return hierarchical_mean(df, value).set_index("subject_id")[value]


def summarize_animals(df: pd.DataFrame, value: str, null: float = 0.0) -> dict:
    """Group-level summary across animals: mean, bootstrap CI, exact sign-flip p, n."""
    a = animal_values(df, value)
    m, lo, hi = bootstrap_mean(a.to_numpy())
    n_ses = int(df.dropna(subset=[value])["ses_idx"].nunique())
    return {"value": value, "mean": m, "ci_lo": lo, "ci_hi": hi, "p_signflip": exact_sign_flip(a.to_numpy(), null),
            "n_animals": int(len(a)), "n_sessions": n_ses, "per_animal": {str(k): float(v) for k, v in a.items()}}


def leave_one_animal_out(df: pd.DataFrame, value: str) -> dict:
    a = animal_values(df, value)
    return {str(k): float(a.drop(k).mean()) for k in a.index} if len(a) > 2 else {}


def mixedlm_intercept(df: pd.DataFrame, value: str) -> dict:
    """Random-intercept (animal) model of session-level values; secondary check."""
    import statsmodels.formula.api as smf
    d = df.groupby(["subject_id", "ses_idx"], as_index=False)[value].mean().dropna()
    if d["subject_id"].nunique() < 3:
        return {}
    try:
        fit = smf.mixedlm(f"{value} ~ 1", d, groups=d["subject_id"]).fit(reml=True, method="lbfgs")
        return {"coef": float(fit.params["Intercept"]), "se": float(fit.bse["Intercept"]),
                "p": float(fit.pvalues["Intercept"]), "n_sessions": int(len(d))}
    except Exception as exc:  # singular fits happen with few animals
        return {"error": str(exc)}


def circular_shift_null(x, y, n_shift: int = 200, min_shift: int = 20, seed: int = config.SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    shifts = rng.integers(min_shift, max(min_shift + 1, n - min_shift), size=n_shift)
    return np.array([np.corrcoef(x, np.roll(y, s))[0, 1] for s in shifts])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value
