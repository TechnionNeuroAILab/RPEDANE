"""Fig S12: time-resolved RPE and value encoding (rolling regression) and DA vs NE encoding latency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.align import baseline_subtract
from analysis.figs import style as st

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]
N_PERM = 1000


def rolling_betas(rows: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Per fiber-session, standardized regression of the baseline-removed trace at each time bin on RPE and Q_chosen."""
    tr = baseline_subtract(cache.traces_for(rows), grid)
    rows = rows.assign(_i=np.arange(len(rows)))
    out = []
    for (ses, ch), g in rows.groupby(["ses_idx", "channel"]):
        X = g[["RPE_all", "Q_chosen"]].to_numpy(float)
        ok = np.isfinite(X).all(axis=1)
        if ok.sum() < 60:
            continue
        X = (X[ok] - X[ok].mean(0)) / X[ok].std(0)
        Y = tr[g["_i"].to_numpy()[ok]]
        Y = (Y - np.nanmean(Y, 0)) / (np.nanstd(Y, 0) + 1e-9)
        Z = np.column_stack([np.ones(len(X)), X])
        B = np.linalg.lstsq(Z, np.nan_to_num(Y), rcond=None)[0]
        out.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "b_rpe": B[1], "b_q": B[2]})
    return pd.DataFrame(out)


def animal_curves(bt: pd.DataFrame, col: str) -> pd.DataFrame:
    d = pd.DataFrame(np.vstack(bt[col].to_numpy()), index=pd.MultiIndex.from_frame(bt[["subject_id", "ses_idx"]]))
    return d.groupby(["subject_id", "ses_idx"]).mean().groupby("subject_id").mean()


def cluster_test(a: pd.DataFrame, t: np.ndarray, t_min=0.0, thr=2.0, seed=0):
    """Sign-flip cluster permutation across animals on the time course; returns significant masks and cluster p."""
    X = a.to_numpy()
    n = len(X)
    sel = t >= t_min

    def tstats(M):
        return M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n) + 1e-12)

    def clusters(tv):
        mask = (tv > thr) & sel
        cl, cur = [], []
        for i, m in enumerate(mask):
            if m:
                cur.append(i)
            elif cur:
                cl.append(cur)
                cur = []
        if cur:
            cl.append(cur)
        return cl

    tv = tstats(X)
    obs = clusters(tv)
    mass = [tv[c].sum() for c in obs]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_PERM):
        s = rng.choice([-1, 1], size=(n, 1))
        tp = tstats(X * s)
        cl = clusters(tp)
        null.append(max([tp[c].sum() for c in cl], default=0))
    null = np.array(null)
    res = [{"t_start": float(t[c[0]]), "t_end": float(t[c[-1]]), "mass": float(m), "p": float(np.mean(null >= m))}
           for c, m in zip(obs, mass)]
    return res


def onset(curve: np.ndarray, t: np.ndarray, frac=0.5):
    sel = t >= 0
    pk = np.nanmax(curve[sel])
    if not np.isfinite(pk) or pk <= 0:
        return np.nan, np.nan
    idx = np.flatnonzero(sel & (curve >= frac * pk))
    return float(t[idx[0]]), float(t[sel][np.nanargmax(curve[sel])])


def build():
    grid = cache.grids()["foraging"]
    sl = (grid >= -1) & (grid <= 4)
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    sim = set(m.loc[m["signal"] == "NE", "ses_idx"])
    P = st.Page(height=270)
    out = {"clusters": {}, "onset": {}, "paired": {}}
    curves = {}
    for r_i, (col, lab) in enumerate((("b_rpe", "RPE"), ("b_q", "Q_chosen"))):
        P.letter(6, 6 + r_i * 130, "ab"[r_i], f"Time-resolved {lab} encoding (standardized β, animal mean ± SEM)")
        for j, (scope, sessions) in enumerate((("all sessions", None), ("simultaneous sessions", sim))):
            ax = P.ax(40 + j * 185, 30 + r_i * 130, 160, 80)
            for sig, reg, color, glab in GROUPS:
                rows = m[(m["signal"] == sig) & (m["region"] == reg)]
                if sessions is not None:
                    rows = rows[rows["ses_idx"].isin(sessions)]
                key = (sig, scope)
                if key not in curves:
                    curves[key] = rolling_betas(rows, grid)
                a = animal_curves(curves[key], col)
                st.trace(ax, grid[sl], a.mean().to_numpy()[sl], (a.std(ddof=1) / np.sqrt(len(a))).to_numpy()[sl],
                         color=color, label=f"{glab} (n={len(a)})")
                cl = cluster_test(a, grid)
                out["clusters"][f"{sig}_{col}_{scope}"] = cl
                for c in cl:
                    if c["p"] < 0.05:
                        ax.plot([c["t_start"], c["t_end"]], [ax.get_ylim()[1] * (0.95 if sig == "DA" else 0.88)] * 2,
                                color=color, lw=2.5, solid_capstyle="butt")
                if col == "b_rpe":
                    ons = [onset(row.to_numpy(), grid) for _, row in a.iterrows()]
                    out["onset"][f"{sig}_{scope}"] = {"onset_s": S.bootstrap_mean(np.array([o[0] for o in ons])),
                                                      "peak_s": S.bootstrap_mean(np.array([o[1] for o in ons])),
                                                      "per_animal": dict(zip(a.index, ons))}
            st.event_line(ax)
            ax.axhline(0, color="#808285", lw=0.4)
            st.time_axis(ax)
            ax.set_ylabel(f"β({lab})")
            ax.set_title(scope, fontsize=6)
            ax.legend(fontsize=4.5, loc="center right")

    P.letter(400, 6, "c", "RPE encoding latency (simultaneous)")
    ax = P.ax(425, 30, 150, 80)
    rows_out = {}
    for k, what in enumerate(("onset", "peak")):
        per = {}
        for sig, reg, color, glab in GROUPS:
            o = out["onset"][f"{sig}_simultaneous sessions"]["per_animal"]
            per[sig] = {a: v[0 if what == "onset" else 1] for a, v in o.items()}
        both = [a for a in per["DA"] if a in per["NE"]]
        for a in both:
            ax.plot([k * 2.5, k * 2.5 + 1], [per["DA"][a], per["NE"][a]], color="#a7a9ac", lw=0.6, marker="o", ms=2)
        d = np.array([per["DA"][a] - per["NE"][a] for a in both])
        rows_out[what] = {"DA_minus_NE": S.bootstrap_mean(d), "p_signflip": S.exact_sign_flip(d), "n": len(d)}
    out["paired"] = rows_out
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["DA", "NE", "DA", "NE"])
    ax.text(0.2, -0.22, "onset (50% of peak)", transform=ax.transAxes, ha="center", fontsize=5)
    ax.text(0.8, -0.22, "peak", transform=ax.transAxes, ha="center", fontsize=5)
    ax.set_ylabel("time from choice (s)")

    P.letter(400, 136, "d", "Summary")
    po, pp_ = rows_out["onset"], rows_out["peak"]
    lines = [f"RPE encoding onset DA − NE: {po['DA_minus_NE'][0]:.2f} s [{po['DA_minus_NE'][1]:.2f}, {po['DA_minus_NE'][2]:.2f}]",
             f"(sign-flip p = {po['p_signflip']:.3f}, n = {po['n']} mice); peak DA − NE:",
             f"{pp_['DA_minus_NE'][0]:.2f} s [{pp_['DA_minus_NE'][1]:.2f}, {pp_['DA_minus_NE'][2]:.2f}].",
             "Bars in a/b: clusters surviving a sign-flip cluster",
             "permutation across animals (t > 2, p < 0.05).",
             "With RPE in the model, Q_chosen β is positive after",
             "choice, strongest for DA: value is added to the outcome",
             "response rather than subtracted (cf. Figs S3, S4)."]
    P.text(425, 160, "\n".join(lines), fontsize=5.2, linespacing=1.55)
    cache.write_json(config.STATS / "figS12.json", out)
    return P.save("figS12")


if __name__ == "__main__":
    print(build())
