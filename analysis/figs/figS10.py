"""Fig S10: response amplitude and decay kinetics of DA vs NE (replaces the DA vs NAc-ACh decay analysis; no ACh
channels in DA_NE_cleaned_data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.align import baseline_subtract, grouped_animal_traces
from analysis.figs import style as st

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]


def exp_tau(t, y):
    """Single exponential A*exp(-(t-tp)/tau)+C fitted from the peak (within 0-1 s) to 4 s."""
    sel = (t >= 0) & (t <= 1.0)
    if not np.isfinite(y[sel]).any():
        return np.nan, np.nan, np.nan
    ip = np.flatnonzero(sel)[np.nanargmax(y[sel])]
    tt, yy = t[ip:] - t[ip], y[ip:]
    ok = np.isfinite(yy) & (tt <= 3.0)
    tt, yy = tt[ok], yy[ok]
    try:
        p, _ = curve_fit(lambda x, a, tau, c: a * np.exp(-x / tau) + c, tt, yy, p0=[yy[0] - yy[-1], 0.5, yy[-1]],
                         bounds=([0, 0.05, -np.inf], [np.inf, 5.0, np.inf]), maxfev=4000)
        pred = p[0] * np.exp(-tt / p[1]) + p[2]
        r2 = 1 - np.sum((yy - pred) ** 2) / np.sum((yy - yy.mean()) ** 2)
        return float(p[1]), float(t[ip]), float(r2)
    except Exception:
        return np.nan, float(t[ip]), np.nan


def session_kinetics(rows: pd.DataFrame, kind: str, grid) -> pd.DataFrame:
    tr = baseline_subtract(cache.traces_for(rows, kind), grid)
    rows = rows.assign(_i=np.arange(len(rows)))
    out = []
    for (ses, ch), g in rows.groupby(["ses_idx", "channel"]):
        for rew in (True, False):
            sel = g[g["rewarded"] == rew]["_i"].to_numpy()
            if len(sel) < 15:
                continue
            m = np.nanmean(tr[sel], axis=0)
            tau, tpk, r2 = exp_tau(grid, m)
            win = (grid >= 0) & (grid <= 1)
            out.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                        "region": g["region"].iloc[0], "rewarded": rew, "tau": tau, "t_peak": tpk, "fit_r2": r2,
                        "amp": float(np.nanmax(m[win]))})
    return pd.DataFrame(out)


def build():
    grid = cache.grids()["foraging"]
    sl = (grid >= -1) & (grid <= 4)
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    sim = set(m.loc[m["signal"] == "NE", "ses_idx"])
    P = st.Page(height=345)
    out = {"amp": {}, "tau": {}, "paired": {}}

    for r_i, (kind, lab, grp) in enumerate((("choice", "choice/outcome-aligned", "rewarded"),
                                            ("gocue", "go-cue-aligned (split by upcoming outcome)", "rewarded"))):
        P.letter(6, 6 + r_i * 130, "ab"[r_i], f"Mean responses, {lab}")
        kin = []
        for j, (sig, reg, col, glab) in enumerate(GROUPS):
            rows = m[(m["signal"] == sig) & (m["region"] == reg) & m["ses_idx"].isin(sim)]
            ax = P.ax(40 + j * 130, 30 + r_i * 130, 105, 75)
            tr = baseline_subtract(cache.traces_for(rows, kind), grid)
            for rew, ls in ((True, "-"), (False, "--")):
                mm, se, n = grouped_animal_traces(rows, tr, (rows["rewarded"] == rew).to_numpy())
                st.trace(ax, grid[sl], mm[sl], se[sl], color=col, ls=ls, label=f"{'R+' if rew else 'R−'} (n={n})")
            st.event_line(ax)
            st.time_axis(ax, label=f"Time - {kind if kind == 'choice' else 'go cue'} (s)")
            ax.set_ylabel("z (baseline removed)")
            ax.set_title(f"{glab} (simultaneous sessions)", fontsize=5.8)
            ax.legend(fontsize=4.5)
            kin.append(session_kinetics(rows, kind, grid))
        kin = pd.concat(kin)
        kin.to_csv(config.TABLES / f"kinetics_{kind}.csv", index=False)
        for c_i, (col_name, ylab) in enumerate((("amp", "peak amplitude (z)"), ("tau", "decay τ (s)"))):
            ax = P.ax(320 + c_i * 135, 30 + r_i * 130, 105, 75)
            for j, (sig, reg, col, glab) in enumerate(GROUPS):
                for k, rew in enumerate((True, False)):
                    d = kin[(kin["signal"] == sig) & (kin["rewarded"] == rew)]
                    if col_name == "tau":
                        d = d[d["fit_r2"] > 0.5]
                    r = S.summarize_animals(d, col_name)
                    out[col_name][f"{kind}_{sig}_{'R' if rew else 'U'}"] = r
                    x = j * 2.5 + k
                    v = list(r["per_animal"].values())
                    ax.scatter(np.full(len(v), x) - 0.15, v, s=4, color=col, lw=0, alpha=0.6)
                    ax.errorbar(x + 0.15, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o",
                                ms=2.8, color=col, mfc=col if rew else "white", elinewidth=0.9)
            # paired DA vs NE within animal (rewarded)
            a = kin[kin["rewarded"]].pipe(lambda d: d[d["fit_r2"] > 0.5] if col_name == "tau" else d)
            av = a.groupby(["subject_id", "ses_idx", "signal"])[col_name].mean().unstack().dropna()
            diff = (av["DA"] - av["NE"]).rename("d").reset_index()
            out["paired"][f"{kind}_{col_name}_DA_minus_NE_R"] = S.summarize_animals(diff, "d")
            ax.set_xticks([0, 1, 2.5, 3.5])
            ax.set_xticklabels(["DA R+", "DA R−", "NE R+", "NE R−"], fontsize=4.8)
            ax.set_ylabel(ylab)
            pr = out["paired"][f"{kind}_{col_name}_DA_minus_NE_R"]
            st.stat_text(ax, f"DA−NE (R+): {st.fmt_ci(pr)}, {st.fmt_p(pr['p_signflip'])}", y=1.12, fontsize=4.3)

    P.letter(6, 270, "c", "Summary")
    pa = out["paired"]
    lines = [f"Outcome-aligned, rewarded trials, same sessions: DA − NE peak amplitude {st.fmt_ci(pa['choice_amp_DA_minus_NE_R'])} z,",
             f"decay τ {st.fmt_ci(pa['choice_tau_DA_minus_NE_R'])} s (exponential fits with R² > 0.5).",
             f"Go-cue-aligned: amplitude {st.fmt_ci(pa['gocue_amp_DA_minus_NE_R'])}, τ {st.fmt_ci(pa['gocue_tau_DA_minus_NE_R'])} s.",
             "The earlier version of this analysis compared DA with NAc acetylcholine (rAch); rAch is not recorded in "
             f"{config.FORAGING_ROOT.name}, so the comparison is DA vs PL NE in simultaneous sessions (n = 5 mice)."]
    P.text(40, 292, "\n".join(lines), fontsize=5.4, linespacing=1.6)
    cache.write_json(config.STATS / "figS10.json", out)
    return P.save("figS10")


if __name__ == "__main__":
    print(build())
