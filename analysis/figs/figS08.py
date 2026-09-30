"""Fig S8: Pavlovian extras (supports Figs 3 and 4d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.align import grouped_animal_traces
from analysis.figs import pav_panels as pp
from analysis.figs import style as st

P_LEVELS = [0.1, 0.5, 0.9]


def lick_adjusted(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ses, ch), g in meta.groupby(["ses_idx", "channel"]):
        a = S.ols(g["dff_cs"], {"p": g["p_reward"]})
        b = S.ols(g["dff_cs"], {"p": g["p_reward"], "licks": g["lick_anticip"], "pre": g["lick_pre"]})
        if a and b:
            rows.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                         "cs_slope": a["p"], "cs_slope_lick_adj": b["p"], "lick_beta": b["licks"]})
    return pd.DataFrame(rows)


def build():
    pav = pd.read_csv(config.TABLES / "pavlovian_summary.csv", dtype={"subject_id": str})
    licks = pd.read_csv(config.TABLES / "pavlovian_licks.csv", dtype={"subject_id": str})
    meta = cache.pav_meta()
    meta = meta[meta["valid"]]
    grid = cache.grids()["pavlovian"]
    P = st.Page(height=455)
    out = {"per_animal": {}, "acquired": {}, "learning": {}, "airpuff": {}, "lick_adjusted": {}}

    P.letter(6, 6, "a", "Per-animal cue and outcome responses (dF/F %, session means per animal)")
    for j, sig in enumerate(("DA", "NE")):
        d = pav[pav["signal"] == sig]
        colors = st.animal_colors(d["subject_id"].unique())
        for k, (prefix, lab) in enumerate((("dff_cs", "CS response"), ("dff_rew", "US response (rewarded)"),
                                           ("dff_om", "US response (omitted)"))):
            ax = P.ax(40 + (j * 3 + k) * 92, 30, 72, 65)
            a = d.groupby(["subject_id", "ses_idx"])[[f"{prefix}_{c}" for c in pp.CUES]].mean().groupby("subject_id").mean()
            for an, row in a.iterrows():
                ax.plot(P_LEVELS, row.to_numpy(), color=colors[an], marker="o", ms=2, lw=0.8)
            ax.plot(P_LEVELS, a.mean().to_numpy(), color="black", lw=1.4)
            ax.set_xticks(P_LEVELS)
            ax.set_xticklabels(["10", "50", "90"])
            ax.set_title(f"{sig}: {lab}", fontsize=5.5)
            ax.set_xlabel("P(reward) %")
            ax.axhline(0, color="#808285", lw=0.4)
            out["per_animal"][f"{sig}_{prefix}"] = a.to_dict()

    P.letter(6, 118, "b", "Sessions with significant lick discrimination only (CS3 > CS1, p < 0.05)")
    acq = licks.loc[licks["acquired"], "ses_idx"]
    for j, sig in enumerate(("DA", "NE")):
        rows = pp.rows_for(sig, sessions=acq)
        if rows.empty:
            continue
        axes = [P.ax(40 + j * 280 + i * 80, 150, 72, 55) for i in range(3)]
        pp.cue_traces(axes, rows)
        axes[0].set_ylabel(f"{sig} ΔF/F (%)")
        d = pav[(pav["signal"] == sig) & pav["ses_idx"].isin(acq)]
        out["acquired"][sig] = {k: S.summarize_animals(d, k) for k in ("dff_cs_slope", "dff_rew_slope", "dff_om_slope", "rpe_split")}
        out["acquired"][sig]["n_sessions"] = int(d["ses_idx"].nunique())

    P.letter(6, 232, "c", "Learning across sessions")
    lk = licks.copy()
    lk["session_n"] = lk.groupby("subject_id")["ses_idx"].rank(method="first")
    da = pav[pav["signal"] == "DA"].groupby(["subject_id", "ses_idx"])["dff_cs_slope"].mean().reset_index()
    da = da.merge(lk[["ses_idx", "session_n", "lick_slope", "has_airpuff"]], on="ses_idx")
    colors = st.animal_colors(lk["subject_id"].unique())
    for k, (d, col, lab) in enumerate(((lk, "lick_slope", "Lick slope (licks per unit P)"),
                                       (da, "dff_cs_slope", "DA CS slope (%ΔF/F per unit P)"))):
        ax = P.ax(40 + k * 150, 255, 120, 70)
        for an, g in d.groupby("subject_id"):
            ax.plot(g["session_n"], g[col], color=colors[an], marker="o", ms=1.8, lw=0.6)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.set_xlabel("Session # (within animal)")
        ax.set_ylabel(lab, fontsize=5)
        rho = S.summarize_animals(d.groupby("subject_id").apply(
            lambda g: S.spearman(g["session_n"], g[col])[0]).rename("rho").reset_index().assign(ses_idx=lambda x: x["subject_id"]),
            "rho")
        out["learning"][col] = rho
        st.stat_text(ax, f"within-animal ρ {st.fmt_ci(rho)}")

    P.letter(330, 232, "d", "Airpuff cue (CS4) vs 10% cue, airpuff sessions")
    for j, sig in enumerate(("DA", "NE")):
        ax = P.ax(355 + j * 115, 255, 100, 70)
        rows = meta[(meta["signal"] == sig) & meta["ses_idx"].isin(pav.loc[pav["has_airpuff"], "ses_idx"])]
        tr = cache.pav_traces_for(rows, "dff")
        tr = tr - np.nanmean(tr[:, (grid >= -1) & (grid <= 0)], axis=1, keepdims=True)
        for cs in ("CS1", "CS4"):
            mask = (rows["CS_type"] == cs).to_numpy()
            if mask.sum() < 3:
                continue
            mm, se, n = grouped_animal_traces(rows, tr, mask)
            sl = (grid >= -1) & (grid <= 5)
            st.trace(ax, grid[sl], mm[sl], se[sl], color=st.CS_COLORS[cs], label=f"{st.CS_LABEL[cs]} (n={n})")
        for x in (0, 2):
            ax.axvline(x, color="#808285", lw=0.5, ls=(0, (6, 4)))
        ax.set_xlabel("Time – CS (s)")
        ax.set_title(sig, fontsize=6)
        ax.legend(fontsize=4.5)
        r = rows.groupby(["subject_id", "ses_idx", "CS_type"])["dff_us"].mean().unstack()
        if "CS4" in r:
            diff = (r["CS4"] - r["CS1"]).rename("d").reset_index()
            out["airpuff"][f"{sig}_US_CS4_minus_CS1"] = S.summarize_animals(diff, "d")

    P.letter(6, 345, "e", "Lick-adjusted CS slope")
    la = lick_adjusted(meta[meta["p_reward"].notna()])
    ax = P.ax(40, 370, 150, 70)
    for j, (sig, color) in enumerate((("DA", st.DA_COLOR), ("NE", st.NE_COLOR))):
        d = la[la["signal"] == sig]
        for k, c in enumerate(("cs_slope", "cs_slope_lick_adj")):
            r = S.summarize_animals(d, c)
            out["lick_adjusted"][f"{sig}_{c}"] = r
            x = j * 2.5 + k
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=3, color=color,
                        mfc=color if k == 0 else "white", elinewidth=0.9)
        out["lick_adjusted"][f"{sig}_lick_beta"] = S.summarize_animals(d, "lick_beta")
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["DA", "DA\n+licks", "NE", "NE\n+licks"], fontsize=5)
    ax.set_ylabel("CS slope (%ΔF/F per unit P)")

    P.letter(220, 345, "f", "Summary")
    a = out["acquired"]
    lines = [f"Fig 3/4d use all 76 sessions. Restricting to the {a['DA']['n_sessions']} sessions with lick discrimination:",
             f"  DA CS slope {st.fmt_ci(a['DA']['dff_cs_slope'])}, rewarded-US slope {st.fmt_ci(a['DA']['dff_rew_slope'])},"
             f" omission slope {st.fmt_ci(a['DA']['dff_om_slope'])} (%ΔF/F per unit P).",
             f"  NE CS slope {st.fmt_ci(a['NE']['dff_cs_slope'])}, RPE index {st.fmt_ci(a['NE']['rpe_split'])}." if 'NE' in a else "",
             f"The DA CS slope survives anticipatory-lick covariates ({st.fmt_ci(out['lick_adjusted']['DA_cs_slope_lick_adj'])}).",
             f"Airpuff US vs 10% cue US: DA {st.fmt_ci(out['airpuff'].get('DA_US_CS4_minus_CS1', {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}))},"
             f" NE {st.fmt_ci(out['airpuff'].get('NE_US_CS4_minus_CS1', {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}))}.",
             "Rewards are defined from df_trials.US_type (df_events 'reward' also contains airpuffs)."]
    P.text(245, 370, "\n".join(lines), fontsize=5.2, linespacing=1.6)
    cache.write_json(config.STATS / "figS08.json", out)
    return P.save("figS08")


if __name__ == "__main__":
    print(build())
