"""Fig 2: NAc DA spans value-dominant, mixed and RPE-dominant regimes; d replaces the CCF map with a regime plane."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.classify import icc_oneway, pick_regime_examples
from analysis.figs import panels
from analysis.figs import style as st
from analysis.models import coding

ROWS = [("value_dominant", "a", "Tonic value coding example"),
        ("mixed", "b", "Mixed tonic value and phasic RPE coding example"),
        ("rpe_dominant", "c", "Phasic RPE coding example")]
COLS = [(42, 100), (180, 88), (310, 100), (450, 118)]


def summary_table() -> pd.DataFrame:
    path = config.TABLES / "coding_summary.csv"
    s = coding.foraging_summary()
    config.ensure_dirs()
    s.to_csv(path, index=False)
    return s


def regime_plane(P: st.Page, s: pd.DataFrame, y0: float):
    da = s[s["signal"] == "DA"].copy()
    ne = s[s["signal"] == "NE"].copy()
    ax = P.ax(42, y0, 150, 95)
    for d, marker, size in ((ne, "x", 8), (da[da["region"] == "medNAcc"], "^", 9), (da[da["region"] == "latNAcc"], "o", 7)):
        split = d[["slope_pos", "slope_neg"]].mean(axis=1)
        if marker == "x":
            ax.scatter(d["r_history"], split, s=size, marker="x", color="#808285", lw=0.6, zorder=2)
        else:
            ax.scatter(d["r_history"], split, s=size, marker=marker, c=[st.REGIME_COLORS[r] for r in d["regime"]],
                       edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(config.REGIME_R_HISTORY, color=st.INK, lw=0.5, ls=(0, (3, 2)))
    ax.axhline(0, color=st.INK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("Tonic value index\n(baseline vs past-outcome run, r)")
    ax.set_ylabel("Phasic RPE slope\n(mean split fit)")
    lo, hi = np.nanpercentile(pd.concat([da, ne])[["slope_pos", "slope_neg"]].mean(axis=1), [1, 99])
    ax.set_ylim(lo - 0.2, hi + 0.2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=st.REGIME_COLORS[k], markersize=3) for k in st.REGIME_LABEL]
    h += [Line2D([], [], marker="^", ls="", color="#58595b", markersize=3),
          Line2D([], [], marker="x", ls="", color="#808285", markersize=3)]
    ax.legend(h, [st.REGIME_LABEL[k] for k in st.REGIME_LABEL] + ["medNAcc DA", "PL NE"], fontsize=4.5,
              loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, labelspacing=0.3)

    ax2 = P.ax(262, y0, 190, 95)
    groups = [(a, da[(da["subject_id"] == a) & (da["region"] == "latNAcc")]) for a in sorted(da["subject_id"].unique())]
    groups = [(a, d) for a, d in groups if len(d)]
    groups += [("med", da[da["region"] == "medNAcc"]), ("NE", ne)]
    bottom = np.zeros(len(groups))
    order = ["value_dominant", "mixed", "rpe_dominant", "weak"]
    for k in order:
        frac = np.array([(d["regime"] == k).mean() for _, d in groups])
        ax2.bar(np.arange(len(groups)), frac, bottom=bottom, color=st.REGIME_COLORS[k], width=0.8, lw=0)
        bottom += frac
    ax2.set_xticks(np.arange(len(groups)))
    ax2.set_xticklabels([a if a in ("med", "NE") else a[-3:] for a, _ in groups], fontsize=4.8, rotation=0)
    for i, (_, d) in enumerate(groups):
        ax2.text(i, 1.02, str(len(d)), ha="center", va="bottom", fontsize=4.2)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Fraction of fiber-sessions")
    ax2.set_xlabel("latNAcc DA, per animal (ID suffix)           medNAcc   PL NE")
    return groups


def build():
    s = summary_table()
    m = cache.for_meta()
    feat = m[m["session_ok"] & m["valid"]]
    picks = pick_regime_examples(s, feat, "DA")
    P = st.Page(height=615)
    out = {"picks": {}, "rows": {}}
    for regime, letter, title in ROWS:
        y = {"a": 8, "b": 160, "c": 312}[letter]
        P.letter(6, y, letter, title)
        ses, ch = picks[regime]
        rows = panels.session_rows(ses, ch)
        yy = y + 24
        a1 = P.ax(COLS[0][0], yy, COLS[0][1], 95)
        panels.q_outcome_panel(a1, rows, level="session", legend=(letter == "a"))
        a2 = P.ax(COLS[1][0], yy, COLS[1][1], 95)
        hist = panels.history_bars(a2, rows)
        a3 = P.ax(COLS[2][0], yy, COLS[2][1], 95)
        panels.rpe_panel(a3, rows, level="session", legend=(letter == "a"))
        a4 = P.ax(COLS[3][0], yy, COLS[3][1], 95)
        fit = panels.split_scatter(a4, rows)
        rec = s[(s["ses_idx"] == ses) & (s["channel"] == ch)].iloc[0]
        out["picks"][regime] = {"ses_idx": ses, "channel": ch}
        out["rows"][regime] = {"n_trials": int(len(rows)), "r_history": rec["r_history"], "p_history": rec["p_history"],
                               "slope_neg": fit["slope_neg"], "p_neg": fit["p_neg"], "slope_pos": fit["slope_pos"],
                               "p_pos": fit["p_pos"], "classified_as": rec["regime"], "history_bar_means": hist}
        a4.text(1.0, -0.3, f"{ses}  {ch}", transform=a4.transAxes, ha="right", fontsize=4, color="#808285")

    P.letter(6, 462, "d", "Coding regimes across recordings")
    P.text(210, 466, "(data-driven replacement for the CCF site map; fiber coordinates not available)", fontsize=5, color="#808285")
    groups = regime_plane(P, s, 488)
    da_lat = s[(s["signal"] == "DA") & (s["region"] == "latNAcc")]
    counts = s.groupby(["signal", "region"])["regime"].value_counts().unstack(fill_value=0)
    out["regime_counts"] = {f"{a}_{b}": row.to_dict() for (a, b), row in counts.iterrows()}
    out["icc_r_history_latDA"] = icc_oneway(da_lat["r_history"], da_lat["subject_id"])
    out["icc_rpe_slope_latDA"] = icc_oneway(da_lat[["slope_pos", "slope_neg"]].mean(axis=1), da_lat["subject_id"])
    out["n_animals_with_multiple_regimes"] = int((da_lat.groupby("subject_id")["regime"].nunique() >= 3).sum())
    out["animal_tests_latDA"] = {c: S.summarize_animals(da_lat, c) for c in ("r_history", "slope_pos", "slope_neg", "rpe_graded")}
    P.text(470, 492, "ICC (between-animal share of variance)\n"
           f"tonic value index: {out['icc_r_history_latDA']:.2f}\n"
           f"RPE slope: {out['icc_rpe_slope_latDA']:.2f}\n"
           f"animals with ≥3 regimes: {out['n_animals_with_multiple_regimes']}/{da_lat['subject_id'].nunique()}",
           fontsize=5.5)
    cache.write_json(config.STATS / "fig2.json", out)
    return P.save("fig2")


if __name__ == "__main__":
    print(build())
