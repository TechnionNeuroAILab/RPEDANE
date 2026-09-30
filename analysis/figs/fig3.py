"""Fig 3: NAc DA reproduces canonical Pavlovian RPE coding."""
from __future__ import annotations

from matplotlib.patches import Patch

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import pav_panels as pp
from analysis.figs import style as st
from analysis.models import coding

SRC = "figure3_DANE.pdf"


def pick_example(pav: "pd.DataFrame", signal: str) -> tuple[str, str]:
    d = pav[(pav["signal"] == signal) & ~pav["has_airpuff"] & pav["canonical"]].copy()
    if d.empty:
        d = pav[pav["signal"] == signal].copy()
    d["score"] = d["dff_cs_slope"] - d["dff_rew_slope"] - d["dff_om_slope"]
    r = d.sort_values("score", ascending=False).iloc[0]
    return r["ses_idx"], r["channel"]


def build():
    pav = coding.pav_session_summary()
    licks = coding.pav_lick_summary()
    pav.to_csv(config.TABLES / "pavlovian_summary.csv", index=False)
    licks.to_csv(config.TABLES / "pavlovian_licks.csv", index=False)
    P = st.Page(height=205)
    P.letter(6, 6, "a", "Pavlovian Task and behavior")
    P.snippet(SRC, (25, 20, 125, 75), (22, 24, 100, 55))
    ses, ch = pick_example(pav, "DA")
    ax_ev = P.ax(150, 16, 400, 10)
    ax = P.ax(150, 26, 400, 52)
    ex = pp.example_trace(ax_ev, ax, ses, ch, pp.best_window(ses))
    lg = [Patch(color=st.EVENT["reward"], label="Reward")] + [Patch(color=st.CS_COLORS[c], label=st.CS_LABEL[c])
                                                             for c in ("CS1", "CS2", "CS3", "CS4")]
    P.fig.legend(handles=lg, loc="upper left", bbox_to_anchor=(556 / P.width, 1 - 18 / P.height), fontsize=5,
                 handlelength=0.8, labelspacing=0.2, title="CS", title_fontsize=5)

    P.letter(6, 94, "b", "RPE in DA response")
    P.snippet(SRC, (10, 105, 170, 185), (8, 108, 158, 80))
    rows = pp.rows_for("DA")
    axes = [P.ax(200 + i * 76, 118, 70, 58) for i in range(3)]
    counts = pp.cue_traces(axes, rows)
    from matplotlib.lines import Line2D
    axes[2].legend([Line2D([], [], color=st.INK, lw=1), Line2D([], [], color=st.INK, lw=1, ls=(0, (4, 2)))],
                   ["reward", "omission"], loc="upper right", fontsize=4.5, handlelength=1.6)
    ax_l = P.ax(460, 108, 80, 68)
    pp.lick_strip(ax_l, licks)

    da = pav[pav["signal"] == "DA"]
    lk = licks.assign(d31=licks["lick_CS3"] - licks["lick_CS1"])
    stats = {"example": ex, "trace_n_animals": counts,
             "n_animals": int(rows["subject_id"].nunique()), "n_sessions": int(rows["ses_idx"].nunique()),
             "n_fiber_sessions": int(rows.groupby(["ses_idx", "channel"]).ngroups),
             "cs_slope_dff_per_prob": S.summarize_animals(da, "dff_cs_slope"),
             "reward_response_slope_vs_prob": S.summarize_animals(da, "dff_rew_slope"),
             "omission_response_slope_vs_prob": S.summarize_animals(da, "dff_om_slope"),
             "omission_dip_90pct": S.summarize_animals(da, "dff_om_CS3"),
             "reward_10_minus_90": S.summarize_animals(da.assign(d=da["dff_rew_CS1"] - da["dff_rew_CS3"]), "d"),
             "graded_rpe_partial_r": S.summarize_animals(da, "rpe_graded"),
             "cue_value_r": S.summarize_animals(da, "r_cue_value"),
             "fraction_canonical_sessions": float(da["canonical"].mean()),
             "licks_CS3_minus_CS1": S.summarize_animals(lk, "d31"),
             "licks_slope_vs_prob": S.summarize_animals(licks, "lick_slope"),
             "sessions_lick_acquired": int(licks["acquired"].sum()), "sessions_total": int(len(licks))}
    cache.write_json(config.STATS / "fig3.json", stats)
    return P.save("fig3")


if __name__ == "__main__":
    print(build())
