"""Fig 4: PL LC-NE axons encode action-outcome RPE in foraging but not canonical Pavlovian RPE."""
from __future__ import annotations

import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import panels
from analysis.figs import pav_panels as pp
from analysis.figs import style as st
from analysis.models import coding

SRC = "figure4_DANE.pdf"


def build():
    s = pd.read_csv(config.TABLES / "coding_summary.csv", dtype={"subject_id": str})
    pav = pd.read_csv(config.TABLES / "pavlovian_summary.csv", dtype={"subject_id": str})
    licks = pd.read_csv(config.TABLES / "pavlovian_licks.csv", dtype={"subject_id": str})
    P = st.Page(height=240)
    P.letter(6, 4, "a", "Fiber photometry NE recordings")
    P.snippet(SRC, (10, 28, 135, 100), (10, 24, 125, 72))
    P.letter(150, 4, "b", "NE response Dynamic foraging task")
    rows = cache.foraging_fibers("NE")
    ax_b = P.ax(180, 24, 105, 70)
    counts = panels.rpe_panel(ax_b, rows, level="animal", legend=True)

    P.letter(305, 4, "c", "RPE coding")
    ne = s[s["signal"] == "NE"]
    ex = ne[ne["n_trials"] >= 250].assign(score=lambda d: d[["slope_pos", "slope_neg"]].min(axis=1))
    ex = ex.sort_values("score", ascending=False).iloc[0]
    ax_c = P.ax(335, 24, 90, 70)
    fit = panels.split_scatter(ax_c, panels.session_rows(ex["ses_idx"], ex["channel"]), label=None, s=4)
    ax_c.text(1.0, -0.32, f"{ex['ses_idx']}", transform=ax_c.transAxes, ha="right", fontsize=4, color="#808285")
    t_pos = S.summarize_animals(ne, "slope_pos")
    t_neg = S.summarize_animals(ne, "slope_neg")
    P.text(440, 28, "Cohort (n = %d mice, %d sessions)\nRPE ≥ 0 slope: %s\n  %s\nRPE < 0 slope: %s\n  %s\ngraded RPE (partial r): %s"
           % (t_pos["n_animals"], t_pos["n_sessions"], st.fmt_ci(t_pos), st.fmt_p(t_pos["p_signflip"]),
              st.fmt_ci(t_neg), st.fmt_p(t_neg["p_signflip"]), st.fmt_ci(S.summarize_animals(ne, "rpe_graded"))),
           fontsize=5.2)

    P.letter(6, 108, "d", "NE response Pavlovian Task")
    P.snippet(SRC, (10, 140, 160, 225), (8, 134, 150, 85))
    rows_p = pp.rows_for("NE")
    axes = [P.ax(185 + i * 78, 142, 72, 62) for i in range(3)]
    pcounts = pp.cue_traces(axes, rows_p)
    ax_l = P.ax(470, 132, 80, 72)
    ne_ses = rows_p["ses_idx"].unique()
    pp.lick_strip(ax_l, licks[licks["ses_idx"].isin(ne_ses)])
    ax_l.set_title("Anticipatory licks\n(NE sessions)", fontsize=6)

    pne = pav[pav["signal"] == "NE"]
    cs_sd = pne["dff_cs_slope"].std()
    stats = {"foraging": {"n_animals": int(rows["subject_id"].nunique()), "n_sessions": int(rows["ses_idx"].nunique()),
                          "rpe_bin_trial_counts": counts, "slope_pos": t_pos, "slope_neg": t_neg,
                          "rpe_graded": S.summarize_animals(ne, "rpe_graded"),
                          "r_history": S.summarize_animals(ne, "r_history"),
                          "slope_pos_leave_one_animal_out": S.leave_one_animal_out(ne, "slope_pos"),
                          "example": {"ses_idx": ex["ses_idx"], "channel": ex["channel"], **fit}},
             "pavlovian": {"n_animals": int(rows_p["subject_id"].nunique()), "n_sessions": int(rows_p["ses_idx"].nunique()),
                           "trace_n_animals": pcounts,
                           **{k: S.summarize_animals(pne, k) for k in
                              ("dff_cs_slope", "dff_rew_slope", "dff_om_slope", "rpe_graded", "r_cue_value")},
                           "fraction_canonical_sessions": float(pne["canonical"].mean()),
                           "tost_rpe_graded_bound_0.1": S.tost_mean(
                               S.animal_values(pne, "rpe_graded").to_numpy(), 0.1),
                           "note": "n=4 animals: minimum exact sign-flip p = 0.125; absence claim rests on CIs/TOST",
                           "cs_slope_sd": cs_sd}}
    cache.write_json(config.STATS / "fig4.json", stats)
    return P.save("fig4")


if __name__ == "__main__":
    print(build())
