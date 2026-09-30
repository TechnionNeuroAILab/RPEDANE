"""Fig 1: foraging task, behavior, Q-learning fit, NAc DA recordings, average DA by value x outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.dataio import foraging as F
from analysis.figs import panels
from analysis.figs import style as st

SRC = "figure1_DANE.pdf"


def pick_behavior_session() -> str:
    t = cache.for_trials()
    rows = []
    meta = cache.for_meta()
    lat_ses = set(meta.loc[meta["region"] == config.DA_MAIN_REGION, "ses_idx"])
    for ses, g in t.groupby("ses_idx"):
        g = g.iloc[:520]
        r = g[g["responded"]]
        if len(g) < 450 or ses not in lat_ses:
            continue
        blocks = int((g["reward_probabilityR"].diff().fillna(0) != 0).sum())
        smooth = r["choice_right"].rolling(9, center=True, min_periods=3).mean()
        fit = np.corrcoef(smooth, r["R_prob"])[0, 1]
        rows.append({"ses_idx": ses, "blocks": blocks, "fit": fit, "ignore_rate": g["ignored"].mean(),
                     "p_right": r["choice_right"].mean()})
    d = pd.DataFrame(rows)
    d = d[(d["blocks"] >= 10) & (d["ignore_rate"] < 0.1) & d["p_right"].between(0.3, 0.7)]
    return str(d.sort_values("fit", ascending=False).iloc[0]["ses_idx"])


def panel_b(ax, g: pd.DataFrame, n_show: int):
    g = g.iloc[:n_show]
    x = np.arange(len(g))
    pr, pl = g["reward_probabilityR"].to_numpy(), g["reward_probabilityL"].to_numpy()
    ax.fill_between(x, 0, pr, step="mid", color=st.PROB_R, lw=0, alpha=0.8)
    ax.fill_between(x, 0, -pl, step="mid", color=st.PROB_L, lw=0, alpha=0.8)
    for side, y0 in ((1.0, 1.15), (0.0, -1.15)):
        sel = g["choice_right"].eq(side).to_numpy()
        rew = g["rewarded"].to_numpy()
        sgn = 1 if side == 1.0 else -1
        ax.vlines(x[sel & rew], y0, y0 + sgn * 0.28, color="black", lw=0.5)
        ax.vlines(x[sel & ~rew], y0, y0 + sgn * 0.28, color="#bcbec0", lw=0.5)
    ign = g["ignored"].to_numpy()
    ax.vlines(x[ign], -0.12, 0.12, color=st.EVENT["ignored"], lw=0.9)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_xlim(-2, len(g) + 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["1", "0", "1"])
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.text(-0.012, 0.73, "$P(R|r)$", transform=ax.transAxes, ha="right", fontsize=4.5, rotation=0)
    ax.text(-0.012, 0.22, "$P(R|l)$", transform=ax.transAxes, ha="right", fontsize=4.5)
    ax.text(0.0, 1.02, "Right choice", transform=ax.transAxes, fontsize=6)
    ax.text(0.0, -0.02, "Left choice", transform=ax.transAxes, fontsize=6, va="top")
    ax.plot([len(g) - 150, len(g)], [-1.62, -1.62], color=st.INK, lw=1, clip_on=False)
    ax.text(len(g) - 75, -1.66, "150 trials", ha="center", va="top", fontsize=6)


def panel_d(ax, g: pd.DataFrame, n_show: int):
    g = g.iloc[:n_show]
    x = np.arange(len(g))
    ch = g["choice_right"].where(g["responded"])
    smooth = ch.rolling(9, center=True, min_periods=3).mean().interpolate(limit_direction="both")
    ax.plot(x, smooth, color=st.SCATTER, lw=1.3, label="Smoothed mouse\nchoice history")
    ax.plot(x, g["R_prob"].interpolate(limit_direction="both"), color="black", lw=1.0, ls=(0, (3, 1.5)),
            label="Single-state\nQ-learning model")
    ax.set_xlim(-2, len(g) + 2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 1])
    ax.set_ylabel("$P(c(t)=r)$", fontsize=5)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.text(0.0, 1.02, "Right choice", transform=ax.transAxes, fontsize=6)
    ax.text(0.0, -0.02, "Left choice", transform=ax.transAxes, fontsize=6, va="top")
    return float(np.corrcoef(smooth, g["R_prob"].interpolate(limit_direction="both"))[0, 1])


def panel_e(ax_ev, ax, ses: str, channel: str, t_start: float, dur: float = 260.0):
    subj = ses.split("_")[0]
    d = config.FORAGING_ROOT / subj / ses
    fip = F.load_fip(d)[channel]
    trials = cache.for_trials().query("ses_idx == @ses")
    sel = (fip["t"] >= t_start) & (fip["t"] <= t_start + dur)
    t, y = fip["t"][sel] - t_start, fip["data"][sel] * 100
    g = trials[(trials["goCue_start_time_in_session"] >= t_start) & (trials["goCue_start_time_in_session"] <= t_start + dur)]
    go = g["goCue_start_time_in_session"] - t_start
    rw = g.loc[g["rewarded"], "choice_time_in_session"] - t_start
    ax.vlines(go, 0, 1, transform=ax.get_xaxis_transform(), color=st.EVENT["gocue"], lw=0.5, alpha=0.8)
    ax.vlines(rw, 0, 1, transform=ax.get_xaxis_transform(), color=st.EVENT["reward"], lw=0.6)
    ax.plot(t, y, color="black", lw=0.45)
    ax.set_xlim(0, dur)
    ax.axis("off")
    rng = np.nanpercentile(y, 99.5) - np.nanpercentile(y, 0.5)
    bar = st.nice_bar(rng)
    ax.plot([dur + 3, dur + 3], [np.nanpercentile(y, 50), np.nanpercentile(y, 50) + bar], color=st.INK, lw=1, clip_on=False)
    ax.text(dur + 6, np.nanpercentile(y, 50) + bar / 2, f"ΔF/F\n{bar:.0f}%", va="center", fontsize=6)
    btr = ax.get_xaxis_transform()
    ax.plot([dur - 50, dur], [-0.05, -0.05], color=st.INK, lw=1, clip_on=False, transform=btr)
    ax.text(dur - 25, -0.08, "50 s", ha="center", va="top", fontsize=6, transform=btr)
    rows = [("reward", rw), ("right", g.loc[g["choice_right"] == 1, "choice_time_in_session"] - t_start),
            ("left", g.loc[g["choice_right"] == 0, "choice_time_in_session"] - t_start), ("gocue", go)]
    for i, (k, xs) in enumerate(rows):
        ax_ev.scatter(xs, np.full(len(xs), 3 - i), s=1.2, marker="s", color=st.EVENT[k], lw=0)
    ax_ev.set_xlim(0, dur)
    ax_ev.set_ylim(-0.5, 3.5)
    ax_ev.axis("off")
    return {"t_start": t_start, "duration": dur, "scale_pct": bar}


def build():
    ses = pick_behavior_session()
    trials = cache.for_trials()
    g = trials[trials["ses_idx"] == ses].reset_index(drop=True)
    n_show = min(len(g), 520)
    P = st.Page(height=300)

    P.letter(6, 6, "a", "Animal behavior")
    P.snippet(SRC, (20, 22, 137, 95), (20, 24, 117, 73))
    P.letter(150, 6, "b")
    ax_b = P.ax(175, 22, 340, 60)
    panel_b(ax_b, g, n_show)
    lg = [Patch(color=st.PROB_R, label="R  Reward"), Patch(color=st.PROB_L, label="L  Probabilities"),
          Line2D([], [], color="black", lw=2, label="Rewarded choice"),
          Line2D([], [], color="#bcbec0", lw=2, label="Unrewarded choice"),
          Line2D([], [], color=st.EVENT["ignored"], lw=2, label="Ignored trials")]
    P.fig.legend(handles=lg, loc="upper left", bbox_to_anchor=(522 / P.width, 1 - 24 / P.height), fontsize=5.5,
                 handlelength=0.8, labelspacing=0.3)

    P.letter(6, 98, "c", "Q-learning model")
    P.snippet(SRC, (5, 110, 157, 170), (5, 112, 150, 60))
    P.letter(150, 98, "d")
    ax_d = P.ax(175, 112, 340, 50)
    fit_r = panel_d(ax_d, g, n_show)
    ax_d.legend(loc="upper left", bbox_to_anchor=(1.01, 1.05), fontsize=5.5, handlelength=1.8, labelspacing=0.8)

    P.letter(6, 188, "e", "Recordings")
    P.snippet(SRC, (20, 205, 118, 275), (18, 206, 98, 72))
    lat = [c for c in cache.for_meta().query("ses_idx == @ses")["channel"].unique() if c.startswith("latNAcc")]
    channel = sorted(lat)[0]
    t_rew = g.loc[g["rewarded"], "choice_time_in_session"].to_numpy()
    starts = np.arange(g["goCue_start_time_in_session"].min(), g["goCue_start_time_in_session"].max() - 260, 20)
    t_start = float(starts[np.argmax([np.sum((t_rew >= s) & (t_rew < s + 260)) for s in starts])])
    ax_ev = P.ax(125, 188, 200, 20)
    ax_e = P.ax(125, 208, 200, 72)
    e_info = panel_e(ax_ev, ax_e, ses, channel, t_start)
    lg = [Patch(color=st.EVENT[k], label=lab) for k, lab in
          (("reward", "Reward"), ("right", "Right choice"), ("left", "Left choice"), ("gocue", "Go cue"))]
    P.fig.legend(handles=lg, loc="upper left", bbox_to_anchor=(340 / P.width, 1 - 186 / P.height), fontsize=5.5,
                 handlelength=0.8, labelspacing=0.3)

    P.letter(393, 186, "f", "Average DA release")
    P.snippet(SRC, (397, 205, 470, 272), (393, 206, 73, 67))
    ax_f = P.ax(492, 206, 98, 70)
    rows = cache.foraging_fibers("DA", config.DA_MAIN_REGION)
    counts = panels.q_outcome_panel(ax_f, rows, level="animal", legend=True)

    per_fiber = rows.groupby(["subject_id", "ses_idx", "channel"]).apply(
        lambda d: S.ols(d["baseline"], {"q": d["Q_chosen"]})["q"] if len(d) > 30 else np.nan).rename("b").reset_index()
    base_q = S.summarize_animals(per_fiber, "b")
    t = trials[trials["responded"]]
    better = t.assign(better=np.where(t["reward_probabilityR"] > t["reward_probabilityL"], t["choice_right"] == 1,
                                      np.where(t["reward_probabilityR"] < t["reward_probabilityL"], t["choice_right"] == 0,
                                               np.nan)).astype(float))
    better = better.dropna(subset=["better"]).groupby(["subject_id", "ses_idx"], as_index=False)["better"].mean()
    stats = {"example_session": ses, "example_channel": channel, "n_trials_shown": n_show,
             "model_vs_smoothed_choice_r": fit_r, "trace_window": e_info, "qxoutcome_trial_counts": counts,
             "cohort": {"n_animals": int(rows["subject_id"].nunique()), "n_sessions": int(rows["ses_idx"].nunique()),
                        "n_fiber_sessions": int(rows.groupby(["ses_idx", "channel"]).ngroups)},
             "baseline_vs_Qchosen_slope": base_q,
             "p_choose_better_side": S.summarize_animals(better, "better", null=0.5)}
    cache.write_json(config.STATS / "fig1.json", stats)
    return P.save("fig1")


if __name__ == "__main__":
    print(build())
