"""Reusable data panels drawn in the paper style."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis import config
from analysis.core import cache
from analysis.core.align import baseline_subtract, grouped_animal_traces, trial_mean_sem
from analysis.core.classify import split_rpe_slopes
from analysis.figs import style as st


def _grid():
    return cache.grids()["foraging"]


def _plot_slice(grid, lo=config.PLOT_T_PRE, hi=config.PLOT_T_POST):
    return (grid >= lo - 1e-9) & (grid <= hi + 1e-9)


def _mean_sem(rows, tr, mask, level):
    if level == "animal":
        return grouped_animal_traces(rows, tr, mask)
    return trial_mean_sem(tr, mask)


def q_outcome_panel(ax, rows: pd.DataFrame, level: str = "session", legend: bool = True,
                    ylabel: str = "Z-scored df/f", box: bool = True, min_n: int = 5):
    """Raw choice-aligned traces for R+/R- (solid/dashed) x Q_chosen tertile."""
    grid = _grid()
    sl = _plot_slice(grid)
    tr = cache.traces_for(rows)
    counts = {}
    if box:
        st.baseline_box(ax)
    for q in ("low", "mid", "high"):
        for rew, ls in ((True, "-"), (False, "--")):
            mask = ((rows["q_bin"] == q) & (rows["rewarded"] == rew)).to_numpy()
            counts[f"{q}_{'R+' if rew else 'R-'}"] = int(mask.sum())
            if mask.sum() < min_n:
                continue
            m, s, _ = _mean_sem(rows, tr, mask, level)
            st.trace(ax, grid[sl], m[sl], s[sl], color=st.Q_COLORS[q], ls=ls, lw=1.0, alpha=0.2)
    st.event_line(ax)
    st.time_axis(ax)
    ax.set_ylabel(ylabel)
    if legend:
        st.legend_q(ax)
    return counts


def rpe_panel(ax, rows: pd.DataFrame, level: str = "session", legend: bool = True,
              ylabel: str = "Z-scored df/f\n(baseline removed)", window: bool = True, min_n: int = 5):
    grid = _grid()
    sl = _plot_slice(grid)
    tr = baseline_subtract(cache.traces_for(rows), grid)
    counts = {}
    for b in st.RPE_ORDER[::-1]:
        mask = (rows["rpe_bin"] == b).to_numpy()
        counts[b] = int(mask.sum())
        if mask.sum() < min_n:
            continue
        m, s, _ = _mean_sem(rows, tr, mask, level)
        st.trace(ax, grid[sl], m[sl], s[sl], color=st.RPE_COLORS[b], lw=1.1, alpha=0.2)
    ax.axhline(0, color=st.INK, lw=0.5, ls=(0, (6, 4)), zorder=0)
    st.event_line(ax)
    if window:
        st.window_box(ax, 0.3, 1.0, "outcome")
    st.time_axis(ax)
    ax.set_ylabel(ylabel)
    if legend:
        st.legend_rpe(ax)
    return counts


def history_bars(ax, rows: pd.DataFrame, level: str = "session", col: str = "baseline",
                 ylabel: str = "Z-scored df/f AUC", label: str = "baseline"):
    """Mean baseline by signed run of consecutive past outcomes (-6..-1, 1..6)."""
    h = rows["streak_past"].clip(-config.HISTORY_MAX, config.HISTORY_MAX)
    xs = [k for k in range(-config.HISTORY_MAX, config.HISTORY_MAX + 1) if k != 0]
    means, errs = [], []
    for k in xs:
        sel = rows[h == k]
        if level == "animal":
            a = sel.groupby(["subject_id", "ses_idx"])[col].mean().groupby("subject_id").mean()
            means.append(a.mean())
            errs.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan)
        else:
            v = sel[col].dropna()
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
    colors = st.HIST_NEG + st.HIST_POS
    pos = np.arange(len(xs))
    ax.bar(pos, means, color=colors, width=0.85, lw=0)
    ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor="#58595b", elinewidth=1.2, capsize=0)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_xticks([0, 5, 6, 11])
    ax.set_xticklabels(["-6", "-1", "1", "6"])
    ax.set_xlabel("")
    ax.text(0.5, -0.16, "# Consecutive ", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.text(0.5, -0.16, "R-", transform=ax.transAxes, ha="left", va="top", fontsize=6, color=st.HIST_NEG[0])
    ax.text(0.585, -0.16, " / ", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax.text(0.66, -0.16, "R+", transform=ax.transAxes, ha="left", va="top", fontsize=6, color=st.HIST_POS[-1])
    ax.text(0.77, -0.16, " trials", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax.set_ylabel(ylabel)
    if label:
        ax.text(0.03, 0.98, label, transform=ax.transAxes, va="top", fontsize=5.5,
                bbox=dict(fc=st.BASELINE_BOX, ec="none", pad=1.2))
    return dict(zip([str(x) for x in xs], means))


def split_scatter(ax, rows: pd.DataFrame, col: str = "outcome_bs", ylabel: str = "Z-scored df/f AUC\n(baseline removed)",
                  label: str = "outcome", max_points: int = 600, seed: int = 0, s: float = 5):
    d = rows[["RPE_all", col]].dropna()
    fit = split_rpe_slopes(d["RPE_all"], d[col])
    show = d.sample(min(max_points, len(d)), random_state=seed) if len(d) > max_points else d
    neg, pos = show[show["RPE_all"] < 0], show[show["RPE_all"] >= 0]
    ax.scatter(neg["RPE_all"], neg[col], s=s, color=st.SCATTER, edgecolor="white", lw=0.25, zorder=2)
    ax.scatter(pos["RPE_all"], pos[col], s=s * 1.1, color=st.SCATTER, marker="x", lw=0.5, zorder=2)
    for dom, color, lo, hi in (("neg", st.FIT_NEG, -1, 0), ("pos", st.FIT_POS, 0, 1)):
        if np.isfinite(fit[f"slope_{dom}"]):
            xx = np.array([max(lo, d["RPE_all"].min()), min(hi, d["RPE_all"].max())])
            ax.plot(xx, fit[f"intercept_{dom}"] + fit[f"slope_{dom}"] * xx, color=color, lw=2.2, zorder=3)
    h = [Line2D([], [], color=st.FIT_NEG, lw=2), Line2D([], [], color=st.FIT_POS, lw=2)]
    leg = ax.legend(h, [f"RPE < 0: {fit['slope_neg']:.3f}", f"RPE >= 0: {fit['slope_pos']:.3f}"], loc="upper left",
                    title="LM fitted slopes", title_fontsize=4.5, fontsize=4.5, frameon=True, borderpad=0.3,
                    bbox_to_anchor=(0.06, 0.93))
    leg.get_frame().set_linewidth(0.4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("RPE")
    ax.set_ylabel(ylabel)
    if label:
        ax.text(0.03, 0.99, label, transform=ax.transAxes, va="top", fontsize=5.5,
                bbox=dict(fc=st.BASELINE_BOX, ec="none", pad=1.2))
    return fit


def session_rows(ses_idx: str, channel: str) -> pd.DataFrame:
    m = cache.for_meta()
    return m[(m["ses_idx"] == ses_idx) & (m["channel"] == channel) & m["valid"]]
