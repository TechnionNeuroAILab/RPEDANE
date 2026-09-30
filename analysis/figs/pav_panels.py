"""Pavlovian panels: CS-aligned traces per cue, example session trace, anticipatory-lick strip plot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core.align import grouped_animal_traces
from analysis.dataio import pavlovian as PV
from analysis.figs import style as st

CUES = ["CS1", "CS2", "CS3"]


def rows_for(signal: str, sessions=None) -> pd.DataFrame:
    m = cache.pav_meta()
    m = m[(m["signal"] == signal) & m["valid"] & m["p_reward"].notna()]
    if sessions is not None:
        m = m[m["ses_idx"].isin(sessions)]
    return m


def cue_traces(axes, rows: pd.DataFrame, kind: str = "dff", t_lo=-1.0, t_hi=5.0, ylim=None, ylabel="ΔF/F (%)"):
    grid = cache.grids()["pavlovian"]
    sl = (grid >= t_lo) & (grid <= t_hi)
    tr = cache.pav_traces_for(rows, kind)
    base = np.nanmean(tr[:, (grid >= -1) & (grid <= 0)], axis=1, keepdims=True)
    tr = tr - base
    out = {}
    for ax, cs in zip(axes, CUES):
        for water, ls in ((True, "-"), (False, (0, (4, 2)))):
            mask = ((rows["CS_type"] == cs) & (rows["water"] == water)).to_numpy()
            if mask.sum() < 3:
                continue
            m, s, n = grouped_animal_traces(rows, tr, mask)
            st.trace(ax, grid[sl], m[sl], s[sl], color=st.CS_COLORS[cs], ls=ls, lw=1.1, alpha=0.25)
            out[f"{cs}_{'rew' if water else 'omit'}_n_animals"] = n
        for x in (0, 2):
            ax.axvline(x, color="#808285", lw=0.5, ls=(0, (6, 4)), ymax=0.85)
        ax.axhline(0, color="#808285", lw=0.5, ls=(0, (6, 4)))
        ax.set_xlim(t_lo, t_hi)
        ax.set_xticks([0, 2])
        ax.set_xlabel("Time – CS (s)")
        _cue_header(ax, cs)
    if ylim is None:
        lims = [a.get_ylim() for a in axes]
        ylim = (min(l[0] for l in lims), max(l[1] for l in lims) * 1.12)
    for i, ax in enumerate(axes):
        ax.set_ylim(*ylim)
        if i:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel)
    return out


def _cue_header(ax, cs):
    """Colored probability label plus the CS / Delay / US bar."""
    from matplotlib.patches import Rectangle
    tr = ax.get_xaxis_transform()
    ax.text(0.0, 1.07, "♪", transform=tr, color=st.CS_COLORS[cs], fontsize=8, ha="right", va="center",
            fontfamily="DejaVu Sans")
    ax.text(0.05, 1.07, st.CS_LABEL[cs], transform=tr, color=st.INK, fontsize=6.5, va="center")
    ax.text(2.2, 1.07, "R or Ø", transform=tr, fontsize=4.8, va="center", color=st.INK)
    for x0, w, fc in ((0, 1, "#bcbec0"), (1, 1, "white"), (2, 0.5, "#bcbec0")):
        ax.add_patch(Rectangle((x0, 0.93), w, 0.06, transform=tr, fc=fc, ec=st.INK, lw=0.4, clip_on=False))


def lick_strip(ax, lick_ses: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i, cs in enumerate(CUES):
        v = lick_ses[f"lick_{cs}"].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)) - 0.12, v, s=4, color=st.CS_COLORS[cs], alpha=0.45, lw=0)
        ax.errorbar(i + 0.18, v.mean(), yerr=v.std(ddof=1), fmt="o", color="black", ms=3, elinewidth=1.6, capsize=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([st.CS_LABEL[c] for c in CUES])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Anticipatory lick #\n(per trial, session mean)")
    ax.set_ylim(bottom=0)


def example_trace(ax_ev, ax, ses_idx: str, channel: str, t_start: float, dur: float = 300.0, bar_pct=None):
    d = config.PAV_ROOT / ses_idx
    fip = PV.load_fip(d)[channel]
    trials = cache.pav_trials().query("ses_idx == @ses_idx")
    sel = (fip["t"] >= t_start) & (fip["t"] <= t_start + dur)
    t, y = fip["t"][sel] - t_start, fip["data"][sel] * 100
    g = trials[(trials["CS_start_time_in_session"] >= t_start) & (trials["CS_start_time_in_session"] <= t_start + dur - 3)]
    cs_t = g["CS_start_time_in_session"] - t_start
    rw_t = g.loc[g["water"], "US_start_time_in_session"] - t_start
    ax.vlines(cs_t, 0, 1, transform=ax.get_xaxis_transform(), color="#9d9fa2", lw=0.6)
    ax.vlines(rw_t, 0, 1, transform=ax.get_xaxis_transform(), color=st.EVENT["reward"], lw=0.8)
    ax.plot(t, y, color="black", lw=0.5)
    ax.set_xlim(0, dur)
    ax.axis("off")
    rng = np.nanpercentile(y, 99.7) - np.nanpercentile(y, 0.3)
    bar = bar_pct or st.nice_bar(rng)
    y0 = np.nanpercentile(y, 60)
    ax.plot([-4, -4], [y0, y0 + bar], color=st.INK, lw=1, clip_on=False)
    ax.text(-6, y0 + bar / 2, f"ΔF/F\n{bar:.0f}%", ha="right", va="center", fontsize=5.5)
    btr = ax.get_xaxis_transform()
    ax.plot([dur - 50, dur], [-0.12, -0.12], color=st.INK, lw=1, clip_on=False, transform=btr)
    ax.text(dur - 25, -0.16, "50 s", ha="center", va="top", fontsize=6, transform=btr)
    for cs in CUES + ["CS4"]:
        xs = g.loc[g["CS_type"] == cs, "CS_start_time_in_session"] - t_start
        ax_ev.scatter(xs, np.full(len(xs), 0.0), s=2.5, marker="|", color=st.CS_COLORS[cs], lw=0.9)
    ax_ev.scatter(rw_t, np.full(len(rw_t), 1.0), s=2.5, marker="s", color=st.EVENT["reward"], lw=0)
    ax_ev.set_xlim(0, dur)
    ax_ev.set_ylim(-0.6, 1.6)
    ax_ev.axis("off")
    return {"ses_idx": ses_idx, "channel": channel, "t_start": t_start, "duration": dur, "scale_pct": bar}


def best_window(ses_idx: str, dur: float = 300.0) -> float:
    t = cache.pav_trials().query("ses_idx == @ses_idx")
    cs = t["CS_start_time_in_session"].to_numpy()
    rw = t.loc[t["water"], "US_start_time_in_session"].to_numpy()
    starts = np.arange(cs.min(), cs.max() - dur, 10)
    score = [np.sum((rw >= s) & (rw < s + dur)) - 5 * np.sum(t["is_airpuff_cs"].to_numpy() & (cs >= s) & (cs < s + dur))
             for s in starts]
    return float(starts[int(np.argmax(score))])
