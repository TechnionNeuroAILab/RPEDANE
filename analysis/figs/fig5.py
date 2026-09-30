"""Fig 5: simultaneous NAc DA / PL NE recordings and the data-driven task x system coding matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.dataio import foraging as F
from analysis.figs import panels
from analysis.figs import style as st

SRC = "figure5_DANE.pdf"


def simultaneous_sessions() -> set:
    m = cache.for_meta()
    return set(m.loc[(m["signal"] == "NE") & m["session_ok"], "ses_idx"])


def column_rows(sim: set) -> list[tuple[str, pd.DataFrame]]:
    m = cache.for_meta()
    m = m[m["ses_idx"].isin(sim) & m["valid"] & m["session_ok"]]
    return [("NAc (L)", m[(m["signal"] == "DA") & (m["hemi"] == "L")]),
            ("NAc (R)", m[(m["signal"] == "DA") & (m["hemi"] == "R")]),
            ("LC axon in PL", m[m["signal"] == "NE"])]


def panel_b(P: st.Page, x0, y0, w, h):
    m = cache.for_meta()
    trip = m.groupby("ses_idx")["channel"].apply(lambda c: {"latNAcc(L)-DA", "latNAcc(R)-DA", "PL(L)-LCAxonCa"} <= set(c))
    ses = sorted(trip[trip].index)[0]
    fip = F.load_fip(config.FORAGING_ROOT / ses.split("_")[0] / ses)
    trials = cache.for_trials().query("ses_idx == @ses")
    dur = 130.0
    go_all = trials["goCue_start_time_in_session"].to_numpy()
    rw_all = trials.loc[trials["rewarded"], "choice_time_in_session"].to_numpy()
    starts = np.arange(go_all.min(), go_all.max() - dur, 5)
    t0 = float(starts[np.argmax([np.sum((rw_all >= s) & (rw_all < s + dur)) for s in starts])])
    names = [("latNAcc(L)-DA", "NAc (L)"), ("latNAcc(R)-DA", "NAc (R)"), ("PL(L)-LCAxonCa", "LC axon in PL")]
    rh = h / 3
    info = {"ses_idx": ses, "t_start": t0, "duration": dur, "scale_pct": {}}
    for i, (ch, lab) in enumerate(names):
        ax = P.ax(x0, y0 + i * rh + 6, w, rh - 8)
        r = fip[ch]
        sel = (r["t"] >= t0) & (r["t"] <= t0 + dur)
        y = r["data"][sel] * 100
        tt = r["t"][sel] - t0
        go = go_all[(go_all >= t0) & (go_all <= t0 + dur)] - t0
        ax.vlines(go, 0, 1, transform=ax.get_xaxis_transform(), color="#bcbec0", lw=1.2)
        ax.plot(tt, y, color=st.TRACE_GREEN, lw=0.8)
        if i == 0:
            rw = rw_all[(rw_all >= t0) & (rw_all <= t0 + dur)] - t0
            ax.scatter(rw, np.full(len(rw), 1.08), transform=ax.get_xaxis_transform(), marker="s", s=3,
                       color=st.EVENT["reward"], clip_on=False)
        ax.set_xlim(0, dur)
        ax.axis("off")
        ax.text(0, 1.12 if i else 1.28, lab, transform=ax.transAxes, fontsize=6.5, va="bottom")
        span = np.nanpercentile(y, 99.5) - np.nanpercentile(y, 0.5)
        bar = st.nice_bar(span)
        base = np.nanpercentile(y, 50)
        ax.plot([dur + 1.5] * 2, [base, base + bar], color=st.INK, lw=1, clip_on=False)
        ax.text(dur + 3, base + bar / 2, f"{bar:.0f}%" if i else f"ΔF/F\n{bar:.0f}%", fontsize=5.5, va="center")
        info["scale_pct"][lab] = bar
        if i == 2:
            ax.plot([dur - 25, dur], [-0.15] * 2, transform=ax.get_xaxis_transform(), color=st.INK, lw=1, clip_on=False)
            ax.text(dur - 12.5, -0.2, "25 s", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
    return info


def coding_matrix(P: st.Page, x0, y0, w, h, cells: dict):
    ax = P.ax(x0, y0, w, h)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.invert_yaxis()

    def box(x, y, fc, text, sub=None, alpha=1.0, fs=6.5):
        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.06), 0.9, 0.88, boxstyle="square,pad=0", fc=fc, ec="none", alpha=alpha))
        ax.text(x + 0.5, y + (0.38 if sub else 0.5), text, ha="center", va="center", fontsize=fs, color=st.INK)
        if sub:
            ax.text(x + 0.5, y + 0.72, sub, ha="center", va="center", fontsize=4.2, color=st.INK)

    ax.plot([0.08, 0.95], [0.12, 0.92], color=st.INK, lw=0.9)
    ax.text(0.78, 0.35, "Task", fontsize=6.5, ha="center")
    ax.text(0.35, 0.78, "System", fontsize=6.5, ha="center")
    box(1, 0, st.BOX_COLORS["task_forage"], "Dynamic\nForaging")
    box(2, 0, st.BOX_COLORS["task_pav"], "Pavlovian\nConditioning")
    box(0, 1, st.BOX_COLORS["sys_da"], "Dopamine\nNAc")
    box(0, 2, st.BOX_COLORS["sys_ne"], "Norepinephrine\nPL")
    colors = {"Mixed\nValue/RPE": "#cf8bb0", "RPE": "#b877c0", "Mixed\nresponse": "#ead3e3", "Value": "#e4a5a5",
              "Weak": "#e6e7e8"}
    for (r, c), cell in cells.items():
        v, p = cell["value"], cell["rpe"]
        sub = (f"{cell['value_name']} {v['mean']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]\n"
               f"RPE {p['mean']:.2f} [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}]\nn = {p['n_animals']} mice")
        x, y = c + 1, r + 1
        box(x, y, colors[cell["label"]], cell["label"])
        ax.texts[-1].set_position((x + 0.5, y + 0.33))
        ha = "left" if c == 0 else "right"
        ax.text(x + (0.1 if c == 0 else 0.9), y + (0.72 if r == 0 else 0.72), sub, ha=ha, va="center",
                fontsize=3.9, color=st.INK, linespacing=1.15)
    ax.add_patch(Circle((2.0, 2.0), 0.26, fc="white", ec=st.INK, lw=0.8, zorder=5))
    ax.text(2.0, 2.0, "Coding\nspace", ha="center", va="center", fontsize=6, zorder=6)
    ax.text(0.05, 3.05, "cell label: animal-bootstrap 95% CI excluding 0\nvalue = tonic history r (foraging) / "
            "cue r (Pavlovian); RPE = split slope", fontsize=3.8, va="top", color="#58595b")


def classify_cell(task: str, value: dict, rpe: dict) -> str:
    v = value["ci_lo"] > 0
    r = rpe["ci_lo"] > 0
    if task == "foraging":
        return "Mixed\nValue/RPE" if (v and r) else "RPE" if r else "Value" if v else "Weak"
    return "RPE" if (v and r) else "Mixed\nresponse"


def build():
    s = pd.read_csv(config.TABLES / "coding_summary.csv", dtype={"subject_id": str})
    pav = pd.read_csv(config.TABLES / "pavlovian_summary.csv", dtype={"subject_id": str})
    sim = simultaneous_sessions()
    P = st.Page(height=292)
    P.letter(6, 4, "a", "Simultaneous fiber photometry\n      NE/DA recordings", title_size=8)
    P.snippet(SRC, (30, 25, 125, 92), (18, 34, 110, 70))
    P.letter(145, 4, "b")
    b_info = panel_b(P, 165, 8, 360, 92)
    from matplotlib.patches import Patch
    P.fig.legend(handles=[Patch(color=st.EVENT["reward"], label="Reward"), Patch(color="#bcbec0", label="Go cue")],
                 loc="upper left", bbox_to_anchor=(545 / P.width, 1 - 14 / P.height), fontsize=5.5, handlelength=0.8)

    P.letter(6, 116, "c", "NE and DA responses")
    counts = {}
    for i, (lab, rows) in enumerate(column_rows(sim)):
        x = 38 + i * 125
        a1 = P.ax(x, 140, 105, 58)
        c1 = panels.q_outcome_panel(a1, rows, level="animal", legend=(i == 0), ylabel="Z-scored df/f" if i == 0 else "",
                                    box=False)
        a1.set_xlabel("")
        a1.set_title(lab, fontsize=7, pad=2)
        a2 = P.ax(x, 214, 105, 58)
        c2 = panels.rpe_panel(a2, rows, level="animal", legend=(i == 0), window=False,
                              ylabel="Z-scored df/f\n(baseline removed)" if i == 0 else "")
        counts[lab] = {"n_animals": int(rows["subject_id"].nunique()), "n_sessions": int(rows["ses_idx"].nunique()),
                       "qxoutcome": c1, "rpe_bins": c2}

    P.letter(420, 116, "d", "Joint NE and DA coding space")
    da_sim = s[(s["signal"] == "DA") & s["ses_idx"].isin(sim)]
    ne = s[s["signal"] == "NE"]
    pda, pne = pav[pav["signal"] == "DA"], pav[pav["signal"] == "NE"]
    cells = {}
    for (r, c, task, d_val, vname, vcol) in ((0, 0, "foraging", da_sim, "value", "r_history"),
                                             (1, 0, "foraging", ne, "value", "r_history"),
                                             (0, 1, "pavlovian", pda, "cue", "r_cue_value"),
                                             (1, 1, "pavlovian", pne, "cue", "r_cue_value")):
        value = S.summarize_animals(d_val, vcol)
        rpe = S.summarize_animals(d_val, "rpe_split")
        cells[(r, c)] = {"task": task, "system": ["DA", "NE"][r], "value_name": vname, "value": value, "rpe": rpe,
                         "label": classify_cell(task, value, rpe)}
    coding_matrix(P, 425, 136, 165, 145, cells)

    stats = {"panel_b": b_info, "panel_c": counts, "simultaneous_sessions": sorted(sim),
             "matrix": {f"{v['system']}_{v['task']}": {k: v[k] for k in ("label", "value", "rpe", "value_name")}
                        for v in cells.values()},
             "definitions": {"rpe_split": "mean of RPE<0 and RPE>=0 split slopes of z-scored outcome response per unit "
                                          "RPE; Pavlovian RPE = 1-p (reward) or -p (omission)",
                             "value_foraging": "r(baseline, signed run of past outcomes)",
                             "value_pavlovian": "r(CS response, P(reward))",
                             "label_rule": "CI (animal bootstrap) excluding 0"}}
    cache.write_json(config.STATS / "fig5.json", stats)
    return P.save("fig5")


if __name__ == "__main__":
    print(build())
