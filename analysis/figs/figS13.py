"""Fig S13: single-trial decoding of upcoming stay/switch and of left/right choice from DA/NE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st

FEAT_COLOR = {"DA": st.DA_COLOR, "NE": st.NE_COLOR, "DA+NE": "#58595b"}


def build():
    d = pd.read_parquet(config.CACHE / "decoding.parquet")
    d["ses_idx"] = d["subject_id"]
    P = st.Page(height=300)
    out = {"tasks": {}, "time": {}}
    titles = {"stay_switch": "Next-trial stay vs switch\n(features: baseline, go-cue and outcome responses)",
              "left_right": "Left vs right choice\n(pre-choice features only)"}
    for k, task in enumerate(("stay_switch", "left_right")):
        P.letter(6 + k * 200, 6, "ab"[k], None)
        ax = P.ax(40 + k * 200, 30, 160, 90)
        sub = d[d["task"] == task]
        x = 0
        ticks, labels = [], []
        for feats in ("DA", "NE", "DA+NE"):
            g = sub[sub["features"] == feats]
            if g.empty:
                continue
            for c, lab in (("ba_neural", "neural"), ("ba_neural_null_mean", "shuffle"), ("ba_behavior", "behavior"),
                           ("ba_neural_behavior", "neural+beh")):
                r = S.summarize_animals(g, c)
                out["tasks"][f"{task}_{feats}_{c}"] = r
                col = FEAT_COLOR[feats] if c.startswith("ba_neural") and c != "ba_neural_null_mean" else "#a7a9ac"
                ax.bar(x, r["mean"], color=col, width=0.8, lw=0, alpha=0.9 if c == "ba_neural" else 0.55)
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none",
                            ecolor="black", elinewidth=0.7)
                ticks.append(x)
                labels.append(lab)
                x += 1
            diff = g.assign(v=g["ba_neural_behavior"] - g["ba_behavior"])
            out["tasks"][f"{task}_{feats}_neural_gain_over_behavior"] = S.summarize_animals(diff, "v")
            above = g.assign(v=g["ba_neural"] - g["ba_neural_null_mean"])
            out["tasks"][f"{task}_{feats}_neural_minus_shuffle"] = S.summarize_animals(above, "v")
            out["tasks"][f"{task}_{feats}_frac_animals_p_perm_lt_0.05"] = float((g["p_perm"] < 0.05).mean())
            ax.text(x - 2.5, 0.98, feats, transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5,
                    color=FEAT_COLOR[feats])
            x += 0.7
        ax.axhline(0.5, color=st.INK, lw=0.5, ls=(0, (3, 2)))
        ax.set_ylim(0.4, max(0.8, ax.get_ylim()[1]))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=4.3, rotation=60)
        ax.set_ylabel("balanced accuracy (LOSO)")
        ax.set_title(titles[task], fontsize=5.6)

    P.letter(410, 6, "c", "When is next-trial switching decodable?")
    ax = P.ax(430, 30, 150, 90)
    tt = d[d["task"] == "stay_switch_time"]
    for feats, g in tt.groupby("features"):
        a = g.pivot_table(index="subject_id", columns="t", values="ba_neural")
        st.trace(ax, a.columns.to_numpy(), a.mean().to_numpy(), (a.std(ddof=1) / np.sqrt(len(a))).to_numpy(),
                 color=FEAT_COLOR[feats], label=f"{feats} (n={len(a)})")
        out["time"][feats] = {"peak_t": float(a.mean().idxmax()), "peak_ba": float(a.mean().max())}
    ax.axhline(0.5, color=st.INK, lw=0.5, ls=(0, (3, 2)))
    st.event_line(ax)
    ax.set_xlabel("Time from choice on trial t (s)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=4.5)

    P.letter(6, 175, "d", "Summary")
    t = out["tasks"]

    def g(k):
        return t.get(k, {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})

    lines = [f"Stay/switch: DA neural BA {st.fmt_ci(g('stay_switch_DA_ba_neural'))}, above shuffle by "
             f"{st.fmt_ci(g('stay_switch_DA_neural_minus_shuffle'))}; adding DA to behavior changes BA by "
             f"{st.fmt_ci(g('stay_switch_DA_neural_gain_over_behavior'))}.",
             f"NE neural BA {st.fmt_ci(g('stay_switch_NE_ba_neural'))}; gain over behavior {st.fmt_ci(g('stay_switch_NE_neural_gain_over_behavior'))}.",
             f"Left/right from pre-choice signals: DA {st.fmt_ci(g('left_right_DA_ba_neural'))}, NE {st.fmt_ci(g('left_right_NE_ba_neural'))}.",
             "Outcome-related signals predict the next choice mostly through the outcome itself (behavior-only model already "
             "captures it);",
             "single-trial DA/NE add little beyond reward and choice, cautioning against strong claims of single-trial "
             "policy read-out.",
             "Leave-one-session-out within animal; chance from within-session label permutation (100 per animal)."]
    P.text(40, 195, "\n".join(lines), fontsize=5.3, linespacing=1.6)
    cache.write_json(config.STATS / "figS13.json", out)
    return P.save("figS13")


if __name__ == "__main__":
    print(build())
