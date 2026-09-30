"""Fig S6: left/right side bias and perseveration, and whether choice side matters for the neural signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st
from analysis.models import coding
from analysis.models import preference as PR

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]


def build():
    trials = cache.for_trials()
    pf = PR.fit_all(trials, cache.for_sess())
    pf.to_csv(config.TABLES / "side_bias_sessions.csv", index=False)
    animals = sorted(pf["subject_id"].unique())
    colors = st.animal_colors(animals)
    P = st.Page(height=470)
    out = {"agreement": {}, "cohort": {}, "per_animal": {}, "neural": {}, "robustness": {}}

    P.letter(6, 6, "a", "Bias estimates (+ = rightward)")
    pairs = [("b_side", "bias_R_model", "logistic b_side", "−biasL (Q-model fit)"),
             ("b_side", "side_bias_upstream", "logistic b_side", "upstream side_bias"),
             ("p_right", "side_bias_upstream", "P(right)", "upstream side_bias")]
    for i, (x, y, xl, yl) in enumerate(pairs):
        ax = P.ax(40 + i * 110, 30, 80, 70)
        ax.scatter(pf[x], pf[y], s=4, c=[colors[a] for a in pf["subject_id"]], lw=0)
        r = S.pearson(pf[x], pf[y])
        out["agreement"][f"{x}_vs_{y}"] = {"r": r[0], "p": r[1], "n_sessions": r[2]}
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        st.stat_text(ax, f"r = {r[0]:.2f}")
        if i == 0:
            ax.set_title("same likelihood given Q:\nconsistency check", fontsize=4.8)
        else:
            ax.set_title("independent estimate", fontsize=4.8)

    P.letter(360, 6, "b", "Bias across sessions")
    ax = P.ax(385, 30, 195, 70)
    for a in animals:
        d = pf[pf["subject_id"] == a].sort_values("day")
        ax.errorbar(d["day"], d["b_side"], yerr=[d["b_side"] - d["b_side_lo"], d["b_side_hi"] - d["b_side"]],
                    color=colors[a], lw=0.6, marker="o", ms=1.8, elinewidth=0.4, label=a[-3:])
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Session # (within animal)")
    ax.set_ylabel("b_side (logit, +R)")
    ax.legend(fontsize=4, ncol=4, loc="upper right", handlelength=0.8, columnspacing=0.6)

    P.letter(6, 118, "c", "Animal-level bias, perseveration and value sensitivity")
    for i, (col, lab, null) in enumerate((("b_side", "b_side (+R)", 0.0), ("p_right", "P(right)", 0.5),
                                          ("rho", "perseveration ρ", 0.0), ("beta_Q", "β_Q (Q_R − Q_L)", 0.0))):
        ax = P.ax(40 + i * 90, 142, 65, 70)
        d = pf[pf["identifiable"]]
        r = S.summarize_animals(d, col, null=null)
        out["cohort"][col] = r
        for k, (a, v) in enumerate(r["per_animal"].items()):
            ax.scatter(k * 0.08 - 0.3, v, s=8, color=colors[a], lw=0)
        ax.errorbar(0.2, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", color="black",
                    ms=3, elinewidth=1)
        ax.axhline(null, color="#808285", lw=0.5, ls=(0, (3, 2)))
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_title(lab, fontsize=6)
        st.stat_text(ax, st.fmt_p(r["p_signflip"]), y=1.0)
    out["n_sessions"] = int(len(pf))
    out["frac_identifiable"] = float(pf["identifiable"].mean())
    out["frac_stubborn"] = float(pf["stubborn"].mean())
    out["frac_sessions_bias_ci_excludes_0"] = float(((pf["b_side_lo"] > 0) | (pf["b_side_hi"] < 0)).mean())

    P.letter(400, 118, "d", "Choice vs value difference")
    ax = P.ax(425, 142, 155, 70)
    ps = PR.psychometric(trials)
    for a in animals:
        d = ps[ps["subject_id"] == a]
        ax.plot(d["qc"], d["choice_right"], color=colors[a], lw=0.7, marker="o", ms=1.5)
    ax.axhline(0.5, color="#808285", lw=0.4, ls=(0, (3, 2)))
    ax.axvline(0, color="#808285", lw=0.4, ls=(0, (3, 2)))
    ax.set_xlabel("Q_R − Q_L")
    ax.set_ylabel("P(right)")

    P.letter(6, 232, "e", "Does choice side (relative to the fiber) change the signals?")
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    hs = PR.hemisphere_side_interaction(m)
    metrics = [("outcome_bs_contra_minus_ipsi_R", "outcome R+"), ("outcome_bs_contra_minus_ipsi_U", "outcome R−"),
               ("baseline_contra_minus_ipsi", "baseline"), ("gocue_resp_contra_minus_ipsi", "go cue"),
               ("rpe_x_contra", "RPE slope ×\ncontra")]
    ax = P.ax(40, 256, 300, 75)
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = hs[(hs["signal"] == sig) & (hs["region"] == reg)]
        for i, (c, name) in enumerate(metrics):
            r = S.summarize_animals(d, c)
            out["neural"][f"{sig}_{c}"] = r
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=3, color=col,
                        elinewidth=0.9, label=lab if i == 0 else None)
            ax.text(x, r["ci_hi"], st.fmt_p(r["p_signflip"]).replace("p = ", "").replace("p < ", "<"), fontsize=3.8,
                    ha="center", va="bottom")
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([n for _, n in metrics], fontsize=5)
    ax.set_ylabel("contra − ipsi (z)")
    ax.legend(fontsize=4.8)

    P.letter(360, 232, "f", "Coding indices after excluding biased sessions")
    s = coding.foraging_summary()
    biased = set(pf.loc[(pf["b_side"].abs() > 1) | pf["stubborn"], "ses_idx"])
    ax = P.ax(385, 256, 195, 75)
    idx = [("r_history", "tonic value r"), ("slope_pos", "RPE≥0 slope"), ("slope_neg", "RPE<0 slope")]
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d0 = s[(s["signal"] == sig) & (s["region"] == reg)]
        for i, (c, name) in enumerate(idx):
            for k, (tag, d) in enumerate((("all", d0), ("unbiased", d0[~d0["ses_idx"].isin(biased)]))):
                r = S.summarize_animals(d, c)
                out["robustness"][f"{sig}_{c}_{tag}"] = r
                x = i * 2.6 + j * 1.1 + k * 0.45
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=2.6,
                            color=col, mfc=col if k == 0 else "white", elinewidth=0.8)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks([0.8 + i * 2.6 for i in range(3)])
    ax.set_xticklabels([n for _, n in idx], fontsize=5)
    ax.text(0.98, 0.98, f"open: excluding {len(biased)} biased/stubborn sessions", transform=ax.transAxes, ha="right",
            va="top", fontsize=4.3)
    out["n_biased_sessions_excluded"] = len(biased)

    P.letter(6, 350, "g", "Summary")
    c = out["cohort"]
    lines = [
        f"No group side bias: b_side {st.fmt_ci(c['b_side'])} ({st.fmt_p(c['b_side']['p_signflip'])}); P(right) {st.fmt_ci(c['p_right'])}. "
        f"{100 * out['frac_sessions_bias_ci_excludes_0']:.0f}% of sessions show a session-level bias (CI excluding 0), in both directions.",
        f"Strong perseveration ρ = {st.fmt_ci(c['rho'])} and value sensitivity β_Q = {st.fmt_ci(c['beta_Q'])} in every animal.",
        "b_side and β_Q reproduce the per-session Q-model biasL and inverse temperature (same likelihood given Q); upstream side_bias "
        "and raw P(right) are independent and agree (r ≥ 0.89).",
        "Sign conventions unified to +R: upstream biasL is a left bias (flipped); upstream Q_Delta is Q_chosen − Q_unchosen, so the "
        "side-referenced regressor is Q_R − Q_L",
        "(using Q_Delta, as in the earlier new-cohort analysis, gives a spurious negative β_Q and inflates ρ).",
        f"Excluding {out['n_biased_sessions_excluded']} strongly biased (|b_side| > 1) or stubborn sessions leaves the DA tonic value index unchanged; "
        "RPE slopes keep their sign but shrink, most for NE (panel f), so part of the NE RPE effect comes from biased sessions.",
    ]
    P.text(40, 372, "\n".join(lines), fontsize=5.4, linespacing=1.6)
    cache.write_json(config.STATS / "figS06.json", out)
    return P.save("figS06")


if __name__ == "__main__":
    print(build())
