"""Fig S1 (replaces the VTA soma/axon figure, whose data are not available): consistency of NAc DA recordings across
simultaneously recorded fibers (left vs right lateral NAc; lateral vs medial NAc)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.dataio import foraging as F
from analysis.figs import panels
from analysis.figs import style as st

REG = ["value_dominant", "mixed", "rpe_dominant", "weak"]


def paired_trials(a: str, b: str) -> pd.DataFrame:
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    A = m[m["channel"] == a][["ses_idx", "subject_id", "trial", "baseline", "outcome_bs"]]
    B = m[m["channel"] == b][["ses_idx", "trial", "baseline", "outcome_bs"]]
    return A.merge(B, on=["ses_idx", "trial"], suffixes=("_a", "_b"))


def trialwise_r(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ses, g in d.groupby("ses_idx"):
        rows.append({"ses_idx": ses, "subject_id": g["subject_id"].iloc[0],
                     "r_baseline": S.pearson(g["baseline_a"], g["baseline_b"])[0],
                     "r_outcome": S.pearson(g["outcome_bs_a"], g["outcome_bs_b"])[0]})
    return pd.DataFrame(rows)


def build():
    s = pd.read_csv(config.TABLES / "coding_summary.csv", dtype={"subject_id": str})
    cp = pd.read_parquet(config.CACHE / "continuous_pairs.parquet")
    P = st.Page(height=545)
    out = {"continuous": {}, "trialwise": {}, "concordance": {}, "lat_vs_med": {}}

    P.letter(6, 6, "a", "Simultaneous left and right lateral NAc DA")
    lr = cp[cp["pair"] == "latL-latR"]
    ses = lr.sort_values("duration_s", ascending=False).iloc[0]["ses_idx"]
    fip = F.load_fip(config.FORAGING_ROOT / ses.split("_")[0] / ses)
    trials = cache.for_trials().query("ses_idx == @ses")
    t0 = float(trials["goCue_start_time_in_session"].iloc[len(trials) // 2])
    for i, ch in enumerate(("latNAcc(L)-DA", "latNAcc(R)-DA")):
        ax = P.ax(40, 28 + i * 32, 330, 30)
        r = fip[ch]
        sel = (r["t"] >= t0) & (r["t"] <= t0 + 90)
        ax.vlines(trials["goCue_start_time_in_session"].to_numpy() - t0, 0, 1, transform=ax.get_xaxis_transform(),
                  color="#bcbec0", lw=0.6)
        ax.plot(r["t"][sel] - t0, r["data_z"][sel], color=st.TRACE_GREEN, lw=0.6)
        ax.set_xlim(0, 90)
        ax.axis("off")
        ax.text(0, 1.0, ch, transform=ax.transAxes, fontsize=5.5)
    ax.plot([70, 90], [ax.get_ylim()[0]] * 2, color=st.INK, lw=1)
    ax.text(80, ax.get_ylim()[0], "20 s", ha="center", va="top", fontsize=5.5)
    ax = P.ax(410, 28, 170, 62)
    for k, (pair, lab) in enumerate((("latL-latR", "latL–latR DA"), ("lat-med", "lat–med DA"), ("DA-NE", "DA–NE"))):
        d = cp[cp["pair"] == pair]
        ms = [S.summarize_animals(d, f"r_win{w}")["mean"] for w in (1, 2, 5, 10, 20, 30)]
        nl = [S.summarize_animals(d, f"r_win{w}_null")["mean"] for w in (1, 2, 5, 10, 20, 30)]
        col = ["#138a3e", "#8e5ea2", "#58595b"][k]
        ax.plot([1, 2, 5, 10, 20, 30], ms, marker="o", ms=2.5, color=col, label=f"{lab} (n={d['subject_id'].nunique()})")
        ax.plot([1, 2, 5, 10, 20, 30], nl, color=col, lw=0.6, ls=(0, (3, 2)))
        out["continuous"][pair] = {f"win{w}": S.summarize_animals(d, f"r_win{w}") for w in (1, 5, 30)}
        out["continuous"][pair]["r_full"] = S.summarize_animals(d, "r_full")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "30"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Mean within-window r")
    ax.legend(fontsize=4.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.text(0.02, 0.98, "dashed: circular-shift null", transform=ax.transAxes, va="top", fontsize=4.5)

    P.letter(6, 110, "b", "Trial-wise L vs R")
    tw = trialwise_r(paired_trials("latNAcc(L)-DA", "latNAcc(R)-DA"))
    ax = P.ax(40, 132, 110, 70)
    for k, (c, lab) in enumerate((("r_baseline", "baseline"), ("r_outcome", "outcome"))):
        r = S.summarize_animals(tw, c)
        out["trialwise"][c] = r
        ax.scatter(np.full(len(tw), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(tw)), tw[c], s=3, color="#a7a9ac", lw=0)
        ax.errorbar(k + 0.3, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", color="black", ms=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["baseline", "outcome"])
    ax.set_ylabel("Session r (L vs R)")
    ax.set_ylim(0, 1)

    P.letter(170, 110, "c", "Regime concordance L vs R")
    both = s[s["region"] == "latNAcc"].pivot_table(index="ses_idx", columns="hemi", values="regime", aggfunc="first").dropna()
    conf = pd.crosstab(both["L"], both["R"]).reindex(index=REG, columns=REG, fill_value=0)
    ax = P.ax(200, 132, 80, 70)
    ax.imshow(conf.to_numpy(), cmap="Greys", vmin=0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, conf.iloc[i, j], ha="center", va="center", fontsize=5,
                    color="white" if conf.iloc[i, j] > conf.to_numpy().max() / 2 else "black")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([st.REGIME_LABEL[r] for r in REG], fontsize=4.5, rotation=45)
    ax.set_yticklabels([st.REGIME_LABEL[r] for r in REG], fontsize=4.5)
    ax.set_xlabel("right fiber")
    ax.set_ylabel("left fiber")
    agree = float(np.trace(conf.to_numpy()) / conf.to_numpy().sum())
    chance = float((conf.sum(1) / conf.to_numpy().sum() * conf.sum(0) / conf.to_numpy().sum()).sum())
    out["concordance"] = {"n_sessions": int(len(both)), "agreement": agree, "chance": chance,
                          "table": conf.to_dict()}
    ax.set_title(f"agree {agree:.0%} (chance {chance:.0%})", fontsize=5.5)

    P.letter(310, 110, "d", "Coding indices, L vs R fiber in the same session")
    wide = s[s["region"] == "latNAcc"].pivot_table(index=["ses_idx", "subject_id"], columns="hemi",
                                                    values=["r_history", "rpe_split"]).dropna()
    for k, (c, lab) in enumerate((("r_history", "tonic value r"), ("rpe_split", "RPE slope"))):
        ax = P.ax(340 + k * 125, 132, 95, 70)
        ax.scatter(wide[(c, "L")], wide[(c, "R")], s=4, color=st.DA_COLOR, lw=0)
        lim = [np.nanmin(wide[c].to_numpy()), np.nanmax(wide[c].to_numpy())]
        ax.plot(lim, lim, color="#808285", lw=0.5, ls=(0, (3, 2)))
        r = S.pearson(wide[(c, "L")], wide[(c, "R")])
        out["concordance"][f"{c}_LR_r"] = {"r": r[0], "p": r[1], "n": r[2]}
        ax.set_xlabel(f"left: {lab}")
        ax.set_ylabel(f"right: {lab}")
        st.stat_text(ax, f"r = {r[0]:.2f}")

    P.letter(6, 225, "e", "Lateral vs medial NAc DA (same sessions, n = 2 mice)")
    med_ses = s.loc[s["region"] == "medNAcc", "ses_idx"].unique()
    rows_lat = cache.foraging_fibers("DA", "latNAcc")
    rows_lat = rows_lat[rows_lat["ses_idx"].isin(med_ses)]
    rows_med = cache.foraging_fibers("DA", "medNAcc")
    P.text(40, 240, "Tonic component (pre-choice baseline, shaded box)", fontsize=5, color="#58595b")
    P.text(310, 240, "Phasic component (RPE-binned, outcome-locked)", fontsize=5, color="#58595b")
    for k, (rows, lab) in enumerate(((rows_lat, "lateral NAc"), (rows_med, "medial NAc"))):
        a1 = P.ax(40 + k * 270, 258, 105, 66)
        panels.q_outcome_panel(a1, rows, level="animal", legend=(k == 0))
        a1.set_title(lab, fontsize=6, pad=3)
        a2 = P.ax(175 + k * 270, 258, 105, 66)
        panels.rpe_panel(a2, rows, level="animal", legend=(k == 0), window=True)
        a2.set_title(lab, fontsize=6, pad=9)
    pw = s[s["ses_idx"].isin(med_ses) & (s["signal"] == "DA")].groupby(["ses_idx", "subject_id", "region"])[
        ["r_history", "rpe_split", "slope_pos"]].mean().unstack("region").dropna()
    for c in ("r_history", "rpe_split", "slope_pos"):
        diff = (pw[(c, "medNAcc")] - pw[(c, "latNAcc")]).rename("d").reset_index()
        out["lat_vs_med"][c] = {"med_minus_lat_sessions": S.bootstrap_mean(diff["d"].to_numpy()),
                                "per_animal": diff.groupby("subject_id")["d"].mean().to_dict(), "n_sessions": int(len(diff))}

    P.letter(6, 358, "f", "Tonic value and phasic RPE indices, lateral vs medial (paired sessions, n = 2 mice)")
    animals_here = sorted(pw.index.get_level_values("subject_id").unique())
    colors_here = st.animal_colors(animals_here)
    subj = pw.index.get_level_values("subject_id").to_numpy()
    metrics = [("r_history", "Tonic value index\n(baseline vs reward-history r)"),
               ("rpe_split", "Phasic RPE index\n(mean split-fit slope)")]
    for i, (c, lab) in enumerate(metrics):
        ax = P.ax(40 + i * 220, 382, 190, 78)
        lat_arr = pw[(c, "latNAcc")].to_numpy()
        med_arr = pw[(c, "medNAcc")].to_numpy()
        for a in animals_here:
            sel = subj == a
            for l, mv in zip(lat_arr[sel], med_arr[sel]):
                ax.plot([0, 1], [l, mv], color=colors_here[a], lw=0.5, alpha=0.3, zorder=1)
            ax.plot([0, 1], [lat_arr[sel].mean(), med_arr[sel].mean()], color=colors_here[a], lw=2, marker="o", ms=4,
                    zorder=3, label=f"{a[-3:]} (n={int(sel.sum())} sessions)")
        ax.axhline(0, color="#808285", lw=0.4, ls=(0, (3, 2)))
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["lateral", "medial"])
        ax.set_ylabel(lab, fontsize=5.2)
        d = out["lat_vs_med"][c]["med_minus_lat_sessions"]
        ax.text(0.98, 0.02, f"Δ(medial−lateral)\n{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=4.5)
        if i == 0:
            ax.legend(fontsize=4.3, loc="upper right", handlelength=1.2)
    P.text(480, 402, "Both indices drop from lateral to\nmedial in both mice: medial DA\nlooks like a weaker version of\n"
                     "the same tonic+phasic code, not\na qualitatively different one\n(no selective tonic-vs-phasic\n"
                     "dissociation across subregion).", fontsize=5, linespacing=1.5, color="#58595b")

    P.letter(6, 486, "g", "Note")
    P.text(40, 506, "The original Fig S1 (simultaneous VTA DA soma, VTA→NAc axon and NAc DA release) cannot be regenerated: those "
                    f"recordings are not in {config.FORAGING_ROOT.name}.\nThis figure instead documents that the NAc DA photometry signal is "
                    f"consistent across simultaneously recorded fibers (L–R whole-session r = "
                    f"{st.fmt_ci(out['continuous']['latL-latR']['r_full'])}), so the heterogeneity in Fig 2 is not fiber noise:\n"
                    f"regime labels agree across hemispheres in {agree:.0%} of sessions (chance {chance:.0%}).", fontsize=5.4,
           linespacing=1.6)
    cache.write_json(config.STATS / "figS01.json", out)
    return P.save("figS01")


if __name__ == "__main__":
    print(build())
