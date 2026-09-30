"""Fig S7: robustness of the foraging coding indices (Fig 2) to analysis choices, plus trial-shuffle nulls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.classify import classify_regime, split_rpe_slopes
from analysis.figs import fig2
from analysis.figs import style as st

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]
N_SHUFFLE = 200


def controlled_slopes(rpe, outcome_raw, baseline, min_n: int = 8) -> dict:
    """Split RPE slopes from the RAW outcome response with the same-trial baseline as a covariate, so the
    baseline is regressed out rather than subtracted (subtraction imports -baseline, which tracks -Q, into the response)."""
    rpe, y, b = (np.asarray(v, float) for v in (rpe, outcome_raw, baseline))
    out = {}
    for name, sel in (("neg", rpe < 0), ("pos", rpe >= 0)):
        fit = S.ols(y[sel], {"rpe": rpe[sel], "baseline": b[sel]}, min_n=min_n) if sel.sum() >= min_n else None
        out[f"slope_{name}"] = fit["rpe"] if fit else np.nan
        out[f"p_{name}"] = fit["p_rpe"] if fit else np.nan
    return out


def variant_summary(m: pd.DataFrame, outcome: str, baseline: str, rpe: str = "RPE_all", control: bool = False) -> pd.DataFrame:
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        sl = controlled_slopes(g[rpe], g["outcome_raw"], g[baseline]) if control else split_rpe_slopes(g[rpe], g[outcome])
        r = S.pearson(g["streak_past"], g[baseline])[0]
        rows.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                     "region": g["region"].iloc[0], "r_history": r, "rpe_split": np.nanmean([sl["slope_pos"], sl["slope_neg"]]),
                     **{k: sl[k] for k in ("slope_pos", "slope_neg", "p_pos", "p_neg")}})
    return pd.DataFrame(rows)


def shuffle_null(m: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Circularly shift RPE relative to responses within each fiber-session; z-score the observed mean split slope."""
    rng = np.random.default_rng(seed)
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        x, y = g["RPE_all"].to_numpy(), g["outcome_bs"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 60:
            continue

        def stat(xx):
            sl = split_rpe_slopes(xx, y)
            return np.nanmean([sl["slope_pos"], sl["slope_neg"]])

        obs = stat(x)
        null = np.array([stat(np.roll(x, s)) for s in rng.integers(20, len(x) - 20, N_SHUFFLE)])
        hx, hb = g["streak_past"].to_numpy()[ok], g["baseline"].to_numpy()[ok]
        r_obs = np.corrcoef(hx, hb)[0, 1]
        r_null = np.array([np.corrcoef(np.roll(hx, s), hb)[0, 1] for s in rng.integers(20, len(x) - 20, N_SHUFFLE)])
        rows.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                     "region": g["region"].iloc[0], "rpe_obs": obs, "rpe_null_mean": np.nanmean(null),
                     "rpe_z": (obs - np.nanmean(null)) / np.nanstd(null), "rpe_p": float(np.mean(null >= obs)),
                     "hist_obs": r_obs, "hist_null_mean": np.nanmean(r_null),
                     "hist_z": (r_obs - np.nanmean(r_null)) / np.nanstd(r_null), "hist_p": float(np.mean(r_null >= r_obs))})
    return pd.DataFrame(rows)


def baseline_rpe_corr(m: pd.DataFrame) -> pd.DataFrame:
    """Same-trial baseline vs RPE, split by outcome (within an outcome RPE is a monotone function of Q), and vs Q."""
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        r, u = g[g["rewarded"] == True], g[g["rewarded"] == False]
        rows.append({"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
                     "region": g["region"].iloc[0], "r_b_rpe_R": S.pearson(r["RPE_all"], r["baseline"])[0],
                     "r_b_rpe_U": S.pearson(u["RPE_all"], u["baseline"])[0], "r_b_q": S.pearson(g["Q_chosen"], g["baseline"])[0]})
    return pd.DataFrame(rows)


def ne_regime_plane(P, s: pd.DataFrame, y0: float, out: dict):
    ne = s[s["signal"] == "NE"].copy()
    ax = P.ax(42, y0, 150, 95)
    ax.scatter(ne["r_history"], ne[["slope_pos", "slope_neg"]].mean(axis=1), s=9, c=[st.REGIME_COLORS[r] for r in ne["regime"]],
               edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(config.REGIME_R_HISTORY, color=st.INK, lw=0.5, ls=(0, (3, 2)))
    ax.axhline(0, color=st.INK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("Tonic value index\n(baseline vs past-outcome run, r)")
    ax.set_ylabel("Phasic RPE slope\n(mean split fit)")
    lo, hi = np.nanpercentile(ne[["slope_pos", "slope_neg"]].mean(axis=1), [1, 97])
    ax.set_ylim(lo - 0.2, hi + 0.2)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", color=st.REGIME_COLORS[k], markersize=3) for k in st.REGIME_LABEL],
              list(st.REGIME_LABEL.values()), fontsize=4.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, labelspacing=0.3)
    ax2 = P.ax(262, y0, 190, 95)
    animals = sorted(ne["subject_id"].unique())
    groups = [(a, ne[ne["subject_id"] == a]) for a in animals] + [("all", ne)]
    bottom = np.zeros(len(groups))
    for k in ("value_dominant", "mixed", "rpe_dominant", "weak"):
        frac = np.array([(d["regime"] == k).mean() for _, d in groups])
        ax2.bar(np.arange(len(groups)), frac, bottom=bottom, color=st.REGIME_COLORS[k], width=0.8, lw=0)
        bottom += frac
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels([a[-3:] if a != "all" else "all" for a, _ in groups], fontsize=4.8)
    for i, (_, d) in enumerate(groups):
        ax2.text(i, 1.02, str(len(d)), ha="center", va="bottom", fontsize=4.2)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Fraction of fiber-sessions")
    ax2.set_xlabel("PL NE, per animal (ID suffix)")
    counts = ne["regime"].value_counts()
    out["ne_regimes"] = {"counts": counts.to_dict(), "n_animals": len(animals)}
    P.text(470, y0 + 4, "PL NE: " + ", ".join(f"{st.REGIME_LABEL[k]} {counts.get(k, 0)}" for k in st.REGIME_LABEL)
           + f"\n({len(ne)} fiber-sessions, {len(animals)} animals)", fontsize=5.5)


def build():
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]].copy()
    lat = pd.read_parquet(config.CACHE / "qlearn_refit_latents.parquet")[["ses_idx", "trial", "refit_RPE"]]
    m = m.merge(lat, on=["ses_idx", "trial"], how="left")
    m["outcome_bs_altbase"] = m["outcome_raw"] - m["baseline_alt"]
    variants = [("main (0–2 s, base −1–0)", "outcome_bs", "baseline", "RPE_all"),
                ("outcome 0–0.5 s", "outcome_bs_0-0.5", "baseline", "RPE_all"),
                ("outcome 0–1 s", "outcome_bs_0-1", "baseline", "RPE_all"),
                ("baseline −2 to −1 s", "outcome_bs_altbase", "baseline_alt", "RPE_all"),
                ("refit RPE (pooled Q-model)", "outcome_bs", "baseline", "refit_RPE"),
                ("baseline regressed out\n(raw ~ RPE + baseline)", "outcome_raw", "baseline", "RPE_all", True)]
    P = st.Page(height=890)
    out = {"variants": {}, "thresholds": {}, "shuffle": {}, "baseline_rpe": {}, "ne_regimes": {}}

    P.letter(6, 6, "a", "Coding indices under alternative windows and RPE estimates (animal mean ± 95% CI)")
    summaries = {v[0]: variant_summary(m, *v[1:]) for v in variants}
    for c_i, (col, lab) in enumerate((("r_history", "tonic value index (r)"), ("slope_pos", "RPE ≥ 0 slope"),
                                      ("slope_neg", "RPE < 0 slope"))):
        ax = P.ax(130 + c_i * 155, 30, 125, 100)
        for j, (sig, reg, color, glab) in enumerate(GROUPS):
            for i, (name, *_) in enumerate(variants):
                d = summaries[name]
                d = d[(d["signal"] == sig) & (d["region"] == reg)]
                r = S.summarize_animals(d, col)
                out["variants"][f"{sig}|{name}|{col}"] = r
                y = i + (j - 0.5) * 0.3
                ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=2.5,
                            color=color, elinewidth=0.8)
        ax.axvline(0, color="#808285", lw=0.5)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([v[0] for v in variants] if c_i == 0 else [], fontsize=5)
        ax.invert_yaxis()
        ax.set_title(lab, fontsize=6)
    P.text(40, 133, "DA (blue) / NE (green)", fontsize=5)

    P.letter(6, 770, "g", "Regime fractions vs classification thresholds")
    base = summaries[variants[0][0]]
    for j, (sig, reg, color, glab) in enumerate(GROUPS):
        ax = P.ax(40 + j * 160, 795, 130, 75)
        d = base[(base["signal"] == sig) & (base["region"] == reg)]
        grid_r = [0.05, 0.10, 0.15, 0.20, 0.25]
        fr = {k: [] for k in st.REGIME_LABEL}
        for thr in grid_r:
            for alpha in (0.05,):
                reg_lab = d.apply(lambda r: classify_regime(r, alpha=alpha, r_thresh=thr), axis=1)
                for k in fr:
                    fr[k].append(float((reg_lab == k).mean()))
        bottom = np.zeros(len(grid_r))
        for k in ("value_dominant", "mixed", "rpe_dominant", "weak"):
            ax.bar(range(len(grid_r)), fr[k], bottom=bottom, color=st.REGIME_COLORS[k], width=0.8, lw=0,
                   label=st.REGIME_LABEL[k])
            bottom += np.array(fr[k])
        out["thresholds"][sig] = {"r_thresholds": grid_r, **fr}
        ax.set_xticks(range(len(grid_r)))
        ax.set_xticklabels([str(g) for g in grid_r])
        ax.set_xlabel("tonic value threshold (r)")
        ax.set_ylabel("fraction of fiber-sessions")
        ax.set_title(glab, fontsize=6)
        if j == 0:
            ax.legend(fontsize=4.5, loc="upper left", bbox_to_anchor=(2.35, 1.0))

    P.letter(6, 160, "b", "Baseline is tied to RPE only through Q")
    ax = P.ax(40, 185, 230, 75)
    cols = [("r_b_rpe_R", "R+"), ("r_b_rpe_U", "R−"), ("r_b_q", "Q")]
    bq = baseline_rpe_corr(m)
    x = 0
    ticks, labels = [], []
    for sig, reg, color, glab in GROUPS:
        d = bq[(bq["signal"] == sig) & (bq["region"] == reg)]
        for k, (c, lab) in enumerate(cols):
            r = S.summarize_animals(d, c)
            out["baseline_rpe"][f"{sig}_{c}"] = r
            ax.bar(x, r["mean"], color=color, alpha=[0.9, 0.55, 0.3][k], width=0.8, lw=0)
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none", ecolor="black", elinewidth=0.8)
            ticks.append(x)
            labels.append(f"r(base,\nRPE|{lab})" if k < 2 else "r(base,\nQ)")
            x += 1
        x += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=4)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_ylabel("r (per fiber-session)")
    ax.text(0.25, 1.02, "DA", transform=ax.transAxes, color=st.DA_COLOR, fontsize=5.5, ha="center")
    ax.text(0.75, 1.02, "NE", transform=ax.transAxes, color=st.NE_COLOR, fontsize=5.5, ha="center")

    P.letter(6, 285, "c", "Trial-shuffle nulls (RPE and outcome history circularly shifted within session)")
    sh = shuffle_null(m)
    sh.to_csv(config.TABLES / "shuffle_nulls.csv", index=False)
    for k, (col, lab) in enumerate((("rpe_z", "RPE slope vs shuffle (z)"), ("hist_z", "tonic value r vs shuffle (z)"))):
        ax = P.ax(40 + k * 170, 310, 140, 75)
        for j, (sig, reg, color, glab) in enumerate(GROUPS):
            d = sh[(sh["signal"] == sig) & (sh["region"] == reg)]
            ax.hist(d[col].clip(-6, 15), bins=np.linspace(-6, 15, 43), color=color, alpha=0.6, lw=0, density=True, label=glab)
            r = S.summarize_animals(d, col)
            frac = float((d[col.replace("_z", "_p")] < 0.05).mean())
            out["shuffle"][f"{sig}_{col}"] = {**r, "frac_sessions_p_lt_0.05": frac}
        ax.axvline(1.64, color=st.INK, lw=0.5, ls=(0, (3, 2)))
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.legend(fontsize=4.5)

    P.letter(380, 285, "d", "Summary")
    o = out["shuffle"]
    lines = [f"Sessions beating the shuffle (p < 0.05):",
             f"  RPE slope: DA {o['DA_rpe_z']['frac_sessions_p_lt_0.05']:.0%}, NE {o['NE_rpe_z']['frac_sessions_p_lt_0.05']:.0%}",
             f"  tonic value: DA {o['DA_hist_z']['frac_sessions_p_lt_0.05']:.0%}, NE {o['NE_hist_z']['frac_sessions_p_lt_0.05']:.0%}",
             "Trials are autocorrelated, so per-session parametric",
             "p-values overstate single-session RPE coding; the",
             "cohort claims rest on animal-level consistency.",
             "The DA tonic value index and the NE RPE≥0 slope keep",
             "their sign across windows, baselines and RPE estimates",
             "(a), except NE at 0–0.5 s (NE responses peak later).",
             "Baseline tracks Q, and within an outcome RPE is a",
             "function of Q, so base and RPE correlate (b) and",
             "subtracting baseline biases the RPE slopes. With the",
             "baseline regressed out (a, last row) DA slopes shrink",
             "toward 0 / negative; NE slopes persist.",
             "DA stays value/mixed-rich and NE RPE/weak-rich at",
             "every threshold (g). Signals: session z-scored dF/F."]
    P.text(405, 310, "\n".join(lines), fontsize=5.3, linespacing=1.55)
    summ = fig2.summary_table()
    P.letter(6, 430, "e", "Coding regimes across recordings (copy of Fig 2d)")
    fig2.regime_plane(P, summ, 460)
    P.letter(6, 595, "f", "Coding regimes, PL NE only")
    ne_regime_plane(P, summ, 625, out)
    cache.write_json(config.STATS / "figS07.json", out)
    return P.save("figS07")


if __name__ == "__main__":
    print(build())
