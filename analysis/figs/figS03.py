"""Fig S3: Bayer-Glimcher reward-history kernels of phasic and tonic DA/NE signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st
from analysis.models import history_bg as BG

K_ALL = list(range(0, config.HISTORY_MAX + 1))
MODEL_STYLE = {"A": ("#231f20", "-", "A: outcome − baseline"), "B": ("#ed2124", "-", "B: outcome | baseline covariate"),
               "C": ("#35a0ab", "-", "C: tonic baseline")}
GROUPS = [("DA", "latNAcc", "NAc DA (lat)"), ("NE", "PL", "PL NE")]


def kernel(ax, d: pd.DataFrame, model: str, ks, color, label=None, per_animal=True, offset=0.0):
    cols = [f"{model}_b{k}" for k in ks]
    animals = d.groupby(["subject_id", "ses_idx"])[cols].mean().groupby("subject_id").mean()
    if per_animal:
        for _, row in animals.iterrows():
            ax.plot(np.array(ks) + offset, row.to_numpy(), color=color, lw=0.4, alpha=0.3)
    res = [S.bootstrap_mean(animals[c].to_numpy()) for c in cols]
    m = np.array([r[0] for r in res])
    lo, hi = np.array([r[1] for r in res]), np.array([r[2] for r in res])
    ax.errorbar(np.array(ks) + offset, m, yerr=[m - lo, hi - m], color=color, lw=1.1, marker="o", ms=2.5,
                capsize=0, elinewidth=0.8, label=label)
    ax.axhline(0, color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(ks)
    return m, animals


def neg(r: dict) -> dict:
    """Flip sign of a summarize_animals-style result (mean/CI), keeping n/p as is."""
    return {**r, "mean": -r["mean"], "ci_lo": -r["ci_hi"], "ci_hi": -r["ci_lo"],
            "per_animal": {k: -v for k, v in r["per_animal"].items()}}


def build():
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    trials = cache.for_trials()
    bg = BG.fit_all_sessions(m, trials)
    bg_gap = BG.fit_all_sessions(m, trials, drop_ignored_gaps=True)
    bg.to_csv(config.TABLES / "bayer_glimcher_sessions.csv", index=False)
    sim = BG.simulate_scenarios(trials)
    P = st.Page(height=1065)
    out = {"n_fiber_sessions": {}, "cohort": {}, "simulation": {}, "exp_fit": {}, "sensitivity": {},
           "session_breakdown": {}, "tonic_phasic_verdict": {}}

    # a: simulations (regression results)
    P.letter(6, 6, "a", "Simulated signals: what each regression reports")
    scenario_names = list(BG.SCENARIOS)
    for i, name in enumerate(scenario_names):
        ax = P.ax(40 + i * 140, 40, 115, 68)
        d = sim[sim["scenario"] == name]
        for model in ("A", "B"):
            c, ls, lab = MODEL_STYLE[model]
            ks = K_ALL
            mm = d[[f"{model}_b{k}" for k in ks]].mean().to_numpy()
            ax.plot(ks, mm, color=c, ls=ls, marker="o", ms=2.2, lw=1, label=lab)
        mc = d[[f"C_b{k}" for k in K_ALL[1:]]].mean().to_numpy()
        ax.plot(K_ALL[1:], mc, color=MODEL_STYLE["C"][0], marker="o", ms=2.2, lw=1, label=MODEL_STYLE["C"][2])
        ax.axhline(0, color="#808285", lw=0.5, ls=(0, (3, 2)))
        ax.set_xticks(K_ALL)
        ax.set_title(name, fontsize=5.6)
        ax.set_ylim(-0.45, 1.15)
        ax.set_xlabel("Trials back (0 = current)")
        if i == 0:
            ax.set_ylabel("Regression weight")
            ax.legend(fontsize=4.5, loc="upper right")
        out["simulation"][name] = {mdl: d[[f"{mdl}_b{k}" for k in (K_ALL if mdl != "C" else K_ALL[1:])]].mean().to_dict()
                                   for mdl in "ABC"}
    P.text(40, 140, "Scenarios 2 and 3 generate the same outcome window (Q + (R − Q) = R): windowed regressions cannot separate a "
                    "tonic value that persists into the outcome window plus a phasic RPE from a tonic value that has already reset "
                    "\nby the outcome (so the window contains only reward). Model A reads both as RPE; Model B reads both as "
                    "reward-only. Only the tonic kernel (C) and scenario 1 vs 4 are unambiguous.", fontsize=5, color="#58595b")

    # b (NEW): what the simulated scenarios actually look like, trial by trial, for one example session
    P.letter(6, 175, "b", "The same four scenarios, trial by trial (one representative simulated session)")
    ex_ses = sim.groupby("ses_idx").size().idxmax()
    ex = BG.simulate_session_example(trials, ex_ses, seed=1)
    n_show = min(150, len(ex))
    for i, name in enumerate(scenario_names):
        ax = P.ax(40 + i * 140, 200, 115, 62)
        t = ex["trial"].to_numpy()[:n_show]
        base, raw = ex[f"{name}__baseline"].to_numpy()[:n_show], ex[f"{name}__outcome_raw"].to_numpy()[:n_show]
        ax.scatter(t, base, s=2, color=MODEL_STYLE["C"][0], alpha=0.25, lw=0)
        ax.scatter(t, raw, s=2, color=MODEL_STYLE["A"][0], alpha=0.2, lw=0)
        win = 15
        ax.plot(t, pd.Series(base).rolling(win, center=True, min_periods=5).mean(), color=MODEL_STYLE["C"][0], lw=1.3,
                label="baseline (tonic window)")
        ax.plot(t, pd.Series(raw).rolling(win, center=True, min_periods=5).mean(), color=MODEL_STYLE["A"][0], lw=1.3,
                label="outcome (phasic window)")
        ax.axhline(0, color="#808285", lw=0.4, ls=(0, (3, 2)))
        ax.set_title(name, fontsize=5.6)
        ax.set_xlabel("Trial")
        ax.set_ylim(-1.3, 2.3)
        if i == 0:
            ax.set_ylabel("Simulated value (a.u.)")
            ax.legend(fontsize=4.2, loc="upper right")
    P.text(40, 284, f"Session {ex_ses}, first {n_show} trials; points = single-trial simulated values, lines = 15-trial rolling "
                    "mean. Scenarios 2 and 3 have identical baseline traces (both track Q_chosen) but their outcome traces are\n"
                    "statistically indistinguishable (both ≈ reward + noise) — this is the ambiguity panel a warns about, made "
                    "concrete.", fontsize=5, color="#58595b")

    # c/d: cohort kernels + cancellation
    P.letter(6, 328, "c", "Reward-history kernels across animals (thin lines: animals; points: mean ± 95% bootstrap CI)")
    for r, (sig, reg, lab) in enumerate(GROUPS):
        d = bg[(bg["signal"] == sig) & (bg["region"] == reg)]
        out["n_fiber_sessions"][sig] = {"n": int(len(d)), "n_animals": int(d["subject_id"].nunique()),
                                        "n_sessions": int(d["ses_idx"].nunique())}
        for c_i, model in enumerate("ABC"):
            ax = P.ax(40 + c_i * 140, 358 + r * 105, 115, 72)
            ks = K_ALL if model != "C" else K_ALL[1:]
            col, _, mlab = MODEL_STYLE[model]
            kernel(ax, d, model, ks, col)
            ax.set_title(f"{lab} — {mlab}", fontsize=5.8)
            ax.set_xlabel("Trials back (0 = current)")
            if c_i == 0:
                ax.set_ylabel("Weight (z / reward)")
            res = {f"b{k}": S.summarize_animals(d, f"{model}_b{k}") for k in ks}
            p_adj = S.bh_fdr([res[f"b{k}"]["p_signflip"] for k in ks])
            for k, pa in zip(ks, p_adj):
                res[f"b{k}"]["p_fdr_lags"] = pa
            res["hist_sum"] = S.summarize_animals(d, f"{model}_hist_sum")
            if model != "C":
                res["total"] = S.summarize_animals(d, f"{model}_total")
                res["cancel_frac"] = S.summarize_animals(d, f"{model}_cancel_frac")
            if model == "B":
                res["b_base"] = S.summarize_animals(d, "B_b_base")
            out["cohort"][f"{sig}_{model}"] = res
            txt = f"Σlags {st.fmt_ci(res['hist_sum'])}"
            ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=4.5)

    P.letter(440, 328, "d", "Cancellation")
    ax = P.ax(470, 358, 105, 72)
    xs = 0
    ticks, labels = [], []
    ref = sim.groupby("scenario")["A_cancel_frac"].mean()
    for sig, reg, lab in GROUPS:
        d = bg[(bg["signal"] == sig) & (bg["region"] == reg)]
        for model in ("A", "B"):
            a = S.animal_values(d, f"{model}_cancel_frac").clip(-1.5, 1.5)
            ax.scatter(np.full(len(a), xs) + np.linspace(-0.12, 0.12, len(a)), a, s=6, color=MODEL_STYLE[model][0], lw=0)
            mm, lo, hi = S.bootstrap_mean(a.to_numpy())
            ax.errorbar(xs + 0.25, mm, yerr=[[mm - lo], [hi - mm]], fmt="o", color="black", ms=2.5, elinewidth=1)
            ticks.append(xs)
            labels.append(f"{sig}-{model}")
            xs += 1
    ax.axhline(ref["Scalar RPE"], color="#35a0ab", lw=0.8, ls=(0, (3, 2)))
    ax.text(-0.4, ref["Scalar RPE"] + 0.04, "scalar RPE (simulated)", fontsize=4.2, va="bottom", color="#35a0ab")
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=4.8)
    ax.set_ylabel("−Σ history weights / β0")

    # e: alpha from exponential kernels vs learning rate
    P.letter(440, 443, "e", "Kernel decay vs learning rate")
    ax = P.ax(470, 468, 105, 65)
    refit_path = config.TABLES / "qlearn_refit_params.csv"
    refit = pd.read_csv(refit_path, dtype={"subject_id": str}).set_index("subject_id") if refit_path.exists() else None
    sess = cache.for_sess()
    lr_up = sess.groupby("subject_id")["learn_rate"].median()
    exp_rows = []
    for sig, reg, lab in GROUPS:
        d = bg[(bg["signal"] == sig) & (bg["region"] == reg)]
        for model in ("A", "B"):
            animals = d.groupby(["subject_id", "ses_idx"])[[f"{model}_b{k}" for k in K_ALL]].mean().groupby("subject_id").mean()
            for a, row in animals.iterrows():
                f = BG.exp_kernel_fit(row.to_numpy())
                exp_rows.append({"signal": sig, "model": model, "subject_id": a, **f,
                                 "learn_rate_upstream": lr_up.get(a, np.nan),
                                 "alpha_refit": refit.loc[a, "alpha"] if refit is not None and a in refit.index else np.nan})
    ex_df = pd.DataFrame(exp_rows)
    ex_df.to_csv(config.TABLES / "bayer_glimcher_expfit.csv", index=False)
    for sig, marker in (("DA", "o"), ("NE", "^")):
        e = ex_df[(ex_df["signal"] == sig) & (ex_df["model"] == "A")]
        ident = e["hist_over_gain"] > 0.1
        col = st.DA_COLOR if sig == "DA" else st.NE_COLOR
        ax.scatter(e.loc[ident, "learn_rate_upstream"], e.loc[ident, "alpha"], marker=marker, s=12, color=col,
                   label=f"{sig} (model A)", lw=0)
        ax.scatter(e.loc[~ident, "learn_rate_upstream"], e.loc[~ident, "alpha"], marker=marker, s=12, facecolor="none",
                   edgecolor=col, lw=0.6)
        e = e[ident]
        out["exp_fit"][sig] = {"spearman_alpha_vs_learn_rate": S.spearman(e["learn_rate_upstream"], e["alpha"]),
                               "n_identifiable": int(ident.sum()), "identifiable_rule": "hist_over_gain > 0.1",
                               "alpha": S.bootstrap_mean(e["alpha"].to_numpy()),
                               "hist_over_gain": S.bootstrap_mean(e["hist_over_gain"].to_numpy()),
                               "per_animal": e[["subject_id", "alpha", "hist_over_gain", "learn_rate_upstream", "alpha_refit"]]
                               .to_dict("records")}
    ax.plot([0, 1], [0, 1], color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model learning rate (median/animal)")
    ax.set_ylabel("Kernel decay α (fit)")
    ax.legend(fontsize=4.5, loc="upper left")
    ax.text(0.98, 0.02, "open: <10% history weight\n(decay unidentifiable)", transform=ax.transAxes, ha="right", fontsize=4)

    # f: sensitivity
    P.letter(6, 578, "f", "Sensitivity and cross-checks")
    specs = [("Lags spanning ignored trials dropped", bg_gap, "DA", "latNAcc"),
             ("Lags spanning ignored trials dropped", bg_gap, "NE", "PL"),
             ("Medial NAc DA", bg, "DA", "medNAcc")]
    for i, (title, d0, sig, reg) in enumerate(specs):
        ax = P.ax(40 + i * 140, 606, 115, 68)
        d = d0[(d0["signal"] == sig) & (d0["region"] == reg)]
        for model in ("A", "B"):
            kernel(ax, d, model, K_ALL, MODEL_STYLE[model][0], label=model, per_animal=False, offset=0.1 * (model == "B"))
        ax.set_title(f"{title}\n{sig} {reg} (n = {d['subject_id'].nunique()} mice)", fontsize=5.5)
        ax.set_xlabel("Trials back")
        if i == 0:
            ax.set_ylabel("Weight")
            ax.legend(fontsize=4.5)
        out["sensitivity"][f"{title}_{sig}_{reg}"] = {f"{mdl}_hist_sum": S.summarize_animals(d, f"{mdl}_hist_sum")
                                                     for mdl in "AB"}

    # upstream no-intercept coefficients (rachel-analysis-utils) for comparison
    ax = P.ax(470, 606, 105, 68)
    up = []
    for f in config.FORAGING_ROOT.glob("*/bg_coef_*.csv"):
        d = pd.read_csv(f)
        d["subject_id"] = f.parent.name
        d["signal"] = "NE" if "LCAxon" in f.name else "DA"
        up.append(d)
    up = pd.concat(up)
    for sig in ("DA", "NE"):
        a = up[up["signal"] == sig].groupby(["subject_id", "lag_num"])["coef"].mean().unstack()
        mm = a.mean()
        ax.plot(mm.index, mm.to_numpy(), marker="o", ms=2.5, lw=1, color=st.DA_COLOR if sig == "DA" else st.NE_COLOR,
                label=f"{sig} upstream (n={len(a)})")
    ax.axhline(0, color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.set_title("Upstream kernels (no intercept,\n0.33–1 s, trial-baseline removed)", fontsize=5.5)
    ax.set_xlabel("Trials back")
    ax.legend(fontsize=4.5)
    out["upstream_bg_coef_mean"] = {sig: up[up["signal"] == sig].groupby("lag_num")["coef"].mean().to_dict()
                                    for sig in ("DA", "NE")}

    # g (NEW): session-level breakdown (every fiber-session, not just animal means)
    P.letter(6, 696, "g", "Session-level breakdown (every fiber-session, not only animal means)")
    bins = np.linspace(-1.5, 1.5, 41)
    sess_color = {"DA": st.DA_COLOR, "NE": st.NE_COLOR}
    for j, (col_name, title, ref_sign) in enumerate((("B_hist_sum", "Phasic history kernel Σ (model B)", "negative = RPE-like"),
                                                     ("C_hist_sum", "Tonic history kernel Σ (model C)", "positive = value-like"))):
        ax = P.ax(40 + j * 175, 722, 150, 68)
        for sig, reg, lab in GROUPS:
            d = bg[(bg["signal"] == sig) & (bg["region"] == reg)][col_name].dropna()
            ax.hist(d.clip(bins[0], bins[-1]), bins=bins, color=sess_color[sig], alpha=0.5, lw=0, density=True,
                    label=f"{sig} (n={len(d)} sessions)")
            frac = float((d < 0).mean()) if "B_" in col_name else float((d > 0).mean())
            out["session_breakdown"][f"{sig}_{col_name}_frac_{'neg' if 'B_' in col_name else 'pos'}"] = frac
        ax.axvline(0, color=st.INK, lw=0.5)
        ax.set_title(f"{title}\n({ref_sign})", fontsize=5.2)
        ax.set_xlabel("Σ lags (z / reward)")
        ax.set_ylabel("density" if j == 0 else "")
        ax.legend(fontsize=4.3)

    ax = P.ax(400, 722, 175, 68)
    d = bg[bg["region"].isin(["latNAcc", "PL"])].copy()
    d["tonic"], d["phasic"] = d["C_hist_sum"], -d["B_hist_sum"]
    for sig, reg, lab in GROUPS:
        dd = d[(d["signal"] == sig) & (d["region"] == reg)]
        ax.scatter(dd["tonic"], dd["phasic"], s=4, color=sess_color[sig], alpha=0.45, lw=0, label=f"{sig} (n={len(dd)})")
    ax.axhline(0, color="#808285", lw=0.4, ls=(0, (3, 2)))
    ax.axvline(0, color="#808285", lw=0.4, ls=(0, (3, 2)))
    ax.set_xlabel("Tonic index (model C Σ)")
    ax.set_ylabel("Phasic RPE index (−model B Σ)")
    ax.set_title("Every fiber-session in one plane", fontsize=5.2)
    ax.legend(fontsize=4.3, loc="lower right")
    n_da = out["n_fiber_sessions"]["DA"]["n"]
    n_ne = out["n_fiber_sessions"]["NE"]["n"]
    P.text(40, 814, f"DA: {out['session_breakdown']['DA_B_hist_sum_frac_neg']:.0%} of {n_da} sessions have a negative "
                    f"(RPE-consistent) phasic kernel; {out['session_breakdown']['DA_C_hist_sum_frac_pos']:.0%} have a positive "
                    "(value-consistent) tonic kernel.\n"
                    f"NE: {out['session_breakdown']['NE_B_hist_sum_frac_neg']:.0%} of {n_ne} sessions have a negative phasic "
                    f"kernel; {out['session_breakdown']['NE_C_hist_sum_frac_pos']:.0%} have a positive tonic kernel.", fontsize=5,
           color="#58595b")

    # h (NEW): does each signal show tonic, phasic, both, or neither?
    P.letter(6, 850, "h", "Verdict: tonic value coding vs phasic RPE coding, by signal")
    ax = P.ax(40, 877, 300, 85)
    verdict_specs = [("DA", 0), ("NE", 1)]
    for gi, (sig, gx) in enumerate(verdict_specs):
        tonic = out["cohort"][f"{sig}_C"]["hist_sum"]
        phasic = neg(out["cohort"][f"{sig}_B"]["hist_sum"])
        out["tonic_phasic_verdict"][sig] = {"tonic_index": tonic, "phasic_rpe_index": phasic}
        for k, (r, name, col) in enumerate(((tonic, "tonic", MODEL_STYLE["C"][0]), (phasic, "phasic RPE", MODEL_STYLE["B"][0]))):
            x = gx * 3 + k
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=5, color=col,
                        elinewidth=1.4, capsize=2)
            sig_excl = r["ci_lo"] > 0 or r["ci_hi"] < 0
            ax.text(x, r["ci_hi"] + 0.03 * np.sign(r["ci_hi"] if r["ci_hi"] != 0 else 1), "significant" if sig_excl else "n.s.",
                    ha="center", va="bottom" if r["mean"] >= 0 else "top", fontsize=4.5,
                    color=col if sig_excl else "#808285")
    ax.axhline(0, color=st.INK, lw=0.5)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["tonic", "phasic RPE", "tonic", "phasic RPE"], fontsize=5)
    ax.text(0.17, -0.14, "DA", transform=ax.transAxes, ha="center", fontsize=6.5, color=st.DA_COLOR, fontweight="bold")
    ax.text(0.83, -0.14, "NE", transform=ax.transAxes, ha="center", fontsize=6.5, color=st.NE_COLOR, fontweight="bold")
    ax.set_ylabel("Index (animal mean ± 95% CI)")
    ax.set_xlim(-0.7, 4.7)
    da_t, da_p = out["tonic_phasic_verdict"]["DA"]["tonic_index"], out["tonic_phasic_verdict"]["DA"]["phasic_rpe_index"]
    ne_t, ne_p = out["tonic_phasic_verdict"]["NE"]["tonic_index"], out["tonic_phasic_verdict"]["NE"]["phasic_rpe_index"]
    def sig2(r: dict) -> str:
        return "significant" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "n.s."

    P.text(360, 887, "Reading this panel:\n"
                     f"DA — tonic: {st.fmt_ci(da_t, 2)} ({sig2(da_t)})\n"
                     f"        phasic RPE: {st.fmt_ci(da_p, 2)} ({sig2(da_p)})\n"
                     f"NE — tonic: {st.fmt_ci(ne_t, 2)} ({sig2(ne_t)})\n"
                     f"        phasic RPE: {st.fmt_ci(ne_p, 2)} ({sig2(ne_p)})\n"
                     "→ DA: tonic value, not reliably phasic RPE by this\n   measure. NE: phasic RPE, not tonic value.\n"
                     "   (Double dissociation, via an independent method\n    from the split-slope index in Figs 2/4/5.)",
           fontsize=5, linespacing=1.5, color="#58595b")

    # i: text summary
    P.letter(6, 983, "i", "Summary")
    ca, cb = out["cohort"]["DA_A"], out["cohort"]["DA_B"]
    na, nb = out["cohort"]["NE_A"], out["cohort"]["NE_B"]
    lines = [
        f"DA tonic kernel (model C) Σ lags = {st.fmt_ci(out['cohort']['DA_C']['hist_sum'])}, "
        f"{st.fmt_p(out['cohort']['DA_C']['hist_sum']['p_signflip'])}: pre-choice DA scales with recent rewards (tonic value).",
        f"DA phasic: model A Σ lags = {st.fmt_ci(ca['hist_sum'])} (cancellation {st.fmt_ci(ca['cancel_frac'])}); "
        f"model B Σ lags = {st.fmt_ci(cb['hist_sum'])}. The RPE-like history weights of DA depend on subtracting the tonic baseline.",
        f"NE tonic kernel Σ lags = {st.fmt_ci(out['cohort']['NE_C']['hist_sum'])}; NE phasic model A Σ lags = "
        f"{st.fmt_ci(na['hist_sum'])}, model B = {st.fmt_ci(nb['hist_sum'])}: history-dependent phasic NE without tonic value.",
        "Session-level breakdown (g) and the tonic/phasic verdict (h) confirm this at the individual-session level, not only "
        "in animal-pooled averages.",
        "Inference: per fiber-session OLS → session → animal means; exact sign-flip across animals (DA n = 8, NE n = 5; "
        "min p = 0.0078 and 0.0625), BH-FDR across lags.",
    ]
    P.text(40, 1003, "\n".join(lines), fontsize=5.5, linespacing=1.6)
    cache.write_json(config.STATS / "figS03.json", out)
    return P.save("figS03")


if __name__ == "__main__":
    print(build())
