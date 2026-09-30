"""Fig S4: multi-regressor GLM of continuous DA/NE photometry, plus trial-level nested models with lick covariates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st
from analysis.models import glm as G
from analysis.models import neural_trial as NT

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]
KTITLE = {"gocue": "Go cue", "choice": "Choice", "outcome": "Outcome (reward − omission)", "lick_bout": "ITI lick bout",
          "RPE_all": "RPE-weighted", "abs_RPE": "|RPE|-weighted", "delta_L": "ΔL-weighted"}
FAM_LABEL = {"events": "Cue+choice", "outcome": "Outcome", "lick": "Licks", "value": "Value", "rpe": "RPE",
             "policy": "ΔL", "side": "Side", "latents": "All latents"}


def animal_kernel(k: pd.DataFrame, name: str):
    d = k[k["kernel"] == name]
    a = d.groupby(["subject_id", "ses_idx", "lag_s"])["coef"].mean().groupby(["subject_id", "lag_s"]).mean().unstack()
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else a.mean() * 0
    return m.index.to_numpy(), m.to_numpy(), se.to_numpy(), len(a)


def build():
    fits = pd.read_parquet(config.CACHE / "glm_fits.parquet")
    kern = pd.read_parquet(config.CACHE / "glm_kernels.parquet")
    ex = np.load(config.CACHE / "glm_examples.npy", allow_pickle=True).item()
    tm = pd.read_parquet(config.CACHE / "trial_models.parquet")
    vif = G.vif_table(cache.for_trials())
    P = st.Page(height=675)
    out = {"n": {}, "dR2": {}, "betas": {}, "trial_models": {}, "vif_median": vif.median().to_dict()}

    P.letter(6, 6, "a", "Continuous GLM: example fits (held-in, 60 s)")
    sel = {"DA": None, "NE": None}
    for ses, e in ex.items():
        sig = "NE" if "LCAxon" in e["channel"] else "DA"
        if sel[sig] is None and (sig == "NE" or e["channel"].startswith("latNAcc")):
            sel[sig] = (ses, e)
    for i, sig in enumerate(("DA", "NE")):
        if sel[sig] is None:
            continue
        ses, e = sel[sig]
        ax = P.ax(40 + i * 280, 30, 250, 50)
        ax.plot(e["t"], e["y"], color="#808285", lw=0.5, label="data (z)")
        ax.plot(e["t"], e["pred"], color=st.DA_COLOR if sig == "DA" else st.NE_COLOR, lw=0.9, label="GLM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z")
        ax.set_title(f"{sig}  {ses}  {e['channel']}", fontsize=5.5)
        ax.legend(fontsize=4.5, loc="upper right", ncol=2)

    P.letter(6, 100, "b", "Event kernels (animal mean ± SEM)")
    for i, name in enumerate(["gocue", "choice", "outcome", "lick_bout"]):
        ax = P.ax(40 + i * 140, 125, 115, 70)
        for sig, reg, col, lab in GROUPS:
            k = kern[(kern["signal"] == sig) & (kern["region"] == reg)]
            t, m, se, n = animal_kernel(k, name)
            st.trace(ax, t, m, se, color=col, label=f"{lab} (n={n})")
        ax.axvline(0, color="#bcbec0", lw=1.5, zorder=0)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.set_title(KTITLE[name], fontsize=6)
        ax.set_xlabel("Lag (s)")
        if i == 0:
            ax.set_ylabel("Kernel (z)")
            ax.legend(fontsize=4.5, loc="upper right")

    P.letter(6, 215, "c", "Trial-variable kernels: response per 1 SD of the variable")
    for i, name in enumerate(G.MOD_KERNELS[:3]):
        ax = P.ax(40 + i * 140, 240, 115, 70)
        for sig, reg, col, lab in GROUPS:
            k = kern[(kern["signal"] == sig) & (kern["region"] == reg)]
            t, m, se, n = animal_kernel(k, name)
            st.trace(ax, t, m, se, color=col, label=lab)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.set_title(KTITLE[name], fontsize=6)
        ax.set_xlabel("Time from choice (s)")
        if i == 0:
            ax.set_ylabel("Kernel (z per unit)")

    P.letter(440, 215, "d", "Modulator weights")
    ax = P.ax(465, 240, 115, 70)
    mods = G.PRE_MODS + ["side", "prev_choice", "licks_0_2"]
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = fits[(fits["signal"] == sig) & (fits["region"] == reg)]
        for i, mname in enumerate(mods):
            r = S.summarize_animals(d, f"beta_{mname}")
            out["betas"][f"{sig}_{mname}"] = r
            y = i + (j - 0.5) * 0.3
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=2.5,
                        color=col, elinewidth=0.8)
        for mname in G.MOD_KERNELS:
            out["betas"][f"{sig}_{mname}_0_1s"] = S.summarize_animals(d, f"beta_{mname}")
    ax.axvline(0, color="#808285", lw=0.5)
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(["Q_chosen (pre)", "Q_sum (pre)", "|ΔQ| (pre)", "Σ past rewards (pre)", "side (0–1 s)",
                        "prev. choice (0–1 s)", "licks (0–1 s)"], fontsize=4.8)
    ax.invert_yaxis()
    ax.set_xlabel("β (z per SD)")

    P.letter(6, 330, "e", "Unique cross-validated variance (drop-one family)")
    ax = P.ax(40, 355, 250, 85)
    fams = list(G.FAMILIES)
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = fits[(fits["signal"] == sig) & (fits["region"] == reg)]
        out["n"][sig] = {"fiber_sessions": int(len(d)), "animals": int(d["subject_id"].nunique()),
                         "cv_r2_full": S.summarize_animals(d, "cv_r2_full"),
                         "lambda_median": float(d["lambda"].median())}
        for i, f in enumerate(fams):
            r = S.summarize_animals(d.assign(v=100 * d[f"dR2_{f}"]), "v")
            out["dR2"][f"{sig}_{f}"] = r
            x = i + (j - 0.5) * 0.35
            ax.bar(x, r["mean"], width=0.33, color=col, alpha=0.8, lw=0, label=lab if i == 0 else None)
            ax.scatter(np.full(len(r["per_animal"]), x), list(r["per_animal"].values()), s=3, color="black", zorder=3, lw=0)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FAM_LABEL[f] for f in fams], fontsize=5)
    ax.set_ylabel("Unique ΔR² (%)")
    ax.legend(fontsize=5)
    da_full, ne_full = out["n"]["DA"]["cv_r2_full"], out["n"]["NE"]["cv_r2_full"]
    st.stat_text(ax, f"full-model CV R²: DA {st.fmt_ci(da_full)}, NE {st.fmt_ci(ne_full)}", y=1.12)

    P.letter(310, 330, "f", "Collinearity of trial regressors (median VIF)")
    ax = P.ax(335, 355, 100, 85)
    v = vif.median().sort_values().clip(upper=1e3)
    ax.barh(range(len(v)), v.to_numpy(), color="#a7a9ac", lw=0)
    for i, (name, val) in enumerate(v.items()):
        if val >= 100:
            ax.text(val, i, " ∞", va="center", fontsize=4.5)
    ax.text(1.12, 0.5, "reward, Q_chosen and RPE\nare exactly collinear\n(RPE = R − Q); the GLM\nenters reward only as\nthe outcome kernel",
            transform=ax.transAxes, ha="left", va="center", fontsize=4.5)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(v.index, fontsize=4.5)
    ax.axvline(5, color=st.FIT_POS, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("VIF")
    ax.set_xscale("log")

    P.letter(6, 465, "g", "Trial-level nested models (blocked CV): ΔR² of each added regressor")
    specs = (("out", [s for s, _ in NT.STEPS_OUTCOME][1:], "Outcome response (baseline removed)"),
             ("base", [s for s, _ in NT.STEPS_BASELINE][1:], "Pre-choice baseline"))
    for c_i, (prefix, steps, title) in enumerate(specs):
        ax = P.ax(40 + c_i * 200, 490, 170, 80)
        for j, (sig, reg, col, lab) in enumerate(GROUPS):
            d = tm[(tm["signal"] == sig) & (tm["region"] == reg)]
            for i, sname in enumerate(steps):
                r = S.summarize_animals(d.assign(v=100 * d[f"{prefix}_d_{sname}"]), "v")
                out["trial_models"][f"{sig}_{prefix}_{sname}"] = r
                x = i + (j - 0.5) * 0.35
                ax.bar(x, r["mean"], width=0.33, color=col, alpha=0.8, lw=0)
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none",
                            ecolor="black", elinewidth=0.7)
        ax.axhline(0, color=st.INK, lw=0.4)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=5)
        ax.set_title(title, fontsize=6)
        if c_i == 0:
            ax.set_ylabel("ΔR² (%)")

    P.letter(420, 465, "h", "RPE slopes with/without lick covariates")
    ax = P.ax(445, 490, 135, 80)
    labels = []
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = tm[(tm["signal"] == sig) & (tm["region"] == reg)]
        for i, (m, name) in enumerate((("slope_pos", "RPE≥0"), ("slope_neg", "RPE<0"), ("reward_jump", "R jump"))):
            for k, tag in enumerate(("nolick", "lick")):
                r = S.summarize_animals(d, f"{m}_{tag}")
                out["trial_models"][f"{sig}_{m}_{tag}"] = r
                x = j * 7 + i * 2 + k * 0.8
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o" if k else "s",
                            ms=2.8, color=col, mfc="white" if k else col, elinewidth=0.8)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks([0.4, 2.4, 4.4, 7.4, 9.4, 11.4])
    ax.set_xticklabels(["RPE≥0", "RPE<0", "R jump"] * 2, fontsize=4.5, rotation=45)
    ax.text(0.2, 1.02, "DA", transform=ax.transAxes, color=st.DA_COLOR, fontsize=5.5)
    ax.text(0.7, 1.02, "NE", transform=ax.transAxes, color=st.NE_COLOR, fontsize=5.5)
    ax.set_ylabel("Coefficient (z)")
    ax.text(0.98, 0.98, "filled: no licks\nopen: + lick covariates", transform=ax.transAxes, ha="right", va="top", fontsize=4.2)

    P.letter(6, 595, "i", "Summary")
    t = out["trial_models"]
    lines = [
        f"Continuous GLM: task events explain most variance (unique ΔR² DA {out['dR2']['DA_events']['mean']:.1f}%, NE "
        f"{out['dR2']['NE_events']['mean']:.1f}%); licks {out['dR2']['DA_lick']['mean']:.2f}% / {out['dR2']['NE_lick']['mean']:.2f}%; "
        f"all latent variables jointly {out['dR2']['DA_latents']['mean']:.2f}% / {out['dR2']['NE_latents']['mean']:.2f}%.",
        f"Trial-level: licks add {t['DA_out_+licks']['mean']:.1f}% (DA) and {t['NE_out_+licks']['mean']:.1f}% (NE) to outcome responses; "
        f"the DA reward jump falls from {t['DA_reward_jump_nolick']['mean']:.2f} to {t['DA_reward_jump_lick']['mean']:.2f} with lick covariates,",
        f"while the DA RPE≥0 slope rises to {st.fmt_ci(t['DA_slope_pos_lick'])} ({st.fmt_p(t['DA_slope_pos_lick']['p_signflip'])}). "
        f"Q_chosen adds {t['DA_base_+Q_chosen']['mean']:.1f}% to the DA baseline and {t['NE_base_+Q_chosen']['mean']:.1f}% to NE.",
        "Caveat: consummatory licking follows reward, so lick covariates and reward share variance (panel f); "
        "the lick analysis in Fig S5 isolates licking outside the task.",
    ]
    P.text(40, 615, "\n".join(lines), fontsize=5.4, linespacing=1.6)
    cache.write_json(config.STATS / "figS04.json", out)
    return P.save("figS04")


if __name__ == "__main__":
    print(build())
