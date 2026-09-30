"""Fig S9: RPE vs value vs policy-update latents (Su & Cohen / Bari-style analyses) for DA and NE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]
LATENTS = ["RPE_all", "Q_chosen", "delta_L", "P_change", "abs_RPE"]
LAT_LABEL = {"RPE_all": "RPE", "Q_chosen": "Q_chosen", "delta_L": "ΔL (Su)", "P_change": "P_change (upstream)",
             "abs_RPE": "|RPE|"}


def latent_fits(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ses, ch), g in m.groupby(["ses_idx", "channel"]):
        g = g.dropna(subset=LATENTS + ["outcome_bs"]).sort_values("trial")
        if len(g) < 100:
            continue
        fit = S.ols(g["outcome_bs"], {c: g[c] for c in LATENTS}, standardize=True)
        full = S.blocked_cv_r2(g["outcome_bs"], {c: g[c] for c in LATENTS})
        row = {"ses_idx": ses, "channel": ch, "subject_id": g["subject_id"].iloc[0], "signal": g["signal"].iloc[0],
               "region": g["region"].iloc[0], "cv_full": full}
        for c in LATENTS:
            row[f"beta_{c}"] = fit[c]
            row[f"upr2_{c}"] = full - S.blocked_cv_r2(g["outcome_bs"], {k: g[k] for k in LATENTS if k != c})
            row[f"r_{c}"] = S.pearson(g[c], g["outcome_bs"])[0]
        # delta_L beyond RPE (partial r)
        z = np.column_stack([np.ones(len(g)), g["RPE_all"]])
        res = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
        row["partial_deltaL_given_RPE"] = float(np.corrcoef(res(g["delta_L"].to_numpy()), res(g["outcome_bs"].to_numpy()))[0, 1])
        row["partial_Pchange_given_RPE"] = float(np.corrcoef(res(g["P_change"].to_numpy()), res(g["outcome_bs"].to_numpy()))[0, 1])
        # Bari-style tonic value regressions
        b = S.ols(g["baseline"], {"Q_sum": g["Q_sum"], "abs_Q_Delta": g["Q_Delta"].abs()}, standardize=True)
        row["tonic_beta_Q_sum"], row["tonic_beta_absdQ"] = b["Q_sum"], b["abs_Q_Delta"]
        rows.append(row)
    return pd.DataFrame(rows)


def binned(ax, m, x, y, edges, color, label):
    d = m.dropna(subset=[x, y]).copy()
    d["b"] = pd.cut(d[x], edges, include_lowest=True)
    d["c"] = d["b"].apply(lambda v: v.mid).astype(float)
    a = d.groupby(["subject_id", "ses_idx", "c"], observed=True)[y].mean().groupby(["subject_id", "c"]).mean().unstack()
    mm, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    st.trace(ax, mm.index.to_numpy(), mm.to_numpy(), se.to_numpy(), color=color, label=f"{label} (n={len(a)})")
    return a


def build():
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    trials = cache.for_trials()
    lf = latent_fits(m)
    lf.to_csv(config.TABLES / "policy_latents.csv", index=False)
    P = st.Page(height=445)
    out = {"behavior": {}, "betas": {}, "unique_r2": {}, "partial": {}, "tonic": {}, "rt": {}}

    P.letter(6, 6, "a", "Behavior: policy update vs RPE (Su 1f)")
    r = trials[trials["responded"]].dropna(subset=["RPE_all", "delta_L", "P_change"])
    edges = np.linspace(-1, 1, 11)
    for k, (col, lab, colr) in enumerate((("delta_L", "ΔL = L(t+1) − L(t), chosen side", "black"),
                                          ("P_change", "upstream P_change", "#a7a9ac"))):
        ax = P.ax(40 + k * 125, 30, 100, 70)
        binned(ax, r, "RPE_all", col, edges, colr, lab)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.axvline(0, color="#808285", lw=0.4)
        ax.set_xlabel("RPE")
        ax.set_ylabel(lab, fontsize=5)
        per = r.groupby(["subject_id", "ses_idx"]).apply(lambda g: np.corrcoef(g["RPE_all"], g[col])[0, 1]).rename("r").reset_index()
        out["behavior"][f"r_RPE_{col}"] = S.summarize_animals(per, "r")
        st.stat_text(ax, f"r = {st.fmt_ci(out['behavior'][f'r_RPE_{col}'])}")
    P.text(40, 116, "P_change in df_trials is not the Su & Cohen ΔL (it tracks the change in P(chosen) across trials); "
                    "ΔL is recomputed from the model's choice probabilities.", fontsize=4.8, color="#58595b")

    P.letter(300, 6, "b", "Outcome response vs RPE (Su 3e)")
    ax = P.ax(325, 30, 110, 70)
    for sig, reg, col, lab in GROUPS:
        binned(ax, m[(m["signal"] == sig) & (m["region"] == reg)], "RPE_all", "outcome_bs", edges, col, lab)
    ax.axhline(0, color="#808285", lw=0.4)
    ax.set_xlabel("RPE")
    ax.set_ylabel("outcome response (z)")
    ax.legend(fontsize=4.5)
    P.letter(455, 6, "c", "vs ΔL (Su 3f)")
    ax = P.ax(475, 30, 105, 70)
    for sig, reg, col, lab in GROUPS:
        binned(ax, m[(m["signal"] == sig) & (m["region"] == reg)], "delta_L", "outcome_bs", np.linspace(-3, 3, 11), col, lab)
    ax.axhline(0, color="#808285", lw=0.4)
    ax.set_xlabel("ΔL")

    P.letter(6, 125, "d", "Joint regression (standardized β)")
    ax = P.ax(40, 150, 170, 80)
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = lf[(lf["signal"] == sig) & (lf["region"] == reg)]
        for i, c in enumerate(LATENTS):
            rr = S.summarize_animals(d, f"beta_{c}")
            out["betas"][f"{sig}_{c}"] = rr
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, rr["mean"], yerr=[[rr["mean"] - rr["ci_lo"]], [rr["ci_hi"] - rr["mean"]]], fmt="o", ms=3,
                        color=col, elinewidth=0.9, label=lab if i == 0 else None)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(range(len(LATENTS)))
    ax.set_xticklabels([LAT_LABEL[c] for c in LATENTS], fontsize=4.8, rotation=20)
    ax.set_ylabel("β (SD/SD)")
    ax.legend(fontsize=4.5)

    P.letter(235, 125, "e", "Unique CV R² (drop-one)")
    ax = P.ax(260, 150, 140, 80)
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = lf[(lf["signal"] == sig) & (lf["region"] == reg)]
        for i, c in enumerate(LATENTS):
            rr = S.summarize_animals(d.assign(v=100 * d[f"upr2_{c}"]), "v")
            out["unique_r2"][f"{sig}_{c}"] = rr
            x = i + (j - 0.5) * 0.35
            ax.bar(x, rr["mean"], width=0.33, color=col, lw=0, alpha=0.85)
            ax.errorbar(x, rr["mean"], yerr=[[rr["mean"] - rr["ci_lo"]], [rr["ci_hi"] - rr["mean"]]], fmt="none",
                        ecolor="black", elinewidth=0.7)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_xticks(range(len(LATENTS)))
    ax.set_xticklabels([LAT_LABEL[c] for c in LATENTS], fontsize=4.8, rotation=20)
    ax.set_ylabel("unique ΔR² (%)")

    P.letter(420, 125, "f", "Policy beyond RPE")
    ax = P.ax(445, 150, 135, 80)
    for j, (sig, reg, col, lab) in enumerate(GROUPS):
        d = lf[(lf["signal"] == sig) & (lf["region"] == reg)]
        for i, c in enumerate(("partial_deltaL_given_RPE", "partial_Pchange_given_RPE")):
            rr = S.summarize_animals(d, c)
            out["partial"][f"{sig}_{c}"] = rr
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, rr["mean"], yerr=[[rr["mean"] - rr["ci_lo"]], [rr["ci_hi"] - rr["mean"]]], fmt="o", ms=3,
                        color=col, elinewidth=0.9)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ΔL | RPE", "P_change | RPE"], fontsize=5)
    ax.set_ylabel("partial r")

    P.letter(6, 255, "g", "Tonic baseline vs value (Bari 3)")
    for k, (x, lab, e) in enumerate((("Q_sum", "Q_sum", np.linspace(0, 1.6, 9)), ("Q_chosen", "Q_chosen", np.linspace(0, 1, 9)))):
        ax = P.ax(40 + k * 125, 280, 100, 70)
        for sig, reg, col, glab in GROUPS:
            binned(ax, m[(m["signal"] == sig) & (m["region"] == reg)], x, "baseline", e, col, glab)
        ax.set_xlabel(lab)
        ax.set_ylabel("baseline (z)")
    for sig, reg, *_ in GROUPS:
        d = lf[(lf["signal"] == sig) & (lf["region"] == reg)]
        out["tonic"][sig] = {c: S.summarize_animals(d, c) for c in ("tonic_beta_Q_sum", "tonic_beta_absdQ")}

    P.letter(300, 255, "h", "Response time vs value (Bari 4)")
    ax = P.ax(325, 280, 110, 70)
    rt = trials[trials["responded"]].dropna(subset=["response_time", "Q_sum"])
    rt = rt[rt["response_time"].between(0, 2)]
    binned(ax, rt, "Q_sum", "response_time", np.linspace(0, 1.6, 9), "black", "all mice")
    per = rt.groupby(["subject_id", "ses_idx"]).apply(lambda g: S.spearman(g["Q_sum"], g["response_time"])[0]).rename("rho").reset_index()
    out["rt"]["spearman_Qsum_RT"] = S.summarize_animals(per, "rho")
    ax.set_xlabel("Q_sum")
    ax.set_ylabel("response time (s)")
    st.stat_text(ax, f"ρ = {st.fmt_ci(out['rt']['spearman_Qsum_RT'])}")

    P.letter(6, 375, "i", "Summary")
    b = out["betas"]
    lines = [f"Behavior: ΔL tracks RPE (r = {st.fmt_ci(out['behavior']['r_RPE_delta_L'])}); upstream P_change does not "
             f"(r = {st.fmt_ci(out['behavior']['r_RPE_P_change'])}).",
             f"Joint regression: RPE is the dominant latent for NE (β = {st.fmt_ci(b['NE_RPE_all'])}) and DA (β = {st.fmt_ci(b['DA_RPE_all'])}); "
             f"partial r of ΔL given RPE: NE {st.fmt_ci(out['partial']['NE_partial_deltaL_given_RPE'])}, "
             f"DA {st.fmt_ci(out['partial']['DA_partial_deltaL_given_RPE'])}: no positive policy-update signal beyond RPE.",
             "RPE and ΔL are strongly collinear (r ≈ 0.6 behaviorally), so their separate βs trade off; the policy-update "
             "hypothesis for NE is not distinguishable from RPE coding here.",
             f"Tonic DA baseline rises with Q_sum (β = {st.fmt_ci(out['tonic']['DA']['tonic_beta_Q_sum'])}); NE does not "
             f"(β = {st.fmt_ci(out['tonic']['NE']['tonic_beta_Q_sum'])})."]
    P.text(40, 395, "\n".join(lines), fontsize=5.3, linespacing=1.6)
    cache.write_json(config.STATS / "figS09.json", out)
    return P.save("figS09")


if __name__ == "__main__":
    print(build())
