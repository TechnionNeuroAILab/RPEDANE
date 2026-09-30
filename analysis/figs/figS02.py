"""Fig S2: behavior and model validation for dynamic foraging (supports Fig 1b-d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.figs import style as st
from analysis.models import qlearn

LAGS = 6


def block_switch_curves(t: pd.DataFrame, pre=10, post=30) -> pd.DataFrame:
    rows = []
    for ses, g in t.groupby("ses_idx"):
        g = g.sort_values("trial").reset_index(drop=True)
        better_r = g["reward_probabilityR"] > g["reward_probabilityL"]
        flips = np.flatnonzero((better_r != better_r.shift()).to_numpy() & (g["reward_probabilityR"] != g["reward_probabilityL"]).to_numpy())[1:]
        for f in flips:
            new_r = bool(better_r.iloc[f])
            for k in range(-pre, post):
                i = f + k
                if 0 <= i < len(g) and g.loc[i, "responded"]:
                    rows.append({"subject_id": g.loc[i, "subject_id"], "ses_idx": ses, "k": k,
                                 "chose_new_best": float(g.loc[i, "choice_right"] == (1.0 if new_r else 0.0))})
    return pd.DataFrame(rows)


def lau_glimcher(t: pd.DataFrame) -> pd.DataFrame:
    """Per session logistic regression of choice (R=1) on past rewarded / unrewarded choices (signed +R/-L)."""
    rows = []
    for ses, g in t[t["responded"]].groupby("ses_idx"):
        g = g.sort_values("trial")
        c = np.where(g["choice_right"] == 1, 1.0, -1.0)
        r = g["rewarded"].to_numpy(float)
        cols = {}
        for k in range(1, LAGS + 1):
            ck = np.r_[np.full(k, np.nan), c[:-k]]
            rk = np.r_[np.full(k, np.nan), r[:-k]]
            cols[f"rew{k}"] = ck * rk
            cols[f"unr{k}"] = ck * (1 - rk)
        fit = S.logistic_mle(g["choice_right"].to_numpy(float), cols, min_n=100)
        if fit:
            rows.append({"ses_idx": ses, "subject_id": g["subject_id"].iloc[0], **{k: fit[k] for k in cols}})
    return pd.DataFrame(rows)


def build():
    t = cache.for_trials()
    sess = cache.for_sess()
    refit = pd.read_csv(config.TABLES / "qlearn_refit_params.csv", dtype={"subject_id": str})
    P = st.Page(height=455)
    out = {}
    animals = sorted(t["subject_id"].unique())
    colors = st.animal_colors(animals)

    P.letter(6, 6, "a", "Adaptation to block switches")
    bs = block_switch_curves(t)
    ax = P.ax(40, 30, 150, 75)
    a = bs.groupby(["subject_id", "ses_idx", "k"])["chose_new_best"].mean().groupby(["subject_id", "k"]).mean().unstack()
    for an, row in a.iterrows():
        ax.plot(row.index, row.to_numpy(), color=colors[an], lw=0.5, alpha=0.6)
    st.trace(ax, a.columns.to_numpy(), a.mean().to_numpy(), (a.std(ddof=1) / np.sqrt(len(a))).to_numpy(), color="black")
    ax.axvline(0, color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.axhline(0.5, color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("Trials from block switch")
    ax.set_ylabel("P(choose new better side)")
    out["block_switch_last10"] = S.bootstrap_mean(a.loc[:, 20:].mean(axis=1).to_numpy())

    P.letter(210, 6, "b", "Win-stay / lose-shift")
    ax = P.ax(235, 30, 80, 75)
    r = t[t["responded"]].copy()
    ws = r.dropna(subset=["prev_rewarded", "stay"]).groupby(["subject_id", "ses_idx", "prev_rewarded"])["stay"].mean()
    ws = ws.groupby(["subject_id", "prev_rewarded"]).mean().unstack()
    for an, row in ws.iterrows():
        ax.plot([0, 1], [row[1.0], 1 - row[0.0]], color=colors[an], marker="o", ms=2.5, lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["P(stay | R+)", "P(shift | R−)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 1)
    out["win_stay"] = S.bootstrap_mean(ws[1.0].to_numpy())
    out["lose_shift"] = S.bootstrap_mean((1 - ws[0.0]).to_numpy())

    P.letter(335, 6, "c", "Model-free choice history kernels")
    lg = lau_glimcher(t)
    ax = P.ax(360, 30, 120, 75)
    for key, col, lab in (("rew", st.FIT_POS, "rewarded choice"), ("unr", st.FIT_NEG, "unrewarded choice")):
        cols = [f"{key}{k}" for k in range(1, LAGS + 1)]
        aa = lg.groupby("subject_id")[cols].mean()
        st.trace(ax, np.arange(1, LAGS + 1), aa.mean().to_numpy(), (aa.std(ddof=1) / np.sqrt(len(aa))).to_numpy(),
                 color=col, label=lab)
        out[f"lau_glimcher_{key}"] = {c: S.summarize_animals(lg, c) for c in cols}
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xlabel("Trials back")
    ax.set_ylabel("Logistic weight (→ same side)")
    ax.legend(fontsize=4.5)

    P.letter(500, 6, "d", "Behavior")
    ax = P.ax(520, 30, 60, 75)
    bh = sess.groupby("subject_id")[["reward_rate", "finished_rate"]].mean()
    for k, c in enumerate(("reward_rate", "finished_rate")):
        ax.scatter(np.full(len(bh), k) + np.linspace(-0.15, 0.15, len(bh)), bh[c], s=6, c=[colors[a] for a in bh.index], lw=0)
        out[c] = S.bootstrap_mean(bh[c].to_numpy())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["reward\nrate", "finished\nrate"], fontsize=5)
    ax.set_ylim(0, 1)

    P.letter(6, 125, "e", "Per-session Q-learning parameters (upstream fits used for all latents)")
    params = [("learn_rate", "α (learning rate)"), ("forget_rate_unchosen", "forgetting (unchosen)"),
              ("softmax_inverse_temperature", "β (inverse temperature)"), ("choice_kernel_relative_weight", "choice-kernel weight"),
              ("bias_R_model", "bias (+R)")]
    out["upstream_params"] = {}
    for i, (c, lab) in enumerate(params):
        ax = P.ax(40 + i * 110, 150, 85, 65)
        for k, an in enumerate(animals):
            v = sess.loc[sess["subject_id"] == an, c]
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(v)), v, s=2, color=colors[an], lw=0)
            ax.scatter(k, v.median(), s=14, marker="_", color="black", lw=1.2)
        ax.set_xticks(range(len(animals)))
        ax.set_xticklabels([a[-3:] for a in animals], fontsize=4, rotation=90)
        ax.set_title(lab, fontsize=5.8)
        if c == "softmax_inverse_temperature":
            ax.set_ylim(0, 20)
        out["upstream_params"][c] = sess.groupby("subject_id")[c].median().to_dict()

    P.letter(6, 240, "f", "Animal-level refit (pooled sessions) vs upstream per-session fits")
    for i, (rc, uc, lab) in enumerate((("alpha", "learn_rate", "α"), ("forget", "forget_rate_unchosen", "forgetting"),
                                       ("beta", "softmax_inverse_temperature", "β"), ("bias_R", "bias_R_model", "bias (+R)"))):
        ax = P.ax(40 + i * 110, 262, 80, 65)
        med = sess.groupby("subject_id")[uc].median()
        x = med.reindex(refit["subject_id"]).to_numpy()
        ax.scatter(x, refit[rc], s=10, c=[colors[a] for a in refit["subject_id"]], lw=0)
        lim = [min(np.nanmin(x), refit[rc].min()), max(np.nanmax(x), refit[rc].max())]
        ax.plot(lim, lim, color="#808285", lw=0.5, ls=(0, (3, 2)))
        ax.set_xlabel(f"upstream median {lab}")
        ax.set_ylabel(f"refit {lab}")
        rr = S.spearman(x, refit[rc])
        out[f"refit_vs_upstream_{rc}"] = {"spearman": rr[0], "p": rr[1]}
        st.stat_text(ax, f"ρ = {rr[0]:.2f}")
    ax = P.ax(480, 262, 100, 65)
    lat = pd.read_parquet(config.CACHE / "qlearn_refit_latents.parquet")
    j = t[t["responded"]][["ses_idx", "trial", "RPE_all", "subject_id"]].merge(lat, on=["ses_idx", "trial"])
    per = j.groupby(["subject_id", "ses_idx"]).apply(lambda g: np.corrcoef(g["RPE_all"], g["refit_RPE"])[0, 1]).rename("r").reset_index()
    ax.hist(per["r"], bins=25, color="#a7a9ac", lw=0)
    ax.set_xlabel("r(stored RPE, refit RPE)")
    ax.set_ylabel("# sessions")
    out["rpe_stored_vs_refit"] = S.summarize_animals(per, "r")
    out["refit"] = refit.to_dict("records")

    P.letter(6, 350, "g", "Model fit quality")
    ax = P.ax(40, 372, 120, 65)
    ax.bar(range(len(refit)), refit["pseudo_r2"], color=[colors[a] for a in refit["subject_id"]], lw=0)
    ax.set_xticks(range(len(refit)))
    ax.set_xticklabels([a[-3:] for a in refit["subject_id"]], fontsize=4.5)
    ax.set_ylabel("McFadden pseudo-R²")
    ax.set_ylim(0, 1)
    out["pseudo_r2"] = S.bootstrap_mean(refit["pseudo_r2"].to_numpy())

    ax = P.ax(200, 372, 150, 65)
    rng = np.random.default_rng(0)
    g = t[(t["ses_idx"] == t["ses_idx"].unique()[40])]
    pl, pr = g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()
    fit = refit.iloc[0]
    ch, rw = qlearn.simulate_agent([fit[p] for p in qlearn.PARAMS], pl, pr, seed=1)
    ax.plot(pd.Series(ch).rolling(9, center=True, min_periods=3).mean(), color=st.SCATTER, lw=0.9, label="simulated agent")
    ax.plot(pr / (pl + pr + 1e-9), color="black", lw=0.7, ls=(0, (3, 2)), label="P(R)/(P(R)+P(L))")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(right)")
    ax.legend(fontsize=4.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Generative check (refit parameters, one session's blocks)", fontsize=5.5)
    out["generative_reward_rate"] = float(rw.mean())

    P.text(420, 375, "Summary\n"
           f"Mice track block switches (P(new best), trials 20–30: {st.fmt_ci({'mean': out['block_switch_last10'][0], 'ci_lo': out['block_switch_last10'][1], 'ci_hi': out['block_switch_last10'][2]})}).\n"
           f"Win-stay {out['win_stay'][0]:.2f}, lose-shift {out['lose_shift'][0]:.2f}.\n"
           "Rewarded choices are repeated with decaying weights (c).\n"
           f"Refit RPE matches stored RPE: r = {st.fmt_ci(out['rpe_stored_vs_refit'])}.\n"
           f"Pooled refit pseudo-R² = {out['pseudo_r2'][0]:.2f}.", fontsize=5.3, linespacing=1.6)
    cache.write_json(config.STATS / "figS02.json", out)
    return P.save("figS02")


if __name__ == "__main__":
    print(build())
