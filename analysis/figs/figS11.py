"""Fig S11: co-fluctuation of simultaneously recorded NAc DA and PL NE across timescales."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.align import grouped_animal_traces
from analysis.dataio import foraging as F
from analysis.figs import style as st
from analysis.models import continuous as C

N_SHIFT = 200
COUPLE = "#7b4f9d"  # DA-NE coupling
LAGS = range(-3, 4)


def paired_trials() -> pd.DataFrame:
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    ne = m[m["signal"] == "NE"][["ses_idx", "trial", "baseline", "outcome_bs", "gocue_base", "gocue_resp"]]
    da = m[(m["signal"] == "DA") & (m["region"] == "latNAcc")]
    da = da.groupby(["ses_idx", "subject_id", "trial"], as_index=False)[
        ["baseline", "outcome_bs", "gocue_base", "gocue_resp", "rewarded", "RPE_all", "streak_past"]].mean()
    return da.merge(ne, on=["ses_idx", "trial"], suffixes=("_da", "_ne"))


def session_coupling(d: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ses, g in d.groupby("ses_idx"):
        g = g.sort_values("trial")
        row = {"ses_idx": ses, "subject_id": g["subject_id"].iloc[0], "n": len(g)}
        for tag, a, b in (("tonic_precue", "gocue_base_da", "gocue_base_ne"), ("baseline", "baseline_da", "baseline_ne"),
                          ("phasic", "outcome_bs_da", "outcome_bs_ne"), ("gocue", "gocue_resp_da", "gocue_resp_ne")):
            row[f"r_{tag}"] = S.pearson(g[a], g[b])[0]
            x, y = g[a].to_numpy(), g[b].to_numpy()
            null = [np.corrcoef(x, np.roll(y, s))[0, 1] for s in rng.integers(10, len(g) - 10, N_SHIFT)]
            row[f"r_{tag}_null"] = float(np.nanmean(null))
        for rew in (True, False):
            h = g[g["rewarded"] == rew]
            row[f"r_phasic_{'R' if rew else 'U'}"] = S.pearson(h["outcome_bs_da"], h["outcome_bs_ne"])[0]
        z = np.column_stack([np.ones(len(g)), g["RPE_all"], g["rewarded"].astype(float)])
        res = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
        row["r_phasic_resid"] = float(np.corrcoef(res(g["outcome_bs_da"].to_numpy()), res(g["outcome_bs_ne"].to_numpy()))[0, 1])
        # co-states: median split of tonic baselines, concordance vs circular-shift null
        hi_da = g["baseline_da"] > g["baseline_da"].median()
        hi_ne = (g["baseline_ne"] > g["baseline_ne"].median()).to_numpy()
        conc = float(np.mean(hi_da.to_numpy() == hi_ne))
        null = [np.mean(hi_da.to_numpy() == np.roll(hi_ne, s)) for s in rng.integers(10, len(g) - 10, N_SHIFT)]
        row["costate_concordance"], row["costate_null"] = conc, float(np.mean(null))
        rows.append(row)
    return pd.DataFrame(rows)


def lagged_trial_corr(d: pd.DataFrame) -> pd.DataFrame:
    """Session-wise r between DA outcome response on trial t and NE outcome response on trial t+k."""
    rows = []
    for ses, g in d.groupby("ses_idx"):
        g = g.sort_values("trial")
        x, y = g["outcome_bs_da"].to_numpy(), g["outcome_bs_ne"].to_numpy()
        row = {"ses_idx": ses, "subject_id": g["subject_id"].iloc[0]}
        for k in LAGS:
            a, b = (x[: len(x) - k], y[k:]) if k >= 0 else (x[-k:], y[: len(y) + k])
            row[f"lag{k}"] = S.pearson(a, b)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def dot_ci(ax, x, r, color, dot_color=None, dx=0.14):
    v = list(r["per_animal"].values())
    ax.scatter(np.full(len(v), x) - dx, v, s=5, color=dot_color or color, alpha=0.7, lw=0)
    ax.errorbar(x + dx, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=3,
                color=color, ecolor=color, elinewidth=1)


def example_traces(ax, dn):
    """~80 s of simultaneously recorded NAc DA and PL NE (z) with reward ticks, from a median-coupling session."""
    med = dn["r_full"].median()
    row = dn.iloc[(dn["r_full"] - med).abs().argmin()]
    subject, ses, d = next(x for x in F.discover_sessions() if x[1] == row["ses_idx"])
    fip = F.load_fip(d)
    tr = F.load_trials(d, subject, ses)
    t0 = float(tr["goCue_start_time_in_session"].iloc[0]) + 300.0
    dur = 80.0
    for ch, col, off in ((row["ch_a"], st.DA_COLOR, 4.0), (row["ch_b"], st.NE_COLOR, 0.0)):
        r = fip[ch]
        m = (r["t"] >= t0) & (r["t"] <= t0 + dur)
        ax.plot(r["t"][m] - t0, r["data_z"][m] + off, color=col, lw=0.6)
    ch_t = tr["choice_time_in_session"]
    for rew, c, y in ((True, st.EVENT["reward"], -2.6), (False, "#808285", -2.6)):
        x = ch_t[(tr["rewarded"] == rew) & (ch_t >= t0) & (ch_t <= t0 + dur)] - t0
        ax.vlines(x, y, y + 0.9, color=c, lw=0.9)
    ax.set_xlim(0, dur)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.text(-0.01, 0.75, "NAc DA", color=st.DA_COLOR, transform=ax.transAxes, fontsize=5.5, ha="right")
    ax.text(-0.01, 0.4, "PL NE", color=st.NE_COLOR, transform=ax.transAxes, fontsize=5.5, ha="right")
    ax.text(0.99, 1.09, "no reward", color="#808285", transform=ax.transAxes, fontsize=4.8, ha="right")
    ax.text(0.86, 1.09, "reward", color=st.EVENT["reward"], transform=ax.transAxes, fontsize=4.8, ha="right")
    ax.text(0.0, 1.09, f"{ses}, r = {row['r_full']:.2f}", transform=ax.transAxes, fontsize=4.8, ha="left")
    ax.set_ylim(-3, 8.5)


def build():
    cp = pd.read_parquet(config.CACHE / "continuous_pairs.parquet")
    dn = cp[cp["pair"] == "DA-NE"].copy()
    pt = paired_trials()
    sc = session_coupling(pt)
    sc.to_csv(config.TABLES / "da_ne_coupling_sessions.csv", index=False)
    sess = cache.for_sess()
    sc = sc.merge(sess[["ses_idx", "reward_rate"]], on="ses_idx", how="left")
    P = st.Page(height=495)
    out = {"continuous": {}, "trialwise": {}, "costate": {}, "behavior": {}, "pair_types": {}, "lagged_trials": {}}

    P.letter(6, 6, "a", "Simultaneous NAc DA and PL NE")
    example_traces(P.ax(40, 32, 250, 72), dn)

    P.letter(320, 6, "b", "Coupling vs window length")
    ax = P.ax(350, 32, 105, 72)
    ws = list(C.WINDOWS_S)
    for suffix, col, lab in (("", COUPLE, "observed"), ("_null", "#a7a9ac", "shift null")):
        rr = [S.summarize_animals(dn, f"r_win{w}{suffix}") for w in ws]
        ax.errorbar(ws, [r["mean"] for r in rr], yerr=[[r["mean"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["mean"] for r in rr]],
                    color=col, marker="o", ms=2.5, lw=1, label=lab)
        if not suffix:
            out["continuous"] = {f"win{w}": r for w, r in zip(ws, rr)}
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws], fontsize=4.8)
    ax.set_xlabel("window (s)")
    ax.set_ylabel("DA–NE r")
    ax.legend(fontsize=4.5, loc="lower right")
    out["continuous"]["r_full"] = S.summarize_animals(dn, "r_full")
    out["continuous"]["frac_sessions_p_shift_lt_0.05"] = float((dn["r_full_p_shift"] < 0.05).mean())

    P.letter(475, 6, "c", "Cross-correlation")
    ax = P.ax(500, 32, 80, 72)
    lags = np.arange(-int(C.MAX_LAG_S / config.DT), int(C.MAX_LAG_S / config.DT) + 1) * config.DT
    xc = pd.DataFrame(np.vstack(dn["xcorr"].to_numpy()), index=pd.MultiIndex.from_frame(dn[["subject_id", "ses_idx"]]))
    a = xc.groupby("subject_id").mean()
    for v in a.to_numpy():
        ax.plot(lags, v, color=COUPLE, lw=0.4, alpha=0.4)
    st.trace(ax, lags, a.mean().to_numpy(), (a.std(ddof=1) / np.sqrt(len(a))).to_numpy(), color=COUPLE)
    ax.axvline(0, color="#808285", lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("lag (s; + = NE follows DA)")
    ax.set_ylabel("cross-corr.")
    out["continuous"]["peak_lag_s"] = S.summarize_animals(dn, "xcorr_peak_lag_s")
    st.stat_text(ax, f"peak lag\n{st.fmt_ci(out['continuous']['peak_lag_s'])} s", fontsize=4.5)

    # ---- row 2: outcome responses -------------------------------------------------------------------------
    P.letter(6, 130, "d", "Outcome responses, paired sessions")
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"] & m["ses_idx"].isin(pt["ses_idx"].unique())
          & (((m["signal"] == "DA") & (m["region"] == "latNAcc")) | (m["signal"] == "NE"))].reset_index(drop=True)
    ctr = cache.for_traces("choice")
    cg = config.time_grid(config.FOR_T_PRE, config.FOR_T_POST)
    tr = np.asarray(ctr[m["row"].to_numpy()], dtype=float)
    tr = tr - np.nanmean(tr[:, (cg >= config.BASELINE[0]) & (cg <= config.BASELINE[1])], axis=1, keepdims=True)
    pl = (cg >= -1) & (cg <= 3)
    for i, (sig, col, lab) in enumerate((("DA", st.DA_COLOR, "NAc DA"), ("NE", st.NE_COLOR, "PL NE"))):
        ax = P.ax(40 + i * 112, 155, 95, 75)
        for rew, c, name in ((True, col, "rewarded"), (False, "#808285", "unrewarded")):
            mask = ((m["signal"] == sig) & (m["rewarded"] == rew)).to_numpy()
            mu, se, n = grouped_animal_traces(m, tr[:, pl], mask)
            st.trace(ax, cg[pl], mu, se, color=c, label=f"{name}")
        ax.axvline(0, color="#bcbec0", lw=1, zorder=0)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.set_title(lab, fontsize=6, color=col)
        ax.set_xlabel("Time from choice (s)")
        if i == 0:
            ax.set_ylabel("z (baseline removed)")
            ax.legend(fontsize=4.3, loc="upper right")

    P.letter(270, 130, "e", "Trial-wise outcome responses")
    ax = P.ax(300, 155, 110, 75)
    rng = np.random.default_rng(0)
    sub = pt.iloc[rng.permutation(len(pt))[:4000]]
    for rew, c in ((False, "#808285"), (True, st.EVENT["reward"])):
        h = sub[sub["rewarded"] == rew]
        ax.scatter(h["outcome_bs_da"], h["outcome_bs_ne"], s=1.2, color=c, alpha=0.35, lw=0, rasterized=True)
    ax.set_xlabel("DA outcome response (z)")
    ax.set_ylabel("NE outcome response (z)")
    ax.text(0.03, 0.97, "rewarded", color=st.EVENT["reward"], transform=ax.transAxes, fontsize=4.8, va="top")
    ax.text(0.03, 0.88, "unrewarded", color="#808285", transform=ax.transAxes, fontsize=4.8, va="top")

    P.letter(430, 130, "f", "Coupling by pair type")
    ax = P.ax(460, 155, 120, 75)
    pairs = (("latL-latR", "DA–DA\nL vs R", st.DA_COLOR), ("lat-med", "DA–DA\nlat vs med", st.DA_COLOR),
             ("DA-NE", "DA–NE", COUPLE))
    for i, (pk, lab, col) in enumerate(pairs):
        r = S.summarize_animals(cp[cp["pair"] == pk], "r_full")
        out["pair_types"][pk] = r
        dot_ci(ax, i, r, col)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[1] for p in pairs], fontsize=4.6)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("whole-session r")

    # ---- row 3: trial-wise coupling ----------------------------------------------------------------------
    P.letter(6, 255, "g", "Trial-wise DA–NE coupling (n = %d mice)" % sc["subject_id"].nunique())
    ax = P.ax(40, 280, 290, 75)
    keys = [("r_tonic_precue", "pre-cue\ntonic"), ("r_baseline", "pre-choice\nbaseline"), ("r_gocue", "go-cue\nresponse"),
            ("r_phasic", "outcome\nresponse"), ("r_phasic_R", "outcome\nR+ only"), ("r_phasic_U", "outcome\nR− only"),
            ("r_phasic_resid", "outcome\n| RPE, R")]
    kcol = ["#a58bbd", "#a58bbd", COUPLE, COUPLE, st.EVENT["reward"], "#808285", "#4d2f66"]
    for i, (k, lab) in enumerate(keys):
        r = S.summarize_animals(sc, k)
        out["trialwise"][k] = r
        dot_ci(ax, i, r, kcol[i], dx=0.13)
        if f"{k}_null" in sc:
            ax.scatter(i + 0.13, S.summarize_animals(sc, f"{k}_null")["mean"], marker="_", s=30, color=st.INK, zorder=4)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([l for _, l in keys], fontsize=4.5)
    ax.set_ylabel("DA–NE r (per session)")
    ax.text(0.99, 0.98, "dots = mice, ● = mean ± 95% CI, — = shift null", transform=ax.transAxes, fontsize=4.3, ha="right", va="top")

    P.letter(355, 255, "h", "Tonic co-states")
    ax = P.ax(385, 280, 75, 75)
    for k, c in enumerate(("costate_concordance", "costate_null")):
        r = S.summarize_animals(sc, c)
        out["costate"][c] = r
        ax.bar(k, r["mean"], color=[COUPLE, "#a7a9ac"][k], width=0.7, lw=0)
        ax.errorbar(k, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none", ecolor="black")
    diff = sc.assign(d=sc["costate_concordance"] - sc["costate_null"])
    out["costate"]["difference"] = S.summarize_animals(diff, "d")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["observed", "shift null"], fontsize=4.8)
    ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("P(same high/low state)")
    st.stat_text(ax, f"Δ {st.fmt_ci(out['costate']['difference'])}\n{st.fmt_p(out['costate']['difference']['p_signflip'])}", fontsize=4.3)

    P.letter(480, 255, "i", "vs session reward rate")
    ax = P.ax(505, 280, 75, 75)
    ax.scatter(sc["reward_rate"], sc["r_phasic"], s=5, color=COUPLE, alpha=0.8, lw=0)
    per = sc.groupby("subject_id").apply(lambda g: S.spearman(g["reward_rate"], g["r_phasic"])[0]).rename("rho").reset_index()
    per["ses_idx"] = per["subject_id"]
    out["behavior"]["within_animal_rho_rewardrate_phasic_coupling"] = S.summarize_animals(per.dropna(), "rho")
    ax.set_xlabel("reward rate")
    ax.set_ylabel("outcome r")

    # ---- row 4 -------------------------------------------------------------------------------------------
    P.letter(6, 380, "j", "Cross-trial lag (outcome responses)")
    ax = P.ax(40, 405, 150, 58)
    lt = lagged_trial_corr(pt)
    for k in LAGS:
        r = S.summarize_animals(lt, f"lag{k}")
        out["lagged_trials"][f"lag{k}"] = r
    ms = [out["lagged_trials"][f"lag{k}"] for k in LAGS]
    ax.errorbar(list(LAGS), [r["mean"] for r in ms], yerr=[[r["mean"] - r["ci_lo"] for r in ms], [r["ci_hi"] - r["mean"] for r in ms]],
                color=COUPLE, marker="o", ms=3, lw=1)
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xlabel("trial lag k (NE trial t+k vs DA trial t)")
    ax.set_ylabel("DA–NE r")

    P.letter(225, 380, "k", "Summary")
    t = out["trialwise"]
    lines = [f"Whole-session DA–NE r = {st.fmt_ci(out['continuous']['r_full'])} (shift null ≈ 0), lower than DA–DA (L vs R",
             f"r = {st.fmt_ci(out['pair_types']['latL-latR'])}); coupling grows with window length and the",
             f"cross-correlation peaks near zero lag ({st.fmt_ci(out['continuous']['peak_lag_s'])} s).",
             f"Trial-wise, outcome responses co-vary (r = {st.fmt_ci(t['r_phasic'])}), partly through shared reward/RPE drive:",
             f"residual r after RPE and reward = {st.fmt_ci(t['r_phasic_resid'])}. Coupling is confined to the trial lag 0.",
             f"Pre-cue tonic coupling r = {st.fmt_ci(t['r_tonic_precue'])}; high/low baseline states co-occur above",
             f"chance by {st.fmt_ci(out['costate']['difference'])}. Shared event timing with partly independent trial-by-trial",
             "content supports the distinct coding in Fig 5 despite simultaneous recording."]
    P.text(250, 408, "\n".join(lines), fontsize=5.2, linespacing=1.55)
    cache.write_json(config.STATS / "figS11.json", out)
    return P.save("figS11")


if __name__ == "__main__":
    print(build())
