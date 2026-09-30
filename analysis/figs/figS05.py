"""Fig S5: DA and NE responses to spontaneous licking during the inter-trial interval."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config
from analysis.core import cache
from analysis.core import stats as S
from analysis.core.align import grouped_animal_traces
from analysis.figs import style as st
from analysis.models import iti_licks as IL

GROUPS = [("DA", "latNAcc", st.DA_COLOR, "NAc DA"), ("NE", "PL", st.NE_COLOR, "PL NE")]


def pos_area(traces: np.ndarray, grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean of the rectified (positive-part) baseline-subtracted trace in [lo, hi]: the area under the positive
    peak only, so a later undershoot cannot cancel it."""
    m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
    return np.nanmean(np.clip(np.asarray(traces[:, m], dtype=float), 0, None), axis=1)


def contrasts(ev: pd.DataFrame, by: list[str] | None = None, metric: str = "pos_0_1") -> pd.DataFrame:
    """bout minus pseudo per fiber-session (optionally within strata of `by`)."""
    by = by or []
    keys = ["subject_id", "ses_idx", "channel", "signal", "region", "gap"]
    b = ev[ev["kind"] == "bout"].groupby(keys + by)[metric].mean()
    p = ev[ev["kind"] == "pseudo"].groupby(keys)[metric].mean()
    d = (b - p.reindex(b.index.droplevel(by) if by else b.index).to_numpy()).rename("delta").reset_index()
    return d


def build():
    ev = pd.read_parquet(config.CACHE / "iti_events.parquet")
    tr = np.load(config.CACHE / "iti_traces.npy", mmap_mode="r")
    ev["row"] = np.arange(len(ev))
    for tag, hi in (("0_05", 0.5), ("0_1", 1.0), ("0_2", 2.0)):
        ev[f"pos_{tag}"] = pos_area(tr, IL.GRID, 0.0, hi)
    ev = ev[ev["valid"] & ev["ses_idx"].isin(cache.for_meta().query("session_ok")["ses_idx"].unique())]
    itis = pd.read_parquet(config.CACHE / "itis.parquet")
    summ = pd.read_parquet(config.CACHE / "iti_session_summary.parquet")
    ev["ipsi"] = np.where(ev["side"] == "none", np.nan, (ev["side"] == ev["hemi"]).astype(float))
    ev["late_iti"] = (ev["iti_frac"] > 0.5).astype(float)
    grid = IL.GRID
    P = st.Page(height=470)
    out = {"n": {}, "contrast": {}, "splits": {}, "gap": {}, "ratio": {}}

    P.letter(6, 6, "a", "Inter-trial licking")
    summ["bouts_per_min"] = 60 * summ["n_bouts_gap1"] / summ["iti_total_s"]
    ses = summ.sort_values("bouts_per_min", ascending=False).iloc[3]["ses_idx"]
    lk = cache.for_licks().query("ses_idx == @ses")
    trials = cache.for_trials().query("ses_idx == @ses")
    it = itis.query("ses_idx == @ses")
    bt = ev[(ev["ses_idx"] == ses) & (ev["kind"] == "bout") & (ev["gap"] == 1.0)]["t0"].to_numpy()
    cand = trials["goCue_start_time_in_session"].dropna().to_numpy()
    t0 = float(cand[np.argmax([np.sum((bt >= c) & (bt < c + 120)) for c in cand])])
    dur = 120.0
    ax = P.ax(40, 28, 300, 45)
    for _, r in it[(it["iti_end"] > t0) & (it["iti_start"] < t0 + dur)].iterrows():
        ax.axvspan(max(r["iti_start"], t0) - t0, min(r["iti_end"], t0 + dur) - t0, color="#f1f2f2", lw=0)
    g = trials[(trials["goCue_start_time_in_session"] >= t0) & (trials["goCue_start_time_in_session"] <= t0 + dur)]
    ax.vlines(g["goCue_start_time_in_session"] - t0, 0, 3, color="#bcbec0", lw=0.8)
    for side, y, col in (("L", 1.0, st.EVENT["left"]), ("R", 2.0, st.EVENT["right"])):
        x = lk[(lk["side"] == side) & (lk["timestamps"] >= t0) & (lk["timestamps"] <= t0 + dur)]["timestamps"] - t0
        ax.vlines(x, y - 0.35, y + 0.35, color=col, lw=0.5)
    bouts = ev[(ev["ses_idx"] == ses) & (ev["kind"] == "bout") & (ev["gap"] == 1.0)].drop_duplicates("t0")
    bx = bouts[(bouts["t0"] >= t0) & (bouts["t0"] <= t0 + dur)]["t0"] - t0
    ax.scatter(bx, np.full(len(bx), 2.75), marker="v", s=8, color="black", zorder=3)
    ax.set_xlim(0, dur)
    ax.set_ylim(0.3, 3)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["L licks", "R licks"])
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    ax.text(0, 1.05, f"{ses}: grey = go cue, shaded = ITI, ▼ = isolated bout onset (≥1 s lick-free, window inside ITI)",
            transform=ax.transAxes, fontsize=4.8)

    ax = P.ax(375, 28, 90, 45)
    ax.hist(itis["iti_duration"].clip(upper=30), bins=40, color="#a7a9ac", lw=0)
    ax.set_xlabel("ITI duration (s)")
    ax.set_ylabel("# ITIs")
    ax = P.ax(495, 28, 85, 45)
    a = summ.groupby("subject_id")["bouts_per_min"].mean()
    ax.bar(range(len(a)), a.to_numpy(), color="#a7a9ac", lw=0)
    ax.set_xticks(range(len(a)))
    ax.set_xticklabels([s[-3:] for s in a.index], fontsize=4.5)
    ax.set_ylabel("Isolated bouts / min ITI")
    out["n"]["bouts_per_min_ITI"] = S.bootstrap_mean(a.to_numpy())
    out["n"]["n_itis"] = int(len(itis))
    out["n"]["median_iti_s"] = float(itis["iti_duration"].median())

    P.letter(6, 95, "b", "Bout-aligned responses vs matched no-lick times (gap 1 s)")
    e1 = ev[ev["gap"] == 1.0]
    for i, (sig, reg, col, lab) in enumerate(GROUPS):
        ax = P.ax(40 + i * 150, 118, 120, 75)
        for kind, c, ls in (("bout", col, "-"), ("pseudo", "#808285", "--")):
            rows = e1[(e1["signal"] == sig) & (e1["region"] == reg) & (e1["kind"] == kind)]
            m, se, n = grouped_animal_traces(rows, np.asarray(tr[rows["row"].to_numpy()]))
            st.trace(ax, grid, m, se, color=c, ls=ls, label=f"{kind} (n={n} mice)")
            out["n"][f"{sig}_{kind}"] = {"events": int(len(rows)), "sessions": int(rows["ses_idx"].nunique()), "animals": n}
        ax.axvline(0, color="#bcbec0", lw=1.5, zorder=0)
        ax.axhline(0, color="#808285", lw=0.4)
        ax.set_title(lab, fontsize=6.5)
        ax.set_xlabel("Time from bout onset (s)")
        ax.set_ylabel("z (baseline removed)")
        ax.legend(fontsize=4.5, loc="upper right")

    P.letter(330, 95, "c", "Per-animal contrast (bout − pseudo), positive-peak area")
    ax = P.ax(360, 118, 220, 75)
    x = 0
    ticks, labels = [], []
    for sig, reg, col, lab in GROUPS:
        for metric, mlab in (("pos_0_05", "0–0.5 s"), ("pos_0_1", "0–1 s"), ("pos_0_2", "0–2 s")):
            d = contrasts(e1[(e1["signal"] == sig) & (e1["region"] == reg)], metric=metric)
            r = S.summarize_animals(d, "delta")
            out["contrast"][f"{sig}_{metric}"] = r
            v = list(r["per_animal"].values())
            ax.scatter(np.full(len(v), x) + np.linspace(-0.15, 0.15, len(v)), v, s=5, color=col, lw=0)
            ax.errorbar(x + 0.3, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", ms=2.5,
                        color="black", elinewidth=0.9)
            ax.text(x, max(v) + 0.05, st.fmt_p(r["p_signflip"]), ha="center", fontsize=4.2)
            ticks.append(x)
            labels.append(f"{sig}\n{mlab}")
            x += 1
    ax.axhline(0, color="#808285", lw=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=4.3)
    ax.set_ylabel("Δ positive area (z)")

    P.letter(6, 225, "d", "Splits: lick side relative to fiber hemisphere, previous outcome, ITI position")
    split_defs = [("ipsi", {1.0: "ipsi", 0.0: "contra"}), ("prev_rewarded", {1.0: "prev R+", 0.0: "prev R−"}),
                  ("late_iti", {0.0: "early ITI", 1.0: "late ITI"})]
    for j, (col_name, names) in enumerate(split_defs):
        ax = P.ax(40 + j * 185, 250, 155, 70)
        for i, (sig, reg, col, lab) in enumerate(GROUPS):
            sub = e1[(e1["signal"] == sig) & (e1["region"] == reg)]
            d = contrasts(sub, by=[col_name])
            vals = {}
            for k, (level, lname) in enumerate(names.items()):
                r = S.summarize_animals(d[d[col_name] == level], "delta")
                vals[lname] = r
                xx = i * 3 + k
                ax.bar(xx, r["mean"], color=col, alpha=0.4 + 0.5 * k, width=0.8, lw=0)
                ax.errorbar(xx, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none",
                            ecolor="black", elinewidth=0.8)
            wide = d.pivot_table(index=["subject_id", "ses_idx", "channel"], columns=col_name, values="delta").dropna()
            diff = (wide[list(names)[1]] - wide[list(names)[0]]).rename("diff").reset_index()
            vals["difference"] = S.summarize_animals(diff, "diff")
            out["splits"][f"{sig}_{col_name}"] = vals
            ax.text(i * 3 + 0.5, ax.get_ylim()[1], st.fmt_p(vals["difference"]["p_signflip"]), ha="center", va="bottom", fontsize=4.5)
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels([*names.values()] * 2, fontsize=4.8)
        ax.axhline(0, color=st.INK, lw=0.4)
        ax.set_ylabel("Δ pos. area (bout − pseudo, 0–1 s)")
        ax.text(0.25, -0.22, "DA", transform=ax.transAxes, color=st.DA_COLOR, fontsize=5.5, ha="center")
        ax.text(0.75, -0.22, "NE", transform=ax.transAxes, color=st.NE_COLOR, fontsize=5.5, ha="center")

    P.letter(6, 350, "e", "Bout-definition sensitivity")
    ax = P.ax(40, 375, 150, 70)
    for i, (sig, reg, col, lab) in enumerate(GROUPS):
        ms, lo, hi = [], [], []
        for gap in IL.GAPS:
            d = contrasts(ev[(ev["gap"] == gap) & (ev["signal"] == sig) & (ev["region"] == reg)])
            r = S.summarize_animals(d, "delta")
            out["gap"][f"{sig}_{gap}"] = r
            ms.append(r["mean"]), lo.append(r["ci_lo"]), hi.append(r["ci_hi"])
        xs = np.array(IL.GAPS) * (1 + 0.04 * i)
        ax.errorbar(xs, ms, yerr=[np.array(ms) - lo, np.array(hi) - ms], color=col, marker="o", ms=3, label=lab, lw=1)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(IL.GAPS)
    ax.set_xticklabels([str(g) for g in IL.GAPS])
    ax.axhline(0, color="#808285", lw=0.4)
    ax.set_xlabel("Minimum lick-free gap (s)")
    ax.set_ylabel("Δ pos. area, z (0–1 s)")
    ax.legend(fontsize=4.5)

    P.letter(215, 350, "f", "ITI lick vs task outcome responses")
    ax = P.ax(245, 375, 150, 70)
    m = cache.for_meta()
    m = m[m["session_ok"] & m["valid"]]
    ctr = cache.for_traces("choice")
    cg = config.time_grid(config.FOR_T_PRE, config.FOR_T_POST)
    cbs = np.asarray(ctr[m["row"].to_numpy()], dtype=float)
    cbs = cbs - np.nanmean(cbs[:, (cg >= config.BASELINE[0]) & (cg <= config.BASELINE[1])], axis=1, keepdims=True)
    m = m.assign(outcome_pos=pos_area(cbs, cg, 0.0, 1.0))
    task = m.groupby(["subject_id", "ses_idx", "channel", "signal", "region", "rewarded"])["outcome_pos"].mean().unstack()
    task.columns = ["task_U", "task_R"]
    lickd = contrasts(e1).set_index(["subject_id", "ses_idx", "channel", "signal", "region"])["delta"]
    joined = task.join(lickd, how="inner").reset_index()
    for i, (sig, reg, col, lab) in enumerate(GROUPS):
        d = joined[(joined["signal"] == sig) & (joined["region"] == reg)]
        for k, (c, name) in enumerate((("delta", "ITI lick"), ("task_R", "rewarded"), ("task_U", "unrewarded"))):
            r = S.summarize_animals(d, c)
            out["ratio"][f"{sig}_{c}"] = r
            ax.bar(i * 4 + k, r["mean"], color=col, alpha=[1.0, 0.6, 0.3][k], lw=0, width=0.8)
            ax.errorbar(i * 4 + k, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="none",
                        ecolor="black", elinewidth=0.8)
        rr = d.assign(ratio=d["delta"].abs() / (d["task_R"] - d["task_U"]).abs())
        out["ratio"][f"{sig}_abs_lick_over_outcome_contrast"] = S.summarize_animals(rr, "ratio")
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(["ITI lick", "R+", "R−"] * 2, fontsize=4.8)
    ax.axhline(0, color=st.INK, lw=0.4)
    ax.set_ylabel("positive area, z (0–1 s)")

    P.letter(420, 350, "g", "Summary")
    c = out["contrast"]
    lines = [f"Isolated ITI lick bouts: DA {st.fmt_ci(c['DA_pos_0_1'])} ({st.fmt_p(c['DA_pos_0_1']['p_signflip'])}),",
             f"NE {st.fmt_ci(c['NE_pos_0_1'])} ({st.fmt_p(c['NE_pos_0_1']['p_signflip'])}) z (positive-peak",
             "area, 0–1 s) vs no-lick times.",
             f"|lick| / |R+ − R−| outcome contrast:",
             f"DA {st.fmt_ci(out['ratio']['DA_abs_lick_over_outcome_contrast'])},",
             f"NE {st.fmt_ci(out['ratio']['NE_abs_lick_over_outcome_contrast'])}.",
             "ITIs are rebuilt from the raw trial table, so an",
             "ITI never spans an ignored trial's go cue.",
             "Animal-level sign-flip tests (DA n = 8, NE n = 5)."]
    P.text(445, 378, "\n".join(lines), fontsize=5.2, linespacing=1.5)
    cache.write_json(config.STATS / "figS05.json", out)
    return P.save("figS05")


if __name__ == "__main__":
    print(build())
