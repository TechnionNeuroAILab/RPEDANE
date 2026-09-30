"""Paths, analysis windows, bin edges, and inclusion thresholds shared by the whole pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Package lives at <project>/code_repo/analysis, so the project root is two levels up.
PROJECT = Path(__file__).resolve().parents[2]
# DANE_FORAGING_ROOT / DANE_OUT let the same pipeline run against a different foraging dataset
# (e.g. the permissive-QC cohort) without touching the default run's cache/figures/stats.
FORAGING_ROOT = Path(os.environ.get("DANE_FORAGING_ROOT", PROJECT / "data" / "DA_NE_cleaned_data"))
PAV_ROOT = PROJECT / "data" / "DA_NE_pavlovian" / "nwbs"
SNIPPET_SOURCE = PROJECT / "overleaf_repo" / "figures"

OUT = Path(os.environ.get("DANE_OUT", PROJECT / "paper_figures_v2"))
CACHE = OUT / "cache"
TABLES = OUT / "tables"
STATS = OUT / "stats"
FIGURES = OUT / "figures"
SNIPPETS = OUT / "snippets"

FS = 20.0
DT = 0.05

# choice-aligned foraging window (cached wider than plotted to allow baseline sensitivity)
FOR_T_PRE, FOR_T_POST = -2.0, 4.0
PLOT_T_PRE, PLOT_T_POST = -1.0, 4.0
BASELINE = (-1.0, 0.0)
BASELINE_ALT = (-2.0, -1.0)
OUTCOME = (0.0, 2.0)
OUTCOME_ALTS = {"0-0.5": (0.0, 0.5), "0-1": (0.0, 1.0), "0-2": (0.0, 2.0)}
GOCUE_WINDOW = (0.0, 1.0)

# Pavlovian CS-aligned window; US arrives 2 s after CS onset
PAV_T_PRE, PAV_T_POST = -2.0, 6.0
PAV_BASELINE = (-1.0, 0.0)
PAV_CS_WINDOW = (0.0, 1.0)
PAV_DELAY_WINDOW = (1.0, 2.0)
PAV_US_WINDOW = (2.0, 3.0)
US_DELAY = 2.0
CS_PROB = {"CS1": 0.1, "CS2": 0.5, "CS3": 0.9}
CS_AIRPUFF = "CS4"

# paper conventions (rachel_analysis_utils.enrich_df_trials): RPE bins of width 1/3 on (-1, 1]
RPE_EDGES = np.array([-1.001, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.001])
RPE_LABELS = ["-1.0", "-0.67", "-0.33", "0.0", "0.33", ">0.67"]
Q_EDGES = np.array([-0.001, 1 / 3, 2 / 3, 1.001])
Q_LABELS = ["low", "mid", "high"]
HISTORY_MAX = 6

MIN_RESPONDED_TRIALS = 100
MIN_BG_TRIALS = 40
MAX_NAN_FRAC = 0.05
REGIME_R_HISTORY = 0.15
REGIME_ALPHA = 0.05
N_BOOT = 5000
SEED = 0

DA_MAIN_REGION = "latNAcc"


def ensure_dirs() -> None:
    for p in (CACHE, TABLES, STATS, FIGURES, SNIPPETS):
        p.mkdir(parents=True, exist_ok=True)


def time_grid(t_pre: float, t_post: float, dt: float = DT) -> np.ndarray:
    n = int(round((t_post - t_pre) / dt)) + 1
    return t_pre + np.arange(n) * dt
