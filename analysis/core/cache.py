"""Memoized accessors for the stage-1 caches."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from analysis import config


@lru_cache(None)
def grids() -> dict[str, np.ndarray]:
    g = np.load(config.CACHE / "grids.npz")
    return {"foraging": g["foraging"], "pavlovian": g["pavlovian"]}


@lru_cache(None)
def for_trials() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "for_trials.parquet")


@lru_cache(None)
def for_meta() -> pd.DataFrame:
    inv = inventory()
    ok = set(inv.loc[(inv["task"] == "foraging") & inv["included"], "ses_idx"])
    m = pd.read_parquet(config.CACHE / "for_meta.parquet")
    m["row"] = np.arange(len(m))
    m["session_ok"] = m["ses_idx"].isin(ok)
    return m


@lru_cache(None)
def for_traces(kind: str = "choice") -> np.ndarray:
    return np.load(config.CACHE / f"for_{kind}.npy", mmap_mode="r")


@lru_cache(None)
def for_licks() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "for_licks.parquet")


@lru_cache(None)
def for_sess() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "for_sess.parquet")


@lru_cache(None)
def for_chan_stats() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "for_chan_stats.parquet")


@lru_cache(None)
def pav_trials() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "pav_trials.parquet")


@lru_cache(None)
def pav_meta() -> pd.DataFrame:
    m = pd.read_parquet(config.CACHE / "pav_meta.parquet")
    m["row"] = np.arange(len(m))
    return m


@lru_cache(None)
def pav_traces(kind: str = "dff") -> np.ndarray:
    return np.load(config.CACHE / f"pav_cs_{kind}.npy", mmap_mode="r")


@lru_cache(None)
def pav_licks() -> pd.DataFrame:
    return pd.read_parquet(config.CACHE / "pav_licks.parquet")


@lru_cache(None)
def inventory() -> pd.DataFrame:
    return pd.read_csv(config.CACHE / "session_inventory.csv", dtype={"subject_id": str})


def foraging_fibers(signal: str, region: str | None = None, valid_only: bool = True) -> pd.DataFrame:
    """Rows of the foraging meta table for one signal family (optionally one NAc subregion)."""
    m = for_meta()
    sel = (m["signal"] == signal) & m["session_ok"]
    if region:
        sel &= m["region"] == region
    if valid_only:
        sel &= m["valid"]
    return m[sel]


def traces_for(rows: pd.DataFrame, kind: str = "choice") -> np.ndarray:
    return np.asarray(for_traces(kind)[rows["row"].to_numpy()])


def pav_traces_for(rows: pd.DataFrame, kind: str = "dff") -> np.ndarray:
    return np.asarray(pav_traces(kind)[rows["row"].to_numpy()])


def write_json(path, obj) -> None:
    from analysis.core.stats import jsonable
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2))


def read_json(path):
    return json.loads(path.read_text())
