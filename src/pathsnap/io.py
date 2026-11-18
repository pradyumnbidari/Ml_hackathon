"""Raw-file readers for PathSnap trips.

A trip folder `<id>/` contains `<id>.track`, `<id>.nodes`, `<id>.arcs` and
(optionally) `<id>.route`. All files are headerless and whitespace-separated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (K, 3) lon, lat, t
    nodes: np.ndarray          # (N, 2) lon, lat
    arcs: np.ndarray           # (A, 2) from_node, to_node
    route: Optional[np.ndarray] = None  # (M,) arc ids, chronological
    meta: dict = field(default_factory=dict)


def _load(path: str, dtype=float, ncols: int | None = None) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    """Load one trip. `with_route=False` is used by the prediction pipeline so
    that reference labels can never leak into inference."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    base = os.path.join(trip_dir, tid)
    track = _load(base + ".track", float, 3)
    nodes = _load(base + ".nodes", float, 2)
    arcs = _load(base + ".arcs", np.int64, 2)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = _load(base + ".route", np.int64, 1).ravel()
    return Trip(tid, track, nodes, arcs, route)


def list_trips(partition_dir: str) -> list[str]:
    return sorted(d for d in os.listdir(partition_dir)
                  if os.path.isdir(os.path.join(partition_dir, d)))


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove invalid / duplicate observations and sort by time.

    Returns the cleaned track and a dict of counts of what was removed so the
    EDA can report it.
    """
    report = {"n_raw": int(len(track))}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    report["n_invalid"] = int((~ok).sum())
    tr = track[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    report["n_out_of_order"] = int((order != np.arange(len(order))).sum())
    tr = tr[order]
    dup = np.r_[False, np.diff(tr[:, 2]) == 0]
    report["n_duplicate_ts"] = int(dup.sum())
    tr = tr[~dup]
    report["n_clean"] = int(len(tr))
    return tr, report
