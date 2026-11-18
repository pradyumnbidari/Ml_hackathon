"""Candidate generation + feature engineering.

One row per (GPS observation, candidate arc). The classifier scores how
likely each candidate is the arc the vehicle was actually on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geo import LocalProjector, angle_diff, bearing_xy, point_segment
from .io import Trip, clean_track
from .labels import align_points_to_route
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "t_proj", "t_raw_clip", "at_endpoint",
    "head_diff", "head_cos", "speed", "head_reliable",
    "arc_len_log", "from_in_deg", "to_out_deg", "has_twin",
    "ctx_dist_mean", "ctx_dist_max", "ctx_within20", "ctx_head_diff",
    "ctx_dist_rel", "dt_prev", "gap_flag",
]

LABEL_RELIABLE_M = 50.0  # aligned distance above this => label not trusted


def point_kinematics(p: np.ndarray, t: np.ndarray, w: int = 2):
    """Smoothed speed (m/s) and heading (deg) using a +-w window."""
    K = len(p)
    i0 = np.clip(np.arange(K) - w, 0, K - 1)
    i1 = np.clip(np.arange(K) + w, 0, K - 1)
    disp = p[i1] - p[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    dist = np.linalg.norm(disp, axis=1)
    speed = dist / dt
    head = bearing_xy(disp[:, 0], disp[:, 1])
    reliable = (dist > 4.0).astype(float)
    return speed, head, reliable


def prepare_trip(trip: Trip, radius=C.CAND_RADIUS_M, k_max=C.CAND_K_MAX,
                 with_labels=True):
    """Clean, project, build network and candidates for one trip."""
    trk, rep = clean_track(trip.track)
    proj = LocalProjector.for_points(trk[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    p = proj.fwd(trk[:, :2])
    cands = net.candidates(p, radius, k_max)
    out = {"trip": trip, "track": trk, "proj": proj, "net": net, "p": p,
           "cands": cands, "clean_report": rep}
    if with_labels and trip.route is not None:
        lab, ridx, adist = align_points_to_route(p, net, trip.route)
        out.update(label=lab, route_idx=ridx, align_dist=adist,
                   reliable=adist <= LABEL_RELIABLE_M)
    return out


def build_features(prep: dict, win: int = C.CTX_WIN) -> pd.DataFrame:
    p, trk, net, cands = prep["p"], prep["track"], prep["net"], prep["cands"]
    K = len(p)
    t = trk[:, 2]
    speed, head, head_rel = point_kinematics(p, t)
    dt_prev = np.r_[1.0, np.diff(t)]

    n_c = np.array([len(c[0]) for c in cands])
    pt = np.repeat(np.arange(K), n_c)
    if len(pt) == 0:
        return pd.DataFrame(columns=["trip_id", "obs", "arc"] + FEATURES)
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    tcl = np.concatenate([c[2] for c in cands])
    rank = np.concatenate([np.arange(n) for n in n_c])
    dmin = np.repeat(np.array([c[1][0] if len(c[1]) else np.nan for c in cands]), n_c)

    a, b = net.a[arc], net.b[arc]
    _, _, traw = point_segment(p[pt], a, b)

    # context window: how well do neighbouring observations fit this arc?
    offs = np.arange(-win, win + 1)
    nb = np.clip(pt[:, None] + offs[None, :], 0, K - 1)
    dctx, _, _ = point_segment(p[nb], a[:, None, :], b[:, None, :])
    hd_nb = angle_diff(head[nb], net.bearing[arc][:, None])
    w_nb = head_rel[nb]
    ctx_head = (hd_nb * w_nb).sum(1) / np.maximum(w_nb.sum(1), 1e-6)
    ctx_head[w_nb.sum(1) == 0] = 90.0
    # mean of the minimum candidate distance in the window (normaliser)
    dmin_pt = np.array([c[1][0] if len(c[1]) else C.CAND_RADIUS_M for c in cands])
    ctx_min_mean = dmin_pt[nb].mean(1)

    hd = angle_diff(head[pt], net.bearing[arc])
    # does the reverse twin of this arc also appear in the candidate set?
    key = net.arcs[arc, 0] * (net.n_nodes + 1) + net.arcs[arc, 1]
    rkey = net.arcs[arc, 1] * (net.n_nodes + 1) + net.arcs[arc, 0]
    df = pd.DataFrame({
        "trip_id": prep["trip"].trip_id, "obs": pt, "arc": arc,
        "dist": dist, "dist_rank": rank, "dist_minus_min": dist - dmin,
        "dist_ratio": dist / (dmin + 1.0), "n_cand": n_c[pt],
        "t_proj": tcl, "t_raw_clip": np.clip(traw, -1.0, 2.0),
        "at_endpoint": ((traw < 0) | (traw > 1)).astype(float),
        "head_diff": np.where(head_rel[pt] > 0, hd, 90.0),
        "head_cos": np.where(head_rel[pt] > 0, np.cos(np.deg2rad(hd)), 0.0),
        "speed": speed[pt], "head_reliable": head_rel[pt],
        "arc_len_log": np.log1p(net.length[arc]),
        "from_in_deg": net.in_deg[net.arcs[arc, 0]],
        "to_out_deg": net.out_deg[net.arcs[arc, 1]],
        "ctx_dist_mean": dctx.mean(1), "ctx_dist_max": dctx.max(1),
        "ctx_within20": (dctx <= 20).mean(1), "ctx_head_diff": ctx_head,
        "ctx_dist_rel": dctx.mean(1) - ctx_min_mean,
        "dt_prev": np.minimum(dt_prev[pt], 600.0),
        "gap_flag": (dt_prev[pt] > C.GAP_S).astype(float),
    })
    twin = pd.MultiIndex.from_arrays([pt, rkey]).isin(pd.MultiIndex.from_arrays([pt, key]))
    df["has_twin"] = twin.astype(float)
    if "label" in prep:
        df["y"] = (arc == prep["label"][pt]).astype(np.int8)
        df["reliable"] = prep["reliable"][pt]
    return df
