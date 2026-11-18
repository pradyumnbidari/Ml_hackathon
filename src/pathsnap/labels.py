"""Derive per-observation reference labels from the arc-level `.route`.

The `.route` file is an ordered list of arc ids but says nothing about which
GPS observation belongs to which arc. We therefore solve a monotonic
alignment: assign each observation k a route position m_k such that
m_0 <= m_1 <= ... (the vehicle never goes backwards along its own route) and
the sum of point-to-arc distances is minimal. This is a dynamic program

    D[k, m] = C[k, m] + min_{m' <= m} D[k-1, m']

solved in O(K*M) with a running minimum. Arcs may receive zero points (short
arcs passed between two 1 Hz samples), which the running minimum allows.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def route_cost_matrix(p_xy: np.ndarray, a: np.ndarray, b: np.ndarray,
                      chunk: int = 2048) -> np.ndarray:
    K, M = len(p_xy), len(a)
    C = np.empty((K, M), dtype=np.float32)
    for s in range(0, K, chunk):
        d, _, _ = point_segment(p_xy[s:s + chunk, None, :], a[None], b[None])
        C[s:s + chunk] = d
    return C


def align_points_to_route(p_xy: np.ndarray, net, route: np.ndarray):
    """Returns (arc_label per point, route index per point, aligned distance)."""
    a = net.a[route]
    b = net.b[route]
    C = route_cost_matrix(p_xy, a, b)
    K, M = C.shape
    back = np.empty((K, M), dtype=np.int32)
    ar = np.arange(M, dtype=np.int32)
    D = C[0].astype(np.float64)
    back[0] = ar
    for k in range(1, K):
        cm = np.minimum.accumulate(D)
        arg = np.maximum.accumulate(np.where(D <= cm, ar, 0))
        back[k] = arg
        D = C[k] + cm
    # back[k, m] = best position at step k-1 given position m at step k
    idx2 = np.empty(K, dtype=np.int64)
    idx2[-1] = int(np.argmin(D))
    for k in range(K - 1, 0, -1):
        idx2[k - 1] = back[k, idx2[k]]
    dist = C[np.arange(K), idx2]
    return route[idx2], idx2, dist
