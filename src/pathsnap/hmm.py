"""HMM / Viterbi route reconstruction.

Hidden state at observation k : one of the candidate arcs of observation k
                                 (plus the projected position t on that arc).
Emission  log P(z_k | s)       : alpha * log p_clf(s | features)  (classifier)
                                 or a Gaussian on perpendicular distance
                                 (baseline, Newson & Krumm 2009).
Transition log P(s_k | s_k-1)  : exponential on |d_route - d_gc| / beta, where
                                 d_route is the driving distance through the
                                 directed road graph and d_gc the straight-line
                                 distance between the two GPS fixes.
Unreachable transitions get -inf. If every transition into step k is -inf, or
the time gap is too large, the chain is broken and restarted at k (an "HMM
break"); this is reported in the output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


@dataclass
class HMMParams:
    beta: float = 10.0          # transition scale (m)
    alpha: float = 1.0          # weight on classifier log-probability
    sigma: float = 8.0          # GPS sd for the Gaussian emission baseline (m)
    emission: str = "classifier"  # "classifier" | "gaussian"
    max_gap_s: float = 180.0    # break the chain above this time gap
    backward_tol_m: float = 15.0  # tolerated backward motion on the same arc
    eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


def _emission(cand, score, P: HMMParams):
    if P.emission == "gaussian":
        d = cand[1]
        return -0.5 * (d / P.sigma) ** 2
    return P.alpha * np.log(np.clip(score, P.eps, 1.0))


def viterbi(prep: dict, scores: list, P: HMMParams):
    """scores[k] = classifier probabilities aligned with prep['cands'][k]."""
    net, cands, p, t = prep["net"], prep["cands"], prep["p"], prep["track"][:, 2]
    K = len(cands)
    pred = np.full(K, -1, dtype=np.int64)
    breaks = 0
    valid = [k for k in range(K) if len(cands[k][0]) > 0]
    if not valid:
        return pred, {"breaks": 0, "segments": 0}

    segments = []           # list of (obs_indices, backpointers, final delta)
    cur_obs, cur_bp = [], []
    delta = None
    prev = None
    for k in valid:
        ids, dist, tt = cands[k]
        em = _emission(cands[k], scores[k], P)
        if prev is None:
            delta = em.copy(); cur_obs = [k]; cur_bp = [None]; prev = k
            continue
        dt = t[k] - t[prev]
        d_gc = float(np.linalg.norm(p[k] - p[prev]))
        restart = dt > P.max_gap_s
        if not restart:
            pids, _, ptt = cands[prev]
            beta = P.beta * max(1.0, dt / 2.0) if dt > C.GAP_S else P.beta
            limit = max(400.0, 3.0 * d_gc + 200.0)
            n0, n1 = len(pids), len(ids)
            tr = np.full((n0, n1), -np.inf)
            for i in range(n0):
                ai, ti = int(pids[i]), float(ptt[i])
                Li = net.length[ai]
                for j in range(n1):
                    aj, tj = int(ids[j]), float(tt[j])
                    if ai == aj:
                        back = (ti - tj) * Li
                        if back > P.backward_tol_m:
                            dr = net.route_distance(ai, ti, aj, tj, limit)
                        else:
                            dr = abs(tj - ti) * Li
                    else:
                        dr = net.route_distance(ai, ti, aj, tj, limit)
                    if np.isfinite(dr):
                        tr[i, j] = -abs(dr - d_gc) / beta
            tot = delta[:, None] + tr
            best = np.argmax(tot, axis=0)
            newd = tot[best, np.arange(n1)] + em
            if not np.isfinite(newd).any():
                restart = True
        if restart:
            breaks += 1
            segments.append((cur_obs, cur_bp, delta))
            delta = em.copy(); cur_obs = [k]; cur_bp = [None]
        else:
            delta = newd
            cur_obs.append(k); cur_bp.append(best)
        prev = k
    segments.append((cur_obs, cur_bp, delta))

    for obs, bp, d in segments:
        s = int(np.argmax(d))
        for idx in range(len(obs) - 1, -1, -1):
            k = obs[idx]
            pred[k] = cands[k][0][s]
            if bp[idx] is not None:
                s = int(bp[idx][s])
    return pred, {"breaks": breaks, "segments": len(segments)}


def classifier_only(prep: dict, scores: list):
    pred = np.full(len(scores), -1, dtype=np.int64)
    for k, (c, s) in enumerate(zip(prep["cands"], scores)):
        if len(c[0]):
            pred[k] = c[0][int(np.argmax(s))]
    return pred


def stitch_route(net, pred_arcs):
    """Collapse per-observation arcs to a route and fill holes between
    non-adjacent consecutive arcs with shortest paths."""
    seq = []
    for a in pred_arcs:
        if a >= 0 and (not seq or seq[-1] != a):
            seq.append(int(a))
    if not seq:
        return [], 0
    out, unfilled = [seq[0]], 0
    for a in seq[1:]:
        prev = out[-1]
        if net.arcs[prev, 1] != net.arcs[a, 0]:
            path = net.shortest_arc_path(prev, a, limit=2000.0)
            if path is None:
                unfilled += 1
            else:
                out.extend(path)
        out.append(a)
    return out, unfilled
