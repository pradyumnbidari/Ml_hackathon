"""Road-network representation: directed arcs as straight segments between
nodes, a spatial index for candidate search and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjector, point_segment, bearing_xy


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray,
                 proj: LocalProjector):
        self.proj = proj
        self.nodes_ll = nodes_lonlat
        self.nodes = proj.fwd(nodes_lonlat)
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(self.nodes)
        self.a = self.nodes[self.arcs[:, 0]]
        self.b = self.nodes[self.arcs[:, 1]]
        d = self.b - self.a
        self.length = np.linalg.norm(d, axis=1)
        self.bearing = bearing_xy(d[:, 0], d[:, 1])
        self.mid = 0.5 * (self.a + self.b)
        self._index_spacing = None
        self._tree = None
        # CSR adjacency over nodes (outgoing arcs)
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self.out_arc = order
        self.out_ptr = np.searchsorted(self.arcs[order, 0], np.arange(self.n_nodes + 1))
        self.out_deg = np.diff(self.out_ptr)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    # ------------------------------------------------------------------ #
    def _build_index(self, spacing: float):
        """KD-tree over points sampled every `spacing` m along every arc."""
        n_samp = np.maximum(1, np.ceil(self.length / spacing).astype(int)) + 1
        arc_of = np.repeat(np.arange(len(self.arcs)), n_samp)
        # fractional positions 0..1 inside each arc
        starts = np.r_[0, np.cumsum(n_samp)[:-1]]
        idx_in = np.arange(n_samp.sum()) - np.repeat(starts, n_samp)
        frac = idx_in / np.repeat(n_samp - 1, n_samp)
        pts = self.a[arc_of] + frac[:, None] * (self.b[arc_of] - self.a[arc_of])
        self._tree = cKDTree(pts)
        self._samp_arc = arc_of
        self._index_spacing = spacing

    def candidates(self, p_xy: np.ndarray, radius: float, k_max: int):
        """For each point return up to k_max arcs whose segment lies within
        `radius` metres, sorted by perpendicular distance.

        Returns list of (arc_ids, dist, t) per point.
        """
        spacing = max(radius / 2.0, 5.0)
        if self._index_spacing != spacing:
            self._build_index(spacing)
        out = []
        lists = self._tree.query_ball_point(p_xy, r=radius + spacing / 2.0 + 1e-6)
        for i, sids in enumerate(lists):
            if len(sids) == 0:
                out.append((np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
                continue
            ids = np.unique(self._samp_arc[np.asarray(sids)])
            d, t, _ = point_segment(p_xy[i], self.a[ids], self.b[ids])
            keep = d <= radius
            ids, d, t = ids[keep], d[keep], t[keep]
            o = np.argsort(d, kind="stable")[:k_max]
            out.append((ids[o], d[o], t[o]))
        return out

    # ------------------------------------------------------------------ #
    def node_dists(self, src: int, limit: float) -> dict:
        """Bounded Dijkstra from node `src` over directed arcs (cached)."""
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(self.out_ptr[u], self.out_ptr[u + 1]):
                ai = self.out_arc[k]
                v = int(self.arcs[ai, 1])
                nd = d + self.length[ai]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._sp_cache[src] = (limit, dist)
        return dist

    def route_distance(self, arc_i: int, t_i: float, arc_j: int, t_j: float,
                       limit: float) -> float:
        """Driving distance from position t_i on arc_i to t_j on arc_j."""
        if arc_i == arc_j and t_j >= t_i - 1e-6:
            return (t_j - t_i) * self.length[arc_i]
        rem = (1.0 - t_i) * self.length[arc_i]
        head = t_j * self.length[arc_j]
        d = self.node_dists(int(self.arcs[arc_i, 1]), limit).get(int(self.arcs[arc_j, 0]))
        if d is None:
            return np.inf
        return rem + d + head

    def shortest_arc_path(self, arc_i: int, arc_j: int, limit: float = 3000.0):
        """Arc sequence strictly between arc_i and arc_j (for route stitching)."""
        src, dst = int(self.arcs[arc_i, 1]), int(self.arcs[arc_j, 0])
        if src == dst:
            return []
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(self.out_ptr[u], self.out_ptr[u + 1]):
                ai = self.out_arc[k]
                v = int(self.arcs[ai, 1])
                nd = d + self.length[ai]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = ai
                    heapq.heappush(pq, (nd, v))
        if dst not in prev:
            return None
        path, v = [], dst
        while v != src:
            ai = prev[v]
            path.append(int(ai))
            v = int(self.arcs[ai, 0])
        return path[::-1]

    def connected(self, arc_i: int, arc_j: int) -> bool:
        return arc_i == arc_j or self.arcs[arc_i, 1] == self.arcs[arc_j, 0]


def connectivity_stats(n_nodes: int, arcs: np.ndarray) -> dict:
    """Weak/strong components and isolated nodes (for EDA)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    g = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.ravel(), minlength=n_nodes)
    return {
        "weak_components": int(nw),
        "largest_weak_frac": float(np.bincount(lw).max() / n_nodes),
        "strong_components": int(ns),
        "largest_strong_frac": float(np.bincount(ls).max() / n_nodes),
        "isolated_nodes": int((deg == 0).sum()),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "duplicate_arcs": int(len(arcs) - len(np.unique(arcs, axis=0))),
    }
