"""Stage 1 - Exploratory data analysis.

Partitions used: TRAIN (80 trips) and STUDENT_TEST (10 trips) for descriptive
statistics; candidate-radius selection uses TRAIN ONLY.
Outputs -> results/eda/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.geo import LocalProjector, haversine
from pathsnap.io import clean_track, list_trips, load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.network import RoadNetwork, connectivity_stats
from pathsnap.viz import trip_map

OUT = os.path.join(C.RESULTS, "eda")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

rows, gaps_all, align_all, speed_all = [], {"train": [], "student_test": []}, [], []
radius_grid = [15, 20, 30, 40, 50, 60, 80, 100]
radius_rows = []

for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
    for tid in list_trips(pdir):
        tdir = os.path.join(pdir, tid)
        files = {ext: os.path.getsize(os.path.join(tdir, f"{tid}.{ext}"))
                 if os.path.exists(os.path.join(tdir, f"{tid}.{ext}")) else None
                 for ext in ["track", "nodes", "arcs", "route"]}
        trip = load_trip(tdir)
        trk, rep = clean_track(trip.track)
        dt = np.diff(trk[:, 2])
        gaps_all[part].append(dt)
        proj = LocalProjector.for_points(trk[:, :2])
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        p = proj.fwd(trk[:, :2])
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        speed = step / np.maximum(dt, 1e-9)
        speed_all.append(speed)
        # projection sanity check versus haversine
        hv = haversine(trk[:-1, 0], trk[:-1, 1], trk[1:, 0], trk[1:, 1])
        proj_err = float(np.max(np.abs(hv - step) / np.maximum(hv, 1.0)))
        conn = connectivity_stats(len(trip.nodes), trip.arcs)
        route = trip.route
        r_conn = float(np.mean(trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0])) if len(route) > 1 else np.nan
        lab, idx, dist = align_points_to_route(p, net, route)
        align_all.append(pd.DataFrame({"part": part, "trip": tid, "dist": dist}))
        if part == "train":
            cand = net.candidates(p, max(radius_grid), 50)
            for r in radius_grid:
                for kmax in [5, 8, 10, 15]:
                    miss = np.mean([not np.any((c[0][:kmax] == l) & (c[1][:kmax] <= r)) for c, l in zip(cand, lab)])
                    ncand = np.mean([min(kmax, int((c[1] <= r).sum())) for c in cand])
                    radius_rows.append({"trip": tid, "radius": r, "k_max": kmax, "miss": miss, "n_cand": ncand})
        arcset = set(map(tuple, trip.arcs))
        lon, lat = trk[:, 0], trk[:, 1]
        ext = proj.fwd(np.array([[lon.min(), lat.min()], [lon.max(), lat.max()]]))
        rows.append({
            "partition": part, "trip_id": tid,
            **{f"bytes_{k}": v for k, v in files.items()},
            **rep,
            "duration_s": float(trk[-1, 2] - trk[0, 2]),
            "path_length_km": float(step.sum() / 1000),
            "dt_median": float(np.median(dt)), "dt_max": float(dt.max()),
            "frac_dt_eq_1": float(np.mean(dt == 1)), "n_gaps_gt10s": int((dt > C.GAP_S).sum()),
            "speed_p95_mps": float(np.percentile(speed, 95)), "speed_max_mps": float(speed.max()),
            "n_jumps_gt60mps": int((speed > 60).sum()),
            "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
            "extent_x_km": float((ext[1, 0] - ext[0, 0]) / 1000), "extent_y_km": float((ext[1, 1] - ext[0, 1]) / 1000),
            "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
            "arc_len_median_m": float(np.median(net.length)), "arc_len_max_m": float(net.length.max()),
            "frac_two_way": float(np.mean([(b, a) in arcset for a, b in map(tuple, trip.arcs)])),
            **conn,
            "route_len_arcs": len(route), "route_len_km": float(net.length[route].sum() / 1000),
            "route_consecutive_connected": r_conn,
            "route_unique_arcs": int(len(np.unique(route))),
            "route_arcs_with_points": float(len(np.unique(idx)) / len(route)),
            "align_dist_median": float(np.median(dist)), "align_dist_p99": float(np.percentile(dist, 99)),
            "align_dist_max": float(dist.max()),
            "proj_rel_err_max": proj_err,
        })
        print(part, tid, len(trk), "pts", len(trip.arcs), "arcs")
        if tid in ("00000048", "00000000"):
            m = trip_map(net, trk[:, :2], reference_arcs=route, title=f"{part} trip {tid}: raw GPS vs reference")
            m.save(os.path.join(OUT, f"map_{part}_{tid}.html"))

inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT, "trip_inventory.csv"), index=False)
align = pd.concat(align_all)
align.to_csv(os.path.join(OUT, "alignment_distances.csv.gz"), index=False)
rad = pd.DataFrame(radius_rows).groupby(["radius", "k_max"])[["miss", "n_cand"]].mean().reset_index()
rad.to_csv(os.path.join(OUT, "candidate_radius_study_train.csv"), index=False)

# ---------------- summary tables ----------------
summ = inv.groupby("partition").agg(
    trips=("trip_id", "count"), obs_total=("n_clean", "sum"), obs_min=("n_clean", "min"),
    obs_median=("n_clean", "median"), obs_max=("n_clean", "max"),
    invalid=("n_invalid", "sum"), dup_ts=("n_duplicate_ts", "sum"), out_of_order=("n_out_of_order", "sum"),
    gaps_gt10s=("n_gaps_gt10s", "sum"), nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
    route_arcs_median=("route_len_arcs", "median"), km_median=("route_len_km", "median"))
summ.to_csv(os.path.join(OUT, "partition_summary.csv"))
print(summ.T)

dq = {
    "missing_or_invalid_gps": int(inv.n_invalid.sum()),
    "duplicate_timestamps": int(inv.n_duplicate_ts.sum()),
    "out_of_order_timestamps": int(inv.n_out_of_order.sum()),
    "frac_steps_exactly_1s": float(np.mean(np.concatenate(gaps_all["train"] + gaps_all["student_test"]) == 1)),
    "gaps_gt10s_total": int(inv.n_gaps_gt10s.sum()),
    "trips_with_gaps_gt10s": inv.loc[inv.n_gaps_gt10s > 0, "trip_id"].tolist(),
    "max_gap_s": float(inv.dt_max.max()),
    "speed_jumps_gt60mps_total": int(inv.n_jumps_gt60mps.sum()),
    "routes_fully_connected": int((inv.route_consecutive_connected == 1).sum()),
    "trips_with_isolated_nodes": int((inv.isolated_nodes > 0).sum()),
    "max_weak_components": int(inv.weak_components.max()),
    "projection_max_rel_error": float(inv.proj_rel_err_max.max()),
    "align_dist_p50_train": float(align[align.part == "train"].dist.median()),
    "align_dist_p99_train": float(align[align.part == "train"].dist.quantile(.99)),
    "align_frac_gt30m_train": float((align[align.part == "train"].dist > 30).mean()),
}
json.dump(dq, open(os.path.join(OUT, "data_quality.json"), "w"), indent=2)
print(json.dumps(dq, indent=2))

# ---------------- plots ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for part, c in [("train", "#3b6ea8"), ("student_test", "#d1495b")]:
    s = inv[inv.partition == part]
    ax[0].hist(s.n_clean, bins=25, alpha=.7, color=c, label=part)
    ax[1].hist(s.route_len_km, bins=25, alpha=.7, color=c, label=part)
    ax[2].hist(np.log10(s.n_arcs), bins=25, alpha=.7, color=c, label=part)
ax[0].set_xlabel("GPS observations per trip"); ax[1].set_xlabel("Reference route length (km)")
ax[2].set_xlabel("log10(# arcs in road network)")
for a in ax: a.set_ylabel("trips"); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_trip_size_distributions.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
dts = np.concatenate(gaps_all["train"])
vals, cnts = np.unique(np.minimum(dts, 30), return_counts=True)
ax[0].bar(vals, cnts, color="#3b6ea8"); ax[0].set_yscale("log")
ax[0].set_xlabel("time step between observations (s, capped at 30)"); ax[0].set_ylabel("count (log)")
ax[0].set_title("Sampling intervals - TRAIN")
spd = np.concatenate(speed_all)
ax[1].hist(np.clip(spd, 0, 60), bins=60, color="#6a994e"); ax[1].set_yscale("log")
ax[1].set_xlabel("implied speed (m/s, capped at 60)"); ax[1].set_title("Implied speed - ALL partitions")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sampling_and_speed.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(np.clip(align[align.part == "train"].dist, 0, 80), bins=80, color="#8d6a9f")
ax.set_xlabel("distance GPS point -> aligned reference arc (m)"); ax.set_ylabel("observations")
ax.set_title("GPS noise relative to reference route - TRAIN")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gps_noise_vs_reference.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for k in [5, 8, 10, 15]:
    s = rad[rad.k_max == k]
    ax[0].plot(s.radius, 100 * s.miss, marker="o", label=f"k_max={k}")
    ax[1].plot(s.radius, s.n_cand, marker="o", label=f"k_max={k}")
ax[0].set_xlabel("candidate radius (m)"); ax[0].set_ylabel("candidate miss rate (%)"); ax[0].set_yscale("log")
ax[1].set_xlabel("candidate radius (m)"); ax[1].set_ylabel("mean candidates / observation")
ax[0].legend(); ax[0].set_title("Candidate miss rate - TRAIN"); ax[1].set_title("Candidate set size - TRAIN")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_candidate_radius_study.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for part, c in [("train", "#3b6ea8"), ("student_test", "#d1495b")]:
    s = inv[inv.partition == part]
    ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=12, c=c, label=part)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(); ax.set_title("Trip centroids (worldwide)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_trip_locations.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].scatter(inv.n_arcs, inv.weak_components, s=10, c="#3b6ea8"); ax[0].set_xscale("log")
ax[0].set_xlabel("# arcs"); ax[0].set_ylabel("# weakly connected components")
ax[1].hist(inv.largest_strong_frac, bins=20, color="#6a994e")
ax[1].set_xlabel("largest strongly-connected component (fraction of nodes)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_network_connectivity.png")); plt.close(fig)

# example trajectory plot (static) for one train trip
trip = load_trip(os.path.join(C.TRAIN_DIR, "00000048"))
fig, ax = plt.subplots(figsize=(5, 5))
P = LocalProjector.for_points(trip.track[:, :2]); net = RoadNetwork(trip.nodes, trip.arcs, P)
xy = P.fwd(trip.track[:, :2]); lo, hi = xy.min(0) - 200, xy.max(0) + 200
sel = ((net.mid > lo) & (net.mid < hi)).all(1)
for a, b in zip(net.a[sel], net.b[sel]): ax.plot([a[0], b[0]], [a[1], b[1]], c="#bbbbbb", lw=.6)
for ai in trip.route: ax.plot([net.a[ai, 0], net.b[ai, 0]], [net.a[ai, 1], net.b[ai, 1]], c="#1b9e3e", lw=2.5)
ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#1f4e9e")
ax.set_aspect("equal"); ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
ax.set_title("TRAIN trip 00000048: GPS (blue), route (green), network (grey)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_example_trajectory.png")); plt.close(fig)
print("EDA done")
