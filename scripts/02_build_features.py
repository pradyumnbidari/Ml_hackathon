"""Stage 2 - candidate generation, label alignment and feature extraction.

TRAIN and STUDENT_TEST trips -> data/processed/{train,student_test}.parquet
Also writes per-observation tables (labels, reliability, candidate miss).
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.io import list_trips, load_trip
from pathsnap.features import prepare_trip, build_features

os.makedirs(C.PROCESSED, exist_ok=True)
summary = []
for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
    frames, obs_frames = [], []
    for tid in list_trips(pdir):
        t0 = time.time()
        prep = prepare_trip(load_trip(os.path.join(pdir, tid)))
        df = build_features(prep)
        frames.append(df)
        K = len(prep["p"])
        in_set = np.array([l in c[0] for l, c in zip(prep["label"], prep["cands"])])
        obs_frames.append(pd.DataFrame({
            "trip_id": tid, "obs": np.arange(K), "t": prep["track"][:, 2],
            "label": prep["label"], "align_dist": prep["align_dist"],
            "reliable": prep["reliable"], "label_in_cands": in_set,
            "n_cand": [len(c[0]) for c in prep["cands"]]}))
        rel = prep["reliable"]
        summary.append({"partition": part, "trip_id": tid, "obs": K, "rows": len(df),
                        "reliable_frac": rel.mean(),
                        "miss_rate_reliable": 1 - in_set[rel].mean() if rel.any() else np.nan,
                        "no_candidate_frac": float(np.mean([len(c[0]) == 0 for c in prep["cands"]])),
                        "secs": time.time() - t0})
        print(part, tid, K, len(df), f"{time.time()-t0:.1f}s")
    pd.concat(frames).to_parquet(os.path.join(C.PROCESSED, f"{part}.parquet"), index=False)
    pd.concat(obs_frames).to_parquet(os.path.join(C.PROCESSED, f"{part}_obs.parquet"), index=False)
s = pd.DataFrame(summary)
s.to_csv(os.path.join(C.RESULTS, "eda", "feature_build_summary.csv"), index=False)
agg = s.groupby("partition").apply(lambda g: pd.Series({
    "obs": g.obs.sum(), "rows": g.rows.sum(), "rows_per_obs": g.rows.sum() / g.obs.sum(),
    "reliable_frac": np.average(g.reliable_frac, weights=g.obs),
    "miss_rate_reliable": np.average(g.miss_rate_reliable.fillna(0), weights=g.obs * g.reliable_frac),
    "no_candidate_frac": np.average(g.no_candidate_frac, weights=g.obs)}))
print(agg)
agg.to_csv(os.path.join(C.RESULTS, "eda", "candidate_summary.csv"))
