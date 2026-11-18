"""Feature selection by group ablation (TRAIN only).

LightGBM (untuned config, 200 trees for speed) is retrained with each feature
group removed, and with only the top-k features by gain. 3-fold GroupKFold on
a fixed 30% sample of TRAIN fixes (all candidates of a sampled fix kept).
Outputs -> results/feature_selection/ablation.csv
"""
import os, sys, time
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.metrics import classification_metrics
import joblib

OUT = os.path.join(C.RESULTS, "feature_selection"); os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(os.path.join(C.PROCESSED, "train.parquet")); tr = tr[tr.reliable]
keep = tr[["trip_id", "obs"]].drop_duplicates().sample(frac=0.3, random_state=C.SEED)
d = tr.merge(keep, on=["trip_id", "obs"]).reset_index(drop=True)
folds = list(GroupKFold(n_splits=3).split(d, groups=d.trip_id))
GROUPS = {
    "context window": ["ctx_dist_mean", "ctx_dist_max", "ctx_within20", "ctx_head_diff", "ctx_dist_rel"],
    "direction": ["head_diff", "head_cos", "speed", "head_reliable"],
    "geometry (relative)": ["dist_rank", "dist_minus_min", "dist_ratio", "n_cand"],
    "projection position": ["t_proj", "t_raw_clip", "at_endpoint"],
    "topology": ["arc_len_log", "from_in_deg", "to_out_deg", "has_twin"],
    "timing": ["dt_prev", "gap_flag"],
}
imp = pd.Series(joblib.load(os.path.join(C.ARTIFACTS, "models", "LightGBM.joblib")).booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
sets = {"all 23 features": FEATURES}
for g, fs in GROUPS.items():
    sets[f"without {g}"] = [f for f in FEATURES if f not in fs]
for k in [5, 10, 15, 20]:
    sets[f"top-{k} by gain"] = list(imp.index[:k])
rows = []
for name, fs in sets.items():
    t1, f1, ll = [], [], []
    t0 = time.time()
    for a, b in folds:
        m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=C.SEED, n_jobs=2, verbose=-1)
        m.fit(d.loc[a, fs].values, d.y.values[a])
        p = m.predict_proba(d.loc[b, fs].values)[:, 1]
        cm = classification_metrics(d.y.values[b], p)
        x = d.loc[b, ["trip_id", "obs", "y"]].copy(); x["s"] = p
        t1.append(x.loc[x.groupby(["trip_id", "obs"]).s.idxmax()].y.mean()); f1.append(cm["f1_macro"]); ll.append(cm["log_loss"])
    rows.append({"feature_set": name, "n_features": len(fs), "cv_top1": np.mean(t1), "cv_f1_macro": np.mean(f1),
                 "cv_log_loss": np.mean(ll), "seconds": time.time() - t0})
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); base = R.iloc[0]
R["delta_top1_pts"] = 100 * (R.cv_top1 - base.cv_top1)
R.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
imp.to_csv(os.path.join(OUT, "untuned_lightgbm_gain.csv"))
print(R.round(4).to_string())
