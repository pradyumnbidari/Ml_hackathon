"""Stage 3 - train and compare candidate classifiers.

* 5-fold GroupKFold on the 80 TRAIN trips (groups = trip id) -> CV metrics
* refit on all TRAIN trips -> evaluate on the 10 STUDENT_TEST trips
Usage: python scripts/03_train_models.py [ModelName ...]   (default: all)
Outputs -> results/models/<model>.json, artifacts/models/<model>.joblib,
           data/processed/scores_<model>.parquet (student-test probabilities)
"""
import json, os, sys, time
import joblib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.geo import LocalProjector
from pathsnap.metrics import (classification_metrics, observation_accuracy, route_continuity,
                              sequence_metrics, top1_from_scores)
from pathsnap.models import MODEL_SPECS

OUT = os.path.join(C.RESULTS, "models"); os.makedirs(OUT, exist_ok=True)
MOD = os.path.join(C.ARTIFACTS, "models"); os.makedirs(MOD, exist_ok=True)

tr = pd.read_parquet(os.path.join(C.PROCESSED, "train.parquet"))
tr_obs = pd.read_parquet(os.path.join(C.PROCESSED, "train_obs.parquet"))
te = pd.read_parquet(os.path.join(C.PROCESSED, "student_test.parquet"))
te_obs = pd.read_parquet(os.path.join(C.PROCESSED, "student_test_obs.parquet"))
tr = tr[tr.reliable].reset_index(drop=True)          # only trusted labels for fitting
groups = tr.trip_id.values
rng = np.random.default_rng(C.SEED)


def top1_acc(df, prob, obs_tab):
    d = df[["trip_id", "obs", "arc"]].copy(); d["score"] = prob
    t1 = top1_from_scores(d)
    m = obs_tab[obs_tab.reliable][["trip_id", "obs", "label"]].merge(t1, on=["trip_id", "obs"], how="left")
    return float((m.pred_arc == m.label).mean()), t1


def fit(name, X, y):
    spec = MODEL_SPECS[name]
    model = spec["build"]()
    sub = spec.get("subsample")
    if sub and len(X) > sub:
        i = rng.choice(len(X), sub, replace=False); X, y = X[i], y[i]
    t0 = time.time(); model.fit(X, y); return model, time.time() - t0


def run(name):
    print(f"==== {name}", flush=True)
    spec = MODEL_SPECS[name]
    X, y = tr[FEATURES].values, tr.y.values
    folds = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m, ttime = fit(name, X[a], y[a])
        prob = m.predict_proba(X[b])[:, 1]
        cm = classification_metrics(y[b], prob)
        val_trips = set(groups[b])
        acc, _ = top1_acc(tr.iloc[b], prob, tr_obs[tr_obs.trip_id.isin(val_trips)])
        ptr = m.predict_proba(X[a][:200_000])[:, 1]
        folds.append({"fold": k, "f1_macro": cm["f1_macro"], "log_loss": cm["log_loss"],
                      "top1": acc, "train_f1_macro": classification_metrics(y[a][:200_000], ptr)["f1_macro"],
                      "train_time_s": ttime})
        print(f"  fold {k}: F1m={cm['f1_macro']:.4f} top1={acc:.4f} ll={cm['log_loss']:.4f} ({ttime:.0f}s)", flush=True)
    # ---- refit on all TRAIN, evaluate on STUDENT_TEST ----
    model, ttime = fit(name, X, y)
    Xte = te[FEATURES].values
    t0 = time.time(); prob = model.predict_proba(Xte)[:, 1]; inf = time.time() - t0
    rel_rows = te.reliable.values
    cm = classification_metrics(te.y.values[rel_rows], prob[rel_rows])
    acc, t1 = top1_acc(te, prob, te_obs)
    # per-trip route metrics on top-1 predictions
    per_trip = []
    for tid, g in te_obs.groupby("trip_id"):
        trip = load_trip(os.path.join(C.STUDENT_TEST_DIR, tid))
        pr = g[["obs"]].merge(t1[t1.trip_id == tid], on="obs", how="left").pred_arc.fillna(-1).astype(int).values
        nxy = LocalProjector.for_points(trip.track[:, :2]).fwd(trip.nodes)
        L = np.linalg.norm(nxy[trip.arcs[:, 1]] - nxy[trip.arcs[:, 0]], axis=1)
        sm = sequence_metrics(pr, trip.route, L)
        per_trip.append({"trip_id": tid, "top1": observation_accuracy(pr, g.label.values, g.reliable.values),
                         "continuity": route_continuity(pr, trip.arcs), **sm})
    pt = pd.DataFrame(per_trip)
    res = {
        "model": name, "family": spec["family"], "config": spec["config"], "loss": spec["loss"],
        "cv_folds": folds,
        "cv_f1_macro_mean": float(np.mean([f["f1_macro"] for f in folds])),
        "cv_f1_macro_std": float(np.std([f["f1_macro"] for f in folds])),
        "cv_top1_mean": float(np.mean([f["top1"] for f in folds])),
        "cv_log_loss_mean": float(np.mean([f["log_loss"] for f in folds])),
        "cv_train_f1_macro_mean": float(np.mean([f["train_f1_macro"] for f in folds])),
        "test": {k: v for k, v in cm.items()},
        "test_top1": acc,
        "test_miss_rate": float(1 - te_obs[te_obs.reliable].label_in_cands.mean()),
        "test_continuity_mean": float(pt.continuity.mean()),
        "test_arc_f1_mean": float(pt.arc_f1.mean()),
        "test_len_f1_mean": float(pt.len_f1.mean()),
        "train_time_s": ttime, "inference_time_s": inf,
        "inference_ms_per_1k_rows": 1000 * inf / len(Xte) * 1000,
        "per_trip": per_trip,
    }
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    joblib.dump(model, os.path.join(MOD, f"{name}.joblib"), compress=3)
    pd.DataFrame({"trip_id": te.trip_id, "obs": te.obs, "arc": te.arc, "score": prob}).to_parquet(
        os.path.join(C.PROCESSED, f"scores_{name}.parquet"), index=False)
    print(f"  TEST: F1m={cm['f1_macro']:.4f} top1={acc:.4f} ll={cm['log_loss']:.4f} "
          f"cont={pt.continuity.mean():.4f} train={ttime:.0f}s inf={inf:.2f}s", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(MODEL_SPECS)):
        run(n)
