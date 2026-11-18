"""Stage 5 - tuned vs untuned comparison, HMM/Viterbi tuning and final
evaluation on STUDENT_TEST, folium maps, and export of inference artifacts.

1. 5-fold GroupKFold out-of-fold (OOF) probabilities on TRAIN for the tuned
   model (same folds as stage 3) -> CV metrics for the tuned model and
   unbiased scores for decoder tuning.
2. Decoder grid search on 30 TRAIN trips using OOF scores (no student-test
   or evaluator data involved).
3. Student-test comparison: classifier-only vs distance-only HMM vs
   classifier+HMM.
4. Export artifacts/pathsnap_config.json for the API.
"""
import json, os, sys, time, shutil
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES, prepare_trip, build_features
from pathsnap.hmm import HMMParams, viterbi, classifier_only, stitch_route
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import (classification_metrics, observation_accuracy, route_continuity,
                              sequence_metrics, top1_from_scores)
from pathsnap.pipeline import feasible_continuity, split_scores
from pathsnap.viz import trip_map

OUT = os.path.join(C.RESULTS, "hmm"); os.makedirs(OUT, exist_ok=True)
OPT = os.path.join(C.RESULTS, "optuna")
MAPS = os.path.join(C.RESULTS, "maps"); os.makedirs(MAPS, exist_ok=True)
best = json.load(open(os.path.join(OPT, "best_params.json")))["best_params"]

tr_all = pd.read_parquet(os.path.join(C.PROCESSED, "train.parquet"))
tr_obs = pd.read_parquet(os.path.join(C.PROCESSED, "train_obs.parquet"))
te = pd.read_parquet(os.path.join(C.PROCESSED, "student_test.parquet"))
te_obs = pd.read_parquet(os.path.join(C.PROCESSED, "student_test_obs.parquet"))
rel = tr_all.reliable.values

# ------------------------------------------------------------------ 1. OOF
oof = np.zeros(len(tr_all))
fold_rows = []
groups = tr_all.trip_id.values
trips_rel = tr_all[rel].trip_id.values
gkf = list(GroupKFold(n_splits=5).split(tr_all[rel], groups=trips_rel))
rel_idx = np.where(rel)[0]
for k, (a, b) in enumerate(gkf):
    val_trips = set(trips_rel[b])
    m = LGBMClassifier(**best, subsample_freq=1, random_state=C.SEED, n_jobs=2, verbose=-1)
    t0 = time.time(); m.fit(tr_all.iloc[rel_idx[a]][FEATURES].values, tr_all.iloc[rel_idx[a]].y.values)
    vmask = tr_all.trip_id.isin(val_trips).values
    oof[vmask] = m.predict_proba(tr_all.loc[vmask, FEATURES].values)[:, 1]
    vb = rel_idx[b]
    cm = classification_metrics(tr_all.y.values[vb], oof[vb])
    d = tr_all.loc[vmask, ["trip_id", "obs", "arc"]].copy(); d["score"] = oof[vmask]
    t1 = top1_from_scores(d)
    mm = tr_obs[tr_obs.trip_id.isin(val_trips) & tr_obs.reliable].merge(t1, on=["trip_id", "obs"], how="left")
    fold_rows.append({"fold": k, "f1_macro": cm["f1_macro"], "log_loss": cm["log_loss"],
                      "top1": float((mm.pred_arc == mm.label).mean()), "train_time_s": time.time() - t0})
    print("OOF fold", fold_rows[-1], flush=True)
pd.DataFrame(fold_rows).to_csv(os.path.join(OPT, "tuned_cv_folds.csv"), index=False)

# ------------------------------------------------- tuned vs untuned on STUDENT_TEST
cmp_rows = []
for name, path in [("LightGBM (untuned)", "LightGBM.joblib"), ("LightGBM (tuned)", "LightGBM_tuned.joblib")]:
    model = joblib.load(os.path.join(C.ARTIFACTS, "models", path))
    t0 = time.time(); p = model.predict_proba(te[FEATURES].values)[:, 1]; inf = time.time() - t0
    r = te.reliable.values
    cm = classification_metrics(te.y.values[r], p[r])
    d = te[["trip_id", "obs", "arc"]].copy(); d["score"] = p
    mm = te_obs[te_obs.reliable].merge(top1_from_scores(d), on=["trip_id", "obs"], how="left")
    cv = (json.load(open(os.path.join(C.RESULTS, "models", "LightGBM.json"))) if "untuned" in name else None)
    cmp_rows.append({"model": name,
                     "cv_f1_macro": cv["cv_f1_macro_mean"] if cv else float(np.mean([f["f1_macro"] for f in fold_rows])),
                     "cv_top1": cv["cv_top1_mean"] if cv else float(np.mean([f["top1"] for f in fold_rows])),
                     "cv_log_loss": cv["cv_log_loss_mean"] if cv else float(np.mean([f["log_loss"] for f in fold_rows])),
                     "test_top1": float((mm.pred_arc == mm.label).mean()), "test_f1_macro": cm["f1_macro"],
                     "test_log_loss": cm["log_loss"], "inference_s_student_test": inf,
                     "n_trees": model.n_estimators})
    if "tuned" in name and "untuned" not in name:
        te_scores = p
pd.DataFrame(cmp_rows).to_csv(os.path.join(OPT, "tuned_vs_untuned.csv"), index=False)
print(pd.DataFrame(cmp_rows).T)

# ------------------------------------------------------- 2. decoder tuning (TRAIN, OOF)
rng = np.random.default_rng(C.SEED)
cand_trips = [t for t in list_trips(C.TRAIN_DIR)
              if tr_obs[tr_obs.trip_id == t].reliable.mean() > 0.8 and (tr_obs.trip_id == t).sum() < 6000]
tune_trips = sorted(rng.choice(cand_trips, size=min(30, len(cand_trips)), replace=False).tolist())
preps = {}
for tid in tune_trips:
    prep = prepare_trip(load_trip(os.path.join(C.TRAIN_DIR, tid)))
    msk = (tr_all.trip_id == tid).values
    preps[tid] = (prep, split_scores(prep, oof[msk]))
grid = ([HMMParams(emission="classifier", alpha=a, beta=b) for a in [0.5, 1.0, 2.0] for b in [3, 5, 10, 20, 40]]
        + [HMMParams(emission="gaussian", sigma=s, beta=b) for s in [5, 10, 20] for b in [3, 5, 10, 20, 40]])
grows = []
for P in grid:
    accs, t0 = [], time.time()
    for tid, (prep, sc) in preps.items():
        pred, _ = viterbi(prep, sc, P)
        accs.append(observation_accuracy(pred, prep["label"], prep["reliable"]))
    grows.append({**P.to_dict(), "train_top1_mean": float(np.mean(accs)), "seconds": time.time() - t0})
    print(grows[-1], flush=True)
base = np.mean([observation_accuracy(classifier_only(p, s), p["label"], p["reliable"]) for p, s in preps.values()])
G = pd.DataFrame(grows); G["classifier_only_top1"] = base
G.to_csv(os.path.join(OUT, "decoder_grid_train.csv"), index=False)
bc = G[G.emission == "classifier"].sort_values("train_top1_mean").iloc[-1]
bg = G[G.emission == "gaussian"].sort_values("train_top1_mean").iloc[-1]
P_best = HMMParams(emission="classifier", alpha=float(bc.alpha), beta=float(bc.beta))
P_gauss = HMMParams(emission="gaussian", sigma=float(bg.sigma), beta=float(bg.beta))
json.dump({"tune_trips": tune_trips, "best_classifier_hmm": P_best.to_dict(), "best_gaussian_hmm": P_gauss.to_dict(),
           "classifier_only_top1_train": base}, open(os.path.join(OUT, "decoder_selection.json"), "w"), indent=2)

fig, ax = plt.subplots(figsize=(6, 3.4))
for a in sorted(G[G.emission == "classifier"].alpha.unique()):
    s = G[(G.emission == "classifier") & (G.alpha == a)]
    ax.plot(s.beta, s.train_top1_mean, marker="o", label=f"classifier emission, alpha={a}")
for sg in sorted(G[G.emission == "gaussian"].sigma.unique()):
    s = G[(G.emission == "gaussian") & (G.sigma == sg)]
    ax.plot(s.beta, s.train_top1_mean, marker="x", ls="--", label=f"gaussian emission, sigma={sg}")
ax.axhline(base, c="k", lw=.8, label="classifier only")
ax.set_xscale("log"); ax.set_xlabel("transition scale beta (m)"); ax.set_ylabel("Top-1 accuracy (30 TRAIN trips, OOF)")
ax.legend(fontsize=6); ax.set_title("HMM decoder tuning")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_decoder_grid.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------- 3. STUDENT_TEST evaluation
rows = []
for tid in list_trips(C.STUDENT_TEST_DIR):
    prep = prepare_trip(load_trip(os.path.join(C.STUDENT_TEST_DIR, tid)))
    msk = (te.trip_id == tid).values
    sc = split_scores(prep, te_scores[msk])
    net, p, ref = prep["net"], prep["p"], prep["trip"].route
    res = {}
    t0 = time.time(); res["classifier_only"] = classifier_only(prep, sc); tc = time.time() - t0
    t0 = time.time(); res["gaussian_hmm"], ig = viterbi(prep, sc, P_gauss); tg = time.time() - t0
    t0 = time.time(); res["classifier_hmm"], ih = viterbi(prep, sc, P_best); th = time.time() - t0
    for name, pred in res.items():
        rows.append({"trip_id": tid, "method": name, "n_obs": len(p),
                     "top1": observation_accuracy(pred, prep["label"], prep["reliable"]),
                     "continuity_strict": route_continuity(pred, net.arcs),
                     "continuity_feasible": feasible_continuity(pred, net, p),
                     **sequence_metrics(pred, ref, net.length),
                     "breaks": {"classifier_only": np.nan, "gaussian_hmm": ig["breaks"], "classifier_hmm": ih["breaks"]}[name],
                     "decode_s": {"classifier_only": tc, "gaussian_hmm": tg, "classifier_hmm": th}[name]})
    st, unf = stitch_route(net, res["classifier_hmm"])
    rows.append({"trip_id": tid, "method": "classifier_hmm_stitched", "n_obs": len(p),
                 **sequence_metrics(st, ref, net.length), "unfilled_holes": unf})
    m = trip_map(net, prep["track"][:, :2], reference_arcs=ref, predicted_arcs=st,
                 classifier_arcs=[a for a in res["classifier_only"] if a >= 0],
                 title=f"STUDENT_TEST {tid}: Top-1 clf {rows[-4]['top1']:.3f} -> HMM {rows[-2]['top1']:.3f}")
    m.save(os.path.join(MAPS, f"student_test_{tid}.html"))
    print(tid, {r["method"]: round(r.get("top1", np.nan), 4) for r in rows[-4:]}, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "student_test_route_eval.csv"), index=False)
summ = R.groupby("method").agg(top1=("top1", "mean"), continuity_strict=("continuity_strict", "mean"),
                               continuity_feasible=("continuity_feasible", "mean"), arc_f1=("arc_f1", "mean"),
                               len_f1=("len_f1", "mean"), breaks=("breaks", "sum"), decode_s=("decode_s", "sum"))
# observation-weighted Top-1 as well
w = R[R.top1.notna()].groupby("method").apply(lambda g: np.average(g.top1, weights=g.n_obs))
summ["top1_obs_weighted"] = w
summ.to_csv(os.path.join(OUT, "student_test_summary.csv")); print(summ)

piv = R[R.method.isin(["classifier_only", "gaussian_hmm", "classifier_hmm"])].pivot(index="trip_id", columns="method", values="top1")
fig, ax = plt.subplots(figsize=(8, 3.4))
x = np.arange(len(piv)); wdt = 0.27
for i, (c, col) in enumerate([("classifier_only", "#ff7f0e"), ("gaussian_hmm", "#9e9e9e"), ("classifier_hmm", "#d62728")]):
    ax.bar(x + (i - 1) * wdt, piv[c], wdt, label=c, color=col)
ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=45, fontsize=7); ax.set_ylim(0.6, 1.0)
ax.set_ylabel("Top-1 accuracy"); ax.set_title("STUDENT_TEST: classifier-only vs HMM decoding per trip"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_hmm_per_trip.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------- 4. export artifacts
shutil.copy(os.path.join(C.ARTIFACTS, "models", "LightGBM_tuned.joblib"), os.path.join(C.ARTIFACTS, "final_model.joblib"))
json.dump({"model_file": "final_model.joblib", "model_name": "LightGBM (Optuna-tuned)",
           "features": FEATURES, "cand_radius_m": C.CAND_RADIUS_M, "cand_k_max": C.CAND_K_MAX,
           "ctx_win": C.CTX_WIN, "gap_s": C.GAP_S, "hmm_params": P_best.to_dict(),
           "lightgbm_params": best},
          open(os.path.join(C.ARTIFACTS, "pathsnap_config.json"), "w"), indent=2)
print("exported artifacts")
