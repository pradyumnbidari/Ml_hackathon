"""Stage 6 - consolidated model comparison table and figures for the report.

Also decodes every model's STUDENT_TEST probabilities with the selected HMM
parameters, to check how compatible each model is with the sequential decoder.
Outputs -> results/figures/, results/model_comparison.csv
"""
import json, os, sys
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES, prepare_trip
from pathsnap.hmm import HMMParams, viterbi
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import observation_accuracy
from pathsnap.models import MODEL_SPECS
from pathsnap.pipeline import feasible_continuity, split_scores

FIG = os.path.join(C.RESULTS, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})
names = [n for n in MODEL_SPECS if os.path.exists(os.path.join(C.RESULTS, "models", f"{n}.json"))]
R = {n: json.load(open(os.path.join(C.RESULTS, "models", f"{n}.json"))) for n in names}
P = HMMParams(**json.load(open(os.path.join(C.ARTIFACTS, "pathsnap_config.json")))["hmm_params"])
te = pd.read_parquet(os.path.join(C.PROCESSED, "student_test.parquet"))

# ---- per-model continuity (feasible) and HMM decoding on STUDENT_TEST ----
preps = {t: prepare_trip(load_trip(os.path.join(C.STUDENT_TEST_DIR, t))) for t in list_trips(C.STUDENT_TEST_DIR)}
tuned = joblib.load(os.path.join(C.ARTIFACTS, "models", "LightGBM_tuned.joblib"))
score_sets = {n: pd.read_parquet(os.path.join(C.PROCESSED, f"scores_{n}.parquet")).score.values for n in names}
score_sets["LightGBM (tuned)"] = tuned.predict_proba(te[FEATURES].values)[:, 1]
hmm_rows = []
for n, prob in score_sets.items():
    accs_c, accs_h, cont_c, cont_h, w = [], [], [], [], []
    for tid, prep in preps.items():
        sc = split_scores(prep, prob[(te.trip_id == tid).values])
        from pathsnap.hmm import classifier_only
        pc = classifier_only(prep, sc); ph, _ = viterbi(prep, sc, P)
        accs_c.append(observation_accuracy(pc, prep["label"], prep["reliable"]))
        accs_h.append(observation_accuracy(ph, prep["label"], prep["reliable"]))
        cont_c.append(feasible_continuity(pc, prep["net"], prep["p"]))
        cont_h.append(feasible_continuity(ph, prep["net"], prep["p"]))
        w.append(prep["reliable"].sum())
    hmm_rows.append({"model": n, "test_top1_clf_obsw": np.average(accs_c, weights=w),
                     "test_top1_hmm_obsw": np.average(accs_h, weights=w),
                     "continuity_clf": np.mean(cont_c), "continuity_hmm": np.mean(cont_h)})
    print(hmm_rows[-1], flush=True)
H = pd.DataFrame(hmm_rows).set_index("model")
H.to_csv(os.path.join(C.RESULTS, "model_hmm_compatibility.csv"))

tv = pd.read_csv(os.path.join(C.RESULTS, "optuna", "tuned_vs_untuned.csv")).set_index("model")
rows = []
for n in names:
    r = R[n]
    rows.append({"Model": n, "Family": r["family"], "Key configuration": r["config"],
                 "CV Macro F1": r["cv_f1_macro_mean"], "CV Macro F1 std": r["cv_f1_macro_std"],
                 "CV Top-1": r["cv_top1_mean"], "Train-fold Macro F1": r["cv_train_f1_macro_mean"],
                 "Test Top-1": r["test_top1"], "Test Macro F1": r["test"]["f1_macro"],
                 "Test Precision (macro)": r["test"]["precision_macro"], "Test Recall (macro)": r["test"]["recall_macro"],
                 "Test Log Loss": r["test"]["log_loss"],
                 "Route continuity (top-1)": H.loc[n, "continuity_clf"],
                 "Test Top-1 after HMM": H.loc[n, "test_top1_hmm_obsw"],
                 "Train time (s)": r["train_time_s"], "Inference ms / 1k rows": r["inference_ms_per_1k_rows"],
                 "Candidate miss rate": r["test_miss_rate"]})
t = tv.loc["LightGBM (tuned)"]
rows.append({"Model": "LightGBM (tuned)", "Family": "Boosting", "Key configuration": "Optuna TPE best (see Sec. 4)",
             "CV Macro F1": t.cv_f1_macro, "CV Macro F1 std": np.nan, "CV Top-1": t.cv_top1, "Train-fold Macro F1": np.nan,
             "Test Top-1": t.test_top1, "Test Macro F1": t.test_f1_macro, "Test Precision (macro)": np.nan,
             "Test Recall (macro)": np.nan, "Test Log Loss": t.test_log_loss,
             "Route continuity (top-1)": H.loc["LightGBM (tuned)", "continuity_clf"],
             "Test Top-1 after HMM": H.loc["LightGBM (tuned)", "test_top1_hmm_obsw"],
             "Train time (s)": json.load(open(os.path.join(C.RESULTS, "optuna", "refit.json")))["refit_seconds"],
             "Inference ms / 1k rows": 1000 * t.inference_s_student_test / len(te) * 1000,
             "Candidate miss rate": R[names[0]]["test_miss_rate"]})
T = pd.DataFrame(rows); T.to_csv(os.path.join(C.RESULTS, "model_comparison.csv"), index=False)
print(T.drop(columns=["Key configuration"]).round(4).to_string())

base = T[T.Model != "LightGBM (tuned)"]
cols = ["#3b6ea8" if n not in ("LightGBM", "HistGradientBoosting") else "#d1495b" for n in base.Model]
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(base.Model, base["CV Macro F1"], yerr=base["CV Macro F1 std"], color=cols, capsize=3)
ax.set_ylim(0.75, 1.0); ax.set_ylabel("Macro F1 (5-fold GroupKFold, TRAIN)"); ax.set_title("Cross-validation Macro F1 by model")
plt.xticks(rotation=25, ha="right"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cv_macro_f1.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3))
x = np.arange(len(T)); w = 0.27
ax.bar(x - w, T["Test Top-1"], w, label="Top-1 (classifier only)", color="#ff7f0e")
ax.bar(x, T["Test Top-1 after HMM"], w, label="Top-1 after HMM", color="#d62728")
ax.bar(x + w, T["Test Macro F1"], w, label="Macro F1", color="#3b6ea8")
ax.set_xticks(x); ax.set_xticklabels(T.Model, rotation=25, ha="right"); ax.set_ylim(0.75, 1.0)
ax.set_title("STUDENT_TEST: Top-1 accuracy and Macro F1"); ax.legend(fontsize=7, ncol=3, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_test_top1_f1.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].bar(T.Model, T["Test Log Loss"], color="#6a994e"); ax[0].set_yscale("log"); ax[0].set_ylabel("log loss (log scale)")
ax[0].set_title("STUDENT_TEST log loss (lower is better)")
ax[1].bar(T.Model, T["Inference ms / 1k rows"], color="#8d6a9f"); ax[1].set_yscale("log")
ax[1].set_ylabel("ms per 1,000 candidate rows"); ax[1].set_title("Inference cost")
for a in ax: a.tick_params(axis="x", rotation=35); [l.set_ha("right") for l in a.get_xticklabels()]
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_logloss_inference.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.scatter(base["Train-fold Macro F1"], base["CV Macro F1"], c="#3b6ea8")
for _, r in base.iterrows(): ax.annotate(r.Model, (r["Train-fold Macro F1"], r["CV Macro F1"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
lo = min(base["CV Macro F1"].min(), base["Train-fold Macro F1"].min()) - .01
ax.plot([lo, 1], [lo, 1], "--", c="grey", lw=.8); ax.set_xlabel("train-fold Macro F1"); ax.set_ylabel("validation-fold Macro F1")
ax.set_title("Over/under-fitting check (points below diagonal = gap)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_train_vs_val.png")); plt.close(fig)

n = len(names); fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(2.3 * ((n + 1) // 2), 4.6))
for a, nm in zip(axs.ravel(), names):
    cm = np.array(R[nm]["test"]["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > .5 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticklabels(["true 0", "true 1"])
    a.set_title(nm, fontsize=7)
for a in axs.ravel()[n:]: a.axis("off")
fig.suptitle("STUDENT_TEST confusion matrices (threshold 0.5)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_confusion_matrices.png")); plt.close(fig)

imp = pd.Series(tuned.booster_.feature_importance("gain"), index=FEATURES).sort_values()
imp = imp / imp.sum()
fig, ax = plt.subplots(figsize=(5, 4.2)); ax.barh(imp.index, imp.values, color="#3b6ea8")
ax.set_xlabel("share of total split gain"); ax.set_title("Tuned LightGBM feature importance")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_feature_importance.png")); plt.close(fig)
imp.to_csv(os.path.join(C.RESULTS, "feature_importance.csv"))

fig, ax = plt.subplots(figsize=(7, 3))
cvt = pd.DataFrame({n: [f["top1"] for f in R[n]["cv_folds"]] for n in names})
ax.boxplot(cvt.values, tick_labels=list(cvt.columns)); ax.set_ylabel("Top-1 accuracy per CV fold"); ax.set_title("Stability across GroupKFold folds (TRAIN)")
plt.xticks(rotation=25, ha="right"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cv_fold_stability.png")); plt.close(fig)
print("done")
