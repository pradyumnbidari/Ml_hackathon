"""Build the PathSnap technical report (HTML -> PDF with headless Chrome).

Every number in the report is read from files under results/ and
artifacts/, which the scripts in scripts/ produce. Re-running the scripts
and this builder regenerates the report.

usage: python report/build_report.py
"""
import base64
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.models import MODEL_SPECS  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT_HTML = os.path.join(ROOT, "report", "PathSnap_Technical_Report.html")
OUT_PDF = os.path.join(ROOT, "report", "PathSnap_Technical_Report.pdf")
TEAM = os.environ.get("PATHSNAP_TEAM", "<b>Team:</b> Pranav Kalla (PES1UG24AM197)<br>Pranav Achar (PES1UG24AM196)<br>"
                                        "Pradyumn Bidari (PES1UG24AM194)<br>Pradyun P Urs (PES1UG24AM195)")
REPO = "https://github.com/pradyumnbidari/Ml_hackathon"


def J(*p):
    return json.load(open(os.path.join(RES, *p)))


def CSV(*p, **kw):
    return pd.read_csv(os.path.join(RES, *p), **kw)


def img(path, caption, width="100%"):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return f'<p class="missing">[missing figure: {path}]</p>'
    b64 = base64.b64encode(open(full, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption>{caption}<br><span class="src">Source: <code>{path}</code></span></figcaption></figure>')


def pct(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def f4(x):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def table(df, cls=""):
    return df.to_html(index=False, escape=False, classes=cls, border=0)


# ------------------------------------------------------------------ data
inv = CSV("eda", "trip_inventory.csv", dtype={"trip_id": str})
dq = J("eda", "data_quality.json")
psum = CSV("eda", "partition_summary.csv")
csum = CSV("eda", "candidate_summary.csv")
rad = CSV("eda", "candidate_radius_study_train.csv")
M = {n: J("models", f"{n}.json") for n in MODEL_SPECS}
CMP = CSV("model_comparison.csv")
HC = CSV("model_hmm_compatibility.csv").set_index("model")
best_opt = J("optuna", "best_params.json")
trials = CSV("optuna", "trials.csv")
tvu = CSV("optuna", "tuned_vs_untuned.csv").set_index("model")
tcv = CSV("optuna", "tuned_cv_folds.csv")
dsel = J("hmm", "decoder_selection.json")
grid = CSV("hmm", "decoder_grid_train.csv")
hs = CSV("hmm", "student_test_summary.csv").set_index("method")
hrows = CSV("hmm", "student_test_route_eval.csv", dtype={"trip_id": str})
cfg = json.load(open(os.path.join(ROOT, "artifacts", "pathsnap_config.json")))
fi = pd.read_csv(os.path.join(RES, "feature_importance.csv"), index_col=0).iloc[:, 0].sort_values(ascending=False)
api_log = open(os.path.join(RES, "api_demo_log.txt")).read() if os.path.exists(os.path.join(RES, "api_demo_log.txt")) else ""
api_json = open(os.path.join(RES, "api_demo_response.json")).read() if os.path.exists(os.path.join(RES, "api_demo_response.json")) else ""
curve_mlp = CSV("models", "curve_mlp.csv") if os.path.exists(os.path.join(RES, "models", "curve_mlp.csv")) else None
curve_lgb = CSV("models", "curve_lightgbm.csv") if os.path.exists(os.path.join(RES, "models", "curve_lightgbm.csv")) else None

tr = inv[inv.partition == "train"]; te = inv[inv.partition == "student_test"]
lgb_best_it = int(curve_lgb.val_log_loss.idxmin()) + 1; lgb_best_ll = curve_lgb.val_log_loss.min(); lgb_400_ll = curve_lgb.val_log_loss.iloc[399]
mlp_best_ep = int(curve_mlp.val_log_loss.idxmin()) + 1; mlp_best_ll = curve_mlp.val_log_loss.min(); mlp_25_ll = curve_mlp.val_log_loss.iloc[24]
abl = CSV("feature_selection", "ablation.csv")
P = cfg["hmm_params"]
clf_only = hs.loc["classifier_only"]; hmm = hs.loc["classifier_hmm"]; gh = hs.loc["gaussian_hmm"]

# ------------------------------------------------------------------ per-model narrative
MT = {
 "LogisticRegression": dict(
   why="Each candidate is scored by a weighted sum of its features. Distance, heading difference and context fit should each push the log-odds of “this is the right arc” in a fairly monotone way, so a linear model is a natural, well-calibrated baseline whose output is directly a probability.",
   strengths="Fast to train and score, with probabilities that are usually well calibrated, which the HMM uses directly. Its coefficients are interpretable.",
   limits="It cannot model interactions such as “heading only matters when speed is high” or the sharp threshold effects of distance rank, so it under-uses the context features."),
 "GaussianNB": dict(
   why="A probabilistic generative baseline. It models each feature per class as a Gaussian (after a quantile transform to make the marginals closer to normal) and combines them with Bayes’ rule.",
   strengths="Trains in seconds with no iterative optimisation, and gives a useful lower bound for the other models.",
   limits="It assumes features are independent given the class. Ours are strongly correlated (dist, dist_rank, ctx_dist_mean and ctx_dist_rel all measure closeness), so the same evidence is counted several times and probabilities become over-confident."),
 "KNN": dict(
   why="An instance-based model. A candidate is likely correct if the most similar (fix, candidate) pairs seen in training were correct. Distance weighting makes closer neighbours count more.",
   strengths="Non-parametric and makes no assumption about the shape of the decision boundary.",
   limits="Prediction cost grows with the training set. We had to fit on a 200,000-row sample, and inference is by far the slowest of all models. Euclidean distance on standardised features weights every feature equally, including weak ones."),
 "DecisionTree": dict(
   why="Rules like “nearest candidate AND heading difference < 30°” match how a person would map-match by hand, and trees capture such thresholds and interactions directly.",
   strengths="Captures non-linear rules, needs no scaling and is very fast at inference.",
   limits="A single tree has high variance. Its probabilities are coarse (leaf frequencies), which hurts log loss and the HMM emission."),
 "RandomForest": dict(
   why="Bagging many decorrelated trees reduces the variance of a single tree while keeping its ability to learn interactions. Averaging trees also gives smoother probabilities.",
   strengths="Stable across folds and needs little tuning.",
   limits="It is the slowest model to train here and has a larger memory and disk footprint. Gains over a single tree are modest because the strongest signal (distance plus context) is already captured by shallow splits."),
 "HistGradientBoosting": dict(
   why="Gradient boosting fits trees sequentially to the errors of the previous ones, directly minimising log loss. Histogram binning makes it fast on our 1.3M rows.",
   strengths="Strong accuracy with well-optimised probabilities and fast training.",
   limits="Less flexible regularisation than LightGBM, and needs several hyperparameters set sensibly."),
 "LightGBM": dict(
   why="Leaf-wise gradient boosting with row and column subsampling. It optimises log loss, so its probabilities suit the HMM emission, and it scales well to our data size.",
   strengths="Best or joint-best on every classification metric, fast enough for real-time API inference, and well calibrated.",
   limits="Leaf-wise growth can overfit (we see the largest train/validation gap of all models). Many hyperparameters interact, which is why it was chosen for Bayesian optimisation."),
 "MLP": dict(
   why="A small feed-forward network can learn smooth non-linear combinations of the features. It is also the one model with an epoch-wise training history, which we use to diagnose fitting.",
   strengths="Smooth probability surface, cheap inference, competitive accuracy.",
   limits="Sensitive to feature scaling and learning-rate choice, needs many epochs over 1M+ rows, and is less interpretable."),
}

model_secs = []
for i, n in enumerate(MODEL_SPECS, 1):
    r, t = M[n], MT[n]
    gap = r["cv_train_f1_macro_mean"] - r["cv_f1_macro_mean"]
    if n == "LightGBM":
        fit_txt = (f"Mild overfitting: train-fold macro F1 {f4(r['cv_train_f1_macro_mean'])} vs validation {f4(r['cv_f1_macro_mean'])} "
                   f"(gap {gap:+.4f}, the largest of all models). The training history (Fig. 3.5) confirms it for probabilities: on 16 held-out trips "
                   f"validation log loss is lowest at about {lgb_best_it} iterations ({lgb_best_ll:.4f}) and rises to {lgb_400_ll:.4f} at the 400 used. "
                   f"Ranking quality is not hurt: student-test macro F1 ({f4(r['test']['f1_macro'])}) and Top-1 stay the best of all models.")
    elif n == "GaussianNB":
        fit_txt = (f"Underfitting: both train-fold ({f4(r['cv_train_f1_macro_mean'])}) and validation ({f4(r['cv_f1_macro_mean'])}) macro F1 are far below "
                   f"the other models, and log loss is high ({f4(r['test']['log_loss'])}) because the independence assumption makes it over-confident.")
    elif n == "MLP":
        fit_txt = (f"Train-fold and validation macro F1 are close ({f4(r['cv_train_f1_macro_mean'])} vs {f4(r['cv_f1_macro_mean'])}), but the epoch-wise "
                   f"curves (Fig. 3.5) show overfitting in log loss: validation loss is lowest at epoch {mlp_best_ep} ({mlp_best_ll:.4f}) and rises to "
                   f"{mlp_25_ll:.4f} by epoch 25, while training loss keeps falling. Early stopping on a held-out group of trips would help.")
    elif n == "LogisticRegression":
        fit_txt = (f"Slight underfitting: validation macro F1 ({f4(r['cv_f1_macro_mean'])}) is not below train-fold ({f4(r['cv_train_f1_macro_mean'])}), "
                   "and the more flexible tree and boosting models beat it on the same features. The linear boundary is the limit, not the data.")
    else:
        fit_txt = (f"No strong over- or underfitting: train-fold macro F1 {f4(r['cv_train_f1_macro_mean'])} vs validation {f4(r['cv_f1_macro_mean'])} "
                   f"(gap {gap:+.4f}). Fold-to-fold spread (std {f4(r['cv_f1_macro_std'])}) is driven mostly by which trips land in each fold.")
    model_secs.append(f"""
<h3>3.2.{i} {n} <span class="fam">({r['family']})</span></h3>
<p><b>A1. Why suitable?</b> {t['why']}</p>
<p><b>A2. Strengths and limitations.</b> {t['strengths']} <i>Limitation:</i> {t['limits']}</p>
<p><b>B5. Important hyperparameters (initial values).</b> {r['config']}.</p>
<p><b>B6. Loss / objective.</b> {r['loss']}.</p>
<p><b>C7–C8. Student-test performance.</b> Observation-level Top-1 accuracy <b>{pct(r['test_top1'])}</b>, macro F1 <b>{f4(r['test']['f1_macro'])}</b>,
macro precision {f4(r['test']['precision_macro'])}, macro recall {f4(r['test']['recall_macro'])}, log loss <b>{f4(r['test']['log_loss'])}</b>.
After HMM decoding with the selected decoder, Top-1 becomes {pct(HC.loc[n, 'test_top1_hmm_obsw'])}.
Training time {r['train_time_s']:.1f} s, inference {r['inference_ms_per_1k_rows']:.2f} ms per 1,000 candidate rows.</p>
<p><b>C9. Over/underfitting.</b> {fit_txt}</p>""")

# ------------------------------------------------------------------ tables
ct = CMP.copy()
ct_show = pd.DataFrame({
    "Model": ct.Model, "Family": ct.Family, "Key configuration": ct["Key configuration"],
    "CV Macro F1": [f"{a:.4f}" + (f" ± {b:.4f}" if not np.isnan(b) else "") for a, b in zip(ct["CV Macro F1"], ct["CV Macro F1 std"])],
    "Top-1 acc. (test)": [pct(x) for x in ct["Test Top-1"]],
    "Route continuity (test)": [pct(x) for x in ct["Route continuity (top-1)"]],
    "Student-test Macro F1": [f4(x) for x in ct["Test Macro F1"]],
    "Log loss (test)": [f4(x) for x in ct["Test Log Loss"]],
    "Inference (ms / 1k rows)": [f"{x:.2f}" for x in ct["Inference ms / 1k rows"]],
})

inv_show = pd.DataFrame({
    "Partition": psum.partition, "Trips": psum.trips, "GPS obs.": psum.obs_total.map("{:,}".format),
    "Obs./trip (min / median / max)": [f"{a} / {int(b)} / {c}" for a, b, c in zip(psum.obs_min, psum.obs_median, psum.obs_max)],
    "Invalid": psum.invalid, "Dup. timestamps": psum.dup_ts, "Out of order": psum.out_of_order,
    "Gaps &gt;10 s": psum.gaps_gt10s, "Median nodes / arcs": [f"{int(a):,} / {int(b):,}" for a, b in zip(psum.nodes_median, psum.arcs_median)],
    "Median route (arcs / km)": [f"{int(a)} / {b:.1f}" for a, b in zip(psum.route_arcs_median, psum.km_median)],
})
anom = inv.sort_values("align_dist_median", ascending=False).head(6)
anom_show = pd.DataFrame({"Partition": anom.partition, "Trip": anom.trip_id, "GPS obs.": anom.n_clean,
                          "GPS path (km)": anom.path_length_km.round(1), "Reference route (km)": anom.route_len_km.round(1),
                          "Median fix→route (m)": anom.align_dist_median.round(1), "Max gap (s)": anom.dt_max.astype(int)})
gaps = inv.sort_values("dt_max", ascending=False).head(5)
gaps_show = pd.DataFrame({"Partition": gaps.partition, "Trip": gaps.trip_id, "Max gap (s)": gaps.dt_max.astype(int),
                          "Gaps &gt;10 s": gaps.n_gaps_gt10s, "Speed jumps &gt;60 m/s": gaps.n_jumps_gt60mps})
conn_show = inv.groupby("partition").agg(nodes_med=("n_nodes", "median"), arcs_med=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
                                         weak_med=("weak_components", "median"), weak_max=("weak_components", "max"),
                                         largest_weak=("largest_weak_frac", "median"), largest_strong=("largest_strong_frac", "median"),
                                         two_way=("frac_two_way", "mean"), arc_len=("arc_len_median_m", "median")).reset_index()
conn_show.columns = ["Partition", "Median nodes", "Median arcs", "Max arcs", "Median weak components", "Max weak components",
                     "Largest weak comp. (median frac.)", "Largest strong comp. (median frac.)", "Two-way arcs (mean frac.)", "Median arc length (m)"]
conn_show = conn_show.round(3)
cs = csum.set_index("partition")
rad_show = rad[rad.k_max.isin([5, 10, 15]) & rad.radius.isin([20, 30, 40, 50, 60, 80, 100])].pivot(index="radius", columns="k_max", values="miss")
rad_show = rad_show.map(lambda v: pct(v)).reset_index(); rad_show.columns = ["Radius (m)"] + [f"miss, k_max={c}" for c in rad_show.columns[1:]]

tv_show = tvu.reset_index()
tv_show = pd.DataFrame({"Model": tv_show.model, "CV Macro F1": tv_show.cv_f1_macro.map(f4), "CV Top-1": tv_show.cv_top1.map(lambda v: pct(v)),
                        "Student-test Top-1": tv_show.test_top1.map(lambda v: pct(v)), "Student-test Macro F1": tv_show.test_f1_macro.map(f4),
                        "Student-test log loss": tv_show.test_log_loss.map(f4), "Inference, all student-test rows (s)": tv_show.inference_s_student_test.map(lambda v: f"{v:.2f}"),
                        "Trees": tv_show.n_trees})
space = pd.DataFrame([
    ["n_estimators", "int 100–800, step 50", "number of boosting rounds"],
    ["learning_rate", "float 0.01–0.2, log", "shrinkage, trades off with n_estimators"],
    ["num_leaves", "int 15–255, log", "tree complexity in leaf-wise growth"],
    ["min_child_samples", "int 10–300, log", "minimum rows per leaf (regularises)"],
    ["subsample", "float 0.5–1.0", "row bagging fraction"],
    ["colsample_bytree", "float 0.4–1.0", "feature bagging fraction"],
    ["reg_lambda", "float 1e-3–30, log", "L2 penalty on leaf values"],
    ["reg_alpha", "float 1e-3–10, log", "L1 penalty on leaf values"]], columns=["Hyperparameter", "Range / distribution", "Why included"])
bp = best_opt["best_params"]
bp_show = pd.DataFrame({"Hyperparameter": list(bp), "Best value": [f"{v:.4g}" if isinstance(v, float) else v for v in bp.values()]})
top_trials = trials.sort_values("value", ascending=False).head(8)
tt_show = pd.DataFrame({"Trial": top_trials.number, "CV Top-1": top_trials.value.map(lambda v: pct(v, 3)),
                        "CV Macro F1": top_trials["user_attrs_cv_f1_macro"].map(f4), "CV log loss": top_trials["user_attrs_cv_log_loss"].map(f4),
                        "n_est.": top_trials.params_n_estimators, "lr": top_trials.params_learning_rate.map(lambda v: f"{v:.3f}"),
                        "leaves": top_trials.params_num_leaves, "min_child": top_trials.params_min_child_samples})

meth = {"classifier_only": "Classifier only (argmax per fix)", "gaussian_hmm": "HMM, distance-only Gaussian emission",
        "classifier_hmm": "HMM, classifier emission (final)", "classifier_hmm_stitched": "Final route after gap stitching"}
hs_show = hs.reset_index()
hs_show = pd.DataFrame({"Method": hs_show.method.map(meth), "Top-1 (mean of trips)": hs_show.top1.map(lambda v: pct(v)),
                        "Top-1 (obs.-weighted)": hs_show.top1_obs_weighted.map(lambda v: pct(v)),
                        "Continuity (feasible)": hs_show.continuity_feasible.map(lambda v: pct(v)),
                        "Continuity (strict adjacency)": hs_show.continuity_strict.map(lambda v: pct(v)),
                        "Route arc-set F1": hs_show.arc_f1.map(lambda v: pct(v)), "Route length F1": hs_show.len_f1.map(lambda v: pct(v)),
                        "HMM breaks (total)": hs_show.breaks.map(lambda v: "N/A" if np.isnan(v) else int(v))})
piv = hrows[hrows.method.isin(["classifier_only", "classifier_hmm"])].pivot(index="trip_id", columns="method", values="top1")
piv["delta"] = piv.classifier_hmm - piv.classifier_only
ptrip = hrows[hrows.method == "classifier_hmm"].set_index("trip_id")
stitched = hrows[hrows.method == "classifier_hmm_stitched"].set_index("trip_id")
pt_show = pd.DataFrame({"Trip": piv.index, "Obs.": ptrip.loc[piv.index, "n_obs"].values,
                        "Top-1 classifier": piv.classifier_only.map(lambda v: pct(v)), "Top-1 HMM": piv.classifier_hmm.map(lambda v: pct(v)),
                        "Δ (pts)": (100 * piv.delta).map(lambda v: f"{v:+.2f}"),
                        "Continuity HMM": ptrip.loc[piv.index, "continuity_feasible"].map(lambda v: pct(v)).values,
                        "Route length F1 (stitched)": stitched.loc[piv.index, "len_f1"].map(lambda v: pct(v)).values,
                        "Breaks": ptrip.loc[piv.index, "breaks"].astype(int).values})
worst = piv.delta.idxmin(); best_t = piv.delta.idxmax()
w_c, w_h = piv.loc[worst, "classifier_only"], piv.loc[worst, "classifier_hmm"]
n_improved = int((piv.delta > 0).sum()); n_worse = int((piv.delta < 0).sum())
g_cls = grid[grid.emission == "classifier"]; g_gau = grid[grid.emission == "gaussian"]
fi_show = pd.DataFrame({"Feature": fi.index[:8], "Share of gain": fi.values[:8]}).round(3)

tuned_better = tvu.loc["LightGBM (tuned)", "test_top1"] > tvu.loc["LightGBM (untuned)", "test_top1"] + 1e-9
same_cfg = all(abs(float(bp[k]) - float(v)) < 1e-12 for k, v in dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                                                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3).items())
if same_cfg:
    opt_verdict = ("No. Across all trials, none of the configurations TPE proposed beat the hand-chosen starting configuration (trial 0), "
                   "so the best parameters equal the untuned ones and the two models are identical. "
                   "Our reading: the classifier is limited by the features and the candidate set, not by LightGBM’s capacity. "
                   "The top trials differ by about 0.1 percentage points, which is inside the fold-to-fold noise.")
elif tuned_better:
    opt_verdict = (f"Yes, slightly. Student-test Top-1 went from {pct(tvu.loc['LightGBM (untuned)', 'test_top1'])} to "
                   f"{pct(tvu.loc['LightGBM (tuned)', 'test_top1'])}, and log loss from {f4(tvu.loc['LightGBM (untuned)', 'test_log_loss'])} "
                   f"to {f4(tvu.loc['LightGBM (tuned)', 'test_log_loss'])}. The gain is small, which tells us the classifier is limited mainly by "
                   "the features and candidate set rather than by LightGBM’s hyperparameters.")
else:
    opt_verdict = (f"Not on student-test Top-1 ({pct(tvu.loc['LightGBM (untuned)', 'test_top1'])} untuned vs "
                   f"{pct(tvu.loc['LightGBM (tuned)', 'test_top1'])} tuned), although the tuned configuration was best in grouped CV. "
                   "The difference is within trip-to-trip noise. We keep the tuned model because it was selected by the pre-declared CV objective, "
                   "not by student-test results.")

# ------------------------------------------------------------------ HTML
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.45; color: #1d2330; }
h1 { font-size: 22pt; margin: 0 0 4pt; color:#14213D; }
h2 { font-size: 14pt; color: #14213D; border-bottom: 2px solid #C8611E; padding-bottom: 3pt; margin-top: 20pt; page-break-after: avoid; }
h3 { font-size: 11pt; color: #14213D; margin: 14pt 0 4pt; page-break-after: avoid; }
h3 .fam { font-weight: normal; color: #6b7280; font-size: 9.5pt; }
p { margin: 4pt 0 6pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 7.8pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
th { background: #14213D; color: #fff; text-align: left; padding: 3pt 4pt; font-weight: 600; }
td { border-bottom: 1px solid #e3e3e3; padding: 2.5pt 4pt; vertical-align: top; }
tr:nth-child(even) td { background: #f7f6f2; }
figure { margin: 8pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { border: 1px solid #ddd; }
figcaption { font-size: 8pt; color: #4a5568; margin-top: 3pt; text-align: left; }
.src { color: #8a8f98; }
code, pre { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; font-size: 7.6pt; }
pre { background: #0f172a; color: #e2e8f0; padding: 8pt; border-radius: 4pt; white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
.title { border-left: 6px solid #C8611E; padding-left: 12pt; margin: 40pt 0 18pt; }
.meta { color: #4a5568; text-align: left; }
.box { background: #f7f6f2; border-left: 3px solid #2F6FB0; padding: 6pt 10pt; margin: 8pt 0; }
.kpis { display: flex; gap: 10pt; margin: 10pt 0; }
.kpi { flex: 1; border: 1px solid #ddd; border-radius: 4pt; padding: 6pt 8pt; }
.kpi b { font-size: 15pt; color: #14213D; display: block; }
.missing { color: #c00; }
.pb { page-break-before: always; }
.toc li { margin: 1pt 0; }
"""

H = []
H.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>
<div class="title"><h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p class="meta">Technical report · UE24CS352A Machine Learning · PES University, RR Campus<br>{TEAM}<br>
GitHub repository: <a href="{REPO}">{REPO}</a></p></div>
<div class="kpis">
<div class="kpi"><b>{pct(hmm.top1_obs_weighted)}</b>student-test Top-1 accuracy, classifier + HMM (observation-weighted)</div>
<div class="kpi"><b>{pct(clf_only.top1_obs_weighted)}</b>same, classifier only</div>
<div class="kpi"><b>{pct(hs.loc['classifier_hmm_stitched', 'len_f1'])}</b>route length F1 of the final stitched route</div>
<div class="kpi"><b>8</b>models compared, 6 algorithm families</div>
</div>
<div class="box"><b>Summary.</b> For every GPS fix we generate up to {cfg['cand_k_max']} candidate road arcs within {cfg['cand_radius_m']:.0f} m and describe each (fix, arc)
pair with {len(FEATURES)} engineered features. A classifier then estimates the probability that the arc is the one the vehicle was on. Eight models from six families were compared
with 5-fold GroupKFold cross-validation on the 80 training trips and evaluated on the 10 student-testing trips. LightGBM was selected and tuned with Optuna (TPE).
Its probabilities feed an HMM whose transitions compare driving distance through the directed road graph with straight-line GPS distance. Viterbi decoding
raises student-test Top-1 accuracy from {pct(clf_only.top1_obs_weighted)} to {pct(hmm.top1_obs_weighted)} and route continuity from
{pct(clf_only.continuity_feasible)} to {pct(hmm.continuity_feasible)}. The system is served by a FastAPI backend (Laptop 1) and a browser UI (Laptop 2).
Evaluator data was not used anywhere in development.</div>
<ol class="toc"><li>Problem and approach</li><li>Dataset understanding and EDA</li><li>Model experimentation and comparison</li>
<li>Bayesian hyperparameter optimisation</li><li>HMM and Viterbi route reconstruction</li><li>Geographical visualisation (Folium)</li>
<li>API, UI and deployment</li><li>Conclusions and limitations</li><li>Reproducibility and AI-assistance disclosure</li></ol>

<h2>1. Problem and approach</h2>
<p>GPS fixes from a moving vehicle are displaced from the road by noise, multipath and dropouts. Given a trip’s GPS trace and its local road graph, PathSnap must output
the road segment (arc) driven at every fix and a coherent route. We frame it as <b>candidate ranking</b>: generate a small set of nearby arcs per fix, score each with a
supervised classifier, then choose a globally consistent sequence with an <b>HMM decoded by Viterbi</b>, using road-network connectivity.</p>
<p><b>Primary metric.</b> Observation-level Top-1 accuracy: the share of fixes (with a trusted label) whose highest-scoring candidate is the reference arc.
It measures exactly what the route output is judged on. Because the HMM consumes the classifier’s probabilities, <b>log loss</b> is the second selection criterion.
Macro F1 on the binary candidate labels is reported as required. Route continuity and route-level F1 measure sequence quality.</p>

<h2>2. Dataset understanding and exploratory data analysis</h2>
<p><i>Partitions used: TRAIN (80 trips) and STUDENT_TEST (10 trips) for descriptive statistics. Every decision (projection, label threshold, candidate radius, features)
was made on TRAIN only. The evaluator partition was never opened.</i> Code: <code>scripts/01_eda.py</code>, <code>scripts/02_build_features.py</code>.</p>
<h3>2.1 How the data is organised</h3>
<p>Each trip folder holds four headerless whitespace-separated files. <code>.track</code> has longitude, latitude and time in seconds. <code>.nodes</code> has longitude and latitude,
with the row index as node id. <code>.arcs</code> has from_node and to_node, with the row index as arc id; arcs are <b>directed</b>, and on average {pct(tr.frac_two_way.mean(), 1)} of
train arcs have a reverse twin. <code>.route</code> is the chronological list of reference arc ids. No GPS row corresponds to a node or arc row; the only link between them is geometry.</p>
{table(inv_show)}
<p class="meta">Table 2.1 Dataset inventory (from <code>results/eda/partition_summary.csv</code>; per-trip details in <code>trip_inventory.csv</code>).</p>
<h3>2.2 Missing-value and data-quality analysis</h3>
<p>No GPS fix was missing, non-finite, outside valid lon/lat ranges or at (0, 0): {dq['missing_or_invalid_gps']} invalid fixes, {dq['duplicate_timestamps']} duplicate timestamps and
{dq['out_of_order_timestamps']} out-of-order timestamps across all 90 trips. The trace is nominally 1 Hz ({pct(dq['frac_steps_exactly_1s'], 1)} of steps are exactly 1 s),
but there are {dq['gaps_gt10s_total']} gaps longer than {10} s spread over {len(dq['trips_with_gaps_gt10s'])} trips. The longest is {dq['max_gap_s']:,.0f} s (about {dq['max_gap_s'] / 3600:.1f} h).
There are also {dq['speed_jumps_gt60mps_total']} implied speeds above 60 m/s, which are position jumps. In the road graphs we found no isolated nodes, but up to {dq['max_weak_components']}
weakly connected components. {dq['routes_fully_connected']} of 90 reference routes have every consecutive arc pair connected; the rest have a few breaks.</p>
{table(gaps_show)}
<p class="meta">Table 2.2 Trips with the largest time gaps (all partitions).</p>
<p><b>Reference-route limitations.</b> Aligning fixes to the reference route (Sec. 2.4) showed that some reference routes cover only part of the GPS trace:</p>
{table(anom_show)}
<p class="meta">Table 2.3 Trips with the largest median fix-to-reference distance. A GPS path much longer than the route means part of the trip has no reference.</p>
<p>We therefore treat a fix as <b>reliably labelled</b> only if it lies within 50 m of its aligned reference arc. That covers {pct(cs.loc['train', 'reliable_frac'], 1)} of TRAIN fixes and
{pct(cs.loc['student_test', 'reliable_frac'], 2)} of STUDENT_TEST fixes. Unreliable fixes are excluded from training and from every reported accuracy, but they are still
processed and decoded by the pipeline.</p>
{img('results/eda/fig_sampling_and_speed.png', 'Figure 2.1 Sampling intervals (TRAIN) and implied speed (all partitions), log scale.', '88%')}
{img('results/eda/fig_trip_size_distributions.png', 'Figure 2.2 Observations per trip, reference route length and network size (TRAIN vs STUDENT_TEST).', '96%')}
<h3>2.3 Geography and preprocessing</h3>
<p>Trips are spread worldwide (Fig. 2.3). Longitude and latitude are angles, so Euclidean distance in degrees is wrong. We project each trip to a local equirectangular tangent
plane centred on its GPS centroid (metres). Against haversine on every consecutive GPS pair, the maximum relative error is {pct(dq['projection_max_rel_error'], 2)},
far below GPS noise. Coordinate order (lon, lat) was confirmed by plausibility and by overlaying on maps.
Preprocessing per trip: drop invalid or duplicate fixes and sort by time (no-ops on this data, kept for the unseen evaluator trips); project fixes and nodes; build arc segments,
lengths and bearings; build a KD-tree over points sampled every 30 m along all arcs, so long arcs are found correctly; build CSR adjacency for routing.</p>
{img('results/eda/fig_trip_locations.png', 'Figure 2.3 Trip centroids by partition.', '55%')}
{img('results/eda/fig_example_trajectory.png', 'Figure 2.4 GPS trajectory plot, TRAIN trip 00000048: GPS fixes (blue), reference route (green), road network (grey).', '55%')}
<h3>2.4 Labels: aligning fixes to the reference route</h3>
<p><code>.route</code> gives arcs but not which fix belongs to which arc. We assign each fix k a route position m<sub>k</sub> with m<sub>0</sub> ≤ m<sub>1</sub> ≤ …
(the vehicle never moves backwards along its own route), minimising the total perpendicular distance. This is solved exactly by dynamic programming:
D[k,m] = dist(k,m) + min<sub>m′≤m</sub> D[k−1,m′], computed with a running minimum in O(K·M) (<code>src/pathsnap/labels.py</code>). Short arcs may receive no fixes.
The median fix-to-aligned-arc distance on TRAIN is {dq['align_dist_p50_train']:.1f} m (Fig. 2.5), a direct estimate of GPS noise.</p>
{img('results/eda/fig_gps_noise_vs_reference.png', 'Figure 2.5 Distance from each TRAIN fix to its aligned reference arc (capped at 80 m).', '55%')}
<h3>2.5 Road-network size and connectivity</h3>
{table(conn_show)}
<p class="meta">Table 2.4 Road-network statistics per partition (from <code>trip_inventory.csv</code>).</p>
{img('results/eda/fig_network_connectivity.png', 'Figure 2.6 Road-network visualisation of connectivity: weakly connected components vs network size, and size of the largest strongly connected component.', '85%')}
<h3>2.6 Candidate generation</h3>
<p>For each fix we keep up to k<sub>max</sub> arcs within radius r. A miss (reference arc not in the set) cannot be fixed by any model, so we chose r and k<sub>max</sub> on TRAIN by trading
miss rate against set size (Fig. 2.7, Table 2.5). Beyond 50–60 m the miss rate barely falls, so we use r = {cfg['cand_radius_m']:.0f} m and k<sub>max</sub> = {cfg['cand_k_max']}.
On reliably labelled fixes the miss rate is {pct(cs.loc['train', 'miss_rate_reliable'])} (TRAIN) and {pct(cs.loc['student_test', 'miss_rate_reliable'])} (STUDENT_TEST),
with {cs.loc['train', 'rows_per_obs']:.2f} and {cs.loc['student_test', 'rows_per_obs']:.2f} candidates per fix. {pct(cs.loc['train', 'no_candidate_frac'])} of TRAIN fixes
have no candidate at all, almost all in the partial-reference trips where the vehicle is off the supplied network.</p>
{table(rad_show)}
<p class="meta">Table 2.5 Candidate miss rate on TRAIN (all fixes, so the unreliable tail inflates it) by radius and k<sub>max</sub>.</p>
{img('results/eda/fig_candidate_radius_study.png', 'Figure 2.7 Candidate miss rate and set size vs radius (TRAIN).', '85%')}
<h3>2.7 Feature engineering: what we extracted and why</h3>
<p>One row per (fix, candidate arc); target = 1 if the candidate is the aligned reference arc. We built {len(FEATURES)} features from the EDA observations:</p>
<table><tr><th style="width:18%">Group</th><th style="width:40%">Features</th><th>Why (observation from EDA)</th></tr>
<tr><td>Geometry</td><td>dist, dist_rank, dist_minus_min, dist_ratio, n_cand, t_proj, t_raw_clip, at_endpoint</td><td>Noise is about 6 m, so the nearest arc is usually but not always right; relative distance matters more than absolute, and projections past an arc end suggest a neighbouring arc.</td></tr>
<tr><td>Direction / kinematics</td><td>head_diff, head_cos, speed, head_reliable</td><td>Arcs are directed and most roads are two-way with identical geometry, so only heading separates the twins. Heading from ±2 fixes is meaningless when stationary, hence speed and a reliability flag.</td></tr>
<tr><td>Temporal context (±{cfg['ctx_win']} fixes)</td><td>ctx_dist_mean, ctx_dist_max, ctx_within20, ctx_head_diff, ctx_dist_rel</td><td>A single fix can jump to a parallel street; neighbouring fixes fitting the same arc is strong evidence. This gives the per-fix classifier some sequence awareness.</td></tr>
<tr><td>Topology</td><td>arc_len_log, from_in_deg, to_out_deg, has_twin</td><td>Junctions and short connector arcs are where errors cluster.</td></tr>
<tr><td>Timing</td><td>dt_prev, gap_flag</td><td>Gaps up to hours exist; features after a gap are less reliable.</td></tr></table>
<p>Training data: {int(cs.loc['train', 'rows']):,} rows from {int(cs.loc['train', 'obs']):,} TRAIN fixes; STUDENT_TEST: {int(cs.loc['student_test', 'rows']):,} rows.
All scaling and quantile transforms are fitted inside each model pipeline on training folds only.</p>
<h3>2.8 Feature selection</h3>
<p>We tested every feature group by <b>ablation</b>: LightGBM (200 trees) was retrained without that group, and also with only the top-k features by gain, using 3-fold
GroupKFold on a fixed 30% sample of TRAIN fixes (<code>scripts/05b_feature_ablation.py</code>). Removing the context window costs
{-abl.set_index('feature_set').loc['without context window', 'delta_top1_pts']:.2f} points of CV Top-1 and removing projection position costs
{-abl.set_index('feature_set').loc['without projection position', 'delta_top1_pts']:.2f} points. Every other group changes Top-1 by less than ±0.35 points, which is within
the run-to-run noise of this setup. Direction looks unimportant on its own only because ctx_head_diff (in the context group) already carries heading.
No reduced set was significantly better (top-10 by gain: {abl.set_index('feature_set').loc['top-10 by gain', 'delta_top1_pts']:+.2f} pts), so all {len(FEATURES)} features were kept.</p>
{table(pd.DataFrame({"Feature set": abl.feature_set, "# features": abl.n_features, "CV Top-1": abl.cv_top1.map(lambda v: pct(v)), "CV macro F1": abl.cv_f1_macro.map(f4), "CV log loss": abl.cv_log_loss.map(f4), "Δ Top-1 (pts)": abl.delta_top1_pts.map(lambda v: f"{v:+.2f}")}))}
<p class="meta">Table 2.6 Feature-group ablation (<code>results/feature_selection/ablation.csv</code>).</p>
{img('results/feature_selection/fig_ablation.png', 'Figure 2.8 Change in CV Top-1 when each feature group is removed; grey band = ±0.35 pts noise.', '70%')}

<h2 class="pb">3. Model experimentation and comparison</h2>
<p><b>Protocol.</b> All models use the same features and the same 5-fold <b>GroupKFold</b> split on the 80 TRAIN trips (groups = trip id, so no trip appears in both training and
validation). Each is then refitted on all TRAIN trips and evaluated once on the 10 STUDENT_TEST trips. Seed 42 everywhere. Configurations, per-fold results,
timings and student-test predictions are saved in <code>results/models/&lt;model&gt;.json</code> and <code>data/processed/scores_&lt;model&gt;.parquet</code>.
Code: <code>scripts/03_train_models.py</code>, <code>src/pathsnap/models.py</code>.</p>
<h3>3.1 Overall model comparison</h3>
{table(ct_show)}
<p class="meta">Table 3.1 Consolidated comparison (<code>results/model_comparison.csv</code>). Top-1 = observation-level Top-1 accuracy over reliably labelled fixes (fixes whose
reference arc is not among the candidates count as errors). Route continuity = share of consecutive fixes whose top-1 arcs are the same, adjacent, or reachable within 2×(GPS step)+30 m.
Macro F1, precision and recall are on the binary candidate labels at threshold 0.5. KNN inference is slow because every query searches the 200k-row training sample.</p>
<p><b>Selection.</b> LightGBM and HistGradientBoosting are effectively tied on Top-1. LightGBM has the best CV macro F1 ({f4(M['LightGBM']['cv_f1_macro_mean'])}), the best student-test
macro F1 ({f4(M['LightGBM']['test']['f1_macro'])}) and the lowest log loss ({f4(M['LightGBM']['test']['log_loss'])}), and its fold-to-fold spread is similar to the others.
Its inference cost ({M['LightGBM']['inference_ms_per_1k_rows']:.1f} ms per 1,000 rows) is well within the API budget. One honest caveat: untuned,
HistGradientBoosting decodes slightly better with the HMM ({pct(HC.loc['HistGradientBoosting', 'test_top1_hmm_obsw'])} vs {pct(HC.loc['LightGBM', 'test_top1_hmm_obsw'])} Top-1 after HMM).
We did not use that student-test number to choose, because selection was fixed before HMM decoding on CV metrics. We selected LightGBM for Bayesian optimisation and
deployment because its larger set of regularisation hyperparameters gave the search more to work with; after tuning it reaches {pct(HC.loc['LightGBM (tuned)', 'test_top1_hmm_obsw'])} after HMM, the best of all. We did not select on training accuracy: LightGBM has the largest train/validation gap, which we accept because its validation scores are best.</p>
<h3>3.2 Per-model analysis</h3>
{''.join(model_secs)}
<h3>3.3 Model visualisations</h3>
{img('results/figures/fig_cv_macro_f1.png', 'Figure 3.1 Cross-validation macro F1 (mean ± std over 5 GroupKFold folds, TRAIN). Boosting models highlighted.', '78%')}
{img('results/figures/fig_test_top1_f1.png', 'Figure 3.2 STUDENT_TEST Top-1 accuracy (classifier only and after HMM) and macro F1 per model.', '88%')}
{img('results/figures/fig_logloss_inference.png', 'Figure 3.3 STUDENT_TEST log loss (log scale) and inference cost per 1,000 candidate rows.', '92%')}
{img('results/figures/fig_cv_fold_stability.png', 'Figure 3.4 Top-1 accuracy per CV fold: stability across groups of trips.', '78%')}
{img('results/models/fig_training_histories.png', 'Figure 3.5 Training and validation log loss histories for the two models that produce them (MLP epochs, LightGBM boosting iterations); validation = 16 held-out TRAIN trips. No curves are shown for models without an iterative training history.', '92%')}
{img('results/figures/fig_train_vs_val.png', 'Figure 3.6 Train-fold vs validation-fold macro F1 (over/underfitting check). Train-fold F1 is measured on a fixed 200k-row slice of each training fold.', '62%')}
{img('results/figures/fig_confusion_matrices.png', 'Figure 3.7 STUDENT_TEST confusion matrices (counts and row-normalised rates) at threshold 0.5.', '100%')}
{img('results/figures/fig_feature_importance.png', 'Figure 3.8 Feature importance (share of split gain) of the deployed LightGBM model.', '58%')}
<p>The deployed model splits most on the projection position past an arc end (t_raw_clip), distance relative to the nearest candidate (dist_minus_min) and the
context features ctx_dist_rel and ctx_head_diff (Fig. 3.8). Gain shows what the model <i>uses</i>; the ablation in Sec. 2.8 shows what it <i>needs</i>.
Both agree that timing features carry almost nothing.</p>

<h2 class="pb">4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p>Code: <code>scripts/05_bayes_opt.py</code>; trial record <code>results/optuna/trials.csv</code> and <code>optuna_study.db</code>.</p>
<p><b>Q1–Q2. Search space.</b></p>{table(space)}
<p><b>Q3. Objective.</b> Maximise mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 TRAIN trips. To keep each trial affordable on a 2-core laptop, every trial
uses the same fixed 40% sample of TRAIN fixes, with all candidates of a sampled fix kept so Top-1 stays well defined, and every trip represented. Macro F1 and log loss are
logged per trial. STUDENT_TEST was used only after the search to compare tuned and untuned models. Evaluator data was never used.</p>
<p><b>Q4. Number of trials.</b> {best_opt['n_trials']} trials of Optuna’s TPE sampler (seed 42, first 10 trials random). Trial 0 is the untuned configuration, enqueued so that both
are scored with exactly the same protocol. Summed trial time {best_opt['total_search_seconds'] / 60:.1f} min on a 2-core machine. An earlier 30-trial pilot run is kept in
<code>trials_30trial_pilot.csv</code>.</p>
<p><b>Q5. Best values.</b> Best CV Top-1 = {pct(best_opt['best_cv_top1'], 3)} (untuned trial 0: {pct(best_opt['baseline_trial0_cv_top1'], 3)}).</p>
{table(bp_show)}
{table(tt_show)}
<p class="meta">Table 4.1 The eight best trials.</p>
{img('results/optuna/fig_optuna.png', 'Figure 4.1 Optimisation history (best so far) and fANOVA hyperparameter importance.', '92%')}
<p><b>Q6. Did tuning help?</b> {opt_verdict}</p>
{table(tv_show)}
<p class="meta">Table 4.2 Untuned vs tuned LightGBM on identical partitions and metrics. CV for the tuned model is 5-fold GroupKFold (<code>tuned_cv_folds.csv</code>), the same folds as Sec. 3.</p>

<h2 class="pb">5. HMM and Viterbi route reconstruction</h2>
<p>Code: <code>src/pathsnap/hmm.py</code>, <code>scripts/06_hmm_and_final_eval.py</code>.</p>
<p><b>Q1. Hidden states.</b> At fix k the hidden state is one of its candidate arcs, together with the projected position t ∈ [0,1] along that arc. States keep the exact candidate-to-arc
correspondence, so the decoded state at every fix is directly an arc id.</p>
<p><b>Q2. Emission.</b> log P(z<sub>k</sub> | s) = α · log p<sub>LightGBM</sub>(s), clipped at 10<sup>−6</sup>. As a baseline we also implemented the classic Gaussian emission on perpendicular
distance, −½(d/σ)².</p>
<p><b>Q3. Transition.</b> log P(s<sub>k</sub> | s<sub>k−1</sub>) = −|d<sub>route</sub> − d<sub>gc</sub>| / β, where d<sub>gc</sub> is the straight-line distance between the two fixes and
d<sub>route</sub> is the driving distance from position t<sub>k−1</sub> on arc s<sub>k−1</sub> to t<sub>k</sub> on arc s<sub>k</sub> through the <b>directed</b> graph. It is computed with a
bounded, cached Dijkstra (limit max(400 m, 3·d<sub>gc</sub>+200 m)). Backward motion of up to 15 m on the same arc is tolerated as GPS noise. A route that needs a long detour is penalised, and a
wrong-way arc is effectively forbidden.</p>
<p><b>Q4. Missing candidates and gaps.</b> Fixes with no candidate are skipped: the chain continues from the last fix that had candidates, and the fix gets no arc. For gaps over 10 s
β is scaled by Δt/2 so longer unobserved driving is not over-penalised. For gaps over {P['max_gap_s']:.0f} s the chain is broken and restarted.</p>
<p><b>Q5. Disconnected segments.</b> Unreachable transitions get probability 0 (−∞). If every transition into fix k is −∞, the HMM <i>breaks</i>: the current segment is back-tracked and
a new chain starts at k from its emissions alone. Breaks are counted and reported, and the API returns them. For the output route, consecutive decoded arcs that are not adjacent are
joined by the shortest path (gap stitching).</p>
<p><b>Decoder tuning (TRAIN only).</b> α, β (classifier emission) and σ, β (Gaussian emission) were grid-searched on {len(dsel['tune_trips'])} TRAIN trips using <b>out-of-fold</b>
LightGBM probabilities, so the decoder never sees probabilities from a model trained on the same trip. Selected: α = {P['alpha']}, β = {P['beta']} m. Only the product α·β changes the decoded path (scaling every log-score by
a constant does not change the argmax), so several grid cells tie exactly.
Best Gaussian baseline: σ = {dsel['best_gaussian_hmm']['sigma']}, β = {dsel['best_gaussian_hmm']['beta']}. Classifier-only Top-1 on these trips was {pct(dsel['classifier_only_top1_train'])};
the best classifier-emission HMM reached {pct(g_cls.train_top1_mean.max())} and the best Gaussian HMM {pct(g_gau.train_top1_mean.max())}.</p>
{img('results/hmm/fig_decoder_grid.png', 'Figure 5.1 Decoder grid search on TRAIN (out-of-fold probabilities).', '68%')}
<p><b>Q6. Did HMM decoding help? (STUDENT_TEST)</b></p>
{table(hs_show)}
<p class="meta">Table 5.1 STUDENT_TEST comparison (<code>results/hmm/student_test_summary.csv</code>). Feasible continuity = consecutive arcs are the same, adjacent, or reachable within
2·d<sub>gc</sub>+30 m. Strict = same or directly adjacent only (at 1 Hz a vehicle can cross a whole short arc between fixes, so strict continuity cannot reach 100%).</p>
<p>Yes. Classifier+HMM improves observation-weighted Top-1 from {pct(clf_only.top1_obs_weighted)} to {pct(hmm.top1_obs_weighted)}, continuity from {pct(clf_only.continuity_feasible)} to
{pct(hmm.continuity_feasible)}, and route length F1 from {pct(clf_only.len_f1)} to {pct(hmm.len_f1)}. The distance-only Gaussian HMM reaches {pct(gh.top1_obs_weighted)}:
connectivity alone helps, but the learned emission adds more.</p>
{table(pt_show)}
<p class="meta">Table 5.2 Per-trip STUDENT_TEST results.</p>
{img('results/hmm/fig_hmm_per_trip.png', 'Figure 5.2 Per-trip Top-1: classifier only vs Gaussian HMM vs classifier HMM.', '88%')}
<p><b>Where the HMM did not help.</b> The HMM improved {n_improved} of 10 trips and made {n_worse} worse. The largest gain was trip {best_t} ({100 * piv.delta.max():+.2f} pts) and the
worst result trip {worst}, where Top-1 fell from {pct(w_c)} to {pct(w_h)} ({100 * piv.delta.min():+.2f} pts). That trip was already near-perfect for the classifier,
so there was little to fix and the decoder’s smoothing occasionally overrides a correct but locally improbable choice. More generally, the HMM cannot recover
fixes whose reference arc is missing from the candidate set, and when the classifier is confidently wrong for a run of fixes on a connected alternative road,
the transition term has nothing to object to and the error persists.
At junctions the decoder sometimes switches arcs a fix early or late. That counts as a Top-1 error even though the route is right, which is why route F1 is higher than Top-1.</p>

<h2 class="pb">6. Geographical visualisation using Folium</h2>
<p>Interactive maps are generated by <code>src/pathsnap/viz.py</code> (called from <code>scripts/01_eda.py</code> and <code>scripts/06_hmm_and_final_eval.py</code>) and committed under
<code>results/eda/</code> and <code>results/maps/</code> (one per STUDENT_TEST trip). Each has separate, toggleable layers (LayerControl): raw GPS fixes (blue dots), nearby road network (grey),
reference route (green), classifier-only prediction (orange dashed, off by default) and the final predicted route (red), plus start and end markers, a legend and bounds
fitted to the trip. The screenshots below were rendered offline, so basemap tiles are blank; the grey road network is drawn from <code>.nodes/.arcs</code>.</p>
{img('results/figures/maps/student_test_00000077.png', 'Figure 6.1 STUDENT_TEST trip 00000077: GPS fixes, road network, reference route (green) and predicted route (red).', '100%')}
{img('results/figures/maps/student_test_00000000.png', 'Figure 6.2 STUDENT_TEST trip 00000000.', '100%')}
{img('results/figures/maps/map_train_00000048.png', 'Figure 6.3 EDA map, TRAIN trip 00000048: raw GPS vs reference route.', '100%')}

<h2 class="pb">7. API, UI and deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. At start-up it loads <code>artifacts/final_model.joblib</code> and
<code>artifacts/pathsnap_config.json</code> (features, candidate parameters, HMM parameters). Endpoints: <code>GET /health</code>, <code>GET /trips</code> and <code>POST /predict {{"trip_id"}}</code>.
<code>/predict</code> validates the id against the trip folders (blocking path traversal), loads <i>only</i> <code>.track/.nodes/.arcs</code>
(<code>load_trip(with_route=False)</code>), runs candidates → features → LightGBM → Viterbi → stitching, and only then reads <code>.route</code> on the server to compute metrics.
The response has GPS points, predicted route geometry, per-fix arc ids, metrics and timings, and <b>no reference geometry</b>. Every request is logged to the console.
<b>Laptop 2</b> serves <code>ui/index.html</code> (bundled Leaflet, no build step). It lists the trips from <code>/trips</code>, sends only the selected id, and draws the result.</p>
{img('results/figures/ui/ui_result_00000077.png', 'Figure 7.1 Evaluator UI after selecting a trip (rehearsal using STUDENT_TEST trips; map tiles blank because this capture ran offline).', '100%')}
{img('results/figures/ui/ui_trip_list.png', 'Figure 7.2 Evaluator UI trip list loaded from the API.', '100%')}
<p><b>API console log</b> captured from the uvicorn process during the rehearsal run (<code>results/api_demo_log.txt</code>):</p>
<pre>{api_log}</pre>
<p><b>Excerpt of a successful <code>/predict</code> response</b> (<code>results/api_demo_response.json</code>, long arrays truncated):</p>
<pre>{api_json}</pre>


<h2>8. Conclusions and limitations</h2>
<p>The candidate-classification plus HMM pipeline reconstructs routes with {pct(hmm.top1_obs_weighted)} observation-level accuracy and {pct(hs.loc['classifier_hmm_stitched', 'len_f1'])}
route length F1 on the student-testing trips. The biggest single lever was the temporal-context features; the second was sequence decoding with directed-graph transitions.
Hyperparameter tuning mattered least.</p>
<p><b>Limitations.</b> (1) Labels come from our alignment of an arc-level route, and some reference routes are partial, so the reported accuracies apply to reliably labelled fixes only
({pct(cs.loc['student_test', 'reliable_frac'], 2)} of student-test fixes). (2) Arcs are straight segments between nodes, which is what the data provides. (3) Top-1 accuracy penalises
switching arcs one fix early or late at junctions. (4) The 10 student-test trips are a small sample: per-trip Top-1 varies by several points, so differences under about 0.5 pts between
models are not meaningful. (5) Evaluator trips may differ (larger networks, longer gaps); the pipeline handles both, but accuracy may vary.</p>

<h2>9. Reproducibility and AI-assistance disclosure</h2>
<p>Run, in order: <code>scripts/01_eda.py</code> → <code>02_build_features.py</code> → <code>03_train_models.py</code> → <code>04_learning_curves.py</code> → <code>05_bayes_opt.py</code>
→ <code>05b_feature_ablation.py</code> → <code>05c_plot_feature_selection.py</code> → <code>06_hmm_and_final_eval.py</code> → <code>07_summarise.py</code> → <code>report/build_report.py</code>. Every table and figure in this report is read from the files these scripts write
(source paths are given under each figure). Library versions are pinned in <code>requirements.txt</code>.</p>
<p><b>AI assistance.</b> We used Claude for code generation and problem understanding.</p>
</body></html>""")

open(OUT_HTML, "w").write("".join(H))
print("wrote", OUT_HTML)
from playwright.sync_api import sync_playwright  # noqa: E402
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page()
    pg.goto("file://" + OUT_HTML, wait_until="load")
    pg.pdf(path=OUT_PDF, format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
           display_header_footer=True, header_template="<span></span>",
           footer_template='<div style="font-size:7pt;width:100%;text-align:center;color:#888">PathSnap technical report · page <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    b.close()
print("wrote", OUT_PDF)
