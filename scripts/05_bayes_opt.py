"""Stage 4 - Bayesian hyperparameter optimisation (Optuna TPE) for LightGBM.

Objective  : maximise mean observation-level Top-1 accuracy over 3-fold
             GroupKFold on the 80 TRAIN trips (groups = trip). Macro-F1 and
             log loss are logged per trial as secondary metrics.
Speed      : each trial uses a fixed 40 % sample of TRAIN observations (all
             candidates of a sampled observation are kept, so Top-1 stays
             well-defined). Every trip is still represented.
The STUDENT_TEST partition is used only after the search, to compare tuned
and untuned models. Evaluator data is never touched.
Outputs -> results/optuna/trials.csv, best_params.json, figs;
           artifacts/models/LightGBM_tuned.joblib
"""
import json, os, sys, time
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, optuna, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.metrics import classification_metrics, top1_from_scores

OUT = os.path.join(C.RESULTS, "optuna"); os.makedirs(OUT, exist_ok=True)
N_TRIALS = int(os.environ.get("N_TRIALS", 60))

tr = pd.read_parquet(os.path.join(C.PROCESSED, "train.parquet"))
obs = pd.read_parquet(os.path.join(C.PROCESSED, "train_obs.parquet"))
tr = tr[tr.reliable].reset_index(drop=True)
rng = np.random.default_rng(C.SEED)
keys = tr[["trip_id", "obs"]].drop_duplicates()
keep = keys.sample(frac=0.4, random_state=C.SEED)
sub = tr.merge(keep, on=["trip_id", "obs"]).reset_index(drop=True)
X, y, g = sub[FEATURES].values, sub.y.values, sub.trip_id.values
folds = list(GroupKFold(n_splits=3).split(X, y, g))
print("rows", len(sub), "obs", len(keep))


def top1(df, prob):
    """Top-1 over sampled observations that have >=1 candidate. Observations
    with no candidates are identical across trials, so they do not change the
    ranking of hyperparameter configurations."""
    d = df[["trip_id", "obs", "y"]].copy(); d["score"] = prob
    best = d.loc[d.groupby(["trip_id", "obs"])["score"].idxmax()]
    return float(best.y.mean())


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    t1s, f1s, lls = [], [], []
    t0 = time.time()
    for a, b in folds:
        m = LGBMClassifier(**params, subsample_freq=1, random_state=C.SEED, n_jobs=2, verbose=-1)
        m.fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        cm = classification_metrics(y[b], p)
        t1s.append(top1(sub.iloc[b], p)); f1s.append(cm["f1_macro"]); lls.append(cm["log_loss"])
    trial.set_user_attr("cv_f1_macro", float(np.mean(f1s)))
    trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
    trial.set_user_attr("cv_top1_std", float(np.std(t1s)))
    trial.set_user_attr("seconds", time.time() - t0)
    print(f"trial {trial.number}: top1={np.mean(t1s):.4f} f1={np.mean(f1s):.4f} ll={np.mean(lls):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return float(np.mean(t1s))


# untuned baseline evaluated with the SAME protocol (trial 0 = enqueued defaults)
study = optuna.create_study(direction="maximize", study_name="pathsnap_lgbm",
                            storage=f"sqlite:///{os.path.join(OUT, 'optuna_study.db')}", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=10))
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not study.trials:
    study.enqueue_trial(dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3))
# resumable: only run the trials still missing (study is stored in SQLite)
remaining = max(0, N_TRIALS - len(done))
print("completed trials:", len(done), "-> running", remaining, flush=True)
study.optimize(objective, n_trials=remaining)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
total = sum(t.user_attrs.get("seconds", 0) for t in done)

df = study.trials_dataframe()
df = df[df.state == "COMPLETE"].reset_index(drop=True)
df.to_csv(os.path.join(OUT, "trials.csv"), index=False)
best = study.best_params
json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": len(done),
           "objective": "mean 3-fold GroupKFold Top-1 accuracy (40% obs sample of TRAIN)",
           "baseline_trial0_cv_top1": study.trials[0].value,
           "best_trial": study.best_trial.number,
           "best_cv_f1_macro": study.best_trial.user_attrs["cv_f1_macro"],
           "best_cv_log_loss": study.best_trial.user_attrs["cv_log_loss"],
           "baseline_cv_f1_macro": study.trials[0].user_attrs["cv_f1_macro"],
           "baseline_cv_log_loss": study.trials[0].user_attrs["cv_log_loss"],
           "total_search_seconds": total}, open(os.path.join(OUT, "best_params.json"), "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
vals = df.value.values
ax[0].scatter(df.number, vals, s=14, label="trial")
ax[0].plot(df.number, np.maximum.accumulate(vals), c="crimson", label="best so far")
ax[0].axhline(vals[0], ls="--", c="grey", lw=.8, label="untuned (trial 0)")
ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Top-1 accuracy"); ax[0].legend(); ax[0].set_title("Optuna TPE optimisation history")
try:
    imp = optuna.importance.get_param_importances(study)
    ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#3b6ea8")
    ax[1].set_title("Hyperparameter importance (fANOVA)")
    json.dump(imp, open(os.path.join(OUT, "param_importance.json"), "w"), indent=2)
except Exception as e:
    print("importance failed", e)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_optuna.png"), dpi=130)

# refit the tuned model on ALL reliable TRAIN rows
full = tr
m = LGBMClassifier(**best, subsample_freq=1, random_state=C.SEED, n_jobs=2, verbose=-1)
t0 = time.time(); m.fit(full[FEATURES].values, full.y.values)
json.dump({"refit_seconds": time.time() - t0}, open(os.path.join(OUT, "refit.json"), "w"))
joblib.dump(m, os.path.join(C.ARTIFACTS, "models", "LightGBM_tuned.joblib"), compress=3)
print("best", best, study.best_value)
