"""Stage 3b - epoch/iteration-wise training histories.

Only for models that genuinely produce them (MLP epochs, LightGBM boosting
iterations). Holdout = 16 of the 80 TRAIN trips (group split, seed 42).
Outputs -> results/models/curve_*.csv and fig_curve_*.png
"""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.features import FEATURES

OUT = os.path.join(C.RESULTS, "models")
tr = pd.read_parquet(os.path.join(C.PROCESSED, "train.parquet"))
tr = tr[tr.reliable].reset_index(drop=True)
a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED).split(tr, groups=tr.trip_id))
X, y = tr[FEATURES].values, tr.y.values

# ---- MLP (same architecture as models.py) ----
sc = StandardScaler().fit(X[a])
Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                    learning_rate_init=1e-3, random_state=C.SEED)
rows = []
rng = np.random.default_rng(C.SEED)
for ep in range(1, 41):
    perm = rng.permutation(len(Xa))
    mlp.partial_fit(Xa[perm], y[a][perm], classes=[0, 1])
    rows.append({"epoch": ep,
                 "train_log_loss": log_loss(y[a][:300_000], mlp.predict_proba(Xa[:300_000])[:, 1]),
                 "val_log_loss": log_loss(y[b], mlp.predict_proba(Xb)[:, 1])})
    print(rows[-1], flush=True)
mc = pd.DataFrame(rows); mc.to_csv(os.path.join(OUT, "curve_mlp.csv"), index=False)

# ---- LightGBM iterations ----
ev = {}
lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, n_jobs=2, verbose=-1)
lgb.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], eval_names=["train", "val"],
        eval_metric="binary_logloss", callbacks=[lambda env: None])
r = lgb.evals_result_
lc = pd.DataFrame({"iteration": np.arange(1, len(r["train"]["binary_logloss"]) + 1),
                   "train_log_loss": r["train"]["binary_logloss"], "val_log_loss": r["val"]["binary_logloss"]})
lc.to_csv(os.path.join(OUT, "curve_lightgbm.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(mc.epoch, mc.train_log_loss, label="train"); ax[0].plot(mc.epoch, mc.val_log_loss, label="validation (16 held-out trips)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("log loss"); ax[0].set_title("MLP 64-32 training history"); ax[0].legend()
ax[1].plot(lc.iteration, lc.train_log_loss, label="train"); ax[1].plot(lc.iteration, lc.val_log_loss, label="validation (16 held-out trips)")
ax[1].axvline(400, ls="--", c="grey", lw=.8); ax[1].text(405, lc.val_log_loss.max() * .9, "n_estimators used", fontsize=7)
ax[1].set_xlabel("boosting iteration"); ax[1].set_title("LightGBM training history"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_training_histories.png"), dpi=130)
print("best val epoch MLP", int(mc.val_log_loss.idxmin()) + 1, "best LGBM iter", int(lc.val_log_loss.idxmin()) + 1)
