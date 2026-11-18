"""Plot the feature-group ablation (results/feature_selection/ablation.csv)."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C

D = os.path.join(C.RESULTS, "feature_selection")
R = pd.read_csv(os.path.join(D, "ablation.csv"))
A = R[R.feature_set.str.startswith("without")].sort_values("delta_top1_pts")
fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
ax.barh(A.feature_set.str.replace("without ", ""), A.delta_top1_pts,
        color=["#C8611E" if v < -0.5 else "#9fb7d6" for v in A.delta_top1_pts])
ax.axvspan(-0.35, 0.35, color="#ddd", alpha=.5, lw=0)
ax.text(0.37, -0.45, "within noise (±0.35)", fontsize=8, color="#555")
ax.axvline(0, c="k", lw=.8)
for i, v in enumerate(A.delta_top1_pts):
    ax.text(v - 0.05 if v < 0 else v + 0.05, i, f"{v:+.2f}", va="center", ha="right" if v < 0 else "left", fontsize=9)
ax.set_xlabel("change in CV Top-1 when the group is removed (percentage points)"); ax.set_xlim(-3, 1)
ax.set_title("Feature-group ablation (LightGBM, 3-fold GroupKFold, TRAIN)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_ablation.png"))
print("saved", os.path.join(D, "fig_ablation.png"))
