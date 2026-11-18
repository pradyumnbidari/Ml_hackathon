"""Model zoo: candidate classifiers from several algorithm families.

Every model is a scikit-learn compatible estimator exposing predict_proba.
Initial (untuned) configurations are recorded in results/models/configs.json.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

MODEL_SPECS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        build=lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", random_state=SEED)),
        config="StandardScaler; L2, C=1.0, lbfgs, max_iter=500",
        loss="Binary cross-entropy (log loss) + L2 penalty"),
    "GaussianNB": dict(
        family="Linear / probabilistic",
        build=lambda: make_pipeline(QuantileTransformer(output_distribution="normal",
                                                        n_quantiles=200, subsample=200_000,
                                                        random_state=SEED), GaussianNB()),
        config="QuantileTransformer->normal; var_smoothing=1e-9",
        loss="Maximum likelihood of class-conditional Gaussians (no iterative loss)"),
    "KNN": dict(
        family="Instance-based",
        build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=25, weights="distance", n_jobs=2)),
        config="StandardScaler; k=25, distance-weighted, Euclidean; fit on 200k-row sample",
        loss="None (lazy learner; vote of neighbours)", subsample=200_000),
    "DecisionTree": dict(
        family="Tree-based",
        build=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
        config="max_depth=12, min_samples_leaf=50, gini",
        loss="Gini impurity (greedy splits)"),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        build=lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                             max_samples=0.25, n_jobs=2, random_state=SEED),
        config="150 trees, max_depth=18, min_samples_leaf=20, bootstrap 25%",
        loss="Gini impurity per tree; averaged probabilities"),
    "HistGradientBoosting": dict(
        family="Boosting",
        build=lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                     max_leaf_nodes=31, random_state=SEED),
        config="300 iters, lr=0.1, 31 leaves, no early stopping",
        loss="Binary log loss (gradient boosting)"),
    "LightGBM": dict(
        family="Boosting",
        build=lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=SEED, n_jobs=2, verbose=-1),
        config="400 trees, lr=0.05, 63 leaves, row/col subsample 0.8",
        loss="Binary log loss (gradient boosting, leaf-wise)"),
    "MLP": dict(
        family="Neural network",
        build=lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
            max_iter=25, early_stopping=False, random_state=SEED)),
        config="StandardScaler; 64-32 ReLU, Adam lr=1e-3, batch 1024, alpha=1e-4, 25 epochs",
        loss="Binary cross-entropy + L2 (Adam)"),
}
