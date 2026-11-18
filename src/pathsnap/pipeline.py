"""End-to-end inference pipeline (used by evaluation scripts and the API).

PathSnapPredictor.predict() only ever reads .track/.nodes/.arcs - the
reference .route is loaded separately by `evaluate_against_reference`, which
runs on the server after prediction, so labels never reach the model.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from . import config as C
from .features import FEATURES, build_features, prepare_trip
from .hmm import HMMParams, classifier_only, stitch_route, viterbi
from .io import load_trip
from .labels import align_points_to_route
from .metrics import observation_accuracy, route_continuity, sequence_metrics


def split_scores(prep, prob):
    n = [len(c[0]) for c in prep["cands"]]
    return np.split(prob, np.cumsum(n)[:-1]) if len(n) else []


def feasible_continuity(pred, net, p, slack=30.0):
    """Fraction of consecutive observation pairs whose predicted arcs are
    reachable from each other by driving no further than 2*d_gc + slack."""
    ok = tot = 0
    for k in range(1, len(pred)):
        a, b = pred[k - 1], pred[k]
        if a < 0 or b < 0:
            continue
        tot += 1
        d_gc = float(np.linalg.norm(p[k] - p[k - 1]))
        if a == b or net.arcs[a, 1] == net.arcs[b, 0]:
            ok += 1; continue
        d = net.route_distance(int(a), 1.0, int(b), 0.0, 2 * d_gc + slack)
        ok += np.isfinite(d)
    return ok / tot if tot else float("nan")


class PathSnapPredictor:
    def __init__(self, artifact_dir: str = C.ARTIFACTS):
        meta = json.load(open(os.path.join(artifact_dir, "pathsnap_config.json")))
        self.meta = meta
        self.model = joblib.load(os.path.join(artifact_dir, meta["model_file"]))
        self.hmm = HMMParams(**meta["hmm_params"])
        self.radius = meta["cand_radius_m"]
        self.k_max = meta["cand_k_max"]

    def predict(self, trip_dir: str) -> dict:
        timings = {}
        t0 = time.time()
        trip = load_trip(trip_dir, with_route=False)        # <- no labels
        prep = prepare_trip(trip, self.radius, self.k_max, with_labels=False)
        timings["preprocess_s"] = time.time() - t0
        t1 = time.time()
        df = build_features(prep)
        timings["features_s"] = time.time() - t1
        t2 = time.time()
        prob = self.model.predict_proba(df[FEATURES].values)[:, 1] if len(df) else np.zeros(0)
        timings["classifier_s"] = time.time() - t2
        scores = split_scores(prep, prob)
        clf_pred = classifier_only(prep, scores)
        t3 = time.time()
        hmm_pred, info = viterbi(prep, scores, self.hmm)
        timings["viterbi_s"] = time.time() - t3
        route, unfilled = stitch_route(prep["net"], hmm_pred)
        timings["total_s"] = time.time() - t0
        return {"prep": prep, "classifier_pred": clf_pred, "hmm_pred": hmm_pred,
                "route": route, "hmm_info": {**info, "unfilled_holes": unfilled},
                "timings": timings, "n_candidate_rows": int(len(df))}


def evaluate_against_reference(trip_dir: str, result: dict) -> dict:
    """Server-side scoring. Loads the reference route AFTER prediction."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    ref = np.loadtxt(os.path.join(trip_dir, f"{tid}.route"), dtype=np.int64, ndmin=1)
    prep = result["prep"]
    net, p = prep["net"], prep["p"]
    lab, _, adist = align_points_to_route(p, net, ref)
    rel = adist <= 50.0
    out = {}
    for name, pred in [("classifier_only", result["classifier_pred"]), ("hmm", result["hmm_pred"])]:
        out[name] = {
            "top1_accuracy": observation_accuracy(pred, lab, rel),
            "continuity_strict": route_continuity(pred, net.arcs),
            "continuity_feasible": feasible_continuity(pred, net, p),
            **sequence_metrics(pred, ref, net.length),
        }
    out["stitched_route"] = sequence_metrics(result["route"], ref, net.length)
    out["n_obs"] = int(len(p)); out["n_reliable_obs"] = int(rel.sum())
    out["candidate_miss_rate"] = float(np.mean([l not in c[0] for l, c, r in zip(lab, prep["cands"], rel) if r])) if rel.any() else None
    return out
