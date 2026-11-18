"""PathSnap FastAPI backend (Laptop 1).

Run:   uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:   PATHSNAP_EVAL_DIR=/path/to/evaluator_trips   (one sub-folder per trip)

The prediction pipeline only reads .track/.nodes/.arcs. The .route file is
read afterwards, server-side, to compute accuracy; it is never sent to the UI.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor, evaluate_against_reference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("pathsnap.api")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", C.EVAL_DIR)
predictor = PathSnapPredictor(C.ARTIFACTS)
log.info("Loaded model: %s | HMM params: %s", predictor.meta["model_name"], predictor.meta["hmm_params"])
log.info("Evaluation trips folder: %s", EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:6]
    t0 = time.time()
    client = request.client.host if request.client else "?"
    log.info("[%s] <-- %s %s from %s", rid, request.method, request.url.path, client)
    response = await call_next(request)
    log.info("[%s] --> %s in %.0f ms", rid, response.status_code, 1000 * (time.time() - t0))
    return response


def _trips() -> list[str]:
    if not os.path.isdir(EVAL_DIR):
        log.warning("Evaluation folder %s does not exist", EVAL_DIR)
        return []
    # only folders that really are trips (<id>/<id>.track present)
    trips = [t for t in list_trips(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, t, f"{t}.track"))]
    if not trips:
        log.warning("No trip folders found in %s (expected %s/<trip_id>/<trip_id>.track)", EVAL_DIR, EVAL_DIR)
    return trips


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta["model_name"], "n_trips": len(_trips())}


@app.get("/trips")
def trips():
    out = []
    for t in _trips():
        with open(os.path.join(EVAL_DIR, t, f"{t}.track")) as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": t, "n_observations": n})
    return {"trips": out}


def _route_geometry(net, route):
    """Predicted route as lat/lon polylines (broken where not adjacent)."""
    segs, cur = [], []
    for i, a in enumerate(route):
        f, t = net.arcs[a]
        if cur and net.arcs[route[i - 1], 1] != f:
            segs.append(cur); cur = []
        if not cur:
            cur.append([float(net.nodes_ll[f, 1]), float(net.nodes_ll[f, 0])])
        cur.append([float(net.nodes_ll[t, 1]), float(net.nodes_ll[t, 0])])
    if cur:
        segs.append(cur)
    return segs


@app.post("/predict")
def predict(req: PredictRequest):
    valid = _trips()
    if req.trip_id not in valid:          # also blocks path traversal
        raise HTTPException(404, f"unknown trip_id {req.trip_id!r}")
    trip_dir = os.path.join(EVAL_DIR, req.trip_id)
    log.info("PREDICT trip=%s : running pipeline (candidates -> classifier -> Viterbi)", req.trip_id)
    res = predictor.predict(trip_dir)
    tm = res["timings"]
    log.info("PREDICT trip=%s : %d obs, %d candidate rows, HMM breaks=%d, route arcs=%d, %.2fs",
             req.trip_id, len(res["prep"]["p"]), res["n_candidate_rows"], res["hmm_info"]["breaks"],
             len(res["route"]), tm["total_s"])
    # ---- server-side scoring (reference loaded only now) ----
    ev = evaluate_against_reference(trip_dir, res)
    log.info("EVAL    trip=%s : Top-1 classifier-only=%.4f | classifier+HMM=%.4f | route F1(len)=%.4f",
             req.trip_id, ev["classifier_only"]["top1_accuracy"], ev["hmm"]["top1_accuracy"],
             ev["stitched_route"]["len_f1"])
    net = res["prep"]["net"]
    trk = res["prep"]["track"]
    step = max(1, len(trk) // 3000)
    return {
        "trip_id": req.trip_id,
        "model": predictor.meta["model_name"],
        "n_observations": int(len(trk)),
        "gps_points": [[float(la), float(lo)] for lo, la in trk[::step, :2]],
        "predicted_route": _route_geometry(net, res["route"]),
        "predicted_route_arc_ids": [int(a) for a in res["route"]],
        "per_observation_arc": [int(a) for a in res["hmm_pred"]],
        "hmm_info": res["hmm_info"],
        "timings_s": {k: round(v, 3) for k, v in tm.items()},
        "metrics": {
            "top1_accuracy_hmm": ev["hmm"]["top1_accuracy"],
            "top1_accuracy_classifier_only": ev["classifier_only"]["top1_accuracy"],
            "continuity_hmm": ev["hmm"]["continuity_feasible"],
            "continuity_classifier_only": ev["classifier_only"]["continuity_feasible"],
            "route_len_precision": ev["stitched_route"]["len_precision"],
            "route_len_recall": ev["stitched_route"]["len_recall"],
            "route_len_f1": ev["stitched_route"]["len_f1"],
            "route_arc_f1": ev["stitched_route"]["arc_f1"],
            "candidate_miss_rate": ev["candidate_miss_rate"],
            "n_scored_observations": ev["n_reliable_obs"],
        },
    }
