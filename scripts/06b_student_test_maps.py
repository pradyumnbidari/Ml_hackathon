"""Regenerate the STUDENT_TEST Folium maps with the exported final pipeline.
(06_hmm_and_final_eval.py makes the same maps; this is the quick way to redo them.)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import PathSnapPredictor, evaluate_against_reference
from pathsnap.viz import trip_map

pred = PathSnapPredictor(C.ARTIFACTS)
out = os.path.join(C.RESULTS, "maps"); os.makedirs(out, exist_ok=True)
for tid in list_trips(C.STUDENT_TEST_DIR):
    d = os.path.join(C.STUDENT_TEST_DIR, tid)
    r = pred.predict(d); ev = evaluate_against_reference(d, r)
    net = r["prep"]["net"]
    m = trip_map(net, r["prep"]["track"][:, :2], reference_arcs=load_trip(d).route, predicted_arcs=r["route"],
                 classifier_arcs=[a for a in r["classifier_pred"] if a >= 0],
                 title=f"STUDENT_TEST {tid}: Top-1 clf {ev['classifier_only']['top1_accuracy']:.3f} -> HMM {ev['hmm']['top1_accuracy']:.3f}")
    m.save(os.path.join(out, f"student_test_{tid}.html")); print(tid, round(ev["hmm"]["top1_accuracy"], 4))
