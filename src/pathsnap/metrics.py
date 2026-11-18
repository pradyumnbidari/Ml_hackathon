"""Evaluation metrics: candidate classification, observation-level and
sequence-level route metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)


def classification_metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "f1_pos": f1_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def top1_from_scores(df: pd.DataFrame, score_col="score") -> pd.DataFrame:
    """Per (trip, obs) take the arc with the highest score."""
    idx = df.groupby(["trip_id", "obs"], sort=False)[score_col].idxmax()
    return df.loc[idx, ["trip_id", "obs", "arc", score_col]].rename(columns={"arc": "pred_arc"})


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive observation pairs whose predicted arcs are the
    same arc or directly connected (to_node of first == from_node of second)."""
    pa = np.asarray(pred_arcs)
    ok = pa >= 0
    a, b = pa[:-1], pa[1:]
    v = ok[:-1] & ok[1:]
    if v.sum() == 0:
        return float("nan")
    same = a[v] == b[v]
    conn = arcs[a[v], 1] == arcs[b[v], 0]
    return float(np.mean(same | conn))


def collapse(seq):
    out = []
    for s in seq:
        if s >= 0 and (not out or out[-1] != s):
            out.append(int(s))
    return out


def sequence_metrics(pred_arcs, ref_route, arc_len) -> dict:
    """Arc-set and length-weighted precision/recall/F1 between the predicted
    route (distinct arcs visited) and the reference route."""
    P, R = set(collapse(pred_arcs)), set(int(r) for r in ref_route)
    inter = P & R
    prec = len(inter) / max(len(P), 1)
    rec = len(inter) / max(len(R), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    Lp = sum(arc_len[i] for i in P); Lr = sum(arc_len[i] for i in R)
    Li = sum(arc_len[i] for i in inter)
    lp, lr = Li / max(Lp, 1e-9), Li / max(Lr, 1e-9)
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f1,
            "len_precision": lp, "len_recall": lr,
            "len_f1": 2 * lp * lr / max(lp + lr, 1e-9)}


def observation_accuracy(pred_arcs, labels, reliable) -> float:
    m = np.asarray(reliable, bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.asarray(pred_arcs)[m] == np.asarray(labels)[m]))
