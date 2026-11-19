"""Métricas: EER, precision, recall, F1 (classe positiva = spoof).

Convenção dos scores: quanto MAIOR o score, mais provável ser spoof.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def compute_eer(labels, scores) -> tuple[float, float]:
    """Retorna (EER, limiar no ponto em que FAR == FRR)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])


def compute_metrics(labels, scores, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary",
                                                  pos_label=1, zero_division=0)
    eer, eer_thr = compute_eer(labels, scores)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "eer": eer,
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "threshold": float(threshold),
        "eer_threshold_this_split": eer_thr,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
