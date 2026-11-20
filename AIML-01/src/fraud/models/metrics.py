"""Imbalance-aware evaluation. Accuracy is intentionally absent."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def recall_at_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec[prec >= target]
    return float(ok.max()) if ok.size else 0.0


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thr = best_f1_threshold(y, p) if threshold is None else threshold
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    k = max(1, int(round(0.01 * len(y))))  # top 1% -> analyst review capacity
    top = np.argsort(-p)[:k]
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.any() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else 0.5,
        "baseline_pr_auc": float(y.mean()),
        "threshold": thr,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision_at_top1pct": float(y[top].mean()),
        "recall_at_p90": recall_at_precision(y, p, 0.90),
    }
