"""Blend transaction-level (XGBoost) and account-level (GNN) risk.

The blend is a convex combination fitted on the validation split by grid
search over PR-AUC; using a rank-preserving, single-parameter blend keeps it
interpretable for analysts and impossible to overfit badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud.models.metrics import evaluate


@dataclass(frozen=True)
class Blender:
    weight_xgb: float = 0.7

    def __call__(self, p_xgb: np.ndarray | float, p_graph: np.ndarray | float) -> np.ndarray | float:
        return self.weight_xgb * p_xgb + (1.0 - self.weight_xgb) * p_graph

    @classmethod
    def fit(cls, y: np.ndarray, p_xgb: np.ndarray, p_graph: np.ndarray,
            grid: np.ndarray | None = None) -> tuple[Blender, dict[float, float]]:
        grid = np.linspace(0, 1, 21) if grid is None else grid
        scores = {round(float(w), 3): evaluate(y, w * p_xgb + (1 - w) * p_graph)["pr_auc"]
                  for w in grid}
        best = max(scores, key=lambda w: (scores[w], w))  # tie-break toward XGB (cheaper path)
        return cls(best), scores
