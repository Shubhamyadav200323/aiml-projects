"""Gradient-boosted tabular model.

Imbalance handling is selectable:
  * ``"weight"`` (default): `scale_pos_weight = n_neg / n_pos`
  * ``"smote"``: oversample the training fold only (never validation/test).

Early stopping monitors validation **aucpr**. Per-prediction explanations
come from XGBoost's exact TreeSHAP (`pred_contribs=True`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.features.definitions import FEATURE_COLUMNS

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "lambda": 1.0,
}


@dataclass
class XGBFraudModel:
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    imbalance: str = "weight"
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    booster: xgb.Booster | None = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray, seed: int = 42) -> XGBFraudModel:
        Xt, yt = X_train[self.features].to_numpy(), np.asarray(y_train)
        params = {**self.params, "seed": seed}
        if self.imbalance == "smote":
            from imblearn.over_sampling import SMOTE

            Xt, yt = SMOTE(random_state=seed, sampling_strategy=0.1).fit_resample(Xt, yt)
        elif self.imbalance == "weight":
            pos = max(int(yt.sum()), 1)
            params["scale_pos_weight"] = (len(yt) - pos) / pos
        dtrain = xgb.DMatrix(Xt, label=yt, feature_names=self.features)
        dval = xgb.DMatrix(X_val[self.features].to_numpy(), label=y_val, feature_names=self.features)
        self.booster = xgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            evals=[(dval, "val")], early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return self

    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X[self.features].to_numpy(dtype=np.float64), feature_names=self.features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "model is not fitted"
        it = (0, self.booster.best_iteration + 1) if hasattr(self.booster, "best_iteration") else (0, 0)
        return self.booster.predict(self._dmatrix(X), iteration_range=it)

    def explain(self, X: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, float]]]:
        """Top-k SHAP contributions (log-odds) pushing each row towards fraud."""
        assert self.booster is not None
        contribs = self.booster.predict(self._dmatrix(X), pred_contribs=True)[:, :-1]
        out = []
        for row in contribs:
            idx = np.argsort(-row)[:top_k]
            out.append([{"feature": self.features[i], "contribution": round(float(row[i]), 4)}
                        for i in idx if row[i] > 0])
        return out

    def feature_importance(self) -> dict[str, float]:
        assert self.booster is not None
        gain = self.booster.get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1]))

    def save(self, directory: str | Path) -> Path:
        assert self.booster is not None
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(d / "xgb.json")
        meta = {"features": self.features, "best_iteration": int(self.booster.best_iteration)}
        (d / "xgb_meta.json").write_text(json.dumps(meta, indent=2))
        return d / "xgb.json"

    @classmethod
    def load(cls, directory: str | Path) -> XGBFraudModel:
        d = Path(directory)
        meta = json.loads((d / "xgb_meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(d / "xgb.json")
        booster.set_attr(best_iteration=str(meta["best_iteration"]))
        return cls(features=meta["features"], booster=booster)
