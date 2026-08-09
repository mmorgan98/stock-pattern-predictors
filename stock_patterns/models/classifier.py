"""Calibrated multiclass booster (legacy/fallback path)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_patterns.models.binary_ensemble import PatternEnsemble


class PatternClassifier:
    """Multiclass calibrated HistGradientBoosting classifier."""

    def __init__(self, class_names: list[str] | None = None, random_state: int = 42):
        self.class_names = class_names or []
        self.random_state = random_state
        base = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=random_state,
        )
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=base,
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        target_names = [self.class_names[i] for i in sorted(set(y_test))] if self.class_names else None
        report = classification_report(
            y_test,
            preds,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        return report

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        idxs = self.predict(X)
        if not self.class_names:
            return [str(i) for i in idxs]
        return [self.class_names[int(i)] for i in idxs]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "class_names": self.class_names,
                "kind": "pattern_multiclass_v2",
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PatternClassifier":
        blob = joblib.load(path)
        obj = cls(class_names=blob.get("class_names", []))
        obj.pipeline = blob["pipeline"]
        return obj


def load_any_model(path: str | Path) -> PatternEnsemble | PatternClassifier:
    """Load either a PatternEnsemble index or a legacy/multiclass classifier."""
    path = Path(path)
    blob = joblib.load(path)
    kind = blob.get("kind")
    if kind == "pattern_ensemble_v1" or "pattern_files" in blob:
        return PatternEnsemble.load(path)
    return PatternClassifier.load(path)
