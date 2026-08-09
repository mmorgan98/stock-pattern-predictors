"""One calibrated binary booster per candlestick / chart pattern."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _base_estimator(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=random_state,
    )


class PatternEnsemble:
    """
    Binary detector bank: one model file/estimator per pattern.

    Each model predicts P(pattern) vs not-pattern using a calibrated
    HistGradientBoostingClassifier.
    """

    def __init__(self, pattern_names: list[str] | None = None, random_state: int = 42):
        self.pattern_names = [n for n in (pattern_names or []) if n != "no_pattern"]
        self.random_state = random_state
        self.models: dict[str, Pipeline] = {}
        self.metrics: dict[str, dict] = {}

    def _make_pipeline(self) -> Pipeline:
        # cv='prefit' unsupported for fitting from scratch; use sigmoid calibration with CV.
        clf = CalibratedClassifierCV(
            estimator=_base_estimator(self.random_state),
            method="sigmoid",
            cv=3,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit_pattern(
        self,
        name: str,
        X: np.ndarray,
        y_binary: np.ndarray,
    ) -> dict:
        """Fit one binary pattern detector. y_binary is 1 for pattern, 0 otherwise."""
        if len(np.unique(y_binary)) < 2:
            raise ValueError(f"Need both classes to train {name}")

        strat = y_binary if np.min(np.bincount(y_binary.astype(int))) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_binary,
            test_size=0.2,
            random_state=self.random_state,
            stratify=strat,
        )
        pipe = self._make_pipeline()
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0,
            "average_precision": float(average_precision_score(y_test, proba))
            if len(np.unique(y_test)) > 1
            else 0.0,
            "n_train": int(len(y_train)),
            "n_pos": int(y_train.sum()),
            "n_neg": int((y_train == 0).sum()),
        }
        self.models[name] = pipe
        self.metrics[name] = metrics
        if name not in self.pattern_names:
            self.pattern_names.append(name)
        return metrics

    def fit_from_multiclass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list[str],
        negative_multiplier: float = 2.0,
    ) -> dict[str, dict]:
        """
        Train one binary model per pattern from a multiclass labeled matrix.

        Negatives are other classes (including no_pattern), subsampled for balance.
        """
        rng = np.random.default_rng(self.random_state)
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        reports: dict[str, dict] = {}

        for name in class_names:
            if name == "no_pattern":
                continue
            pos_idx = name_to_idx[name]
            pos_mask = y == pos_idx
            neg_mask = ~pos_mask
            X_pos = X[pos_mask]
            X_neg = X[neg_mask]
            if len(X_pos) < 5 or len(X_neg) < 5:
                continue

            n_neg = min(len(X_neg), max(len(X_pos), int(len(X_pos) * negative_multiplier)))
            choice = rng.choice(len(X_neg), size=n_neg, replace=False)
            X_bin = np.vstack([X_pos, X_neg[choice]])
            y_bin = np.concatenate(
                [np.ones(len(X_pos), dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
            )
            reports[name] = self.fit_pattern(name, X_bin, y_bin)
        return reports

    def predict_proba_dict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return {pattern: P(pattern)} for each trained detector."""
        out: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            out[name] = model.predict_proba(X)[:, 1]
        return out

    def score_vector(self, x: np.ndarray) -> dict[str, float]:
        """Score a single feature vector."""
        X = np.atleast_2d(x)
        return {name: float(proba[0]) for name, proba in self.predict_proba_dict(X).items()}

    def save(self, path: str | Path) -> Path:
        """
        Save ensemble index plus one joblib file per pattern under <path>_patterns/.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pattern_dir = path.parent / f"{path.stem}_patterns"
        pattern_dir.mkdir(parents=True, exist_ok=True)

        pattern_files: dict[str, str] = {}
        for name, model in self.models.items():
            fpath = pattern_dir / f"{name}.joblib"
            joblib.dump(model, fpath)
            pattern_files[name] = str(fpath)

        joblib.dump(
            {
                "pattern_names": self.pattern_names,
                "random_state": self.random_state,
                "metrics": self.metrics,
                "pattern_files": pattern_files,
                "kind": "pattern_ensemble_v1",
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PatternEnsemble":
        path = Path(path)
        blob = joblib.load(path)
        obj = cls(
            pattern_names=blob.get("pattern_names", []),
            random_state=blob.get("random_state", 42),
        )
        obj.metrics = blob.get("metrics", {})
        pattern_files = blob.get("pattern_files", {})
        for name, fpath in pattern_files.items():
            obj.models[name] = joblib.load(fpath)
        # Backward compatible: models embedded directly.
        if not obj.models and "models" in blob:
            obj.models = blob["models"]
        return obj
