"""Tests for binary ensemble training and peak-window ranking."""

from __future__ import annotations

from pathlib import Path

from stock_patterns.models.binary_ensemble import PatternEnsemble
from stock_patterns.patterns.hammer import generate as gen_hammer
from stock_patterns.pipeline.dataset import build_synthetic_dataset
from stock_patterns.pipeline.features import sequence_to_feature_vector
from stock_patterns.pipeline.ranking import rank_patterns


def test_ensemble_trains_and_saves(tmp_path: Path):
    X, y, names = build_synthetic_dataset(samples_per_pattern=20, seed=1, no_pattern_multiplier=2)
    ens = PatternEnsemble(pattern_names=names, random_state=1)
    reports = ens.fit_from_multiclass(X, y, names)
    assert "hammer" in reports
    assert reports["hammer"]["roc_auc"] >= 0.8

    out = tmp_path / "pattern_ensemble.joblib"
    ens.save(out)
    assert out.exists()
    assert (tmp_path / "pattern_ensemble_patterns" / "hammer.joblib").exists()

    loaded = PatternEnsemble.load(out)
    x = sequence_to_feature_vector(gen_hammer(seed=7))
    score = loaded.score_vector(x)
    assert score["hammer"] > 0.5


def test_rank_patterns_returns_confidence():
    X, y, names = build_synthetic_dataset(samples_per_pattern=15, seed=2, no_pattern_multiplier=2)
    ens = PatternEnsemble(pattern_names=names, random_state=2)
    ens.fit_from_multiclass(X, y, names)
    df = gen_hammer(seed=3)
    ranked = rank_patterns(df, top=3, model=ens, min_confidence=0.3)
    assert ranked
    assert "confidence" in ranked[0]
    assert ranked[0]["confidence"] >= ranked[-1]["confidence"]
