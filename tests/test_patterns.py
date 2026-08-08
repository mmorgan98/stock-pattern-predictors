"""Smoke tests for pattern generators and self-match heuristics."""

from __future__ import annotations

from stock_patterns.patterns.registry import PATTERN_GENERATORS, PATTERN_MATCHERS, PATTERN_NAMES
from stock_patterns.pipeline.dataset import build_synthetic_dataset


def test_all_generators_self_match():
    for name in PATTERN_NAMES:
        gen = PATTERN_GENERATORS[name]
        matcher = PATTERN_MATCHERS[name]
        for seed in range(25):
            df = gen(seed=seed)
            assert not df.empty, name
            assert {"open", "high", "low", "close"}.issubset(df.columns)
            assert matcher(df), f"{name} failed self-match for seed={seed}"


def test_synthetic_dataset_shapes():
    X, y, names = build_synthetic_dataset(samples_per_pattern=5, seed=0)
    assert len(names) == len(PATTERN_NAMES)
    assert X.shape[0] == len(PATTERN_NAMES) * 5
    assert y.shape[0] == X.shape[0]
