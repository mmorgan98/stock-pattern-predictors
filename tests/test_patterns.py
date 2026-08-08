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
            if name == "no_pattern":
                assert not matcher(df), "no_pattern must stay a negative class"
            else:
                assert matcher(df), f"{name} failed self-match for seed={seed}"


def test_synthetic_dataset_shapes():
    X, y, names = build_synthetic_dataset(samples_per_pattern=5, seed=0)
    assert len(names) == len(PATTERN_NAMES)
    # no_pattern gets an extra batch of negative samples.
    expected = len(PATTERN_NAMES) * 5 + 5
    assert X.shape[0] == expected
    assert y.shape[0] == X.shape[0]
