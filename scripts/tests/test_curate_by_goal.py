"""Tests for curate_by_goal.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_curate_by_dimension_selects_top():
    from curate_by_goal import curate_by_dimension

    samples = [
        {"_original_index": 0, "scores": {"accuracy": 0.9, "relevance": 0.3}},
        {"_original_index": 1, "scores": {"accuracy": 0.3, "relevance": 0.9}},
        {"_original_index": 2, "scores": {"accuracy": 0.7, "relevance": 0.5}},
        {"_original_index": 3, "scores": {"accuracy": 0.5, "relevance": 0.7}},
    ]
    acc = curate_by_dimension(samples, "accuracy", retention=0.5)
    rel = curate_by_dimension(samples, "relevance", retention=0.5)

    assert len(acc) == 2
    assert len(rel) == 2
    assert set(acc) != set(rel), "Different dimensions should select different subsets"
    # Top accuracy: indices 0, 2
    assert 0 in acc
    assert 2 in acc


def test_curate_universal():
    from curate_by_goal import curate_universal

    samples = [
        {"_original_index": i, "scores": {"a": i / 10, "b": (10 - i) / 10}}
        for i in range(10)
    ]
    result = curate_universal(samples, ["a", "b"], retention=0.3)
    assert len(result) == 3


def test_curate_random_is_deterministic():
    from curate_by_goal import curate_random

    samples = [{"_original_index": i} for i in range(100)]
    r1 = curate_random(samples, retention=0.3, seed=42)
    r2 = curate_random(samples, retention=0.3, seed=42)
    assert r1 == r2
    assert len(r1) == 30
