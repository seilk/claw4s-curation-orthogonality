"""Tests for rank_correlation.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tau_matrix_symmetry():
    from rank_correlation import compute_tau_matrix

    samples = [
        {"scores": {"a": i / 100, "b": (100 - i) / 100, "c": (i * 3 % 100) / 100}}
        for i in range(100)
    ]
    result = compute_tau_matrix(samples, ["a", "b", "c"])
    tau = result["tau"]

    # Symmetric
    assert tau["a"]["b"] == tau["b"]["a"]
    # Diagonal = 1
    assert tau["a"]["a"] == 1.0
    # a and b are anti-correlated
    assert tau["a"]["b"] < 0


def test_jaccard_identical_sets():
    from rank_correlation import compute_jaccard_matrix

    subsets = {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 2, 3, 4, 5],
        "z": [6, 7, 8, 9, 10],
    }
    jac = compute_jaccard_matrix(subsets)
    assert jac["x"]["y"] == 1.0
    assert jac["x"]["z"] == 0.0
