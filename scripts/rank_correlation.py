"""Compute Kendall's τ and Jaccard similarity between curated subsets.

Usage:
    python scripts/rank_correlation.py output/alpaca/scores.jsonl \
        output/alpaca/subsets.json \
        --output output/alpaca/
"""
import json
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


def load_scores_and_subsets(scores_path: Path, subsets_path: Path):
    with open(scores_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(subsets_path) as f:
        subsets = json.load(f)
    return samples, subsets


def compute_tau_matrix(samples: list[dict], dimensions: list[str]) -> dict:
    """Kendall's τ between score vectors (all samples, not just subsets)."""
    n = len(samples)
    score_vecs = {}
    for d in dimensions:
        score_vecs[d] = np.array([s["scores"][d] for s in samples])

    matrix = {}
    p_values = {}
    for d in dimensions:
        matrix[d] = {}
        p_values[d] = {}

    for a, b in combinations(dimensions, 2):
        tau, p = kendalltau(score_vecs[a], score_vecs[b], method="asymptotic")
        matrix.setdefault(a, {})[b] = round(float(tau), 4)
        matrix.setdefault(b, {})[a] = round(float(tau), 4)
        p_values.setdefault(a, {})[b] = round(float(p), 8)
        p_values.setdefault(b, {})[a] = round(float(p), 8)

    for d in dimensions:
        matrix[d][d] = 1.0
        p_values[d][d] = 0.0

    return {"tau": matrix, "p_values": p_values}


def compute_jaccard_matrix(subsets: dict[str, list[int]]) -> dict:
    """Jaccard similarity between all pairs of subsets."""
    names = sorted(subsets.keys())
    sets = {k: set(v) for k, v in subsets.items()}
    matrix = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            intersection = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            matrix[a][b] = round(intersection / max(union, 1), 4)
    return matrix


def compute_quality_loss(
    samples: list[dict], subsets: dict[str, list[int]], dimensions: list[str]
) -> dict:
    """Compare goal-specific vs universal subset quality per dimension."""
    from scipy.stats import mannwhitneyu

    universal_set = set(subsets.get("universal", []))
    results = {}

    for dim in dimensions:
        goal_set = set(subsets.get(dim, []))
        goal_scores = [samples[i]["scores"][dim] for i in range(len(samples)) if samples[i]["_original_index"] in goal_set]
        univ_scores = [samples[i]["scores"][dim] for i in range(len(samples)) if samples[i]["_original_index"] in universal_set]

        if not goal_scores or not univ_scores:
            continue

        goal_mean = sum(goal_scores) / len(goal_scores)
        univ_mean = sum(univ_scores) / len(univ_scores)
        delta = goal_mean - univ_mean

        stat, p_val = mannwhitneyu(goal_scores, univ_scores, alternative="greater")
        # Rank-biserial effect size (positive = goal > universal)
        n1, n2 = len(goal_scores), len(univ_scores)
        effect_size = (2 * stat) / (n1 * n2) - 1

        results[dim] = {
            "goal_specific_mean": round(goal_mean, 4),
            "universal_mean": round(univ_mean, 4),
            "delta": round(delta, 4),
            "p_value": round(float(p_val), 8),
            "effect_size": round(float(effect_size), 4),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("subsets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples, subsets = load_scores_and_subsets(args.scores, args.subsets)
    dimensions = [d for d in subsets.keys() if d not in ("universal", "random")]

    print("[rank_corr] Computing Kendall's τ matrix...")
    tau_result = compute_tau_matrix(samples, dimensions)

    print("[rank_corr] Computing Jaccard matrix...")
    jaccard = compute_jaccard_matrix(subsets)

    print("[rank_corr] Computing quality loss (goal-specific vs universal)...")
    quality_loss = compute_quality_loss(samples, subsets, dimensions)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "tau_matrix.json", "w") as f:
        json.dump(tau_result, f, indent=2)
    with open(args.output / "jaccard_matrix.json", "w") as f:
        json.dump(jaccard, f, indent=2)
    with open(args.output / "quality_loss.json", "w") as f:
        json.dump(quality_loss, f, indent=2)

    print(f"[rank_corr] Results saved to {args.output}")
    print("\nKendall's τ matrix:")
    for d in dimensions:
        row = " | ".join(f"{tau_result['tau'][d].get(d2, 'N/A'):>7}" for d2 in dimensions)
        print(f"  {d:>15}: {row}")
