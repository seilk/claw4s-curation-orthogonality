"""Permutation test for Kendall's τ significance.

Shuffles dimension labels N times to establish null distribution.
Tests whether observed τ values are significantly different from random.

Usage:
    python scripts/permutation_test.py output/alpaca/scores.jsonl \
        --n-permutations 10000 \
        --subsample 5000 \
        --output output/alpaca/permutation_results.json
"""
import json
import random
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


def permutation_test(
    scores_path: Path,
    n_permutations: int = 10000,
    subsample: int = 5000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    with open(scores_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Subsample for performance
    if len(samples) > subsample:
        indices = sorted(rng.sample(range(len(samples)), subsample))
        samples = [samples[i] for i in indices]

    dimensions = list(samples[0]["scores"].keys())
    n = len(samples)

    # Extract score matrix: n_samples x n_dimensions
    score_matrix = np.array(
        [[s["scores"][d] for d in dimensions] for s in samples]
    )

    # Observed Kendall's τ for all pairs
    pairs = list(combinations(range(len(dimensions)), 2))
    observed = {}
    for i, j in pairs:
        tau, _ = kendalltau(score_matrix[:, i], score_matrix[:, j], method="asymptotic")
        pair_name = f"{dimensions[i]}_vs_{dimensions[j]}"
        observed[pair_name] = float(tau)

    print(f"[perm] {len(pairs)} dimension pairs, {n_permutations} permutations on {n} samples...")

    # Permutation null distribution: for each pair, shuffle one column's rows
    null_distributions: dict[str, list[float]] = {k: [] for k in observed}

    for perm_i in range(n_permutations):
        if (perm_i + 1) % 1000 == 0:
            print(f"  permutation {perm_i + 1}/{n_permutations}")
        for i, j in pairs:
            # Shuffle rows of column j while keeping column i fixed
            shuffled_j = score_matrix[:, j].copy()
            rng.shuffle(shuffled_j)
            tau, _ = kendalltau(score_matrix[:, i], shuffled_j, method="asymptotic")
            pair_name = f"{dimensions[i]}_vs_{dimensions[j]}"
            null_distributions[pair_name].append(float(tau))

    # Compute p-values
    results = {}
    for pair_name, obs_tau in observed.items():
        null = np.array(null_distributions[pair_name])
        p_value = float(np.mean(np.abs(null) >= np.abs(obs_tau)))
        results[pair_name] = {
            "observed_tau": round(obs_tau, 4),
            "p_value": round(p_value, 6),
            "null_mean": round(float(null.mean()), 4),
            "null_std": round(float(null.std()), 4),
            "null_5th": round(float(np.percentile(null, 5)), 4),
            "null_95th": round(float(np.percentile(null, 95)), 4),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--subsample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = permutation_test(
        args.scores, args.n_permutations, args.subsample, args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[perm] Results saved to {args.output}")
    print("\nPermutation test summary:")
    for pair, data in results.items():
        sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else "ns"
        print(f"  {pair}: τ={data['observed_tau']:.4f}, p={data['p_value']:.4f} {sig}")
