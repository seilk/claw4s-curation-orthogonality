"""Bootstrap subsampling: score K random subsamples of size N fully via LLM.

This eliminates the median-imputation problem by computing τ only on
fully-scored subsamples. Reports mean ± std across bootstrap iterations.

Usage:
    python scripts/bootstrap_scoring.py data/raw/alpaca.jsonl \
        --k 5 --n 1000 --output data/results/alpaca/bootstrap/ \
        --cache-dir data/cache/bootstrap/
"""
import json
import random
import argparse
import time
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau

from score_dimensions import (
    score_conciseness,
    score_info_density,
    score_diversity_batch,
    build_judge_prompt,
    call_llm_judge,
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_subsample_full(
    samples: list[dict],
    cache_path: Path | None = None,
) -> list[dict]:
    """Score ALL samples on all 5 dimensions — no imputation."""
    n = len(samples)

    # Statistical dimensions (fast, no API)
    conciseness = [score_conciseness(s) for s in samples]
    diversity = score_diversity_batch(samples)
    info_density = [score_info_density(s) for s in samples]

    # LLM dimensions — score EVERY sample
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    accuracy_scores = []
    relevance_scores = []

    for i, sample in enumerate(samples):
        key = f"{sample.get('id', i)}"

        # Accuracy
        if f"accuracy_{key}" in cache:
            acc = cache[f"accuracy_{key}"]
        else:
            prompt = build_judge_prompt(sample, "accuracy")
            acc = call_llm_judge(prompt)
            cache[f"accuracy_{key}"] = acc
        accuracy_scores.append(acc / 10.0)

        # Relevance
        if f"relevance_{key}" in cache:
            rel = cache[f"relevance_{key}"]
        else:
            prompt = build_judge_prompt(sample, "relevance")
            rel = call_llm_judge(prompt)
            cache[f"relevance_{key}"] = rel
        relevance_scores.append(rel / 10.0)

        if (i + 1) % 100 == 0:
            print(f"    scored {i+1}/{n}")
            # Save cache periodically
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

    # Final cache save
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    # Build scored samples
    scored = []
    for i, sample in enumerate(samples):
        scored.append({
            **sample,
            "_original_index": sample.get("id", i),
            "scores": {
                "accuracy": accuracy_scores[i],
                "relevance": relevance_scores[i],
                "conciseness": conciseness[i],
                "diversity": diversity[i],
                "info_density": info_density[i],
            },
        })

    return scored


def compute_tau_matrix(scored: list[dict]) -> dict:
    dims = list(scored[0]["scores"].keys())
    vecs = {d: np.array([s["scores"][d] for s in scored]) for d in dims}

    tau_matrix = {}
    for a, b in combinations(dims, 2):
        tau, p = kendalltau(vecs[a], vecs[b])
        tau_matrix[f"{a}_vs_{b}"] = {"tau": round(float(tau), 4), "p": round(float(p), 8)}

    return tau_matrix


def run_bootstrap(
    data_path: Path,
    k: int = 5,
    n: int = 1000,
    output_dir: Path = Path("data/results/alpaca/bootstrap"),
    cache_dir: Path = Path("data/cache/bootstrap"),
    seed_base: int = 42,
):
    all_samples = load_jsonl(data_path)
    print(f"[bootstrap] Loaded {len(all_samples)} samples from {data_path}")
    print(f"[bootstrap] Running {k} iterations, {n} samples each")

    all_tau_results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(k):
        seed = seed_base + iteration
        rng = random.Random(seed)
        subsample = rng.sample(all_samples, min(n, len(all_samples)))

        print(f"\n[bootstrap] === Iteration {iteration+1}/{k} (seed={seed}) ===")
        print(f"  Scoring {len(subsample)} samples fully (no imputation)...")

        cache_path = cache_dir / f"iter_{iteration}.json"
        scored = score_subsample_full(subsample, cache_path)

        tau_result = compute_tau_matrix(scored)
        all_tau_results.append(tau_result)

        # Save per-iteration results
        iter_path = output_dir / f"iter_{iteration}.json"
        with open(iter_path, "w") as f:
            json.dump(tau_result, f, indent=2)

        print(f"  τ results:")
        for pair, data in sorted(tau_result.items()):
            print(f"    {pair}: τ={data['tau']:.4f} p={data['p']:.4f}")

    # Aggregate: mean ± std
    pairs = sorted(all_tau_results[0].keys())
    summary = {}
    for pair in pairs:
        taus = [r[pair]["tau"] for r in all_tau_results]
        summary[pair] = {
            "mean_tau": round(float(np.mean(taus)), 4),
            "std_tau": round(float(np.std(taus)), 4),
            "min_tau": round(float(np.min(taus)), 4),
            "max_tau": round(float(np.max(taus)), 4),
            "all_taus": [round(t, 4) for t in taus],
            "n_iterations": k,
            "subsample_size": n,
        }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[bootstrap] === SUMMARY (K={k}, N={n}) ===")
    for pair, data in sorted(summary.items()):
        print(f"  {pair:>35}: τ = {data['mean_tau']:.4f} ± {data['std_tau']:.4f}  [{data['min_tau']:.4f}, {data['max_tau']:.4f}]")

    print(f"\n[bootstrap] Results saved to {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--k", type=int, default=5, help="Number of bootstrap iterations")
    parser.add_argument("--n", type=int, default=1000, help="Subsample size per iteration")
    parser.add_argument("--output", type=Path, default=Path("data/results/alpaca/bootstrap"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/bootstrap"))
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args()
    run_bootstrap(args.data, args.k, args.n, args.output, args.cache_dir, args.seed_base)
