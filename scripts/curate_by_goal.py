"""Generate goal-specific curated subsets by selecting top-K% per dimension.

Usage:
    python scripts/curate_by_goal.py output/alpaca/scores.jsonl \
        --output output/alpaca/subsets.json \
        --retention 0.3
"""
import json
import math
import random
import argparse
from pathlib import Path


def load_scored(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def curate_by_dimension(
    samples: list[dict], dimension: str, retention: float = 0.3
) -> list[int]:
    scored = [(s["_original_index"], s["scores"][dimension]) for s in samples]
    scored.sort(key=lambda x: x[1], reverse=True)
    k = max(1, math.ceil(len(samples) * retention))
    return [idx for idx, _ in scored[:k]]


def curate_universal(
    samples: list[dict], dimensions: list[str], retention: float = 0.3
) -> list[int]:
    composite = []
    for s in samples:
        avg = sum(s["scores"][d] for d in dimensions) / len(dimensions)
        composite.append((s["_original_index"], avg))
    composite.sort(key=lambda x: x[1], reverse=True)
    k = max(1, math.ceil(len(samples) * retention))
    return [idx for idx, _ in composite[:k]]


def curate_random(
    samples: list[dict], retention: float = 0.3, seed: int = 42
) -> list[int]:
    rng = random.Random(seed)
    k = max(1, math.ceil(len(samples) * retention))
    all_indices = [s["_original_index"] for s in samples]
    return sorted(rng.sample(all_indices, k))


def generate_all_subsets(
    scored_path: Path, output_path: Path, retention: float = 0.3
) -> dict:
    samples = load_scored(scored_path)
    dimensions = list(samples[0]["scores"].keys())

    subsets = {}
    for dim in dimensions:
        indices = curate_by_dimension(samples, dim, retention)
        subsets[dim] = indices
        print(f"  [{dim}] {len(indices)} samples selected")

    subsets["universal"] = curate_universal(samples, dimensions, retention)
    print(f"  [universal] {len(subsets['universal'])} samples selected")

    subsets["random"] = curate_random(samples, retention)
    print(f"  [random] {len(subsets['random'])} samples selected")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subsets, f, indent=2)

    print(f"[curate] {len(subsets)} subsets -> {output_path}")
    return subsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scored_data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention", type=float, default=0.3)
    args = parser.parse_args()
    generate_all_subsets(args.scored_data, args.output, args.retention)
