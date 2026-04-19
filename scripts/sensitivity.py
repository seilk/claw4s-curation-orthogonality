"""Sensitivity analysis: vary retention rate and check if finding holds.

Usage:
    python scripts/sensitivity.py output/alpaca/scores.jsonl \
        --rates 0.2,0.3,0.5 \
        --output output/alpaca/sensitivity.json
"""
import json
import argparse
from pathlib import Path

from curate_by_goal import load_scored, curate_by_dimension, curate_universal
from rank_correlation import compute_tau_matrix


def sensitivity_analysis(
    scores_path: Path, rates: list[float], output_path: Path
) -> dict:
    from rank_correlation import compute_jaccard_matrix, compute_quality_loss
    samples = load_scored(scores_path)
    dimensions = list(samples[0]["scores"].keys())

    results = {}
    for rate in rates:
        print(f"\n[sensitivity] retention={rate}")
        # Generate subsets at this retention rate
        subsets = {}
        for dim in dimensions:
            subsets[dim] = curate_by_dimension(samples, dim, rate)
        subsets["universal"] = curate_universal(samples, dimensions, rate)

        # Compute Jaccard at this rate
        jaccard = compute_jaccard_matrix(subsets)

        # Extract off-diagonal Jaccard for goal-specific dimensions only
        off_diag_jac = []
        for a in dimensions:
            for b in dimensions:
                if a != b:
                    off_diag_jac.append(jaccard.get(a, {}).get(b, 0))

        # Quality loss at this rate
        ql = compute_quality_loss(samples, subsets, dimensions)

        results[str(rate)] = {
            "retention_rate": rate,
            "jaccard_off_diag_mean": round(sum(off_diag_jac) / max(len(off_diag_jac), 1), 4),
            "jaccard_off_diag_max": round(max(off_diag_jac), 4),
            "jaccard_off_diag_min": round(min(off_diag_jac), 4),
            "quality_loss_summary": {d: round(v["delta"], 4) for d, v in ql.items()},
            "all_jaccard_below_0.6": all(j < 0.6 for j in off_diag_jac),
        }
        print(f"  Jaccard mean={results[str(rate)]['jaccard_off_diag_mean']}, max={results[str(rate)]['jaccard_off_diag_max']}, all<0.6={results[str(rate)]['all_jaccard_below_0.6']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[sensitivity] Results -> {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--rates", type=str, default="0.2,0.3,0.5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rates = [float(r) for r in args.rates.split(",")]
    sensitivity_analysis(args.scores, rates, args.output)
