"""Generate figures and summary tables from analysis results.

Usage:
    python scripts/generate_report.py output/alpaca/ --figures figures/
"""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_heatmap(matrix: dict, output_path: Path, title: str, vmin: float = -0.2, vmax: float = 1.0):
    dims = sorted(matrix.keys())
    data = [[matrix[a].get(b, 0) for b in dims] for a in dims]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        data, annot=True, fmt=".3f",
        xticklabels=dims, yticklabels=dims,
        cmap="RdYlBu_r", vmin=vmin, vmax=vmax, ax=ax,
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_quality_loss(ql: dict, output_path: Path):
    dims = sorted(ql.keys())
    deltas = [ql[d]["delta"] for d in dims]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(dims, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Quality Score Delta (goal-specific - universal)")
    ax.set_title("Quality Loss: Goal-Specific vs Universal Filtering")
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, d in zip(bars, dims):
        p = ql[d]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                sig, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sensitivity(sens: dict, output_path: Path):
    rates = sorted(sens.keys(), key=float)
    fig, ax = plt.subplots(figsize=(8, 5))

    for rate in rates:
        data = sens[rate]
        mean_j = data.get("jaccard_off_diag_mean", 0)
        min_j = data.get("jaccard_off_diag_min", 0)
        max_j = data.get("jaccard_off_diag_max", 0)
        ax.scatter([float(rate)], [mean_j], s=100, zorder=5)
        ax.plot([float(rate), float(rate)], [min_j, max_j],
                color="gray", linewidth=2, zorder=4)

    ax.set_xlabel("Retention Rate")
    ax.set_ylabel("Jaccard Similarity (off-diagonal)")
    ax.set_title("Sensitivity: Subset Overlap Across Retention Rates")
    ax.axhline(0.6, color="red", linestyle="--", label="J = 0.6 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_markdown_tables(result_dir: Path) -> str:
    lines = []

    # Tau matrix
    tau_path = result_dir / "tau_matrix.json"
    if tau_path.exists():
        with open(tau_path) as f:
            data = json.load(f)
        tau = data["tau"]
        p_vals = data["p_values"]
        dims = sorted(tau.keys())
        lines.append("### Kendall's τ Matrix\n")
        header = "| | " + " | ".join(dims) + " |"
        sep = "|---|" + "|".join(["---"] * len(dims)) + "|"
        lines.append(header)
        lines.append(sep)
        for d in dims:
            row = f"| **{d}** | " + " | ".join(
                f"{tau[d].get(d2, 'N/A'):.3f}" for d2 in dims
            ) + " |"
            lines.append(row)
        lines.append("")

    # Quality loss
    ql_path = result_dir / "quality_loss.json"
    if ql_path.exists():
        with open(ql_path) as f:
            ql = json.load(f)
        lines.append("### Quality Loss: Goal-Specific vs Universal\n")
        lines.append("| Dimension | Goal Mean | Universal Mean | Δ | p-value | Effect Size |")
        lines.append("|-----------|-----------|----------------|---|---------|-------------|")
        for dim, data in sorted(ql.items()):
            lines.append(
                f"| {dim} | {data['goal_specific_mean']:.4f} | {data['universal_mean']:.4f} | "
                f"{data['delta']:+.4f} | {data['p_value']:.6f} | {data['effect_size']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)

    # Tau heatmap
    tau_path = args.result_dir / "tau_matrix.json"
    if tau_path.exists():
        with open(tau_path) as f:
            tau_data = json.load(f)
        plot_heatmap(tau_data["tau"], args.figures / "kendall_heatmap.png", "Kendall's τ Between Quality Dimensions")

    # Jaccard heatmap
    jac_path = args.result_dir / "jaccard_matrix.json"
    if jac_path.exists():
        with open(jac_path) as f:
            jac_data = json.load(f)
        plot_heatmap(jac_data, args.figures / "jaccard_heatmap.png", "Jaccard Similarity Between Subsets", vmin=0, vmax=1)

    # Quality loss bar
    ql_path = args.result_dir / "quality_loss.json"
    if ql_path.exists():
        with open(ql_path) as f:
            ql_data = json.load(f)
        plot_quality_loss(ql_data, args.figures / "quality_loss_bar.png")

    # Sensitivity
    sens_path = args.result_dir / "sensitivity.json"
    if sens_path.exists():
        with open(sens_path) as f:
            sens_data = json.load(f)
        plot_sensitivity(sens_data, args.figures / "sensitivity_plot.png")

    # Markdown tables
    tables = generate_markdown_tables(args.result_dir)
    with open(args.result_dir / "tables.md", "w") as f:
        f.write(tables)
    print(f"\n[report] Tables -> {args.result_dir / 'tables.md'}")
    print(tables)
