# Curation Orthogonality in Instruction-Tuning Data

Reproduction repository for a Claw4S 2026 submission analyzing the near-independence of goal-specific curation strategies across five quality dimensions in instruction-tuning corpora (Alpaca, WizardLM), with downstream validation on Qwen3.5.

## Artifacts

| File | Purpose |
|------|---------|
| [`paper.md`](paper.md) | Full paper (Markdown with KaTeX math) |
| [`tex/main.pdf`](tex/main.pdf) | 4-page LaTeX research note |
| [`tex/main.tex`](tex/main.tex) | LaTeX source of the research note |
| [`SKILL.md`](SKILL.md) | Executable reproduction skill (agent-parseable) |
| [`scripts/`](scripts/) | Analysis pipeline |
| [`figures/`](figures/) | Publication figures |

## Reproduce

```bash
pip install -r scripts/requirements.txt
cp .env.example .env    # fill in LLM_API_KEY

python scripts/download_data.py
python scripts/score_dimensions.py data/raw/alpaca.jsonl --output data/scored/alpaca_scores.jsonl
python scripts/curate_by_goal.py
python scripts/rank_correlation.py
python scripts/permutation_test.py
python scripts/bootstrap_scoring.py
python scripts/sensitivity.py
python scripts/generate_report.py
```

For the agent-driven path, see [`SKILL.md`](SKILL.md).

## Key findings

- Near-zero Kendall's τ between diversity and information density (τ = −0.025), with top-30% Jaccard J = 0.154 (random null J = 0.176)
- Bootstrap-validated accuracy–relevance τ = 0.442 ± 0.014 (Alpaca, K = 5, N = 1,000, no imputation)
- Downstream Qwen3.5 finetuning: diversity-optimized subsets yield +5.2pp Distinct-2 and −6.9pp Self-BLEU vs. universal filtering
- Cross-dataset: WizardLM shows elevated inter-correlation among statistical dimensions (τ up to 0.499) — synthetic evolutionary generation introduces shared structural artifacts

## License

MIT. See [`LICENSE`](LICENSE).
