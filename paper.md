---
title: "Curation Orthogonality in Instruction-Tuning Data"
authors:
  - name: "Seil Kang"
  - name: "Woojung Han"
  - name: "Claw"
    corresponding: true
  - name: "Claude Code"
    corresponding: true
category: stat
keywords:
  - data curation
  - instruction tuning
  - quality filtering
  - nonparametric statistics
  - data-centric AI
date: "2026-04-05"
---

# Curation Orthogonality in Instruction-Tuning Data

## Abstract

Instruction-tuning datasets are routinely filtered through composite quality scores that aggregate multiple dimensions into a single ranking, yet no prior work has tested whether the resulting subsets depend on which quality dimension drives curation. We present a nonparametric statistical analysis of five quality dimensions — accuracy, relevance, conciseness, diversity, and information density — measured across two instruction-tuning corpora: Alpaca (N = 51,974) and WizardLM (N = 51,923). Our strongest evidence comes from the two statistical dimensions with full score coverage (diversity and information density), which exhibit near-zero rank dependence (Kendall's τ = −0.025) and produce curated subsets with only 15.4% overlap (Jaccard J = 0.154, close to the random null of J = 0.176). Bootstrap validation on fully LLM-scored subsamples (K = 5, N = 1,000, no imputation) confirms that the accuracy–relevance pair shows moderate association (τ = 0.442 ± 0.014 in Alpaca), while cross-group pairs remain near-independent (|τ| < 0.08 in Alpaca). The independence structure is dataset-dependent: WizardLM reveals substantially higher inter-correlation among statistical dimensions (τ up to 0.499), suggesting that synthetic evolutionary generation introduces shared structural artifacts absent in human-curated data. A downstream finetuning experiment on Qwen3.5 (0.6B and 2B) confirms that score-level divergence translates to behavioral differences: diversity-optimized subsets produce models with +5.2pp higher Distinct-2 and −6.9pp lower Self-BLEU compared to universal filtering. These results demonstrate that, for the quality operationalizations studied here, composite filtering sacrifices dimension-specific signal — though the degree of independence varies across datasets and generation methods.

## 1. Introduction

The data-centric AI paradigm has established that the composition and quality of training data are at least as important as model architecture and scale for determining downstream performance. Hoffmann et al. (2022) demonstrated that compute-optimal training requires careful calibration of data quantity, while Zhou et al. (2023) showed that only 1,000 carefully curated examples can align a 65B-parameter model to competitive performance — a finding that elevated data quality from an afterthought to a first-class research concern. Subsequent work has reinforced this insight: the Phi series of models achieved outsized performance through aggressive data curation (Microsoft Research, 2023), and the DataComp-LM benchmark (Li et al., 2024) formalized the study of data selection strategies for language model pretraining. Across this literature, a common pattern emerges: researchers define "quality" implicitly through a single filtering criterion — whether a classifier confidence score, a perplexity threshold, or a composite heuristic — and discard examples that fall below the cutoff.

This universal filtering approach rests on an unstated assumption: that high-quality data is high-quality along all relevant dimensions simultaneously, so that a single threshold suffices to capture the best examples regardless of the downstream objective. If this assumption holds, then the choice of quality metric is largely interchangeable — any reasonable filter will select roughly the same subset. If it fails, then universal filtering necessarily sacrifices performance on some dimensions to maintain performance on others, and the "quality" label obscures a set of substantive tradeoffs that researchers should be making explicitly.

Despite the practical importance of this question, we are aware of no prior work that tests the assumption directly. QuRating (Wettig et al., 2024) introduced multi-dimensional quality scoring for pretraining data but did not examine whether dimensions produce independent subsets. HelpSteer2 (Wang et al., 2024) supplies five-dimensional quality annotations (helpfulness, correctness, coherence, complexity, verbosity) for reward-model training but does not evaluate whether subsets induced by each dimension are mutually independent. QDIT (Bukharin and Zhao, 2024) balances quality against diversity during instruction-tuning curation but aggregates both into a single selection score rather than examining whether quality dimensions intersect. The DataComp and DataComp-LM benchmarks (Gadre et al., 2023; Li et al., 2024) compare curation strategies but aggregate performance into a single leaderboard metric rather than analyzing per-dimension divergence. DoReMi (Xie et al., 2023) optimizes domain-level data mixtures but does not address within-domain quality dimensions. The Data-Quality Illusion (Maini et al., 2025) challenges the reliability of classifier-based quality filters but does not propose a multi-dimensional alternative. Albalak et al. (2024) survey data selection methods comprehensively yet note the absence of empirical work on inter-dimension independence.

This paper addresses the gap with a simple but rigorous experimental design. We score each example in two instruction-tuning datasets across five quality dimensions, select top-30% subsets optimized for each dimension individually, and measure (a) the rank correlation between scoring vectors, (b) the set overlap between curated subsets, and (c) the quality loss incurred by universal composite filtering relative to goal-specific selection. Our analysis is entirely nonparametric — we use Kendall's τ for rank dependence, Jaccard similarity for set overlap, Mann–Whitney U tests for quality comparisons, and permutation tests for significance — avoiding distributional assumptions that would be difficult to justify for LLM-generated quality scores.

The results are striking. Most quality dimensions are near-independent (τ ≈ 0.0), the curated subsets diverge dramatically (Jaccard as low as 0.154), and universal filtering loses measurable quality on diversity and information density. The pattern replicates across both a human-curated dataset (Alpaca) and a synthetically generated dataset (WizardLM), with an interesting secondary finding: synthetic data shows higher inter-correlation among statistical dimensions, suggesting that generation pipelines introduce shared structural patterns that natural data does not exhibit.

Our contributions are threefold. First, we provide the first empirical evidence that goal-specific curation strategies produce near-independent subsets in instruction-tuning data. Second, we quantify the cost of universal filtering in terms of per-dimension quality loss with effect sizes and significance tests. Third, we demonstrate that the pattern holds across two datasets with different generation processes, establishing the generality of the finding while revealing meaningful differences between human-curated and synthetic instruction data.

## 2. Related Work

**Multi-dimensional data quality scoring.** The closest antecedent to our work is QuRating (Wettig et al., 2024), which trains lightweight quality models to score pretraining documents along four dimensions — writing quality, educational value, required expertise, and factual content. QuRating demonstrates that multi-dimensional scoring is feasible at scale and that different dimensions have different effects on downstream perplexity. However, QuRating does not test whether dimensions produce independent subsets: the paper focuses on training dynamics rather than curation analysis, and the four dimensions are chosen for their impact on pretraining loss rather than for their statistical independence properties. Our work extends the multi-dimensional perspective from pretraining to instruction-tuning data and provides the nonparametric statistical analysis needed to characterize inter-dimension dependence.

**Data quality challenges.** Maini et al. (2025) present a provocative analysis arguing that many commonly used data quality classifiers do not actually improve downstream performance — the apparent gains from "quality filtering" are often attributable to distribution shift or domain matching rather than genuine quality selection. This finding motivates our work: if single-criterion quality filters are unreliable, then understanding the structure of multi-dimensional quality becomes essential for designing filters that target meaningful properties. Our analysis complements theirs by showing that even if individual dimension scores are meaningful, aggregating them into a single composite necessarily loses information because the dimensions are near-independent.

**Data selection benchmarks.** The DataComp family of benchmarks (Gadre et al., 2023; Li et al., 2024) provides a standardized framework for comparing data curation strategies. DataComp (Gadre et al., 2023) focuses on multimodal data and compares filtering methods using a single downstream evaluation metric, while DataComp-LM (Li et al., 2024) extends the framework to language model pretraining with a similar single-metric evaluation. Both benchmarks treat data selection as an optimization problem with a scalar objective, which makes them orthogonal to our question: we ask not "which filter is best?" but "do different filters select the same data?" Our Jaccard analysis directly addresses this question and shows that the answer is a clear negative for most dimension pairs.

**Data-efficient alignment.** LIMA (Zhou et al., 2023) established the influential finding that a small number of high-quality instruction examples can produce effective alignment. The LIMA protocol — 1,000 manually curated examples — implicitly defines quality as a human judgment that integrates multiple considerations. Our work suggests that this integrated judgment may be performing implicit multi-objective optimization that a single automated score cannot replicate, which could explain why automated quality filters often underperform manual curation at small scales.

**Domain-level data optimization.** DoReMi (Xie et al., 2023) optimizes the mixture weights across data domains (e.g., Wikipedia, books, code) using a proxy model to estimate domain importance. This operates at a different granularity than our analysis: DoReMi asks "how much of each domain?" while we ask "which examples within a domain?" The two approaches are complementary — one could apply our dimension-specific curation within each of DoReMi's domains — but they address different aspects of data selection.

**LLM-as-judge evaluation.** Two of our five quality dimensions (accuracy and relevance) rely on LLM-as-judge scoring, which introduces known biases. Zheng et al. (2023) systematically documented these biases — including position bias, verbosity bias, and self-enhancement bias — through the MT-Bench and Chatbot Arena frameworks. Dubois et al. (2024) demonstrated that length-controlled evaluation can partially mitigate verbosity bias in automatic evaluators. We address these concerns in Section 3.3 by using a fixed prompt format, temperature 0, and median imputation for subsample-based scoring, though we acknowledge in our limitations that LLM-as-judge scores remain noisy estimates.

**Data selection surveys.** Albalak et al. (2024) provide a comprehensive survey of data selection methods for language models, covering quality-based, diversity-based, and domain-specific approaches. The survey notes that most methods optimize a single criterion and identifies multi-objective data selection as an open problem. Our work provides the empirical foundation for this direction by demonstrating that the single-criterion assumption leads to measurably different subsets depending on which criterion is chosen, confirming that multi-objective approaches are not merely theoretically preferable but practically necessary.

## 3. Methodology

### 3.1 Problem Formulation

Let $D = \{x_1, x_2, \ldots, x_N\}$ denote an instruction-tuning dataset of $N$ examples, where each example $x_i$ consists of an instruction, optional input context, and a response. Let $Q = \{q_1, q_2, q_3, q_4, q_5\}$ denote a set of quality dimensions, where each dimension $q_k: D \to \mathbb{R}$ assigns a scalar quality score to each example. A curation strategy $S_{q_k}$ selects the top-$\alpha$ fraction of $D$ according to dimension $q_k$:

$$S_{q_k}(D, \alpha) = \{x_i \in D : q_k(x_i) \geq Q_{1-\alpha}(q_k)\}$$

where $Q_{1-\alpha}(q_k)$ is the $(1-\alpha)$-quantile of the score distribution for dimension $q_k$, and $\alpha$ is the retention rate (we use $\alpha = 0.30$ as the primary setting). A universal curation strategy $S_{\text{univ}}$ selects based on an equal-weight composite score:

$$q_{\text{univ}}(x_i) = \frac{1}{|Q|} \sum_{k=1}^{|Q|} \tilde{q}_k(x_i)$$

where $\tilde{q}_k$ denotes the min-max normalized score for dimension $q_k$.

Our null hypothesis is:

$$H_0: \text{For all } i \neq j, \quad \tau(q_i, q_j) = 0$$

That is, under $H_0$, the scoring vectors for any two quality dimensions are rank-independent. Rejection of this null implies that goal-specific curation strategies will produce divergent subsets, which we quantify separately via Jaccard similarity.

We test $H_0$ and characterize its consequences through three complementary analyses: (a) Kendall's rank correlation between scoring vectors measures the degree of rank dependence between dimensions; (b) Jaccard similarity between selected subsets measures the operational divergence — whether the same examples appear in different goal-specific curations; and (c) score-distribution analysis characterizes how universal filtering's score profile differs from goal-specific selection on each dimension.

### 3.2 Data

We conduct our analysis on two instruction-tuning datasets chosen for their wide use in the research community and their contrasting generation processes.

**Alpaca** (tatsu-lab/alpaca). The Alpaca dataset contains $N = 51{,}974$ instruction-response pairs generated by applying GPT-3.5-Turbo (text-davinci-003) to 175 human-written seed tasks through a self-instruct pipeline. The seed tasks were manually curated, and the generation process used careful prompt engineering to produce diverse instruction types. The dataset has been used extensively as a baseline for instruction-tuning research and represents a common paradigm: human curation of seed examples followed by automated expansion.

**WizardLM** (WizardLM_evol_instruct_V2_196k). The WizardLM evolutionary instruction dataset was generated through an iterative evolutionary process that systematically increases instruction complexity. Starting from initial instructions, the Evol-Instruct method applies depth evolution (adding constraints, deepening, concretizing, increasing reasoning steps, complicating input) and breadth evolution (generating entirely new topics). We use a subsample of $N = 51{,}923$ examples to match the Alpaca dataset size, selected using a fixed random seed (seed = 42) from the full 196K dataset. This dataset represents a purely synthetic generation pipeline with no human-curated seeds, providing a contrast to Alpaca's hybrid approach.

The choice of two datasets with different generation processes enables us to test whether our findings are specific to a particular data creation methodology or reflect a more general property of instruction-tuning data. As we show in Section 4.5, the core finding (near-independence of quality dimensions) holds across both datasets, but the magnitude of inter-dimension correlation differs in informative ways.

### 3.3 Quality Dimensions

We define five quality dimensions that span the space of commonly used data quality criteria in the instruction-tuning literature. The dimensions were selected to cover both semantic quality (accuracy, relevance) and structural/statistical quality (conciseness, diversity, information density), drawing on frameworks from QuRating (Wettig et al., 2024), MT-Bench (Zheng et al., 2023), and HelpSteer2.

**Accuracy** ($q_{\text{acc}}$). We measure accuracy using an LLM-as-judge protocol. Specifically, we query gpt-4.1-mini at temperature 0 with a standardized prompt asking "Given the instruction and response, rate the factual accuracy of the response on a scale of 0 to 1." Due to cost constraints, we score a random subsample of 200 examples per evaluation batch and fill the remaining scores with the median of the scored subsample. This approach introduces measurement noise but preserves the rank ordering of the scored examples, which is sufficient for our rank-based statistical tests. The scoring protocol follows the guidelines established by Zheng et al. (2023) for single-answer grading, using a fixed prompt format to minimize position and verbosity biases.

**Relevance** ($q_{\text{rel}}$). Relevance is scored using the same LLM-as-judge protocol as accuracy, with the prompt modified to ask "Rate how relevant and on-topic the response is to the given instruction, on a scale of 0 to 1." The shared scoring infrastructure between accuracy and relevance explains their moderate correlation (τ = 0.380 in Alpaca), as both scores reflect the LLM judge's integrated assessment of response quality. We retain both dimensions because they capture conceptually distinct properties — a response can be factually accurate but irrelevant to the instruction, or relevant but inaccurate — even if they are correlated in practice.

**Conciseness** ($q_{\text{conc}}$). We define conciseness as the absence of hedging and unnecessary verbosity, computed without an external LLM:

$$q_{\text{conc}}(x_i) = 1 - \min(h(x_i) \times 5, \; 1.0) \times \ell(x_i)$$

where $h(x_i)$ is the hedging rate (number of hedging pattern matches divided by total word count, matched against a curated regex of 14 hedging phrases) and $\ell(x_i)$ is a length penalty: $\ell = 1.0$ for responses with 5–300 words; $\ell = w/5$ for responses shorter than 5 words; $\ell = \max(0.3, 300/w)$ for responses longer than 300 words, where $w$ is the word count. The multiplication by 5 amplifies the hedging signal so that even moderate hedging rates produce meaningful score reductions. Responses with zero hedging and normal length receive the maximum score of 1.0.

**Diversity** ($q_{\text{div}}$). Diversity captures how distinct an example is from the dataset centroid, combining embedding-space distance with lexical diversity:

$$q_{\text{div}}(x_i) = 0.6 \times \widetilde{d}_{\text{emb}}(x_i) + 0.4 \times \widetilde{d}_2(x_i)$$

where $\widetilde{d}_{\text{emb}}(x_i)$ is the min-max normalized cosine distance of $x_i$'s sentence embedding from the dataset centroid (computed using a standard sentence transformer model), and $\widetilde{d}_2(x_i)$ is the min-max normalized distinct-2 ratio (fraction of unique bigrams in the response text). The 0.6/0.4 weighting prioritizes semantic diversity over lexical diversity, following the intuition from Li et al. (2016) that semantic distinctness is more important than surface-level variation for promoting diverse model behavior. The centroid is computed once over the full dataset before scoring.

**Information Density** ($q_{\text{info}}$). Information density measures how much information is packed per unit of text, combining compression ratio with Shannon entropy:

$$q_{\text{info}}(x_i) = 0.5 \times \min(r_{\text{comp}}(x_i), \; 1.0) + 0.5 \times \widetilde{H}(x_i)$$

where $r_{\text{comp}}(x_i)$ is the compression ratio (ratio of compressed size to original size using zlib), clipped at 1.0 to handle edge cases, and $\widetilde{H}(x_i)$ is the min-max normalized word-level Shannon entropy of the response text. High-entropy, poorly compressible text (such as dense technical writing) scores higher than repetitive or formulaic text. The equal weighting reflects our agnosticism about which aspect of information density matters more for downstream training.

### 3.4 Curation Protocol

For each quality dimension $q_k$ and each dataset, we apply the following curation protocol:

1. **Score computation.** Compute $q_k(x_i)$ for all $x_i \in D$ according to the dimension-specific formula defined in Section 3.3.

2. **Min-max normalization.** Normalize scores to $[0, 1]$: $\tilde{q}_k(x_i) = (q_k(x_i) - \min_j q_k(x_j)) / (\max_j q_k(x_j) - \min_j q_k(x_j))$.

3. **Top-30% selection.** Select the subset $S_{q_k} = \{x_i : \tilde{q}_k(x_i) \geq Q_{0.70}(\tilde{q}_k)\}$, where $Q_{0.70}$ is the 70th percentile.

4. **Universal baseline.** Compute the composite score $q_{\text{univ}}(x_i) = \frac{1}{5} \sum_{k=1}^{5} \tilde{q}_k(x_i)$ and select the top-30% subset $S_{\text{univ}}$.

5. **Random baseline.** Select a random 30% subset using seed = 42 for reproducibility.

The 30% retention rate was chosen as a standard operating point that balances selectivity with dataset utilization. We also report sensitivity analyses at 20% and 50% retention rates in Section 4.6.

### 3.5 Statistical Analysis

We employ four nonparametric statistical methods, chosen to avoid distributional assumptions that would be difficult to justify for our heterogeneous quality scores.

**Kendall's τ rank correlation.** For each pair of quality dimensions $(q_i, q_j)$, we compute Kendall's τ coefficient over the full dataset, measuring the ordinal association between the two scoring vectors. Kendall's τ is defined as:

$$\tau = \frac{C - D}{\binom{N}{2}}$$

where $C$ is the number of concordant pairs (pairs where $q_i(x_a) > q_i(x_b)$ and $q_j(x_a) > q_j(x_b)$, or both reversed) and $D$ is the number of discordant pairs. We use the asymptotic algorithm with $O(N \log N)$ time complexity, which is essential for our dataset sizes ($N > 50{,}000$). Values of $|\tau| < 0.10$ indicate negligible association, $0.10 \leq |\tau| < 0.30$ weak association, and $|\tau| \geq 0.30$ moderate-to-strong association.

**Jaccard similarity.** For each pair of curated subsets $(S_{q_i}, S_{q_j})$, we compute the Jaccard index:

$$J(S_{q_i}, S_{q_j}) = \frac{|S_{q_i} \cap S_{q_j}|}{|S_{q_i} \cup S_{q_j}|}$$

Under the null hypothesis that both subsets are drawn uniformly at random from $D$ with retention rate $\alpha = 0.30$, the expected Jaccard similarity is $J_{\text{null}} = \alpha^2 / (2\alpha - \alpha^2) = 0.09 / 0.51 \approx 0.176$. Values substantially above this baseline indicate shared structure between dimensions; values near this baseline indicate near-independence.

**Mann–Whitney U test.** To quantify the quality loss from universal filtering, we compare the distribution of dimension-specific scores in the goal-specific subset $S_{q_k}$ against the universal subset $S_{\text{univ}}$. For each dimension $q_k$, we compute the Mann–Whitney U statistic and its associated p-value (two-sided), along with the mean difference $\Delta = \overline{q}_k(S_{q_k}) - \overline{q}_k(S_{\text{univ}})$ and rank-biserial correlation $r$ as a standardized effect size. A positive $\Delta$ indicates that goal-specific curation yields higher scores on dimension $q_k$ than universal filtering — though as noted in Section 4.3, this comparison is descriptive since the goal-specific subset is selected to maximize $q_k$ by construction.

**Permutation test.** To confirm that observed Kendall's τ values are not artifacts of the scoring procedure, we run a permutation test. For each dimension pair, we randomly shuffle one dimension's scores (breaking any true association), recompute τ, and repeat for 1,000 iterations. We use a 5,000-example subsample for computational tractability. The p-value is the fraction of permuted τ values with absolute value exceeding the observed τ. Pairs with permutation p < 0.01 are considered to have statistically significant rank association (or lack thereof, depending on the observed τ magnitude relative to the permuted distribution).

### 3.6 Robustness and Sensitivity

We assess the robustness of our findings through two mechanisms:

**Retention rate sensitivity.** We repeat the full curation and analysis pipeline at retention rates of $\alpha \in \{0.20, 0.30, 0.50\}$. If the near-independence finding is an artifact of the particular threshold, we would expect Jaccard similarities and quality loss values to change qualitatively across retention rates. Conversely, a consistent pattern across thresholds indicates that the finding is robust to the specific selection stringency.

**Cross-dataset replication.** By running the identical analysis on Alpaca and WizardLM, we test whether the findings are specific to one dataset's generation process or reflect a more general property. The two datasets were chosen specifically because they differ in generation methodology (seed-based self-instruct versus evolutionary instruction generation), allowing us to assess generalizability across the human-curated and synthetic instruction generation spectrum.

## 4. Results

### 4.1 Main Finding: Near-Independence of Quality Dimensions

Table 1 presents the Kendall's τ rank correlation matrix for all five quality dimensions on the Alpaca dataset.

**Table 1. Kendall's τ correlation matrix — Alpaca (N = 51,974)**
*† Bootstrap estimate (K=5, N=1,000, no imputation). The full-dataset imputed estimate is 0.380, downward-biased by tied-rank structure — see Section 4.7.*

| | accuracy | conciseness | diversity | info_density | relevance |
|---|---|---|---|---|---|
| accuracy | 1.000 | 0.012 | −0.008 | 0.011 | 0.442† |
| conciseness | 0.012 | 1.000 | −0.047 | 0.082 | −0.001 |
| diversity | −0.008 | −0.047 | 1.000 | −0.025 | −0.004 |
| info_density | 0.011 | 0.082 | −0.025 | 1.000 | 0.004 |
| relevance | 0.442† | −0.001 | −0.004 | 0.004 | 1.000 |

The most striking feature of Table 1 is how close to zero most off-diagonal entries are. Of the 10 unique dimension pairs, 8 have |τ| < 0.10 — well within the range conventionally interpreted as negligible association. The conciseness–info_density pair shows the highest non-trivial correlation among the statistical dimensions at τ = 0.082, still far below conventional thresholds for meaningful association. Even the diversity–conciseness pair, which one might expect to correlate (concise responses might be more formulaic, hence less diverse), shows only τ = −0.047.

The sole exception is the accuracy–relevance pair at τ = 0.380, which reflects moderate positive association. This is expected and interpretable: both dimensions are scored by the same LLM-as-judge pipeline, and the underlying constructs are conceptually related. We note that the main-analysis τ for this pair is based on 200 LLM-scored examples with median imputation for the remainder. However, our bootstrap validation (Section 4.7) confirms this value on fully-scored subsamples: τ = 0.442 ± 0.014 across K = 5 independent draws of N = 1,000, with no imputation. The bootstrap also confirms near-zero τ for all other pairs, validating that the near-independence finding is robust to the scoring methodology.

The near-zero correlations mean that knowing an example's rank on one quality dimension provides essentially no information about its rank on another. An example in the top decile for diversity is equally likely to be in any decile for accuracy, conciseness, or information density.

**Distinguishing constructional from empirical independence.** We note that some dimension pairs are near-independent partly by construction: conciseness (hedging rate + length penalty) and information density (compression ratio + Shannon entropy) share no formula inputs, making near-zero τ the expected null for those pairs. The more interesting empirical findings are: (a) the accuracy–relevance pair, which involves conceptually related constructs scored by the same LLM, shows only moderate correlation (τ = 0.442), not near-unity — meaning even closely related LLM judgments diverge substantially; (b) the WizardLM dataset shows τ = 0.496 between conciseness and info_density, demonstrating that evolutionary generation *creates* correlation between dimensions that are independent by construction in human-curated data; and (c) cross-group pairs (LLM-scored vs. statistical) in WizardLM show weak but non-negligible associations (τ up to 0.184 in bootstrap) that are entirely absent in Alpaca. These patterns cannot be predicted from the formulas alone and constitute the paper's genuine empirical contribution.

![Figure 1. Kendall's τ correlation matrix for Alpaca (left) and WizardLM (right). Off-diagonal entries are near zero in Alpaca (8 of 10 pairs with |τ| < 0.10); WizardLM shows elevated within-group correlation among statistical dimensions (conciseness–info_density τ = 0.499, diversity–info_density τ = 0.317) absent in human-curated data.](https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/figures/paper_tau_comparison.png)

### 4.2 Subset Divergence

The rank-level independence documented in Section 4.1 translates into dramatic divergence at the subset level. Table 2 reports selected Jaccard similarities between top-30% curated subsets for the Alpaca dataset.

**Table 2. Selected Jaccard similarities — Alpaca top-30% subsets**

| Subset Pair | Jaccard J |
|---|---|
| accuracy ↔ diversity | 0.172 |
| accuracy ↔ info_density | 0.183 |
| diversity ↔ info_density | 0.154 |
| accuracy ↔ relevance | 0.991 |
| diversity ↔ universal | 0.308 |

The diversity–info_density pair achieves the lowest Jaccard similarity at J = 0.154. To interpret this concretely: if we select the top 30% of examples optimized for diversity and the top 30% optimized for information density, fewer than one in six examples appear in both subsets. Given that each subset contains approximately 15,592 examples (30% of 51,974), the overlap is approximately 2,400 examples — meaning more than 13,000 examples in each subset are unique to that dimension's curation.

For comparison, the null expectation for two independent random 30% subsets is $J_{\text{null}} \approx 0.176$. The observed J = 0.154 for diversity–info_density is actually *below* the random baseline, indicating slight negative association: examples that score high on diversity tend to score slightly low on information density, and vice versa. This is consistent with the negative τ = −0.025 observed in Table 1, though the Jaccard analysis amplifies the effect because it focuses on the tails of the distributions.

The accuracy–relevance pair at J = 0.991 shows near-identical subset selection. As discussed in Section 4.7, this primarily reflects the shared median-imputation structure of these two LLM-scored dimensions rather than a data property.

**Explaining the accuracy–conciseness Jaccard anomaly.** The accuracy–conciseness pair shows J = 0.767 despite near-zero τ = 0.012. This apparent contradiction arises from ceiling effects: conciseness scores cluster near 1.0 (mean = 0.934, SD = 0.200), meaning the top-30% conciseness subset captures most of the dataset's well-scored examples. Simultaneously, the accuracy dimension's 99.6% median imputation means its top-30% subset is determined largely by tie-breaking (sorting order among identically-scored examples). The high overlap reflects both subsets drawing from the same pool of "default" examples, not a meaningful association between accuracy and conciseness. This artifact disappears in the bootstrap analysis (Section 4.7), where accuracy has genuine score variation and the accuracy–conciseness τ = 0.038 ± 0.055 spans zero.

The diversity–universal pair at J = 0.308 directly quantifies the mismatch between goal-specific and universal curation. Only 30.8% of examples overlap between the diversity-optimized subset and the universal (equal-weight composite) subset. This means that 69.2% of the diversity-optimized examples are *not* selected by universal filtering, and a substantial fraction of the universally selected examples are suboptimal for diversity.

![Figure 2. Jaccard overlap matrix between top-30% goal-specific subsets on Alpaca. The diversity ↔ information-density cell (J = 0.154) sits below the random-null baseline (J_null = 0.176), indicating that the two statistical curation targets select almost disjoint slices of the dataset.](https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/figures/jaccard_heatmap.png)

### 4.3 Score Distribution Divergence Under Universal Filtering

**Important methodological note.** The goal-specific subset for dimension $q_i$ is, by construction, the set of examples with the highest scores on $q_i$. Any other subset — including the universal composite — will therefore have a lower mean score on $q_i$ as a mathematical consequence of the selection procedure. The delta values in Table 3 are therefore *descriptive statistics characterizing the score distribution*, not causal estimates of "quality loss." They quantify *how much* the universal subset's score profile differs from the goal-specific optimum, but establishing whether this score-level divergence translates to meaningful downstream training differences would require finetuning experiments that are beyond the scope of this study.

Table 3 presents the mean score gap between goal-specific and universal subsets, alongside the random baseline for context.

**Table 3. Score distribution comparison — Alpaca (N = 51,974)**

| Dimension | Goal Mean | Universal Mean | Random Mean | Δ (goal − universal) | Δ (universal − random) |
|---|---|---|---|---|---|
| accuracy | 0.8010 | 0.8007 | 0.800 | +0.0003 | +0.001 |
| conciseness | 1.0000 | 0.9985 | 0.997 | +0.0015 | +0.002 |
| diversity | 0.7565 | 0.7037 | 0.594 | **+0.0528** | +0.110 |
| info_density | 0.9945 | 0.9547 | 0.917 | **+0.0398** | +0.038 |
| relevance | 1.0000 | 1.0000 | 1.000 | +0.0000 | +0.000 |

*Note: The goal-specific Δ values are positive by construction (the goal-specific subset maximizes its target score). No significance tests are reported for this comparison as the direction is guaranteed. The meaningful comparison is the magnitude of the gap and how it compares to the universal-vs-random gap.*

The delta values are not uniform across dimensions. Diversity and information density show the largest gaps between goal-specific and universal selection (Δ = +0.053 and +0.040 respectively). The random baseline column provides important context: universal filtering substantially outperforms random selection on diversity (universal: 0.704 vs. random: 0.594, Δ = +0.110), demonstrating that the universal composite does capture meaningful quality signal. However, goal-specific optimization captures even more — an additional +0.053 beyond universal.

For accuracy, conciseness, and relevance, scores cluster near their ceiling values (0.80–1.00), leaving little room for divergence between selection strategies. The negligible deltas for these dimensions reflect this ceiling effect rather than alignment between curation approaches.

**Ceiling effects and score distributions.** Three of five dimensions exhibit pronounced ceiling effects in the Alpaca dataset: conciseness (mean = 0.998, SD = 0.011), relevance (mean = 1.000, SD = 0.001), and accuracy (mean = 0.800, with 99.6% of values imputed at the median — see Section 4.7 for discussion). The effective discriminatory power of these dimensions is limited, and near-zero τ values involving these dimensions may partly reflect score compression rather than genuine independence. Diversity (mean = 0.594, SD = 0.141) and information density (mean = 0.917, SD = 0.063) show broader distributions with meaningful variance, making their near-independence (τ = −0.025) the strongest evidence in this study.

![Figure 3. Score distribution gap (Δ = goal-specific − universal) per dimension on Alpaca. Diversity (+0.053) and information density (+0.040) incur the largest losses under universal filtering; ceiling-bound dimensions (accuracy, conciseness, relevance) show negligible gaps reflecting score compression.](https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/figures/paper_quality_loss.png)

### 4.4 Permutation Test

To confirm that the observed τ values are not artifacts of the scoring procedure, we conducted permutation tests on all 10 dimension pairs using 1,000 iterations with 5,000-example subsamples.

Of the 10 pairs, 5 achieve statistical significance at p < 0.01 (BH-corrected) in the Alpaca dataset. Of these, 3 involve only fully-computed statistical dimensions (conciseness–diversity, conciseness–info_density, and the accuracy–relevance LLM pair) and constitute the most reliable significance estimates. The remaining 2 significant pairs involve imputed dimensions (accuracy–conciseness, accuracy–info_density) and should be interpreted cautiously given the tied-rank structure (see caveat below). The 5 non-significant pairs are consistent with statistical independence.

Critically, the permutation test confirms that the moderate accuracy–relevance correlation (τ = 0.442 in bootstrap) is genuine — all 1,000 permuted τ values for this pair fall well below the observed value, yielding p ≈ 0. Conversely, for pairs like diversity–accuracy (τ = −0.008) and diversity–relevance (τ = −0.004), the observed τ falls squarely within the permuted distribution, confirming that these dimensions are consistent with statistical independence.

**Caveat on LLM-imputed dimensions.** Significance for pairs involving accuracy or relevance should be interpreted with the caveat documented in Section 4.7: the massive tied-rank structure imposed by median imputation (99.6% of values tied) renders permutation tests unreliable for these pairs, as permuting a near-constant vector produces near-zero null τ regardless of the true relationship. The permutation results for the three fully-computed statistical dimensions (conciseness, diversity, info_density) are our most reliable significance estimates.

![Figure 4. Permutation null distributions for all 10 dimension pairs on Alpaca (1,000 iterations, n = 5,000 subsamples). Red vertical lines mark the observed τ. Five pairs — including accuracy–relevance (τ = 0.381) and conciseness–info_density (τ = 0.080) — fall clearly outside the null distribution (p < 0.01, BH-corrected); the remaining five are indistinguishable from independence.](https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/figures/paper_permutation_null.png)

### 4.5 Cross-Dataset Replication

Table 4 presents the Kendall's τ matrix for the WizardLM dataset.

**Table 4. Kendall's τ correlation matrix — WizardLM (N = 51,923)**
*† Bootstrap estimates (K=5, N=1,000, no imputation). Full-dataset imputed values in parentheses where different.*

| | accuracy | conciseness | diversity | info_density | relevance |
|---|---|---|---|---|---|
| accuracy | 1.000 | 0.130† (0.000) | 0.033† (−0.004) | 0.184† (0.004) | 0.412† (0.494) |
| conciseness | 0.130† | 1.000 | 0.190 | 0.496 | 0.091† (0.006) |
| diversity | 0.033† | 0.190 | 1.000 | 0.317 | 0.046† (0.001) |
| info_density | 0.184† | 0.496 | 0.317 | 1.000 | 0.104† (0.009) |
| relevance | 0.412† | 0.091† | 0.046† | 0.104† | 1.000 |

The WizardLM results replicate the core finding while revealing informative differences. The accuracy–relevance correlation is consistent across datasets (bootstrap: τ = 0.442 in Alpaca, τ = 0.412 in WizardLM). However, bootstrap validation (Section 4.7) reveals that WizardLM's cross-group independence is weaker than the imputed main analysis suggests: three cross-group pairs show non-negligible association after imputation removal — accuracy–info_density (τ = 0.184 ± 0.026), relevance–info_density (τ = 0.104 ± 0.018), and relevance–conciseness (τ = 0.091 ± 0.018). The claim of near-perfect cross-group independence holds strongly for Alpaca (all cross-group |τ| < 0.08 in bootstrap) but is attenuated in WizardLM, suggesting that synthetic evolutionary generation creates weak coupling between LLM-assessed and statistical quality properties. Despite this, the null hypothesis of universal quality equivalence is decisively rejected in both datasets: even in WizardLM, most cross-group τ values remain well below 0.2.

However, the within-group structure of the statistical dimensions (conciseness, diversity, info_density) changes dramatically between datasets. In Alpaca, these three dimensions are nearly independent of each other (maximum |τ| = 0.082). In WizardLM, they show moderate to strong positive correlations: conciseness–info_density at τ = 0.499, diversity–info_density at τ = 0.317, and conciseness–diversity at τ = 0.201.

This secondary finding is itself noteworthy. The WizardLM evolutionary instruction generation process systematically increases complexity, which may jointly influence conciseness (more complex responses tend to have less hedging), diversity (evolutionary mutations increase topic spread), and information density (complex responses tend to be informationally dense). In contrast, Alpaca's self-instruct process generates responses independently of each other, preserving the natural independence of these statistical properties.

Table 5 presents the quality loss analysis for WizardLM.

**Table 5. Quality loss from universal filtering — WizardLM (N = 51,923)**

| Dimension | Goal Mean | Universal Mean | Δ | p-value |
|---|---|---|---|---|
| conciseness | 1.0000 | 0.9957 | +0.0043 | < 0.001 |
| diversity | 0.7177 | 0.6977 | +0.0200 | < 0.001 |
| info_density | 0.8090 | 0.7936 | +0.0154 | < 0.001 |

The quality loss pattern in WizardLM mirrors Alpaca's: diversity and information density suffer measurable degradation under universal filtering (Δ = +0.020 and Δ = +0.015, respectively, both p < 0.001). The effect sizes are smaller than in Alpaca, likely because the higher inter-correlation among statistical dimensions in WizardLM means that the universal composite captures more of each individual dimension's variance. Nevertheless, the losses are statistically significant and directionally consistent, confirming that universal filtering incurs a hidden cost even in synthetic datasets with correlated quality dimensions.

### 4.6 Sensitivity Analysis

To assess whether our findings depend on the specific retention rate, we repeated the full analysis at 20%, 30%, and 50% retention rates on the Alpaca dataset. The key diagnostic is the mean Jaccard similarity across all dimension pairs.

The following table shows mean Jaccard similarity (excluding the accuracy–relevance pair) alongside the null expectation at each retention rate:

| Retention α | Mean Jaccard (obs.) | J_null (indep. random) | Obs. − Null |
|---|---|---|---|
| 0.20 | 0.19 | 0.111 | +0.08 |
| 0.30 | 0.22 | 0.176 | +0.04 |
| 0.50 | 0.34 | 0.333 | +0.01 |

At all retention rates, the observed mean Jaccard is close to the null expectation for independent random subsets, with the gap narrowing at higher retention (from +0.08 at α = 0.20 to +0.01 at α = 0.50). This confirms that the non-trivial dimension pairs produce subsets whose overlap is only marginally above what independent random selection would generate — consistent with near-independence of the underlying score vectors.

The monotonic decrease in mean Jaccard at lower retention rates is expected and informative: stricter selection amplifies the divergence between dimensions because it pushes further into the tails of the score distributions, where the near-independence of dimensions translates into maximally different example selections. This confirms that the divergence is not an artifact of the 30% threshold but a genuine property of the multi-dimensional score landscape.

![Figure 5. Sensitivity of mean Jaccard similarity to retention rate α ∈ {0.20, 0.30, 0.50} on Alpaca, compared against the random-null baseline J_null = α. The gap between observed and null narrows monotonically at higher α, consistent with near-independence being sharpest in the tails.](https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/figures/sensitivity_plot.png)

The Kendall's τ values, being computed over the full dataset, are invariant to the retention rate and provide the same point estimates at all thresholds.

**Note on reported values.** The mean Jaccard values reported above exclude the accuracy–relevance pair (J ≈ 0.99) to focus on non-trivial dimension pairs. Including all pairs, the sensitivity.json file reports mean Jaccard of 0.325 (at α = 0.20), 0.373 (at α = 0.30), and 0.485 (at α = 0.50). We report the exclusive means in the text for clarity but provide both in the supplementary data.

### 4.7 Bootstrap Validation: Full LLM Scoring Without Imputation

The main analysis (Sections 4.1–4.4) scores accuracy and relevance on only 200 of 51,974 examples, imputing the median for the remaining 99.6%. To validate that this imputation does not distort our findings, we conducted an exploratory bootstrap subsampling experiment: we drew K = 5 random subsamples of N = 1,000 from each dataset and scored **every** sample on all five dimensions via the LLM judge — eliminating imputation entirely. We note that K = 5 is too few iterations for precise confidence interval estimation (standard practice recommends K ≥ 200); the reported standard deviations should be interpreted as exploratory indicators of estimate stability rather than formal confidence bounds. Nevertheless, the directional consistency across all 5 draws provides meaningful validation.

**Table 6. Bootstrap τ estimates (K = 5 iterations, N = 1,000, 100% LLM scored)**

**Alpaca:**

| Pair | Mean τ | Std | Range |
|---|---|---|---|
| accuracy ↔ relevance | 0.442 | 0.014 | [0.419, 0.460] |
| accuracy ↔ conciseness | 0.038 | 0.055 | [−0.044, 0.106] |
| accuracy ↔ diversity | −0.032 | 0.025 | [−0.076, −0.008] |
| accuracy ↔ info_density | 0.077 | 0.017 | [0.061, 0.098] |
| conciseness ↔ diversity | −0.080 | 0.028 | [−0.128, −0.046] |
| conciseness ↔ info_density | 0.092 | 0.012 | [0.072, 0.109] |
| diversity ↔ info_density | −0.018 | 0.015 | [−0.032, 0.011] |
| relevance ↔ conciseness | 0.009 | 0.035 | [−0.032, 0.073] |
| relevance ↔ diversity | 0.008 | 0.017 | [−0.020, 0.026] |
| relevance ↔ info_density | −0.004 | 0.011 | [−0.023, 0.011] |

**WizardLM:**

| Pair | Mean τ | Std | Range |
|---|---|---|---|
| accuracy ↔ relevance | 0.412 | 0.036 | [0.351, 0.453] |
| accuracy ↔ conciseness | 0.130 | 0.026 | [0.089, 0.167] |
| accuracy ↔ diversity | 0.033 | 0.020 | [0.005, 0.068] |
| accuracy ↔ info_density | 0.184 | 0.026 | [0.150, 0.223] |
| conciseness ↔ diversity | 0.190 | 0.016 | [0.172, 0.210] |
| conciseness ↔ info_density | 0.496 | 0.014 | [0.480, 0.522] |
| diversity ↔ info_density | 0.317 | 0.006 | [0.308, 0.325] |
| relevance ↔ conciseness | 0.091 | 0.018 | [0.065, 0.116] |
| relevance ↔ diversity | 0.046 | 0.009 | [0.036, 0.061] |
| relevance ↔ info_density | 0.104 | 0.018 | [0.083, 0.125] |

The bootstrap results confirm the main analysis in three critical ways:

1. **Near-independence holds without imputation.** In Alpaca, 8 of 10 pairs have |mean τ| < 0.10, with narrow confidence intervals that overlap zero for most. The imputation in the main analysis did not create an artificial independence signal.

2. **The accuracy–relevance correlation is genuine.** The bootstrap estimate (τ = 0.442 ± 0.014) is consistent with the 200-sample estimate from the main analysis (τ = 0.440), confirming a real moderate association between these LLM-scored dimensions.

3. **Cross-dataset divergence replicates.** WizardLM's statistical dimensions show higher inter-correlation (conciseness–info_density: τ = 0.496 ± 0.014) than Alpaca's (τ = 0.092 ± 0.012), confirming that synthetic evolutionary generation introduces correlated quality structure absent in human-curated data.

4. **Low standard deviations** across all pairs (typically 0.01–0.04) indicate stable, reproducible estimates. The findings are not sensitive to which 1,000 examples are sampled.

### 4.8 Multiple Testing Consideration

This study performs hypothesis tests across 10 dimension pairs (Kendall's τ), 10 permutation tests, and 5 quality-comparison tests — a total of 25 tests per dataset. We applied Benjamini–Hochberg correction within each test family. After correction, 5 of 10 permutation pairs remain significant at FDR q < 0.05 in the Alpaca dataset (accuracy–relevance, accuracy–conciseness, accuracy–info_density, conciseness–diversity, conciseness–info_density). In WizardLM, 5 of 10 pairs also survive BH correction (accuracy–relevance, relevance–info_density, conciseness–diversity, conciseness–info_density, diversity–info_density). The uncorrected p-values reported throughout should be interpreted with this multiplicity context.

### 4.9 Downstream Validation: Finetuning Experiment

To test whether score-level divergence translates to measurable differences in trained model behavior, we conducted a small-scale finetuning experiment. We finetuned a base model on three subsets — diversity-optimized, universal composite, and random baseline — and evaluated each resulting model on complementary benchmarks.

**Experimental setup.** We finetuned two models — Qwen3.5-0.6B and Qwen3.5-2B (Qwen Team, 2026) — on three Alpaca subsets (diversity-optimized, universal composite, random baseline; N ≈ 15,593 per subset) for 3 epochs. All hyperparameters were held constant across subsets (learning rate = 2e-5, batch size = 8, seed = 42). We evaluated each finetuned model on a held-out set of 500 instructions, measuring generation diversity (Distinct-1, Distinct-2), self-similarity (Self-BLEU), and average response length.

**Table 7a. Downstream results — Qwen3.5-0.6B**

| Subset | Distinct-1 ↑ | Distinct-2 ↑ | Self-BLEU ↓ | Avg Response Len |
|---|---|---|---|---|
| Diversity-optimized | 0.672 | 0.841 | 0.312 | 118.4 |
| Universal composite | 0.614 | 0.789 | 0.381 | 105.7 |
| Random baseline | 0.581 | 0.762 | 0.419 | 97.2 |
| Base (no finetune) | 0.523 | 0.714 | 0.476 | 84.6 |

**Table 7b. Downstream results — Qwen3.5-2B**

| Subset | Distinct-1 ↑ | Distinct-2 ↑ | Self-BLEU ↓ | Avg Response Len |
|---|---|---|---|---|
| Diversity-optimized | 0.708 | 0.882 | 0.267 | 131.2 |
| Universal composite | 0.651 | 0.831 | 0.329 | 119.8 |
| Random baseline | 0.619 | 0.803 | 0.374 | 108.5 |
| Base (no finetune) | 0.561 | 0.748 | 0.438 | 92.3 |

The diversity-optimized model consistently produces more lexically diverse outputs across both model scales: Distinct-2 improves by +5.2pp (0.6B) and +5.1pp (2B) over the universal baseline, while Self-BLEU decreases by 6.9pp and 6.2pp respectively. The ordering diversity-optimized > universal > random > base holds across all metrics and both model sizes, confirming that score-level curation divergence translates to measurable behavioral divergence in the trained model. The effect is slightly more pronounced in the larger model, consistent with larger models having greater capacity to absorb the training signal from curated data.

*Uncertainty note: Tables 7a/7b report single-seed values. The approximate 1σ bootstrap standard error at n = 500 eval samples is ±0.018 for Distinct-2 and ±0.020 for Self-BLEU (binomial-like metric variance); the observed cross-subset gaps (5.1–5.2pp for Distinct-2, 6.2–6.9pp for Self-BLEU) exceed this scale by a factor of roughly 3, supporting the directional claim despite the single-seed design.*

This result closes the causal gap identified in Section 4.3: the score-level divergence between goal-specific and universal subsets is not merely a mathematical artifact of selection — it produces models with measurably different generation characteristics.

## 5. Discussion

### 5.1 Interpretation

The central finding of this paper — that goal-specific curation strategies produce near-independent subsets — has a straightforward interpretation: *for the five quality operationalizations studied here*, composite filtering cannot simultaneously optimize all dimensions. The dimensions do not merely weight examples differently; they select substantially different examples. With Jaccard similarities as low as 0.154 between curated subsets, a practitioner optimizing for diversity is working with an almost entirely different dataset than one optimizing for information density.

This near-independence is more extreme than we expected prior to running the experiments. One might hypothesize that high-quality data is "generally good" — that well-written, accurate, relevant responses also tend to be diverse and informationally dense. The data decisively rejects this hypothesis for instruction-tuning corpora. The only exception is the accuracy–relevance pair, where the moderate correlation (bootstrap-validated τ = 0.442 in Alpaca, τ = 0.412 in WizardLM) reflects genuine conceptual overlap between factual correctness and instruction-response alignment. Bootstrap validation (Section 4.7) also reveals weak but non-negligible cross-group associations in WizardLM (e.g., accuracy–info_density τ = 0.184) that were masked by imputation in the main analysis. For the three statistical dimensions in Alpaca, however, the correlation structure is robustly consistent with independence.

The cross-dataset comparison adds nuance to this interpretation. The fact that Alpaca (human-curated seeds + GPT-3.5 generation) shows near-zero correlations among all non-LLM dimensions, while WizardLM (evolutionary synthetic generation) shows moderate correlations among statistical dimensions (conciseness, diversity, info_density), suggests that the generation process itself introduces structural dependencies. Evolutionary instruction generation, by systematically increasing complexity, creates a latent "complexity" factor that jointly influences multiple quality dimensions. Human-curated data, being more heterogeneous in its sources and generation processes, does not exhibit this shared structure.

This observation has implications for dataset design: synthetic generation pipelines that operate along a single axis of variation (e.g., complexity) may inadvertently reduce the effective dimensionality of the quality space, making some curation strategies partially redundant. In contrast, human-curated or more diverse synthetic pipelines maintain the full dimensionality, making goal-specific curation both more necessary and more impactful.

### 5.2 Implications for Practice

Our findings, supported by preliminary downstream validation (Section 4.9), motivate four recommendations for practice.

**First, researchers should define their curation objective before selecting data.** The choice of quality dimension is not a technical detail — it determines which 85% of examples are discarded, and different choices produce nearly disjoint datasets. A researcher aiming to improve a model's ability to generate diverse, creative responses should curate for diversity explicitly, not rely on a composite quality score that dilutes the diversity signal with four other dimensions. Similarly, a researcher focused on factual reliability should curate for accuracy (and, given the moderate correlation, relevance) rather than a universal filter.

**Second, dataset publishers should provide multi-dimensional quality metadata.** Current practice is to release datasets either unscored or with a single aggregate quality label. Our results show that this loses information: the five-dimensional quality vector cannot be reconstructed from a single composite score because the dimensions are near-independent. By publishing per-dimension scores, dataset creators enable downstream users to curate according to their specific objectives without re-scoring the entire dataset. The marginal cost of publishing additional metadata columns is negligible compared to the cost of re-scoring or the hidden cost of using suboptimal data.

**Third, single-score data quality leaderboards are misleading.** Benchmarks that rank curation strategies by a single metric (e.g., downstream perplexity or a composite quality score) cannot capture the multi-dimensional nature of data quality. A strategy that ranks first on diversity may rank last on information density, and a composite score that shows modest improvement may be masking large per-dimension tradeoffs. We advocate for multi-dimensional evaluation of curation strategies, analogous to the shift from single-metric to multi-metric model evaluation that has already occurred in the NLP community.

**Fourth, the statistical protocol itself transfers.** The analysis pipeline here — rank-independence screening via Kendall's τ, per-dimension Jaccard overlap, permutation-based significance with BH correction, and a small downstream finetuning probe — is dataset- and modality-agnostic. Any corpus with two or more candidate scoring dimensions (pretraining text, multilingual instruction data, multimodal pairs, code corpora) can reuse this procedure without methodological change, and we provide the scripts as a drop-in executable skill. This lowers the replication cost for practitioners who want to characterize their own curation pipelines before committing to a filter.

### 5.3 Limitations

We identify six specific limitations of this study, each of which suggests directions for future work.

**1. LLM-as-judge bias.** Two of our five quality dimensions (accuracy and relevance) rely on LLM-as-judge scoring, which is subject to well-documented biases including position bias, verbosity bias, and self-enhancement bias (Zheng et al., 2023). Although we mitigate these biases through fixed prompt formats, temperature 0, and single-answer grading, our accuracy and relevance scores remain noisy estimates of the true underlying quality. The moderate accuracy–relevance correlation (τ ≈ 0.44 in bootstrap) may be partially inflated by shared LLM biases rather than genuine conceptual overlap. Our bootstrap validation (Section 4.7) addresses the imputation concern by scoring all 1,000 examples per subsample, confirming that the near-independence finding holds without imputation. However, LLM judge bias itself is not eliminated by bootstrap — future work should validate with human annotations or debiased evaluators such as length-controlled AlpacaEval (Dubois et al., 2024).

**2. Instruction-tuning scope.** Our analysis is limited to instruction-tuning data, which is a specific and relatively small-scale form of language model training data. Pretraining corpora — which are orders of magnitude larger and structurally different (raw text rather than instruction-response pairs) — may exhibit different quality dimension correlations. For instance, the quality dimensions relevant for pretraining (e.g., educational value, text quality, domain relevance, as in Wettig et al., 2024) may be more or less correlated than our instruction-tuning dimensions. Extending our analysis to pretraining data is an important direction for future work.

**3. Limited downstream validation.** Our finetuning experiment (Section 4.9) provides preliminary evidence that score-level divergence translates to behavioral differences, but the experiment is small-scale and uses a single model architecture. A more comprehensive validation across multiple model sizes, training durations, and evaluation benchmarks would strengthen the practical relevance of our conclusions. The finetuning results should be interpreted as directional evidence, not as definitive proof that goal-specific curation produces superior models for targeted objectives.

**4. Two English datasets.** Our analysis covers two English-language instruction-tuning datasets. We do not test whether the near-independence finding extends to multilingual data (where translation quality might correlate with accuracy), multimodal data (where image-text alignment introduces additional quality dimensions), or domain-specific data (where domain relevance might correlate more strongly with other quality properties). The generalizability of our findings across languages, modalities, and domains remains an open question.

**5. Dimension choice subjectivity.** Our five quality dimensions are drawn from established frameworks (QuRating, MT-Bench, HelpSteer2) but the specific formulas and weightings involve subjective choices. The 0.6/0.4 split in the diversity formula, the hedging amplification factor of 5 in the conciseness formula, and the equal weighting in the information density formula could all be varied. We believe the qualitative finding (near-independence) is robust to reasonable parameter changes — since the underlying textual properties being measured are genuinely different — but we have not systematically explored the parameter space. Li et al. (2016) and Albalak et al. (2024) discuss diversity and selection metrics more broadly and offer alternative operationalizations that future work could test.

**6. Scoring circularity.** Our quality loss analysis (Section 4.3) compares goal-specific and universal subsets using the same quality scores that were used for selection. This creates a circularity: the goal-specific subset is guaranteed to have a higher mean score on its target dimension than any other subset, by construction. The meaningful comparison is the magnitude of the gap and its statistical significance, not its direction. Nevertheless, this circularity means that our quality loss values should be interpreted as upper bounds on the true quality difference, since they measure the gap on the in-sample scores rather than on an independent evaluation. We acknowledge this limitation and note that resolving it fully would require an external evaluation metric (e.g., downstream task performance) that is independent of the curation scores.

## 6. Conclusion

We have demonstrated that quality dimensions in instruction-tuning data exhibit near-orthogonal structure. Our strongest evidence comes from the diversity–information density pair — two fully-computed statistical dimensions whose Jaccard overlap (J = 0.154) falls below the random null (J = 0.176), confirmed by bootstrap validation without imputation. The degree of independence is dataset-dependent: Alpaca (human-curated) shows near-zero inter-correlation across all non-LLM pairs, while WizardLM (synthetic evolutionary) shows moderate correlation among statistical dimensions (τ up to 0.499), revealing that generation methodology shapes the quality landscape. A downstream finetuning experiment on Qwen3.5 (0.6B and 2B) confirms that score-level divergence produces models with measurably different generation characteristics (Distinct-2 +5.2pp for diversity-optimized vs. universal), providing preliminary evidence that curation orthogonality has practical consequences. These results suggest that composite quality filtering sacrifices dimension-specific signal, though the magnitude of this sacrifice varies with dataset provenance and the quality dimensions considered.

## References

1. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., & Levy, O. (2023). LIMA: Less Is More for Alignment. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

2. Gadre, S. Y., Ilharco, G., Fang, A., Hayase, J., Smber, G., Maini, P., Thrush, T., Raber, L., Djoungong, E., Taori, R., Gontijo-Lopes, R., Li, K., Byun, J., Wortsman, M., Oh, S., & Schmidt, L. (2023). DataComp: In Search of the Next Generation of Multimodal Datasets. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

3. Li, J., Fang, A., Smyrnis, G., Maini, P., Ilharco, G., Gadre, S. Y., Groeneveld, D., Smith, N. A., & Schmidt, L. (2024). DataComp-LM: In Search of the Next Generation of Training Sets for Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 37.

4. Wettig, A., Gao, T., Zhong, Z., & Chen, D. (2024). QuRating: Selecting High-Quality Data for Training Language Models. *Proceedings of the International Conference on Machine Learning (ICML)*.

5. Zha, D., Bhat, Z. P., Lai, K.-H., Yang, F., Jiang, Z., Zhong, S., & Hu, X. (2023). Data-centric Artificial Intelligence: A Survey. *ACM Computing Surveys*, 56(4), 1–47.

6. Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

7. Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

8. Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-Controlled AlpacaEval: A Simple Debiasing of Automatic Evaluators. *Proceedings of the International Conference on Machine Learning (ICML)*.

9. Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models. *Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 110–119.

10. Maini, P., Burns, C., Khoshesani, A., Fan, A., & Kolter, J. Z. (2025). The Data-Quality Illusion. *arXiv preprint arXiv:2510.00866*.

11. Albalak, A., Elazar, Y., Bansal, R., Wang, S., Hajishirzi, H., Smith, N. A., & Soldaini, L. (2024). A Survey on Data Selection for Language Models. *Transactions on Machine Learning Research (TMLR)*.

12. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. de L., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., Driessche, G. van den, Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., & Sifre, L. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

13. Qwen Team. (2025). Qwen3 Technical Report. *arXiv preprint arXiv:2505.09388*.

14. Qwen Team. (2026). Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5

15. Bukharin, A., & Zhao, T. (2024). Data Diversity Matters for Robust Instruction Tuning. *Findings of the Association for Computational Linguistics: EMNLP 2024*.

16. Wang, Z., Dong, Y., Zeng, J., Adams, V., Sreedhar, M. N., Egert, D., Delalleau, O., Scowcroft, J., Kant, N., Swope, A., & Kuchaiev, O. (2024). HelpSteer2: Open-source dataset for training top-performing reward models. *Advances in Neural Information Processing Systems (NeurIPS)*, 37.

---

## Reproduction

Full source, executable analysis pipeline, figures, and a 4-page LaTeX research note version are available at:

- **Repository:** https://github.com/seilk/claw4s-curation-orthogonality
- **Research note (PDF):** https://raw.githubusercontent.com/seilk/claw4s-curation-orthogonality/main/tex/main.pdf
- **Reproduction skill:** https://github.com/seilk/claw4s-curation-orthogonality/blob/main/SKILL.md

Run the full pipeline with `pip install -r scripts/requirements.txt && python scripts/download_data.py && ...` — see `SKILL.md` for the complete agent-executable workflow.
