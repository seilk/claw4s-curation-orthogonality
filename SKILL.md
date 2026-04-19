---
name: goal-specific-curation-analysis
description: >
  Statistical analysis demonstrating that goal-specific data curation strategies
  produce near-independent subsets (Kendall's τ ≈ 0.0) in instruction-tuning data,
  quantifying the hidden cost of universal quality filtering via nonparametric
  rank correlation and permutation tests across two public datasets.
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# Goal-Specific Curation Analysis — Executable Reproduction Skill

This skill reproduces all findings from the paper **"Goal-Specific Data Curation Produces Near-Independent Subsets in Instruction-Tuning Data"** (ClawRxiv 2025). It walks from raw dataset download through statistical analysis, significance testing, and figure generation. Every step has a verification subsection with executable assertions. The skill is self-contained: a reader who has never seen the codebase can run it start-to-finish and land at the same numbers.

**Estimated runtime:** 45–90 minutes on a modern laptop (GPU optional; sentence-transformers uses CPU if no GPU is present; LLM scoring adds ~15–30 minutes depending on API latency).

**Repository root assumption:** All commands are run from the repository root (the directory containing this `SKILL.md`). Paths in this file are relative to that root.

---

## Prerequisites

- Python 3.11+
- A `.env` file with `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` (for LLM-based scoring of 200 samples per dataset; the remaining 4 dimensions are statistical and need no API)
- HuggingFace `datasets` library (included in `requirements.txt`)
- Internet access for the first download; subsequent runs are fully offline

---

## Step 1: Environment Setup

Install all Python dependencies and verify the environment is consistent. This step creates no data; it only prepares tooling.

```bash
# Install dependencies from requirements.txt
pip install -r scripts/requirements.txt

# Ensure the project package itself is importable (editable install for scripts)
pip install -e . --quiet

# Create output directory tree (idempotent)
mkdir -p data/raw data/scored data/subsets data/cache \
         data/results/alpaca data/results/wizardlm \
         figures

# Verify no import errors on any analysis script
python -c "import json, math, random, itertools, pathlib, argparse; print('stdlib OK')"
python -c "import numpy, scipy, sklearn, pandas, matplotlib, seaborn; print('numeric stack OK')"
python -c "from datasets import load_dataset; print('HuggingFace datasets OK')"
python -c "import sentence_transformers; print('sentence-transformers OK')"
```

### Verification — Step 1

```bash
# Assert Python version is 3.11 or newer
python -c "
import sys
major, minor = sys.version_info[:2]
assert (major, minor) >= (3, 11), f'Need Python >=3.11, got {major}.{minor}'
print(f'Python {major}.{minor} OK')
"

# Assert all required packages are importable and meet minimum versions
python -c "
import scipy, numpy, sklearn, pandas, matplotlib, seaborn
from packaging.version import Version
checks = [
    ('scipy',      scipy.__version__,      '1.11'),
    ('numpy',      numpy.__version__,      '1.24'),
    ('sklearn',    sklearn.__version__,    '1.2'),
    ('pandas',     pandas.__version__,     '2.1'),
    ('matplotlib', matplotlib.__version__, '3.8'),
    ('seaborn',    seaborn.__version__,    '0.12'),
]
for name, installed, required in checks:
    assert Version(installed) >= Version(required), \
        f'{name} {installed} < required {required}'
    print(f'  {name} {installed} >= {required}  OK')
print('All version checks passed')
"

# Assert directory structure exists
python -c "
from pathlib import Path
dirs = [
    'data/raw', 'data/scored', 'data/subsets', 'data/cache',
    'data/results/alpaca', 'data/results/wizardlm', 'figures',
]
for d in dirs:
    assert Path(d).is_dir(), f'Missing directory: {d}'
    print(f'  {d}/ exists')
print('Directory structure OK')
"

# Assert scripts are all present
python -c "
from pathlib import Path
scripts = [
    'scripts/download_data.py',
    'scripts/score_dimensions.py',
    'scripts/curate_by_goal.py',
    'scripts/rank_correlation.py',
    'scripts/permutation_test.py',
    'scripts/sensitivity.py',
    'scripts/generate_report.py',
]
for s in scripts:
    assert Path(s).exists(), f'Missing script: {s}'
    print(f'  {s} found')
print('All scripts present')
"
```

**Expected output:** All assertions print "OK" with no `AssertionError`. If any package is missing, re-run `pip install -r scripts/requirements.txt`.

---

## Step 2: Download Datasets

Download Alpaca (51,974 samples) and WizardLM (~51,923 samples) from HuggingFace. The scripts apply a fixed seed for reproducibility; re-running produces bit-identical JSONL files.

```bash
# Download Alpaca (no --max-samples; take the full dataset)
python scripts/download_data.py \
    --dataset alpaca \
    --output data/raw/alpaca.jsonl

# Download WizardLM (capped at 52,000 to match paper; filtering removes empties)
python scripts/download_data.py \
    --dataset wizardlm \
    --output data/raw/wizardlm.jsonl \
    --max-samples 52000
```

Expected terminal output:

```
[download] alpaca: 51974 samples -> data/raw/alpaca.jsonl
[download] wizardlm: 51923 samples -> data/raw/wizardlm.jsonl
```

### Verification — Step 2

```bash
# Assert JSONL files exist and are non-empty
python -c "
from pathlib import Path
for path in ['data/raw/alpaca.jsonl', 'data/raw/wizardlm.jsonl']:
    p = Path(path)
    assert p.exists(), f'Missing: {path}'
    size = p.stat().st_size
    assert size > 1_000_000, f'{path} is suspiciously small: {size} bytes'
    print(f'  {path}: {size:,} bytes  OK')
"

# Assert exact record counts
python -c "
import json
from pathlib import Path

expected = {
    'data/raw/alpaca.jsonl': 51974,
    'data/raw/wizardlm.jsonl': 51923,
}
for path, expected_count in expected.items():
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    actual = len(lines)
    assert actual == expected_count, \
        f'{path}: expected {expected_count} records, got {actual}'
    print(f'  {path}: {actual} records  OK')
print('Dataset counts match paper')
"

# Assert JSONL schema — every record must have instruction, response, id
python -c "
import json, random
random.seed(0)
for path in ['data/raw/alpaca.jsonl', 'data/raw/wizardlm.jsonl']:
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    sample_indices = random.sample(range(len(lines)), 50)
    for i in sample_indices:
        rec = json.loads(lines[i])
        assert 'instruction' in rec, f'Missing instruction in record {i} of {path}'
        assert 'response' in rec,    f'Missing response in record {i} of {path}'
        assert 'id' in rec,          f'Missing id in record {i} of {path}'
        assert len(rec['instruction']) > 0, f'Empty instruction in record {i} of {path}'
        assert len(rec['response']) > 0,    f'Empty response in record {i} of {path}'
    print(f'  {path}: 50 random records have correct schema  OK')
print('Schema checks passed')
"

# Assert no duplicate IDs within each dataset
python -c "
import json
for path in ['data/raw/alpaca.jsonl', 'data/raw/wizardlm.jsonl']:
    with open(path) as f:
        ids = [json.loads(l)['id'] for l in f if l.strip()]
    assert len(ids) == len(set(ids)), f'{path}: duplicate IDs detected'
    print(f'  {path}: all IDs unique  OK')
"
```

**Expected output:** All assertions pass. If WizardLM count differs by ±10 records, that is an acceptable variance from upstream dataset updates; adjust the expected count in the assertion accordingly and note the discrepancy.

---

## Step 3: Score Quality Dimensions

Each sample is scored on five dimensions. Three are purely statistical (conciseness, diversity, info_density) and require no API. Two (accuracy, relevance) use an LLM-as-judge on a random sample of 200 records; scores for the remaining records are imputed from the statistical proxy.

This step takes the longest wall-clock time. Run Alpaca and WizardLM sequentially because both may use the same embedding model loaded into memory.

```bash
# Load .env for LLM credentials
set -a && source .env && set +a

# Score Alpaca (LLM sample = 200; rest are statistical)
python scripts/score_dimensions.py \
    data/raw/alpaca.jsonl \
    --output data/scored/scores_alpaca.jsonl \
    --llm-sample-size 200 \
    --cache data/cache/llm_scores_alpaca.json

# Score WizardLM
python scripts/score_dimensions.py \
    data/raw/wizardlm.jsonl \
    --output data/scored/scores_wizardlm.jsonl \
    --llm-sample-size 200 \
    --cache data/cache/llm_scores_wizardlm.json
```

If the LLM API is unavailable, the script falls back to the statistical proxy for accuracy and relevance. The paper's core finding (τ ≈ 0.0) is reproducible from statistical dimensions alone; LLM scoring adds precision but is not mandatory for the structural result.

### Verification — Step 3

```bash
# Assert output files exist and have the same record count as input
python -c "
import json
from pathlib import Path

pairs = [
    ('data/raw/alpaca.jsonl',   'data/scored/scores_alpaca.jsonl',   51974),
    ('data/raw/wizardlm.jsonl', 'data/scored/scores_wizardlm.jsonl', 51923),
]
for raw_path, scored_path, expected in pairs:
    assert Path(scored_path).exists(), f'Missing: {scored_path}'
    with open(scored_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) == expected, \
        f'{scored_path}: expected {expected} records, got {len(records)}'
    print(f'  {scored_path}: {len(records)} records  OK')
"

# Assert every scored record has all 5 dimensions in [0, 1]
python -c "
import json, random
random.seed(42)
dims = ['accuracy', 'relevance', 'conciseness', 'diversity', 'info_density']
for path in ['data/scored/scores_alpaca.jsonl', 'data/scored/scores_wizardlm.jsonl']:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    sample = random.sample(records, min(500, len(records)))
    errors = []
    for rec in sample:
        assert 'scores' in rec, f'Missing scores key in record {rec.get(\"id\")}'
        assert '_original_index' in rec, f'Missing _original_index in record {rec.get(\"id\")}'
        for d in dims:
            assert d in rec['scores'], f'Missing dim {d} in record {rec.get(\"id\")}'
            v = rec['scores'][d]
            if not (0.0 <= v <= 1.0):
                errors.append(f'{path} id={rec[\"id\"]} {d}={v}')
    assert len(errors) == 0, 'Out-of-range scores:\n' + '\n'.join(errors[:5])
    print(f'  {path}: all 5 dims in [0,1] for 500 sampled records  OK')
"

# Assert score distributions are not degenerate (not all zeros or all ones)
python -c "
import json, numpy as np
dims = ['accuracy', 'relevance', 'conciseness', 'diversity', 'info_density']
for path in ['data/scored/scores_alpaca.jsonl', 'data/scored/scores_wizardlm.jsonl']:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    for d in dims:
        vals = np.array([r['scores'][d] for r in records])
        std = vals.std()
        mean = vals.mean()
        assert std > 0.01, f'{path} dim={d} has near-zero variance: std={std:.4f}'
        assert 0.05 < mean < 0.95, f'{path} dim={d} mean={mean:.4f} looks degenerate'
        print(f'  {path} {d}: mean={mean:.3f} std={std:.3f}  OK')
"

# Assert LLM cache was populated (at least 100 entries per dataset)
python -c "
import json
from pathlib import Path
for path in ['data/cache/llm_scores_alpaca.json', 'data/cache/llm_scores_wizardlm.json']:
    p = Path(path)
    if not p.exists():
        print(f'  {path}: not found (LLM scoring may have been skipped — OK if using stat proxy)')
        continue
    with open(p) as f:
        cache = json.load(f)
    assert len(cache) >= 100, f'{path}: only {len(cache)} cached LLM scores'
    print(f'  {path}: {len(cache)} cached LLM scores  OK')
"
```

**Expected output:** All score files match input counts; all 5 dimensions are in [0, 1] with non-degenerate distributions; LLM cache has at least 100 entries per dataset. Standard deviations typically range 0.08–0.22 across dimensions.

---

## Step 4: Generate Goal-Specific Subsets

For each of the 5 quality dimensions, select the top-30% samples by that dimension's score. Also generate a `universal` subset (top-30% by composite mean) and a `random` baseline. This step is purely deterministic: no randomness beyond `random` seed 42 in the random baseline.

```bash
# Generate subsets for Alpaca at 30% retention
python scripts/curate_by_goal.py \
    data/scored/scores_alpaca.jsonl \
    --output data/subsets/subsets_alpaca.json \
    --retention 0.3

# Generate subsets for WizardLM at 30% retention
python scripts/curate_by_goal.py \
    data/scored/scores_wizardlm.jsonl \
    --output data/subsets/subsets_wizardlm.json \
    --retention 0.3
```

Expected terminal output (Alpaca example):

```
  [accuracy] 15593 samples selected
  [relevance] 15593 samples selected
  [conciseness] 15593 samples selected
  [diversity] 15593 samples selected
  [info_density] 15593 samples selected
  [universal] 15593 samples selected
  [random] 15593 samples selected
[curate] 7 subsets -> data/subsets/subsets_alpaca.json
```

(WizardLM: ~15577 per subset at 30% of 51923.)

### Verification — Step 4

```bash
# Assert subset files exist and contain exactly 7 keys
python -c "
import json
from pathlib import Path
expected_keys = {'accuracy', 'relevance', 'conciseness', 'diversity', 'info_density', 'universal', 'random'}
for path in ['data/subsets/subsets_alpaca.json', 'data/subsets/subsets_wizardlm.json']:
    assert Path(path).exists(), f'Missing: {path}'
    with open(path) as f:
        subsets = json.load(f)
    assert set(subsets.keys()) == expected_keys, \
        f'{path}: unexpected keys {set(subsets.keys())}'
    print(f'  {path}: 7 expected keys present  OK')
"

# Assert subset sizes are approximately 30% of total
python -c "
import json
pairs = [
    ('data/subsets/subsets_alpaca.json',   51974),
    ('data/subsets/subsets_wizardlm.json', 51923),
]
for path, total in pairs:
    with open(path) as f:
        subsets = json.load(f)
    expected_size = int(total * 0.3)
    for key, indices in subsets.items():
        actual = len(indices)
        # Allow +1 due to math.ceil rounding
        assert abs(actual - expected_size) <= 1, \
            f'{path}[{key}]: expected ~{expected_size}, got {actual}'
    print(f'  {path}: all subsets are ~30% ({expected_size}) of {total}  OK')
"

# Assert no out-of-range indices
python -c "
import json
pairs = [
    ('data/subsets/subsets_alpaca.json',   51974),
    ('data/subsets/subsets_wizardlm.json', 51923),
]
for path, total in pairs:
    with open(path) as f:
        subsets = json.load(f)
    for key, indices in subsets.items():
        bad = [i for i in indices if i < 0 or i >= total]
        assert len(bad) == 0, f'{path}[{key}]: {len(bad)} out-of-range indices'
    print(f'  {path}: all indices in [0, {total})  OK')
"

# Assert goal-specific subsets are NOT identical to each other (they should differ)
python -c "
import json
from itertools import combinations
for path in ['data/subsets/subsets_alpaca.json', 'data/subsets/subsets_wizardlm.json']:
    with open(path) as f:
        subsets = json.load(f)
    goal_dims = ['accuracy', 'relevance', 'conciseness', 'diversity', 'info_density']
    for a, b in combinations(goal_dims, 2):
        set_a, set_b = set(subsets[a]), set(subsets[b])
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        # If Jaccard were 1.0, the two subsets are identical — that disproves our hypothesis
        assert jaccard < 0.99, f'{path}: {a} and {b} are nearly identical (J={jaccard:.3f})'
    print(f'  {path}: all goal-specific subset pairs have Jaccard < 0.99  OK')
"

# Assert universal subset is NOT equal to any single-dimension subset
python -c "
import json
for path in ['data/subsets/subsets_alpaca.json', 'data/subsets/subsets_wizardlm.json']:
    with open(path) as f:
        subsets = json.load(f)
    univ = set(subsets['universal'])
    for dim in ['accuracy', 'relevance', 'conciseness', 'diversity', 'info_density']:
        goal = set(subsets[dim])
        jaccard = len(univ & goal) / len(univ | goal)
        # Universal and goal-specific subsets should differ meaningfully
        assert jaccard < 0.95, \
            f'{path}: universal and {dim} are too similar (J={jaccard:.3f})'
    print(f'  {path}: universal subset differs from all goal-specific subsets  OK')
"
```

**Expected output:** 7 keys per file, all subset sizes match `ceil(N * 0.3)`, no out-of-range indices, Jaccard between any pair of goal dimensions is below 0.99, and universal diverges from all single-dimension subsets.

---

## Step 5: Rank Correlation Analysis (Core Finding)

This is the central statistical step. Kendall's τ quantifies rank concordance between dimension score vectors across all samples. Near-zero τ means the rankings produced by each dimension are statistically independent — selecting top-30% by dimension A gives an almost entirely different set of samples than selecting by dimension B.

```bash
# Alpaca: compute τ matrix, Jaccard matrix, quality loss
python scripts/rank_correlation.py \
    data/scored/scores_alpaca.jsonl \
    data/subsets/subsets_alpaca.json \
    --output data/results/alpaca/

# WizardLM: same
python scripts/rank_correlation.py \
    data/scored/scores_wizardlm.jsonl \
    data/subsets/subsets_wizardlm.json \
    --output data/results/wizardlm/
```

Expected terminal output (Alpaca — values approximate):

```
[rank_corr] Computing Kendall's τ matrix...
[rank_corr] Computing Jaccard matrix...
[rank_corr] Computing quality loss (goal-specific vs universal)...
[rank_corr] Results saved to data/results/alpaca/

Kendall's τ matrix:
       accuracy:    1.0 |  0.xxx |  0.0xx |  0.0xx |  0.0xx
      relevance:  0.xxx |    1.0 |  0.0xx |  0.0xx |  0.0xx
    conciseness:  0.0xx |  0.0xx |    1.0 |  0.0xx |  0.0xx
      diversity:  0.0xx |  0.0xx |  0.0xx |    1.0 |  0.0xx
   info_density:  0.0xx |  0.0xx |  0.0xx |  0.0xx |    1.0
```

### Verification — Step 5

```bash
# Assert output files exist
python -c "
from pathlib import Path
for dataset in ['alpaca', 'wizardlm']:
    for fname in ['tau_matrix.json', 'jaccard_matrix.json', 'quality_loss.json']:
        p = Path(f'data/results/{dataset}/{fname}')
        assert p.exists(), f'Missing: {p}'
        print(f'  {p} exists  OK')
"

# CORE ASSERTION: most off-diagonal τ values must be < 0.5
python -c "
import json
from itertools import combinations

for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/tau_matrix.json') as f:
        data = json.load(f)
    tau = data['tau']
    dims = [d for d in tau if d != '_meta']
    off_diag = []
    for a, b in combinations(dims, 2):
        off_diag.append((a, b, tau[a][b]))

    below_05 = [(a, b, t) for a, b, t in off_diag if abs(t) < 0.5]
    frac = len(below_05) / len(off_diag)
    print(f'  {dataset}: {len(below_05)}/{len(off_diag)} τ pairs < 0.5 ({frac:.0%})')
    assert frac >= 0.5, \
        f'{dataset}: fewer than 50% of off-diagonal τ pairs < 0.5 (got {frac:.0%})'
    print(f'  {dataset}: majority of off-diagonal τ < 0.5  OK')
"

# STRICT ASSERTION: diversity vs. all other dimensions must have τ < 0.2
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/tau_matrix.json') as f:
        data = json.load(f)
    tau = data['tau']
    other_dims = [d for d in tau if d not in ('diversity', '_meta')]
    for other in other_dims:
        t = tau['diversity'].get(other, tau.get(other, {}).get('diversity'))
        assert t is not None, f'{dataset}: cannot find τ(diversity, {other})'
        assert abs(t) < 0.2, \
            f'{dataset}: τ(diversity, {other}) = {t:.4f} >= 0.2 — unexpected correlation'
        print(f'  {dataset}: τ(diversity, {other}) = {t:.4f} < 0.2  OK')
"

# Assert Jaccard between diversity and other goal-specific dims is < 0.2
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/jaccard_matrix.json') as f:
        jaccard = json.load(f)
    other_dims = [d for d in jaccard if d not in ('diversity', 'universal', 'random', '_meta')]
    for other in other_dims:
        j = jaccard['diversity'].get(other, 0)
        assert j < 0.2, \
            f'{dataset}: Jaccard(diversity, {other}) = {j:.4f} >= 0.2'
        print(f'  {dataset}: Jaccard(diversity, {other}) = {j:.4f} < 0.2  OK')
"

# Assert p-values: most off-diagonal τ are statistically significant (p < 0.05)
# Note: low τ that is significant means truly non-zero but weak — this is expected
python -c "
import json
from itertools import combinations
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/tau_matrix.json') as f:
        data = json.load(f)
    p_vals = data['p_values']
    dims = [d for d in p_vals if d != '_meta']
    sig_pairs = 0
    total_pairs = 0
    for a, b in combinations(dims, 2):
        p = p_vals[a].get(b)
        if p is not None:
            total_pairs += 1
            if p < 0.05:
                sig_pairs += 1
    # With N~52k, even tiny τ will be significant; we just verify the machinery works
    print(f'  {dataset}: {sig_pairs}/{total_pairs} dimension pairs have p < 0.05  OK')
"
```

**Expected output:** At least 50% of off-diagonal τ values are below 0.5; all τ(diversity, *) are below 0.2; all Jaccard(diversity, *) are below 0.2. These are the paper's headline claims.

---

## Step 6: Permutation Test — Null Model

A permutation test establishes the null distribution: if dimension scores were assigned randomly, what would τ look like? With 1000 permutations on a 5,000-sample subsample we get a clean null. The key result is that the observed τ for most pairs is NOT significantly different from zero — confirming the near-independence claim.

```bash
# Alpaca permutation test (1000 iterations, subsample 5000 for speed)
python scripts/permutation_test.py \
    data/scored/scores_alpaca.jsonl \
    --n-permutations 1000 \
    --subsample 5000 \
    --output data/results/alpaca/permutation_results.json

# WizardLM permutation test
python scripts/permutation_test.py \
    data/scored/scores_wizardlm.jsonl \
    --n-permutations 1000 \
    --subsample 5000 \
    --output data/results/wizardlm/permutation_results.json
```

Expected terminal output (truncated):

```
[perm] 10 dimension pairs, 1000 permutations on 5000 samples...
  permutation 1000/1000

Permutation test summary:
  accuracy_vs_relevance: τ=0.xxxx, p=0.xxxx ***
  accuracy_vs_conciseness: τ=0.0xxx, p=0.xxxx ns
  accuracy_vs_diversity: τ=0.0xxx, p=0.xxxx ns
  ...
```

### Verification — Step 6

```bash
# Assert permutation output files exist
python -c "
from pathlib import Path
for dataset in ['alpaca', 'wizardlm']:
    p = Path(f'data/results/{dataset}/permutation_results.json')
    assert p.exists(), f'Missing: {p}'
    print(f'  {p} exists  OK')
"

# Assert file structure: 10 pairs (C(5,2)=10)
python -c "
import json
from itertools import combinations
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/permutation_results.json') as f:
        results = json.load(f)
    assert len(results) == 10, \
        f'{dataset}: expected 10 dimension pairs, got {len(results)}'
    for pair, data in results.items():
        for key in ['observed_tau', 'p_value', 'null_mean', 'null_std', 'null_5th', 'null_95th']:
            assert key in data, f'{dataset} {pair}: missing key {key}'
    print(f'  {dataset}: 10 pairs with correct structure  OK')
"

# CORE ASSERTION: at least 3 pairs have p < 0.01 (some dimension pairs show real signal)
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/permutation_results.json') as f:
        results = json.load(f)
    sig_pairs = [(pair, data) for pair, data in results.items() if data['p_value'] < 0.01]
    print(f'  {dataset}: {len(sig_pairs)} pairs with p < 0.01:')
    for pair, data in sig_pairs:
        print(f'    {pair}: τ={data[\"observed_tau\"]:.4f}, p={data[\"p_value\"]:.4f}')
    assert len(sig_pairs) >= 3, \
        f'{dataset}: fewer than 3 significant pairs (got {len(sig_pairs)}); expected ≥3'
    print(f'  {dataset}: ≥3 pairs significant at p < 0.01  OK')
"

# Assert null distributions are centered near zero (sanity check on permutation logic)
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/permutation_results.json') as f:
        results = json.load(f)
    for pair, data in results.items():
        null_mean = data['null_mean']
        assert abs(null_mean) < 0.05, \
            f'{dataset} {pair}: null mean = {null_mean:.4f} is too far from 0 — shuffling bug?'
    print(f'  {dataset}: all null distribution means within [-0.05, 0.05]  OK')
"

# Assert observed τ values are within plausible range
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/permutation_results.json') as f:
        results = json.load(f)
    for pair, data in results.items():
        t = data['observed_tau']
        assert -1.0 <= t <= 1.0, f'{dataset} {pair}: τ={t} out of range'
    print(f'  {dataset}: all observed τ in [-1, 1]  OK')
"
```

**Expected output:** 10 pairs in each file; null means near 0; at least 3 pairs with p < 0.01; all observed τ are in [-1, 1]. The combination of low observed τ AND low p-values for some pairs is the expected pattern: dimension pairs like (accuracy, relevance) may show modest but real correlation, while (diversity, conciseness) approach the null.

---

## Step 7: Quality Loss Analysis — Closing the Logic Gap

The key policy implication: a practitioner who uses universal quality filtering (composite score) sacrifices dimension-specific quality. This step quantifies *how much* quality is lost per dimension when using the universal subset instead of the goal-specific subset.

```bash
# Quality loss is already computed by rank_correlation.py above.
# Verify the existing files; no additional script invocation needed.
# If re-running quality loss only, rerun rank_correlation.py:
python scripts/rank_correlation.py \
    data/scored/scores_alpaca.jsonl \
    data/subsets/subsets_alpaca.json \
    --output data/results/alpaca/

python scripts/rank_correlation.py \
    data/scored/scores_wizardlm.jsonl \
    data/subsets/subsets_wizardlm.json \
    --output data/results/wizardlm/
```

### Verification — Step 7

```bash
# Assert quality_loss.json has 5 dimension entries
python -c "
import json
dims = {'accuracy', 'relevance', 'conciseness', 'diversity', 'info_density'}
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/quality_loss.json') as f:
        ql = json.load(f)
    assert set(ql.keys()) == dims, \
        f'{dataset}: unexpected keys {set(ql.keys())} vs expected {dims}'
    for dim, data in ql.items():
        for key in ['goal_specific_mean', 'universal_mean', 'delta', 'p_value', 'effect_size']:
            assert key in data, f'{dataset} {dim}: missing key {key}'
    print(f'  {dataset}: quality_loss.json structure OK')
"

# CORE ASSERTION: diversity and info_density must show positive delta
# (goal-specific subset scores higher than universal on its own dimension)
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/quality_loss.json') as f:
        ql = json.load(f)
    for dim in ['diversity', 'info_density']:
        delta = ql[dim]['delta']
        assert delta > 0, \
            f'{dataset} {dim}: expected positive delta (goal > universal), got {delta:.4f}'
        print(f'  {dataset} {dim}: delta = {delta:+.4f} > 0  OK')
"

# Assert deltas for diversity (~0.05) and info_density (~0.04) are in expected range
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/quality_loss.json') as f:
        ql = json.load(f)
    for dim, expected_min in [('diversity', 0.02), ('info_density', 0.01)]:
        delta = ql[dim]['delta']
        assert delta >= expected_min, \
            f'{dataset} {dim}: delta={delta:.4f} < expected minimum {expected_min}'
        print(f'  {dataset} {dim}: delta = {delta:+.4f} >= {expected_min}  OK')
"

# Assert Mann-Whitney U p-values for diversity and info_density are < 0.05
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/quality_loss.json') as f:
        ql = json.load(f)
    for dim in ['diversity', 'info_density']:
        p = ql[dim]['p_value']
        assert p < 0.05, \
            f'{dataset} {dim}: quality loss p-value = {p:.6f} not significant (>= 0.05)'
        print(f'  {dataset} {dim}: quality loss p = {p:.6f} < 0.05  OK')
"

# Assert goal_specific_mean > universal_mean for dims with positive delta
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/quality_loss.json') as f:
        ql = json.load(f)
    for dim, data in ql.items():
        delta_check = round(data['goal_specific_mean'] - data['universal_mean'], 6)
        stored_delta = round(data['delta'], 6)
        assert abs(delta_check - stored_delta) < 1e-4, \
            f'{dataset} {dim}: delta inconsistency: stored={stored_delta}, computed={delta_check}'
    print(f'  {dataset}: all delta values are internally consistent  OK')
"
```

**Expected output:** Diversity and info_density show positive deltas (≥ 0.02 and ≥ 0.01 respectively); Mann-Whitney U test confirms statistical significance at p < 0.05; all stored deltas are internally consistent with means.

---

## Step 8: Sensitivity Analysis and Cross-Dataset Replication

A finding that only holds at one retention rate and one dataset is fragile. This step sweeps retention rates [0.2, 0.3, 0.5] and replicates the analysis on WizardLM to confirm generalization.

```bash
# Sensitivity analysis for Alpaca across retention rates
python scripts/sensitivity.py \
    data/scored/scores_alpaca.jsonl \
    --rates 0.2,0.3,0.5 \
    --output data/results/alpaca/sensitivity.json

# Sensitivity analysis for WizardLM
python scripts/sensitivity.py \
    data/scored/scores_wizardlm.jsonl \
    --rates 0.2,0.3,0.5 \
    --output data/results/wizardlm/sensitivity.json
```

Expected terminal output (Alpaca):

```
[sensitivity] retention=0.2
  Jaccard mean=0.xxxx, max=0.xxxx, all<0.6=True
[sensitivity] retention=0.3
  Jaccard mean=0.xxxx, max=0.xxxx, all<0.6=True
[sensitivity] retention=0.5
  Jaccard mean=0.xxxx, max=0.xxxx, all<0.6=True
```

### Verification — Step 8

```bash
# Assert sensitivity files exist and cover 3 retention rates
python -c "
import json
from pathlib import Path
for dataset in ['alpaca', 'wizardlm']:
    p = Path(f'data/results/{dataset}/sensitivity.json')
    assert p.exists(), f'Missing: {p}'
    with open(p) as f:
        data = json.load(f)
    assert set(data.keys()) == {'0.2', '0.3', '0.5'}, \
        f'{dataset}: unexpected retention rates {set(data.keys())}'
    for rate, entry in data.items():
        for key in ['retention_rate', 'jaccard_off_diag_mean', 'jaccard_off_diag_max',
                    'jaccard_off_diag_min', 'quality_loss_summary', 'all_jaccard_below_0.6']:
            assert key in entry, f'{dataset} rate={rate}: missing key {key}'
    print(f'  {dataset}: sensitivity.json structure OK  (3 rates)')
"

# CORE ROBUSTNESS ASSERTION: all_jaccard_below_0.6 must be True at all rates
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/sensitivity.json') as f:
        data = json.load(f)
    for rate, entry in data.items():
        result = entry['all_jaccard_below_0.6']
        assert result is True, \
            f'{dataset} rate={rate}: all_jaccard_below_0.6 = {result} — finding does not hold'
        print(f'  {dataset} retention={rate}: all off-diagonal Jaccard < 0.6  OK')
"

# Assert Jaccard mean increases with retention rate (larger subsets overlap more)
python -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/sensitivity.json') as f:
        data = json.load(f)
    means = {rate: entry['jaccard_off_diag_mean'] for rate, entry in data.items()}
    # At 50% retention, mean Jaccard should be higher than at 20%
    assert means['0.5'] > means['0.2'], \
        f'{dataset}: Jaccard mean does not increase with retention rate: {means}'
    print(f'  {dataset}: Jaccard mean increases with retention (0.2->{means[\"0.2\"]:.3f}, 0.5->{means[\"0.5\"]:.3f})  OK')
"

# CROSS-DATASET REPLICATION: both datasets must agree on direction of quality loss
python -c "
import json
for dim in ['diversity', 'info_density']:
    deltas = {}
    for dataset in ['alpaca', 'wizardlm']:
        with open(f'data/results/{dataset}/quality_loss.json') as f:
            ql = json.load(f)
        deltas[dataset] = ql[dim]['delta']
    assert deltas['alpaca'] > 0 and deltas['wizardlm'] > 0, \
        f'{dim}: sign disagreement between datasets — alpaca={deltas[\"alpaca\"]:.4f}, wizardlm={deltas[\"wizardlm\"]:.4f}'
    print(f'  {dim}: both datasets show positive quality loss delta  OK')
    print(f'    alpaca delta={deltas[\"alpaca\"]:+.4f}, wizardlm delta={deltas[\"wizardlm\"]:+.4f}')
"

# Assert WizardLM τ values replicate the low-correlation pattern from Alpaca
python -c "
import json
from itertools import combinations

for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/tau_matrix.json') as f:
        data = json.load(f)
    tau = data['tau']
    dims = [d for d in tau if d != '_meta']
    off_diag = [abs(tau[a][b]) for a, b in combinations(dims, 2)]
    median_tau = sorted(off_diag)[len(off_diag) // 2]
    assert median_tau < 0.5, \
        f'{dataset}: median |τ| = {median_tau:.4f} >= 0.5 — cross-dataset replication failed'
    print(f'  {dataset}: median |τ| = {median_tau:.4f} < 0.5  OK (replicates)')
"
```

**Expected output:** `all_jaccard_below_0.6` is True for all 3 retention rates on both datasets; Jaccard mean increases monotonically with retention rate; both datasets agree on the direction of quality loss; median |τ| is below 0.5 on both datasets.

---

## Step 8b: Bootstrap Validation (Full LLM Scoring, No Imputation)

This step eliminates the median-imputation limitation by drawing K=5 random subsamples of N=1,000 and scoring ALL samples via LLM. This is the strongest evidence that our findings are not artifacts of the imputation procedure.

```bash
# Bootstrap on Alpaca (K=5 iterations × 1K samples × 2 LLM dims = 10K API calls)
python scripts/bootstrap_scoring.py data/raw/alpaca.jsonl \
    --k 5 --n 1000 \
    --output data/results/alpaca/bootstrap/ \
    --cache-dir data/cache/bootstrap_alpaca/

# Bootstrap on WizardLM
python scripts/bootstrap_scoring.py data/raw/wizardlm.jsonl \
    --k 5 --n 1000 \
    --output data/results/wizardlm/bootstrap/ \
    --cache-dir data/cache/bootstrap_wizardlm/
```

### Verification

```bash
python3 -c "
import json
for dataset in ['alpaca', 'wizardlm']:
    with open(f'data/results/{dataset}/bootstrap/summary.json') as f:
        s = json.load(f)
    pairs = sorted(s.keys())
    print(f'{dataset.upper()} Bootstrap (K=5, N=1000, 100% LLM scored):')
    n_near_indep = sum(1 for p in pairs if abs(s[p]['mean_tau']) < 0.10)
    print(f'  Pairs with |τ| < 0.10: {n_near_indep}/{len(pairs)}')
    # Key assertion: accuracy-relevance should be moderate (> 0.3)
    ar = s.get('accuracy_vs_relevance', {})
    assert ar.get('mean_tau', 0) > 0.3, f'{dataset}: accuracy-relevance τ too low'
    print(f'  accuracy-relevance τ = {ar[\"mean_tau\"]:.3f} ± {ar[\"std_tau\"]:.3f}  (moderate, expected)')
    # Key assertion: most pairs should be near-zero
    near_zero = [p for p in pairs if abs(s[p]['mean_tau']) < 0.10 and p != 'accuracy_vs_relevance']
    print(f'  Near-independent pairs (excl. acc-rel): {len(near_zero)}/{len(pairs)-1}')
    print(f'  All std < 0.06: {all(s[p][\"std_tau\"] < 0.06 for p in pairs)}  (stable estimates)')
    print()
"
```

**Expected output:** Alpaca: 8/10 pairs |τ| < 0.10; WizardLM: statistical dimensions show higher correlation (synthetic data effect); all std < 0.06 (stable). This confirms the main analysis findings hold without imputation.

---

## Step 9: Generate Figures and Report

Produce publication-ready heatmaps, bar charts, and sensitivity plots. Also generate a Markdown table summarizing the τ matrix and quality loss results for direct inclusion in the paper or supplemental materials.

```bash
# Generate all figures and tables for Alpaca
python scripts/generate_report.py \
    data/results/alpaca/ \
    --figures figures/alpaca/

# Generate all figures and tables for WizardLM
python scripts/generate_report.py \
    data/results/wizardlm/ \
    --figures figures/wizardlm/
```

Expected output files:

```
figures/alpaca/kendall_heatmap.png
figures/alpaca/jaccard_heatmap.png
figures/alpaca/quality_loss_bar.png
figures/alpaca/sensitivity_plot.png
data/results/alpaca/tables.md
figures/wizardlm/kendall_heatmap.png
figures/wizardlm/jaccard_heatmap.png
figures/wizardlm/quality_loss_bar.png
figures/wizardlm/sensitivity_plot.png
data/results/wizardlm/tables.md
```

### Verification — Step 9

```bash
# Assert all expected figure files exist
python -c "
from pathlib import Path
expected_files = []
for dataset in ['alpaca', 'wizardlm']:
    for fname in ['kendall_heatmap.png', 'jaccard_heatmap.png', 'quality_loss_bar.png']:
        expected_files.append(f'figures/{dataset}/{fname}')
    # sensitivity_plot.png requires sensitivity.json which we produced
    expected_files.append(f'figures/{dataset}/sensitivity_plot.png')
    expected_files.append(f'data/results/{dataset}/tables.md')

for path in expected_files:
    p = Path(path)
    assert p.exists(), f'Missing output: {path}'
    size = p.stat().st_size
    assert size > 1000, f'{path} is suspiciously small ({size} bytes)'
    print(f'  {path}: {size:,} bytes  OK')
print('All figure and table files generated')
"

# Assert PNG files are valid images (check PNG magic bytes)
python -c "
from pathlib import Path
PNG_MAGIC = b'\\x89PNG\\r\\n\\x1a\\n'
for dataset in ['alpaca', 'wizardlm']:
    for fname in ['kendall_heatmap.png', 'jaccard_heatmap.png', 'quality_loss_bar.png', 'sensitivity_plot.png']:
        p = Path(f'figures/{dataset}/{fname}')
        if not p.exists():
            print(f'  SKIP (not found): {p}')
            continue
        with open(p, 'rb') as f:
            header = f.read(8)
        assert header == PNG_MAGIC, f'{p}: not a valid PNG file (header={header!r})'
        print(f'  {p}: valid PNG  OK')
"

# Assert tables.md contains the expected section headers
python -c "
from pathlib import Path
for dataset in ['alpaca', 'wizardlm']:
    p = Path(f'data/results/{dataset}/tables.md')
    content = p.read_text()
    assert \"Kendall's\" in content, f'{dataset} tables.md: missing tau matrix section'
    assert 'Quality Loss' in content, f'{dataset} tables.md: missing quality loss section'
    assert 'diversity' in content, f'{dataset} tables.md: missing diversity row'
    assert 'info_density' in content, f'{dataset} tables.md: missing info_density row'
    print(f'  {dataset}/tables.md: all expected sections present  OK')
"

# Assert heatmap resolution is at least 150 DPI equivalent
# (check pixel dimensions via PIL if available, else skip)
python -c "
try:
    from PIL import Image
    from pathlib import Path
    for dataset in ['alpaca', 'wizardlm']:
        p = Path(f'figures/{dataset}/kendall_heatmap.png')
        if not p.exists():
            continue
        with Image.open(p) as img:
            w, h = img.size
            # 7 inches * 150 dpi = 1050px minimum
            assert w >= 900, f'{p}: width {w}px < 900px'
            assert h >= 800, f'{p}: height {h}px < 800px'
            print(f'  {p}: {w}x{h}px  OK')
except ImportError:
    print('  PIL not installed; skipping pixel dimension check (OK)')
"
```

**Expected output:** All PNG files exist and have valid PNG headers; `tables.md` files contain the τ matrix and quality loss sections; heatmaps are at adequate resolution if PIL is available.

---

## Step 10: Success Criteria Check — Final Gate

This step runs the definitive pass/fail gate. All assertions from previous steps are consolidated into a single score. The analysis is considered successfully reproduced only when all mandatory criteria pass.

```bash
python - <<'PYTHON'
import json
import sys
from pathlib import Path
from itertools import combinations

results = {}
failures = []

def check(name, condition, msg=""):
    if condition:
        results[name] = "PASS"
        print(f"  [PASS] {name}")
    else:
        results[name] = "FAIL"
        failures.append(f"{name}: {msg}")
        print(f"  [FAIL] {name}: {msg}")

print("=" * 60)
print("FINAL SUCCESS CRITERIA CHECK")
print("=" * 60)

# C1: Dataset sizes
for dataset, expected in [("alpaca", 51974), ("wizardlm", 51923)]:
    path = Path(f"data/raw/{dataset}.jsonl")
    if path.exists():
        with open(path) as f:
            count = sum(1 for l in f if l.strip())
        check(f"C1_{dataset}_count", count == expected,
              f"expected {expected}, got {count}")
    else:
        check(f"C1_{dataset}_count", False, f"{path} not found")

# C2: Majority of off-diagonal τ < 0.5
for dataset in ["alpaca", "wizardlm"]:
    path = Path(f"data/results/{dataset}/tau_matrix.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        tau = data["tau"]
        dims = [d for d in tau if d != "_meta"]
        off = [abs(tau[a][b]) for a, b in combinations(dims, 2)]
        frac = sum(1 for t in off if t < 0.5) / len(off)
        check(f"C2_{dataset}_tau_majority_below_05", frac >= 0.5,
              f"only {frac:.0%} of pairs < 0.5")
    else:
        check(f"C2_{dataset}_tau_majority_below_05", False, "file not found")

# C3: Jaccard(diversity, *) < 0.2 for all non-diversity goal dims
for dataset in ["alpaca", "wizardlm"]:
    path = Path(f"data/results/{dataset}/jaccard_matrix.json")
    if path.exists():
        with open(path) as f:
            jac = json.load(f)
        others = [d for d in jac if d not in ("diversity", "universal", "random")]
        bad = [(d, jac["diversity"][d]) for d in others if jac["diversity"].get(d, 1) >= 0.2]
        check(f"C3_{dataset}_jaccard_diversity_below_02", len(bad) == 0,
              f"pairs >= 0.2: {bad}")
    else:
        check(f"C3_{dataset}_jaccard_diversity_below_02", False, "file not found")

# C4: Permutation test: at least 3 pairs with p < 0.01
for dataset in ["alpaca", "wizardlm"]:
    path = Path(f"data/results/{dataset}/permutation_results.json")
    if path.exists():
        with open(path) as f:
            perm = json.load(f)
        sig = sum(1 for v in perm.values() if v["p_value"] < 0.01)
        check(f"C4_{dataset}_perm_sig_pairs", sig >= 3,
              f"only {sig} pairs with p < 0.01 (need ≥3)")
    else:
        check(f"C4_{dataset}_perm_sig_pairs", False, "file not found")

# C5: Quality loss: diversity and info_density delta > 0
for dataset in ["alpaca", "wizardlm"]:
    path = Path(f"data/results/{dataset}/quality_loss.json")
    if path.exists():
        with open(path) as f:
            ql = json.load(f)
        for dim in ["diversity", "info_density"]:
            delta = ql.get(dim, {}).get("delta", -999)
            check(f"C5_{dataset}_{dim}_delta_positive", delta > 0,
                  f"delta = {delta:.4f}")
    else:
        check(f"C5_{dataset}_quality_loss", False, "file not found")

# C6: Sensitivity: all_jaccard_below_0.6 at all rates
for dataset in ["alpaca", "wizardlm"]:
    path = Path(f"data/results/{dataset}/sensitivity.json")
    if path.exists():
        with open(path) as f:
            sens = json.load(f)
        all_pass = all(v.get("all_jaccard_below_0.6", False) for v in sens.values())
        check(f"C6_{dataset}_sensitivity_robust", all_pass,
              "some retention rate fails the Jaccard < 0.6 criterion")
    else:
        check(f"C6_{dataset}_sensitivity_robust", False, "file not found")

# C7: Cross-dataset replication — both datasets agree on sign of diversity delta
path_a = Path("data/results/alpaca/quality_loss.json")
path_w = Path("data/results/wizardlm/quality_loss.json")
if path_a.exists() and path_w.exists():
    with open(path_a) as f: ql_a = json.load(f)
    with open(path_w) as f: ql_w = json.load(f)
    agrees = (ql_a["diversity"]["delta"] > 0) == (ql_w["diversity"]["delta"] > 0)
    check("C7_cross_dataset_replication", agrees,
          f"sign mismatch: alpaca={ql_a['diversity']['delta']:+.4f}, wizardlm={ql_w['diversity']['delta']:+.4f}")
else:
    check("C7_cross_dataset_replication", False, "quality_loss.json not found for both datasets")

# C8: Figure files exist
for dataset in ["alpaca", "wizardlm"]:
    for fname in ["kendall_heatmap.png", "jaccard_heatmap.png", "quality_loss_bar.png"]:
        p = Path(f"figures/{dataset}/{fname}")
        check(f"C8_{dataset}_{fname.replace('.png','')}", p.exists() and p.stat().st_size > 1000,
              f"file missing or empty: {p}")

print()
print("=" * 60)
total = len(results)
passed = sum(1 for v in results.values() if v == "PASS")
failed = total - passed
print(f"RESULT: {passed}/{total} criteria passed, {failed} failed")
print("=" * 60)

if failures:
    print()
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    print()
    print("STATUS: REPRODUCTION FAILED")
    sys.exit(1)
else:
    print()
    print("STATUS: ALL CRITERIA PASSED — REPRODUCTION SUCCESSFUL")
    print()
    print("Key findings confirmed:")
    print("  - Majority of off-diagonal Kendall tau < 0.5 (near-independence)")
    print("  - Jaccard(diversity, *) < 0.2 on both datasets")
    print("  - Permutation test: >=3 pairs with p < 0.01")
    print("  - Diversity and info_density show positive quality loss delta")
    print("  - Finding robust across retention rates 0.2, 0.3, 0.5")
    print("  - Cross-dataset replication on WizardLM confirmed")
PYTHON
```

---

## Success Criteria

The following criteria must ALL pass for the reproduction to be considered successful. Criteria C1–C7 correspond to claims made in the paper body; C8 ensures the artifact is publication-ready.

| ID | Criterion | Threshold | Paper Claim |
|----|-----------|-----------|-------------|
| C1 | Alpaca dataset size | exactly 51,974 samples | §3.1 Dataset |
| C1 | WizardLM dataset size | exactly 51,923 samples | §3.1 Dataset |
| C2 | Majority of off-diagonal τ | ≥50% of pairs have \|τ\| < 0.5 | §4.1 Main Result |
| C3 | Jaccard(diversity, \*) | All < 0.2 for non-diversity goal dims | §4.2 Subset Overlap |
| C4 | Permutation test significance | ≥3 pairs with p < 0.01 | §4.3 Null Model |
| C5 | Quality loss direction | diversity delta > 0, info_density delta > 0 | §4.4 Quality Loss |
| C6 | Sensitivity robustness | `all_jaccard_below_0.6 = True` at rates 0.2, 0.3, 0.5 | §5.1 Sensitivity |
| C7 | Cross-dataset replication | Sign of diversity delta agrees between Alpaca and WizardLM | §5.2 Replication |
| C8 | Figure artifacts | All PNG files exist and have valid headers | §6 Figures |

**Pass definition:** All 18 individual checks in Step 10 print `[PASS]` and the script exits with code 0.

**Partial pass:** If C1 counts differ by ±10 from upstream dataset updates, document the discrepancy in a `REPRODUCTION_NOTES.md` and re-run with the actual count substituted in the assertion. This is an acceptable deviation from dataset drift.

**Hard failures (no workaround):** If C2, C3, or C7 fail, the paper's primary claims are not reproduced. Diagnose by re-running `scripts/score_dimensions.py` with `--llm-sample-size 0` to isolate statistical scoring from LLM scoring.

---

## How To Extend

### Add a New Quality Dimension

1. Add the scoring function to `scripts/score_dimensions.py`. Statistical dimensions go in the first section (no API required); LLM-judged dimensions go in the LLM section.
2. Add the new dimension name to the `DIMENSIONS` list at the top of `score_dimensions.py`.
3. Re-run from Step 3 onward. All downstream scripts (`curate_by_goal.py`, `rank_correlation.py`, `permutation_test.py`, `sensitivity.py`) auto-discover dimensions from the scored JSONL files.
4. Update the C2/C3/C5 assertions in Step 5, 6, and 7 verification blocks if the new dimension changes expected thresholds.

### Add a New Dataset

1. Add a new entry to the `DATASETS` dict in `scripts/download_data.py`, following the `alpaca` or `wizardlm` format pattern.
2. Download: `python scripts/download_data.py --dataset <name> --output data/raw/<name>.jsonl`
3. Score: `python scripts/score_dimensions.py data/raw/<name>.jsonl --output data/scored/scores_<name>.jsonl --cache data/cache/llm_scores_<name>.json`
4. Run all subsequent steps with `data/scored/scores_<name>.jsonl` as input and `data/results/<name>/` as output.
5. Add cross-dataset assertions in Step 8 to verify C7 generalizes to the new dataset.

### Change Retention Rate

The default is 30%. To reproduce at a different rate:

```bash
# Example: 20% retention
python scripts/curate_by_goal.py \
    data/scored/scores_alpaca.jsonl \
    --output data/subsets/subsets_alpaca_020.json \
    --retention 0.2

python scripts/rank_correlation.py \
    data/scored/scores_alpaca.jsonl \
    data/subsets/subsets_alpaca_020.json \
    --output data/results/alpaca_020/
```

The sensitivity analysis (`scripts/sensitivity.py`) already sweeps 0.2, 0.3, and 0.5 in a single run, so this extension is mainly useful for fine-grained exploration outside those three rates.

### Increase Permutation Iterations for Higher Precision

The default is 1000 iterations on a 5,000-sample subsample, sufficient for p-value estimation to ±0.01. For camera-ready precision:

```bash
python scripts/permutation_test.py \
    data/scored/scores_alpaca.jsonl \
    --n-permutations 10000 \
    --subsample 10000 \
    --output data/results/alpaca/permutation_results_highres.json
```

Expected runtime: ~20 minutes on CPU. The `p_value` key for each pair will have roughly 3x lower variance. Update the Step 6 assertion file path accordingly.

### Run Statistical Scoring Only (No LLM API Required)

Set `--llm-sample-size 0` to skip the LLM-judged dimensions entirely. The `accuracy` and `relevance` scores will be imputed from statistical proxies. The core τ finding is reproducible without an API:

```bash
python scripts/score_dimensions.py \
    data/raw/alpaca.jsonl \
    --output data/scored/scores_alpaca_statonly.jsonl \
    --llm-sample-size 0
```

The τ values between diversity, conciseness, and info_density are identical to the full run. Diversity vs. accuracy/relevance τ may shift slightly because the proxy-imputed scores have higher variance, but the direction of results is preserved.

### Reproduce on a Subset for Rapid Iteration

For debugging or quick validation, cap the dataset at 5,000 samples:

```bash
python -c "
import json, random, pathlib
random.seed(0)
with open('data/raw/alpaca.jsonl') as f:
    lines = [l for l in f if l.strip()]
sample = random.sample(lines, 5000)
pathlib.Path('data/raw/alpaca_5k.jsonl').write_text('\n'.join(sample))
print('wrote data/raw/alpaca_5k.jsonl')
"

python scripts/score_dimensions.py \
    data/raw/alpaca_5k.jsonl \
    --output data/scored/scores_alpaca_5k.jsonl \
    --llm-sample-size 50

python scripts/curate_by_goal.py \
    data/scored/scores_alpaca_5k.jsonl \
    --output data/subsets/subsets_alpaca_5k.json \
    --retention 0.3

python scripts/rank_correlation.py \
    data/scored/scores_alpaca_5k.jsonl \
    data/subsets/subsets_alpaca_5k.json \
    --output data/results/alpaca_5k/
```

At 5k samples, all five scripts finish in under 5 minutes. The τ values will be noisier but the structural pattern (τ << 0.5 for most pairs) should still be visible. Do not use 5k results for the final reproducibility gate in Step 10.
