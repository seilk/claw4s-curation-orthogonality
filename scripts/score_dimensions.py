"""Score dataset samples on 5 quality dimensions.

Dimensions:
    accuracy     - LLM-as-judge factual correctness (0-1)
    relevance    - LLM-as-judge instruction-response alignment (0-1)
    conciseness  - Statistical: inverse hedging + verbosity penalty (0-1)
    diversity    - Distinct-2 + embedding distance from centroid (0-1)
    info_density - Compression ratio + Shannon entropy (0-1)

Usage:
    python scripts/score_dimensions.py data/alpaca.jsonl \
        --output output/alpaca/scores.jsonl \
        --llm-sample-size 200
"""
import json
import math
import os
import re
import zlib
import random
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

DIMENSIONS = ["accuracy", "relevance", "conciseness", "diversity", "info_density"]

# ─── Conciseness (statistical) ───────────────────────────────────────────────

HEDGING = re.compile(
    r"\b(I'm not sure|might|perhaps|possibly|I think|it seems|arguably|"
    r"to some extent|it could be|not certain|I believe|it's hard to say|"
    r"I'm not confident|may or may not)\b",
    re.IGNORECASE,
)


def score_conciseness(sample: dict) -> float:
    resp = sample.get("response", "")
    words = resp.split()
    n_words = len(words)
    if n_words == 0:
        return 0.0
    hedging_count = len(HEDGING.findall(resp))
    hedging_rate = hedging_count / n_words
    length_penalty = 1.0
    if n_words < 5:
        length_penalty = n_words / 5
    elif n_words > 300:
        length_penalty = max(0.3, 300 / n_words)
    raw = (1.0 - min(hedging_rate * 5, 1.0)) * length_penalty
    return round(min(max(raw, 0.0), 1.0), 4)


# ─── Diversity (Distinct-2 + embedding distance) ────────────────────────────

def _distinct_n(text: str, n: int = 2) -> float:
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def score_diversity_batch(samples: list[dict]) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        s.get("instruction", "") + " " + s.get("response", "") for s in samples
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    emb_distances = 1 - embeddings @ centroid

    distinct_scores = np.array([_distinct_n(t) for t in texts])

    def norm01(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else np.full(len(arr), 0.5)

    combined = 0.6 * norm01(emb_distances) + 0.4 * norm01(distinct_scores)
    return [round(float(d), 4) for d in combined]


# ─── Information Density (compression ratio + entropy) ──────────────────────

def score_info_density(sample: dict) -> float:
    resp = sample.get("response", "")
    if not resp or len(resp) < 10:
        return 0.0
    compressed = len(zlib.compress(resp.encode()))
    ratio = compressed / max(len(resp.encode()), 1)
    words = resp.lower().split()
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    max_entropy = math.log2(max(len(freq), 2))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    score = 0.5 * min(ratio, 1.0) + 0.5 * norm_entropy
    return round(min(max(score, 0.0), 1.0), 4)


# ─── LLM-as-Judge (accuracy + relevance) ────────────────────────────────────

def build_judge_prompt(sample: dict, dimension: str) -> str:
    inst = sample.get("instruction", "")
    resp = sample.get("response", "")
    if dimension == "accuracy":
        criteria = (
            "factual correctness — are all claims in the response verifiable and correct? "
            "Score 1 (completely wrong) to 10 (perfectly accurate)."
        )
    elif dimension == "relevance":
        criteria = (
            "relevance — does the response directly and fully address the instruction? "
            "Score 1 (completely off-topic) to 10 (perfectly relevant)."
        )
    else:
        criteria = f"{dimension} on a scale of 1 to 10."
    return (
        f"Rate this instruction-response pair on {criteria}\n\n"
        f"Instruction: {inst}\n\n"
        f"Response: {resp}\n\n"
        "Return ONLY a single integer (1-10)."
    )


def _load_env():
    """Load .env file if exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def call_llm_judge(prompt: str) -> int:
    import openai

    _load_env()
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("LLM_API_KEY", "")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    # Extract first integer
    for token in text.split():
        token = token.strip(".,;:!?")
        if token.isdigit():
            return min(max(int(token), 1), 10)
    return 5  # fallback


def score_llm_dimensions(
    samples: list[dict],
    dimensions: list[str],
    sample_size: int = 200,
    seed: int = 42,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    n = len(samples)
    indices = sorted(rng.sample(range(n), min(sample_size, n)))

    # Load cache if exists
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    scores: dict[str, list[float | None]] = {d: [None] * n for d in dimensions}

    for d in dimensions:
        cached_dim = cache.get(d, {})
        scored_values = []
        for idx in indices:
            key = str(idx)
            if key in cached_dim:
                val = cached_dim[key]
            else:
                prompt = build_judge_prompt(samples[idx], d)
                val = call_llm_judge(prompt)
                cached_dim[key] = val
            scores[d][idx] = val / 10.0  # normalize to 0-1
            scored_values.append(val / 10.0)

        # Fill unscored with median
        median = sorted(scored_values)[len(scored_values) // 2] if scored_values else 0.5
        scores[d] = [s if s is not None else median for s in scores[d]]

        # Update cache
        cache[d] = cached_dim
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)

        print(f"  [{d}] scored {len(indices)} samples, median={median:.2f}")

    return scores


# ─── Main pipeline ──────────────────────────────────────────────────────────

def score_all(
    data_path: Path,
    output_path: Path,
    llm_sample_size: int = 200,
    cache_path: Path | None = None,
):
    with open(data_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"[score] Loaded {len(samples)} samples from {data_path}")

    # Statistical dimensions (no API needed)
    print("[score] Computing conciseness...")
    conciseness = [score_conciseness(s) for s in samples]

    print("[score] Computing diversity (embedding + Distinct-2)...")
    diversity = score_diversity_batch(samples)

    print("[score] Computing info_density...")
    info_density = [score_info_density(s) for s in samples]

    # LLM dimensions (API calls)
    print(f"[score] Scoring accuracy + relevance via LLM ({llm_sample_size} samples)...")
    llm_scores = score_llm_dimensions(
        samples, ["accuracy", "relevance"], llm_sample_size, cache_path=cache_path
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, sample in enumerate(samples):
            row = {
                **sample,
                "_original_index": i,
                "scores": {
                    "accuracy": llm_scores["accuracy"][i],
                    "relevance": llm_scores["relevance"][i],
                    "conciseness": conciseness[i],
                    "diversity": diversity[i],
                    "info_density": info_density[i],
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[score] Done. {len(samples)} scored samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm-sample-size", type=int, default=200)
    parser.add_argument("--cache", type=Path, default=None, help="LLM score cache file")
    args = parser.parse_args()
    score_all(args.data, args.output, args.llm_sample_size, args.cache)
