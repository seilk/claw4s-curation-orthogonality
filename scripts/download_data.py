"""Download instruction-tuning datasets from HuggingFace.

Usage:
    python scripts/download_data.py --dataset alpaca --output data/alpaca.jsonl
    python scripts/download_data.py --dataset wizardlm --output data/wizardlm.jsonl --max-samples 52000
"""
import json
import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "alpaca": {
        "repo": "tatsu-lab/alpaca",
        "split": "train",
        "format": "flat",  # instruction/output columns
        "fields": {"instruction": "instruction", "response": "output"},
    },
    "wizardlm": {
        "repo": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
        "split": "train",
        "format": "conversations",  # conversations list
        "fields": {},
    },
}


def download(name: str, output: Path, max_samples: int | None = None, seed: int = 42):
    cfg = DATASETS[name]
    ds = load_dataset(cfg["repo"], split=cfg["split"])

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w") as f:
        for i, row in enumerate(ds):
            if cfg["format"] == "conversations":
                convs = row.get("conversations", [])
                inst = next((c["value"] for c in convs if c["from"] == "human"), "")
                resp = next((c["value"] for c in convs if c["from"] == "gpt"), "")
            else:
                inst = row.get(cfg["fields"]["instruction"], "")
                resp = row.get(cfg["fields"]["response"], "")
            # Skip empty rows
            if not inst.strip() or not resp.strip():
                continue
            f.write(json.dumps({
                "id": i,
                "instruction": inst,
                "response": resp,
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"[download] {name}: {count} samples -> {output}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download instruction-tuning datasets")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download(args.dataset, args.output, args.max_samples, args.seed)
