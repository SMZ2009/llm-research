"""Freeze deterministic Dolly splits for the Qwen3 LoRA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    ifeval_prompts = {
        json.loads(line)["prompt"].strip().casefold()
        for line in args.ifeval.open(encoding="utf-8")
    }
    seen = set()
    cleaned = []
    dropped = {"empty": 0, "duplicate_prompt": 0, "ifeval_exact_prompt": 0}
    for index, row in enumerate(rows):
        instruction = row["instruction"].strip()
        context = row["context"].strip()
        response = row["response"].strip()
        if not instruction or not response:
            dropped["empty"] += 1
            continue
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        key = " ".join(prompt.casefold().split())
        if key in seen:
            dropped["duplicate_prompt"] += 1
            continue
        if instruction.casefold() in ifeval_prompts or prompt.casefold() in ifeval_prompts:
            dropped["ifeval_exact_prompt"] += 1
            continue
        seen.add(key)
        cleaned.append({
            "id": f"dolly_{index:05d}",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        })

    random.Random(args.seed).shuffle(cleaned)
    n = len(cleaned)
    n_test = round(n * 0.05)
    n_validation = round(n * 0.05)
    splits = {
        "test": cleaned[:n_test],
        "validation": cleaned[n_test:n_test + n_validation],
        "train": cleaned[n_test + n_validation:],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "source_sha256": digest(args.source),
        "ifeval_sha256": digest(args.ifeval),
        "raw_rows": len(rows),
        "deduplication": "first row per whitespace-normalized, casefolded instruction plus context",
        "ifeval_leakage_check": "exact casefolded instruction or full prompt match",
        "dropped": dropped,
        "splits": {},
    }
    for split, records in splits.items():
        target = args.output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": len(records), "sha256": digest(target)}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
