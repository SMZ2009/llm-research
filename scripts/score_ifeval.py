"""Score fixed responses with the vendored official IFEval functions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_root.parent))
    from instruction_following_eval import evaluation_lib

    inputs = evaluation_lib.read_prompt_list(str(args.input))
    responses = {}
    for name, path in (("base", args.base), ("adapter", args.adapter)):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        keys = [r["key"] for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != {r.key for r in inputs}:
            raise ValueError(f"{name}: incomplete or duplicate response keys")
        by_key = {r["key"]: r for r in rows}
        if any(by_key[r.key]["prompt"] != r.prompt for r in inputs):
            raise ValueError(f"{name}: prompt mismatch")
        responses[name] = {r["prompt"]: by_key[r.key]["response"] for r in inputs}
    per_sample = []
    for item in inputs:
        record = {"key": item.key, "prompt": item.prompt}
        for name in ("base", "adapter"):
            strict = evaluation_lib.test_instruction_following_strict(item, responses[name])
            loose = evaluation_lib.test_instruction_following_loose(item, responses[name])
            record[name] = {
                "strict": strict.follow_all_instructions,
                "loose": loose.follow_all_instructions,
                "per_instruction_strict": strict.follow_instruction_list,
            }
        per_sample.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "ifeval_per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_sample:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(per_sample)
    base = sum(r["base"]["strict"] for r in per_sample) / n
    adapter = sum(r["adapter"]["strict"] for r in per_sample) / n
    paired = [int(r["adapter"]["strict"]) - int(r["base"]["strict"]) for r in per_sample]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(paired, k=n)) / n for _ in range(10000))
    summary = {
        "n": n,
        "primary_metric": "IFEval strict prompt accuracy",
        "base": base,
        "adapter": adapter,
        "difference": adapter - base,
        "paired_bootstrap_95pct_ci": [boot[250], boot[9750]],
        "base_only_pass": sum(x == -1 for x in paired),
        "adapter_only_pass": sum(x == 1 for x in paired),
        "both_pass": sum(r["base"]["strict"] and r["adapter"]["strict"] for r in per_sample),
        "neither_pass": sum(not r["base"]["strict"] and not r["adapter"]["strict"] for r in per_sample),
    }
    (args.output / "ifeval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
