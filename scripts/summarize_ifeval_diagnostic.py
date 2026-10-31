"""Summarize preselected post hoc decoding checks without changing the main run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from investigate_ifeval_drop import repetition_fraction


def load(path: Path) -> dict:
    return {row["key"]: row for row in (json.loads(line) for line in path.open(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dir / "selection.json").read_text(encoding="utf-8"))
    original = load(args.dir / "adapter_512.jsonl")
    outputs = {
        name: load(args.dir / f"{name}.jsonl")
        for name in ("adapter_512", "greedy_1024", "sampling_512")
    }
    scores = {
        name: load(args.dir / f"{name}_score/ifeval_per_sample.jsonl")
        for name in outputs
    }
    all_keys = set(original)
    if any(set(value) != all_keys for value in [*outputs.values(), *scores.values()]):
        raise ValueError("Diagnostic output or scoring keys differ")
    result = {"post_hoc_selection": manifest, "conditions": {}}
    for name, by_key in outputs.items():
        cap = 1024 if name == "greedy_1024" else 512
        result["conditions"][name] = {}
        for stratum, keys in manifest["selected"].items():
            rows = [by_key[key] for key in keys]
            score_rows = [scores[name][key] for key in keys]
            result["conditions"][name][stratum] = {
                "n": len(keys),
                "strict_pass": sum(row["adapter"]["strict"] for row in score_rows),
                "loose_pass": sum(row["adapter"]["loose"] for row in score_rows),
                "capped": sum(row["new_tokens"] == cap for row in rows),
                "mean_new_tokens": statistics.mean(row["new_tokens"] for row in rows),
                "mean_repetition_fraction": statistics.mean(
                    repetition_fraction(row["response"]) for row in rows
                ),
                "sum_generation_seconds": sum(row["seconds"] for row in rows),
                "passed_keys": [key for key in keys if scores[name][key]["adapter"]["strict"]],
            }
    result["greedy_1024_text_prefix_match"] = sum(
        outputs["greedy_1024"][key]["response"].startswith(original[key]["response"])
        for key in all_keys
    )
    result["greedy_1024_text_prefix_total"] = len(all_keys)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
