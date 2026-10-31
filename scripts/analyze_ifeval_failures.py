"""Summarize paired IFEval errors and output lengths without re-scoring."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def load(path: Path, key: str) -> dict:
    return {row[key]: row for row in (json.loads(line) for line in path.open(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = load(args.input, "key")
    scores = load(args.scores, "key")
    responses = {"base": load(args.base, "key"), "adapter": load(args.adapter, "key")}
    if not (inputs.keys() == scores.keys() == responses["base"].keys() == responses["adapter"].keys()):
        raise ValueError("Paired input, scores, and responses must have identical keys")
    cases = collections.defaultdict(list)
    categories = collections.defaultdict(lambda: {"n": 0, "base_pass": 0, "adapter_pass": 0})
    for key, row in scores.items():
        base_pass = row["base"]["strict"]
        adapter_pass = row["adapter"]["strict"]
        label = ("both_pass" if base_pass and adapter_pass else
                 "base_only" if base_pass else
                 "adapter_only" if adapter_pass else "neither_pass")
        cases[label].append(key)
        for category in {name.split(":")[0] for name in inputs[key]["instruction_id_list"]}:
            bucket = categories[category]
            bucket["n"] += 1
            bucket["base_pass"] += base_pass
            bucket["adapter_pass"] += adapter_pass
    summary = {"groups": {}, "categories": {}}
    for label, keys in cases.items():
        summary["groups"][label] = {
            "n": len(keys),
            "base_mean_new_tokens": statistics.mean(responses["base"][k]["new_tokens"] for k in keys),
            "adapter_mean_new_tokens": statistics.mean(responses["adapter"][k]["new_tokens"] for k in keys),
            "base_capped": sum(responses["base"][k]["new_tokens"] == 512 for k in keys),
            "adapter_capped": sum(responses["adapter"][k]["new_tokens"] == 512 for k in keys),
            "sample_keys": keys[:5],
        }
    for name, bucket in categories.items():
        summary["categories"][name] = {
            **bucket,
            "base_accuracy": bucket["base_pass"] / bucket["n"],
            "adapter_accuracy": bucket["adapter_pass"] / bucket["n"],
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
