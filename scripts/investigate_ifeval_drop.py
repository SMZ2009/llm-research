"""Audit paired IFEval outputs and Dolly training distribution after the run."""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path


def load(path: Path, key: str) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len(rows) != len({row[key] for row in rows}):
        raise ValueError(f"Duplicate {key} in {path}")
    return {row[key]: row for row in rows}


def repetition_fraction(response: str, first_units: int | None = None) -> float:
    words = re.findall(r"\w+|[^\w\s]", response.casefold())
    if first_units is not None:
        words = words[:first_units]
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dolly", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = [json.loads(line) for line in args.raw_dolly.open(encoding="utf-8")]
    train = [json.loads(line) for line in args.train.open(encoding="utf-8")]
    categories = collections.Counter(original[int(row["id"].split("_")[1])]["category"] for row in train)
    constraint_pattern = re.compile(
        r"\b(exactly|at least|at most|no commas?|without commas?|"
        r"uppercase|lowercase|capital letters|json|paragraphs?|bullet points?|"
        r"only|must start|must end|do not include|don't include)\b", re.IGNORECASE
    )
    train_constraint_hits = sum(
        bool(constraint_pattern.search(row["messages"][0]["content"])) for row in train
    )
    specific_patterns = {
        "exactly": r"\bexactly\b",
        "at_least": r"\bat least\b",
        "no_comma": r"\b(?:no|without) commas?\b",
        "uppercase": r"\b(?:all caps|capital letters|uppercase)\b",
        "json": r"\bjson\b",
        "start_end": r"\b(?:start|end) (?:with|in)\b",
    }
    specific_hits = {
        name: sum(bool(re.search(pattern, row["messages"][0]["content"], re.IGNORECASE))
                  for row in train)
        for name, pattern in specific_patterns.items()
    }
    train_responses = [row["messages"][-1]["content"] for row in train]
    train_repetition = [repetition_fraction(response) for response in train_responses]

    inputs = load(args.ifeval, "key")
    scores = load(args.scores, "key")
    base = load(args.base, "key")
    adapter = load(args.adapter, "key")
    if not (inputs.keys() == scores.keys() == base.keys() == adapter.keys()):
        raise ValueError("IFEval inputs, scores, and outputs have different keys")

    strata = collections.defaultdict(list)
    categories_ifeval = collections.defaultdict(list)
    by_instruction_count = collections.defaultdict(list)
    diagnostics = []
    base_instruction_pass = 0
    adapter_instruction_pass = 0
    instruction_count = 0
    for key in inputs:
        b = base[key]
        a = adapter[key]
        s = scores[key]
        b_cap = b["new_tokens"] == 512
        a_cap = a["new_tokens"] == 512
        stratum = ("both_capped" if b_cap and a_cap else
                   "adapter_only_capped" if a_cap else
                   "base_only_capped" if b_cap else "neither_capped")
        row = {
            "key": key,
            "base_strict": s["base"]["strict"],
            "adapter_strict": s["adapter"]["strict"],
            "base_loose": s["base"]["loose"],
            "adapter_loose": s["adapter"]["loose"],
            "base_tokens": b["new_tokens"],
            "adapter_tokens": a["new_tokens"],
            "base_repetition_fraction": repetition_fraction(b["response"]),
            "adapter_repetition_fraction": repetition_fraction(a["response"]),
            "base_first_100_repetition_fraction": repetition_fraction(b["response"], 100),
            "adapter_first_100_repetition_fraction": repetition_fraction(a["response"], 100),
            "stratum": stratum,
        }
        diagnostics.append(row)
        strata[stratum].append(row)
        by_instruction_count[len(inputs[key]["instruction_id_list"])].append(row)
        instruction_count += len(inputs[key]["instruction_id_list"])
        base_instruction_pass += sum(s["base"]["per_instruction_strict"])
        adapter_instruction_pass += sum(s["adapter"]["per_instruction_strict"])
        for category in {name.split(":")[0] for name in inputs[key]["instruction_id_list"]}:
            categories_ifeval[category].append(row)

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "base_strict_pass": sum(r["base_strict"] for r in rows),
            "adapter_strict_pass": sum(r["adapter_strict"] for r in rows),
            "base_loose_pass": sum(r["base_loose"] for r in rows),
            "adapter_loose_pass": sum(r["adapter_loose"] for r in rows),
            "base_only_pass": sum(r["base_strict"] and not r["adapter_strict"] for r in rows),
            "adapter_only_pass": sum(r["adapter_strict"] and not r["base_strict"] for r in rows),
            "base_mean_tokens": statistics.mean(r["base_tokens"] for r in rows),
            "adapter_mean_tokens": statistics.mean(r["adapter_tokens"] for r in rows),
            "base_mean_repetition_fraction": statistics.mean(r["base_repetition_fraction"] for r in rows),
            "adapter_mean_repetition_fraction": statistics.mean(r["adapter_repetition_fraction"] for r in rows),
            "base_mean_first_100_repetition_fraction": statistics.mean(
                r["base_first_100_repetition_fraction"] for r in rows
            ),
            "adapter_mean_first_100_repetition_fraction": statistics.mean(
                r["adapter_first_100_repetition_fraction"] for r in rows
            ),
            "base_repetition_over_0p2": sum(r["base_repetition_fraction"] > 0.2 for r in rows),
            "adapter_repetition_over_0p2": sum(r["adapter_repetition_fraction"] > 0.2 for r in rows),
        }

    report = {
        "dolly_train": {
            "n": len(train),
            "categories": dict(sorted(categories.items())),
            "constraint_lexeme_hits": train_constraint_hits,
            "constraint_lexeme_rule": constraint_pattern.pattern,
            "specific_constraint_lexeme_hits": specific_hits,
            "mean_response_words": statistics.mean(len(response.split()) for response in train_responses),
            "mean_response_repetition_fraction": statistics.mean(train_repetition),
            "response_repetition_over_0p2": sum(value > 0.2 for value in train_repetition),
            "note": "Lexeme hits are a rough proxy, not verified IFEval-style constraints",
        },
        "ifeval_overall": summarize(diagnostics),
        "ifeval_instruction_level": {
            "n": instruction_count,
            "base_strict_pass": base_instruction_pass,
            "adapter_strict_pass": adapter_instruction_pass,
        },
        "ifeval_by_cap": {name: summarize(rows) for name, rows in sorted(strata.items())},
        "ifeval_by_instruction_count": {
            str(count): summarize(rows) for count, rows in sorted(by_instruction_count.items())
        },
        "ifeval_by_category": {
            name: summarize(rows) for name, rows in sorted(categories_ifeval.items())
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
