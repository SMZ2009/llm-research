"""Select a deterministic post hoc IFEval diagnostic subset from paired failures."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: Path, key: str) -> dict:
    return {row[key]: row for row in (json.loads(line) for line in path.open(encoding="utf-8"))}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--per-stratum", type=int, default=15)
    args = parser.parse_args()
    inputs = load(args.input, "key")
    scores = load(args.scores, "key")
    base = load(args.base, "key")
    adapter = load(args.adapter, "key")
    if not (inputs.keys() == scores.keys() == base.keys() == adapter.keys()):
        raise ValueError("Inputs, scores, and outputs must match")
    capped = sorted(
        key for key, row in scores.items()
        if row["base"]["strict"] and not row["adapter"]["strict"]
        and adapter[key]["new_tokens"] == 512
    )
    uncapped = sorted(
        key for key, row in scores.items()
        if row["base"]["strict"] and not row["adapter"]["strict"]
        and adapter[key]["new_tokens"] < 512
    )
    rng = random.Random(args.seed)
    selected = {
        "capped_base_only": sorted(rng.sample(capped, args.per_stratum)),
        "uncapped_base_only": sorted(rng.sample(uncapped, args.per_stratum)),
    }
    keys = sorted(selected["capped_base_only"] + selected["uncapped_base_only"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "input.jsonl", [inputs[key] for key in keys])
    write_jsonl(args.output_dir / "base_512.jsonl", [base[key] for key in keys])
    write_jsonl(args.output_dir / "adapter_512.jsonl", [adapter[key] for key in keys])
    manifest = {
        "selection": "post hoc: base passes, adapter fails; 15 capped and 15 uncapped selected uniformly without replacement within each stratum",
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "eligible": {"capped": len(capped), "uncapped": len(uncapped)},
        "selected": selected,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
