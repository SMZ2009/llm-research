"""Audit the exact prompt/completion boundary used by the Dolly SFT entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    counts = {"rows": 0, "prompt_prefix_mismatch": 0, "zero_completion_tokens": 0,
              "truncated_before_eos": 0, "eos_retained": 0, "empty_think_preamble": 0,
              "duplicate_eos_in_source": 0}
    examples = {"prefix_mismatch": [], "zero_completion_tokens": []}
    for line in args.train.open(encoding="utf-8"):
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        completion = row["messages"][-1]["content"] + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        counts["rows"] += 1
        counts["empty_think_preamble"] += "<think>\n\n</think>" in prompt
        counts["duplicate_eos_in_source"] += completion.endswith("<|im_end|><|im_end|>")
        if full_ids[:len(prompt_ids)] != prompt_ids:
            counts["prompt_prefix_mismatch"] += 1
            if len(examples["prefix_mismatch"]) < 5:
                examples["prefix_mismatch"].append(row["id"])
        if len(prompt_ids) >= args.max_length:
            counts["zero_completion_tokens"] += 1
            if len(examples["zero_completion_tokens"]) < 5:
                examples["zero_completion_tokens"].append(row["id"])
        if full_ids[-1] != tokenizer.eos_token_id:
            raise RuntimeError(f"Completion does not end in EOS: {row['id']}")
        if len(full_ids) > args.max_length:
            counts["truncated_before_eos"] += 1
        else:
            counts["eos_retained"] += 1
    report = {"eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
              "max_length": args.max_length, "counts": counts, "examples": examples}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
