"""Report training split truncation and response masking coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    report = {}
    for split in ("train", "validation", "test"):
        count = 0
        full_over = 0
        prompt_over = 0
        total_response_tokens = 0
        retained_response_tokens = 0
        for line in (args.split_dir / f"{split}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            completion = row["messages"][-1]["content"] + "<|im_end|>"
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            full_len = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
            count += 1
            full_over += full_len > args.max_length
            prompt_over += prompt_len >= args.max_length
            response_len = full_len - prompt_len
            total_response_tokens += response_len
            retained_response_tokens += max(0, min(full_len, args.max_length) - prompt_len)
        report[split] = {
            "rows": count,
            "full_sequence_over_limit": full_over,
            "prompt_fills_limit": prompt_over,
            "response_tokens": total_response_tokens,
            "retained_response_tokens": retained_response_tokens,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
