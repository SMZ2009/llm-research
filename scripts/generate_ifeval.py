"""Generate resumable IFEval responses with a fixed inference protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            row = json.loads(line)
            completed[row["key"]] = row
    if len(completed) != len({x["key"] for x in completed.values()}):
        raise RuntimeError("Duplicate response keys")
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, 1):
            if row["key"] in completed:
                continue
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            started = time.monotonic()
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_tokens = tokens[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            record = {
                "key": row["key"],
                "prompt": row["prompt"],
                "response": response,
                "new_tokens": int(response_tokens.numel()),
                "seconds": round(time.monotonic() - started, 3),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(rows):
                print(f"{index}/{len(rows)} generated", flush=True)


if __name__ == "__main__":
    main()
