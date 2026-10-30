"""Compare response token NLL on the untouched Dolly test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    completed = {}
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            record = json.loads(line)
            completed[record["id"]] = record
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            prompt = tokenizer.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            completion = row["messages"][-1]["content"] + "<|im_end|>"
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:args.max_length]
            if len(prompt_ids) >= len(full_ids):
                record = {"id": row["id"], "tokens": 0, "nll_sum": None,
                          "excluded": "prompt fills max_length"}
            else:
                ids = torch.tensor([full_ids], device="cuda")
                with torch.inference_mode():
                    logits = model(input_ids=ids).logits[:, :-1, :].float()
                targets = ids[:, 1:]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
                )
                response_losses = losses[len(prompt_ids) - 1:]
                record = {"id": row["id"], "tokens": len(response_losses),
                          "nll_sum": float(response_losses.sum()), "excluded": None}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(rows):
                print(f"{index}/{len(rows)} scored", flush=True)
    valid = [json.loads(line) for line in args.output.open(encoding="utf-8")]
    total_tokens = sum(r["tokens"] for r in valid)
    nll = sum(r["nll_sum"] for r in valid if r["nll_sum"] is not None) / total_tokens
    print(json.dumps({"samples": len(valid), "tokens": total_tokens,
                      "excluded": sum(r["excluded"] is not None for r in valid),
                      "response_token_nll": nll, "perplexity": math.exp(nll)}))


if __name__ == "__main__":
    main()
