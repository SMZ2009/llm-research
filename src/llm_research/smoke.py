"""Read-only GPU, model and non-thinking inference smoke test."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in WSL")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f"Expected RTX 5060 compute capability 12.0, got {capability}")
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError("CUDA BF16 matrix multiplication produced nonfinite values")

    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    cached_snapshot = cache_root / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION
    required_files = ("config.json", "model.safetensors", "tokenizer.json")
    if all((cached_snapshot / name).is_file() for name in required_files):
        model_dir = str(cached_snapshot)
    else:
        model_dir = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello in one short sentence."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<think>\n\n</think>" not in prompt:
        raise RuntimeError("The prompt is not the expected Qwen3 non-thinking template")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    if "<think>" in answer or "</think>" in answer:
        raise RuntimeError("Model generated thinking tags under non-thinking mode")
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cuda_bf16_matmul_ok": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_cache_path": model_dir,
        "enable_thinking": False,
        "prompt_has_empty_think_block": True,
        "sample_answer": answer.strip(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
