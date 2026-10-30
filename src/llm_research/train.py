"""Dataset-gated, auditable non-thinking LoRA SFT entry point.

This module is prepared but must not run until a dataset and task are approved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from llm_research.data_contract import sha256_file, validate_disjoint_splits, validate_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(
    os.environ.get("LLM_RESEARCH_RUNTIME", Path.home() / "llm-research-runtime")
)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping")
    model = config["model"]
    dataset = config["dataset"]
    training = config["training"]
    if model["id"] != "Qwen/Qwen3-0.6B" or model["enable_thinking"] is not False:
        raise ValueError("Only Qwen3-0.6B non-thinking SFT is configured")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("A pinned 40-character model revision is required")
    if training["quantization"] != "none":
        raise ValueError("Only BF16 LoRA is configured; QLoRA needs a reviewed configuration")
    if not dataset["id"] or not re.fullmatch(r"[a-z0-9_]+", dataset["id"]):
        raise ValueError("Set dataset.id to a lowercase snake_case identifier")
    for key in ("task", "source_uri", "license"):
        if not dataset[key]:
            raise ValueError(f"Dataset is not approved: dataset.{key} is unset")
    for key in ("train_jsonl", "validation_jsonl", "test_jsonl"):
        if not dataset[key]:
            raise ValueError(f"Dataset is not approved: dataset.{key} is unset")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/sft_lora_non_thinking.yaml"
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    dataset_cfg = config["dataset"]
    training = config["training"]

    split_paths = {
        split: Path(dataset_cfg[f"{split}_jsonl"])
        for split in ("train", "validation", "test")
    }
    checks = {split: validate_jsonl(path) for split, path in split_paths.items()}
    validate_disjoint_splits(split_paths)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_id = f"qwen3_0p6b_sft_lora_{dataset_cfg['id']}_s{training['seed']}_{stamp}"
    run_dir = RUNTIME_ROOT / "runs" / run_id
    input_dir = RUNTIME_ROOT / "data" / dataset_cfg["id"]
    input_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    local_files = {}
    for split, source in checks.items():
        target = input_dir / f"{split}_{source['sha256'][:12]}.jsonl"
        if not target.exists():
            shutil.copy2(source["path"], target)
        if sha256_file(target) != source["sha256"]:
            raise RuntimeError(f"Copied {split} data failed checksum verification")
        local_files[split] = str(target)

    snapshot = snapshot_download(
        repo_id=config["model"]["id"],
        revision=config["model"]["revision"],
        cache_dir=RUNTIME_ROOT / "huggingface/hub",
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True
    )
    sample_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "sample"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<think>\n\n</think>" not in sample_prompt:
        raise RuntimeError("Qwen3 non-thinking chat template validation failed")

    manifest = {
        "run_id": run_id,
        "model": config["model"],
        "dataset": checks,
        "lockfile_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "mode": "non_thinking",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.config, run_dir / "config.yaml")

    corpus = load_dataset(
        "json",
        data_files={split: local_files[split] for split in ("train", "validation")},
        cache_dir=str(RUNTIME_ROOT / "datasets_cache"),
    )
    lora = LoraConfig(
        r=training["lora_r"],
        lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"],
        target_modules=training["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args_sft = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        logging_dir=str(run_dir / "logs"),
        max_length=training["max_length"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        bf16=training["bf16"],
        gradient_checkpointing=training["gradient_checkpointing"],
        assistant_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to="tensorboard",
        seed=training["seed"],
        data_seed=training["seed"],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args_sft,
        train_dataset=corpus["train"],
        eval_dataset=corpus["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    result = trainer.train()
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    export_dir = PROJECT_ROOT / "models" / "adapters" / run_id
    shutil.copytree(adapter_dir, export_dir)
    for source_file in adapter_dir.rglob("*"):
        if source_file.is_file():
            exported_file = export_dir / source_file.relative_to(adapter_dir)
            if sha256_file(source_file) != sha256_file(exported_file):
                raise RuntimeError(f"Adapter export failed checksum verification: {source_file}")
    public_run_dir = PROJECT_ROOT / "experiments" / run_id
    public_run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config.yaml", "manifest.json", "metrics.json"):
        shutil.copy2(run_dir / name, public_run_dir / name)
    shutil.rmtree(run_dir / "checkpoints")
    shutil.rmtree(adapter_dir)
    print(f"Adapter: {export_dir}")
    print(f"Run manifest: {public_run_dir}")


if __name__ == "__main__":
    main()
