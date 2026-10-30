# Reproducibility

The project pins the Python patch version in `.python-version`, Python dependencies and hashes in `uv.lock`, and the Qwen3-0.6B Hub commit in `configs/sft_lora_non_thinking.yaml`. Run `bash scripts/setup_wsl.sh` and `bash scripts/verify_wsl.sh` from WSL2 to recreate and verify the environment. The scripts derive the project path from their location and keep the Python environment and model cache under `${LLM_RESEARCH_RUNTIME:-$HOME/llm-research-runtime}` on the WSL Linux file system.

Each future experiment must record the Git commit, model revision, data checksums, lockfile SHA-256, configuration, seed, GPU, package versions, and train/validation metrics. Keep the test split held out until the final comparison. The training entry point requires a task, source URI, license, and all three fixed splits before it starts.

For source rollback, check out a prior Git commit or tag. For dependency rollback, run `uv sync --frozen --extra dev --python 3.11.18` in that revision. The pinned model revision can be downloaded again from Hugging Face. A machine-local setup audit from 2026-10-30 remains in the original workspace but is intentionally excluded from the public repository because it contains host-specific paths and identifiers.
