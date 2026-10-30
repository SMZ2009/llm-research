# Storage policy

Routine setup, verification, and training do **not** archive the Python environment, Hugging Face cache, base model, or full WSL distribution. The base model stays in one WSL cache. `reports/local/`, raw data, checkpoints, and adapter binaries are ignored by Git and are not uploaded to the public repository.

For each completed training run, preserve the small config, manifest, metrics, and final adapter. A running job may keep one resumable checkpoint; after a successful export the temporary checkpoint directory is removed. This prevents completed runs from retaining duplicate model or optimizer states. Training data copied to WSL is reused by dataset ID and checksum rather than recopied for every run.

Use Git commits/tags, `uv.lock`, the pinned model revision, and per-run manifests for audit and version rollback. A full runtime backup should be made only for a specific offline recovery need, as a manually requested one-time artifact with an explicit storage location and retention decision. The initial 2026-10-30 runtime backup was verified, then retired to reclaim disk space; its historical SHA-256 remains in the private local audit record.
