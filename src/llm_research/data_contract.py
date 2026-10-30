"""Validate future SFT data before allowing a training run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}
FORBIDDEN_REASONING_MARKERS = ("<think>", "</think>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_jsonl(path: Path) -> dict[str, object]:
    """Require complete, unique, non-thinking conversations in UTF-8 JSONL."""
    if not path.is_file():
        raise FileNotFoundError(path)

    ids: set[str] = set()
    content_hashes: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            record_id = row.get("id") if isinstance(row, dict) else None
            messages = row.get("messages") if isinstance(row, dict) else None
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing string id")
            if record_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id}")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: messages must contain a dialogue")
            if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: final message must be assistant")
            if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
                raise ValueError(f"{path}:{line_number}: a user message is required")
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
                    raise ValueError(f"{path}:{line_number}: unsupported role")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_number}: empty/non-string content")
                if "reasoning_content" in message or any(
                    marker in content for marker in FORBIDDEN_REASONING_MARKERS
                ):
                    raise ValueError(f"{path}:{line_number}: thinking content is forbidden")
            fingerprint = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fingerprint in content_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate conversation")
            ids.add(record_id)
            content_hashes.add(fingerprint)
            count += 1

    if count == 0:
        raise ValueError(f"{path}: no records")
    return {"path": str(path.resolve()), "records": count, "sha256": sha256_file(path)}


def validate_disjoint_splits(paths: dict[str, Path]) -> None:
    """Reject ID or conversation overlap across fixed train/validation/test splits."""
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for split, path in paths.items():
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                record_id = row["id"]
                fingerprint = hashlib.sha256(
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if record_id in seen_ids:
                    raise ValueError(f"id {record_id} overlaps {seen_ids[record_id]} and {split}")
                if fingerprint in seen_content:
                    raise ValueError(f"conversation overlaps {seen_content[fingerprint]} and {split}")
                seen_ids[record_id] = split
                seen_content[fingerprint] = split
