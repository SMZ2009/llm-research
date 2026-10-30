import json

import pytest

from llm_research.data_contract import validate_disjoint_splits, validate_jsonl


def test_rejects_thinking_content(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "<think>secret</think>Hello"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="thinking content is forbidden"):
        validate_jsonl(path)


def test_rejects_cross_split_conversation_overlap(tmp_path):
    paths = {}
    for split, record_id in (("train", "train_1"), ("test", "test_1")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": record_id,
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        validate_jsonl(path)
        paths[split] = path
    with pytest.raises(ValueError, match="conversation overlaps"):
        validate_disjoint_splits(paths)
