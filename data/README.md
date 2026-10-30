# Dataset contract

No dataset is included yet. A future task and source must be identified with its URI, license, and checksum before import.

Use UTF-8 JSONL with one record per conversation:

```json
{"id":"example_000001","messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Keep raw, processed, and fixed train/validation/test splits separate. Do not put secrets or private data in Git. The training path is exclusively non-thinking mode; datasets containing `<think>` sections must be rejected or explicitly transformed and reviewed.
