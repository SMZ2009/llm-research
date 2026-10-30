# Imported sources

Only the original inputs for the cancelled Dolly 15k experiment are retained. No training run, split, adapter, checkpoint, or evaluation result is retained here.

| Import | Local path (Git ignored) | Provenance manifest |
| --- | --- | --- |
| Databricks Dolly 15k, 15,011 JSONL rows | `data/raw/dolly15k/databricks-dolly-15k.jsonl` | [`dolly15k_source.json`](dolly15k_source.json) |
| Google IFEval, 541 prompts and official evaluator files | `data/raw/ifeval_official/instruction_following_eval/` | [`ifeval_source.json`](ifeval_source.json) |

Dolly is pinned to Hugging Face dataset revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` and marked CC-BY-SA-3.0. IFEval is pinned to Google Research commit `697787053593d98a82f0ffb008463f0eab81ff85`. The manifests give source URLs and SHA-256 values so the ignored local files can be reimported and verified.
