# SignalRank Architecture

SignalRank is a hybrid candidate-ranking pipeline for the Redrob Senior AI Engineer challenge.

## Pipeline

1. Parse candidate JSONL into structured parquet artifacts.
2. Apply survivorship and availability filters.
3. Compute behavioral availability scores.
4. Precompute semantic embeddings offline.
5. Run a CPU-only final ranker using local artifacts.
6. Write a validator-ready top-100 CSV.

## Final ranking phase

The final ranker loads only local parquet and numpy artifacts. It performs no network calls, no hosted LLM calls, and no GPU calls.

Required local artifacts include job embeddings, query embeddings, job metadata, candidate base data, survivor data, and availability scores.

## Submission output

The final submission file is outputs/SignalRank.csv.

It is validated with:

python scripts\validate.py outputs\SignalRank.csv

## Reproduction note

Large generated artifacts are intentionally excluded from Git and should be distributed separately as a zip file.
