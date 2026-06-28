\# Redrob Ranker



AI candidate ranking pipeline for the Redrob Data \& AI Challenge.



The system ranks candidates for a Senior AI Engineer role using a staged pipeline:



1\. Parse candidate JSONL into normalized Parquet tables.

2\. Apply structural and JD-specific qualifier gates.

3\. Compute availability multiplier from Redrob behavioral signals.

4\. Compute semantic JD-fit from career history, with recency weighting.

5\. Apply trap / suspicious-profile penalties.

6\. Produce a ranked top-100 candidate output.



Initial exploration was done in Jupyter notebooks. The final pipeline is being organized into reproducible scripts under `scripts/`.



\## Current status



\- Steps 1–2 completed.

\- Step 3 availability multiplier completed.

\- Step 5 semantic scoring under implementation.



\## Reproduction plan



```bash

python scripts/step1\_2\_filters.py

python scripts/step3\_availability.py

python scripts/precompute\_embeddings.py

python scripts/rank.py

