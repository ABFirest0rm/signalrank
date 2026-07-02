# SignalRank — Redrob Candidate Ranking Submission

SignalRank is a hybrid candidate-ranking system built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

The system ranks candidates for the **Senior AI Engineer – Founding Team** role using a combination of semantic retrieval, career-history analysis, behavioral availability signals, and deterministic CPU-only scoring.

---

# Repository Structure

```
scripts/                  Ranking pipeline
outputs/                  Final submission CSV
artifacts/                Precomputed embeddings (download separately)
dataset/artifacts/        Processed parquet artifacts (download separately)
docs/                     Architecture notes
```

---

# Final Submission

The validated submission file is:

```text
outputs/SignalRank.csv
```

---

# Reproducing the Submission

## 1. Clone the repository

```bash
git clone https://github.com/ABFirest0rm/signalrank.git
cd signalrank
```

## 2. Install dependencies

```bash
pip install -r requirements-rank.txt
```

## 3. Download precomputed artifacts

Artifacts are provided separately because GitHub size limits prevent storing the embedding files in the repository.

Download:

https://drive.google.com/file/d/187XhR7Qs7VD4c_F6hfIsKh8LiIFWsCRp/view?usp=sharing

Extract the archive into the repository root so the structure becomes:

```
artifacts/
dataset/artifacts/
```

---

## 4. Generate the submission

```bash
python scripts/rank.py \
    --data-dir dataset/artifacts \
    --artifacts-dir artifacts \
    --output-dir outputs \
    --team-id SignalRank
```

The ranking step:

- uses only local artifacts
- performs no network calls
- requires no GPU
- completes in approximately **4.7 seconds** on the original test machine

---

## 5. Validate the output

```bash
python scripts/validate.py outputs/SignalRank.csv
```

---

# Sandbox Demo

Google Colab:

https://colab.research.google.com/drive/1U-wI02xjG6RBt3UBdvnvmdgCtzTuL50G?usp=sharing

---

# Methodology

SignalRank follows a two-stage architecture.

**Offline preprocessing**

- Parse candidate profiles
- Apply filtering and availability scoring
- Generate semantic embeddings using **BAAI/bge-base-en-v1.5**
- Store lightweight `.npy` and `.parquet` artifacts

**Online ranking (CPU-only)**

- Load precomputed artifacts
- Compute semantic similarity using NumPy matrix multiplication
- Apply must-have, domain, availability, seniority and behavioral multipliers
- Generate deterministic candidate reasoning
- Produce a validator-ready top-100 CSV

The expensive embedding stage is completely separated from the ranking stage, allowing reproducible CPU-only execution within the competition constraints.

---

# Technologies

- Python
- NumPy
- Pandas
- Polars
- SciPy
- Sentence Transformers
- BAAI/bge-base-en-v1.5