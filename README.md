# SignalRank — Redrob Candidate Ranking Submission

SignalRank is a hybrid candidate-ranking system built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

The system ranks the top 100 candidates for the released **Senior AI Engineer — Founding Team** job description. It combines semantic matching, must-have skill scoring, career-history attribution, behavioral availability signals, honeypot-style profile checks, and deterministic CSV generation.

---

## Submission Assets

| Asset | Link / Path |
|---|---|
| GitHub repository | https://github.com/ABFirest0rm/signalrank |
| Sandbox / demo notebook | https://colab.research.google.com/drive/1U-wI02xjG6RBt3UBdvnvmdgCtzTuL50G?usp=sharing |
| Precomputed artifact zip | https://drive.google.com/file/d/187XhR7Qs7VD4c_F6hfIsKh8LiIFWsCRp/view?usp=sharing |
| Final validated CSV | `outputs/SignalRank.csv` |

---

## Final Submission File

The final validated ranking file is:

```text
outputs/SignalRank.csv
```

It contains exactly the required columns:

```text
candidate_id,rank,score,reasoning
```

Validation command:

```bash
python scripts/validate.py outputs/SignalRank.csv
```

The submitted file should remain a `.csv` file unless the portal explicitly prevents CSV upload.

---

## Repository Structure

```text
scripts/
  parse_and_filter.py          # Parse raw candidate JSONL and apply hard filters
  compute_availability.py      # Compute behavioral / availability features
  precompute_embeddings.py     # One-time embedding precompute step
  rank.py                      # CPU-only final ranking step
  validate.py                  # Submission validator

outputs/
  SignalRank.csv               # Final validated top-100 ranking

docs/
  architecture.md              # Architecture notes

artifacts/                     # Download separately
  job_embeddings.npy
  query_embeddings.npy
  job_meta.parquet

dataset/artifacts/             # Download separately
  availability_scores.parquet
  candidate_base.parquet
  candidate_education.parquet
  candidate_jobs.parquet
  candidate_redrob.parquet
  candidate_skills.parquet
  step1_dropped.parquet
  step1_survivors.parquet
  step2_dropped.parquet
  survivors.parquet
```

Large generated artifacts are intentionally excluded from Git and provided separately.

---

## Quick Reproduction: Final CPU-Only Ranking Step

This is the reproducible ranking step used to regenerate the final submission CSV from precomputed local artifacts.

### 1. Clone the repository

```bash
git clone https://github.com/ABFirest0rm/signalrank.git
cd signalrank
```

### 2. Install ranking dependencies

```bash
pip install -r requirements-rank.txt
```

### 3. Download precomputed artifacts

Download the artifact zip:

```text
https://drive.google.com/file/d/187XhR7Qs7VD4c_F6hfIsKh8LiIFWsCRp/view?usp=sharing
```

Extract it into the repository root so the directory layout becomes:

```text
artifacts/
dataset/artifacts/
```

Expected key files:

```text
artifacts/job_embeddings.npy
artifacts/query_embeddings.npy
artifacts/job_meta.parquet

dataset/artifacts/availability_scores.parquet
dataset/artifacts/candidate_base.parquet
dataset/artifacts/survivors.parquet
```

### 4. Generate the submission CSV

```bash
python scripts/rank.py \
  --data-dir dataset/artifacts \
  --artifacts-dir artifacts \
  --output-dir outputs \
  --team-id SignalRank
```

This writes:

```text
outputs/SignalRank.csv
outputs/top100_ranked_diagnostic.csv
```

### 5. Validate the generated CSV

```bash
python scripts/validate.py outputs/SignalRank.csv
```

Expected result:

```text
Validation complete: all required format checks passed.
```

---

## Single Command for Stage-3 Ranking Reproduction

After dependencies and artifacts are present, the final judged ranking step is:

```bash
python scripts/rank.py --data-dir dataset/artifacts --artifacts-dir artifacts --output-dir outputs --team-id SignalRank
```

This command regenerates the final submission CSV using only local artifacts.

---

## Compute Characteristics

The final ranking step is designed to satisfy the hackathon reproduction constraints:

| Constraint | SignalRank final ranking step |
|---|---|
| Runtime | Approximately 4.67 seconds in local testing |
| Compute | CPU-only |
| GPU | Not used during ranking |
| Network | No external API calls |
| Hosted LLM APIs | Not used |
| Intermediate state | Local `.npy` and `.parquet` artifacts |

Runtime observed locally:

```text
RANKING PHASE: loading local artifacts only — no model, no network, no GPU calls.
job_embeddings   : (285346, 768) / float32
query_embeddings : (11, 768) / float32
job_meta rows    : 285,346
Score matrix: (285346, 11) computed in 0.37s
```

The expensive embedding stage is fully decoupled from the timed CPU-only ranking step.

---

## Full Pipeline Overview

SignalRank uses a two-phase architecture.

### Phase 1 — Offline preprocessing and embedding generation

These steps are run once before final ranking:

```bash
python scripts/parse_and_filter.py
python scripts/compute_availability.py
python scripts/precompute_embeddings.py
```

The preprocessing phase:

1. Parses the released `candidates.jsonl`.
2. Applies profile-quality and survivorship filters.
3. Computes behavioral and availability features.
4. Builds enriched job-history text.
5. Embeds candidate job-history text and JD-derived query probes using `BAAI/bge-base-en-v1.5`.
6. Writes compact `.parquet` and `.npy` artifacts.

This phase may use GPU for embedding precomputation and is not the timed final ranking step.

### Phase 2 — CPU-only ranking

The final ranker:

1. Loads precomputed embeddings and parquet artifacts.
2. Computes semantic similarity using NumPy matrix multiplication.
3. Scores four must-have JD dimensions.
4. Applies nice-to-have, domain, hands-on, availability, seniority, job-hopper, and trap multipliers.
5. Produces a deterministic top-100 ranking.
6. Generates candidate-specific reasoning strings.
7. Writes a validator-ready CSV.

---

## Methodology

SignalRank treats candidate fit as a set of auditable signals rather than one opaque blended score.

The JD is decomposed into targeted semantic probes:

- Production retrieval / RAG systems
- Hands-on vector database and search infrastructure
- Ranking and evaluation rigor
- Production implementation experience
- Nice-to-have adjacent AI/ML experience
- Domain fit signals
- Hands-on engineering signals

Candidate job-history entries are embedded once offline. During ranking, the system computes similarity between job-history embeddings and JD query embeddings, then aggregates evidence at the candidate level.

The core must-have score uses a geometric mean across key must-have dimensions so a candidate cannot rank highly by matching only one axis while missing the others.

Final score is produced through transparent multiplicative scoring:

```text
final_score =
  must_have_score
  × nice_to_have_bonus
  × domain_multiplier
  × hands_on_multiplier
  × availability_multiplier
  × seniority_multiplier
  × job_hopper_multiplier
  × trap_multiplier
```

This keeps the scoring interpretable and makes failure modes easier to inspect.

---

## Reasoning Generation

Each ranked candidate receives a reasoning string generated from the same scored features used by the ranker.

The reasoning is:

- deterministic
- candidate-specific
- tied to observed profile signals
- connected to JD requirements
- not generated by a hosted LLM
- not manually edited after ranking

This avoids hallucinated skills, employers, or experience claims.

---

## Honeypot / Trap Handling

SignalRank includes defensive profile-quality checks and a trap multiplier to reduce the chance of ranking impossible or suspicious profiles.

The system checks for signals such as:

- overlapping full-time roles
- claimed experience exceeding plausible career span
- suspiciously low-detail or padded job descriptions
- consulting-only career patterns
- weak availability or engagement signals

In the validated local output:

```text
Trap candidates in output: 0
```

---

## Sandbox Demo

A Google Colab demo is available here:

```text
https://colab.research.google.com/drive/1U-wI02xjG6RBt3UBdvnvmdgCtzTuL50G?usp=sharing
```

The notebook demonstrates:

1. Cloning the repository
2. Installing ranking dependencies
3. Downloading precomputed artifacts
4. Running the CPU-only ranker
5. Validating `outputs/SignalRank.csv`
6. Displaying the top ranked rows

---

## AI Tools Disclosure

This project was developed with AI-assisted tooling.

Tools used:

- ChatGPT — architecture discussion, implementation guidance, debugging, documentation, and submission packaging support
- Claude — code review, notebook review, implementation suggestions, and documentation support

All engineering decisions, integration, testing, scoring methodology, debugging, validation, and final submission preparation were performed by the author.

No hosted LLM APIs are used during the ranking step. The final ranking pipeline is fully deterministic, CPU-only, offline, and operates entirely on local precomputed artifacts.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Polars
- SciPy
- orjson
- PyArrow / Parquet
- Sentence Transformers
- `BAAI/bge-base-en-v1.5`

---

## Team

**Team name:** SignalRank

**Team member:**

- Abey Ajit — ajitabey@yahoo.com

---

## Notes for Reviewers

The repository intentionally does not commit the raw dataset or large generated artifacts.

Ignored files include:

```text
dataset/candidates.jsonl
dataset/artifacts/*.parquet
artifacts/*.npy
artifacts/*.parquet
outputs/diagnostics/
outputs/top100_ranked_diagnostic.csv
```

The final CSV is committed at:

```text
outputs/SignalRank.csv
```

The large artifacts required for reproduction are available through the artifact zip linked above.