"""
Redrob Ranker — Part 2
Precompute Embeddings (run once — output saved to disk)

Loads candidate_base / candidate_jobs / availability_scores / survivors,
builds enriched job text, embeds job texts + query strings with
BAAI/bge-base-en-v1.5, and saves:

    job_embeddings.npy      (N_jobs, 768)
    query_embeddings.npy    (11, 768)
    job_meta.parquet        per-job metadata used later by rank.py

This step may use GPU if available. It is NOT the timed/reproduced ranking
step — rank.py (CPU-only) is what judges run against the compute constraint.

Usage:
    python scripts/precompute_embeddings.py \
        --data-dir dataset/artifacts \
        --artifacts-dir artifacts \
        --reference-date 2026-06-25
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Must-have queries (M1-M4) ─────────────────────────────────────────────
MUST_HAVE_QUERIES = {
    'M1': (
        'production deployment of embedding-based retrieval or semantic search '
        'systems to real users, handling embedding drift, index refresh, and '
        'retrieval quality regression in production'
    ),
    'M2': (
        'hands-on production experience with vector databases or hybrid search '
        'infrastructure such as FAISS, Pinecone, Weaviate, Qdrant, Milvus, '
        'OpenSearch, or Elasticsearch — operational experience not just tutorials'
    ),
    'M3': (
        'designing and operating ranking evaluation frameworks, NDCG, MRR, MAP, '
        'offline-to-online correlation, A/B testing of ranking or recommendation '
        'systems, understanding how to measure ranking quality rigorously'
    ),
    'M4': (
        'personally implemented and deployed ML or backend systems to production, '
        'wrote production Python code, owned deployed services, hands-on '
        'implementation rather than architecture oversight or team management'
    ),
}

# ── Nice-to-have queries (N1-N4) — bonus only, never penalises absence ────
NICE_TO_HAVE_QUERIES = {
    'N1': (
        'LLM fine-tuning experience — LoRA, QLoRA, PEFT, parameter-efficient '
        'fine-tuning, instruction tuning, RLHF'
    ),
    'N2': (
        'learning-to-rank models, XGBoost LTR, neural ranking, '
        'listwise or pairwise ranking losses, LambdaMART'
    ),
    'N3': (
        'HR technology, recruiting platform, marketplace product, '
        'talent intelligence, job matching, candidate search product'
    ),
    'N4': (
        'distributed systems, large-scale ML inference optimization, '
        'model serving at scale, latency optimization, GPU inference'
    ),
}

# ── Domain + hands-on queries ─────────────────────────────────────────────
DOMAIN_QUERIES = {
    'IR': (
        'information retrieval, NLP, natural language processing, search, ranking, '
        'recommendation systems, text embeddings, semantic search, '
        'retrieval augmented generation, question answering'
    ),
    'CV': (
        'computer vision, image recognition, object detection, speech recognition, '
        'audio processing, robotics, pose estimation, image segmentation, '
        'convolutional neural network'
    ),
}

HANDS_ON_QUERY = (
    'personally implemented and deployed ML or backend systems, '
    'wrote production code, built and shipped features to real users, '
    'hands-on engineering not team management or architecture review'
)

QUERY_KEYS = ['M1', 'M2', 'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'IR', 'CV', 'HANDS_ON']

QUERY_TEXTS = [
    MUST_HAVE_QUERIES['M1'], MUST_HAVE_QUERIES['M2'],
    MUST_HAVE_QUERIES['M3'], MUST_HAVE_QUERIES['M4'],
    NICE_TO_HAVE_QUERIES['N1'], NICE_TO_HAVE_QUERIES['N2'],
    NICE_TO_HAVE_QUERIES['N3'], NICE_TO_HAVE_QUERIES['N4'],
    DOMAIN_QUERIES['IR'], DOMAIN_QUERIES['CV'],
    HANDS_ON_QUERY,
]
assert len(QUERY_KEYS) == 11 == len(QUERY_TEXTS), "Query count mismatch"

TECH_ANCHORS = {
    'milvus', 'pinecone', 'faiss', 'elasticsearch', 'qdrant', 'weaviate',
    'bert', 'embedding', 'vector', 'retrieval', 'ndcg', 'ranking', 'transformer',
    'fine-tun', 'pytorch', 'sklearn', 'xgboost', 'rag', 'llm', 'semantic',
    'reranking', 'bm25', 'dense', 'sparse', 'hybrid', 'sentence-transformer',
}


def safe_text(x) -> str:
    """Return empty string for NaN/None; str() everything else."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def description_confidence(text) -> float:
    """
    Scalar [0.20, 1.0] penalising vague short descriptions.
    Short-but-technical gets a floor of 0.75.
    """
    text = safe_text(text)
    if not text:
        return 0.20
    wc = len(text.split())
    has_anchor = any(t in text.lower() for t in TECH_ANCHORS)
    if wc >= 30:
        return 1.0
    elif has_anchor:
        return max(0.75, min(1.0, wc / 30))
    else:
        return min(1.0, wc / 30)


def enrich_job_text(row) -> str:
    """
    Enrich job text for embedding. For sparse job descriptions (<15 words),
    fall back to candidate headline + summary from candidate_base
    (merged in earlier). Do NOT include company name — adds noise.
    """
    title = safe_text(row.get('title'))
    description = safe_text(row.get('description'))
    industry = safe_text(row.get('industry'))
    headline = safe_text(row.get('headline'))
    summary = safe_text(row.get('summary'))

    wc = len(description.split())

    if wc < 15:
        profile_ctx = ' '.join(filter(None, [industry, headline, summary[:300]]))
        return f"{title}. {profile_ctx}. {description}".strip()
    else:
        return f"{title}. {description}".strip()


def encode_with_oom_backoff(model, texts, *, batch_size, device, label):
    """
    Encode texts with normalized BGE embeddings.
    If CUDA OOM happens, halve the batch size and retry automatically.
    """
    current_batch = int(batch_size)
    while True:
        try:
            print(f"Encoding {len(texts):,} {label} on {device} (batch={current_batch}) ...")
            t0 = time.time()
            arr = model.encode(
                texts,
                batch_size=current_batch,
                normalize_embeddings=True,  # L2 norm -> dot product == cosine similarity
                show_progress_bar=True,
                convert_to_numpy=True,
                device=device,
            )
            arr = np.asarray(arr, dtype=np.float32)
            elapsed = time.time() - t0
            print(f"Done in {elapsed / 60:.1f} min — shape: {arr.shape}, dtype={arr.dtype}")
            return arr
        except RuntimeError as e:
            msg = str(e).lower()
            if device.startswith('cuda') and ('out of memory' in msg or 'cuda' in msg):
                if current_batch <= 32:
                    raise
                print(f"CUDA OOM at batch={current_batch}. Retrying with batch={current_batch // 2} ...")
                torch.cuda.empty_cache()
                gc.collect()
                current_batch //= 2
            else:
                raise


def run(data_dir: Path, embed_dir: Path, reference_date: str, force_cpu: bool, batch_size_override: int | None) -> None:
    embed_dir.mkdir(parents=True, exist_ok=True)

    REFERENCE_DATE = pd.Timestamp(reference_date)

    PRECOMPUTE_DEVICE = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"
    if PRECOMPUTE_DEVICE == "cuda":
        BATCH_SIZE = batch_size_override or 512
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Precompute device : CUDA — {torch.cuda.get_device_name(0)}")
        print(f"VRAM              : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        BATCH_SIZE = batch_size_override or 128
        print("Precompute device : CPU")
    print(f"Embedding batch   : {BATCH_SIZE}")
    print(f"Reference date    : {REFERENCE_DATE.date()}")

    # ── Load required artifacts ────────────────────────────────────────────
    required_artifacts = [
        'candidate_base.parquet',
        'candidate_jobs.parquet',
        'availability_scores.parquet',
        'survivors.parquet',
    ]
    missing = [name for name in required_artifacts if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required parquet artifacts: " + ", ".join(missing) +
            "\nRun scripts/parse_and_filter.py and scripts/compute_availability.py first."
        )

    df_base = pd.read_parquet(data_dir / 'candidate_base.parquet')
    df_jobs_raw = pd.read_parquet(data_dir / 'candidate_jobs.parquet')
    df_survivors = pd.read_parquet(data_dir / 'survivors.parquet')

    survivor_ids = set(df_survivors['candidate_id'])
    df_jobs = df_jobs_raw[df_jobs_raw['candidate_id'].isin(survivor_ids)].copy()

    print(f"Survivors           : {len(survivor_ids):,}")
    print(f"Job rows (raw)      : {len(df_jobs_raw):,}")
    print(f"Job rows (survivors): {len(df_jobs):,}")

    # ── Merge candidate profile fields into df_jobs before enrichment ──────
    PROFILE_COLS = [
        c for c in [
            'candidate_id', 'headline', 'summary', 'location',
            'country', 'years_of_experience', 'current_title',
        ]
        if c in df_base.columns
    ]
    df_jobs = df_jobs.merge(
        df_base[PROFILE_COLS].drop_duplicates('candidate_id'),
        on='candidate_id',
        how='left',
    )

    df_jobs['enriched_text'] = df_jobs.apply(enrich_job_text, axis=1)
    df_jobs['description_confidence'] = df_jobs['description'].apply(description_confidence)

    company_col = next(
        (c for c in ['company', 'employer', 'organization', 'company_name'] if c in df_jobs.columns),
        None,
    )
    if company_col is None:
        df_jobs['company'] = 'Unknown'
        company_col = 'company'

    df_jobs['end_date'] = pd.to_datetime(df_jobs['end_date'], errors='coerce')
    df_jobs['start_date'] = pd.to_datetime(df_jobs['start_date'], errors='coerce')
    df_jobs['is_current'] = df_jobs['is_current'].fillna(False).astype(bool)
    df_jobs.loc[df_jobs['is_current'], 'end_date'] = REFERENCE_DATE

    df_jobs['duration_months'] = (
        (df_jobs['end_date'] - df_jobs['start_date']).dt.days / 30.44
    ).clip(lower=0)
    df_jobs['months_since_end'] = (
        (REFERENCE_DATE - df_jobs['end_date']).dt.days / 30.44
    ).clip(lower=0)

    META_COLS = [
        'candidate_id', 'title', company_col,
        'description_confidence', 'is_current',
        'end_date', 'duration_months', 'months_since_end',
    ]
    df_meta_save = (
        df_jobs[META_COLS]
        .rename(columns={company_col: 'company'})
        .reset_index(drop=True)
    )
    df_meta_save.to_parquet(embed_dir / 'job_meta.parquet', index=True)
    print(f"Saved job_meta.parquet: {len(df_meta_save):,} rows -> {embed_dir / 'job_meta.parquet'}")

    # ── Embeddings ───────────────────────────────────────────────────────
    JOB_EMB_PATH = embed_dir / 'job_embeddings.npy'
    QUERY_EMB_PATH = embed_dir / 'query_embeddings.npy'

    from sentence_transformers import SentenceTransformer

    need_job_embeddings = not JOB_EMB_PATH.exists()
    need_query_embeddings = not QUERY_EMB_PATH.exists()

    model = None
    if need_job_embeddings or need_query_embeddings:
        print("Loading BAAI/bge-base-en-v1.5 ...")
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=PRECOMPUTE_DEVICE)
        print("Model loaded.\n")
    else:
        print("Both embedding files already exist — skipping model load.")

    if JOB_EMB_PATH.exists():
        print(f"job_embeddings.npy already exists ({JOB_EMB_PATH.stat().st_size / 1e6:.0f} MB).")
    else:
        texts = df_jobs['enriched_text'].fillna('').astype(str).tolist()
        job_embeddings = encode_with_oom_backoff(
            model, texts, batch_size=BATCH_SIZE, device=PRECOMPUTE_DEVICE, label='job texts',
        )
        np.save(JOB_EMB_PATH, job_embeddings)
        print(f"Saved -> {JOB_EMB_PATH}  ({JOB_EMB_PATH.stat().st_size / 1e6:.0f} MB)")
        del job_embeddings
        gc.collect()
        if PRECOMPUTE_DEVICE == 'cuda':
            torch.cuda.empty_cache()

    if QUERY_EMB_PATH.exists():
        print(f"query_embeddings.npy already exists ({QUERY_EMB_PATH.stat().st_size / 1e6:.2f} MB).")
    else:
        query_embeddings = encode_with_oom_backoff(
            model, QUERY_TEXTS, batch_size=11, device=PRECOMPUTE_DEVICE, label='query strings',
        )
        np.save(QUERY_EMB_PATH, query_embeddings)
        print(f"Saved -> {QUERY_EMB_PATH}  shape: {query_embeddings.shape}")
        del query_embeddings

    if model is not None:
        del model
        gc.collect()
        if PRECOMPUTE_DEVICE == 'cuda':
            torch.cuda.empty_cache()
            print("Released CUDA cache after precompute.")

    print("\nPrecompute complete. Artifacts written to:", embed_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset/artifacts"),
        help="Directory with candidate_base/candidate_jobs/availability_scores/survivors "
             "parquet files (default: dataset/artifacts)",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Output directory for embedding artifacts (job_embeddings.npy, "
             "query_embeddings.npy, job_meta.parquet) (default: artifacts)",
    )
    parser.add_argument(
        "--reference-date", type=str, default="2026-06-25",
        help="Fixed reference date, YYYY-MM-DD, for reproducibility (default: 2026-06-25)",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Force CPU embedding even if a GPU is available",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override embedding batch size (default: 512 on GPU, 128 on CPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.artifacts_dir, args.reference_date, args.force_cpu, args.batch_size)
