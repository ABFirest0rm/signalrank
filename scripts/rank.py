"""
Redrob Ranker — RANKING PHASE (CPU-safe, artifact-only)

Loads the precomputed embedding artifacts (no model, no network, no GPU
calls) and produces the final ranked top-100 candidate list.

Scoring pipeline:
    1. Cosine similarity (dot product on normalized embeddings) of every
       job against 11 queries (M1-M4 must-have, N1-N4 nice-to-have,
       IR/CV domain, HANDS_ON).
    2. Recency-weighted, confidence-adjusted per-job scores.
    3. Per-candidate M-scores (70/30 blend of best-job + recency-weighted avg).
    4. Geometric mean of M1-M4 -> must_have_score.
    5. Nice-to-have bonus (capped +20%).
    6. Domain multiplier (CV-heavy / IR-light penalty).
    7. Hands-on multiplier (recent hands-on evidence).
    8. Job-hopper multiplier (tenure-based).
    9. Availability multiplier (from Step 3 artifact).
   10. Trap multiplier (honeypot penalty, if trap_scores.parquet present).
   11. Final score = must_have (with NTH bonus) x domain x hands_on
                      x availability x job_hopper x trap.

This script performs no model calls and is intended to run in well under
5 minutes / 16GB RAM on CPU only, per the compute constraint.

Usage:
    python scripts/rank.py \
        --data-dir dataset/artifacts \
        --artifacts-dir artifacts \
        --output-dir output \
        --team-id team_xxx \
        --reference-date 2026-06-25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean as scipy_gmean

QUERY_KEYS = ['M1', 'M2', 'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'IR', 'CV', 'HANDS_ON']
M_COLS = ['M1', 'M2', 'M3', 'M4']
N_COLS = ['N1', 'N2', 'N3', 'N4']
GMEAN_FLOOR = 0.05

# ── Domain multiplier thresholds ────────────────────────────────────────────
CV_HIGH_THRESH = 0.65
CV_MED_THRESH = 0.50
IR_LOW_THRESH = 0.40
IR_MED_THRESH = 0.50

# ── Hands-on multiplier thresholds ──────────────────────────────────────────
HO_THRESHOLDS = [(0.75, 1.00), (0.55, 0.85), (0.35, 0.70)]


def domain_mult_fn(ir: float, cv: float) -> float:
    if cv > CV_HIGH_THRESH and ir < IR_LOW_THRESH:
        return 0.65
    elif cv > CV_MED_THRESH and ir < IR_MED_THRESH:
        return 0.90
    return 1.00


def hands_on_mult_fn(score: float) -> float:
    for thresh, mult in HO_THRESHOLDS:
        if score >= thresh:
            return mult
    return 0.55


def hopper_mult_fn(med: float, last3: float, cur: float) -> float:
    if med >= 24 and last3 >= 18:
        mult = 1.00
    elif med >= 18 or last3 >= 18:
        mult = 0.92
    elif med >= 12:
        mult = 0.82
    else:
        mult = 0.70
    if cur < 6:
        mult *= 0.95
    return mult


def build_reason(row) -> str:
    strengths = []
    if row['M1_score'] > 0.60: strengths.append('retrieval/search systems')
    if row['M2_score'] > 0.60: strengths.append('vector DB / hybrid search')
    if row['M3_score'] > 0.60: strengths.append('ranking evaluation frameworks')
    if row['M4_score'] > 0.60: strengths.append('hands-on production coding')

    flags = []
    if row['job_hopper_multiplier'] < 0.85: flags.append('job stability concern')
    if row['domain_multiplier'] < 1.00: flags.append('partial domain mismatch')
    if row['low_description_confidence']: flags.append('limited description detail')
    if row['availability_multiplier'] < 0.80: flags.append('availability friction')
    if row.get('all_sparse', False): flags.append('all jobs have sparse descriptions')
    if row['trap_multiplier'] < 1.00: flags.append('possible honeypot/profile inconsistency')

    strength_str = ', '.join(strengths) if strengths else 'general ML background'
    flag_str = ('. Concern: ' + '; '.join(flags)) if flags else ''
    return (
        f"Evidence of {strength_str} from "
        f"{row['best_matching_job_title']} at {row['best_matching_job_company']} "
        f"with balanced must-have scores "
        f"M1:{row['M1_score']:.2f}, M2:{row['M2_score']:.2f}, "
        f"M3:{row['M3_score']:.2f}, M4:{row['M4_score']:.2f}{flag_str}."
    )


def run(data_dir: Path, embed_dir: Path, output_dir: Path, team_id: str, reference_date: str, top_n: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    REFERENCE_DATE = pd.Timestamp(reference_date)

    JOB_EMB_PATH = embed_dir / 'job_embeddings.npy'
    QUERY_EMB_PATH = embed_dir / 'query_embeddings.npy'
    JOB_META_PATH = embed_dir / 'job_meta.parquet'

    for path in [JOB_EMB_PATH, QUERY_EMB_PATH, JOB_META_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required ranking artifact: {path}\n"
                "Run scripts/precompute_embeddings.py first."
            )

    survivors_path = data_dir / 'survivors.parquet'
    avail_path = data_dir / 'availability_scores.parquet'
    base_path = data_dir / 'candidate_base.parquet'
    for path in [survivors_path, avail_path, base_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    print("RANKING PHASE: loading local artifacts only — no model, no network, no GPU calls.")

    df_survivors = pd.read_parquet(survivors_path)
    df_avail = pd.read_parquet(avail_path)
    df_base = pd.read_parquet(base_path)
    survivor_ids = set(df_survivors['candidate_id'])

    job_embeddings = np.load(JOB_EMB_PATH, mmap_mode='r')
    query_embeddings = np.asarray(np.load(QUERY_EMB_PATH), dtype=np.float32)
    df_meta = pd.read_parquet(JOB_META_PATH)

    assert len(job_embeddings) == len(df_meta), (
        f"Row mismatch: embeddings={len(job_embeddings)}, meta={len(df_meta)}"
    )
    assert query_embeddings.shape[0] == len(QUERY_KEYS), (
        f"Query mismatch: embeddings={query_embeddings.shape[0]}, keys={len(QUERY_KEYS)}"
    )

    print(f"job_embeddings   : {job_embeddings.shape} / {job_embeddings.dtype}")
    print(f"query_embeddings : {query_embeddings.shape} / {query_embeddings.dtype}")
    print(f"job_meta rows    : {len(df_meta):,}")

    # ── Matrix multiply: (N_jobs, 768) @ (768, 11) = (N_jobs, 11) ──────────
    t0 = time.time()
    score_matrix = np.asarray(job_embeddings @ query_embeddings.T, dtype=np.float32)
    print(f"Score matrix: {score_matrix.shape} computed in {time.time() - t0:.2f}s")

    score_matrix = np.clip(score_matrix, 0, None)
    for i, key in enumerate(QUERY_KEYS):
        df_meta[f'score_{key}'] = score_matrix[:, i]

    # ── Recency weights ─────────────────────────────────────────────────────
    df_meta['recency_mh'] = np.select(
        condlist=[
            df_meta['is_current'],
            df_meta['months_since_end'] <= 18,
            df_meta['months_since_end'] <= 36,
            df_meta['months_since_end'] <= 60,
        ],
        choicelist=[1.25, 1.00, 0.65, 0.35],
        default=0.15,
    )
    df_meta['recency_nth'] = np.select(
        condlist=[
            df_meta['is_current'],
            df_meta['months_since_end'] <= 36,
            df_meta['months_since_end'] <= 60,
        ],
        choicelist=[1.15, 1.00, 0.75],
        default=0.50,
    )

    # ── Per-job adjusted M scores ────────────────────────────────────────────
    for m in M_COLS:
        df_meta[f'adj_{m}'] = (
            df_meta[f'score_{m}'] * df_meta['description_confidence'] * df_meta['recency_mh']
        ).clip(0, 1.0)

    df_cand = pd.DataFrame(index=pd.Index(sorted(survivor_ids), name='candidate_id'))

    recency_mh_sum = df_meta.groupby('candidate_id')['recency_mh'].sum()

    for m in M_COLS:
        best = df_meta.groupby('candidate_id')[f'adj_{m}'].max()
        w_avg = (df_meta.groupby('candidate_id')[f'adj_{m}'].sum() / recency_mh_sum).fillna(0)
        df_cand[f'{m}_score'] = (
            (0.70 * best + 0.30 * w_avg)
            .reindex(df_cand.index)
            .fillna(0)
            .clip(0, 1.0)
        )

    # ── Geometric mean of M1-M4 -> must_have_score ──────────────────────────
    m_arr = df_cand[['M1_score', 'M2_score', 'M3_score', 'M4_score']].values.clip(GMEAN_FLOOR, 1.0)
    df_cand['must_have_score'] = scipy_gmean(m_arr, axis=1)

    # ── Nice-to-have bonus ────────────────────────────────────────────────
    for n in N_COLS:
        df_meta[f'adj_{n}'] = (
            df_meta[f'score_{n}'] * df_meta['description_confidence'] * df_meta['recency_nth']
        ).clip(0, 1.0)
    df_meta['best_nth'] = df_meta[[f'adj_{n}' for n in N_COLS]].max(axis=1)

    recency_nth_sum = df_meta.groupby('candidate_id')['recency_nth'].sum()
    df_cand['nth_raw'] = (
        df_meta.groupby('candidate_id')['best_nth'].sum() / recency_nth_sum
    ).reindex(df_cand.index).fillna(0).clip(0, 1.0)
    df_cand['nth_bonus'] = (df_cand['nth_raw'] * 0.20).clip(0, 0.20)

    # ── Domain multiplier ────────────────────────────────────────────────
    cand_ir = df_meta.groupby('candidate_id')['score_IR'].max().reindex(df_cand.index).fillna(0)
    cand_cv = df_meta.groupby('candidate_id')['score_CV'].max().reindex(df_cand.index).fillna(0)
    df_cand['domain_multiplier'] = [
        domain_mult_fn(cand_ir.get(cid, 0), cand_cv.get(cid, 0)) for cid in df_cand.index
    ]

    # ── Hands-on multiplier ──────────────────────────────────────────────
    recent_mask = df_meta['is_current'] | (df_meta['months_since_end'] <= 18)
    recent_ho = (
        df_meta[recent_mask]
        .groupby('candidate_id')['score_HANDS_ON']
        .max()
        .reindex(df_cand.index)
        .fillna(0)
    )
    df_cand['hands_on_multiplier'] = recent_ho.map(hands_on_mult_fn)

    # ── Job-hopper multiplier ────────────────────────────────────────────
    completed = df_meta[~df_meta['is_current']].copy()
    median_tenure = completed.groupby('candidate_id')['duration_months'].median()
    last3_avg = (
        completed.sort_values('months_since_end')
        .groupby('candidate_id')
        .head(3)
        .groupby('candidate_id')['duration_months']
        .mean()
    )
    current_dur = df_meta[df_meta['is_current']].groupby('candidate_id')['duration_months'].max()

    df_cand['job_hopper_multiplier'] = [
        hopper_mult_fn(median_tenure.get(cid, 24), last3_avg.get(cid, 24), current_dur.get(cid, 24))
        for cid in df_cand.index
    ]

    # ── Availability multiplier ──────────────────────────────────────────
    avail = df_avail.set_index('candidate_id')['availability_multiplier']
    df_cand['availability_multiplier'] = df_cand.index.map(avail).fillna(1.0)

    # ── Trap multiplier — require explicit column, no boolean auto-detect ──
    trap_path = data_dir / 'trap_scores.parquet'
    if trap_path.exists():
        df_trap = pd.read_parquet(trap_path).set_index('candidate_id')
        if 'trap_multiplier' not in df_trap.columns:
            raise ValueError(
                "trap_scores.parquet exists but has no 'trap_multiplier' column.\n"
                f"Columns found: {list(df_trap.columns)}\n"
                "Rename the correct column to 'trap_multiplier' (float, 0.0-1.0) before continuing."
            )
        df_cand['trap_multiplier'] = df_cand.index.map(df_trap['trap_multiplier']).fillna(1.0)
        print("Trap multiplier loaded.")
    else:
        df_cand['trap_multiplier'] = 1.0
        print("trap_scores.parquet not found — trap_multiplier = 1.0")

    # ── Final score assembly (multiplicative) ────────────────────────────
    df_cand['semantic_score'] = (
        df_cand['must_have_score']
        * (1 + df_cand['nth_bonus'])
        * df_cand['domain_multiplier']
        * df_cand['hands_on_multiplier']
    )
    df_cand['final_score_raw'] = (
        df_cand['semantic_score']
        * df_cand['availability_multiplier']
        * df_cand['job_hopper_multiplier']
        * df_cand['trap_multiplier']
    )
    # ALWAYS rank on raw — floor is ONLY for the display column
    df_cand['final_score'] = df_cand['final_score_raw'].clip(lower=0.10)

    # ── Best matching job (composite M gmean, not single-M max) ────────────
    df_meta['composite_M'] = scipy_gmean(
        df_meta[[f'adj_{m}' for m in M_COLS]].clip(GMEAN_FLOOR, 1.0), axis=1,
    )
    best_job = (
        df_meta.sort_values('composite_M', ascending=False)
        .groupby('candidate_id')
        .first()
        [['title', 'company', 'description_confidence']]
        .rename(columns={
            'title': 'best_matching_job_title',
            'company': 'best_matching_job_company',
            'description_confidence': 'best_job_confidence',
        })
    )
    df_cand = df_cand.join(best_job)
    df_cand['best_matching_job_title'] = df_cand['best_matching_job_title'].fillna('Unknown')
    df_cand['best_matching_job_company'] = df_cand['best_matching_job_company'].fillna('Unknown')
    df_cand['best_job_confidence'] = df_cand['best_job_confidence'].fillna(0.0)

    df_cand['low_description_confidence'] = df_cand['best_job_confidence'] < 0.75
    all_sparse = df_meta.groupby('candidate_id')['description_confidence'].max().lt(0.75)
    df_cand['all_sparse'] = df_cand.index.map(all_sparse).fillna(True)

    # ── Deterministic Top-N ──────────────────────────────────────────────
    topN = (
        df_cand.sort_values(['final_score_raw'], ascending=[False], kind='mergesort')
        .reset_index()
        .sort_values(['final_score_raw', 'candidate_id'], ascending=[False, True], kind='mergesort')
        .head(top_n)
        .copy()
    )
    topN['rank'] = np.arange(1, len(topN) + 1)
    topN['reasoning'] = topN.apply(build_reason, axis=1)

    DIAG_COLS = [
        'candidate_id', 'rank', 'final_score_raw', 'final_score', 'semantic_score', 'must_have_score',
        'M1_score', 'M2_score', 'M3_score', 'M4_score', 'nth_bonus',
        'domain_multiplier', 'hands_on_multiplier', 'job_hopper_multiplier',
        'availability_multiplier', 'trap_multiplier',
        'best_matching_job_title', 'best_matching_job_company',
        'low_description_confidence', 'all_sparse', 'reasoning',
    ]
    diag = topN[DIAG_COLS].copy()
    diag_path = output_dir / f'top{top_n}_ranked_diagnostic.csv'
    diag.to_csv(diag_path, index=False, encoding='utf-8')

    submission = pd.DataFrame({
        'candidate_id': topN['candidate_id'].astype(str).values,
        'rank': topN['rank'].astype(int).values,
        'score': topN['final_score_raw'].astype(float).round(6).values,
        'reasoning': topN['reasoning'].astype(str).values,
    })
    submission_path = output_dir / f'{team_id}.csv'
    submission.to_csv(submission_path, index=False, encoding='utf-8')

    print(f"Saved validator-ready submission: {submission_path}")
    print(f"Saved diagnostic CSV          : {diag_path}")
    print("\nTop 10 snapshot:")
    print(
        submission.merge(
            topN[['candidate_id', 'must_have_score', 'best_matching_job_title']],
            on='candidate_id', how='left',
        )[['rank', 'candidate_id', 'score', 'must_have_score', 'best_matching_job_title']]
        .head(10)
        .to_string(index=False)
    )

    return submission_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset/artifacts"),
        help="Directory with survivors/availability_scores/candidate_base parquet "
             "(and optional trap_scores.parquet) (default: dataset/artifacts)",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory with precomputed job_embeddings.npy, query_embeddings.npy, "
             "job_meta.parquet (default: artifacts)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to write the submission CSV and diagnostic CSV (default: output)",
    )
    parser.add_argument(
        "--team-id", type=str, default="team_xxx",
        help="Team ID — output file will be output/<TEAM_ID>.csv (default: team_xxx)",
    )
    parser.add_argument(
        "--reference-date", type=str, default="2026-06-25",
        help="Fixed reference date, YYYY-MM-DD, for reproducibility (default: 2026-06-25)",
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of ranked candidates to output (default: 100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.artifacts_dir, args.output_dir, args.team_id, args.reference_date, args.top_n)
