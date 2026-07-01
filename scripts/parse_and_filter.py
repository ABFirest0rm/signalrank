"""
Redrob Ranker — Steps 1 & 2
Parse JSONL -> Hard Filters -> Qualifier Gates

Reads the raw candidates.jsonl(.gz) file and produces the base artifact
tables used by the rest of the pipeline:

    candidate_base.parquet       one row per candidate (profile fields)
    candidate_jobs.parquet       one row per job in career_history
    candidate_skills.parquet     one row per skill entry
    candidate_education.parquet  one row per education record
    candidate_redrob.parquet     23 redrob behavioral/platform signals
    step1_survivors.parquet / step1_dropped.parquet
    survivors.parquet            final candidate_id list feeding Step 3
    step2_dropped.parquet

Usage:
    python scripts/parse_and_filter.py \
        --dataset dataset/candidates.jsonl \
        --artifacts-dir dataset/artifacts \
        --reference-date 2026-06-25
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import orjson
import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "hcl", "capgemini",
    "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree",
}
# Longest names first so "tata consultancy" is tried before "tcs"
CONSULTING_PATTERN = "|".join(
    re.escape(f) for f in sorted(CONSULTING_FIRMS, key=len, reverse=True)
)

OVERLAP_THRESHOLD_MONTHS = 2.0
SPAN_BUFFER_MONTHS = 36


# ── Timing helpers ───────────────────────────────────────────────────────────

_t_start: dict[str, float] = {}
_t_elapsed: dict[str, float] = {}
_counts: dict[str, int] = {}


def tick(label: str) -> None:
    _t_start[label] = time.perf_counter()


def tock(label: str) -> float:
    e = time.perf_counter() - _t_start[label]
    _t_elapsed[label] = e
    return e


# ── Row extractors ───────────────────────────────────────────────────────────

def extract_base(cid: str, rec: dict) -> dict:
    p = rec.get("profile", {}) or {}
    return {
        "candidate_id": cid,
        "anonymized_name": p.get("anonymized_name"),
        "headline": p.get("headline"),
        "summary": p.get("summary"),
        "location": p.get("location"),
        "country": p.get("country"),
        "years_of_experience": p.get("years_of_experience"),
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "current_company_size": p.get("current_company_size"),
        "current_industry": p.get("current_industry"),
    }


def extract_jobs(cid: str, rec: dict) -> list[dict]:
    rows = []
    for idx, job in enumerate(rec.get("career_history", []) or []):
        rows.append({
            "candidate_id": cid,
            "job_index": idx,
            "company": job.get("company"),
            "title": job.get("title"),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            "duration_months": job.get("duration_months"),
            "is_current": job.get("is_current"),
            "industry": job.get("industry"),
            "company_size": job.get("company_size"),
            "description": job.get("description"),
        })
    return rows


def extract_skills(cid: str, rec: dict) -> list[dict]:
    rows = []
    for idx, sk in enumerate(rec.get("skills", []) or []):
        rows.append({
            "candidate_id": cid,
            "skill_index": idx,
            "name": sk.get("name"),
            "proficiency": sk.get("proficiency"),
            "endorsements": sk.get("endorsements", 0),
            "duration_months": sk.get("duration_months", 0),
        })
    return rows


def extract_education(cid: str, rec: dict) -> list[dict]:
    rows = []
    for idx, ed in enumerate(rec.get("education", []) or []):
        rows.append({
            "candidate_id": cid,
            "edu_index": idx,
            "institution": ed.get("institution"),
            "degree": ed.get("degree"),
            "field_of_study": ed.get("field_of_study"),
            "start_year": ed.get("start_year"),
            "end_year": ed.get("end_year"),
            "grade": ed.get("grade"),
            "tier": ed.get("tier"),
        })
    return rows


def extract_redrob(cid: str, rec: dict) -> dict:
    r = rec.get("redrob_signals", {}) or {}
    salary = r.get("expected_salary_range_inr_lpa") or {}
    assessment_raw = r.get("skill_assessment_scores") or {}
    assessment_json = json.dumps(assessment_raw) if assessment_raw else None
    return {
        "candidate_id": cid,
        "profile_completeness_score": r.get("profile_completeness_score"),
        "signup_date": r.get("signup_date"),
        "last_active_date": r.get("last_active_date"),
        "open_to_work_flag": r.get("open_to_work_flag"),
        "profile_views_received_30d": r.get("profile_views_received_30d"),
        "applications_submitted_30d": r.get("applications_submitted_30d"),
        "recruiter_response_rate": r.get("recruiter_response_rate"),
        "avg_response_time_hours": r.get("avg_response_time_hours"),
        "skill_assessment_scores_json": assessment_json,
        "connection_count": r.get("connection_count"),
        "endorsements_received": r.get("endorsements_received"),
        "notice_period_days": r.get("notice_period_days"),
        "expected_salary_min_lpa": salary.get("min"),
        "expected_salary_max_lpa": salary.get("max"),
        "preferred_work_mode": r.get("preferred_work_mode"),
        "willing_to_relocate": r.get("willing_to_relocate"),
        # -1 sentinel = not linked / no history — never coerce to 0
        "github_activity_score": r.get("github_activity_score"),
        "search_appearance_30d": r.get("search_appearance_30d"),
        "saved_by_recruiters_30d": r.get("saved_by_recruiters_30d"),
        "interview_completion_rate": r.get("interview_completion_rate"),
        "offer_acceptance_rate": r.get("offer_acceptance_rate"),  # -1 = no history
        "verified_email": r.get("verified_email"),
        "verified_phone": r.get("verified_phone"),
        "linkedin_connected": r.get("linkedin_connected"),
    }


def _open(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def run(dataset_path: Path, artifacts_dir: Path, today: date) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE 0 — Parse JSONL
    # =========================================================================
    print("=" * 62)
    print("  PHASE 0 — Parse JSONL")
    print("=" * 62)
    tick("phase0")

    base_rows: list[dict] = []
    job_rows: list[dict] = []
    skill_rows: list[dict] = []
    edu_rows: list[dict] = []
    redrob_rows: list[dict] = []
    total_parsed = 0
    parse_errors = 0
    raw_dups = 0
    seen_ids: set[str] = set()

    print(f"Reading: {dataset_path}")

    with _open(dataset_path) as fh:
        for line_num, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = orjson.loads(line)
            except Exception as e:  # noqa: BLE001
                parse_errors += 1
                if parse_errors <= 3:
                    print(f"  [WARN] L{line_num}: {e}", file=sys.stderr)
                continue

            cid = rec.get("candidate_id")
            if not cid:
                parse_errors += 1
                continue

            if cid in seen_ids:
                raw_dups += 1
                continue
            seen_ids.add(cid)

            base_rows.append(extract_base(cid, rec))
            job_rows.extend(extract_jobs(cid, rec))
            skill_rows.extend(extract_skills(cid, rec))
            edu_rows.extend(extract_education(cid, rec))
            redrob_rows.append(extract_redrob(cid, rec))
            total_parsed += 1

            if total_parsed % 10_000 == 0:
                print(f"  ... {total_parsed:,} parsed")

    tock("phase0")
    _counts.update({
        "parsed": total_parsed,
        "parse_errors": parse_errors,
        "raw_dups": raw_dups,
    })
    print(f"Done — {total_parsed:,} candidates  |  errors: {parse_errors}  |  raw dupes: {raw_dups}")
    print(f"Time: {_t_elapsed['phase0']:.1f}s\n")

    # =========================================================================
    # PHASE 1a — Write Parquets
    # =========================================================================
    print("=" * 62)
    print("  PHASE 1a — Write Parquets")
    print("=" * 62)
    tick("phase1a")

    _dfs: dict[str, pl.DataFrame] = {
        "candidate_base": pl.DataFrame(base_rows),
        "candidate_jobs": pl.DataFrame(job_rows),
        "candidate_skills": pl.DataFrame(skill_rows),
        "candidate_education": pl.DataFrame(edu_rows),
        "candidate_redrob": pl.DataFrame(redrob_rows),
    }

    for name, df in _dfs.items():
        out = artifacts_dir / f"{name}.parquet"
        df.write_parquet(out)
        print(f"  {name:28s}  {len(df):>10,} rows  {df.shape[1]} cols")

    tock("phase1a")
    print(f"Time: {_t_elapsed['phase1a']:.1f}s\n")

    # =========================================================================
    # PHASE 1b — Step 1 Hard Filters
    # =========================================================================
    print("=" * 62)
    print("  PHASE 1b — Step 1 Hard Filters")
    print("=" * 62)
    tick("phase1b")

    base = _dfs["candidate_base"]
    jobs_raw = _dfs["candidate_jobs"]
    total_input = len(base)

    # Safety-net dedup in Polars (parse loop already deduped via seen_ids)
    base_dd = base.unique(subset=["candidate_id"], keep="first")
    polars_dedup_drops = total_input - len(base_dd)

    # Drop: missing years_of_experience
    drop_missing_yoe = (
        base_dd
        .filter(pl.col("years_of_experience").is_null())
        .select("candidate_id")
        .with_columns(pl.lit("missing_years_of_experience").alias("drop_reason"))
    )

    # Drop: invalid years_of_experience (<= 0)
    drop_invalid_yoe = (
        base_dd
        .filter(pl.col("years_of_experience").is_not_null())
        .filter(pl.col("years_of_experience") <= 0)
        .select("candidate_id")
        .with_columns(pl.lit("invalid_years_of_experience").alias("drop_reason"))
    )

    # Drop: missing career_history — zero rows in candidate_jobs
    cids_with_jobs = jobs_raw.select("candidate_id").unique()
    drop_missing_career = (
        base_dd
        .filter(pl.col("years_of_experience").is_not_null())
        .filter(pl.col("years_of_experience") > 0)
        .filter(~pl.col("candidate_id").is_in(cids_with_jobs["candidate_id"]))
        .select("candidate_id")
        .with_columns(pl.lit("missing_career_history").alias("drop_reason"))
    )

    step1_dropped = pl.concat([drop_missing_yoe, drop_invalid_yoe, drop_missing_career])
    step1_survivors = (
        base_dd
        .filter(~pl.col("candidate_id").is_in(step1_dropped["candidate_id"]))
        .select("candidate_id")
    )

    step1_survivors.write_parquet(artifacts_dir / "step1_survivors.parquet")
    step1_dropped.write_parquet(artifacts_dir / "step1_dropped.parquet")

    _counts.update({
        "total_input": total_input,
        "polars_dedup_drops": polars_dedup_drops,
        "missing_yoe_drops": len(drop_missing_yoe),
        "invalid_yoe_drops": len(drop_invalid_yoe),
        "missing_career_drops": len(drop_missing_career),
        "step1_total_drops": len(step1_dropped) + polars_dedup_drops,
        "step1_survivors": len(step1_survivors),
    })

    tock("phase1b")
    print(f"  {'Total input':<38} {total_input:>8,}")
    print(f"  {'Dedup (parse-time)':<38} {raw_dups:>8,}")
    print(f"  {'Dedup (Polars safety net)':<38} {polars_dedup_drops:>8,}")
    print(f"  {'Missing years_of_experience':<38} {len(drop_missing_yoe):>8,}")
    print(f"  {'Invalid years_of_experience (<= 0)':<38} {len(drop_invalid_yoe):>8,}")
    print(f"  {'Missing career_history':<38} {len(drop_missing_career):>8,}")
    print(f"  {'-' * 50}")
    print(f"  {'Step 1 survivors':<38} {len(step1_survivors):>8,}")
    print(f"Time: {_t_elapsed['phase1b']:.2f}s\n")

    # =========================================================================
    # PHASE 2 — Step 2 Qualifier Gates
    # =========================================================================
    print("=" * 62)
    print("  PHASE 2 — Step 2 Qualifier Gates")
    print("=" * 62)
    tick("phase2")

    survivors_s1 = pl.read_parquet(artifacts_dir / "step1_survivors.parquet")

    jobs = (
        jobs_raw
        .filter(pl.col("candidate_id").is_in(survivors_s1["candidate_id"]))
        .with_columns([
            pl.col("start_date")
              .cast(pl.Utf8, strict=False)
              .str.to_date(format="%Y-%m-%d", strict=False)
              .alias("start_date"),
            pl.col("end_date")
              .cast(pl.Utf8, strict=False)
              .str.to_date(format="%Y-%m-%d", strict=False)
              .alias("end_date"),
            pl.col("is_current")
              .cast(pl.Boolean, strict=False)
              .fill_null(False)
              .alias("is_current"),
        ])
        .with_columns(
            pl.when(pl.col("end_date").is_null() | pl.col("is_current"))
              .then(pl.lit(today))
              .otherwise(pl.col("end_date"))
              .alias("end_date_filled")
        )
    )

    # ── Gate 1: Overlapping full-time roles ─────────────────────────────────
    print("  Gate 1 — overlapping full-time roles...")
    tick("gate1")

    gate1 = (
        jobs
        .filter(pl.col("start_date").is_not_null())
        .sort(["candidate_id", "start_date"])
        .with_columns(
            pl.col("end_date_filled")
              .shift(1)
              .over("candidate_id")
              .alias("prev_end")
        )
        .filter(pl.col("prev_end").is_not_null())
        .with_columns(
            ((pl.col("prev_end") - pl.col("start_date")).dt.total_days() / 30.44)
              .alias("overlap_months")
        )
        .filter(pl.col("overlap_months") > OVERLAP_THRESHOLD_MONTHS)
        .group_by("candidate_id")
        .agg(pl.col("overlap_months").max().alias("max_overlap"))
        .with_columns([
            pl.lit("overlapping_fulltime_roles").alias("drop_reason"),
            pl.concat_str([
                pl.lit("max_overlap="),
                pl.col("max_overlap").round(1).cast(pl.Utf8),
                pl.lit("mo"),
            ]).alias("drop_detail"),
        ])
        .select(["candidate_id", "drop_reason", "drop_detail"])
    )

    tock("gate1")
    _counts["gate1_drops"] = len(gate1)
    print(f"    -> {len(gate1):,} dropped  ({_t_elapsed['gate1']:.2f}s)")

    # ── Gate 2: Experience exceeds career span ──────────────────────────────
    print("  Gate 2 — experience exceeds career span...")
    tick("gate2")

    span = (
        jobs
        .filter(pl.col("start_date").is_not_null())
        .group_by("candidate_id")
        .agg([
            pl.col("start_date").min().alias("earliest_start"),
            pl.col("end_date_filled").max().alias("latest_end"),
        ])
        .with_columns(
            ((pl.col("latest_end") - pl.col("earliest_start")).dt.total_days() / 30.44)
              .alias("career_span_months")
        )
    )

    gate2 = (
        survivors_s1
        .join(
            _dfs["candidate_base"].select(["candidate_id", "years_of_experience"]),
            on="candidate_id", how="left",
        )
        .filter(pl.col("years_of_experience").is_not_null())
        .join(span, on="candidate_id", how="left")
        .filter(pl.col("career_span_months").is_not_null())
        .with_columns((pl.col("years_of_experience") * 12).alias("claimed_months"))
        .filter(pl.col("claimed_months") > (pl.col("career_span_months") + SPAN_BUFFER_MONTHS))
        .with_columns([
            pl.lit("experience_exceeds_career_span").alias("drop_reason"),
            pl.concat_str([
                pl.lit("claimed="),
                pl.col("claimed_months").round(0).cast(pl.Int32).cast(pl.Utf8),
                pl.lit("mo_span="),
                pl.col("career_span_months").round(0).cast(pl.Int32).cast(pl.Utf8),
                pl.lit("mo"),
            ]).alias("drop_detail"),
        ])
        .select(["candidate_id", "drop_reason", "drop_detail"])
    )

    tock("gate2")
    _counts["gate2_drops"] = len(gate2)
    print(f"    -> {len(gate2):,} dropped  ({_t_elapsed['gate2']:.2f}s)")

    # ── Gate 3: Consulting-only career ──────────────────────────────────────
    print("  Gate 3 — consulting-only career...")
    tick("gate3")

    gate3 = (
        jobs
        .with_columns(
            pl.col("company")
              .cast(pl.Utf8, strict=False)
              .str.to_lowercase()
              .str.strip_chars()
              .str.contains(CONSULTING_PATTERN)
              .fill_null(False)  # null company -> not consulting
              .alias("is_consulting")
        )
        .group_by("candidate_id")
        .agg([
            pl.col("is_consulting").all().alias("all_consulting"),
            pl.col("candidate_id").count().alias("job_count"),
        ])
        .filter(pl.col("all_consulting"))
        .with_columns([
            pl.lit("consulting_only_career").alias("drop_reason"),
            pl.concat_str([
                pl.lit("all_"),
                pl.col("job_count").cast(pl.Utf8),
                pl.lit("_roles_consulting"),
            ]).alias("drop_detail"),
        ])
        .select(["candidate_id", "drop_reason", "drop_detail"])
    )

    tock("gate3")
    _counts["gate3_drops"] = len(gate3)
    print(f"    -> {len(gate3):,} dropped  ({_t_elapsed['gate3']:.2f}s)")

    # ── Merge all drops — Gate 1 reason wins on multi-gate candidates ───────
    step2_dropped = (
        pl.concat([gate1, gate2, gate3])
        .unique(subset=["candidate_id"], keep="first")
    )

    g1_ids = set(gate1["candidate_id"].to_list())
    g2_ids = set(gate2["candidate_id"].to_list())
    g3_ids = set(gate3["candidate_id"].to_list())
    multi_gate_count = (len(g1_ids) + len(g2_ids) + len(g3_ids)) - len(g1_ids | g2_ids | g3_ids)

    survivors_s2 = survivors_s1.filter(
        ~pl.col("candidate_id").is_in(step2_dropped["candidate_id"])
    )

    survivors_s2.write_parquet(artifacts_dir / "survivors.parquet")
    step2_dropped.write_parquet(artifacts_dir / "step2_dropped.parquet")

    _counts.update({
        "step2_input": len(survivors_s1),
        "step2_multi_gate": multi_gate_count,
        "step2_total_drops": len(step2_dropped),
        "step2_survivors": len(survivors_s2),
    })

    tock("phase2")

    print(f"\n  {'Step 1 input':<38} {_counts['step2_input']:>8,}")
    print(f"  {'Gate 1 drops (overlap > 2mo)':<38} {_counts['gate1_drops']:>8,}")
    print(f"  {'Gate 2 drops (exp > span+36m)':<38} {_counts['gate2_drops']:>8,}")
    print(f"  {'Gate 3 drops (consulting-only)':<38} {_counts['gate3_drops']:>8,}")
    print(f"  {'Multi-gate overlaps (merged away)':<38} {_counts['step2_multi_gate']:>8,}")
    print(f"  {'-' * 50}")
    print(f"  {'Step 2 survivors':<38} {_counts['step2_survivors']:>8,}")
    print(f"Time: {_t_elapsed['phase2']:.2f}s\n")

    print("=" * 62)
    print(f"  DONE — artifacts written to {artifacts_dir}")
    print("=" * 62)
    for fname in [
        "candidate_base.parquet", "candidate_jobs.parquet", "candidate_skills.parquet",
        "candidate_education.parquet", "candidate_redrob.parquet",
        "step1_survivors.parquet", "step1_dropped.parquet",
        "survivors.parquet", "step2_dropped.parquet",
    ]:
        print(f"  - {fname}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/candidates.jsonl"),
        help="Path to raw candidates.jsonl or .jsonl.gz (default: dataset/candidates.jsonl)",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("dataset/artifacts"),
        help="Output directory for parquet artifacts (default: dataset/artifacts)",
    )
    parser.add_argument(
        "--reference-date", type=str, default="2026-06-25",
        help="Fixed reference 'today' date, YYYY-MM-DD, for reproducibility (default: 2026-06-25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    today = date.fromisoformat(args.reference_date)
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    run(args.dataset, args.artifacts_dir, today)
