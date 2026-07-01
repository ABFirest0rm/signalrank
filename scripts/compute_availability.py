"""
Redrob Ranker — Step 3
Availability Multiplier

Inputs:
    survivors.parquet          candidate_id list from Steps 1-2
    candidate_base.parquet     profile fields, including country
    candidate_redrob.parquet   behavioral/platform signals

Output:
    availability_scores.parquet

Design:
    availability_multiplier =
        country_gate x recency_gate x engagement_score

    engagement_score =
        response_score  x 0.28
      + notice_score    x 0.30
      + intent_score    x 0.18
      + interview_score x 0.12
      + work_mode_score x 0.12

Notes:
    - City/location and willing_to_relocate are intentionally excluded.
    - JD says Pune/Noida preferred but flexible, no fixed office cadence,
      quarterly travel/offsites, and outside India is case-by-case with no visa sponsorship.
    - Step 3 is a soft filter, not a hard drop.

Usage:
    python scripts/compute_availability.py \
        --artifacts-dir dataset/artifacts \
        --reference-date 2026-06-25
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

FLOOR = 0.05

RATE_FLOOR = 0.10
RATE_CEIL = 0.90

# Engagement weights — observed behavior > self-reported checkbox
W_RESPONSE = 0.28
W_NOTICE = 0.30
W_INTENT = 0.18
W_INTERVIEW = 0.12
W_WORK_MODE = 0.12

assert abs(
    W_RESPONSE + W_NOTICE + W_INTENT + W_INTERVIEW + W_WORK_MODE - 1.0
) < 1e-9, "Engagement weights must sum to 1.00"

OUTPUT_COLS = [
    "candidate_id",
    # Final result
    "availability_multiplier",
    "availability_raw",
    # Multiplicative gates
    "country_gate",
    "recency_gate",
    # Engagement
    "engagement_score",
    "response_score",
    "notice_score",
    "intent_score",
    "interview_score",
    "work_mode_score",
    # Debug / explainability helpers
    "last_active_days_ago",
    "response_rate_flag",
    "country_risk_flag",
    "stale_activity_flag",
    "long_notice_flag",
    "not_open_flag",
    "weak_interview_flag",
    "remote_pref_flag",
]


# ── Load & Join ──────────────────────────────────────────────────────────────

def load_data(survivors_path: Path, base_path: Path, redrob_path: Path) -> pl.DataFrame:
    survivors = (
        pl.read_parquet(survivors_path)
        .select("candidate_id")
        .unique()
    )

    base = (
        pl.read_parquet(base_path)
        .select(["candidate_id", "country"])
    )

    redrob = (
        pl.read_parquet(redrob_path)
        .select([
            "candidate_id",
            "last_active_date",
            "open_to_work_flag",
            "recruiter_response_rate",
            "notice_period_days",
            "preferred_work_mode",
            "interview_completion_rate",
        ])
    )

    return survivors.join(base, on="candidate_id", how="left").join(redrob, on="candidate_id", how="left")


# ── Gate 1 — Country Gate ────────────────────────────────────────────────────
# City/location and willing_to_relocate are intentionally NOT used.
#   country == India       -> 1.00
#   country null/blank     -> 0.75
#   country != India        -> 0.35

def compute_country_gate(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("country")
          .cast(pl.Utf8, strict=False)
          .str.to_lowercase()
          .str.strip_chars()
          .alias("_country_norm")
    )

    country_unknown = (
        pl.col("_country_norm").is_null()
        | (pl.col("_country_norm") == "")
    )

    df = df.with_columns(
        pl.when(pl.col("_country_norm") == "india")
          .then(pl.lit(1.00))
          .when(country_unknown)
          .then(pl.lit(0.75))
          .otherwise(pl.lit(0.35))
          .cast(pl.Float64)
          .alias("country_gate")
    )

    return df.drop("_country_norm")


# ── Gate 2 — Recency Gate ────────────────────────────────────────────────────
#   <=14 days       -> 1.00
#   15-30 days     -> 0.95
#   31-60 days     -> 0.85
#   61-180 days    -> 0.70
#   181-365 days   -> 0.45
#   >365 days      -> 0.20
#   null           -> 0.30

def compute_recency_gate(df: pl.DataFrame, today: date) -> pl.DataFrame:
    today_lit = pl.lit(today)

    df = df.with_columns(
        pl.col("last_active_date")
          .cast(pl.Utf8, strict=False)
          .str.to_date(format="%Y-%m-%d", strict=False)
          .alias("_lad")
    )

    df = df.with_columns(
        pl.when(pl.col("_lad").is_null())
          .then(pl.lit(None, dtype=pl.Int64))
          .otherwise((today_lit - pl.col("_lad")).dt.total_days().cast(pl.Int64))
          .alias("last_active_days_ago")
    )

    days = pl.col("last_active_days_ago")

    df = df.with_columns(
        pl.when(pl.col("_lad").is_null())
          .then(pl.lit(0.30))
          .when(days <= 14)
          .then(pl.lit(1.00))
          .when(days <= 30)
          .then(pl.lit(0.95))
          .when(days <= 60)
          .then(pl.lit(0.85))
          .when(days <= 180)
          .then(pl.lit(0.70))
          .when(days <= 365)
          .then(pl.lit(0.45))
          .otherwise(pl.lit(0.20))
          .cast(pl.Float64)
          .alias("recency_gate")
    )

    return df.drop("_lad")


# ── Engagement Signal 1 — Recruiter Response Rate ────────────────────────────
#   null              -> 0.75
#   clamped < 0.20    -> 0.30
#   0.20-0.40         -> 0.65
#   0.40-0.60         -> 0.85
#   >0.60             -> 1.00

def compute_response_score(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("recruiter_response_rate").cast(pl.Float64, strict=False).alias("_rr")
    )

    clamped = pl.col("_rr").clip(RATE_FLOOR, RATE_CEIL)

    df = df.with_columns(
        pl.when(pl.col("_rr").is_null())
          .then(pl.lit(0.75))
          .when(clamped < 0.20)
          .then(pl.lit(0.30))
          .when(clamped < 0.40)
          .then(pl.lit(0.65))
          .when(clamped <= 0.60)
          .then(pl.lit(0.85))
          .otherwise(pl.lit(1.00))
          .cast(pl.Float64)
          .alias("response_score")
    )

    return df.drop("_rr")


# ── Engagement Signal 2 — Notice Period ──────────────────────────────────────
#   null      -> 0.80
#   <=30       -> 1.00
#   31-60     -> 0.85
#   61-90     -> 0.70
#   >90       -> 0.55

def compute_notice_score(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("notice_period_days").cast(pl.Int64, strict=False).alias("_notice_days")
    )

    days = pl.col("_notice_days")

    df = df.with_columns(
        pl.when(days.is_null())
          .then(pl.lit(0.80))
          .when(days <= 30)
          .then(pl.lit(1.00))
          .when(days <= 60)
          .then(pl.lit(0.85))
          .when(days <= 90)
          .then(pl.lit(0.70))
          .otherwise(pl.lit(0.55))
          .cast(pl.Float64)
          .alias("notice_score")
    )

    return df.drop("_notice_days")


# ── Engagement Signal 3 — Intent / Open to Work ──────────────────────────────
#   true / 1 / "true"       -> 1.00
#   false / 0 / "false"     -> 0.65
#   null / blank / unknown  -> 0.75

def compute_intent_score(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("open_to_work_flag")
          .cast(pl.Utf8, strict=False)
          .str.to_lowercase()
          .str.strip_chars()
          .alias("_otw_str")
    )

    is_true = pl.col("_otw_str").is_in(["true", "1", "yes", "y"])
    is_false = pl.col("_otw_str").is_in(["false", "0", "no", "n"])

    df = df.with_columns(
        pl.when(is_true)
          .then(pl.lit(1.00))
          .when(is_false)
          .then(pl.lit(0.65))
          .otherwise(pl.lit(0.75))
          .cast(pl.Float64)
          .alias("intent_score")
    )

    return df.drop("_otw_str")


# ── Engagement Signal 4 — Interview Completion Rate ─────────────────────────
#   null              -> 0.80
#   clamped >= 0.80    -> 1.00
#   0.50-0.79         -> 0.90
#   <0.50             -> 0.55

def compute_interview_score(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("interview_completion_rate").cast(pl.Float64, strict=False).alias("_icr")
    )

    clamped = pl.col("_icr").clip(RATE_FLOOR, RATE_CEIL)

    df = df.with_columns(
        pl.when(pl.col("_icr").is_null())
          .then(pl.lit(0.80))
          .when(clamped >= 0.80)
          .then(pl.lit(1.00))
          .when(clamped >= 0.50)
          .then(pl.lit(0.90))
          .otherwise(pl.lit(0.55))
          .cast(pl.Float64)
          .alias("interview_score")
    )

    return df.drop("_icr")


# ── Engagement Signal 5 — Work Mode ──────────────────────────────────────────
#   onsite / hybrid / flexible  -> 1.00
#   remote                      -> 0.80
#   null / unknown              -> 0.90

def compute_work_mode_score(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("preferred_work_mode")
          .cast(pl.Utf8, strict=False)
          .str.to_lowercase()
          .str.strip_chars()
          .alias("_wm_norm")
    )

    df = df.with_columns(
        pl.when(pl.col("_wm_norm").is_null() | (pl.col("_wm_norm") == ""))
          .then(pl.lit(0.90))
          .when(pl.col("_wm_norm").is_in(["onsite", "hybrid", "flexible"]))
          .then(pl.lit(1.00))
          .when(pl.col("_wm_norm") == "remote")
          .then(pl.lit(0.80))
          .otherwise(pl.lit(0.90))
          .cast(pl.Float64)
          .alias("work_mode_score")
    )

    return df.drop("_wm_norm")


def compute_engagement_score(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("response_score") * W_RESPONSE
            + pl.col("notice_score") * W_NOTICE
            + pl.col("intent_score") * W_INTENT
            + pl.col("interview_score") * W_INTERVIEW
            + pl.col("work_mode_score") * W_WORK_MODE
        )
        .cast(pl.Float64)
        .alias("engagement_score")
    )


def compute_multiplier(df: pl.DataFrame) -> pl.DataFrame:
    raw = pl.col("country_gate") * pl.col("recency_gate") * pl.col("engagement_score")

    return df.with_columns([
        raw.cast(pl.Float64).alias("availability_raw"),
        pl.max_horizontal(raw, pl.lit(FLOOR)).cast(pl.Float64).alias("availability_multiplier"),
    ])


def compute_flags(df: pl.DataFrame) -> pl.DataFrame:
    rr = pl.col("recruiter_response_rate").cast(pl.Float64, strict=False)
    clamped_rr = rr.clip(RATE_FLOOR, RATE_CEIL)

    notice_days = pl.col("notice_period_days").cast(pl.Int64, strict=False)

    return df.with_columns([
        pl.when(rr.is_null())
          .then(pl.lit(False))
          .when(clamped_rr < 0.40)
          .then(pl.lit(True))
          .otherwise(pl.lit(False))
          .alias("response_rate_flag"),

        (pl.col("country_gate") < 1.00).alias("country_risk_flag"),

        (
            pl.col("last_active_days_ago").is_null()
            | (pl.col("last_active_days_ago") > 180)
        ).alias("stale_activity_flag"),

        (
            notice_days.is_not_null() & (notice_days > 60)
        ).alias("long_notice_flag"),

        (pl.col("intent_score") == 0.65).alias("not_open_flag"),

        (pl.col("interview_score") == 0.55).alias("weak_interview_flag"),

        (pl.col("work_mode_score") == 0.80).alias("remote_pref_flag"),
    ])


def print_summary(df: pl.DataFrame) -> None:
    total = len(df)

    country_india = df.filter(pl.col("country_gate") == 1.00).height
    country_unknown = df.filter(pl.col("country_gate") == 0.75).height
    outside_india = df.filter(pl.col("country_gate") == 0.35).height

    active_14 = df.filter(pl.col("recency_gate") == 1.00).height
    active_15_30 = df.filter(pl.col("recency_gate") == 0.95).height
    active_31_60 = df.filter(pl.col("recency_gate") == 0.85).height
    active_61_180 = df.filter(pl.col("recency_gate") == 0.70).height
    active_181_365 = df.filter(pl.col("recency_gate") == 0.45).height
    active_365p = df.filter(pl.col("recency_gate") == 0.20).height
    active_null = df.filter(pl.col("last_active_days_ago").is_null()).height

    low_response = df.filter(pl.col("response_rate_flag")).height
    long_notice = df.filter(pl.col("long_notice_flag")).height
    not_open = df.filter(pl.col("not_open_flag")).height
    weak_interview = df.filter(pl.col("weak_interview_flag")).height
    remote_pref = df.filter(pl.col("remote_pref_flag")).height

    availability = pl.col("availability_multiplier")

    strong = df.filter(availability >= 0.80).height
    moderate = df.filter((availability >= 0.50) & (availability < 0.80)).height
    weak = df.filter((availability >= 0.20) & (availability < 0.50)).height
    near_floor = df.filter((availability > FLOOR) & (availability < 0.20)).height
    at_floor = df.filter(availability == FLOOR).height
    floor_pct = at_floor / total * 100 if total else 0.0

    print()
    print("=" * 76)
    print("  STEP 3 — AVAILABILITY MULTIPLIER SUMMARY")
    print("=" * 76)
    print(f"  Total processed              : {total:>10,}")
    print()
    print("  Country gate:")
    print(f"    India                      : {country_india:>10,}")
    print(f"    Country unknown/null       : {country_unknown:>10,}")
    print(f"    Outside India              : {outside_india:>10,}")
    print()
    print("  Recency gate:")
    print(f"    Active <= 14 days           : {active_14:>10,}")
    print(f"    Active 15-30 days          : {active_15_30:>10,}")
    print(f"    Active 31-60 days          : {active_31_60:>10,}")
    print(f"    Active 61-180 days         : {active_61_180:>10,}")
    print(f"    Active 181-365 days        : {active_181_365:>10,}")
    print(f"    Active > 365 days          : {active_365p:>10,}")
    print(f"    Last active null           : {active_null:>10,}")
    print()
    print("  Engagement flags:")
    print(f"    Low response rate          : {low_response:>10,}  (< 0.40 after clamping)")
    print(f"    Long notice period         : {long_notice:>10,}  (> 60 days)")
    print(f"    Not open to work           : {not_open:>10,}  (explicit false)")
    print(f"    Weak interview completion  : {weak_interview:>10,}  (< 0.50)")
    print(f"    Remote preference          : {remote_pref:>10,}")
    print()
    print("  Multiplier distribution:")
    print(f"    1.00 - 0.80  strong        : {strong:>10,}  ({strong / total:.1%})")
    print(f"    0.80 - 0.50  moderate      : {moderate:>10,}  ({moderate / total:.1%})")
    print(f"    0.50 - 0.20  weak          : {weak:>10,}  ({weak / total:.1%})")
    print(f"    0.20 - 0.05  near-floor    : {near_floor:>10,}  ({near_floor / total:.1%})")
    print(f"    0.05         floor         : {at_floor:>10,}  ({floor_pct:.1f}%)")
    print()

    if floor_pct > 10.0:
        print(f"  WARNING: {floor_pct:.1f}% of candidates hit the floor — Step 3 may be too harsh")
    if total and strong / total < 0.05:
        print("  WARNING: <5% strong availability — check if gates are too aggressive")
    if total and strong / total > 0.60:
        print("  WARNING: >60% strong availability — check if gates are too lenient")

    print("=" * 76)
    print()


def run(artifacts_dir: Path, today: date) -> None:
    survivors_path = artifacts_dir / "survivors.parquet"
    base_path = artifacts_dir / "candidate_base.parquet"
    redrob_path = artifacts_dir / "candidate_redrob.parquet"
    output_path = artifacts_dir / "availability_scores.parquet"

    for p in [survivors_path, base_path, redrob_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required input: {p}\n"
                "Run scripts/parse_and_filter.py first."
            )

    print("Resolved paths:")
    print(f"  survivors : {survivors_path}")
    print(f"  base      : {base_path}")
    print(f"  redrob    : {redrob_path}")
    print(f"  output    : {output_path}")

    print(f"\nLoading data... TODAY = {today}")
    df = load_data(survivors_path, base_path, redrob_path)
    print(f"Rows after survivor + base + redrob join: {len(df):,}")

    print("\nComputing country + recency gates...")
    df = compute_country_gate(df)
    df = compute_recency_gate(df, today)

    print("Computing engagement scores...")
    df = compute_response_score(df)
    df = compute_notice_score(df)
    df = compute_intent_score(df)
    df = compute_interview_score(df)
    df = compute_work_mode_score(df)
    df = compute_engagement_score(df)

    print("Computing final multiplier + flags...")
    df = compute_multiplier(df)
    df = compute_flags(df)

    out = df.select(OUTPUT_COLS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)

    print(f"\nWrote output -> {output_path}")
    print_summary(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("dataset/artifacts"),
        help="Directory containing survivors.parquet, candidate_base.parquet, "
             "candidate_redrob.parquet, and where availability_scores.parquet will be written "
             "(default: dataset/artifacts)",
    )
    parser.add_argument(
        "--reference-date", type=str, default="2026-06-25",
        help="Fixed reference 'today' date, YYYY-MM-DD, for reproducibility (default: 2026-06-25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    today = date.fromisoformat(args.reference_date)
    run(args.artifacts_dir, today)
