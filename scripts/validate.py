"""
Redrob Ranker — Submission Validator

Checks a ranked output CSV against the required submission format and runs
a handful of sanity checks on scoring quality (medians, correlations,
floor rate, honeypot count) to help catch problems before final upload.

Usage:
    python scripts/validate.py output/team_xxx.csv \
        --candidate-base dataset/artifacts/candidate_base.parquet \
        --diagnostic output/top100_ranked_diagnostic.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLS = ['candidate_id', 'rank', 'score', 'reasoning']


def validate_submission_format(submission_path: Path, candidate_base_path: Path | None, expected_rows: int) -> bool:
    print("=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    ok = True

    print("\n[0] Submission CSV format:")
    sub = pd.read_csv(submission_path)

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if cond:
            print(f"  OK   {msg}")
        else:
            print(f"  FAIL {msg}")
            ok = False

    check(list(sub.columns) == EXPECTED_COLS, f"Column order == {EXPECTED_COLS} (got {list(sub.columns)})")
    check(len(sub) == expected_rows, f"Row count == {expected_rows} (got {len(sub)})")
    check(sub['rank'].tolist() == list(range(1, expected_rows + 1)), "Ranks are exactly 1..N in order")
    check(sub['candidate_id'].is_unique, "candidate_id values are unique")
    check(sub['rank'].is_unique, "rank values are unique")
    check(sub['score'].notna().all(), "No NaN scores")
    check(sub['reasoning'].notna().all(), "No NaN reasoning")
    check(sub['score'].is_monotonic_decreasing, "Scores are non-increasing with rank")

    if candidate_base_path is not None and candidate_base_path.exists():
        df_base = pd.read_parquet(candidate_base_path)
        if 'candidate_id' in df_base.columns:
            missing_ids = set(sub['candidate_id']) - set(df_base['candidate_id'])
            check(not missing_ids, f"All candidate_ids exist in candidate_base (missing: {list(missing_ids)[:5]})")

    return ok


def validate_diagnostics(diagnostic_path: Path) -> None:
    if not diagnostic_path.exists():
        print(f"\n(diagnostic CSV not found at {diagnostic_path} — skipping score-quality checks)")
        return

    diag = pd.read_csv(diagnostic_path)

    print("\n[1] must_have_score distribution:")
    q = diag['must_have_score'].quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    print(q.round(3).to_string())
    if diag['must_have_score'].median() > 0.50:
        print("  WARNING: Median > 0.5 — scoring may be too generous")
    else:
        print("  OK: Median < 0.5 — right-skewed as expected")

    print("\n[2] M1/M2 Pearson correlation:")
    corr = diag[['M1_score', 'M2_score']].corr().iloc[0, 1]
    print(f"    r = {corr:.3f}")
    if corr > 0.90:
        print("  WARNING: >0.90 — M1/M2 redundant, consider averaging before gmean")
    elif corr < 0.50:
        print("  WARNING: <0.50 — unexpectedly low, check query strings")
    else:
        print("  OK: In expected range")

    print("\n[3] Floor check:")
    floored = (diag['final_score'] == 0.10).sum()
    print(f"    Candidates floored at 0.10: {floored:,} ({floored / len(diag) * 100:.1f}%)")
    if floored > len(diag) * 0.10:
        print("  WARNING: >10% floored — check if multipliers are too aggressive")
    else:
        print("  OK: Floor rate looks fine")

    print("\n[4] Honeypot/trap check:")
    if 'trap_multiplier' in diag.columns:
        trap_count = int((diag['trap_multiplier'] < 1.0).sum())
        print(f"    Trap candidates in output: {trap_count}")
        if trap_count >= 10:
            print("  WARNING: >=10 honeypots — likely disqualification risk")
        elif trap_count > 0:
            print("  Some trap candidates remain; inspect diagnostic CSV")
        else:
            print("  OK: No trap candidates")
    else:
        print("  trap_multiplier column missing")

    print("\n[5] Top-10 spot-check:")
    cols = [c for c in ['rank', 'candidate_id', 'final_score_raw', 'M1_score', 'M2_score',
                         'M3_score', 'M4_score', 'best_matching_job_title'] if c in diag.columns]
    print(diag[cols].head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the submission CSV (e.g. output/team_xxx.csv)")
    parser.add_argument(
        "--candidate-base", type=Path, default=Path("dataset/artifacts/candidate_base.parquet"),
        help="Path to candidate_base.parquet, used to check candidate_id validity "
             "(default: dataset/artifacts/candidate_base.parquet)",
    )
    parser.add_argument(
        "--diagnostic", type=Path, default=None,
        help="Path to the diagnostic CSV for score-quality checks "
             "(default: <submission_dir>/top<rows>_ranked_diagnostic.csv, guessed)",
    )
    parser.add_argument(
        "--expected-rows", type=int, default=100,
        help="Expected number of rows in the submission (default: 100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.submission.exists():
        print(f"Submission file not found: {args.submission}", file=sys.stderr)
        sys.exit(1)

    diagnostic_path = args.diagnostic
    if diagnostic_path is None:
        diagnostic_path = args.submission.parent / f"top{args.expected_rows}_ranked_diagnostic.csv"

    ok = validate_submission_format(args.submission, args.candidate_base, args.expected_rows)
    validate_diagnostics(diagnostic_path)

    print()
    if ok:
        print("Validation complete: all required format checks passed.")
        sys.exit(0)
    else:
        print("Validation FAILED: fix the issues above before submitting.")
        sys.exit(1)
