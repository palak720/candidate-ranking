"""
Validate a candidate-ranking submission CSV.

Usage:
  python validate_submission.py output/submission.csv
  python validate_submission.py output/submission.csv --expected-rows 50
"""

import argparse
import csv
import sys


REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def _parse_rank(value, row_number, errors):
    try:
        rank = int(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: rank must be an integer")
        return None
    if rank < 1:
        errors.append(f"row {row_number}: rank must be >= 1")
    return rank


def _parse_score(value, row_number, errors):
    try:
        score = float(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: score must be numeric")
        return None
    if score < 0 or score > 1:
        errors.append(f"row {row_number}: score must be between 0 and 1")
    return score


def validate(path, expected_rows=100):
    errors = []

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except FileNotFoundError:
        return [f"file not found: {path}"]

    if fieldnames != REQUIRED_COLUMNS:
        errors.append(
            "CSV header must be exactly: " + ",".join(REQUIRED_COLUMNS)
        )

    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    ranks = []
    candidate_ids = []
    scores = []

    for row_number, row in enumerate(rows, start=2):
        candidate_id = (row.get("candidate_id") or "").strip()
        reasoning = (row.get("reasoning") or "").strip()

        if not candidate_id:
            errors.append(f"row {row_number}: candidate_id is required")
        if not reasoning:
            errors.append(f"row {row_number}: reasoning is required")

        rank = _parse_rank(row.get("rank"), row_number, errors)
        score = _parse_score(row.get("score"), row_number, errors)

        if candidate_id:
            candidate_ids.append(candidate_id)
        if rank is not None:
            ranks.append(rank)
        if score is not None:
            scores.append(score)

    duplicate_ids = sorted(
        candidate_id for candidate_id in set(candidate_ids) if candidate_ids.count(candidate_id) > 1
    )
    if duplicate_ids:
        errors.append("duplicate candidate_id values: " + ", ".join(duplicate_ids[:10]))

    expected_ranks = set(range(1, expected_rows + 1))
    actual_ranks = set(ranks)
    if actual_ranks != expected_ranks:
        missing = sorted(expected_ranks - actual_ranks)
        extra = sorted(actual_ranks - expected_ranks)
        if missing:
            errors.append("missing ranks: " + ", ".join(map(str, missing[:10])))
        if extra:
            errors.append("unexpected ranks: " + ", ".join(map(str, extra[:10])))

    if len(ranks) != len(set(ranks)):
        errors.append("rank values must be unique")

    for index in range(1, len(scores)):
        if scores[index] > scores[index - 1]:
            errors.append(
                f"scores must be monotonically non-increasing; row {index + 2} is higher than previous row"
            )
            break

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate candidate-ranking CSV output")
    parser.add_argument("csv_path")
    parser.add_argument("--expected-rows", type=int, default=100)
    args = parser.parse_args()

    errors = validate(args.csv_path, expected_rows=args.expected_rows)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Validation passed: {args.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
