"""
bias_audit.py
=============
Post-ranking fairness check.
Analyses score distribution across location, experience buckets.
"""

import json
import csv
from collections import defaultdict
from typing import List, Dict, Any


def exp_bucket(yoe: float) -> str:
    if yoe < 2:   return "0-2 yrs"
    elif yoe < 5: return "2-5 yrs"
    elif yoe < 9: return "5-9 yrs"
    else:         return "9+ yrs"


def audit(candidates: List[Dict[str, Any]],
          results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns fairness metrics across experience buckets and countries.
    """
    cand_map = {c["candidate_id"]: c for c in candidates}

    exp_scores   = defaultdict(list)
    country_scores = defaultdict(list)

    for r in results:
        cid  = r["candidate_id"]
        cand = cand_map.get(cid)
        if not cand:
            continue
        yoe     = cand["profile"]["years_of_experience"]
        country = cand["profile"]["country"]
        score   = r["score"]

        exp_scores[exp_bucket(yoe)].append(score)
        country_scores[country].append(score)

    def summarise(bucket_scores):
        return {
            bucket: {
                "count":   len(scores),
                "avg":     round(sum(scores)/len(scores), 4),
                "max":     round(max(scores), 4),
                "min":     round(min(scores), 4),
            }
            for bucket, scores in bucket_scores.items()
            if scores
        }

    report = {
        "by_experience": summarise(exp_scores),
        "by_country":    summarise(country_scores),
        "total_ranked":  len(results),
    }

    print("\n[Bias Audit] Experience distribution:")
    for bucket, stats in sorted(report["by_experience"].items()):
        print(f"  {bucket:10s}  count={stats['count']:4d}  avg={stats['avg']:.4f}")

    print("\n[Bias Audit] Top countries:")
    top_countries = sorted(report["by_country"].items(),
                           key=lambda x: x[1]["count"], reverse=True)[:8]
    for country, stats in top_countries:
        print(f"  {country:20s}  count={stats['count']:4d}  avg={stats['avg']:.4f}")

    return report


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_candidates_json

    candidates = load_candidates_json("data/sample_candidates.json")

    # Load results if they exist
    results_path = "output/submission_detailed.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        report = audit(candidates, results)
        with open("output/bias_audit.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n  Bias audit saved → output/bias_audit.json")
    else:
        print("Run rank.py first to generate results.")
