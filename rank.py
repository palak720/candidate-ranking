"""
rank.py - Complete pipeline, fully offline (CPU only, no API, no network)
=========================================================================
Satisfies submission_spec.docx constraints:
  - Exactly 100 rows, ranks 1-100 each exactly once
  - Score monotonically non-increasing
  - No external API calls
  - CPU only

Usage:
  python rank.py --candidates data/sample_candidates.json --jd data/job_description.txt
  python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import embeddings as EMB
from bias_audit import audit as run_bias_audit
from jd_parser import parse_jd
from preprocessing import (
    build_candidate_text,
    build_skills_set,
    load_candidates_json,
    load_candidates_jsonl,
)
from scoring import (
    behavioral_score,
    cold_start_penalty,
    final_score,
    generate_reasoning,
    profile_score,
    semantic_score as calc_sem,
)
from reranking import blend_rerank_score, precision_rerank_score

TOP_N = 100


def load_jd_text(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw).get("description", raw)
    except Exception:
        return raw


def load_candidates(path, limit=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return list(load_candidates_jsonl(path, limit=limit))
    data = load_candidates_json(path)
    return data[:limit] if limit else data


def run(candidates_path, jd_path, output_path, use_cache=True, use_hybrid=True, limit=None):
    t0 = time.time()
    os.makedirs("output", exist_ok=True)

    print("\n[1/7] Parsing Job Description ...")
    jd_text = load_jd_text(jd_path)
    parsed_jd = parse_jd(jd_text)
    seniority = parsed_jd.get("seniority", "mid")
    print(f"      Seniority  : {seniority}")
    print(f"      Work mode  : {parsed_jd['work_mode']}")
    print(f"      Exp range  : {parsed_jd['experience']}")
    print(f"      Req skills : {parsed_jd['required_skills'][:8]}")

    print("\n[2/7] Loading candidates ...")
    candidates = load_candidates(candidates_path, limit=limit)
    n_candidates = len(candidates)
    print(f"      Loaded {n_candidates:,} candidates")

    cand_texts = [build_candidate_text(candidate) for candidate in candidates]
    cand_skills = [build_skills_set(candidate) for candidate in candidates]
    cand_ids = [candidate["candidate_id"] for candidate in candidates]

    print("\n[3/7] Building semantic index (TF-IDF + LSA, fully offline) ...")
    if use_cache and EMB.cache_exists():
        try:
            cached_ids, _, candidate_matrix = EMB.load_cache()
            if cached_ids != cand_ids:
                raise ValueError("Cache mismatch - rebuilding")
            print("      Cache HIT")
        except Exception as exc:
            print(f"      {exc}")
            EMB._vectorizer, EMB._svd, candidate_matrix = EMB.fit_and_embed(cand_texts)
            EMB.save_cache(cand_ids, cand_texts, candidate_matrix, None)
    else:
        EMB._vectorizer, EMB._svd, candidate_matrix = EMB.fit_and_embed(cand_texts)
        if use_cache:
            EMB.save_cache(cand_ids, cand_texts, candidate_matrix, None)

    jd_combined = jd_text + " " + " ".join(parsed_jd["all_skills"])
    jd_embedding = EMB.transform_text([jd_combined], EMB._vectorizer, EMB._svd)[0]

    print("\n[4/7] Hybrid retrieval (LSA cosine + BM25 RRF) ...")
    retrieve_k = min(n_candidates, 1000)

    sem_scores_raw, sem_ids = EMB.semantic_search(
        jd_embedding,
        candidate_matrix,
        top_k=retrieve_k,
    )
    sem_score_map = {
        int(idx): float(score) for idx, score in zip(sem_ids, sem_scores_raw)
    }

    if use_hybrid and n_candidates > 1:
        bm25_idx = EMB.build_bm25(cand_texts)
        bm25_scores = EMB.bm25_search(bm25_idx, jd_combined)
        fused = EMB.rrf_fusion(sem_scores_raw, sem_ids, bm25_scores, n_candidates)
        pool_idxs = [idx for idx, _ in fused[:retrieve_k]]
    else:
        pool_idxs = list(sem_ids[:retrieve_k])

    print(f"      Retrieval pool: {len(pool_idxs)}")

    print("\n[5/7] Multi-signal scoring (Semantic + Profile + Behavioral) ...")
    scored = []
    for orig_idx in pool_idxs:
        orig_idx = int(orig_idx)
        if orig_idx < 0 or orig_idx >= n_candidates:
            continue
        candidate = candidates[orig_idx]
        candidate_skills = cand_skills[orig_idx]
        cos_sim = sem_score_map.get(orig_idx, 0.0)

        sem_score = calc_sem(max(0.0, float(cos_sim)))
        prof_score = profile_score(candidate, candidate_skills, parsed_jd)
        beh_score = behavioral_score(candidate, parsed_jd)
        cs_mult = cold_start_penalty(candidate)
        score = final_score(sem_score, prof_score, beh_score, seniority, cs_mult)
        reason = generate_reasoning(
            candidate,
            sem_score,
            prof_score,
            beh_score,
            candidate_skills,
            parsed_jd,
        )

        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "orig_idx": orig_idx,
                "score": score,
                "sem": sem_score,
                "prof": prof_score,
                "beh": beh_score,
                "reasoning": reason,
            }
        )

    scored.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    included = {row["candidate_id"] for row in scored}
    if len(scored) < TOP_N:
        print(f"      Padding to {TOP_N} ...")
        for index, candidate in enumerate(candidates):
            if candidate["candidate_id"] in included:
                continue
            candidate_skills = cand_skills[index]
            sem_score = calc_sem(max(0.0, sem_score_map.get(index, 0.0)))
            prof_score = profile_score(candidate, candidate_skills, parsed_jd)
            beh_score = behavioral_score(candidate, parsed_jd)
            cs_mult = cold_start_penalty(candidate)
            score = final_score(sem_score, prof_score, beh_score, seniority, cs_mult)
            reason = generate_reasoning(
                candidate,
                sem_score,
                prof_score,
                beh_score,
                candidate_skills,
                parsed_jd,
            )
            scored.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "orig_idx": index,
                    "score": score,
                    "sem": sem_score,
                    "prof": prof_score,
                    "beh": beh_score,
                    "reasoning": reason,
                }
            )
            included.add(candidate["candidate_id"])
            if len(scored) >= TOP_N:
                break
        scored.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    print("\n[6/7] Precision re-ranking top candidates ...")
    rerank_window = min(len(scored), 50)
    for row in scored[:rerank_window]:
        candidate_index = row["orig_idx"]
        rerank_score = precision_rerank_score(
            jd_text,
            cand_texts[candidate_index],
            candidates[candidate_index],
            cand_skills[candidate_index],
            parsed_jd,
        )
        row["rerank"] = rerank_score
        row["score"] = blend_rerank_score(row["score"], rerank_score)
    scored.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    top100 = scored[:TOP_N]

    for index in range(1, len(top100)):
        if top100[index]["score"] > top100[index - 1]["score"]:
            top100[index]["score"] = top100[index - 1]["score"]

    for rank_index, row in enumerate(top100, start=1):
        row["rank"] = rank_index

    print("\n[7/7] Writing outputs and trust audit ...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        for row in top100:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "reasoning": row["reasoning"],
                }
            )

    detailed = output_path.replace(".csv", "_detailed.json")
    with open(detailed, "w", encoding="utf-8") as f:
        json.dump(
            [{key: value for key, value in row.items() if key != "orig_idx"} for row in top100],
            f,
            indent=2,
        )

    audit_report = run_bias_audit(candidates, top100)
    audit_path = os.path.join(os.path.dirname(output_path) or ".", "bias_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Candidates loaded    : {n_candidates:,}")
    print(f"  Output rows          : {len(top100)}")
    print(f"  Submission CSV       : {output_path}")
    print(f"  Detailed JSON        : {detailed}")
    print(f"  Bias audit JSON      : {audit_path}")
    print("\n  Top-5:")
    for row in top100[:5]:
        print(
            f"    #{row['rank']}  {row['candidate_id']}  {row['score']:.4f}  |  "
            f"{row['reasoning'][:65]}"
        )
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Candidate Ranking System")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", required=True)
    parser.add_argument("--out", default="output/submission.csv")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(
        args.candidates,
        args.jd,
        args.out,
        use_cache=not args.no_cache,
        use_hybrid=not args.no_hybrid,
        limit=args.limit,
    )
