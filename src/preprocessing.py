"""
preprocessing.py
================
Load, clean and normalise candidate profiles from JSONL / JSON.
"""

import json
import re
from datetime import date, datetime
from typing import List, Dict, Any, Iterator

TODAY = date.today()

SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "llm": "large language models", "k8s": "kubernetes",
    "sklearn": "scikit-learn", "tf": "tensorflow",
    "hf": "huggingface", "st": "sentence-transformers",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25, "intermediate": 0.55, "advanced": 0.80, "expert": 1.0
}

def normalize_skill_name(name: str) -> str:
    n = name.lower().strip()
    return SKILL_ALIASES.get(n, n)

def days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999

def build_candidate_text(c: Dict[str, Any]) -> str:
    """
    Flattens a candidate profile into a single text blob for embedding.
    This is the most important function — richer text = better semantic match.
    """
    parts = []

    p = c.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(f"{p.get('current_title','')} at {p.get('current_company','')}")
    parts.append(f"Industry: {p.get('current_industry','')}")
    parts.append(f"Experience: {p.get('years_of_experience', 0)} years")

    for ch in c.get("career_history", []):
        parts.append(f"{ch.get('title','')} at {ch.get('company','')} ({ch.get('industry','')}) — {ch.get('description','')}")

    for edu in c.get("education", []):
        parts.append(f"{edu.get('degree','')} in {edu.get('field_of_study','')} from {edu.get('institution','')} ({edu.get('tier','')})")

    skill_strs = []
    for sk in c.get("skills", []):
        name = normalize_skill_name(sk.get("name", ""))
        prof = sk.get("proficiency", "beginner")
        skill_strs.append(f"{name}({prof})")
    parts.append("Skills: " + ", ".join(skill_strs))

    for cert in c.get("certifications", []):
        parts.append(f"Certified: {cert.get('name','')} by {cert.get('issuer','')}")

    # Skill assessment scores from redrob signals
    sigs = c.get("redrob_signals", {})
    assessments = sigs.get("skill_assessment_scores", {})
    if assessments:
        parts.append("Assessments: " + ", ".join(f"{k}={v:.0f}" for k, v in assessments.items()))

    return " | ".join(filter(None, parts))

def build_skills_set(c: Dict[str, Any]) -> Dict[str, float]:
    """Returns {normalized_skill: weighted_score} for a candidate."""
    result = {}
    for sk in c.get("skills", []):
        name = normalize_skill_name(sk.get("name", ""))
        prof_w = PROFICIENCY_WEIGHT.get(sk.get("proficiency", "beginner"), 0.25)
        endr   = min(sk.get("endorsements", 0) / 50.0, 1.0)   # cap at 50
        dur    = min(sk.get("duration_months", 0) / 36.0, 1.0) # cap at 3yr
        score  = 0.5 * prof_w + 0.25 * endr + 0.25 * dur
        result[name] = max(result.get(name, 0), score)

    # Boost skills that have assessment scores
    sigs = c.get("redrob_signals", {})
    for skill, ascore in sigs.get("skill_assessment_scores", {}).items():
        nk = normalize_skill_name(skill)
        boost = ascore / 100.0
        result[nk] = max(result.get(nk, 0), result.get(nk, 0) * 0.7 + boost * 0.3)

    return result

def load_candidates_jsonl(path: str, limit: int = None) -> Iterator[Dict]:
    """Stream candidates from a large JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    break
            except json.JSONDecodeError:
                continue

def load_candidates_json(path: str) -> List[Dict]:
    """Load candidates from a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
