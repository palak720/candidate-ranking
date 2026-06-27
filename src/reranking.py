"""
Precision re-ranking helpers.

This is an offline substitute for the architecture's cross-encoder layer. It
adds a second-pass relevance score over the retrieved shortlist using exact
requirement coverage, phrase overlap, and role-intent signals.
"""

import re
from typing import Any, Dict


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _phrase_hit(phrase: str, text: str) -> bool:
    return phrase.lower() in text.lower()


def precision_rerank_score(
    jd_text: str,
    candidate_text: str,
    candidate: Dict[str, Any],
    candidate_skills: Dict[str, float],
    parsed_jd: Dict[str, Any],
) -> float:
    """
    Return a 0-100 precision score for a retrieved candidate.

    The score intentionally rewards explicit must-have alignment more than broad
    semantic similarity, which mirrors the role of a cross-encoder re-ranker.
    """
    required = [skill.lower() for skill in parsed_jd.get("required_skills", [])]
    nice = [skill.lower() for skill in parsed_jd.get("nice_to_have", [])]

    required_total = max(len(required), 1)
    required_hits = sum(1 for skill in required if skill in candidate_skills)
    required_coverage = required_hits / required_total

    phrase_total = max(len(required), 1)
    phrase_hits = sum(1 for skill in required if _phrase_hit(skill, candidate_text))
    phrase_coverage = phrase_hits / phrase_total

    nice_bonus = min(sum(1 for skill in nice if skill in candidate_skills) / 5.0, 1.0)

    jd_tokens = _tokens(jd_text)
    candidate_tokens = _tokens(candidate_text)
    overlap = len(jd_tokens & candidate_tokens) / max(len(jd_tokens), 1)

    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()
    role_text = f"{title} {summary}"
    role_intent = 0.0
    if any(term in role_text for term in ["ranking", "retrieval", "search", "recommendation"]):
        role_intent += 0.45
    if any(term in role_text for term in ["ml", "ai", "machine learning", "nlp", "llm"]):
        role_intent += 0.35
    if any(term in title for term in ["engineer", "scientist", "architect"]):
        role_intent += 0.20

    score = (
        0.42 * required_coverage
        + 0.22 * phrase_coverage
        + 0.16 * overlap
        + 0.12 * min(role_intent, 1.0)
        + 0.08 * nice_bonus
    )
    return round(min(score * 100.0, 100.0), 4)


def blend_rerank_score(base_score: float, rerank_score: float, weight: float = 0.12) -> float:
    """Blend a normalized 0-1 base score with a 0-100 re-rank score."""
    rerank_normalized = rerank_score / 100.0
    return round((1.0 - weight) * base_score + weight * rerank_normalized, 4)
