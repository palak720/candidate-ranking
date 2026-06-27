"""
scoring.py — Updated with real data understanding
===================================================
Three signals based on actual dataset + submission_spec requirements:
  1. semantic_score   — cosine similarity (embedding-based)
  2. profile_score    — skills, experience, education explicit match
  3. behavioral_score — all 23 redrob_signals used correctly
"""

from datetime import date, datetime
from typing import Dict, Any

TODAY = date.today()

# ── Dynamic weights per seniority ────────────────────────────────
SENIORITY_WEIGHTS = {
    "intern":    {"semantic": 0.30, "profile": 0.25, "behavioral": 0.45},
    "junior":    {"semantic": 0.35, "profile": 0.30, "behavioral": 0.35},
    "entry":     {"semantic": 0.35, "profile": 0.30, "behavioral": 0.35},
    "mid":       {"semantic": 0.40, "profile": 0.35, "behavioral": 0.25},
    "senior":    {"semantic": 0.38, "profile": 0.45, "behavioral": 0.17},
    "lead":      {"semantic": 0.35, "profile": 0.50, "behavioral": 0.15},
    "staff":     {"semantic": 0.30, "profile": 0.55, "behavioral": 0.15},
    "principal": {"semantic": 0.30, "profile": 0.55, "behavioral": 0.15},
    "founding":  {"semantic": 0.35, "profile": 0.45, "behavioral": 0.20},
}
DEFAULT_WEIGHTS = {"semantic": 0.40, "profile": 0.35, "behavioral": 0.25}

EDU_TIER_SCORE = {
    "tier_1": 1.00, "tier_2": 0.80, "tier_3": 0.60,
    "tier_4": 0.40, "unknown": 0.30
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25, "intermediate": 0.55, "advanced": 0.80, "expert": 1.0
}


# ── 1. Semantic Score ─────────────────────────────────────────────
def semantic_score(cosine_sim: float) -> float:
    return max(0.0, min(1.0, float(cosine_sim))) * 100.0


# ── 2. Profile Score ─────────────────────────────────────────────
def profile_score(candidate: Dict[str, Any],
                  candidate_skills: Dict[str, float],
                  parsed_jd: Dict[str, Any]) -> float:
    """
    Sub-components (0-100 each):
      a) Required skills coverage (50%)
      b) Nice-to-have skills bonus (up to 15 pts)
      c) Experience years match   (25%)
      d) Education tier           (15%)
      e) Title relevance          (10%)
    """
    required = [s.lower() for s in parsed_jd.get("required_skills", [])]
    nice     = [s.lower() for s in parsed_jd.get("nice_to_have", [])]

    # a) Required skills — weighted by proficiency
    req_total = max(len(required), 1)
    req_hit   = sum(candidate_skills.get(sk, 0) for sk in required)
    req_score = (req_hit / req_total) * 100.0

    # Assessment score boost — from redrob_signals
    sigs        = candidate.get("redrob_signals", {})
    assessments = sigs.get("skill_assessment_scores", {})
    assess_boost = 0.0
    for skill, ascore in assessments.items():
        if skill.lower() in required:
            assess_boost += (ascore / 100.0) * 5.0
    req_score = min(req_score + assess_boost, 100.0)

    # b) Nice-to-have bonus
    nice_bonus = sum(candidate_skills.get(sk, 0) * 5.0 for sk in nice)
    nice_bonus = min(nice_bonus, 15.0)

    # c) Experience
    yoe     = candidate.get("profile", {}).get("years_of_experience", 0)
    exp_req = parsed_jd.get("experience", {"min": 0, "max": 50})
    lo, hi  = exp_req["min"], exp_req["max"]
    if lo <= yoe <= hi:
        exp_score = 100.0
    elif yoe < lo:
        exp_score = max(0.0, 100.0 - (lo - yoe) * 15.0)
    else:
        exp_score = max(60.0, 100.0 - (yoe - hi) * 5.0)

    # d) Education tier
    best_tier = "unknown"
    for edu in candidate.get("education", []):
        t = edu.get("tier", "unknown")
        if EDU_TIER_SCORE.get(t, 0) > EDU_TIER_SCORE.get(best_tier, 0):
            best_tier = t
    edu_score = EDU_TIER_SCORE.get(best_tier, 0.3) * 100.0

    # e) Title relevance
    title      = candidate.get("profile", {}).get("current_title", "").lower()
    seniority  = parsed_jd.get("seniority", "mid").lower()
    title_score = 60.0
    if seniority in title:
        title_score = 100.0
    elif any(w in title for w in ["engineer", "scientist", "ml", "ai", "data", "architect", "researcher"]):
        title_score = 85.0

    final = (0.50 * req_score +
             0.25 * exp_score +
             0.15 * edu_score +
             0.10 * title_score) + nice_bonus

    return round(min(final, 100.0), 4)


# ── 3. Behavioral Score ──────────────────────────────────────────
def behavioral_score(candidate: Dict[str, Any],
                     parsed_jd: Dict[str, Any]) -> float:
    """
    Uses all 23 redrob_signals.
    Sub-components:
      a) Recency & availability   (30%)
      b) Engagement activity      (25%)
      c) Reliability              (25%)
      d) Intent & job-fit         (20%)
    """
    sigs = candidate.get("redrob_signals", {})

    # ── a) Recency & availability ────────────────────────────────
    days_ago = _days_since(sigs.get("last_active_date", "2020-01-01"))
    if days_ago <= 7:    recency = 100.0
    elif days_ago <= 30: recency = 85.0
    elif days_ago <= 90: recency = 65.0
    elif days_ago <= 180:recency = 40.0
    else:                recency = 10.0

    open_bonus     = 20.0 if sigs.get("open_to_work_flag", False) else 0.0
    completeness   = sigs.get("profile_completeness_score", 0) / 100.0 * 10.0
    # Avg response time — lower = better
    resp_time      = sigs.get("avg_response_time_hours", 999)
    resp_time_sc   = max(0, 10.0 - resp_time / 48.0 * 10.0)  # 10pt for <48hr

    recency_score  = min(recency + open_bonus + completeness + resp_time_sc, 100.0)

    # ── b) Engagement activity ────────────────────────────────────
    apps_30d    = min(sigs.get("applications_submitted_30d", 0) / 5.0,  1.0) * 30.0
    views_30d   = min(sigs.get("profile_views_received_30d", 0) / 50.0, 1.0) * 20.0
    saved_30d   = min(sigs.get("saved_by_recruiters_30d", 0) / 10.0,   1.0) * 20.0
    search_30d  = min(sigs.get("search_appearance_30d", 0) / 200.0,    1.0) * 10.0
    github      = sigs.get("github_activity_score", -1)
    gh_score    = (github / 100.0 * 20.0) if github >= 0 else 0.0
    connections = min(sigs.get("connection_count", 0) / 500.0, 1.0) * 10.0

    engagement_score = min(apps_30d + views_30d + saved_30d +
                           search_30d + gh_score + connections, 100.0)

    # ── c) Reliability ────────────────────────────────────────────
    resp_rate   = sigs.get("recruiter_response_rate", 0.0) * 40.0
    intv_rate   = sigs.get("interview_completion_rate", 0.0) * 30.0
    offer_rate  = sigs.get("offer_acceptance_rate", -1)
    offer_sc    = (offer_rate * 20.0) if offer_rate >= 0 else 10.0
    # Notice period — shorter = easier to hire
    notice      = sigs.get("notice_period_days", 90)
    notice_sc   = max(0.0, 10.0 - notice / 180.0 * 10.0)
    # Verified contacts
    verify_sc   = (sigs.get("verified_email", False) * 3.0 +
                   sigs.get("verified_phone", False) * 3.0 +
                   sigs.get("linkedin_connected", False) * 4.0)

    reliability_score = min(resp_rate + intv_rate + offer_sc +
                            notice_sc + verify_sc, 100.0)

    # ── d) Intent & job-fit ───────────────────────────────────────
    # Work mode match
    jd_mode   = parsed_jd.get("work_mode", "flexible").lower()
    cand_mode = sigs.get("preferred_work_mode", "flexible").lower()
    if jd_mode == cand_mode or jd_mode == "flexible" or cand_mode == "flexible":
        mode_sc = 40.0
    elif (jd_mode == "hybrid" and cand_mode in ("onsite","remote")):
        mode_sc = 20.0
    else:
        mode_sc = 10.0

    relocate_sc    = 20.0 if sigs.get("willing_to_relocate", False) else 0.0
    endorsements   = min(sigs.get("endorsements_received", 0) / 100.0, 1.0) * 25.0

    # Salary fit (if JD implies startup budget ~20-60 LPA, penalise very high expectations)
    sal = sigs.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0) if isinstance(sal, dict) else 0
    sal_sc  = 15.0 if sal_min <= 60 else 5.0  # penalise if expecting >60 LPA for Series A

    intent_score = min(mode_sc + relocate_sc + endorsements + sal_sc, 100.0)

    final = (0.30 * recency_score +
             0.25 * engagement_score +
             0.25 * reliability_score +
             0.20 * intent_score)

    return round(min(final, 100.0), 4)


# ── Cold-start multiplier ────────────────────────────────────────
def cold_start_penalty(candidate: Dict[str, Any]) -> float:
    sigs = candidate.get("redrob_signals", {})
    data_points = sum([
        sigs.get("applications_submitted_30d", 0) > 0,
        sigs.get("profile_views_received_30d", 0) > 0,
        sigs.get("recruiter_response_rate", 0) > 0,
        sigs.get("saved_by_recruiters_30d", 0) > 0,
    ])
    if data_points >= 2:  return 1.00
    elif data_points == 1:return 0.90
    else:                 return 0.80


# ── Final combined score (0.0–1.0) ──────────────────────────────
def final_score(sem: float, prof: float, beh: float,
                seniority: str = "mid",
                cold_start_mult: float = 1.0) -> float:
    w = SENIORITY_WEIGHTS.get(seniority, DEFAULT_WEIGHTS)
    raw = (w["semantic"] * sem + w["profile"] * prof + w["behavioral"] * beh) / 100.0
    return round(min(raw * cold_start_mult, 1.0), 4)


# ── Reasoning generator (matches submission format) ──────────────
def generate_reasoning(candidate: Dict[str, Any],
                       sem: float, prof: float, beh: float,
                       candidate_skills: Dict[str, float],
                       parsed_jd: Dict[str, Any]) -> str:
    p     = candidate.get("profile", {})
    sigs  = candidate.get("redrob_signals", {})
    title = p.get("current_title", "Unknown")
    yoe   = p.get("years_of_experience", 0)
    loc   = p.get("location", "")
    country = p.get("country", "")

    required     = [s.lower() for s in parsed_jd.get("required_skills", [])]
    n_matched    = sum(1 for s in required if s in candidate_skills)
    resp_rate    = sigs.get("recruiter_response_rate", 0)
    days_ago     = _days_since(sigs.get("last_active_date", "2020-01-01"))
    active_str   = (f"{days_ago}d ago" if days_ago < 365 else f"{days_ago//365}yr+ ago")
    notice       = sigs.get("notice_period_days", 90)
    github       = sigs.get("github_activity_score", -1)
    open_to_work = sigs.get("open_to_work_flag", False)

    reason = (f"{title} with {yoe:.1f} yrs; "
              f"{n_matched} core skills matched; "
              f"response rate {resp_rate:.2f}; "
              f"last active {active_str}; "
              f"notice {notice}d")

    extras = []
    if open_to_work:           extras.append("open to work")
    if github >= 50:           extras.append(f"github {github:.0f}")
    if sigs.get("willing_to_relocate"): extras.append("willing to relocate")
    if loc:                    extras.append(f"{loc}, {country}")
    if extras:
        reason += "; " + "; ".join(extras)

    return reason


# ── Helper ───────────────────────────────────────────────────────
def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999
