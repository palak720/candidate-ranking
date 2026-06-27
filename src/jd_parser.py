"""
jd_parser.py
============
Parses a raw Job Description text into a structured dictionary.
Works WITHOUT any external LLM API call — uses rule-based NLP + keyword
extraction so the ranker can run fully offline (per submission constraints).
"""

import re
from typing import Dict, List, Any

# ── Skill taxonomy ──────────────────────────────────────────────────────────
AI_ML_SKILLS = [
    "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "huggingface", "transformers", "sentence-transformers", "llm", "llms",
    "fine-tuning", "fine tuning", "embeddings", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "chromadb", "opensearch",
    "elasticsearch", "bm25", "hybrid search", "rag", "retrieval augmented",
    "ranking", "retrieval", "nlp", "natural language processing",
    "deep learning", "machine learning", "neural network",
    "bert", "gpt", "openai", "langchain", "llama", "mistral",
    "a/b testing", "ndcg", "mrr", "map", "evaluation", "mlops",
    "spark", "airflow", "kafka", "dbt", "sql", "nosql",
    "docker", "kubernetes", "aws", "gcp", "azure", "git",
    "recommendation system", "information retrieval",
]

SENIORITY_MAP = {
    "founding": 9, "staff": 8, "principal": 8, "lead": 7,
    "senior": 6, "mid": 5, "junior": 3, "entry": 2, "intern": 1,
}

WORK_MODES = ["remote", "hybrid", "onsite", "flexible"]

def extract_skills(text: str) -> List[str]:
    """Return list of matched skills from the taxonomy."""
    text_lower = text.lower()
    found = []
    for skill in AI_ML_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))  # preserve order, dedupe

def extract_experience_years(text: str) -> Dict[str, int]:
    """Pull min/max years of experience from JD text."""
    patterns = [
        r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+\s*(?:years?|yrs?)',
        r'minimum\s+(\d+)\s*(?:years?|yrs?)',
        r'at least\s+(\d+)\s*(?:years?|yrs?)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                return {"min": int(groups[0]), "max": int(groups[1])}
            return {"min": int(groups[0]), "max": int(groups[0]) + 4}
    return {"min": 0, "max": 50}

def extract_seniority(text: str) -> str:
    text_lower = text.lower()
    best = ("mid", 5)
    for word, level in SENIORITY_MAP.items():
        if word in text_lower and level > best[1]:
            best = (word, level)
    return best[0]

def extract_work_mode(text: str) -> str:
    text_lower = text.lower()
    for mode in WORK_MODES:
        if mode in text_lower:
            return mode
    return "flexible"

def extract_location(text: str) -> List[str]:
    # Simple heuristic: lines with city names or "Location:" prefix
    locs = []
    for line in text.split('\n'):
        if 'location' in line.lower():
            locs.append(line.strip())
    return locs[:3]

def parse_jd(jd_text: str) -> Dict[str, Any]:
    """
    Main entry point. Returns a structured dict from raw JD text.
    """
    skills       = extract_skills(jd_text)
    experience   = extract_experience_years(jd_text)
    seniority    = extract_seniority(jd_text)
    work_mode    = extract_work_mode(jd_text)
    locations    = extract_location(jd_text)

    # Split required vs nice-to-have based on section keywords
    required_skills, nicetohaave_skills = [], []
    in_required = False
    for line in jd_text.split('\n'):
        ll = line.lower()
        if any(k in ll for k in ["absolutely need", "you must", "required", "must have", "must-have"]):
            in_required = True
        elif any(k in ll for k in ["nice to have", "nice-to-have", "bonus", "preferred", "good to have"]):
            in_required = False
        matched = [s for s in skills if s in ll]
        if matched:
            if in_required:
                required_skills.extend(matched)
            else:
                nicetohaave_skills.extend(matched)

    return {
        "all_skills":        skills,
        "required_skills":   list(set(required_skills)) or skills[:8],
        "nice_to_have":      list(set(nicetohaave_skills)),
        "experience":        experience,
        "seniority":         seniority,
        "work_mode":         work_mode,
        "locations":         locations,
        "raw_text":          jd_text,
    }


if __name__ == "__main__":
    sample = """
    Senior AI Engineer — 5-9 years experience required.
    Location: Pune/Noida, Hybrid.
    Things you absolutely need:
      Production experience with embeddings, FAISS, vector databases.
      Strong Python, ranking systems, NDCG, A/B testing.
    Nice to have: Spark, Kafka, MLOps.
    """
    import json
    result = parse_jd(sample)
    print(json.dumps(result, indent=2))
