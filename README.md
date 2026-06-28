# 🎯 Intelligent Candidate Discovery
### Redrob Hackathon 2026 — Track 01: Data & AI Challenge

---

## 👥 Team Details

| Field | Details |
|-------|---------|
| **Team Name** | CodeCrafter |
| **Team Leader** | Varsha Katiyar |
| **Problem Statement** | Intelligent Candidate Discovery |
| **Track** | Track 01 — Data & AI Challenge |

---

## 📌 Project Overview

An AI-powered candidate ranking system that goes **beyond keyword search** to intelligently rank candidates using three complementary signals — semantic understanding, explicit profile matching, and behavioral intelligence from all 23 Redrob platform signals.

The system delivers a **precise, explainable top-100 ranked shortlist** from massive talent pools in under 2 seconds — fully offline, CPU-only, zero API calls.

---

## Architecture Diagram
[View Architecture Diagram](docs/ARCHITECTURE_DIAGRAM%20(1).html)

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add dataset
Download `candidates.jsonl` from hackathon portal → place in `data/` folder

### Step 3 — Run pipeline
```bash
# Full dataset
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt

# Quick test on sample
python rank.py --candidates data/sample_candidates.json --jd data/job_description.txt
```

### Step 4 — Run live demo
```bash
streamlit run demo/app.py
```
Opens at `http://localhost:8501`

### Step 5 — Validate submission
```bash
python validate_submission.py output/submission.csv
```

### Step 6 — Run bias audit
```bash
python src/bias_audit.py
```

---

## 📁 Project Structure

```
candidate-ranking/
├── rank.py                      ← Main entry point
├── requirements.txt             ← All dependencies
├── validate_submission.py       ← Official validator
├── README.md                    ← This file
├── src/
│   ├── jd_parser.py             ← JD understanding (rule-based NLP)
│   ├── preprocessing.py         ← Data cleaning + text blob builder
│   ├── embeddings.py            ← TF-IDF + LSA + BM25 + RRF
│   ├── scoring.py               ← Multi-signal scoring engine
│   └── bias_audit.py            ← Fairness analysis
├── demo/
│   └── app.py                   ← Streamlit live demo
├── data/
│   ├── sample_candidates.json   ← 50-candidate sample (included)
│   ├── job_description.txt      ← JD text (included)
│   └── candidates.jsonl         ← Full dataset (download separately)
└── output/
    ├── submission.csv            ← Final ranked output
    ├── submission_detailed.json  ← Sub-scores for analysis
    └── bias_audit.json          ← Fairness report
```

---

## 🧠 Scoring System

### Three Signals

| Signal | Default Weight | What It Measures |
|--------|---------------|-----------------|
| **Semantic** | 40% | TF-IDF + LSA cosine similarity — meaning, not keywords |
| **Profile** | 35% | Skills, experience, education, title relevance |
| **Behavioral** | 25% | All 23 Redrob signals — recency, engagement, reliability, intent |

### Dynamic Weights by Seniority

| Seniority | Semantic | Profile | Behavioral |
|-----------|----------|---------|------------|
| Intern    | 30%      | 25%     | 45%        |
| Junior    | 35%      | 30%     | 35%        |
| Mid       | 40%      | 35%     | 25%        |
| Senior    | 38%      | 45%     | 17%        |
| Founding  | 35%      | 45%     | 20%        |

### Final Score Formula
```
Final Score = (w_semantic × Semantic + w_profile × Profile + w_behavioral × Behavioral) / 100
              × Cold-Start Multiplier (0.80 – 1.00)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Embeddings | scikit-learn TfidfVectorizer + TruncatedSVD (LSA) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Data Processing | pandas, numpy |
| Demo UI | Streamlit |
| Visualization | plotly |

> **Fully offline** — no GPU, no internet, no API calls during ranking.

---

## 📊 Dataset

`data/candidates.jsonl` (464MB) is **not included** due to GitHub file size limits.

Download from the hackathon portal and place in `data/` folder before running.

Sample data (`data/sample_candidates.json` — 50 candidates) is included for testing.

---

## ✅ Submission Compliance

- Exactly 100 rows output ✅
- Ranks 1–100 each exactly once ✅
- Score monotonically non-increasing ✅
- CPU only, no API calls ✅
- Runtime on sample: ~1.1 seconds ✅

---

## 📤 Submission Assets

- `output/submission.csv` — 100 ranked candidates
- `output/submission_detailed.json` — sub-scores for analysis
- `output/bias_audit.json` — fairness report
- `demo/app.py` — Streamlit live demo
