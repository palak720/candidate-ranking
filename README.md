# 🎯 Intelligent Candidate Discovery

> **Redrob Hackathon 2026 — Track 01: Data & AI Challenge**

An AI-powered candidate ranking engine that intelligently discovers and ranks the most relevant candidates using **Semantic Intelligence**, **Profile Matching**, and **Behavioral Analytics**.

Unlike traditional keyword-based search, our system understands the **meaning** behind candidate profiles and job descriptions to generate an explainable **Top-100 ranked candidate shortlist** in under **2 seconds**, completely offline and CPU-only.

---

# 👥 Team Details

| Field | Details |
|-------|---------|
| **Team Name** | CodeCrafter |
| **Team Leader** | Varsha Katiyar |
| **Team Member** | Palak Kasaudhan |
| **Track** | Track 01 – Data & AI Challenge |
| **Problem Statement** | Intelligent Candidate Discovery |

---

# 📌 Project Overview

Recruiters often struggle to identify the best candidates because traditional Applicant Tracking Systems (ATS) rely heavily on exact keyword matching.

For example:

- Resume contains **"Machine Learning"**
- Job Description contains **"Predictive Analytics"**

Although both represent similar skills, keyword-based systems may fail to identify them as relevant.

Our solution addresses this problem using a hybrid AI ranking pipeline that combines:

- Semantic Similarity
- Explicit Profile Matching
- Behavioral Intelligence

The result is an accurate, explainable, and fair ranking of candidates.

---

# ✨ Key Features

- 🔍 Semantic Resume Understanding
- 📄 Job Description Parsing
- 🎯 Multi-Signal Candidate Ranking
- ⚖ Dynamic Weighting based on Seniority
- 📊 Explainable Candidate Scores
- 🚀 Offline CPU-only Execution
- 🌐 Streamlit Dashboard
- 📈 Bias Audit
- ✅ Top-100 Candidate Shortlisting
- ⚡ Runtime under 2 seconds

---

# 🏗️ System Architecture

```text
                  Job Description
                         │
                         ▼
                JD Parsing Module
                         │
                         ▼
           Candidate Preprocessing
                         │
         ┌───────────────┴──────────────┐
         ▼                              ▼
 Semantic Matching              Profile Matching
(TF-IDF + LSA + BM25)       Skills + Experience
         │
         ▼
 Reciprocal Rank Fusion
         │
         ▼
 Behavioral Intelligence
      (23 Redrob Signals)
         │
         ▼
 Dynamic Weight Engine
         │
         ▼
 Cold Start Adjustment
         │
         ▼
      Final Candidate Score
         │
         ▼
     Top-100 Ranked Output
```

---

# 📂 Project Structure

```text
candidate-ranking/
│
├── rank.py
├── requirements.txt
├── README.md
├── validate_submission.py
│
├── src/
│   ├── preprocessing.py
│   ├── jd_parser.py
│   ├── embeddings.py
│   ├── scoring.py
│   └── bias_audit.py
│
├── demo/
│   └── app.py
│
├── data/
│   ├── sample_candidates.json
│   ├── job_description.txt
│   └── candidates.jsonl
│
└── output/
    ├── submission.csv
    ├── submission_detailed.json
    └── bias_audit.json
```

---

# 🧠 Ranking Pipeline

Our ranking system consists of six major stages.

---

## 1️⃣ Candidate Preprocessing

Each candidate profile is cleaned and converted into a structured representation containing:

- Skills
- Experience
- Education
- Certifications
- Job Titles
- Projects
- Summary

---

## 2️⃣ Job Description Understanding

The JD parser extracts important information such as:

- Required Skills
- Preferred Skills
- Experience
- Education
- Keywords
- Seniority Level

This information becomes the reference profile for ranking.

---

## 3️⃣ Semantic Matching

Instead of relying on exact keywords, we calculate semantic similarity using:

- TF-IDF Vectorization
- Latent Semantic Analysis (LSA)
- Cosine Similarity

This enables matching related concepts rather than identical words.

---

## 4️⃣ Profile Matching

The profile matching engine evaluates:

- Technical Skills
- Work Experience
- Education
- Certifications
- Projects
- Job Titles

Each attribute contributes to the overall profile score.

---

## 5️⃣ Behavioral Intelligence

Behavioral ranking uses all **23 Redrob platform signals**, including:

- Recent Activity
- Recruiter Engagement
- Resume Freshness
- Hiring Intent
- Profile Completeness
- Response Rate
- Platform Reliability

Behavioral signals help distinguish candidates with similar qualifications.

---

## 6️⃣ Final Score Calculation

Three independent scores are combined.

| Component | Weight |
|------------|---------|
| Semantic | 40% |
| Profile | 35% |
| Behavioral | 25% |

Formula:

```text
Final Score =
(Semantic × Weight)
+
(Profile × Weight)
+
(Behavior × Weight)

×

Cold Start Multiplier
```

---

# 🎯 Dynamic Weighting

Weights automatically adjust according to job seniority.

| Seniority | Semantic | Profile | Behavioral |
|------------|----------|----------|------------|
| Intern | 30 | 25 | 45 |
| Junior | 35 | 30 | 35 |
| Mid | 40 | 35 | 25 |
| Senior | 38 | 45 | 17 |
| Founding | 35 | 45 | 20 |

---

# ⚖ Explainability

Every ranked candidate includes:

- Semantic Score
- Profile Score
- Behavioral Score
- Final Score
- Rank

Recruiters can understand **why** a candidate received a particular ranking.

---

# 🛡️ Bias Audit

To promote fairness, the system performs bias analysis by evaluating:

- Score Distribution
- Ranking Consistency
- Feature Importance
- Behavioral Bias
- Protected Attribute Independence

Generated Report:

```text
output/bias_audit.json
```

---

# 🚀 Performance

| Metric | Value |
|---------|-------|
| Runtime | ~1.1 sec |
| Output | Top 100 Candidates |
| API Calls | 0 |
| GPU Required | No |
| CPU Only | Yes |
| Internet Required | No |

---

# 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.11 |
| NLP | Scikit-learn |
| Semantic Search | TF-IDF |
| Dimensionality Reduction | TruncatedSVD |
| Keyword Search | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Dashboard | Streamlit |
| Visualization | Plotly |

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset

Download **candidates.jsonl** from the hackathon portal and place it inside the **data/** directory.

---

# ▶️ Run Ranking Pipeline

```bash
python rank.py \
--candidates data/candidates.jsonl \
--jd data/job_description.txt
```

---

## Sample Dataset

```bash
python rank.py \
--candidates data/sample_candidates.json \
--jd data/job_description.txt
```

---

# 🌐 Run Demo

```bash
streamlit run demo/app.py
```

Visit:

```
http://localhost:8501
```

---

# ✅ Validate Submission

```bash
python validate_submission.py output/submission.csv
```

---

# 🔍 Run Bias Audit

```bash
python src/bias_audit.py
```

---

# 📤 Output Files

| File | Description |
|------|-------------|
| submission.csv | Final ranked candidates |
| submission_detailed.json | Detailed score breakdown |
| bias_audit.json | Fairness report |

---

# 📈 Future Improvements

- Sentence Transformers
- Hybrid Dense Retrieval
- Learning-to-Rank Models
- Recruiter Feedback Loop
- Multilingual Resume Parsing
- Skill Graph Integration
- Personalized Candidate Ranking

---

# 🏆 Submission Compliance

- ✅ Exactly 100 ranked candidates
- ✅ Rank values from 1–100
- ✅ CPU-only execution
- ✅ Offline processing
- ✅ Zero API calls
- ✅ Explainable AI
- ✅ Bias Audited
- ✅ Streamlit Demo Included

---

# 🙏 Acknowledgements

Developed for the **Redrob Hackathon 2026 – Track 01: Data & AI Challenge** by **Team CodeCrafter**.

**Team Members**

- **Varsha Katiyar** – Team Leader
- **Palak Kasaudhan** – Team Member

---