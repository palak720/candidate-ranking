# Intelligent Candidate Discovery

Offline candidate-ranking pipeline for matching candidates to a job description.

## What It Does

- Parses a job description into structured intent, seniority, skills, work mode, and experience range.
- Builds candidate text profiles from structured JSON or JSONL input.
- Ranks candidates using semantic search, BM25 hybrid retrieval, multi-signal scoring, and precision re-ranking.
- Writes submission CSV, detailed JSON, and bias-audit JSON outputs.

## Architecture Alignment

This implementation follows the supplied `architecture.html` flow:

| Architecture stage | Implemented in this repo |
| --- | --- |
| Data ingestion | `rank.py` loads candidate JSON/JSONL and raw JD text. |
| Preprocessing and understanding | `src/preprocessing.py` normalizes candidate profiles; `src/jd_parser.py` extracts structured JD intent. |
| Embedding and semantic index | `src/embeddings.py` builds cached TF-IDF + LSA vectors for offline semantic retrieval. |
| Hybrid keyword pass | `rank_bm25` combines exact keyword retrieval with semantic search using RRF fusion. |
| Multi-signal scoring | `src/scoring.py` combines semantic, profile, behavioral, dynamic seniority weights, and cold-start handling. |
| Cross-encoder re-ranking layer | `src/reranking.py` provides an offline precision re-ranker over the top shortlist. |
| Ranking and trust layer | `rank.py` sorts, explains, enforces ranks, and writes `output/bias_audit.json`. |
| Demo and presentation | `demo/app.py` provides the Streamlit judge-facing cockpit. |

The original architecture mentions optional LLM APIs, sentence-transformers, FAISS/Chroma, and cross-encoder models. This repo uses offline equivalents so it runs without network, API keys, or GPU during judging.

## Setup

```bash
pip install -r requirements.txt
```

## Run Ranking

```bash
python rank.py --candidates data/sample_candidates.json --jd data/job_description.txt
```

For a full dataset:

```bash
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt
```

The default output is:

- `output/submission.csv`
- `output/submission_detailed.json`
- `output/bias_audit.json`

## Validate Output

The final submission format expects 100 rows:

```bash
python validate_submission.py output/submission.csv
```

The included sample candidate file has 50 candidates, so validate that run with:

```bash
python validate_submission.py output/submission.csv --expected-rows 50
```

## Demo App

```bash
streamlit run demo/app.py
```
## Dataset Setup

The large dataset file `data/candidates.jsonl` (464MB) is NOT included
in this repository due to GitHub file size limits.

To run the project:
1. Download `candidates.jsonl` from the hackathon portal
2. Place it in the `data/` folder
3. Run the pipeline:
   python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt