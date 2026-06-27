"""
Streamlit demo for the candidate-ranking system.

Run:
  streamlit run demo/app.py
"""

import json
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import embeddings as EMB
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


DEFAULT_JD = """Senior AI Engineer - Founding Team
Location: Pune/Noida, India (Hybrid)
Experience: 5-9 years

Things you absolutely need:
- Production experience with embeddings-based retrieval systems
- Strong Python and vector databases (FAISS, Pinecone, Weaviate)
- Ranking systems, NDCG, A/B testing, evaluation frameworks
- LLM fine-tuning, NLP, semantic search

Nice to have:
- Spark, Kafka, MLOps experience
- Startup experience
"""


st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="RR",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --bg: #f8fafc;
        --ink: #172033;
        --muted: #64748b;
        --line: #d8e0ea;
        --panel: #ffffff;
        --accent: #0f766e;
        --accent-2: #2563eb;
        --warn: #b45309;
        --good: #047857;
    }

    .main .block-container {
        max-width: 1360px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 58%, #111827 100%);
        border-radius: 8px;
        color: white;
        padding: 28px 30px;
        margin-bottom: 18px;
    }

    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        opacity: 0.84;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        line-height: 1.12;
        margin: 0.35rem 0 0.45rem;
    }

    .hero p {
        max-width: 820px;
        margin: 0;
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.02rem;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 18px;
    }

    .stat-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stat-value {
        color: var(--ink);
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 4px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }

    .chip {
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        border-radius: 999px;
        color: #155e75;
        display: inline-flex;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
    }

    .chip-muted {
        background: #f1f5f9;
        border-color: #cbd5e1;
        color: #475569;
    }

    .candidate-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .candidate-head {
        align-items: flex-start;
        display: flex;
        gap: 12px;
        justify-content: space-between;
    }

    .rank-pill {
        background: #172033;
        border-radius: 999px;
        color: white;
        font-size: 0.82rem;
        font-weight: 800;
        padding: 5px 10px;
        white-space: nowrap;
    }

    .candidate-name {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0;
    }

    .candidate-sub {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 3px 0 0;
    }

    .score-pill {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        color: var(--good);
        font-weight: 800;
        padding: 6px 12px;
        white-space: nowrap;
    }

    .signal-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
    }

    .signal {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
    }

    .signal span {
        color: var(--muted);
        display: block;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .signal strong {
        color: var(--ink);
        font-size: 1.1rem;
    }

    .reason {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-top: 12px;
    }

    .empty-state {
        border: 1px dashed #94a3b8;
        border-radius: 8px;
        color: var(--muted);
        padding: 24px;
        text-align: center;
    }

    @media (max-width: 900px) {
        .stat-grid,
        .signal-row {
            grid-template-columns: 1fr;
        }

        .hero h1 {
            font-size: 1.65rem;
        }

        .candidate-head {
            display: block;
        }

        .score-pill {
            display: inline-flex;
            margin-top: 10px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def format_score(score):
    return f"{score:.4f}"


def format_percent(value):
    return f"{value:.0%}"


def chip_list(items, empty_text="No skills detected", muted=False):
    if not items:
        items = [empty_text]
        muted = True
    class_name = "chip chip-muted" if muted else "chip"
    chips = "".join(f"<span class='{class_name}'>{item}</span>" for item in items)
    return f"<div class='chips'>{chips}</div>"


@st.cache_resource(show_spinner="Indexing candidate database...")
def load_data():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_path = os.path.join(base, "data", "sample_candidates.json")
    full_path = os.path.join(base, "data", "candidates.jsonl")

    if os.path.exists(full_path):
        candidates = list(load_candidates_jsonl(full_path, limit=5000))
        source_name = "data/candidates.jsonl"
    elif os.path.exists(sample_path):
        candidates = load_candidates_json(sample_path)
        source_name = "data/sample_candidates.json"
    else:
        st.error("No candidate data found in the data folder.")
        st.stop()

    texts = [build_candidate_text(candidate) for candidate in candidates]
    skills = [build_skills_set(candidate) for candidate in candidates]

    vec, svd, embeds = EMB.fit_and_embed(texts)
    EMB._vectorizer = vec
    EMB._svd = svd
    bm25 = EMB.build_bm25(texts)

    return candidates, texts, skills, embeds, bm25, source_name


def rank_candidates(
    jd_text,
    candidates,
    cand_texts,
    cand_skills,
    cand_embeds,
    bm25_index,
    top_n,
    use_hybrid,
):
    parsed_jd = parse_jd(jd_text)
    seniority = parsed_jd.get("seniority", "mid")
    jd_combined = jd_text + " " + " ".join(parsed_jd["all_skills"])
    jd_embedding = EMB.transform_text([jd_combined], EMB._vectorizer, EMB._svd)[0]

    retrieve_k = min(len(candidates), max(300, top_n * 6))
    sem_scores_raw, sem_ids = EMB.semantic_search(jd_embedding, cand_embeds, top_k=retrieve_k)
    sem_score_map = {
        int(idx): float(score) for idx, score in zip(sem_ids, sem_scores_raw)
    }

    if use_hybrid and len(candidates) > 1:
        bm25_scores = EMB.bm25_search(bm25_index, jd_combined)
        fused = EMB.rrf_fusion(sem_scores_raw, sem_ids, bm25_scores, len(candidates))
        pool = [idx for idx, _ in fused[:retrieve_k]]
    else:
        pool = list(sem_ids[:retrieve_k])

    results = []
    for orig_idx in pool:
        orig_idx = int(orig_idx)
        if orig_idx < 0 or orig_idx >= len(candidates):
            continue

        candidate = candidates[orig_idx]
        candidate_skills = cand_skills[orig_idx]
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        sem_score = calc_sem(max(0.0, sem_score_map.get(orig_idx, 0.0)))
        prof_score = profile_score(candidate, candidate_skills, parsed_jd)
        beh_score = behavioral_score(candidate, parsed_jd)
        base_score = final_score(
            sem_score,
            prof_score,
            beh_score,
            seniority,
            cold_start_penalty(candidate),
        )
        rerank_score = precision_rerank_score(
            jd_text,
            cand_texts[orig_idx],
            candidate,
            candidate_skills,
            parsed_jd,
        )
        score = blend_rerank_score(base_score, rerank_score)
        reasoning = generate_reasoning(
            candidate,
            sem_score,
            prof_score,
            beh_score,
            candidate_skills,
            parsed_jd,
        )
        matched_skills = [
            skill for skill in parsed_jd.get("required_skills", []) if skill in candidate_skills
        ]
        top_skills = sorted(
            candidate_skills.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:6]

        results.append(
            {
                "candidate_id": candidate.get("candidate_id", ""),
                "score": score,
                "title": profile.get("current_title", "Unknown role"),
                "headline": profile.get("headline", ""),
                "company": profile.get("current_company", ""),
                "industry": profile.get("current_industry", ""),
                "yoe": profile.get("years_of_experience", 0),
                "location": profile.get("location", ""),
                "country": profile.get("country", ""),
                "open_to_work": signals.get("open_to_work_flag", False),
                "last_active": signals.get("last_active_date", ""),
                "github": signals.get("github_activity_score", -1),
                "resp_rate": signals.get("recruiter_response_rate", 0),
                "notice": signals.get("notice_period_days", 90),
                "sem_sc": sem_score,
                "prof_sc": prof_score,
                "beh_sc": beh_score,
                "rerank_sc": rerank_score,
                "reasoning": reasoning,
                "matched_skills": matched_skills,
                "top_skills": [skill for skill, _ in top_skills],
            }
        )

    results.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return parsed_jd, results[:top_n]


def render_stats(cards):
    html = "<div class='stat-grid'>"
    for label, value in cards:
        html += (
            "<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_candidate_card(row):
    open_status = "Open to work" if row["open_to_work"] else "Passive"
    location = ", ".join(filter(None, [row["location"], row["country"]])) or "Location unknown"
    company = row["company"] or "Company unavailable"
    github = "N/A" if row["github"] < 0 else f"{row['github']:.0f}"

    st.markdown(
        f"""
<div class="candidate-card">
    <div class="candidate-head">
        <div>
            <span class="rank-pill">Rank {row['rank']}</span>
            <h3 class="candidate-name">{row['candidate_id']} - {row['title']}</h3>
            <p class="candidate-sub">{row['yoe']:.1f} yrs | {company} | {location}</p>
        </div>
        <span class="score-pill">Score {format_score(row['score'])}</span>
    </div>
    <p class="reason">{row['headline']}</p>
    <div class="signal-row">
        <div class="signal"><span>Semantic</span><strong>{row['sem_sc']:.1f}</strong></div>
        <div class="signal"><span>Profile</span><strong>{row['prof_sc']:.1f}</strong></div>
        <div class="signal"><span>Behavioral</span><strong>{row['beh_sc']:.1f}</strong></div>
    </div>
    <div class="reason">
        Precision re-rank: {row['rerank_sc']:.1f} | {row['reasoning']}<br>
        Status: {open_status} | Response: {format_percent(row['resp_rate'])} | Notice: {row['notice']} days | GitHub: {github}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    skill_col, match_col = st.columns([1, 1])
    with skill_col:
        st.markdown("Top candidate skills")
        st.markdown(chip_list(row["top_skills"], muted=True), unsafe_allow_html=True)
    with match_col:
        st.markdown("Matched required skills")
        st.markdown(chip_list(row["matched_skills"]), unsafe_allow_html=True)


candidates, cand_texts, cand_skills, cand_embeds, bm25_index, source_name = load_data()

with st.sidebar:
    st.markdown("### Ranking Controls")
    top_n = st.slider("Candidates to show", min_value=5, max_value=min(50, len(candidates)), value=min(20, len(candidates)), step=5)
    use_hybrid = st.toggle("Hybrid retrieval", value=True)
    min_score = st.slider("Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    only_open = st.toggle("Open-to-work only", value=False)
    st.divider()
    st.markdown("### Data")
    st.metric("Candidates", f"{len(candidates):,}")
    st.caption(source_name)
    st.caption("TF-IDF + LSA, BM25, profile score, behavioral score")


st.markdown(
    """
<div class="hero">
    <div class="hero-kicker">Redrob Hackathon Demo</div>
    <h1>Candidate intelligence cockpit for high-signal shortlists.</h1>
    <p>Paste a job description, inspect extracted intent, rank the candidate pool, and export a recruiter-ready shortlist with explainable scoring.</p>
</div>
""",
    unsafe_allow_html=True,
)

dataset_label = "sample" if "sample" in source_name else "full"
render_stats(
    [
        ("Dataset", dataset_label.title()),
        ("Profiles", f"{len(candidates):,}"),
        ("Model", "Offline"),
        ("Signals", "3"),
    ]
)

input_col, analysis_col = st.columns([1.18, 0.82], gap="large")

with input_col:
    st.markdown("<div class='panel-title'>Job Description</div>", unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste job description",
        value=DEFAULT_JD,
        height=310,
        label_visibility="collapsed",
    )
    action_col, hint_col = st.columns([0.34, 0.66])
    with action_col:
        run_btn = st.button("Rank candidates", type="primary", use_container_width=True)
    with hint_col:
        st.caption("The demo ranks locally. No API calls or network requests are used.")

with analysis_col:
    parsed_preview = parse_jd(jd_text) if jd_text.strip() else {}
    st.markdown("<div class='panel-title'>JD Intelligence</div>", unsafe_allow_html=True)
    if parsed_preview:
        render_stats(
            [
                ("Seniority", parsed_preview["seniority"].title()),
                ("Mode", parsed_preview["work_mode"].title()),
                ("Experience", f"{parsed_preview['experience']['min']}-{parsed_preview['experience']['max']}"),
                ("Skills", len(parsed_preview["all_skills"])),
            ]
        )
        st.markdown("Required skills")
        st.markdown(
            chip_list(parsed_preview.get("required_skills", [])[:10]),
            unsafe_allow_html=True,
        )
        st.markdown("Nice to have")
        st.markdown(
            chip_list(parsed_preview.get("nice_to_have", [])[:8], "No optional skills detected", True),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='empty-state'>Paste a JD to preview extracted seniority, skills, and work mode.</div>",
            unsafe_allow_html=True,
        )


if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = None
    st.session_state.parsed_jd = None
    st.session_state.elapsed = None

if run_btn:
    if not jd_text.strip():
        st.warning("Paste a job description before ranking.")
    else:
        with st.spinner("Ranking candidates and preparing shortlist..."):
            start = time.time()
            parsed_jd, ranked = rank_candidates(
                jd_text,
                candidates,
                cand_texts,
                cand_skills,
                cand_embeds,
                bm25_index,
                top_n,
                use_hybrid,
            )
            st.session_state.ranking_results = ranked
            st.session_state.parsed_jd = parsed_jd
            st.session_state.elapsed = time.time() - start


results = st.session_state.ranking_results or []
if results:
    visible_results = [
        row for row in results if row["score"] >= min_score and (row["open_to_work"] or not only_open)
    ]

    st.divider()
    st.markdown("<div class='panel-title'>Shortlist Overview</div>", unsafe_allow_html=True)
    render_stats(
        [
            ("Showing", len(visible_results)),
            ("Top Score", format_score(results[0]["score"])),
            ("Open To Work", sum(1 for row in visible_results if row["open_to_work"])),
            ("Runtime", f"{st.session_state.elapsed:.2f}s"),
        ]
    )

    if not visible_results:
        st.markdown(
            "<div class='empty-state'>No candidates match the active filters.</div>",
            unsafe_allow_html=True,
        )
    else:
        table_rows = [
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "score": row["score"],
                "title": row["title"],
                "yoe": row["yoe"],
                "location": ", ".join(filter(None, [row["location"], row["country"]])),
                "open_to_work": row["open_to_work"],
                "semantic": row["sem_sc"],
                "profile": row["prof_sc"],
                "behavioral": row["beh_sc"],
                "rerank": row["rerank_sc"],
            }
            for row in visible_results
        ]
        table_df = pd.DataFrame(table_rows)

        tab_cards, tab_table, tab_export = st.tabs(
            ["Candidate cards", "Score table", "Export"]
        )

        with tab_cards:
            for row in visible_results:
                render_candidate_card(row)

        with tab_table:
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "score",
                        format="%.4f",
                        min_value=0,
                        max_value=1,
                    ),
                    "semantic": st.column_config.NumberColumn("semantic", format="%.1f"),
                    "profile": st.column_config.NumberColumn("profile", format="%.1f"),
                    "behavioral": st.column_config.NumberColumn("behavioral", format="%.1f"),
                    "rerank": st.column_config.NumberColumn("rerank", format="%.1f"),
                },
            )

        with tab_export:
            export_df = pd.DataFrame(
                [
                    {
                        "candidate_id": row["candidate_id"],
                        "rank": index + 1,
                        "score": row["score"],
                        "reasoning": row["reasoning"],
                    }
                    for index, row in enumerate(visible_results)
                ]
            )
            st.download_button(
                "Download submission CSV",
                export_df.to_csv(index=False),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download detailed JSON",
                json.dumps(visible_results, indent=2),
                file_name="shortlist_detailed.json",
                mime="application/json",
                use_container_width=True,
            )
else:
    st.markdown(
        "<div class='empty-state'>Run the ranker to generate a shortlist.</div>",
        unsafe_allow_html=True,
    )
