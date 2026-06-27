"""
embeddings.py — TF-IDF + SVD based semantic embeddings (fully offline, CPU-only)
==================================================================================
Uses sklearn TfidfVectorizer + TruncatedSVD (Latent Semantic Analysis).
No external downloads needed. Satisfies competition constraint: CPU-only, no network.
"""

import os, pickle
import numpy as np
from typing import List, Tuple, Dict, Any

_vectorizer = None
_svd        = None
CACHE_PATH  = "output/embeddings_cache.pkl"
INDEX_PATH  = "output/tfidf_index.pkl"
N_COMPONENTS = 256   # LSA dimensionality


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def fit_and_embed(texts: List[str]) -> Tuple[Any, Any, np.ndarray]:
    """Fit TF-IDF + SVD on corpus, return (vectorizer, svd, embeddings)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline

    print(f"  [embed] Fitting TF-IDF + SVD ({N_COMPONENTS}d) on {len(texts)} docs …")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=1,
    )
    tfidf_mat = vec.fit_transform(texts)

    n_comp = min(N_COMPONENTS, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(tfidf_mat).astype(np.float32)
    embeddings = _normalise(embeddings)

    print(f"  [embed] Done. Shape: {embeddings.shape}")
    return vec, svd, embeddings

def transform_text(texts: List[str], vec, svd) -> np.ndarray:
    tfidf_mat  = vec.transform(texts)
    embeddings = svd.transform(tfidf_mat).astype(np.float32)
    return _normalise(embeddings)

def embed_texts(texts: List[str], show_progress: bool = True, **kwargs) -> np.ndarray:
    """Called externally — only works after fit (use fit_and_embed first)."""
    global _vectorizer, _svd
    if _vectorizer is None or _svd is None:
        raise RuntimeError("Call fit_and_embed first to initialise the model.")
    return transform_text(texts, _vectorizer, _svd)

def build_faiss_index(embeddings: np.ndarray):
    """Returns a simple numpy matrix (used for dot-product cosine search)."""
    return embeddings  # we use numpy dot product directly

def save_cache(candidate_ids, candidate_texts, embeddings, index) -> None:
    os.makedirs("output", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vec": _vectorizer, "svd": _svd, "embeddings": embeddings,
                     "ids": candidate_ids, "texts": candidate_texts}, f)
    print(f"  [cache] Saved → {INDEX_PATH}")

def load_cache() -> Tuple[List[str], List[str], np.ndarray]:
    global _vectorizer, _svd
    with open(INDEX_PATH, "rb") as f:
        d = pickle.load(f)
    _vectorizer = d["vec"]
    _svd        = d["svd"]
    print(f"  [cache] Loaded {len(d['ids'])} vectors from disk")
    return d["ids"], d["texts"], d["embeddings"]

def cache_exists() -> bool:
    return os.path.exists(INDEX_PATH)

def semantic_search(jd_emb: np.ndarray,
                    candidate_matrix: np.ndarray,
                    top_k: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine similarity via dot product (vectors are L2-normalised)."""
    scores = candidate_matrix.dot(jd_emb)
    if top_k >= len(scores):
        ids = np.argsort(-scores)
    else:
        ids = np.argpartition(-scores, top_k)[:top_k]
        ids = ids[np.argsort(-scores[ids])]
    return scores[ids], ids

def build_bm25(candidate_texts: List[str]):
    from rank_bm25 import BM25Okapi
    tokenised = [t.lower().split() for t in candidate_texts]
    bm25 = BM25Okapi(tokenised)
    print(f"  [bm25] Built BM25 index over {len(tokenised)} docs")
    return bm25

def bm25_search(bm25, query_text: str, top_k: int = 500) -> np.ndarray:
    return np.array(bm25.get_scores(query_text.lower().split()), dtype=np.float32)

def rrf_fusion(sem_scores, sem_ids, bm25_scores, n_candidates,
               k=60, sem_weight=0.6, bm25_weight=0.4):
    sem_rank  = {int(idx): rank+1 for rank, idx in enumerate(sem_ids)}
    bm25_order = np.argsort(-bm25_scores)
    bm25_rank  = {int(idx): rank+1 for rank, idx in enumerate(bm25_order)}
    fused = {}
    for idx in range(n_candidates):
        sr = sem_rank.get(idx, n_candidates+1)
        br = bm25_rank.get(idx, n_candidates+1)
        fused[idx] = sem_weight/(k+sr) + bm25_weight/(k+br)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
