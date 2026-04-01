"""
VAVE AI Hybrid Retrieval Engine — Features 1, 4, 5, 9, 10
Implements:
  Feature 1  — Hybrid Retrieval: BM25 (sparse) + FAISS (dense) merged with RRF
  Feature 4  — Metadata Filtering: pre-filter by carline / dept / status in PostgreSQL
  Feature 5  — Multi-Query Retrieval: generate N query reformulations → RRF-merge results
  Feature 9  — (via Feature 5, same code path)
  Feature 10 — Contextual Compression: trim retrieved context via Gemini Flash before main prompt
"""
import logging
import json
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any, Callable

logger = logging.getLogger(__name__)

# ─── Try optional dependency ──────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("[HybridRetrieval] rank_bm25 not installed. BM25 disabled. Run: pip install rank-bm25")


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1 — BM25 Index Builder
# ═══════════════════════════════════════════════════════════════════════

class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi to provide keyword/sparse retrieval
    over the VAVE idea text corpus.
    """
    def __init__(self):
        self._bm25: Optional[Any] = None
        self._corpus_size = 0

    def build(self, idea_texts: List[str]):
        """
        Build BM25 index from the list of idea text strings.
        Call after build_vector_db() in app.py.
        """
        if not _BM25_AVAILABLE:
            logger.warning("[BM25Index] BM25 unavailable — index not built.")
            return
        tokenized = [self._tokenize(t) for t in idea_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_size = len(idea_texts)
        logger.info(f"[BM25Index] Built BM25 index over {self._corpus_size} ideas.")

    def search(self, query: str, top_k: int = 20) -> List[int]:
        """Return list of corpus indices ranked by BM25 score."""
        if not self._bm25:
            return list(range(min(top_k, self._corpus_size)))
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return ranked.tolist()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercase, remove punctuation, split on whitespace
        cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        return [t for t in cleaned.split() if len(t) > 2]


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1 — Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Each index in a ranking at position r contributes 1/(k+r+1) to its score.
    Returns indices sorted by descending fused score.
    """
    scores: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1 — Hybrid Search Core
# ═══════════════════════════════════════════════════════════════════════

def hybrid_search(
    query: str,
    faiss_index,
    embedding_model,
    idea_texts: List[str],
    idea_rows: List[Dict],
    bm25_index: BM25Index,
    top_k: int = 10,
    candidate_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Perform hybrid retrieval:
    1. Dense search via FAISS (semantic)
    2. Sparse search via BM25 (keyword)
    3. Merge with RRF

    If candidate_ids is provided (from metadata filter), only those IDs participate.
    Returns top_k idea dicts.
    """
    n_total = len(idea_texts)
    if n_total == 0:
        return []

    search_k = min(top_k * 3, n_total)

    # ── Dense search (FAISS) ──────────────────────────────────────────
    dense_ranking: List[int] = []
    try:
        query_vec = embedding_model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        if candidate_ids is not None:
            # Restrict to candidate subset
            subset_vecs = np.array([
                embedding_model.encode([idea_texts[i]])[0] for i in candidate_ids
            ], dtype=np.float32)
            sims = subset_vecs @ query_vec
            local_ranking = np.argsort(sims)[::-1][:search_k]
            dense_ranking = [candidate_ids[i] for i in local_ranking]
        else:
            query_vec_2d = np.expand_dims(query_vec, 0).astype(np.float32)
            _, indices = faiss_index.search(query_vec_2d, search_k)
            dense_ranking = [int(i) for i in indices[0] if i != -1]
    except Exception as e:
        logger.error(f"[HybridSearch] Dense search error: {e}")

    # ── Sparse search (BM25) ─────────────────────────────────────────
    sparse_ranking: List[int] = []
    try:
        all_sparse = bm25_index.search(query, top_k=search_k)
        if candidate_ids is not None:
            candidate_set = set(candidate_ids)
            sparse_ranking = [i for i in all_sparse if i in candidate_set]
        else:
            sparse_ranking = all_sparse
    except Exception as e:
        logger.error(f"[HybridSearch] BM25 search error: {e}")

    # ── RRF merge ────────────────────────────────────────────────────
    rankings = [r for r in [dense_ranking, sparse_ranking] if r]
    if not rankings:
        return []

    fused = reciprocal_rank_fusion(rankings)[:top_k]

    results = []
    for idx in fused:
        if 0 <= idx < len(idea_rows):
            results.append(idea_rows[idx])
    return results


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 4 — Metadata Filtering
# ═══════════════════════════════════════════════════════════════════════

def extract_metadata_filters(query: str) -> Dict[str, str]:
    """
    Heuristically extract structured filters from a natural-language query.
    Returns dict with optional keys: carline, dept, status.
    """
    filters = {}
    q = query.lower()

    # CarLine detection
    carline_map = {
        "hector plus": "Hector Plus",
        "hector": "Hector",
        "astor": "Astor",
        "zs ev": "ZS EV",
        "comet": "Comet",
        "seltos": "Seltos",
        "nexon": "Nexon",
    }
    for alias, value in carline_map.items():
        if alias in q:
            filters["carline"] = value
            break

    # Department detection
    dept_map = {
        "body": "Body",
        "biw": "Body",
        "hvac": "HVAC",
        "thermal": "HVAC",
        "electrical": "Electrical",
        "electronic": "Electrical",
        "exterior": "Exterior",
        "interior": "Interior",
        "brake": "Brakes",
        "suspension": "Suspension",
        "engine": "Engine",
        "powertrain": "Powertrain",
    }
    for alias, value in dept_map.items():
        if alias in q:
            filters["dept"] = value
            break

    # Status detection
    if "approved" in q or " ok " in q or "status ok" in q:
        filters["status"] = "OK"
    elif "pending" in q or "tbd" in q:
        filters["status"] = "TBD"
    elif "rejected" in q or " ng " in q:
        filters["status"] = "NG"

    return filters


def filter_candidate_ids(
    pg_conn_func: Callable,
    carline: Optional[str] = None,
    dept: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[List[int]]:
    """
    Query PostgreSQL for idea IDs matching the metadata filters.
    Returns list of row indices (0-based positions in idea_rows),
    or None if no filters applied (search everything).
    """
    if not carline and not dept and not status:
        return None  # No filter → search full corpus

    try:
        conn = pg_conn_func()
        conditions = []
        params = []
        if carline:
            conditions.append("mgi_carline ILIKE %s")
            params.append(f"%{carline}%")
        if dept:
            conditions.append("dept ILIKE %s")
            params.append(f"%{dept}%")
        if status:
            conditions.append("status = %s")
            params.append(status)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT ROW_NUMBER() OVER (ORDER BY id) - 1 AS row_idx FROM ideas WHERE {where_clause}"
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        ids = [int(r[0]) for r in rows if r[0] is not None]
        logger.info(f"[MetadataFilter] Filters {{'carline': {carline}, 'dept': {dept}, 'status': {status}}} -> {len(ids)} candidates")
        return ids if ids else None

    except Exception as e:
        logger.error(f"[MetadataFilter] DB filter error: {e}")
        return None  # Fallback: search everything


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 5 — Multi-Query Retrieval (RAG Fusion)
# ═══════════════════════════════════════════════════════════════════════

def generate_query_reformulations(
    query: str,
    gemini_model,
    n: int = 4,
) -> List[str]:
    """
    Use Gemini Flash to generate N semantic reformulations of a query.
    Falls back to the original query on any error.
    """
    import prompt_registry
    try:
        prompt = prompt_registry.get("multi_query_reformulation", query=query, n=n)
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON array
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            reformulations = json.loads(match.group(0))
            if isinstance(reformulations, list):
                return [str(r) for r in reformulations[:n]]
    except Exception as e:
        logger.error(f"[MultiQuery] Reformulation generation failed: {e}")

    return [query]  # Fallback: single original query


def multi_query_hybrid_search(
    query: str,
    gemini_model,
    faiss_index,
    embedding_model,
    idea_texts: List[str],
    idea_rows: List[Dict],
    bm25_index: BM25Index,
    top_k: int = 10,
    candidate_ids: Optional[List[int]] = None,
    n_reformulations: int = 3,
) -> List[Dict]:
    """
    Feature 5 + 1 combined:
    1. Generate n_reformulations of the query
    2. Run hybrid_search for each reformulation
    3. RRF-merge all result rankings
    4. Return top_k deduplicated results
    """
    all_queries = [query] + generate_query_reformulations(query, gemini_model, n=n_reformulations)
    logger.info(f"[MultiQuery] Running {len(all_queries)} query variants: {all_queries[:2]}...")

    all_rankings: List[List[int]] = []

    for q in all_queries:
        try:
            results = hybrid_search(
                query=q,
                faiss_index=faiss_index,
                embedding_model=embedding_model,
                idea_texts=idea_texts,
                idea_rows=idea_rows,
                bm25_index=bm25_index,
                top_k=top_k * 2,
                candidate_ids=candidate_ids,
            )
            # Convert back to global indices
            result_indices = []
            for r in results:
                try:
                    idx = idea_rows.index(r)
                    result_indices.append(idx)
                except ValueError:
                    pass
            if result_indices:
                all_rankings.append(result_indices)
        except Exception as e:
            logger.error(f"[MultiQuery] Search failed for '{q}': {e}")

    if not all_rankings:
        return []

    fused_indices = reciprocal_rank_fusion(all_rankings)[:top_k]
    seen_ids = set()
    final_results = []
    for idx in fused_indices:
        if 0 <= idx < len(idea_rows):
            idea = idea_rows[idx]
            idea_id = idea.get("idea_id", idx)
            if idea_id not in seen_ids:
                seen_ids.add(idea_id)
                final_results.append(idea)

    return final_results


# ═══════════════════════════════════════════════════════════════════════
# FEATURE 10 — Contextual Compression
# ═══════════════════════════════════════════════════════════════════════

def compress_retrieved_context(
    retrieved_ideas: List[Dict],
    user_query: str,
    gemini_flash_model,
    max_ideas: int = 8,
) -> str:
    """
    Feature 10: Use Gemini Flash to extract only the sentences from
    retrieved ideas that are relevant to the specific user query.
    Reduces token usage by 30-50% before passing to the main Gemini Pro call.
    """
    import prompt_registry

    if not retrieved_ideas:
        return ""

    # Format ideas as compact text
    chunks = []
    for idea in retrieved_ideas[:max_ideas]:
        chunk = (
            f"[{idea.get('idea_id', 'N/A')}] "
            f"{idea.get('cost_reduction_idea', '')} | "
            f"Saving: ₹{idea.get('saving_value_inr', 'N/A')} | "
            f"Dept: {idea.get('dept', 'N/A')} | "
            f"Status: {idea.get('status', 'N/A')} | "
            f"Way forward: {idea.get('way_forward', 'N/A')[:150]}"
        )
        chunks.append(chunk)

    raw_context = "\n".join(chunks)

    try:
        compress_prompt = prompt_registry.get(
            "contextual_compression",
            query=user_query,
            context=raw_context,
        )
        response = gemini_flash_model.generate_content(compress_prompt)
        compressed = response.text.strip()
        logger.info(
            f"[Compression] Reduced context {len(raw_context)} → {len(compressed)} chars "
            f"({100*(1-len(compressed)/max(len(raw_context),1)):.0f}% reduction)"
        )
        return compressed
    except Exception as e:
        logger.error(f"[Compression] Compression failed, using raw: {e}")
        return raw_context


# Module-level singleton for BM25
bm25_index = BM25Index()
