# retrieval/hybrid.py
from typing import List, Dict, Any
from ingestion.index import collection
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

# =====================================================================
# GLOBAL SINGLETON CACHE FOR SPARSE LEXICAL SEARCH (ZERO OVERHEAD RUNTIME)
# =====================================================================
_CACHED_BM25_GLOBAL: Any = None
_CACHED_GLOBAL_RECORDS: Dict[str, Any] = {}

_CACHED_BM25_BY_FY: Dict[str, Any] = {}      
_CACHED_RECORDS_BY_FY: Dict[str, Any] = {} 

def _get_or_init_global_bm25() -> tuple:
    """
    Lazily instantiates and caches the complete tokenized corpus index once.
    Eliminates database reads and tokenization loops on active requests.
    """
    global _CACHED_BM25_GLOBAL, _CACHED_GLOBAL_RECORDS
    if _CACHED_BM25_GLOBAL is not None:
        return _CACHED_BM25_GLOBAL, _CACHED_GLOBAL_RECORDS

    # Fetch the full static database corpus into the runtime layer exactly once
    records = collection.get(include=["documents", "metadatas"])
    if not records or not records["documents"]:
        return None, {}

    corpus = records["documents"]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    
    _CACHED_BM25_GLOBAL = BM25Okapi(tokenized_corpus)
    _CACHED_GLOBAL_RECORDS = records
    return _CACHED_BM25_GLOBAL, _CACHED_GLOBAL_RECORDS

def warm_bm25_cache(fiscal_years: List[str] = None) -> None:
    """
    Pre-builds BM25 indexes for the given fiscal years (defaults to all
    four supported years).  Call this once at application startup to
    eliminate first-request latency.
 
    Example usage in main.py:
        from contextlib import asynccontextmanager
        from retrieval.hybrid import warm_bm25_cache
 
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            warm_bm25_cache()          # pre-warm all FY indexes
            yield
 
        app = FastAPI(lifespan=lifespan, ...)
 
    Args:
        fiscal_years: list of FY strings to pre-warm, e.g. ["FY23","FY24"].
                      Defaults to ["FY22","FY23","FY24","FY25"].
    """
    if fiscal_years is None:
        fiscal_years = ["FY22", "FY23", "FY24", "FY25"]
 
    for fy in fiscal_years:
        if fy not in _CACHED_BM25_BY_FY:
            logger.info(f"🔥 Pre-warming BM25 index for {fy}...")
            filtered_records = collection.get(
                where={"fy": fy},
                include=["documents", "metadatas"]
            )
            if not filtered_records or not filtered_records["documents"]:
                logger.warning(f"⚠️  No documents found for {fy} — skipping pre-warm.")
                continue
            corpus = filtered_records["documents"]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            _CACHED_BM25_BY_FY[fy]    = BM25Okapi(tokenized_corpus)
            _CACHED_RECORDS_BY_FY[fy] = filtered_records
            logger.info(f"✅ BM25 index for {fy} ready ({len(corpus)} documents).")
        else:
            logger.info(f"♻️  BM25 index for {fy} already cached — skipping.")

def execute_dense_semantic_search(query: str, target_fy: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Queries the local database using free local sentence-transformer semantic matching.
    """
    where_filter = {"fy": target_fy} if target_fy else {}
    
    if where_filter:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    
    output_hits = []
    if not results or not results["documents"] or not results["documents"][0]:
        return output_hits
        
    for i in range(len(results["ids"][0])):
        output_hits.append({
            "id": results["ids"][0][i],
            "contextual_text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": results["distances"][0][i]
        })
    return output_hits

def execute_sparse_lexical_search(query: str, target_fy: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
     """
    BM25 keyword search with per-FY lazy caching.
    First call per FY builds the index; all subsequent calls reuse it.
    """
    # ── CASE A: Filtered Search by Year (Narrows down subset dynamically) ──
     if target_fy:
        if target_fy not in _CACHED_BM25_BY_FY:
            # ── First call for this FY: build and store ───────────
            filtered_records = collection.get(
                where={"fy": target_fy},
                include=["documents", "metadatas"]
            )
            if not filtered_records or not filtered_records["documents"]:
                return []
            corpus = filtered_records["documents"]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            _CACHED_BM25_BY_FY[target_fy]    = BM25Okapi(tokenized_corpus)
            _CACHED_RECORDS_BY_FY[target_fy] = filtered_records
        # ── Every call: reuse the cached engine ───────────────────
        bm25_engine   = _CACHED_BM25_BY_FY[target_fy]
        records_source = _CACHED_RECORDS_BY_FY[target_fy]
 
    # ── CASE B: Global Search (Uses the Lazy Cache Singleton instantly) ──
     else:
        bm25_engine, records_source = _get_or_init_global_bm25()
        if not bm25_engine:
            return []
 
    # ── Scoring + hit construction — COMPLETELY UNCHANGED ────────────
     tokenized_query = query.lower().split()
     scores = bm25_engine.get_scores(tokenized_query)
 
    # In Case A, corpus is already set from the cached records
     corpus = records_source["documents"]
 
     hits = []
     for idx, score in enumerate(scores):
        if score > 0:
            hits.append({
                "id":             records_source["ids"][idx],
                "contextual_text": corpus[idx],
                "metadata":       records_source["metadatas"][idx],
                "sparse_score":   float(score)
            })
 
     hits.sort(key=lambda x: x["sparse_score"], reverse=True)
     return hits[:top_k]