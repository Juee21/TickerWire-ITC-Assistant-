# retrieval/reranker.py
import os
import math
import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, List

# =====================================================================
# 1. HARD SYSTEM THREAD CONTROL (MUST RUN FIRST)
# =====================================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # Forces local weight use, prevents unauthenticated web hits
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from sentence_transformers import CrossEncoder
from retrieval.hybrid import execute_dense_semantic_search, execute_sparse_lexical_search

logger = logging.getLogger(__name__)

# =====================================================================
# 2. MAIN-THREAD PRE-WARMING
# =====================================================================
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
logger.info(f"💾 Pre-warming Local Cross-Encoder on Main Thread: {MODEL_NAME}...")

try:
    _GLOBAL_RERANKER_MODEL = CrossEncoder(MODEL_NAME, device="cpu")
    logger.info("✅ Cross-Encoder memory initialization complete.")
except Exception as init_err:
    logger.critical(f"🔥 Critical model loading crash: {init_err}")
    raise RuntimeError(init_err)

# Dedicated thread pool isolated from the primary async loop execution pool
_RERANKER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="tickerwire_rerank_worker"
)

def _extract_text_safely(candidate: Any) -> str:
    if candidate is None:
        return ""
    if isinstance(candidate, dict):
        for key in ["contextual_text", "verbatim", "text", "page_content", "content"]:
            if key in candidate and candidate[key] is not None:
                return str(candidate[key]).strip()
        return str(candidate)
    if hasattr(candidate, "page_content") and candidate.page_content is not None:
        return str(candidate.page_content).strip()
    if hasattr(candidate, "text") and candidate.text is not None:
        return str(candidate.text).strip()
    return str(candidate).strip()

def _rerank_sync_core(query: str, passages: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Performs continuous multi-document matrix classification within the background pool."""
    pairs = [[query, passage] for passage in passages]
    scores = _GLOBAL_RERANKER_MODEL.predict(pairs, batch_size=min(len(pairs), 8), show_progress_bar=False)

    scored = sorted(
        [{"index": i, "score": float(s)} for i, s in enumerate(scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    return scored[:top_k]

# =====================================================================
# ASYNC GRAPH INTERFACE NODE ENTRY
# =====================================================================
async def get_integrated_hybrid_context(query: str, target_fy: Any = None) -> Dict[str, Any]:
    if isinstance(target_fy, list):
        resolved_fy = target_fy[0] if target_fy else None
    else:
        resolved_fy = target_fy

    try:
        dense_hits  = execute_dense_semantic_search(query, target_fy=resolved_fy, top_k=10)
        sparse_hits = execute_sparse_lexical_search(query, target_fy=resolved_fy, top_k=5)
    except Exception as e:
        logger.error(f"❌ Index retrieval execution failure: {e}")
        return _fallback_response("Index retrieval fault.", resolved_fy)

    # Deduplicate matching strings
    combined, seen = [], set()
    for hit in (dense_hits + sparse_hits):
        hit_id = str(hit.get("id", id(hit))) if isinstance(hit, dict) else str(id(hit))
        if hit_id not in seen:
            combined.append(hit)
            seen.add(hit_id)

    if not combined:
        return _fallback_response("No document context segments located.", resolved_fy)

    passages = [_extract_text_safely(c) for c in combined]

    try:
        loop = asyncio.get_running_loop()
        ranked = await loop.run_in_executor(
            _RERANKER_EXECUTOR,
            _rerank_sync_core,
            query,
            passages,
            5  # Extract top 5 elements to ensure rich context injection
        )
    except Exception as process_err:
        logger.error(f"⚠️ Reranker runtime fault: {process_err}. Triggering pass-through.")
        ranked = [{"index": i, "score": 0.0} for i in range(min(5, len(combined)))]

    context_strings = []
    highest_confidence = 0.50
    primary_id = "unknown-chunk"

    for idx, item in enumerate(ranked):
        candidate = combined[item["index"]]
        raw_score = item["score"]

        try:
            confidence = 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            confidence = 1.0 if raw_score > 0 else 0.0

        if idx == 0:
            highest_confidence = round(confidence, 4)
            if isinstance(candidate, dict):
                primary_id = str(candidate.get("id", "unknown-chunk"))

        extracted_text = _extract_text_safely(candidate)
        context_strings.append(f"[Document Segment {idx + 1}]\\n{extracted_text}")

    unified_context = "\\n\\n".join(context_strings)
    
    # FLATTENED SCHEMA TARGET CONTRACT
    return {
        "id": primary_id,
        "contextual_text": unified_context,
        "verbatim": unified_context,
        "fy": str(resolved_fy or "GLOBAL"),
        "confidence": float(highest_confidence),
        "source": "ITC Limited Annual Report (Local Cross-Encoder Ranked Pool)",
        "api_called": False
    }

def _fallback_response(reason: str, fy: Any) -> Dict[str, Any]:
    return {
        "id": "failure-fallback-node",
        "contextual_text": reason,
        "verbatim": reason,
        "fy": str(fy or "UNKNOWN"),
        "confidence": 0.0,
        "source": "ITC Limited Annual Report (System Error Fallback)",
        "api_called": False
    }