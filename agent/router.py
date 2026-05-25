# agent/router.py
from asyncio.log import logger
import re
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from retrieval.reranker import get_integrated_hybrid_context
import logging


# Import the decoupled MCP service pipeline
from mcp.server import mcp_service_node

logger = logging.getLogger(__name__)

class AgentGraphState(TypedDict):
    query: str
    original_query: str
    trace_id: str
    detected_fy: Optional[List[str]]  # Supports single or multi-year analysis
    loop_count: int
    max_loops: int
    router_decision: Optional[str]
    retrieved_context: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    evaluation_score: float  # Reflection metrics

# Utility Function: Financial Text Formatter
_METRIC_PATTERNS: List[tuple] = [
    ("Revenue from Operations",  r"revenue\s+from\s+operations[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Total Revenue",            r"total\s+revenue[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Profit Before Tax (PBT)",  r"profit\s+before\s+tax[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Profit After Tax (PAT)",   r"profit\s+after\s+tax[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Segment EBITDA",           r"(?:segment\s+)?ebitda[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Gross Profit",             r"gross\s+profit[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("EBIT",                     r"\bebit\b[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
    ("Net Sales",                r"net\s+sales[\s:₹]*([\d,]+\.?\d*)\s+([\d,]+\.?\d*)"),
]
 
 
def _format_number(raw: str) -> str:
    """Normalise a raw numeric string to comma-formatted display value."""
    cleaned = raw.replace(",", "")
    try:
        return f"{float(cleaned):,.2f}" if "." in cleaned else f"{int(cleaned):,}"
    except ValueError:
        return raw
 
 
def _split_segments(unified_text: str) -> List[str]:
    """
    Split the unified context string on the [Document Segment N] markers
    that reranker.py writes.  Returns a list of non-empty segment bodies.
    """
    # Handles both the literal \n escape (reranker writes \\n) and real newlines
    normalised = unified_text.replace("\\n", "\n")
    parts = re.split(r"\[Document Segment \d+\]", normalised)
    return [p.strip() for p in parts if p.strip()]
 
 
def _deduplicate_lines(segments: List[str]) -> List[str]:
    """
    Merge all segments into a deduplicated list of lines.
    Preserves first-occurrence order; removes exact duplicates.
    """
    seen: set = set()
    result: List[str] = []
    for seg in segments:
        for line in seg.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                result.append(line)
    return result
 
 
def synthesize_financial_answer(unified_context: str, target_fy_list: list, confidence: float) -> str:
    """
    Converts a raw unified retrieval context string into a clean,
    structured Markdown answer suitable for financial QA.
 
    Steps:
      1. Split into labelled segments.
      2. Deduplicate lines across segments.
      3. Attempt metric table extraction.
      4. Fall back to bullet-point prose summary.
      5. Append confidence badge + citation footer.
 
    Args:
        unified_context:  The 'verbatim' field from the reranker response.
        target_fy_list:   List of fiscal years e.g. ["FY23"].
        confidence:       Sigmoid-normalised reranker confidence (0–1).
 
    Returns:
        A Markdown-formatted string ready to send to the user.
    """
    year_label = ", ".join(target_fy_list) if target_fy_list else "Filing Context"
 
    # ── Step 1 & 2: segment split + deduplication ─────────────────────
    segments     = _split_segments(unified_context)
    unique_lines = _deduplicate_lines(segments)
    clean_body   = "\n".join(unique_lines)
    search_text  = clean_body.lower()
 
    # ── Step 3: attempt numeric metric table extraction ───────────────
    extracted_rows: List[str] = []
    for metric_name, pattern in _METRIC_PATTERNS:
        match = re.search(pattern, search_text)
        if match:
            val_curr = _format_number(match.group(1))
            val_prev = _format_number(match.group(2))
            extracted_rows.append(f"| {metric_name} | {val_curr} | {val_prev} |")
 
    # ── Step 4a: structured table output ─────────────────────────────
    if extracted_rows:
        output = (
            f"### 📊 Financial Highlights — {year_label}\n\n"
            f"| Metric | Current Period (₹ Cr) | Previous Period (₹ Cr) |\n"
            f"| :--- | ---: | ---: |\n"
        )
        output += "\n".join(extracted_rows) + "\n\n"
 
    # ── Step 4b: prose bullet fallback ───────────────────────────────
    else:
        # Take up to 8 most informative lines (skip very short lines)
        prose_lines = [ln for ln in unique_lines if len(ln) > 40][:8]
        if not prose_lines:
            prose_lines = unique_lines[:6]   # last resort: anything
 
        output = f"### 📖 Analysis Summary — {year_label}\n\n"
        for line in prose_lines:
            # Capitalise first character; strip trailing punctuation artifacts
            line = line.rstrip("|").strip()
            if line:
                output += f"- {line[0].upper()}{line[1:]}\n"
        output += "\n"
 
    # ── Step 5: confidence badge ──────────────────────────────────────
    if confidence >= 0.75:
        badge = "🟢 High"
    elif confidence >= 0.45:
        badge = "🟡 Medium"
    else:
        badge = "🔴 Low"
 
    output += (
        f"---\n"
        f"**Retrieval Confidence:** {badge} ({confidence:.0%})  \n"
        f"**Source:** ITC Limited Annual Report ({year_label})  \n"
        f"**Grounding:** Local CrossEncoder-ranked hybrid retrieval\n"
    )
    return output

# Node 1: Dynamic Decision-Making & Guardrail Supervisor
async def supervisor_router_node(state: AgentGraphState) -> Dict[str, Any]:
    query_str = state["query"].lower()
    
    # Refusal Handling: Production-grade regex perimeter check
    banned_patterns = r"\b(reliance|ril|tcs|infosys|competitor|competitors|rival|wipro)\b"
    if re.search(banned_patterns, query_str):
        return {"router_decision": "REFUSE"}
        
    # Catch completely out-of-scope non-financial inputs
    financial_keywords =[
        "revenue", "ebitda", "profit", "fmcg", "hotel", "agri", "report",
        "growth", "financial", "margins", "segment", "business", "fy", "itc",
        "sales", "income", "earnings", "turnover", "expenditure", "dividend",
    ]
    if not any(kw in query_str for kw in financial_keywords):
        return {"router_decision": "REFUSE"}

    # Adaptive Target Extraction: Capture all year targets to handle comparisons
    found_years = []
    for target_year in ["FY22", "FY23", "FY24", "FY25"]:
        if target_year.lower() in query_str:
            found_years.append(target_year)
            
    # Clarification Handling: Activate if user intent lacks sufficient context
    if not found_years:
        return {"router_decision": "ASK_CLARIFYING_QUESTION", "detected_fy": None}
        
    return {"router_decision": "RETRIEVE", "detected_fy": found_years}

# Node 2: Adaptive Retrieval / Model Context Protocol Node
# agent/router.py

# Ensure your function signature is async (which it should already be)
async def mcp_retrieval_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Retrieves grounded context utilizing the local asynchronous cross-encoder pipeline.
    """
    # ── FIX-3: hard loop-count safety guard ──────────────────────────
    current_loop = state.get("loop_count", 0)
    max_loops    = state.get("max_loops", 2)
    if current_loop >= max_loops:
        logger.warning(
            f"⛔ Loop guard triggered: loop_count={current_loop} >= max_loops={max_loops}. "
            "Aborting retrieval to prevent infinite cycle."
        )
        return {
            "router_decision": "REFUSE",
            "final_output": {
                "decision": "REFUSE",
                "answer": "Maximum retrieval attempts reached without satisfactory grounding confidence."
            },
            "loop_count": current_loop,
        }
    # ─────────────────────────────────────────────────────────────────
 
    logger.info(f"🔍 Accessing Local Hybrid Retrieval Tool for Query: {state['query']}")
 
    mcp_response = await get_integrated_hybrid_context(
        query=state["query"],
        target_fy=state.get("detected_fy")
    )
 
    calculated_cost = state.get("cost_accumulated", 0.0)
    if mcp_response.get("api_called", False):
        calculated_cost += 0.002
 
    # ── FIX-4: propagate reranker confidence into evaluation_score ───
    retrieval_confidence = mcp_response.get("confidence", 0.0)
    # ─────────────────────────────────────────────────────────────────
 
    return {
        "retrieved_context": mcp_response,
        "loop_count": current_loop + 1,
        "cost_accumulated": calculated_cost,
        "evaluation_score": retrieval_confidence,   # FIX-4
    }

# Node 3: Synthesis Generation

async def answer_synthesis_node(state: AgentGraphState) -> Dict[str, Any]:
    ctx = state["retrieved_context"]
 
    # Low-confidence or empty retrieval → refuse cleanly
    confidence = ctx.get("confidence", 0.0) if ctx else 0.0
    if not ctx or confidence < 0.20:
        return {
            "router_decision": "REFUSE",
            "final_output": {
                "decision": "REFUSE",
                "answer": (
                    f"Unable to provide a grounded answer — retrieval confidence "
                    f"({confidence:.0%}) fell below the minimum threshold (20%). "
                    "Please try rephrasing your question or specifying a different fiscal year."
                )
            }
        }
 
    unified_context = ctx.get("verbatim", "")
    target_years    = state.get("detected_fy", [])
 
    formatted_answer = synthesize_financial_answer(
        unified_context=unified_context,
        target_fy_list=target_years,
        confidence=confidence,        # passes real signal to badge
    )
 
    return {
        # ── FIX-1b: tell the graph this path is done ────────────
        "router_decision": "FINALIZE",              # FIX-1b  ← THE KEY FIX
        "final_output": {
            "decision": "ANSWER_WITH_CITATION",
            "answer": formatted_answer,
            "citations": [{"doc_id": ctx.get("id"), "scope": ctx.get("source")}]
        }
    }

# Node 4: Reflection & Self-Evaluation Node (The Self-Correction Core)
async def reflection_evaluation_node(state: AgentGraphState) -> Dict[str, Any]:
    score = state["evaluation_score"]
    current_loop = state["loop_count"]
    
    # Confidence-Based Routing Threshold
    if score >= 0.75 or current_loop >= state["max_loops"]:
        # Pass directly to final response mapping
        return {
            "router_decision": "FINALIZE",
            "final_output": {
                "decision": "ANSWER_WITH_CITATION",
                "answer": f"Verified Grounded Text: {state['retrieved_context'].get('verbatim')}",
                "citations": [{"doc_id": state['retrieved_context'].get("id"), "scope": state['retrieved_context'].get("source")}],
                "meta": {"loops_executed": current_loop, "grounding_confidence": score}
            }
        }
    
    # Self-Correction Trigger: Grounding thresholds unmet; clear for query rewrite loop
    print(f"⚠️ Reflection flag raised: Score {score} below acceptable threshold. Triggering retry loop...")
    return {"router_decision": "REWRITE"}

# Node 5: Query Rewriting Node
async def query_rewriter_node(state: AgentGraphState) -> Dict[str, Any]:
    logger.info("🔧 Self-Correction Gate: Optimizing query string for next retrieval pass...")
    current_query = state["query"]
    # Simple semantic expansion fallback logic.
    # Dynamically introduces structural synonyms to unblock parsing traps.
    rewritten_query = current_query
    if "revenue" in current_query.lower() and "gross" not in current_query.lower():
        rewritten_query += " Gross Revenue Financial Highlights"
        
    return {
        "query": rewritten_query, 
        "loop_count": state["loop_count"] + 1,
        "router_decision": "RETRIEVE"
    }

# Node 6: Clarification Response Generator
async def clarification_node(state: AgentGraphState) -> Dict[str, Any]:
    return {"final_output": {
        "decision": "ASK_CLARIFYING_QUESTION",
        "clarifying_question_text": "Could you please specify which fiscal reporting timeline (choices: FY22 through FY25) you want to extract numbers for?"
    }}

# Node 7: Refusal Handler Guardrail
async def refusal_node(state: AgentGraphState) -> Dict[str, Any]:
    return {"final_output": {
        "decision": "REFUSE",
        "answer": "Refusal: The requested operations fall outside authorized tracking parameters or brand safety scopes."
    }}

# =====================================================================
# AGENT STATE GRAPH TOPOLOGY MATRIX
# =====================================================================
workflow = StateGraph(AgentGraphState)

# Component Registration
workflow.add_node("SupervisorRouter", supervisor_router_node)
workflow.add_node("MCPRetrievalTool", mcp_retrieval_node)
workflow.add_node("AnswerSynthesizer", answer_synthesis_node)
workflow.add_node("ReflectionEvaluator", reflection_evaluation_node)
workflow.add_node("QueryRewriter", query_rewriter_node)
workflow.add_node("ClarificationGenerator", clarification_node)
workflow.add_node("RefusalGenerator", refusal_node)

workflow.set_entry_point("SupervisorRouter")

# Perimeter Routing Conditional Branch
workflow.add_conditional_edges(
    "SupervisorRouter",
    lambda state: state["router_decision"],
    {
        "RETRIEVE": "MCPRetrievalTool",
        "ASK_CLARIFYING_QUESTION": "ClarificationGenerator",
        "REFUSE": "RefusalGenerator"
    }
)

# Linear connection from collection to synthesis
workflow.add_edge("MCPRetrievalTool", "AnswerSynthesizer")

# Post-Synthesis Evaluation Filter Path
workflow.add_conditional_edges(
    "AnswerSynthesizer",
    lambda state: state["router_decision"],
    {
        "FINALIZE": END,               
        "REFUSE":   END,                
        "EVALUATE": "ReflectionEvaluator",
        "REWRITE":  "QueryRewriter",
        "RETRIEVE": "QueryRewriter",
    }
)

# Core Feedback Evaluation Loop Path
workflow.add_conditional_edges(
    "ReflectionEvaluator",
    lambda state: state["router_decision"],
    {
        "FINALIZE": END,
        "RETRIEVE": "QueryRewriter",
        "REWRITE": "QueryRewriter"
    }
)

# Connect Rewrite Node back up into the automated retrieval node
workflow.add_edge("QueryRewriter", "MCPRetrievalTool")

# Static execution exit states
workflow.add_edge("ClarificationGenerator", END)
workflow.add_edge("RefusalGenerator", END)

compiled_agent_application = workflow.compile()