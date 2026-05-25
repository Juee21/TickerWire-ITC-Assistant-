### 📘 TickerWire–ITC Assistant
Agentic RAG System for Financial Document Intelligence (FY22–FY25)
🚀 Overview

TickerWire–ITC Assistant is an Agentic Retrieval-Augmented Generation (RAG) system designed to answer financial queries over ITC Limited’s Annual Reports (FY22–FY25).

It solves a real-world newsroom bottleneck where journalists rely on analysts for financial data extraction, leading to delays (25–40 minutes) and occasional inaccuracies in numeric reporting and year attribution.

## This system provides:

⚡ Fast retrieval of financial insights
📊 Structured, citation-backed answers
🧠 Agent-based decision routing
🛑 Hallucination prevention via guardrails
🔍 Hybrid retrieval (BM25 + Dense embeddings)
🎯 Problem Statement

Financial journalists need accurate, grounded answers from large annual reports. Existing workflows are:

Slow (manual analyst dependency)
Error-prone (wrong year / incorrect numbers)
Non-scalable under high query volume
Goal:

## Build an Agentic RAG system that:

Answers ITC financial queries (FY22–FY25)
Uses retrieval + reasoning + routing
Produces structured, citation-backed outputs
Handles ambiguity and out-of-domain queries safely
🏗️ System Architecture
User Query
    ↓
Supervisor Router (LangGraph)
    ↓
-------------------------------------------------
| Answer | Retrieve | Clarify | Refuse         |
-------------------------------------------------
    ↓
Hybrid Retrieval Layer
    ├── Dense Search (ChromaDB embeddings)
    ├── BM25 Keyword Search
    ↓
Cross-Encoder Reranker
    ↓
Answer Synthesizer
    ↓
Structured Financial Response (Markdown + Citations)

⚙️ Key Features
🧠 Agentic Routing (LangGraph)
Decision-based execution flow
Supports:
ANSWER_WITH_CITATION
RETRIEVE
ASK_CLARIFYING_QUESTION
REFUSE

🔎 Hybrid Retrieval System
Dense vector search (ChromaDB)
BM25 lexical search
Combines semantic + exact financial matching

🎯 Cross-Encoder Reranking
Improves relevance ranking of retrieved chunks
Ensures high-quality grounding for final answers

📊 Structured Financial Output
Markdown tables for numerical data
Bullet-point insights for narrative analysis
Always includes citations from source documents

🛑 Guardrails
Refuses out-of-domain queries (e.g., other companies)
Asks clarification when financial year is missing
Prevents hallucinated financial values
🧪 Evaluation Harness

Single-command evaluation:

python -m evaluation.eval

Tests:

FY-specific retrieval
Multi-year comparison
Clarification handling
Refusal behavior


📊 Evaluation Results
Scenario	Type	Output	Status	Latency
FY23 Revenue Query	Retrieval	Structured financial table	✅ PASS	2.40s
FY22 vs FY24 Comparison	Multi-year analysis	Comparative table	✅ PASS	2.05s
Missing FY Query	Clarification	Clarifying question	✅ PASS	~0.00s
Cross-company Query	Refusal	Guardrail triggered	✅ PASS	~0.00s

Overall Metrics
✅ Pass Rate: 100% (4/4)
⚡ Avg Latency: ~1.1s/query
⏱ Total Eval Time: 4.45s
🧠 Routing Accuracy: 100%
🧠 Design Decisions

Why LangGraph?

Used for stateful agent workflows instead of linear RAG pipelines. Enables:

Conditional routing
Multi-path execution
Clear separation of concerns
Why Hybrid Retrieval?

Financial data requires:

Exact numeric matching (BM25)
Semantic understanding (Dense embeddings)

Hybrid approach ensures maximum recall + precision.

Why Cross-Encoder?

Improves ranking quality by evaluating query–document pairs directly, ensuring better grounding before generation.

Why Guardrails?

To prevent:

Hallucinated financial values
Cross-company comparisons
Ambiguous answers without FY context
🧪 Debugging & Key Fixes

During development, a major issue was discovered:

❌ Problem:

LangGraph routing loop caused repeated execution of retrieval and synthesis nodes, leading to:

Increased latency
Repeated BM25 rebuilds
Evaluation instability

✅ Fix:
Corrected router state transitions
Ensured proper update of routing decisions
Introduced BM25 caching
Result:
Evaluation runtime reduced from minutes → ~4.5 seconds
Stable single-pass execution
🚫 What Was NOT Built (Intentional Decisions)

To maintain stability and interpretability:

❌ Multi-agent autonomous planning
❌ Recursive reflection loops
❌ External web data augmentation
❌ Over-complex reasoning chains
Reason:

Priority was given to:

reliability, explainability, and evaluation correctness

🧰 Tech Stack
Python
LangGraph (Agent orchestration)
ChromaDB (Vector store)
BM25 (Keyword retrieval)
Sentence Transformers / CrossEncoder
FastAPI (Backend API)
Streamlit (UI layer)
PyTorch (local inference)

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Start backend
uvicorn main:app --reload
3. Run evaluation
python -m evaluation.eval
4. Start UI (optional)
streamlit run app.py

📌 Example Queries
"What was ITC’s Agri Business revenue in FY23?"
"Compare FMCG revenue FY22 vs FY24"
"What happened to Hotels segment in FY23?"
"What is ITC revenue?" → (Clarification triggered)
"Compare ITC with Reliance" → (Refusal triggered)
📈 System Summary

This project demonstrates a production-style Agentic RAG system combining:

Hybrid retrieval (dense + lexical)
Agent-based routing (LangGraph)
Cross-encoder reranking
Structured financial synthesis
Guardrails for safe AI behavior
Automated evaluation harness

🧾 License

For academic and evaluation purposes only.
