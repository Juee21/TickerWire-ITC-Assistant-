# app.py
import streamlit as st
import pandas as pd
import time
import asyncio


# 🌟 Import your real compiled LangGraph application from your agent package
from agent.router import compiled_agent_application

# =====================================================================
# 1. APP CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="TickerWire Agentic RAG",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Minimalist CSS to match a clean chat layout and render citations cleanly
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .citation-container {
        font-size: 0.85rem;
        background-color: rgba(151, 151, 151, 0.1);
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        border-left: 3px solid #007bff;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. SESSION STATE MANAGEMENT (CHAT MEMORY)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am your financial intelligence assistant. Ask me anything about fiscal reports, EBITDA, or revenue margins.",
            "structured_data": None,
            "citations": None
        }
    ]

# =====================================================================
# 3. SIDEBAR CONFIGURATION
# =====================================================================
with st.sidebar:
    st.title("🤖 TickerWire AI")
    st.caption("Agentic RAG Engine v2.0 (Local/Offline Core)")
    st.write("---")
    
    st.markdown("### Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Chat history cleared! How can I help you today?",
                "structured_data": None,
                "citations": None
            }
        ]
        st.rerun()
        
    st.write("---")
    st.markdown("""
    **System Status:**
    - Vector Store: `ChromaDB (Local)`
    - Model: `all-MiniLM-L6-v2`
    - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    - Engine: `LangGraph Stateful Execution`
    """)

# =====================================================================
# 4. LIVE AGENT BACKEND INTEGRATION (REAL ASYNC LOOP HANDLING)
# =====================================================================
def query_agentic_rag_backend(user_query: str) -> dict:
    """
    Safely routes the synchronous Streamlit text input into the asynchronous 
    LangGraph agent execution engine, returning clean UI payload structures.
    """
    # Initialize state matching your exact strict schema rules
    initial_graph_state = {
        "query": user_query,
        "original_query": user_query,
        "trace_id": f"st-{int(time.time())}",
        "detected_fy": None,
        "loop_count": 0,
        "max_loops": 3,
        "cost_accumulated": 0.0,
        "router_decision": None,
        "retrieved_context": None,
        "final_output": None,
        "evaluation_score": 0.0
    }
    
    try:
        # Create an isolated event loop to handle LangGraph's async flow inside Streamlit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        graph_output = loop.run_until_complete(compiled_agent_application.ainvoke(initial_graph_state))
        loop.close()
        
        # Pull final settlement object computed by your synthesis layer
        final_payload = graph_output.get("final_output", {})
        decision = final_payload.get("decision", "REFUSE")
        
        if decision == "REFUSE":
            return {
                "answer": final_payload.get("answer", "Query refused by system guardrails."),
                "structured_data": None,
                "citations": None
            }
            
        if decision == "ASK_CLARIFYING_QUESTION":
            return {
                "answer": final_payload.get("clarifying_question_text", "Could you please specify your target timeline context?"),
                "structured_data": None,
                "citations": None
            }
            
        # Successfully matched an evaluation track path
        raw_answer = final_payload.get("answer", "")
        citations = final_payload.get("citations", [])
        
        # Unpack structural dictionary listings safely for the UI component
        formatted_citations = []
        for c in citations:
            formatted_citations.append(f"Document ID: {c.get('doc_id')} | Target Scope: {c.get('scope')}")
            
        # Optional Payload Hook: If your data returns structured metric lists, we parse it into a table
        structured_df = None
        # Example checking if data frames are packed into state from retrieval
        if "table_data" in final_payload:
            structured_df = pd.DataFrame(final_payload["table_data"])

        return {
            "answer": raw_answer,
            "structured_data": structured_df,
            "citations": formatted_citations if formatted_citations else ["ChromaDB Local Context Store"]
        }
        
    except Exception as e:
        return {
            "answer": f"❌ Core Engine Exception: {str(e)}",
            "structured_data": None,
            "citations": None
        }

# =====================================================================
# 5. RENDER CONVERSATION HISTORY
# =====================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("structured_data") is not None:
            st.dataframe(message["structured_data"], use_container_width=True, hide_index=True)
            
        if message.get("citations"):
            for citation in message["citations"]:
                st.markdown(f"<div class='citation-container'>📄 <b>Source Reference:</b> {citation}</div>", unsafe_allow_html=True)

# =====================================================================
# 6. USER INPUT CHAT COMPONENT & RUNTIME LOOP
# =====================================================================
if user_prompt := st.chat_input("Ask about financial statements (e.g., 'What milestone did Branded Packaged Foods cross in FY25?')"):
    
    # 1. Immediately post the user message onto the screen
    st.session_state.messages.append({"role": "user", "content": user_prompt, "structured_data": None, "citations": None})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # 2. Open processing window block to handle live backend computation
    with st.chat_message("assistant"):
        with st.spinner("Agent evaluating inputs, resolving path matrices, and executing RAG retrieval..."):
            
            # Fire real pipeline computation
            response_dict = query_agentic_rag_backend(user_prompt)
            
            text_answer = response_dict.get("answer", "")
            table_df = response_dict.get("structured_data")
            citations_list = response_dict.get("citations")
            
            # 3. Stream data elements cleanly to user viewport
            st.markdown(text_answer)
            if table_df is not None:
                st.dataframe(table_df, use_container_width=True, hide_index=True)
            if citations_list:
                for citation in citations_list:
                    st.markdown(f"<div class='citation-container'>📄 <b>Source Reference:</b> {citation}</div>", unsafe_allow_html=True)
            
            # 4. Save completely resolved history parameters back into memory
            st.session_state.messages.append({
                "role": "assistant",
                "content": text_answer,
                "structured_data": table_df,
                "citations": citations_list
            })