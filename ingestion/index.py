import os
from dotenv import load_dotenv
load_dotenv()

from typing import Any
import chromadb
from chromadb.utils import embedding_functions
from ingestion.extract import load_annual_report
from ingestion.chunk import split_document_into_chunks
from ingestion.enrich import enrich_chunks_with_context



# Initialize a persistent local vector store database directory
chroma_client = chromadb.PersistentClient(path="./tw_chroma_db")

# 🌟 SWAP HERE: Use a free local embedding model instead of OpenAI
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="itc_annual_reports", 
    embedding_function=local_ef
)

def build_production_knowledge_base():
    """
    Orchestrates the data extraction, chunking, enrichment, and vector storage steps.
    """
    # Ensure database is only populated if empty to optimize lifecycle startups
    if collection.count() > 0:
        print(f"📦 Found {collection.count()} existing index records. Skipping regeneration.")
        return

    metadata_map = {
        "FY25": "Management Discussion Analysis and FMCG Packaged Foods Performance Highlights",
        "FY24": "Standalone Financial Highlights Statements and EBITDA Reporting Matrix",
        "FY23": "Agri Business Procurement Strategy and Value Crops Revenue Review",
        "FY22": "Hotels Segment Recovery Framework and Travel Occupancy Analysis"
    }
    
    print("🚀 Initializing Knowledge Base build pipeline...")
    for fy, summary in metadata_map.items():
        raw_text = load_annual_report(fy)
        if not raw_text:
            continue
            
        raw_chunks = split_document_into_chunks(raw_text, fy)
        enriched_chunks = enrich_chunks_with_context(raw_chunks, summary)
        
        if not enriched_chunks:
            continue
            
        ids = [item["chunk_id"] for item in enriched_chunks]
        documents = [item["contextual_text"] for item in enriched_chunks]
        metadatas = [{"fy": item["fy"], "raw": item["raw_text"]} for item in enriched_chunks]
        
        # Batch insert into the vector store
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✅ Ingestion complete. Collection size: {collection.count()} records.")

# Automatically invoke processing compilation when module loads
build_production_knowledge_base()