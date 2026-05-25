from typing import List, Dict

def enrich_chunks_with_context(chunks: List[Dict[str, str]], document_summary: str) -> List[Dict[str, str]]:
    """
    Prepends situational structural context snippets onto raw text chunks
    to avoid contextual decay during vector retrieval operations.
    """
    enriched_chunks = []
    for chunk in chunks:
        situational_context = f"ITC Limited Annual Report {chunk['fy']} | Document Segment Focus: {document_summary}."
        contextualized_text = f"<{situational_context}> Content: {chunk['text']}"
        
        enriched_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "raw_text": chunk["text"],
            "contextual_text": contextualized_text,
            "fy": chunk["fy"]
        })
    return enriched_chunks