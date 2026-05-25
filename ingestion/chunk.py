from typing import List, Dict

def split_document_into_chunks(raw_text: str, fiscal_year: str) -> List[Dict[str, str]]:
    """
    Splits text into atomic structural lines or sentences for indexing.
    """
    # Splitting cleanly by newlines and cleaning empty lines
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    chunks = []
    
    current_chunk = []
    current_length = 0
    chunk_index = 0
    
    for line in lines:
        current_chunk.append(line)
        current_length += len(line.split())
        
        # Create a chunk roughly every 150 words
        if current_length >= 150:
            chunks.append({
                "chunk_id": f"chunk_{fiscal_year.lower()}_{chunk_index}",
                "text": " ".join(current_chunk),
                "fy": fiscal_year
            })
            current_chunk = []
            current_length = 0
            chunk_index += 1
            
    if current_chunk:
        chunks.append({
            "chunk_id": f"chunk_{fiscal_year.lower()}_{chunk_index}",
            "text": " ".join(current_chunk),
            "fy": fiscal_year
        })
        
    return chunks