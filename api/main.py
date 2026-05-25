import uuid
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


from agent.router import compiled_agent_application, AgentGraphState


app = FastAPI(title="TickerWire-ITC Production Platform Core")

class QueryPayload(BaseModel):
    query: str

async def streaming_chunks_generator(final_dict: dict):
    serialized_dict = json.dumps(final_dict)
    for offset in range(0, len(serialized_dict), 12):
        yield serialized_dict[offset:offset+12]
        await asyncio.sleep(0.05)

@app.post("/query")
async def execute_query_endpoint(payload: QueryPayload, request: Request):
    trace_id = request.headers.get("X-Distributed-Trace-Id", uuid.uuid4().hex)

    initial_graph_state = {
        "query": payload.query,
        "original_query": payload.query,
        "trace_id": trace_id,
        "detected_fy": None,
        "loop_count": 0,
        "max_loops": 1,  # Prevents infinite computing loops
        "cost_accumulated": 0.0,
        "router_decision": None,
        "retrieved_context": None,
        "final_output": None,
        "evaluation_score": 0.0
    }

    try:
        # Fixed execution endpoint to match the imported compiled graph object
        execution_output = await compiled_agent_application.ainvoke(initial_graph_state)
        return StreamingResponse(
            streaming_chunks_generator(execution_output.get("final_output", {})), 
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Execution Failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main.py:app", host="127.0.0.1", port=8000, reload=True)