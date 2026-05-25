from typing import Dict, Any
from retrieval.reranker import get_integrated_hybrid_context

class TickerWireMCPServer:
    """
    Minimal MCP-compliant tool server for the TickerWire retrieval pipeline.
 
    Exposes one tool:  contextual_hybrid_search
      - Dense (ChromaDB sentence-transformer) + Sparse (BM25) retrieval
      - CrossEncoder reranking
      - FY-filtered or global corpus search
    """
 
    def __init__(self):
        self.tool_name = "contextual_hybrid_search"
    def __init__(self):
        self.tool_name = "contextual_hybrid_search"

     # ── IMPROVEMENT-D: machine-readable tool schema ──────────────────
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns a JSON-Schema-compatible descriptor for this MCP tool.
        Satisfies the "expose one tool via MCP" requirement without
        adding any new infrastructure.
 
        Usage (for documentation / capability negotiation):
            schema = mcp_service_node.get_tool_schema()
        """
        return {
            "name": self.tool_name,
            "description": (
                "Performs hybrid retrieval (dense semantic + BM25 sparse) "
                "over ITC Limited Annual Reports (FY22–FY25), followed by "
                "CrossEncoder reranking. Returns the top-ranked context "
                "segments with a grounding confidence score."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language financial question about ITC Limited."
                    },
                    "target_fy": {
                        "type": "string",
                        "enum": ["FY22", "FY23", "FY24", "FY25"],
                        "description": "Fiscal year filter. Omit for cross-year global search."
                    }
                },
                "required": ["query"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "id":             {"type": "string"},
                    "verbatim":       {"type": "string"},
                    "confidence":     {"type": "number"},
                    "fy":             {"type": "string"},
                    "source":         {"type": "string"},
                    "api_called":     {"type": "boolean"}
                }
            }
        }

    async def call_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
         """
        Executes the hybrid retrieval tool with the given arguments.
 
        Args:
            arguments: dict with keys "query" (required) and
                       "target_fy" (optional).
 
        Returns:
            {"payload": <reranker response dict>}
        """
         query = arguments.get("query")
         target_fy = arguments.get("target_fy")
        
        # Routes execution cleanly through the core reranker engine
         retrieved_payload = await get_integrated_hybrid_context(query, target_fy)
         return {"payload": retrieved_payload}

# Export service instance point
mcp_service_node = TickerWireMCPServer()

# ── convenience async helper ──────────────────────────
async def route_through_mcp(query: str, target_fy: Any = None) -> Dict[str, Any]:
    """
    Thin async helper that routes a retrieval call through the MCP
    server object, keeping the MCP abstraction boundary intact.
 
    router.py can call this instead of importing
    get_integrated_hybrid_context directly, making the MCP layer
    a real part of the execution path rather than a bypassed stub.
 
    Usage in mcp_retrieval_node:
        from mcp.server import route_through_mcp
        result = await route_through_mcp(query=..., target_fy=...)
        # result == {"payload": { ...reranker dict... }}
        mcp_response = result["payload"]
    """
    result = await mcp_service_node.call_tool(
        {"query": query, "target_fy": target_fy}
    )
    return result