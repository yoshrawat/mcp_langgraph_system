from typing import List
from pydantic import BaseModel, Field

# from mcp_server.vectorstore.chroma_store import ChromaVectorStore
from mcp_server.vector_store.chroma_store import ChromaVectorStore


# ---------------------------------------------------------
# INPUT / OUTPUT SCHEMAS
# ---------------------------------------------------------

class RAGQueryInput(BaseModel):
    query: str = Field(..., description="User question to query using embeddings.")
    namespace: str = Field(default="default")
    k: int = Field(default=5, description="Number of results to return.")


class RAGQueryOutput(BaseModel):
    matches: List[dict]
    namespace: str
    query: str


# ---------------------------------------------------------
# TOOL IMPLEMENTATION
# ---------------------------------------------------------

async def rag_query_tool(input: RAGQueryInput) -> RAGQueryOutput:
    """
    Perform vector similarity search in ChromaDB.
    """

    store = ChromaVectorStore(namespace=input.namespace)

    docs = await store.query(
        query=input.query,
        k=input.k
    )

    return RAGQueryOutput(
        matches=docs,
        namespace=input.namespace,
        query=input.query
    )


# ---------------------------------------------------------
# SCHEMA EXPOSURE HELPERS
# ---------------------------------------------------------

def input_schema():
    return RAGQueryInput.model_json_schema()


def output_schema():
    return RAGQueryOutput.model_json_schema()
