from typing import Optional, List
from pydantic import BaseModel, Field

# from .vector_store.chroma_store import ChromaVectorStore
from ..vector_store.chroma_store import get_chroma


# ---------------------------------------------------------
# INPUT / OUTPUT SCHEMAS
# ---------------------------------------------------------

class RAGIndexInput(BaseModel):
    texts: List[str] = Field(..., description="List of text chunks to embed and store.")
    metadatas: Optional[List[dict]] = Field(
        default=None,
        description="Optional metadata per chunk."
    )
    namespace: str = Field(
        default="default",
        description="Namespace / collection name for Chroma persistence."
    )


class RAGIndexOutput(BaseModel):
    success: bool
    stored_count: int
    namespace: str


# ---------------------------------------------------------
# TOOL IMPLEMENTATION
# ---------------------------------------------------------

async def rag_index_tool(input: RAGIndexInput) -> RAGIndexOutput:
    """
    Index text into ChromaDB using Ollama embeddings.
    """

    store = get_chroma()

    count = len(input.texts)
    store.add_texts(
        texts=input.texts,
        metadatas=input.metadatas
    )

    return RAGIndexOutput(
        success=True,
        stored_count=count,
        namespace=input.namespace
    )


# ---------------------------------------------------------
# SCHEMA EXPOSURE HELPERS
# ---------------------------------------------------------

def input_schema():
    return RAGIndexInput.model_json_schema()


def output_schema():
    return RAGIndexOutput.model_json_schema()
