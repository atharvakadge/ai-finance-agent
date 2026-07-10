"""
Query API routes — Ask questions about uploaded documents.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    collection_name: str
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    content: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str
    chunks_used: int


@router.post("/query", response_model=QueryResponse)
async def query_document(request: Request, query: QueryRequest):

    try:
        rag = request.app.state.rag_service

        result = rag.query(
            question=query.question,
            collection_name=query.collection_name,
            top_k=query.top_k,
        )

        sources = [
            SourceChunk(
                content=s["content"],
                chunk_index=s["chunk_index"],
                score=s["score"],
            )
            for s in result["sources"]
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=result["question"],
            chunks_used=result["chunks_used"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")