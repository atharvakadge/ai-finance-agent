"""
Query API routes — Ask questions about uploaded documents.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.database_service import DatabaseService

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
async def query_document(
    request: Request,
    query: QueryRequest,
    db: Session = Depends(get_db),
):
    try:
        rag = request.app.state.rag_service

        result = rag.query(
            question=query.question,
            collection_name=query.collection_name,
            top_k=query.top_k,
        )

        # Log to database
        db_service = DatabaseService(db)
        scores = [s["score"] for s in result["sources"]] if result["sources"] else []
        avg_score = sum(scores) / len(scores) if scores else None

        db_service.save_chat(
            question=query.question,
            answer=result["answer"],
            collection_name=query.collection_name,
            sources_count=result["chunks_used"],
        )

        db_service.log_query(
            question=query.question,
            query_type="rag",
            collection_name=query.collection_name,
            chunks_retrieved=result["chunks_used"],
            avg_relevance_score=avg_score,
            response_length=len(result["answer"]),
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