"""
History API routes - Chat history and analytics.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.database_service import DatabaseService

router = APIRouter()


class HistoryResponse(BaseModel):
    chats: list
    count: int


class StatsResponse(BaseModel):
    total_queries: int
    by_type: dict
    documents_uploaded: int


@router.get("/history")
async def get_history(
    limit: int = 50,
    collection_name: str = None,
    db: Session = Depends(get_db),
):
    """
    GET /api/v1/history — View chat history.

    Depends(get_db) is FastAPI's dependency injection:
    1. FastAPI calls get_db() before the route runs
    2. get_db() creates a database session
    3. The session is passed to this function as 'db'
    4. After the route finishes, get_db() closes the session
    
    This is automatic — no manual session management needed.
    """
    db_service = DatabaseService(db)
    chats = db_service.get_chat_history(limit=limit, collection_name=collection_name)

    return {"chats": chats, "count": len(chats)}


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    GET /api/v1/stats — Query analytics.

    Shows how many queries were made, broken down by type.
    Useful for monitoring and understanding usage patterns.
    """
    db_service = DatabaseService(db)
    stats = db_service.get_query_stats()
    uploads = db_service.get_uploads()

    return {
        "total_queries": stats["total_queries"],
        "by_type": stats["by_type"],
        "documents_uploaded": len(uploads),
        "documents": uploads,
    }