"""
Database Service - All database operations in one place.

CRUD = Create, Read, Update, Delete — the four basic database operations.

Single Responsibility: routes handle HTTP, this service handles database.
Routes never write SQL or touch the session directly.
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.database_models import ChatHistory, DocumentUpload, QueryLog


class DatabaseService:
    """Handles all database operations."""

    def __init__(self, db: Session):
        """
        Takes a database session.

        The session is created per-request by FastAPI's dependency
        injection (get_db) and passed here. This ensures each
        request has its own session — no cross-request data leaks.
        """
        self.db = db

    # ── Chat History ──

    def save_chat(
        self,
        question: str,
        answer: str,
        model: str = None,
        collection_name: str = None,
        tools_used: list = None,
        sources_count: int = 0,
    ) -> ChatHistory:
        """Save a chat interaction to the database."""
        chat = ChatHistory(
            question=question,
            answer=answer,
            model=model,
            collection_name=collection_name,
            tools_used=tools_used,
            sources_count=sources_count,
        )
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def get_chat_history(self, limit: int = 50, collection_name: str = None) -> list[dict]:
        """
        Retrieve recent chat history.

        Optional filter by collection_name to show
        only chats about a specific document.
        """
        query = self.db.query(ChatHistory)

        if collection_name:
            query = query.filter(ChatHistory.collection_name == collection_name)

        chats = query.order_by(desc(ChatHistory.created_at)).limit(limit).all()
        return [chat.to_dict() for chat in chats]

    # ── Document Uploads ──

    def save_upload(
        self,
        filename: str,
        collection_name: str,
        total_chunks: int,
        vector_dimension: int = 384,
        file_size_bytes: int = None,
    ) -> DocumentUpload:
        """Record a document upload."""
        # Check if collection already exists (re-upload)
        existing = self.db.query(DocumentUpload).filter(
            DocumentUpload.collection_name == collection_name
        ).first()

        if existing:
            # Update existing record
            existing.filename = filename
            existing.total_chunks = total_chunks
            existing.vector_dimension = vector_dimension
            existing.file_size_bytes = file_size_bytes
            existing.status = "completed"
            self.db.commit()
            self.db.refresh(existing)
            return existing

        upload = DocumentUpload(
            filename=filename,
            collection_name=collection_name,
            total_chunks=total_chunks,
            vector_dimension=vector_dimension,
            file_size_bytes=file_size_bytes,
            status="completed",
        )
        self.db.add(upload)
        self.db.commit()
        self.db.refresh(upload)
        return upload

    def get_uploads(self) -> list[dict]:
        """Get all uploaded documents."""
        uploads = self.db.query(DocumentUpload).order_by(
            desc(DocumentUpload.created_at)
        ).all()
        return [u.to_dict() for u in uploads]

    def delete_upload(self, collection_name: str) -> bool:
        """Remove upload record."""
        upload = self.db.query(DocumentUpload).filter(
            DocumentUpload.collection_name == collection_name
        ).first()
        if upload:
            self.db.delete(upload)
            self.db.commit()
            return True
        return False

    # ── Query Logs ──

    def log_query(
        self,
        question: str,
        query_type: str,
        collection_name: str = None,
        chunks_retrieved: int = 0,
        avg_relevance_score: float = None,
        response_length: int = 0,
    ) -> QueryLog:
        """Log a query for analytics."""
        log = QueryLog(
            question=question,
            collection_name=collection_name,
            query_type=query_type,
            chunks_retrieved=chunks_retrieved,
            avg_relevance_score=avg_relevance_score,
            response_length=response_length,
        )
        self.db.add(log)
        self.db.commit()
        return log

    def get_query_stats(self) -> dict:
        """Get query analytics summary."""
        total = self.db.query(QueryLog).count()
        by_type = {}
        for qt in ["rag", "chat", "compare", "agent"]:
            count = self.db.query(QueryLog).filter(QueryLog.query_type == qt).count()
            if count > 0:
                by_type[qt] = count

        return {
            "total_queries": total,
            "by_type": by_type,
        }