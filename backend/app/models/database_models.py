"""
Database Models — Define the structure of our database tables.

Each class = one database table.
Each attribute = one column.

SQLAlchemy maps Python classes to SQL tables automatically:
    ChatHistory class → chat_history table
    DocumentUpload class → document_uploads table

These models work IDENTICALLY with SQLite and PostgreSQL.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class ChatHistory(Base):
    """
    Stores every chat interaction.

    Why persist chat history?
    - Users expect to see their previous conversations
    - Useful for analytics (what do users ask most?)
    - Debugging (reproduce issues from real queries)
    - Training data for fine-tuning (future)
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    model = Column(String(100), nullable=True)
    collection_name = Column(String(200), nullable=True)  # which document was queried
    tools_used = Column(JSON, nullable=True)               # ["document_search", "news_fetch"]
    sources_count = Column(Integer, default=0)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        """Convert to plain dict for API responses."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "collection_name": self.collection_name,
            "tools_used": self.tools_used,
            "sources_count": self.sources_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentUpload(Base):
    """
    Tracks all uploaded documents.

    Why track uploads?
    - Know what documents are available for querying
    - Track processing stats (chunks, dimensions)
    - Show upload history to users
    - Clean up orphaned vector store collections
    """
    __tablename__ = "document_uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    collection_name = Column(String(200), nullable=False, unique=True)
    total_chunks = Column(Integer, default=0)
    vector_dimension = Column(Integer, default=384)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(String(50), default="completed")  # processing, completed, failed
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "collection_name": self.collection_name,
            "total_chunks": self.total_chunks,
            "vector_dimension": self.vector_dimension,
            "file_size_bytes": self.file_size_bytes,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryLog(Base):
    """
    Logs every query for analytics.

    Useful for:
    - Understanding what users ask most
    - Measuring retrieval quality (avg score)
    - Identifying failed queries
    - Performance monitoring (response times)
    """
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    collection_name = Column(String(200), nullable=True)
    query_type = Column(String(50), nullable=False)  # "rag", "chat", "compare", "agent"
    chunks_retrieved = Column(Integer, default=0)
    avg_relevance_score = Column(Float, nullable=True)
    response_length = Column(Integer, default=0)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "collection_name": self.collection_name,
            "query_type": self.query_type,
            "chunks_retrieved": self.chunks_retrieved,
            "avg_relevance_score": self.avg_relevance_score,
            "response_length": self.response_length,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }