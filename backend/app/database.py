"""
Database configuration and session management.

Uses SQLAlchemy ORM — the industry standard Python database toolkit.

WHY SQLAlchemy instead of raw SQL?
- Write Python classes instead of SQL strings
- Database-agnostic: same code works with SQLite, PostgreSQL, MySQL
- Prevents SQL injection automatically
- Migration support (schema changes without losing data)
- Relationship management (foreign keys, joins)

Connection string examples:
- SQLite:      sqlite:///./finlens.db          (file on disk, zero setup)
- PostgreSQL:  postgresql://user:pass@host/db   (production)
- Switch by changing DATABASE_URL in .env — zero code changes.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings


# Create the database engine
# echo=False in production (no SQL logging), True for debugging
engine = create_engine(
    settings.database_url,
    # SQLite needs this to work with FastAPI's async
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)

# Session factory — creates database sessions for each request
# autocommit=False: we manually commit (explicit > implicit)
# autoflush=False: we control when data is written
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """
    Base class for all database models.

    Every model (ChatHistory, DocumentUpload, etc.) inherits from this.
    SQLAlchemy uses this to track all models and create tables.
    """
    pass


def init_db():
    """
    Create all database tables.

    Called once at startup. If tables already exist, does nothing.
    In production, you'd use Alembic migrations instead.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Database session dependency for FastAPI routes.

    Usage in a route:
        @router.get("/something")
        def get_something(db: Session = Depends(get_db)):
            ...

    The 'yield' pattern ensures the session is always closed,
    even if the route throws an error. This prevents connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()