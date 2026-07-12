"""
Main entry point for the AI Financial Research Agent API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, upload, query, documents, compare, financial_tools, news, agent
from app.services.rag_service import RAGService
from app.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FinLens AI server...")
    logger.info("Loading AI models (this takes ~30 seconds on first run)")
    app.state.rag_service = RAGService()
    logger.info("Models loaded. Server ready.")

    yield

    logger.info("Shutting down FinLens AI server.")


app = FastAPI(
    title="AI Financial Research Agent",
    description="Production-grade AI agent for financial research and analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["RAG Query"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents Management"])
app.include_router(compare.router, prefix="/api/v1", tags=["Compare"])
app.include_router(financial_tools.router, prefix="/api/v1", tags=["Financial Tools"])
app.include_router(news.router, prefix="/api/v1", tags=["News"])
app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])


@app.get("/")
async def root():
    return {
        "status": "running",
        "project": "AI Financial Research Agent",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}