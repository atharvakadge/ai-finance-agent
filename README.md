# FinLens AI — Financial Research Agent

A production-grade AI-powered financial research terminal that lets you upload company annual reports, ask questions grounded in actual document data, and compare multiple companies side by side — with source citations for every answer.

Built from scratch with **FastAPI**, **Qdrant**, **Sentence Transformers**, and **React**.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What It Does

**Upload** any company's annual report (PDF) → **Ask questions** → Get **cited, analyst-grade answers** pulled directly from the document.

No hallucination. No guessing. Every answer is grounded in the actual report with expandable source citations showing exactly which sections were used.

### Key Features

- **RAG Pipeline** — Retrieval-Augmented Generation: chunks documents, embeds them, searches by meaning, and generates answers grounded in real data
- **Multi-Document Comparison** — Upload Reliance + TCS + HDFC, then ask "Compare the revenue growth of all three companies"
- **Source Citations** — Every response shows the exact document chunks used, with cosine similarity scores
- **Financial-Grade Prompts** — System prompts tuned for financial analysis: bull/bear perspectives, FACT vs OPINION labeling, ratio calculations
- **Professional Terminal UI** — Dark-themed financial research interface with drag-and-drop upload, real-time server health monitoring, and responsive design
- **Model-Agnostic Architecture** — Swap between Groq/Llama, OpenAI, or Gemini with a single config change

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FRONTEND (React)                    │
│         Dark Financial Terminal UI                    │
│   Upload │ Query │ Compare │ Citations Panel          │
└──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │
           ▼          ▼          ▼
┌──────────────────────────────────────────────────────┐
│                 FastAPI Backend                       │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐│
│  │ /upload  │  │ /query  │  │ /compare │  │ /chat  ││
│  └────┬────┘  └────┬────┘  └────┬─────┘  └───┬────┘│
│       │            │            │             │      │
│       ▼            ▼            ▼             ▼      │
│  ┌──────────────────────────────────────────────┐   │
│  │              RAG Service                      │   │
│  │  (Orchestrates the full pipeline)             │   │
│  └──┬──────────┬──────────┬──────────┬──────────┘   │
│     │          │          │          │               │
│     ▼          ▼          ▼          ▼               │
│  ┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │ PDF  │ │Embedding│ │ Vector │ │  LLM   │         │
│  │Service│ │Service  │ │ Store  │ │Service │         │
│  └──────┘ └────────┘ └────────┘ └────────┘         │
│     │          │          │          │               │
│  PyMuPDF  MiniLM-L6   Qdrant    Groq API           │
│            (384d)    (in-memory) (Qwen 3.6)         │
└──────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

```
INGESTION (once per document):
  PDF ──► Extract Text ──► Chunk (1000 chars, 200 overlap) ──► Embed ──► Store in Qdrant

QUERY (every user question):
  Question ──► Embed ──► Cosine Similarity Search ──► Top-K Chunks ──► LLM + Context ──► Cited Answer
```

| Step | Technology | Purpose |
|------|-----------|---------|
| Text Extraction | PyMuPDF | Fast PDF parsing with multi-column support |
| Chunking | LangChain RecursiveCharacterTextSplitter | Smart splitting at paragraph/sentence boundaries |
| Embeddings | all-MiniLM-L6-v2 (384 dimensions) | Free, local, no API costs |
| Vector Search | Qdrant (cosine similarity) | Meaning-based document retrieval |
| Generation | Qwen 3.6 27B via Groq | Fast inference, function calling support |
| Prompt Engineering | Custom financial prompts | FACT vs OPINION, bull/bear, structured output |

---

## Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| **Backend** | FastAPI | Async-native, auto-docs, Pydantic validation |
| **LLM** | Groq (Qwen 3.6 27B) | Free tier, fast inference, model-agnostic design |
| **Embeddings** | Sentence Transformers | Local, free, no API dependency |
| **Vector DB** | Qdrant | Production-ready, runs locally and cloud |
| **PDF Processing** | PyMuPDF | Fastest Python PDF parser |
| **Chunking** | LangChain Text Splitters | Recursive splitting with overlap |
| **Frontend** | React 18 + Custom CSS | Dark terminal UI, no framework bloat |
| **Config** | Pydantic Settings | Type-safe, fail-fast configuration |

---

## Project Structure

```
ai-finance-agent/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── chat.py           # Direct LLM chat
│   │   │   ├── upload.py         # PDF upload + ingestion
│   │   │   ├── query.py          # RAG query endpoint
│   │   │   ├── compare.py        # Multi-document comparison
│   │   │   └── documents.py      # Collection management
│   │   ├── services/
│   │   │   ├── llm_service.py    # LLM API communication
│   │   │   ├── pdf_service.py    # PDF extraction + chunking
│   │   │   ├── embedding_service.py  # Text → vector conversion
│   │   │   ├── vector_store_service.py  # Qdrant operations
│   │   │   └── rag_service.py    # RAG pipeline orchestrator
│   │   ├── config.py             # Environment configuration
│   │   ├── prompts.py            # Financial system prompts
│   │   └── main.py               # FastAPI entry point
│   ├── requirements.txt
│   └── .env                      # API keys (not committed)
├── frontend/
│   └── index.html                # Complete React UI
└── .gitignore
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API key ([console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone
git clone https://github.com/atharvakadge/ai-finance-agent.git
cd ai-finance-agent

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate            # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Configure
cp backend/.env.example backend/.env
# Edit .env and add your GROQ_API_KEY

# Run
cd backend
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`. Open `frontend/index.html` in your browser.

### API Docs
Interactive Swagger documentation at `http://localhost:8000/docs`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload and ingest a PDF |
| `POST` | `/api/v1/query` | Ask a question about a document |
| `POST` | `/api/v1/compare` | Compare multiple documents |
| `POST` | `/api/v1/chat` | Direct LLM chat (no RAG) |
| `GET` | `/api/v1/documents` | List all uploaded documents |
| `DELETE` | `/api/v1/documents/{name}` | Delete a document collection |
| `GET` | `/health` | Server health check |

---

## Engineering Decisions

- **Why FastAPI over Flask?** — Async-native for I/O-bound LLM API calls. Auto-generated docs. Pydantic validation catches bad input at the boundary.
- **Why Qdrant over Pinecone?** — Runs locally (no cloud dependency), production-ready, free. Pinecone locks you into paid cloud from day one.
- **Why local embeddings over OpenAI?** — Zero cost, zero latency, data never leaves the machine. Quality difference is negligible for RAG retrieval.
- **Why Groq?** — Free tier with fast inference. Architecture is model-agnostic — swap to OpenAI/Gemini by changing one environment variable.
- **Why singleton services?** — Embedding model (80MB) loads once at startup via FastAPI lifespan, not per-request. Critical for production performance.
- **Why separate prompts file?** — Prompts are the most frequently iterated part of an AI app. Keeping them separate from code enables rapid tuning without touching business logic.

---

## What I Learned Building This

- **RAG from scratch** — embeddings, chunking strategies, vector similarity search, cosine similarity, retrieval quality tuning
- **LLM API integration** — messages format, token management, temperature tuning, structured output, model-agnostic design
- **Prompt engineering** — Role/Rules/Format/Constraints framework, FACT vs OPINION labeling, financial domain prompts
- **Production patterns** — singleton services, fail-fast configuration, CORS, API versioning, separation of concerns (SOLID)
- **Vector databases** — how HNSW indexing works, why cosine similarity over euclidean for text, collection management

---

## Roadmap

- [ ] LangGraph AI agents for autonomous multi-step research
- [ ] PostgreSQL for persistent chat history and user management
- [ ] Financial news API integration
- [ ] Auto-extract financial statements and calculate ratios
- [ ] Docker + Docker Compose deployment
- [ ] Streaming responses (token-by-token output)

---

## License

MIT

---

*Built as a deep-learning project to master AI engineering concepts through hands-on implementation, not tutorials.*
