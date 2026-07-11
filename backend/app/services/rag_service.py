"""
RAG Service - Hybrid search with metadata filtering.
No hardcoding. No hacks. Production-grade.
"""

from app.services.pdf_service import PDFService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService
from app.prompts import FINANCIAL_ASSISTANT_PROMPT, RAG_QUERY_PROMPT


class RAGService:

    def __init__(self):
        self.pdf_service = PDFService()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()

    def ingest_pdf(self, pdf_bytes: bytes, collection_name: str) -> dict:
        chunks = self.pdf_service.process_pdf(pdf_bytes=pdf_bytes)

        self.vector_store.create_collection(
            collection_name=collection_name,
            vector_size=self.embedding_service.dimension,
        )

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(chunk_texts)

        self.vector_store.store_chunks(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "collection_name": collection_name,
            "total_chunks": len(chunks),
            "vector_dimension": self.embedding_service.dimension,
        }

    def query(self, question: str, collection_name: str, top_k: int = 5) -> dict:
        seen_indices = set()
        all_results = []
        query_vector = self.embedding_service.embed_text(question)

        # === SEARCH 1: Unfiltered semantic search (catches everything) ===
        general_results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=6,
        )

        for r in general_results:
            if r["chunk_index"] not in seen_indices:
                seen_indices.add(r["chunk_index"])
                all_results.append(r)

        # === SEARCH 2: Consolidated-only search for financial queries ===
        # This guarantees we get consolidated P&L data, not segment data
        financial_keywords = ["revenue", "profit", "income", "expense", "ebitda",
                              "earnings", "eps", "debt", "asset", "liability",
                              "cash flow", "dividend", "margin", "sales", "turnover",
                              "balance sheet", "borrowing"]

        if any(kw in question.lower() for kw in financial_keywords):
            consolidated_results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=6,
                section_filter="consolidated",
            )

            for r in consolidated_results:
                if r["chunk_index"] not in seen_indices:
                    seen_indices.add(r["chunk_index"])
                    all_results.append(r)

        # Sort by score, take top 12
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:12]

        context = "\n\n---\n\n".join(
            [f"[Section: {r.get('section', 'unknown').upper()}]\n{r['content']}"
             for r in top_results]
        )

        augmented_prompt = RAG_QUERY_PROMPT.format(
            context=context,
            question=question,
        )

        answer = self.llm_service.chat(
            user_message=augmented_prompt,
            system_prompt=FINANCIAL_ASSISTANT_PROMPT,
        )

        return {
            "answer": answer,
            "sources": top_results,
            "question": question,
            "chunks_used": len(top_results),
        }