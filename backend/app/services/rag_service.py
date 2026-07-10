"""
RAG Service - Orchestrates the complete RAG pipeline.
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
        query_vector = self.embedding_service.embed_text(question)

        search_results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
        )

        context = "\n\n---\n\n".join(
            [result["content"] for result in search_results]
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
            "sources": search_results,
            "question": question,
            "chunks_used": len(search_results),
        }