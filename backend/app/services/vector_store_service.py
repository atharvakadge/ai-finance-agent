"""
Vector Store Service - Manages the Qdrant vector database.

Uses a SINGLETON pattern: one shared Qdrant client for the entire app.
Without this, each request creates a new empty in-memory database,
and data from uploads disappears when queries run.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ── SINGLETON CLIENT ──
# Created ONCE when this module is first imported.
# Every VectorStoreService instance shares this same client.
# This is why the upload's data is still there when a query runs.
_client = QdrantClient(":memory:")


class VectorStoreService:

    def __init__(self):
        self.client = _client  # shared instance, not a new one

    def create_collection(self, collection_name: str, vector_size: int = 384):
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if collection_name in existing_names:
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def store_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        search_results = []
        for result in results:
            search_results.append({
                "content": result.payload["content"],
                "chunk_index": result.payload["chunk_index"],
                "score": result.score,
            })

        return search_results