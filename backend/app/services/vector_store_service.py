"""
Vector Store Service - Qdrant with metadata filtering.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

_client = QdrantClient(":memory:")


class VectorStoreService:

    def __init__(self):
        self.client = _client

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

    def store_chunks(self, collection_name, chunks, embeddings):
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "page": chunk.get("page", 0),
                        "section": chunk.get("section", "unknown"),
                    },
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search(self, collection_name, query_vector, top_k=5, section_filter=None):
        """
        Semantic search with optional section filtering.

        section_filter="consolidated" -> only search consolidated chunks
        section_filter=None -> search everything
        """
        query_filter = None
        if section_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        ).points

        search_results = []
        for result in results:
            search_results.append({
                "content": result.payload["content"],
                "chunk_index": result.payload["chunk_index"],
                "page": result.payload.get("page", 0),
                "section": result.payload.get("section", "unknown"),
                "score": result.score,
            })

        return search_results