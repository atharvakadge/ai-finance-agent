"""
Document management API routes.

List uploaded documents, get collection info, delete documents.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CollectionInfo(BaseModel):
    name: str
    points_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]
    total: int


@router.get("/documents", response_model=CollectionsResponse)
async def list_documents(request: Request):
    """
    GET /api/v1/documents — List all uploaded document collections.

    Useful for:
    - Frontend showing what's been uploaded
    - Checking if a document exists before querying
    - Managing storage
    """
    try:
        rag = request.app.state.rag_service
        vector_store = rag.vector_store

        collections = vector_store.client.get_collections().collections
        result = []

        for col in collections:
            info = vector_store.client.get_collection(col.name)
            result.append(CollectionInfo(
                name=col.name,
                points_count=info.points_count,
            ))

        return CollectionsResponse(
            collections=result,
            total=len(result),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.delete("/documents/{collection_name}")
async def delete_document(request: Request, collection_name: str):
    """
    DELETE /api/v1/documents/{collection_name}

    Remove a document collection from the vector store.
    """
    try:
        rag = request.app.state.rag_service
        rag.vector_store.client.delete_collection(collection_name)

        return {"message": f"Collection '{collection_name}' deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")