"""
Document upload API routes.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    collection_name: str
    total_chunks: int
    vector_dimension: int
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()

        rag = request.app.state.rag_service

        collection_name = (
            file.filename
            .replace(".pdf", "")
            .replace(" ", "_")
            .lower()
        )

        result = rag.ingest_pdf(
            pdf_bytes=pdf_bytes,
            collection_name=collection_name,
        )

        return UploadResponse(
            filename=file.filename,
            collection_name=result["collection_name"],
            total_chunks=result["total_chunks"],
            vector_dimension=result["vector_dimension"],
            message=f"Successfully ingested {file.filename} into collection '{collection_name}'.",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")