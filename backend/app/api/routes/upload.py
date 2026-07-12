"""
Document upload API routes.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.database_service import DatabaseService

router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    collection_name: str
    total_chunks: int
    vector_dimension: int
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
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

        # Save to database
        db_service = DatabaseService(db)
        db_service.save_upload(
            filename=file.filename,
            collection_name=result["collection_name"],
            total_chunks=result["total_chunks"],
            vector_dimension=result["vector_dimension"],
            file_size_bytes=len(pdf_bytes),
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