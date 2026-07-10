"""
Compare API routes — Compare multiple documents side by side.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.prompts import FINANCIAL_ASSISTANT_PROMPT, COMPARE_PROMPT

router = APIRouter()


class CompareRequest(BaseModel):
    collection_names: list[str]
    question: str
    top_k: int = 3


class CompareSource(BaseModel):
    collection: str
    content: str
    chunk_index: int
    score: float


class CompareResponse(BaseModel):
    answer: str
    sources: list[CompareSource]
    question: str
    collections_searched: int


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(request: Request, compare: CompareRequest):

    if len(compare.collection_names) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 collections to compare.")

    try:
        rag = request.app.state.rag_service
        all_sources = []
        context_parts = []

        for collection_name in compare.collection_names:
            query_vector = rag.embedding_service.embed_text(compare.question)

            results = rag.vector_store.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=compare.top_k,
            )

            context_parts.append(f"\n### Document: {collection_name.upper()}\n")
            for r in results:
                context_parts.append(r["content"])
                all_sources.append(CompareSource(
                    collection=collection_name,
                    content=r["content"],
                    chunk_index=r["chunk_index"],
                    score=r["score"],
                ))
            context_parts.append("\n---\n")

        context = "\n".join(context_parts)

        comparison_prompt = COMPARE_PROMPT.format(
            context=context,
            question=compare.question,
        )

        answer = rag.llm_service.chat(
            user_message=comparison_prompt,
            system_prompt=FINANCIAL_ASSISTANT_PROMPT,
        )

        return CompareResponse(
            answer=answer,
            sources=all_sources,
            question=compare.question,
            collections_searched=len(compare.collection_names),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compare error: {str(e)}")