"""
Chat API routes — Direct LLM chat (no RAG).
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.prompts import FINANCIAL_ASSISTANT_PROMPT

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[Message] = []


class ChatResponse(BaseModel):
    response: str
    model: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """POST /api/v1/chat — Direct LLM chat without document context."""

    try:
        llm = request.app.state.rag_service.llm_service

        history = [msg.model_dump() for msg in chat_request.conversation_history]

        response = llm.chat(
            user_message=chat_request.message,
            system_prompt=FINANCIAL_ASSISTANT_PROMPT,
            conversation_history=history,
        )

        return ChatResponse(
            response=response,
            model=llm.model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")