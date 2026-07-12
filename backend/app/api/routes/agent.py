"""
Agent API route - The smart endpoint.

Instead of the user choosing /query, /news, or /ratios,
they just ask a question here and the agent figures out
which tools to use.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class AgentRequest(BaseModel):
    """
    Simple input — just a question and optionally which document to use.

    Examples:
    - {"question": "Analyze Reliance's financial health", "collection_name": "reliance"}
    - {"question": "What's the latest news on TCS?"}
    - {"question": "What was the revenue growth?", "collection_name": "reliance"}
    """
    question: str
    collection_name: str | None = None


class AgentNewsItem(BaseModel):
    title: str
    snippet: str
    url: str
    source: str


class AgentResponse(BaseModel):
    answer: str
    sources: list  # RAG sources if used
    news: list     # News articles if fetched
    tools_used: list[str]
    question: str


@router.post("/agent", response_model=AgentResponse)
async def run_agent(request: Request, agent_request: AgentRequest):
    """
    POST /api/v1/agent — Smart endpoint that auto-routes to the right tools.

    The agent:
    1. Classifies the question
    2. Decides which tools to call (RAG, news, ratios)
    3. Executes them
    4. Synthesizes a combined answer
    """
    try:
        from app.services.agent_service import FinancialAgent

        rag_service = request.app.state.rag_service
        agent = FinancialAgent(rag_service)

        result = agent.run(
            question=agent_request.question,
            collection_name=agent_request.collection_name,
        )

        return AgentResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            news=result.get("news", []),
            tools_used=result["tools_used"],
            question=result["question"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")