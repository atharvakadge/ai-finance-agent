"""
News API routes - Fetch latest financial news.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.news_service import NewsService

router = APIRouter()
news_service = NewsService()


class NewsRequest(BaseModel):
    company: str
    max_results: int = 5


class NewsArticle(BaseModel):
    title: str
    snippet: str
    url: str
    source: str


class NewsResponse(BaseModel):
    company: str
    articles: list[NewsArticle]
    count: int


@router.post("/news", response_model=NewsResponse)
async def get_news(request: NewsRequest):
    try:
        result = news_service.get_company_news(request.company)

        return NewsResponse(
            company=result["company"],
            articles=[NewsArticle(**a) for a in result["articles"]],
            count=result["count"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News error: {str(e)}")