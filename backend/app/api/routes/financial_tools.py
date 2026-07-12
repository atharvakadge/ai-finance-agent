"""
Financial Tools API routes.

Pure calculation endpoints — no LLM, no RAG.
Deterministic: same input always gives same output.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.financial_tools_service import FinancialToolsService

router = APIRouter()
tools = FinancialToolsService()


class RatioRequest(BaseModel):
    """
    Flexible input for ratio calculations.
    Send whichever fields you have — the API calculates
    all ratios possible from the provided data.

    Example:
    {
        "market_price": 2500,
        "eps": 51.45,
        "total_debt": 347530,
        "total_equity": 843200,
        "net_income": 69621,
        "revenue": 980136
    }
    """
    market_price: float | None = None
    eps: float | None = None
    total_debt: float | None = None
    total_equity: float | None = None
    shareholders_equity: float | None = None
    net_income: float | None = None
    revenue: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    nopat: float | None = None
    invested_capital: float | None = None


class SingleRatioRequest(BaseModel):
    """Request for a specific ratio."""
    ratio_name: str  # "pe", "de", "roe", "current", "npm", "roic"
    values: dict     # {"market_price": 2500, "eps": 51.45}


@router.post("/ratios/calculate")
async def calculate_all_ratios(request: RatioRequest):
    """
    POST /api/v1/ratios/calculate

    Send financial data, get back all calculable ratios.
    Only ratios with sufficient input data are returned.
    """
    financials = {k: v for k, v in request.model_dump().items() if v is not None}

    if not financials:
        raise HTTPException(status_code=400, detail="No financial data provided.")

    # Use shareholders_equity for ROE if total_equity not provided
    if "shareholders_equity" not in financials and "total_equity" in financials:
        financials["shareholders_equity"] = financials["total_equity"]

    results = tools.calculate_all(financials)

    if not results:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to calculate any ratios. Need at least 2 related fields.",
        )

    return {
        "ratios": results,
        "count": len(results),
        "input_fields": list(financials.keys()),
    }


@router.post("/ratios/single")
async def calculate_single_ratio(request: SingleRatioRequest):
    """
    POST /api/v1/ratios/single

    Calculate a specific ratio by name.

    Example: {"ratio_name": "pe", "values": {"market_price": 2500, "eps": 51.45}}
    """
    ratio_map = {
        "pe": ("market_price", "eps", tools.pe_ratio),
        "de": ("total_debt", "total_equity", tools.debt_to_equity),
        "roe": ("net_income", "shareholders_equity", tools.roe),
        "current": ("current_assets", "current_liabilities", tools.current_ratio),
        "npm": ("net_income", "revenue", tools.net_profit_margin),
        "roic": ("nopat", "invested_capital", tools.roic),
    }

    name = request.ratio_name.lower()
    if name not in ratio_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ratio: {name}. Available: {list(ratio_map.keys())}",
        )

    param1_name, param2_name, func = ratio_map[name]

    if param1_name not in request.values or param2_name not in request.values:
        raise HTTPException(
            status_code=400,
            detail=f"Ratio '{name}' requires: {param1_name} and {param2_name}",
        )

    result = func(request.values[param1_name], request.values[param2_name])
    return result