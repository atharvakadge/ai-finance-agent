"""
Unit tests for the Financial Tools Service.

WHY test financial calculations specifically?
- A wrong P/E ratio could mislead investment decisions
- These are deterministic (no LLM randomness) — perfect for testing
- Easy to verify against known correct values

Run with: pytest tests/ -v
"""

import pytest
from app.services.financial_tools_service import FinancialToolsService


@pytest.fixture
def tools():
    """Create a fresh FinancialToolsService for each test."""
    return FinancialToolsService()


# ── P/E Ratio Tests ──

class TestPERatio:

    def test_normal_pe(self, tools):
        """Reliance: price 1265, EPS 51.45 → P/E ~24.59"""
        result = tools.pe_ratio(1265, 51.45)
        assert result["value"] == 24.59
        assert result["ratio"] == "P/E Ratio"

    def test_high_pe(self, tools):
        """High growth stock: price 5000, EPS 50 → P/E 100"""
        result = tools.pe_ratio(5000, 50)
        assert result["value"] == 100.0
        assert "high growth" in result["interpretation"].lower()

    def test_low_pe(self, tools):
        """Value stock: price 100, EPS 20 → P/E 5"""
        result = tools.pe_ratio(100, 20)
        assert result["value"] == 5.0

    def test_negative_pe(self, tools):
        """Loss-making company: negative EPS"""
        result = tools.pe_ratio(100, -10)
        assert result["value"] == -10.0
        assert "unprofitable" in result["interpretation"].lower()

    def test_zero_eps(self, tools):
        """Zero EPS → should return error, not crash"""
        result = tools.pe_ratio(100, 0)
        assert "error" in result


# ── Debt-to-Equity Tests ──

class TestDebtToEquity:

    def test_reliance_de(self, tools):
        """Reliance: debt 347530, equity 843200 → D/E ~0.41"""
        result = tools.debt_to_equity(347530, 843200)
        assert result["value"] == 0.41

    def test_conservative(self, tools):
        """Low leverage company"""
        result = tools.debt_to_equity(1000, 10000)
        assert result["value"] == 0.1
        assert "conservative" in result["interpretation"].lower()

    def test_high_leverage(self, tools):
        """Heavily leveraged company"""
        result = tools.debt_to_equity(5000, 1000)
        assert result["value"] == 5.0
        assert "heavy" in result["interpretation"].lower() or "reliance on debt" in result["interpretation"].lower()

    def test_zero_equity(self, tools):
        """Zero equity → error"""
        result = tools.debt_to_equity(1000, 0)
        assert "error" in result


# ── ROE Tests ──

class TestROE:

    def test_normal_roe(self, tools):
        """Reliance: net income 69621, equity 843200 → ROE ~8.26%"""
        result = tools.roe(69621, 843200)
        assert result["value"] == "8.26%"

    def test_excellent_roe(self, tools):
        """High ROE company"""
        result = tools.roe(5000, 10000)
        assert result["value"] == "50.0%"
        assert "excellent" in result["interpretation"].lower()

    def test_zero_equity(self, tools):
        result = tools.roe(1000, 0)
        assert "error" in result


# ── Net Profit Margin Tests ──

class TestNetProfitMargin:

    def test_reliance_npm(self, tools):
        """Reliance: net income 69621, revenue 980136 → NPM ~7.1%"""
        result = tools.net_profit_margin(69621, 980136)
        assert result["value"] == "7.1%"

    def test_high_margin(self, tools):
        """High margin business like software"""
        result = tools.net_profit_margin(3000, 10000)
        assert result["value"] == "30.0%"
        assert "pricing power" in result["interpretation"].lower()

    def test_thin_margin(self, tools):
        """Thin margin business like retail"""
        result = tools.net_profit_margin(200, 10000)
        assert result["value"] == "2.0%"

    def test_zero_revenue(self, tools):
        result = tools.net_profit_margin(1000, 0)
        assert "error" in result


# ── Current Ratio Tests ──

class TestCurrentRatio:

    def test_healthy(self, tools):
        result = tools.current_ratio(20000, 10000)
        assert result["value"] == 2.0
        assert "healthy" in result["interpretation"].lower()

    def test_below_one(self, tools):
        """Liquidity concern"""
        result = tools.current_ratio(5000, 10000)
        assert result["value"] == 0.5
        assert "difficulty" in result["interpretation"].lower()


# ── Calculate All Tests ──

class TestCalculateAll:

    def test_with_full_data(self, tools):
        """All ratios should be calculated with full data."""
        financials = {
            "market_price": 1265,
            "eps": 51.45,
            "total_debt": 347530,
            "total_equity": 843200,
            "shareholders_equity": 843200,
            "net_income": 69621,
            "revenue": 980136,
        }
        results = tools.calculate_all(financials)
        assert len(results) == 4  # PE, DE, ROE, NPM
        ratio_names = [r["ratio"] for r in results]
        assert "P/E Ratio" in ratio_names
        assert "Debt-to-Equity Ratio" in ratio_names

    def test_with_partial_data(self, tools):
        """Only calculable ratios should be returned."""
        financials = {"market_price": 100, "eps": 10}
        results = tools.calculate_all(financials)
        assert len(results) == 1
        assert results[0]["ratio"] == "P/E Ratio"

    def test_with_no_data(self, tools):
        """Empty input → no ratios."""
        results = tools.calculate_all({})
        assert len(results) == 0