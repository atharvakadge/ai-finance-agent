"""
Financial Tools Service - Pure financial calculations.

No LLM, no RAG. Just math.

Why separate from LLM service?
- LLMs are bad at math. They hallucinate calculations.
- Financial ratios need exact precision.
- These functions are deterministic — same input, same output.
- Can be unit tested independently.

In Lesson 10, the LangGraph agent will call these tools
when the user asks about ratios instead of trying to calculate
them with the LLM (which would give wrong numbers).
"""


class FinancialToolsService:
    """
    Calculates and explains financial ratios.

    Every method returns a dict with:
    - value: the calculated ratio
    - formula: how it was calculated
    - interpretation: what the number means
    - benchmark: typical industry ranges
    """

    def pe_ratio(self, market_price: float, eps: float) -> dict:
        """Price-to-Earnings ratio."""
        if eps == 0:
            return {"error": "EPS cannot be zero (division by zero)"}

        value = round(market_price / eps, 2)

        return {
            "ratio": "P/E Ratio",
            "value": value,
            "formula": f"{market_price} / {eps} = {value}",
            "interpretation": self._interpret_pe(value),
            "benchmark": {
                "low": "< 15 (potentially undervalued or low growth)",
                "moderate": "15-25 (fair value for stable companies)",
                "high": "> 25 (high growth expectations or overvalued)",
            },
            "inputs": {"market_price": market_price, "eps": eps},
        }

    def debt_to_equity(self, total_debt: float, total_equity: float) -> dict:
        """Debt-to-Equity ratio."""
        if total_equity == 0:
            return {"error": "Equity cannot be zero"}

        value = round(total_debt / total_equity, 2)

        return {
            "ratio": "Debt-to-Equity Ratio",
            "value": value,
            "formula": f"{total_debt} / {total_equity} = {value}",
            "interpretation": self._interpret_de(value),
            "benchmark": {
                "low": "< 0.5 (conservative, strong balance sheet)",
                "moderate": "0.5-1.5 (balanced leverage)",
                "high": "> 1.5 (heavily leveraged, higher risk)",
            },
            "inputs": {"total_debt": total_debt, "total_equity": total_equity},
        }

    def roe(self, net_income: float, shareholders_equity: float) -> dict:
        """Return on Equity."""
        if shareholders_equity == 0:
            return {"error": "Shareholders equity cannot be zero"}

        value = round((net_income / shareholders_equity) * 100, 2)

        return {
            "ratio": "Return on Equity (ROE)",
            "value": f"{value}%",
            "formula": f"({net_income} / {shareholders_equity}) × 100 = {value}%",
            "interpretation": self._interpret_roe(value),
            "benchmark": {
                "low": "< 10% (poor capital efficiency)",
                "moderate": "10-20% (acceptable for most industries)",
                "high": "> 20% (excellent capital efficiency)",
            },
            "inputs": {"net_income": net_income, "shareholders_equity": shareholders_equity},
        }

    def current_ratio(self, current_assets: float, current_liabilities: float) -> dict:
        """Current Ratio - liquidity measure."""
        if current_liabilities == 0:
            return {"error": "Current liabilities cannot be zero"}

        value = round(current_assets / current_liabilities, 2)

        return {
            "ratio": "Current Ratio",
            "value": value,
            "formula": f"{current_assets} / {current_liabilities} = {value}",
            "interpretation": self._interpret_current(value),
            "benchmark": {
                "low": "< 1.0 (may struggle to meet short-term obligations)",
                "healthy": "1.5-2.5 (good liquidity position)",
                "high": "> 3.0 (excess liquidity, capital not deployed efficiently)",
            },
            "inputs": {"current_assets": current_assets, "current_liabilities": current_liabilities},
        }

    def net_profit_margin(self, net_income: float, revenue: float) -> dict:
        """Net Profit Margin."""
        if revenue == 0:
            return {"error": "Revenue cannot be zero"}

        value = round((net_income / revenue) * 100, 2)

        return {
            "ratio": "Net Profit Margin",
            "value": f"{value}%",
            "formula": f"({net_income} / {revenue}) × 100 = {value}%",
            "interpretation": self._interpret_npm(value),
            "benchmark": {
                "low": "< 5% (thin margins, competitive industry)",
                "moderate": "5-15% (healthy for most sectors)",
                "high": "> 15% (strong pricing power or efficiency)",
            },
            "inputs": {"net_income": net_income, "revenue": revenue},
        }

    def roic(self, nopat: float, invested_capital: float) -> dict:
        """Return on Invested Capital."""
        if invested_capital == 0:
            return {"error": "Invested capital cannot be zero"}

        value = round((nopat / invested_capital) * 100, 2)

        return {
            "ratio": "Return on Invested Capital (ROIC)",
            "value": f"{value}%",
            "formula": f"({nopat} / {invested_capital}) × 100 = {value}%",
            "interpretation": f"The company generates {value}% return on every rupee of invested capital. "
                            + ("Above typical WACC — creating shareholder value." if value > 10
                               else "Below typical WACC — potentially destroying value."),
            "benchmark": {
                "poor": "< 8% (below cost of capital for most firms)",
                "acceptable": "8-15% (meeting cost of capital)",
                "excellent": "> 15% (significant value creation)",
            },
            "inputs": {"nopat": nopat, "invested_capital": invested_capital},
        }

    def calculate_all(self, financials: dict) -> list[dict]:
        """
        Calculate all possible ratios from provided financials.

        Args:
            financials: dict with keys like 'revenue', 'net_income',
                       'total_debt', 'total_equity', etc.

        Returns:
            List of ratio results for all calculable ratios.
        """
        results = []

        if "market_price" in financials and "eps" in financials:
            results.append(self.pe_ratio(financials["market_price"], financials["eps"]))

        if "total_debt" in financials and "total_equity" in financials:
            results.append(self.debt_to_equity(financials["total_debt"], financials["total_equity"]))

        if "net_income" in financials and "shareholders_equity" in financials:
            results.append(self.roe(financials["net_income"], financials["shareholders_equity"]))

        if "current_assets" in financials and "current_liabilities" in financials:
            results.append(self.current_ratio(financials["current_assets"], financials["current_liabilities"]))

        if "net_income" in financials and "revenue" in financials:
            results.append(self.net_profit_margin(financials["net_income"], financials["revenue"]))

        if "nopat" in financials and "invested_capital" in financials:
            results.append(self.roic(financials["nopat"], financials["invested_capital"]))

        return results

    # ── Private interpretation helpers ──

    def _interpret_pe(self, value):
        if value < 0:
            return "Negative P/E indicates the company is currently unprofitable."
        if value < 15:
            return f"P/E of {value} suggests the market expects low growth, or the stock may be undervalued relative to earnings."
        if value < 25:
            return f"P/E of {value} indicates moderate growth expectations — typical for stable, established companies."
        return f"P/E of {value} signals high growth expectations. The market is paying a premium for future earnings potential."

    def _interpret_de(self, value):
        if value < 0.5:
            return f"D/E of {value} indicates a conservative capital structure with low leverage risk."
        if value < 1.5:
            return f"D/E of {value} shows balanced use of debt and equity financing."
        return f"D/E of {value} indicates heavy reliance on debt. Higher financial risk but potentially higher returns."

    def _interpret_roe(self, value):
        if value < 10:
            return f"ROE of {value}% suggests the company is not efficiently using shareholder capital."
        if value < 20:
            return f"ROE of {value}% indicates acceptable profitability relative to equity."
        return f"ROE of {value}% shows excellent capital efficiency — the company generates strong returns for shareholders."

    def _interpret_current(self, value):
        if value < 1:
            return f"Current ratio of {value} is below 1 — the company may face difficulty paying short-term debts."
        if value < 2.5:
            return f"Current ratio of {value} indicates healthy liquidity — the company can comfortably meet short-term obligations."
        return f"Current ratio of {value} suggests excess liquidity. Capital may not be deployed optimally."

    def _interpret_npm(self, value):
        if value < 5:
            return f"Net margin of {value}% is thin — typical for high-volume, low-margin industries like retail or commodities."
        if value < 15:
            return f"Net margin of {value}% indicates healthy profitability for most industries."
        return f"Net margin of {value}% shows strong pricing power and operational efficiency."