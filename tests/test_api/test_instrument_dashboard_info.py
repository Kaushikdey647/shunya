"""Mapping of Yahoo / yfinance ``t.info`` keys into overview valuation + company profile."""

from __future__ import annotations

from api.services.instrument_dashboard import _company_from_info, _valuation_from_info


def test_valuation_prefers_quote_style_eps_keys() -> None:
    v = _valuation_from_info(
        {
            "trailingPE": 34.0,
            "forwardPE": 29.0,
            "epsTrailingTwelveMonths": 6.12,
            "epsForward": 7.5,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.08,
            "priceToBook": 40.0,
            "priceToSalesTrailing12Months": 8.0,
            "debtToEquity": 1.2,
        }
    )
    assert v.trailing_eps == 6.12
    assert v.forward_eps == 7.5
    assert v.return_on_equity == 0.15
    assert v.return_on_assets == 0.08
    assert v.price_to_sales == 8.0
    assert v.debt_to_equity == 1.2


def test_valuation_falls_back_to_legacy_keys() -> None:
    v = _valuation_from_info(
        {
            "trailingEps": 5.0,
            "forwardEps": 6.0,
        }
    )
    assert v.trailing_eps == 5.0
    assert v.forward_eps == 6.0


def test_price_to_sales_derived_from_market_cap_and_revenue() -> None:
    v = _valuation_from_info(
        {
            "marketCap": 3000.0,
            "totalRevenue": 1000.0,
        }
    )
    assert v.price_to_sales == 3.0


def test_company_profile_uses_description_fallback() -> None:
    c = _company_from_info(
        {
            "description": "TestCo makes widgets.",
            "sector": "Technology",
            "industry": "Software",
        }
    )
    assert c is not None
    assert c.long_business_summary == "TestCo makes widgets."
    assert c.sector == "Technology"
    assert c.industry == "Software"
