from __future__ import annotations

from typing import Dict, List

import requests


def suggest_tickers_for_isin(isin: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Usa la búsqueda de Yahoo Finance para sugerir posibles tickers para un ISIN.

    Devuelve una lista de dicts con:
      - symbol
      - shortname
      - exchange
      - currency
    """
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        resp = requests.get(
            url,
            params={"q": isin, "quotesCount": max_results, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0 (portfolio-evolution)"},
            timeout=5,
        )
        print(f"[tickers.suggest_tickers_for_isin] {isin}: HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[tickers.suggest_tickers_for_isin] {isin}: error {e!r}")
        return []

    quotes = data.get("quotes", []) or []
    suggestions: List[Dict[str, str]] = []
    for q in quotes:
        suggestions.append(
            {
                "symbol": q.get("symbol", ""),
                "shortname": q.get("shortname", ""),
                "exchange": q.get("exchange", ""),
                "currency": q.get("currency", ""),
            }
        )
    print(f"[tickers.suggest_tickers_for_isin] {isin}: {len(suggestions)} suggestions")
    return suggestions


def best_ticker_suggestion(isin: str) -> str:
    """
    Devuelve el símbolo sugerido principal para un ISIN (o "" si no hay sugerencias).
    """
    suggestions = suggest_tickers_for_isin(isin, max_results=1)
    if not suggestions:
        print(f"[tickers.best_ticker_suggestion] {isin}: no suggestions")
        return ""
    symbol = suggestions[0].get("symbol", "") or ""
    print(f"[tickers.best_ticker_suggestion] {isin}: picked {symbol!r}")
    return symbol

