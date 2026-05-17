"""Data collection tools for SophiaAgent.

Five tools:
- data_macro: Macroeconomic panel data (World Bank / FRED)
- data_china_finance: Chinese financial data (akshare)
- data_scrape: General-purpose web scraping (Playwright)
- data_scrape_batch: Batch URL scraping
- data_news: News article collection (GDELT + trafilatura)

All tools store results in ResultStore for downstream research tool chaining.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from sophia.tools.data_collection_helpers import (
    HAS_TRAFILATURA,
    HAS_WB,
    extract_article_text,
    gdelt_search,
    resolve_country,
    wb_indicator_lookup,
    wb_list_indicators,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Import guards for optional packages
# ------------------------------------------------------------------

try:
    import wbgapi as wb
except ImportError:
    wb = None

try:
    import pandas_datareader as pdr
except ImportError:
    pdr = None

try:
    import akshare as ak
except ImportError:
    ak = None


# ------------------------------------------------------------------
# Helper: store and return JSON
# ------------------------------------------------------------------

def _store_and_return(store, data, tool, args, shape_info=None):
    """Store data in ResultStore and return standard JSON response."""
    if isinstance(data, pd.DataFrame):
        kind = "dataframe"
        info = {"shape": list(data.shape), "columns": list(data.columns)}
    elif isinstance(data, dict):
        kind = "dict"
        info = {"keys": list(data.keys())}
    elif isinstance(data, str):
        kind = "text"
        info = {"length": len(data)}
    else:
        kind = "dict"
        info = {}

    rid = store.store(data, kind=kind, tool=tool, params=args)
    result = {
        "action": "data_collected",
        "tool": tool,
        "result_id": rid,
        "source": args.get("source", ""),
        "count": data.shape[0] if isinstance(data, pd.DataFrame) else 1,
    }
    if shape_info:
        # Use data_action key to avoid overriding the top-level "action"
        if "action" in shape_info:
            shape_info["data_action"] = shape_info.pop("action")
        result.update(shape_info)
    result.update(info)
    return json.dumps(result, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool 1: data_macro — Macroeconomic panel data
# ------------------------------------------------------------------

def data_macro(args: Dict[str, Any], store, guard) -> str:
    """Fetch macroeconomic panel data from World Bank or FRED.

    Args:
        source: "world_bank" | "fred"
        indicators: list of indicator codes or aliases (e.g. ["人均GDP", "SP.POP.TOTL"])
        countries: list of ISO codes or names, or "all"
        start_year: int (default 2000)
        end_year: int (default 2024)
    """
    source = args.get("source", "world_bank")
    raw_indicators = args.get("indicators", [])
    raw_countries = args.get("countries", "all")
    start_year = int(args.get("start_year", 2000))
    end_year = int(args.get("end_year", 2024))

    # Resolve indicator aliases
    indicators = []
    for ind in raw_indicators:
        resolved = wb_indicator_lookup(ind)
        if resolved:
            indicators.append(resolved)
        else:
            indicators.append(ind)

    if not indicators:
        return json.dumps({
            "error": "No valid indicators specified.",
            "available": wb_list_indicators(),
        }, ensure_ascii=False)

    # Resolve country codes
    if isinstance(raw_countries, str) and raw_countries.lower() == "all":
        countries = "all"
    elif isinstance(raw_countries, list):
        countries = [resolve_country(c) for c in raw_countries]
    else:
        countries = [resolve_country(str(raw_countries))]

    try:
        if source == "world_bank":
            if not HAS_WB or wb is None:
                return json.dumps({"error": "wbgapi not installed. Run: pip install wbgapi"})
            df = wb.data.DataFrame(indicators, economy=countries,
                                   time=range(start_year, end_year + 1),
                                   skipBlanks=True, labels=True)
            if df.empty:
                return json.dumps({"error": "No data returned from World Bank."})
            df = df.reset_index()
            return _store_and_return(store, df, "data_macro", args,
                                     {"source": "world_bank"})

        elif source == "fred":
            if pdr is None:
                return json.dumps({"error": "pandas_datareader not installed. Run: pip install pandas-datareader"})
            series_id = indicators[0]
            df = pdr.get_data_fred(series_id, start=f"{start_year}-01-01",
                                    end=f"{end_year}-12-31")
            if df.empty:
                return json.dumps({"error": f"No data returned for FRED series '{series_id}'."})
            df = df.reset_index()
            df.columns = ["date", "value"]
            return _store_and_return(store, df, "data_macro", args,
                                     {"source": "fred"})

        else:
            return json.dumps({"error": f"Unknown source '{source}'. Use 'world_bank' or 'fred'."})

    except Exception as e:
        logger.exception("data_macro failed")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool 2: data_china_finance — Chinese financial data via akshare
# ------------------------------------------------------------------

def data_china_finance(args: Dict[str, Any], store, guard) -> str:
    """Fetch Chinese financial/economic data via akshare.

    Args:
        action: "stock_hist" | "macro_gdp" | "macro_cpi" | "macro_pmi" | "financial"
        symbol: Stock code (e.g. "000001")
        start_date: YYYYMMDD (default "20200101")
        end_date: YYYYMMDD (default "20241231")
        period: "daily" | "weekly" | "monthly"
    """
    if ak is None:
        return json.dumps({"error": "akshare not installed. Run: pip install akshare"})

    action = args.get("action", "stock_hist")
    symbol = args.get("symbol", "000001")
    start_date = args.get("start_date", "20200101")
    end_date = args.get("end_date", "20241231")
    period = args.get("period", "daily")

    try:
        if action == "stock_hist":
            df = ak.stock_zh_a_hist(symbol=symbol, period=period,
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")
        elif action == "macro_gdp":
            df = ak.macro_china_gdp()
        elif action == "macro_cpi":
            df = ak.macro_china_cpi()
        elif action == "macro_pmi":
            df = ak.macro_china_pmi()
        elif action == "financial":
            symbol_ak = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
            try:
                df = ak.stock_financial_analysis_indicator(symbol=symbol_ak)
            except Exception:
                df = ak.stock_financial_report_sina(stock=f"sh{symbol}", symbol="资产负债表")
        elif action == "index":
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        else:
            return json.dumps({"error": f"Unknown action '{action}'."})

        if df is None or df.empty:
            return json.dumps({"error": f"No data returned for action '{action}'."})

        return _store_and_return(store, df, "data_china_finance", args,
                                 {"source": "akshare", "action": action})

    except Exception as e:
        logger.exception("data_china_finance failed")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool 3: data_scrape — General-purpose web scraping
# ------------------------------------------------------------------

def data_scrape(args: Dict[str, Any], store, guard) -> str:
    """Scrape data from a web page.

    Args:
        url: Target URL
        extract_type: "text" | "tables" | "links"
        selector: CSS selector (optional, defaults to "body" or "table")
    """
    url = args.get("url", "")
    if not url:
        return json.dumps({"error": "url is required"}, ensure_ascii=False)

    extract_type = args.get("extract_type", "text")
    selector = args.get("selector", "")

    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (Academic Research)"})
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch URL: {e}"}, ensure_ascii=False)

    try:
        if extract_type == "text":
            if HAS_TRAFILATURA:
                text = trafilatura.extract(html)
                if not text:
                    text = _basic_strip_html(html)
            else:
                text = _basic_strip_html(html)

            rid = store.store({"url": url, "text": text}, kind="text",
                              tool="data_scrape", params=args)
            return json.dumps({
                "action": "data_collected",
                "tool": "data_scrape",
                "result_id": rid,
                "extract_type": "text",
                "chars": len(text),
                "source": url,
            }, ensure_ascii=False)

        elif extract_type == "tables":
            import io
            tables = pd.read_html(io.StringIO(html))
            if not tables:
                return json.dumps({"error": "No tables found on page."})

            results = []
            for i, df in enumerate(tables):
                rid = store.store(df, kind="dataframe", tool="data_scrape",
                                  params={**args, "table_index": i})
                results.append({
                    "result_id": rid,
                    "table_index": i,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                })

            return json.dumps({
                "action": "data_collected",
                "tool": "data_scrape",
                "tables": results,
                "source": url,
            }, ensure_ascii=False)

        elif extract_type == "links":
            import re
            links = re.findall(r'href=["\']([^"\']+)["\']', html)
            links = list(dict.fromkeys(links))[:500]
            df = pd.DataFrame({"url": links})
            return _store_and_return(store, df, "data_scrape", args,
                                     {"extract_type": "links", "source": url})

        else:
            return json.dumps({"error": f"Unknown extract_type '{extract_type}'"})

    except Exception as e:
        logger.exception("data_scrape extraction failed")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool 4: data_scrape_batch — Batch URL scraping
# ------------------------------------------------------------------

def data_scrape_batch(args: Dict[str, Any], store, guard) -> str:
    """Scrape multiple URLs in batch.

    Args:
        urls: List of URLs to scrape
        extract_type: "text" (default) | "tables"
    """
    urls = args.get("urls", [])
    if not urls:
        return json.dumps({"error": "urls list is required"}, ensure_ascii=False)
    if len(urls) > 50:
        return json.dumps({"error": "Maximum 50 URLs per batch request."}, ensure_ascii=False)

    extract_type = args.get("extract_type", "text")
    results = []

    for i, url in enumerate(urls):
        single_result = data_scrape(
            {"url": url, "extract_type": extract_type}, store, guard
        )
        parsed = json.loads(single_result)
        if "error" not in parsed:
            results.append(parsed)
        else:
            results.append({"url": url, "error": parsed["error"]})

    return json.dumps({
        "action": "batch_collected",
        "tool": "data_scrape_batch",
        "total": len(urls),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results,
    }, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool 5: data_news — News article collection
# ------------------------------------------------------------------

def data_news(args: Dict[str, Any], store, guard) -> str:
    """Collect news articles from GDELT and extract full text.

    Args:
        keyword: Search keyword
        start_date: YYYY-MM-DD (default "2024-01-01")
        end_date: YYYY-MM-DD (default "2024-12-31")
        max_articles: int (default 20, max 250)
        extract_body: bool (default true) — extract full article text
        language: "eng" | "chi"
    """
    keyword = args.get("keyword", "")
    if not keyword:
        return json.dumps({"error": "keyword is required"}, ensure_ascii=False)

    start_date = args.get("start_date", "2024-01-01")
    end_date = args.get("end_date", "2024-12-31")
    max_articles = min(int(args.get("max_articles", 20)), 250)
    extract_body = args.get("extract_body", True)
    language = args.get("language", "eng")

    # Step 1: Search GDELT
    try:
        articles = gdelt_search(keyword, start_date, end_date, max_articles, language)
    except RuntimeError as e:
        return json.dumps({
            "error": str(e),
            "tool": "data_news",
            "keyword": keyword,
        }, ensure_ascii=False)
    if not articles:
        return json.dumps({
            "action": "data_collected",
            "tool": "data_news",
            "keyword": keyword,
            "count": 0,
            "error": "No articles found from GDELT.",
        }, ensure_ascii=False)

    # Step 2: Extract body text
    if extract_body:
        for art in articles:
            if art.get("url"):
                art["body"] = extract_article_text(art["url"])
            else:
                art["body"] = ""

    # Step 3: Store as DataFrame
    df = pd.DataFrame(articles)
    return _store_and_return(store, df, "data_news", args,
                             {"source": "gdelt", "keyword": keyword})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

import re as _re

try:
    import trafilatura
except ImportError:
    trafilatura = None


def _basic_strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _re.sub(r'<script[^>]*>.*?</script>', '', html, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r'<[^>]+>', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text[:20000]


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_data_collection_tools(registry, store, guard):
    """Register all data collection tools."""
    from functools import partial

    registry.register(
        name="data_macro",
        description=(
            "Fetch macroeconomic panel data from World Bank or FRED. "
            "World Bank: 200+ countries, 1600+ indicators (GDP, CPI, trade, population, etc.), 1960-present. "
            "FRED: 800,000+ US economic time series. "
            "Supports Chinese/English indicator aliases (e.g. '人均GDP' → NY.GDP.PCAP.CD). "
            "Results stored in ResultStore for chaining with research tools."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "default": "world_bank",
                    "enum": ["world_bank", "fred"],
                    "description": "Data source",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Indicator codes or aliases (e.g. ['NY.GDP.PCAP.CD', '人均GDP'])",
                },
                "countries": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "default": "all",
                    "description": "Country codes or names, or 'all'",
                },
                "start_year": {"type": "integer", "default": 2000},
                "end_year": {"type": "integer", "default": 2024},
            },
            "required": ["indicators"],
        },
        handler=partial(data_macro, store=store, guard=guard),
    )

    registry.register(
        name="data_china_finance",
        description=(
            "Fetch Chinese financial and economic data via akshare. "
            "Actions: stock_hist (A-share daily/weekly/monthly), "
            "macro_gdp, macro_cpi, macro_pmi, financial (financial statements), "
            "index (stock indices). Zero registration required."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "default": "stock_hist",
                    "enum": ["stock_hist", "macro_gdp", "macro_cpi", "macro_pmi", "financial", "index"],
                    "description": "Data type to fetch",
                },
                "symbol": {
                    "type": "string",
                    "default": "000001",
                    "description": "Stock code (e.g. '000001' for Ping An Bank)",
                },
                "start_date": {
                    "type": "string",
                    "default": "20200101",
                    "description": "Start date YYYYMMDD",
                },
                "end_date": {
                    "type": "string",
                    "default": "20241231",
                    "description": "End date YYYYMMDD",
                },
                "period": {
                    "type": "string",
                    "default": "daily",
                    "enum": ["daily", "weekly", "monthly"],
                },
            },
            "required": [],
        },
        handler=partial(data_china_finance, store=store, guard=guard),
    )

    registry.register(
        name="data_scrape",
        description=(
            "Scrape data from a web page. "
            "extract_type 'text': extract clean body text (via trafilatura). "
            "extract_type 'tables': extract HTML tables as DataFrames. "
            "extract_type 'links': extract all links. "
            "Results stored in ResultStore."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL"},
                "extract_type": {
                    "type": "string",
                    "default": "text",
                    "enum": ["text", "tables", "links"],
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector (optional)",
                },
            },
            "required": ["url"],
        },
        handler=partial(data_scrape, store=store, guard=guard),
    )

    registry.register(
        name="data_scrape_batch",
        description="Scrape multiple URLs in batch. Returns results for each URL.",
        parameters={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to scrape",
                },
                "extract_type": {
                    "type": "string",
                    "default": "text",
                    "enum": ["text", "tables", "links"],
                },
            },
            "required": ["urls"],
        },
        handler=partial(data_scrape_batch, store=store, guard=guard),
    )

    registry.register(
        name="data_news",
        description=(
            "Collect news articles from GDELT global event database. "
            "Searches by keyword and date range, optionally extracts full article text. "
            "Supports English and Chinese news. Free, no API key required."
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Search keyword"},
                "start_date": {
                    "type": "string",
                    "default": "2024-01-01",
                    "description": "YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "default": "2024-12-31",
                    "description": "YYYY-MM-DD",
                },
                "max_articles": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max articles (1-250)",
                },
                "extract_body": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract full article body text",
                },
                "language": {
                    "type": "string",
                    "default": "eng",
                    "enum": ["eng", "chi"],
                },
            },
            "required": ["keyword"],
        },
        handler=partial(data_news, store=store, guard=guard),
    )
