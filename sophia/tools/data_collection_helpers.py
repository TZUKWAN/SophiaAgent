"""Helper functions for data collection tools.

- GDELT news event search
- Article body extraction (trafilatura / httpx fallback)
- World Bank indicator lookup (Chinese/English keyword → indicator code)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Import guards
# ------------------------------------------------------------------

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    import wbgapi as wb
    HAS_WB = True
except ImportError:
    HAS_WB = False


# ------------------------------------------------------------------
# World Bank indicator lookup
# ------------------------------------------------------------------

WB_INDICATOR_ALIASES: Dict[str, str] = {
    "人均GDP".lower(): "NY.GDP.PCAP.CD",
    "GDP per capita".lower(): "NY.GDP.PCAP.CD",
    "GDP".lower(): "NY.GDP.MKTP.CD",
    "人口".lower(): "SP.POP.TOTL",
    "population".lower(): "SP.POP.TOTL",
    "CPI".lower(): "FP.CPI.TOTL.ZG",
    "通货膨胀".lower(): "FP.CPI.TOTL.ZG",
    "inflation".lower(): "FP.CPI.TOTL.ZG",
    "失业率".lower(): "SL.UEM.TOTL.ZS",
    "unemployment".lower(): "SL.UEM.TOTL.ZS",
    "教育支出".lower(): "SE.XPD.TOTL.GD.ZS",
    "education spending".lower(): "SE.XPD.TOTL.GD.ZS",
    "贸易".lower(): "NE.TRD.GNFS.ZS",
    "trade".lower(): "NE.TRD.GNFS.ZS",
    "FDI".lower(): "BX.KLT.DINV.WD.GD.ZS",
    "外商直接投资".lower(): "BX.KLT.DINV.WD.GD.ZS",
    "CO2".lower(): "EN.ATM.CO2E.PC",
    "碳排放".lower(): "EN.ATM.CO2E.PC",
    "预期寿命".lower(): "SP.DYN.LE00.IN",
    "life expectancy".lower(): "SP.DYN.LE00.IN",
    "研发支出".lower(): "GB.XPD.RSDV.GD.ZS",
    "R&D".lower(): "GB.XPD.RSDV.GD.ZS",
    "internet".lower(): "IT.NET.USER.ZS",
    "互联网".lower(): "IT.NET.USER.ZS",
    "Gini".lower(): "SI.POV.GINI",
    "基尼系数".lower(): "SI.POV.GINI",
    "出生率".lower(): "SP.DYN.CBRT.IN",
    "死亡率".lower(): "SP.DYN.CDRT.IN",
    "urbanization".lower(): "SP.URB.TOTL.IN.ZS",
    "城镇化率".lower(): "SP.URB.TOTL.IN.ZS",
    "健康支出".lower(): "SH.XPD.CHEX.GD.ZS",
    "health spending".lower(): "SH.XPD.CHEX.GD.ZS",
}


def wb_indicator_lookup(query: str) -> Optional[str]:
    """Look up a World Bank indicator code from a keyword.

    Accepts indicator codes directly (e.g. 'NY.GDP.PCAP.CD') or
    descriptive aliases in Chinese/English.
    """
    q = query.strip()
    # Direct code match (e.g. NY.GDP.PCAP.CD, SP.POP.TOTL)
    if re.match(r'^[A-Z]{2,3}\.[A-Z0-9]+\.[A-Z0-9.]+$', q):
        return q
    # Alias lookup
    return WB_INDICATOR_ALIASES.get(q.lower())


def wb_list_indicators() -> List[Dict[str, str]]:
    """Return a list of commonly used World Bank indicators."""
    result = []
    seen = set()
    for alias, code in WB_INDICATOR_ALIASES.items():
        if code not in seen:
            seen.add(code)
            result.append({"alias": alias, "code": code})
    return result


# ------------------------------------------------------------------
# GDELT news search
# ------------------------------------------------------------------

GDELT_SEARCH_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def gdelt_search(
    keyword: str,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 50,
    language: str = "eng",
) -> List[Dict[str, str]]:
    """Search GDELT for news articles matching a keyword.

    Args:
        keyword: Search query.
        start_date: YYYY-MM-DD format.
        end_date: YYYY-MM-DD format.
        max_results: Max articles to return (capped at 250).
        language: 'eng' or 'chi'.

    Returns:
        List of dicts with keys: title, url, date, source.
    """
    max_results = min(max_results, 250)
    # GDELT date format: YYYYMMDDHHMMSS
    start_fmt = start_date.replace("-", "") + "000000"
    end_fmt = end_date.replace("-", "") + "235959"

    query = f'"{keyword}"'
    if language == "chi":
        query += " sourcelang:chinese"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_results),
        "startdatetime": start_fmt,
        "enddatetime": end_fmt,
        "format": "json",
    }

    try:
        resp = httpx.get(GDELT_SEARCH_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"GDELT API returned {e.response.status_code}: {e.response.text[:200]}") from e
    except Exception as e:
        raise RuntimeError(f"GDELT search failed: {e}") from e

    articles = data.get("articles", [])
    results = []
    for art in articles[:max_results]:
        results.append({
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "date": art.get("seendate", "")[:8],
            "source": art.get("source", ""),
        })
    return results


# ------------------------------------------------------------------
# Article body extraction
# ------------------------------------------------------------------

def extract_article_text(url: str) -> str:
    """Extract clean article body text from a URL.

    Tries trafilatura first (best quality), falls back to httpx + basic HTML strip.
    """
    if HAS_TRAFILATURA:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    return text
        except Exception as e:
            logger.debug("trafilatura failed for %s: %s", url, e)

    # Fallback: httpx + basic tag stripping
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        # Strip tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:10000]
    except Exception as e:
        logger.debug("Fallback extraction failed for %s: %s", url, e)
        return ""


# ------------------------------------------------------------------
# Country code helpers
# ------------------------------------------------------------------

COUNTRY_CODES = {
    # Chinese names
    "中国": "CHN", "美国": "USA", "日本": "JPN", "德国": "DEU", "英国": "GBR",
    "法国": "FRA", "印度": "IND", "巴西": "BRA", "俄罗斯": "RUS", "韩国": "KOR",
    "意大利": "ITA", "加拿大": "CAN", "澳大利亚": "AUS", "西班牙": "ESP",
    "墨西哥": "MEX", "印度尼西亚": "IDN", "荷兰": "NLD", "沙特阿拉伯": "SAU",
    "土耳其": "TUR", "瑞士": "CHE", "阿根廷": "ARG", "南非": "ZAF",
    "瑞典": "SWE", "波兰": "POL", "挪威": "NOR", "比利时": "BEL",
    "泰国": "THA", "埃及": "EGY", "新加坡": "SGP", "马来西亚": "MYS",
    "菲律宾": "PHL", "越南": "VNM", "以色列": "ISR", "新西兰": "NZL",
    "葡萄牙": "PRT", "爱尔兰": "IRL", "丹麦": "DNK", "芬兰": "FIN",
    "奥地利": "AUT", "希腊": "GRC", "智利": "CHL", "哥伦比亚": "COL",
    "尼日利亚": "NGA", "巴基斯坦": "PAK", "孟加拉国": "BGD",
    # English names
    "china": "CHN", "usa": "USA", "united states": "USA", "japan": "JPN",
    "germany": "DEU", "uk": "GBR", "united kingdom": "GBR", "france": "FRA",
    "india": "IND", "brazil": "BRA", "russia": "RUS", "korea": "KOR",
    "south korea": "KOR", "italy": "ITA", "canada": "CAN",
    "australia": "AUS", "spain": "ESP", "mexico": "MEX",
    "indonesia": "IDN", "netherlands": "NLD", "saudi arabia": "SAU",
    "turkey": "TUR", "switzerland": "CHE", "argentina": "ARG",
    "south africa": "ZAF", "sweden": "SWE", "poland": "POL",
    "norway": "NOR", "belgium": "BEL", "thailand": "THA",
    "egypt": "EGY", "singapore": "SGP", "malaysia": "MYS",
    "philippines": "PHL", "vietnam": "VNM", "israel": "ISR",
    "new zealand": "NZL", "portugal": "PRT", "ireland": "IRL",
    "denmark": "DNK", "finland": "FIN", "austria": "AUT",
    "greece": "GRC", "chile": "CHL", "colombia": "COL",
    "nigeria": "NGA", "pakistan": "PAK", "bangladesh": "BGD",
}


def resolve_country(code: str) -> str:
    """Resolve a country name/code to ISO 3-letter code."""
    code = code.strip()
    if len(code) == 3 and code.isalpha():
        return code.upper()
    return COUNTRY_CODES.get(code.lower(), code.upper())
