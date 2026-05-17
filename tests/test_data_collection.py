"""Tests for data collection tools."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeResultStore:
    """Minimal ResultStore mock that tracks stored items."""

    def __init__(self):
        self._items = {}
        self._counter = 0

    def store(self, data, kind="dataframe", tool="", params=None, parents=None):
        self._counter += 1
        rid = f"res_{self._counter:08x}"
        self._items[rid] = {"data": data, "kind": kind, "tool": tool}
        return rid

    def get(self, rid):
        return self._items.get(rid, {}).get("data")

    def get_dataframe(self, rid):
        data = self.get(rid)
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)


class FakeGuard:
    def resolve_read(self, path):
        return path


@pytest.fixture
def store():
    return FakeResultStore()


@pytest.fixture
def guard():
    return FakeGuard()


# ---------------------------------------------------------------------------
# World Bank indicator lookup
# ---------------------------------------------------------------------------

def test_wb_indicator_lookup_direct_code():
    from sophia.tools.data_collection_helpers import wb_indicator_lookup
    assert wb_indicator_lookup("NY.GDP.PCAP.CD") == "NY.GDP.PCAP.CD"


def test_wb_indicator_lookup_chinese_alias():
    from sophia.tools.data_collection_helpers import wb_indicator_lookup
    assert wb_indicator_lookup("人均GDP") == "NY.GDP.PCAP.CD"


def test_wb_indicator_lookup_english_alias():
    from sophia.tools.data_collection_helpers import wb_indicator_lookup
    assert wb_indicator_lookup("population") == "SP.POP.TOTL"


def test_wb_indicator_lookup_unknown():
    from sophia.tools.data_collection_helpers import wb_indicator_lookup
    assert wb_indicator_lookup("xyz_no_exist") is None


def test_wb_list_indicators():
    from sophia.tools.data_collection_helpers import wb_list_indicators
    result = wb_list_indicators()
    assert len(result) > 5
    codes = [r["code"] for r in result]
    assert "NY.GDP.PCAP.CD" in codes


# ---------------------------------------------------------------------------
# Country code resolution
# ---------------------------------------------------------------------------

def test_resolve_country_chinese():
    from sophia.tools.data_collection_helpers import resolve_country
    assert resolve_country("中国") == "CHN"
    assert resolve_country("美国") == "USA"


def test_resolve_country_iso():
    from sophia.tools.data_collection_helpers import resolve_country
    assert resolve_country("CHN") == "CHN"
    assert resolve_country("usa") == "USA"


# ---------------------------------------------------------------------------
# data_macro — World Bank mock
# ---------------------------------------------------------------------------

def test_data_macro_world_bank(store, guard):
    from sophia.tools.data_collection import data_macro

    fake_df = pd.DataFrame({
        "economy": ["CHN", "CHN", "USA", "USA"],
        "series": ["NY.GDP.PCAP.CD", "SP.POP.TOTL", "NY.GDP.PCAP.CD", "SP.POP.TOTL"],
        "year": ["YR2020", "YR2020", "YR2020", "YR2020"],
        "value": [10435.0, 1411780000.0, 63528.0, 331449281.0],
    })

    with patch("sophia.tools.data_collection.wb") as mock_wb:
        mock_wb.data.DataFrame.return_value = fake_df
        result = json.loads(data_macro(
            {"source": "world_bank", "indicators": ["NY.GDP.PCAP.CD", "SP.POP.TOTL"],
             "countries": ["CHN", "USA"], "start_year": 2020, "end_year": 2020},
            store=store, guard=guard,
        ))

    assert result["action"] == "data_collected"
    assert result["tool"] == "data_macro"
    assert "result_id" in result
    assert result["source"] == "world_bank"
    assert result["count"] == 4


def test_data_macro_no_indicators(store, guard):
    from sophia.tools.data_collection import data_macro
    result = json.loads(data_macro({"indicators": []}, store=store, guard=guard))
    assert "error" in result


def test_data_macro_unknown_source(store, guard):
    from sophia.tools.data_collection import data_macro
    result = json.loads(data_macro(
        {"source": "nonexistent", "indicators": ["X"]}, store=store, guard=guard))
    assert "error" in result


# ---------------------------------------------------------------------------
# data_china_finance — akshare mock
# ---------------------------------------------------------------------------

def test_data_china_finance_stock_hist(store, guard):
    from sophia.tools.data_collection import data_china_finance

    fake_df = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5],
        "收盘": [10.2, 10.3],
        "最高": [10.5, 10.8],
        "最低": [9.8, 10.1],
        "成交量": [1000000, 1200000],
    })

    with patch("sophia.tools.data_collection.ak") as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = fake_df
        result = json.loads(data_china_finance(
            {"action": "stock_hist", "symbol": "000001",
             "start_date": "20240101", "end_date": "20240103"},
            store=store, guard=guard,
        ))

    assert result["action"] == "data_collected"
    assert result["source"] == "akshare"
    assert result["count"] == 2


def test_data_china_finance_macro(store, guard):
    from sophia.tools.data_collection import data_china_finance

    fake_df = pd.DataFrame({"年份": [2020, 2021, 2022], "GDP": [101.6, 114.4, 121.0]})

    with patch("sophia.tools.data_collection.ak") as mock_ak:
        mock_ak.macro_china_gdp.return_value = fake_df
        result = json.loads(data_china_finance(
            {"action": "macro_gdp"}, store=store, guard=guard,
        ))

    assert result["count"] == 3


# ---------------------------------------------------------------------------
# data_scrape — mock HTTP
# ---------------------------------------------------------------------------

def test_data_scrape_text(store, guard):
    from sophia.tools.data_collection import data_scrape

    fake_html = "<html><body><p>Hello academic world. This is a test article.</p></body></html>"

    with patch("sophia.tools.data_collection.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = json.loads(data_scrape(
            {"url": "https://example.com/test", "extract_type": "text"},
            store=store, guard=guard,
        ))

    assert result["action"] == "data_collected"
    assert result["extract_type"] == "text"
    assert result["chars"] > 0


def test_data_scrape_tables(store, guard):
    from sophia.tools.data_collection import data_scrape

    fake_html = """
    <html><body>
    <table><tr><th>Name</th><th>Value</th></tr>
    <tr><td>GDP</td><td>10.5</td></tr></table>
    </body></html>
    """

    with patch("sophia.tools.data_collection.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = json.loads(data_scrape(
            {"url": "https://example.com/table", "extract_type": "tables"},
            store=store, guard=guard,
        ))

    assert result["action"] == "data_collected"
    assert len(result["tables"]) == 1


def test_data_scrape_no_url(store, guard):
    from sophia.tools.data_collection import data_scrape
    result = json.loads(data_scrape({}, store=store, guard=guard))
    assert "error" in result


# ---------------------------------------------------------------------------
# data_news — GDELT mock
# ---------------------------------------------------------------------------

def test_data_news(store, guard):
    from sophia.tools.data_collection import data_news

    fake_articles = [
        {"title": "AI breakthrough", "url": "https://example.com/ai",
         "date": "20240115", "source": "TestNews"},
        {"title": "Climate summit", "url": "https://example.com/climate",
         "date": "20240116", "source": "GreenNews"},
    ]

    with patch("sophia.tools.data_collection.gdelt_search", return_value=fake_articles):
        with patch("sophia.tools.data_collection.extract_article_text",
                   return_value="Full article text here."):
            result = json.loads(data_news(
                {"keyword": "climate", "start_date": "2024-01-01",
                 "end_date": "2024-01-31", "max_articles": 10},
                store=store, guard=guard,
            ))

    assert result["action"] == "data_collected"
    assert result["count"] == 2
    assert result["keyword"] == "climate"


def test_data_news_empty(store, guard):
    from sophia.tools.data_collection import data_news

    with patch("sophia.tools.data_collection.gdelt_search", return_value=[]):
        result = json.loads(data_news(
            {"keyword": "xyz_nonexistent_topic"},
            store=store, guard=guard,
        ))

    assert result["count"] == 0


# ---------------------------------------------------------------------------
# data_scrape_batch
# ---------------------------------------------------------------------------

def test_data_scrape_batch(store, guard):
    from sophia.tools.data_collection import data_scrape_batch

    fake_html = "<html><body><p>Test content.</p></body></html>"

    with patch("sophia.tools.data_collection.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = json.loads(data_scrape_batch(
            {"urls": ["https://a.com", "https://b.com"], "extract_type": "text"},
            store=store, guard=guard,
        ))

    assert result["total"] == 2
    assert result["successful"] == 2


# ---------------------------------------------------------------------------
# ResultStore roundtrip
# ---------------------------------------------------------------------------

def test_collected_data_retrievable(store, guard):
    """Verify collected data can be retrieved by result_id."""
    from sophia.tools.data_collection import data_china_finance

    fake_df = pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]})

    with patch("sophia.tools.data_collection.ak") as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = fake_df
        result = json.loads(data_china_finance(
            {"action": "stock_hist", "symbol": "000001"}, store=store, guard=guard))

    rid = result["result_id"]
    retrieved = store.get_dataframe(rid)
    assert isinstance(retrieved, pd.DataFrame)
    assert list(retrieved.columns) == ["date", "close"]
