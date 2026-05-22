import pytest
from unittest.mock import MagicMock, patch
from sophia.tools.research import _search_cnki_playwright, literature_search

# Mock data for Playwright Elements
class MockElement:
    def __init__(self, text):
        self._text = text
    def inner_text(self):
        return self._text

class MockRow:
    def __init__(self, title, author, source, year, abstract):
        self.title = title
        self.author = author
        self.source = source
        self.year = year
        self.abstract = abstract
        
    def query_selector(self, selector):
        if selector == ".name a" and self.title is not None: return MockElement(self.title)
        if selector == ".author" and self.author is not None: return MockElement(self.author)
        if selector == ".source" and self.source is not None: return MockElement(self.source)
        if selector == ".year" and self.year is not None: return MockElement(self.year)
        if selector == ".abstract" and self.abstract is not None: return MockElement(self.abstract)
        return None

def test_cnki_search_no_playwright(monkeypatch):
    monkeypatch.setattr("sophia.tools.research.HAS_PLAYWRIGHT", False)
    results, err = _search_cnki_playwright("deep learning", 5)
    assert not results
    assert "playwright not installed" in err

@patch("sophia.tools.research.sync_playwright")
def test_cnki_search_exception(mock_sync, monkeypatch):
    monkeypatch.setattr("sophia.tools.research.HAS_PLAYWRIGHT", True)
    mock_pw = MagicMock()
    mock_sync.return_value.__enter__.return_value = mock_pw
    mock_pw.chromium.launch.side_effect = Exception("Browser init failure")
    
    results, err = _search_cnki_playwright("deep learning", 5)
    assert not results
    assert "Browser init failure" in err
    assert "cnki_playwright failed:" in err

@patch("sophia.tools.research.sync_playwright")
def test_cnki_search_success(mock_sync, monkeypatch):
    monkeypatch.setattr("sophia.tools.research.HAS_PLAYWRIGHT", True)
    mock_pw = MagicMock()
    mock_sync.return_value.__enter__.return_value = mock_pw
    
    mock_browser = MagicMock()
    mock_pw.chromium.launch.return_value = mock_browser
    
    mock_page = MagicMock()
    mock_browser.new_page.return_value = mock_page
    
    # Mock row data returned by Playwright query_selector_all
    mock_page.query_selector_all.return_value = [
        MockRow("Test Title 1", "John Doe", "Journal of Testing", "2023", "Abstract 1"),
        MockRow("Test Title 2", None, None, "InvalidYear", "Abstract 2"), # Test invalid year parse
        MockRow(None, "Author", "Source", "2024", "Abstract 3") # Test missing title row skipping
    ]
    
    results, err = _search_cnki_playwright("climate change", 5)
    
    assert err == ""
    assert len(results) == 2
    
    # First valid item
    assert results[0]["title"] == "Test Title 1"
    assert results[0]["authors"] == "John Doe"
    assert results[0]["journal"] == "Journal of Testing"
    assert results[0]["year"] == 2023
    assert results[0]["abstract"] == "Abstract 1"
    
    # Second valid item (Invalid year text)
    assert results[1]["title"] == "Test Title 2"
    assert results[1]["authors"] == ""
    assert results[1]["journal"] == ""
    assert results[1]["year"] is None
    
    # Verify interactions
    mock_page.goto.assert_called_with("https://kns.cnki.net/kns8s/defaultresult/index", timeout=30000)
    mock_page.fill.assert_called_with("input#txt_SearchText", "climate change")
    mock_page.click.assert_called_with("input.search-btn")
    mock_page.wait_for_selector.assert_called_with(".result-table-list", timeout=15000)

@patch("sophia.tools.research._search_cnki_playwright")
@patch("sophia.tools.research._search_semantic_scholar")
@patch("sophia.tools.research._search_arxiv")
@patch("sophia.tools.research._search_crossref")
def test_literature_search_integration(mock_crossref, mock_arxiv, mock_semantic, mock_cnki):
    # Dummy returns for all except CNKI, make them empty
    mock_crossref.return_value = ([], "")
    mock_arxiv.return_value = ([], "")
    mock_semantic.return_value = ([], "")
    
    # Setup our tested item properly returning info
    mock_cnki.return_value = ([
        {"title": "China Economy Meta", "source": "cnki"}
    ], "")
    
    args = {"query": "economy", "max_results": 2, "sources": ["cnki"]}
    result_str = literature_search(args)
    
    assert "China Economy Meta" in result_str
    mock_cnki.assert_called_once_with("economy", 2)
