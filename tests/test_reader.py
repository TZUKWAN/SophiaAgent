"""Tests for PaperReader."""

import json
import os
from unittest.mock import MagicMock

import pytest

from sophia.research.reader import PaperReader, _extract_by_keywords


class TestExtractByKeywords:
    def test_extracts_research_question(self):
        text = "本研究旨在探讨社会资本对居民幸福感的影响机制。"
        result = _extract_by_keywords(text)
        assert len(result["research_question"]) > 0

    def test_extracts_methods(self):
        text = "We used a mixed-method approach combining surveys and interviews."
        result = _extract_by_keywords(text)
        assert len(result["methods"]) > 0

    def test_extracts_sample_size(self):
        text = "A total of 1250 participants were recruited from three cities."
        result = _extract_by_keywords(text)
        assert result["sample_size"] == "1250"

    def test_empty_text(self):
        result = _extract_by_keywords("")
        assert result["sample_size"] is None


class TestPaperReaderExtractKeyElements:
    def test_extract_from_text_no_provider(self):
        reader = PaperReader()
        text = (
            "研究问题：社会资本如何影响居民幸福感？\n"
            "方法：问卷调查与深度访谈\n"
            "数据来源：三个城市的1250名居民\n"
            "主要发现：社会资本显著正向影响幸福感\n"
            "局限：横截面数据，无法推断因果\n"
            "理论框架：基于社会资本理论"
        )
        result = reader.extract_key_elements(text)
        assert isinstance(result, dict)
        assert "research_question" in result
        assert "methods" in result
        assert "main_findings" in result

    def test_extract_empty_text(self):
        reader = PaperReader()
        result = reader.extract_key_elements("")
        assert result["research_question"] == []
        assert result["sample_size"] is None

    def test_extract_with_llm_provider(self):
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "research_question": ["Does X affect Y?"],
            "core_arguments": ["X positively affects Y"],
            "methods": ["Regression analysis"],
            "data_sources": ["Panel data"],
            "main_findings": ["Significant positive effect"],
            "limitations": ["Endogeneity concerns"],
            "theoretical_framework": ["Social capital theory"],
            "sample_size": "5000",
        })
        mock_provider.chat.return_value = mock_response

        reader = PaperReader(provider=mock_provider)
        result = reader.extract_key_elements("Some paper text here.")
        assert result["research_question"] == ["Does X affect Y?"]
        assert result["sample_size"] == "5000"
        mock_provider.chat.assert_called_once()

    def test_llm_fallback_on_failure(self):
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = Exception("LLM error")

        reader = PaperReader(provider=mock_provider)
        text = "研究问题：A对B的影响。方法：回归分析。样本：1000人。"
        result = reader.extract_key_elements(text)
        assert isinstance(result, dict)
        assert "methods" in result


class TestPaperReaderExtractAnnotations:
    def test_missing_pdf(self):
        reader = PaperReader()
        result = reader.extract_annotations("/nonexistent/file.pdf")
        assert "error" in result

    def test_pymupdf_not_installed(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError("No module named fitz")))
        reader = PaperReader()
        # Create a dummy file
        result = reader.extract_annotations("dummy.pdf")
        # The import will fail inside the method
        assert "error" in result


class TestPaperReaderComparePapers:
    def test_compare_two_papers(self):
        reader = PaperReader()
        elements_list = [
            {
                "research_question": ["Does social capital affect happiness?"],
                "theoretical_framework": ["Social capital theory"],
                "methods": ["Survey"],
                "data_sources": ["Urban residents"],
                "sample_size": "1000",
                "main_findings": ["Positive effect"],
                "limitations": ["Cross-sectional"],
            },
            {
                "research_question": ["How does trust influence well-being?"],
                "theoretical_framework": ["Social capital theory"],
                "methods": ["Experiment"],
                "data_sources": ["Students"],
                "sample_size": "500",
                "main_findings": ["Mixed results"],
                "limitations": ["Small sample"],
            },
        ]
        result = reader.compare_papers(elements_list)
        assert result["paper_count"] == 2
        assert "matrix" in result
        assert "consensus" in result
        assert "controversies" in result
        assert "theoretical_framework" in result["matrix"]

    def test_compare_empty_list(self):
        reader = PaperReader()
        result = reader.compare_papers([])
        assert "error" in result

    def test_compare_single_paper(self):
        reader = PaperReader()
        result = reader.compare_papers([{"research_question": ["Q1"]}])
        assert result["paper_count"] == 1
