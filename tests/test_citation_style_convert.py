"""Tests for citation_style_convert tool in citation.py."""

import json
import os

import pytest

from sophia.tools.citation import (
    _format_apa7,
    _format_chicago,
    _format_gbt7714,
    _format_mla,
    _parse_authors_apa,
    _parse_authors_chicago,
    _parse_authors_mla,
    _convert_citation_style,
    ref_add,
    _load_bib,
)


# ---------------------------------------------------------------------------
# Author parser tests
# ---------------------------------------------------------------------------

class TestAuthorParsers:
    def test_parse_authors_apa_single(self):
        result = _parse_authors_apa("John Smith")
        assert result == "Smith, J."

    def test_parse_authors_apa_two(self):
        result = _parse_authors_apa("John Smith, Jane Doe")
        assert result == "Smith, J., & Doe, J."

    def test_parse_authors_apa_three(self):
        result = _parse_authors_apa("John Smith, Jane Doe, Bob Brown")
        assert result == "Smith, J., Doe, J., & Brown, B."

    def test_parse_authors_apa_empty(self):
        assert _parse_authors_apa("") == ""

    def test_parse_authors_mla_single(self):
        result = _parse_authors_mla("John Smith")
        assert result == "Smith, John"

    def test_parse_authors_mla_two(self):
        result = _parse_authors_mla("John Smith, Jane Doe")
        assert result == "Smith, John, and Jane Doe"

    def test_parse_authors_mla_three(self):
        result = _parse_authors_mla("John Smith, Jane Doe, Bob Brown")
        assert result == "Smith, John, et al."

    def test_parse_authors_chicago_single(self):
        result = _parse_authors_chicago("John Smith")
        assert result == "Smith, John"

    def test_parse_authors_chicago_two(self):
        result = _parse_authors_chicago("John Smith, Jane Doe")
        assert result == "Smith, John and Doe, Jane"

    def test_parse_authors_chicago_three(self):
        result = _parse_authors_chicago("John Smith, Jane Doe, Bob Brown")
        assert result == "Smith, John, Doe, Jane, and Brown, Bob"


# ---------------------------------------------------------------------------
# Formatter tests for 8 literature types x 4 styles
# ---------------------------------------------------------------------------

class TestFormatters:
    @pytest.fixture
    def journal_entry(self):
        return {
            "_key": "Smith2024",
            "_type": "article",
            "_lit_type": "journal",
            "author": "John Smith, Jane Doe",
            "year": "2024",
            "title": "A Study of Social Capital",
            "journal": "Journal of Sociology",
            "volume": "45",
            "number": "3",
            "pages": "123-145",
            "doi": "10.1234/example",
        }

    @pytest.fixture
    def book_entry(self):
        return {
            "_key": "Putnam2000",
            "_type": "book",
            "_lit_type": "book",
            "author": "Robert Putnam",
            "year": "2000",
            "title": "Bowling Alone",
            "publisher": "Simon & Schuster",
            "edition": "1st",
        }

    @pytest.fixture
    def chapter_entry(self):
        return {
            "_key": "Granovetter1973",
            "_type": "incollection",
            "_lit_type": "chapter",
            "author": "Mark Granovetter",
            "year": "1973",
            "title": "The Strength of Weak Ties",
            "booktitle": "Social Networks",
            "editor": "Peter Marsden",
            "pages": "105-130",
            "publisher": "Academic Press",
        }

    @pytest.fixture
    def thesis_entry(self):
        return {
            "_key": "Zhang2020",
            "_type": "phdthesis",
            "_lit_type": "thesis",
            "author": "Wei Zhang",
            "year": "2020",
            "title": "Digital Transformation in Education",
            "publisher": "Peking University",
            "url": "https://example.com/thesis",
        }

    @pytest.fixture
    def web_entry(self):
        return {
            "_key": "OECD2023",
            "_type": "misc",
            "_lit_type": "web",
            "author": "OECD",
            "year": "2023",
            "title": "Education at a Glance",
            "publisher": "OECD Publishing",
            "url": "https://oecd.org/education",
        }

    @pytest.fixture
    def report_entry(self):
        return {
            "_key": "WorldBank2022",
            "_type": "techreport",
            "_lit_type": "report",
            "author": "World Bank",
            "year": "2022",
            "title": "World Development Report",
            "publisher": "World Bank Group",
            "url": "https://worldbank.org",
        }

    @pytest.fixture
    def conference_entry(self):
        return {
            "_key": "Chen2023",
            "_type": "inproceedings",
            "_lit_type": "conference",
            "author": "Li Chen",
            "year": "2023",
            "title": "AI in Education",
            "booktitle": "Proceedings of ICML",
            "pages": "45-52",
            "publisher": "ACM",
            "doi": "10.1234/icml",
        }

    def test_apa_journal(self, journal_entry):
        result = _format_apa7(journal_entry)
        assert "Smith, J., & Doe, J." in result
        assert "(2024)." in result
        assert "A Study of Social Capital." in result
        assert "*Journal of Sociology*" in result
        assert "*45*" in result
        assert "(3)" in result
        assert "123-145" in result
        assert "https://doi.org/10.1234/example" in result

    def test_apa_book(self, book_entry):
        result = _format_apa7(book_entry)
        assert "Putnam, R." in result
        assert "*Bowling Alone*." in result
        assert "Simon & Schuster" in result

    def test_apa_chapter(self, chapter_entry):
        result = _format_apa7(chapter_entry)
        assert "Granovetter, M." in result
        assert "The Strength of Weak Ties." in result
        assert "In Marsden, P. (Eds.)," in result
        assert "*Social Networks*" in result
        assert "pp. 105-130." in result

    def test_apa_thesis(self, thesis_entry):
        result = _format_apa7(thesis_entry)
        assert "Zhang, W." in result
        assert "*Digital Transformation in Education*" in result
        assert "Peking University" in result

    def test_apa_web(self, web_entry):
        result = _format_apa7(web_entry)
        assert "OECD" in result
        assert "Education at a Glance." in result
        assert "https://oecd.org/education" in result

    def test_apa_report(self, report_entry):
        result = _format_apa7(report_entry)
        assert "World Bank" in result
        assert "*World Development Report*." in result

    def test_apa_conference(self, conference_entry):
        result = _format_apa7(conference_entry)
        assert "Chen, L." in result
        assert "AI in Education." in result
        assert "In *Proceedings of ICML*" in result
        assert "pp. 45-52" in result

    def test_mla_journal(self, journal_entry):
        result = _format_mla(journal_entry)
        assert "Smith, John, and Jane Doe." in result
        assert '"A Study of Social Capital."' in result
        assert "*Journal of Sociology*" in result
        assert "vol. 45" in result
        assert "no. 3" in result

    def test_mla_book(self, book_entry):
        result = _format_mla(book_entry)
        assert "Putnam, Robert." in result
        assert "*Bowling Alone*." in result

    def test_mla_chapter(self, chapter_entry):
        result = _format_mla(chapter_entry)
        assert "Granovetter, Mark." in result
        assert '"The Strength of Weak Ties."' in result

    def test_chicago_journal(self, journal_entry):
        result = _format_chicago(journal_entry)
        assert "Smith, John and Doe, Jane." in result
        assert '"A Study of Social Capital."' in result
        assert "*Journal of Sociology*" in result
        assert "45" in result
        assert "no. 3" in result
        assert "(2024)" in result

    def test_chicago_book(self, book_entry):
        result = _format_chicago(book_entry)
        assert "Putnam, Robert." in result
        assert "*Bowling Alone*." in result

    def test_chicago_chapter(self, chapter_entry):
        result = _format_chicago(chapter_entry)
        assert "Granovetter, Mark." in result
        assert "In *Social Networks*," in result

    def test_gbt_journal(self, journal_entry):
        result = _format_gbt7714(journal_entry)
        assert "John Smith, Jane Doe" in result
        assert "2024." in result
        assert "A Study of Social Capital[J]." in result
        assert "Journal of Sociology" in result

    def test_gbt_book(self, book_entry):
        result = _format_gbt7714(book_entry)
        assert "Bowling Alone[M]." in result

    def test_gbt_thesis(self, thesis_entry):
        result = _format_gbt7714(thesis_entry)
        assert "Digital Transformation in Education[D]." in result

    def test_gbt_web(self, web_entry):
        result = _format_gbt7714(web_entry)
        assert "Education at a Glance[EB/OL]." in result

    def test_gbt_report(self, report_entry):
        result = _format_gbt7714(report_entry)
        assert "World Development Report[R]." in result

    def test_gbt_conference(self, conference_entry):
        result = _format_gbt7714(conference_entry)
        assert "AI in Education[C]." in result


# ---------------------------------------------------------------------------
# Tool wrapper tests
# ---------------------------------------------------------------------------

class TestCitationStyleConvert:
    def test_missing_styles(self):
        result = json.loads(_convert_citation_style({}, workspace="/tmp/fake"))
        assert "error" in result
        assert "from_style and to_style are required" in result["error"]

    def test_invalid_to_style(self, tmp_path):
        result = json.loads(_convert_citation_style(
            {"from_style": "apa", "to_style": "invalid"},
            workspace=str(tmp_path),
        ))
        assert "error" in result
        assert "Invalid to_style" in result["error"]

    def test_empty_library(self, tmp_path):
        result = json.loads(_convert_citation_style(
            {"from_style": "apa", "to_style": "mla"},
            workspace=str(tmp_path),
        ))
        assert "error" in result
        assert "No references found" in result["error"]

    def test_convert_all_styles(self, tmp_path):
        ws = str(tmp_path)
        # Add a reference
        ref_add({
            "key": "Smith2024",
            "type": "article",
            "fields": {
                "author": "John Smith",
                "year": "2024",
                "title": "Social Capital",
                "journal": "Sociology",
                "volume": "10",
                "pages": "1-20",
            },
        }, ws)

        for to_style in ["apa7", "apa", "chicago", "mla", "gb-t-7714-2015"]:
            result = json.loads(_convert_citation_style(
                {"from_style": "gb-t-7714-2015", "to_style": to_style},
                workspace=ws,
            ))
            assert result["action"] == "style_converted"
            assert result["total_references"] == 1
            assert len(result["converted_references"]) == 1
            ref = result["converted_references"][0]
            assert ref["key"] == "Smith2024"
            assert ref["new_formatted"]
            assert ref["old_formatted"]

    def test_document_text_update(self, tmp_path):
        ws = str(tmp_path)
        ref_add({
            "key": "Smith2024",
            "type": "article",
            "fields": {
                "author": "John Smith",
                "year": "2024",
                "title": "Social Capital",
                "journal": "Sociology",
            },
        }, ws)

        doc_text = "As shown by (Smith, 2024), social capital is important."
        result = json.loads(_convert_citation_style(
            {
                "from_style": "apa",
                "to_style": "mla",
                "document_text": doc_text,
            },
            workspace=ws,
        ))
        assert result["document_updated"] is True
        assert "Smith 2024" in result["updated_text_preview"]

    def test_convert_to_gb_t(self, tmp_path):
        ws = str(tmp_path)
        ref_add({
            "key": "Doe2023",
            "type": "book",
            "fields": {
                "author": "Jane Doe",
                "year": "2023",
                "title": "Research Methods",
                "publisher": "Academic Press",
            },
        }, ws)

        result = json.loads(_convert_citation_style(
            {"from_style": "apa", "to_style": "gb-t-7714-2015"},
            workspace=ws,
        ))
        assert result["action"] == "style_converted"
        ref = result["converted_references"][0]
        assert "[M]" in ref["new_formatted"]
