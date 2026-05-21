"""Tests for Phase L: Academic translation."""

import pytest
from sophia.research.translation import AcademicTranslator, GlossaryManager


@pytest.fixture
def glossary():
    return GlossaryManager()


@pytest.fixture
def translator(glossary):
    return AcademicTranslator(glossary)


# ---------------------------------------------------------------------------
# L-2: Glossary management
# ---------------------------------------------------------------------------

class TestGlossaryManager:
    def test_lookup_existing_term(self, glossary):
        result = glossary.lookup("社会资本")
        assert result is not None
        assert result["en"] == "social capital"
        assert result["discipline"] == "社会学"

    def test_lookup_english_term(self, glossary):
        result = glossary.lookup("social capital")
        assert result is not None
        assert result["term"] == "社会资本"

    def test_lookup_nonexistent(self, glossary):
        result = glossary.lookup("nonexistent_term_xyz")
        assert result is None

    def test_search_partial(self, glossary):
        results = glossary.search("社会")
        assert len(results) > 0
        assert all("社会" in r["term"] or "社会" in r["en"] for r in results)

    def test_search_by_discipline(self, glossary):
        results = glossary.search("", discipline="教育学")
        assert len(results) > 0
        assert all(r["discipline"] == "教育学" for r in results)

    def test_add_custom_term(self, glossary):
        glossary.add_term("测试术语", "test term", "测试学科")
        result = glossary.lookup("测试术语")
        assert result is not None
        assert result["en"] == "test term"
        assert result["custom"] == "true"

    def test_remove_custom_term(self, glossary):
        glossary.add_term("待删除", "to_delete", "")
        assert glossary.remove_term("待删除") is True
        assert glossary.lookup("待删除") is None

    def test_remove_builtin_term_fails(self, glossary):
        assert glossary.remove_term("社会资本") is False
        assert glossary.lookup("社会资本") is not None

    def test_export_csv(self, glossary):
        csv_text = glossary.export_csv()
        assert "cn,en,discipline,custom" in csv_text
        assert "社会资本" in csv_text
        assert "social capital" in csv_text

    def test_import_csv(self, glossary):
        csv_text = "cn,en,discipline,custom\n自定义词,custom word,测试,true"
        count = glossary.import_csv(csv_text)
        assert count == 1
        result = glossary.lookup("自定义词")
        assert result is not None
        assert result["en"] == "custom word"

    def test_list_disciplines(self, glossary):
        disciplines = glossary.list_disciplines()
        assert "教育学" in disciplines
        assert "社会学" in disciplines
        assert "心理学" in disciplines
        assert "经济学" in disciplines
        assert "政治学" in disciplines

    def test_stats(self, glossary):
        stats = glossary.get_stats()
        assert stats["total_terms"] >= 100
        assert stats["builtin_terms"] >= 100
        assert stats["custom_terms"] >= 0
        assert len(stats["disciplines"]) == 5


# ---------------------------------------------------------------------------
# L-1: Academic translation
# ---------------------------------------------------------------------------

class TestAcademicTranslation:
    def test_translate_zh_to_en_basic(self, translator):
        result = translator.translate({
            "text": "社会资本对农民工城市融入的影响",
            "source_lang": "zh",
            "target_lang": "en",
        })
        assert result["source_lang"] == "zh"
        assert result["target_lang"] == "en"
        assert "social capital" in result["translated_text"]
        assert "migrant workers" not in result["translated_text"]  # Not in glossary
        assert len(result["replacements"]) > 0

    def test_translate_en_to_zh(self, translator):
        result = translator.translate({
            "text": "The effect of social capital on urban integration",
            "source_lang": "en",
            "target_lang": "zh",
        })
        assert result["source_lang"] == "en"
        assert result["target_lang"] == "zh"
        assert "社会资本" in result["translated_text"]

    def test_auto_detect_chinese(self, translator):
        result = translator.translate({
            "text": "这是一个中文测试文本",
        })
        assert result["source_lang"] == "zh"
        assert result["target_lang"] == "en"

    def test_auto_detect_english(self, translator):
        result = translator.translate({
            "text": "This is an English test text.",
        })
        assert result["source_lang"] == "en"
        assert result["target_lang"] == "zh"

    def test_discipline_filter(self, translator):
        # "社会" appears in multiple disciplines
        result = translator.translate({
            "text": "社会资本与认知负荷",
            "source_lang": "zh",
            "target_lang": "en",
            "discipline": "社会学",
        })
        assert "social capital" in result["translated_text"]

    def test_empty_text_error(self, translator):
        result = translator.translate({"text": ""})
        assert "error" in result

    def test_warnings_for_unknown_terms(self, translator):
        result = translator.translate({
            "text": "这是一个超复杂且未收录的术语组合",
            "source_lang": "zh",
        })
        # Should flag potentially untranslated terms
        assert isinstance(result["warnings"], list)

    def test_translate_with_glossary_replacements(self, translator):
        result = translator.translate({
            "text": "翻转课堂和合作学习能提升学习动机",
            "source_lang": "zh",
            "target_lang": "en",
            "discipline": "教育学",
        })
        trans = result["translated_text"]
        assert "flipped classroom" in trans
        assert "cooperative learning" in trans
        assert "learning motivation" in trans
        # Check replacements are reported
        r_from = [r["from"] for r in result["replacements"]]
        assert "翻转课堂" in r_from

    def test_terminology_consistency(self, translator):
        # Same term should be translated consistently
        result = translator.translate({
            "text": "社会资本社会资本社会资本",
            "source_lang": "zh",
            "target_lang": "en",
        })
        count = result["translated_text"].count("social capital")
        assert count == 3


# ---------------------------------------------------------------------------
# L-3: Small language abstract translation
# ---------------------------------------------------------------------------

class TestSmallLanguageTranslation:
    def test_japanese_abstract(self, translator):
        result = translator.translate_abstract({
            "text": "これはテストです",
            "source_lang": "ja",
        })
        assert result["source_lang"] == "ja"
        assert result["source_lang_name"] == "日语"
        assert "需要LLM" in result["note"] or "LLM" in result["note"]

    def test_german_abstract(self, translator):
        result = translator.translate_abstract({
            "text": "Dies ist ein Test",
            "source_lang": "de",
        })
        assert result["source_lang"] == "de"
        assert result["source_lang_name"] == "德语"

    def test_unsupported_language(self, translator):
        result = translator.translate_abstract({
            "text": "test",
            "source_lang": "xx",
        })
        assert "error" in result
        assert "supported" in result

    def test_empty_text_error(self, translator):
        result = translator.translate_abstract({
            "text": "",
            "source_lang": "ja",
        })
        assert "error" in result
