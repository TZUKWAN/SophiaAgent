"""Academic translation engine (Phase L).

L-1: Academic translation with terminology consistency
L-2: Glossary management
L-3: Small-language abstract translation (ja/de/fr/ru/ko)
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in glossary (100 core terms across 5 disciplines)
# ---------------------------------------------------------------------------

_BUILTIN_GLOSSARY: List[Dict[str, str]] = []

_EDUCATION_TERMS = [
    ("教育公平", "educational equity", "教育学"),
    ("翻转课堂", "flipped classroom", "教育学"),
    ("核心素养", "core competencies", "教育学"),
    ("学习动机", "learning motivation", "教育学"),
    ("学业成就", "academic achievement", "教育学"),
    ("教师专业发展", "teacher professional development", "教育学"),
    ("课程与教学论", "curriculum and instruction", "教育学"),
    ("教育评价体系", "educational evaluation system", "教育学"),
    ("混合式学习", "blended learning", "教育学"),
    ("教育信息化", "educational informatization", "教育学"),
    ("合作学习", "cooperative learning", "教育学"),
    ("自主学习", "self-directed learning", "教育学"),
    ("建构主义", "constructivism", "教育学"),
    (" formative assessment", "形成性评价", "教育学"),
    ("元认知", "metacognition", "教育学"),
    (" scaffolding", "支架式教学", "教育学"),
    ("最近发展区", "zone of proximal development", "教育学"),
    ("教育治理", "educational governance", "教育学"),
    ("终身学习", "lifelong learning", "教育学"),
    ("知识建构", "knowledge construction", "教育学"),
]

_SOCIOLOGY_TERMS = [
    ("社会资本", "social capital", "社会学"),
    ("社会分层", "social stratification", "社会学"),
    ("社会流动", "social mobility", "社会学"),
    ("文化资本", "cultural capital", "社会学"),
    ("场域", "field", "社会学"),
    ("惯习", "habitus", "社会学"),
    ("镶嵌", "embeddedness", "社会学"),
    ("社会网络", "social network", "社会学"),
    ("弱关系", "weak ties", "社会学"),
    ("强关系", "strong ties", "社会学"),
    ("结构洞", "structural holes", "社会学"),
    ("污名化", "stigmatization", "社会学"),
    ("集体行为", "collective behavior", "社会学"),
    ("社会运动", "social movement", "社会学"),
    ("风险社会", "risk society", "社会学"),
    ("个体化", "individualization", "社会学"),
    ("全球化", "globalization", "社会学"),
    ("现代性", "modernity", "社会学"),
    ("后现代", "postmodern", "社会学"),
    ("反思性现代化", "reflexive modernization", "社会学"),
]

_PSYCHOLOGY_TERMS = [
    ("认知负荷", "cognitive load", "心理学"),
    ("工作记忆", "working memory", "心理学"),
    ("执行功能", "executive function", "心理学"),
    ("心理韧性", "psychological resilience", "心理学"),
    ("情绪调节", "emotion regulation", "心理学"),
    ("自我效能感", "self-efficacy", "心理学"),
    ("归因风格", "attribution style", "心理学"),
    ("内隐态度", "implicit attitude", "心理学"),
    ("心理契约", "psychological contract", "心理学"),
    ("职业倦怠", "burnout", "心理学"),
    ("习得性无助", "learned helplessness", "心理学"),
    ("认知失调", "cognitive dissonance", "心理学"),
    ("刻板印象", "stereotype", "心理学"),
    ("社会认同", "social identity", "心理学"),
    ("依恋类型", "attachment style", "心理学"),
    ("大五人格", "Big Five personality", "心理学"),
    ("正念", "mindfulness", "心理学"),
    ("创伤后成长", "post-traumatic growth", "心理学"),
    ("心流", "flow", "心理学"),
    ("决策偏差", "decision bias", "心理学"),
]

_ECONOMICS_TERMS = [
    ("人力资本", "human capital", "经济学"),
    ("信息不对称", "information asymmetry", "经济学"),
    ("市场失灵", "market failure", "经济学"),
    ("外部性", "externality", "经济学"),
    ("机会成本", "opportunity cost", "经济学"),
    ("边际效用", "marginal utility", "经济学"),
    ("规模经济", "economies of scale", "经济学"),
    ("博弈论", "game theory", "经济学"),
    ("纳什均衡", "Nash equilibrium", "经济学"),
    ("道德风险", "moral hazard", "经济学"),
    ("逆向选择", "adverse selection", "经济学"),
    ("全要素生产率", "total factor productivity", "经济学"),
    ("基尼系数", "Gini coefficient", "经济学"),
    ("帕累托最优", "Pareto optimality", "经济学"),
    ("理性预期", "rational expectations", "经济学"),
    ("行为经济学", "behavioral economics", "经济学"),
    ("内生性", "endogeneity", "经济学"),
    ("工具变量", "instrumental variable", "经济学"),
    ("双重差分", "difference-in-differences", "经济学"),
    ("断点回归", "regression discontinuity", "经济学"),
]

_POLITICAL_TERMS = [
    ("国家治理", "state governance", "政治学"),
    ("公共政策", "public policy", "政治学"),
    ("制度变迁", "institutional change", "政治学"),
    ("威权主义", "authoritarianism", "政治学"),
    ("民主转型", "democratic transition", "政治学"),
    ("政治参与", "political participation", "政治学"),
    ("利益集团", "interest group", "政治学"),
    ("政策工具", "policy instrument", "政治学"),
    ("治理效能", "governance effectiveness", "政治学"),
    ("官僚制", "bureaucracy", "政治学"),
    ("合法性", "legitimacy", "政治学"),
    ("权力下放", "decentralization", "政治学"),
    ("政策执行", "policy implementation", "政治学"),
    ("多中心治理", "polycentric governance", "政治学"),
    ("新自由主义", "neoliberalism", "政治学"),
    ("民粹主义", "populism", "政治学"),
    ("公共选择理论", "public choice theory", "政治学"),
    ("囚徒困境", "prisoner's dilemma", "政治学"),
    ("集体行动", "collective action", "政治学"),
    ("寻租", "rent-seeking", "政治学"),
]

for _terms in [_EDUCATION_TERMS, _SOCIOLOGY_TERMS, _PSYCHOLOGY_TERMS, _ECONOMICS_TERMS, _POLITICAL_TERMS]:
    for cn, en, discipline in _terms:
        _BUILTIN_GLOSSARY.append({
            "cn": cn,
            "en": en,
            "discipline": discipline,
        })


# ---------------------------------------------------------------------------
# L-2: Glossary Manager
# ---------------------------------------------------------------------------

class GlossaryManager:
    """Manage academic terminology glossary."""

    def __init__(self):
        self._terms: Dict[str, Dict[str, str]] = {}  # cn -> {en, discipline, custom}
        self._load_builtin()

    def _load_builtin(self) -> None:
        for term in _BUILTIN_GLOSSARY:
            cn = term["cn"]
            self._terms[cn] = {
                "en": term["en"],
                "discipline": term["discipline"],
                "custom": "false",
            }

    def lookup(self, term: str, discipline: str = "") -> Optional[Dict[str, str]]:
        """Look up a term. If discipline given, prefer same-discipline match."""
        term = term.strip()
        # Exact match
        if term in self._terms:
            return {"term": term, **self._terms[term]}
        # Try reverse (English -> Chinese)
        for cn, data in self._terms.items():
            if data["en"].lower() == term.lower():
                return {"term": cn, **data}
        return None

    def search(self, query: str, discipline: str = "") -> List[Dict[str, str]]:
        """Search terms by partial match."""
        query_lower = query.lower()
        results = []
        for cn, data in self._terms.items():
            en = data["en"].lower()
            disc = data.get("discipline", "")
            if query_lower in cn.lower() or query_lower in en:
                if not discipline or discipline == disc:
                    results.append({"term": cn, **data})
        return results

    def add_term(self, cn: str, en: str, discipline: str = "") -> None:
        self._terms[cn] = {
            "en": en,
            "discipline": discipline,
            "custom": "true",
        }

    def remove_term(self, cn: str) -> bool:
        if cn in self._terms and self._terms[cn].get("custom") == "true":
            del self._terms[cn]
            return True
        return False

    def export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["cn", "en", "discipline", "custom"])
        for cn, data in sorted(self._terms.items()):
            writer.writerow([cn, data["en"], data.get("discipline", ""), data.get("custom", "false")])
        return output.getvalue()

    def import_csv(self, csv_text: str) -> int:
        count = 0
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            cn = row.get("cn", "").strip()
            en = row.get("en", "").strip()
            disc = row.get("discipline", "").strip()
            if cn and en:
                self._terms[cn] = {"en": en, "discipline": disc, "custom": "true"}
                count += 1
        return count

    def list_disciplines(self) -> List[str]:
        disciplines = set()
        for data in self._terms.values():
            d = data.get("discipline", "")
            if d:
                disciplines.add(d)
        return sorted(disciplines)

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._terms)
        custom = sum(1 for t in self._terms.values() if t.get("custom") == "true")
        return {
            "total_terms": total,
            "builtin_terms": total - custom,
            "custom_terms": custom,
            "disciplines": self.list_disciplines(),
        }


# ---------------------------------------------------------------------------
# L-1 & L-3: Academic Translator
# ---------------------------------------------------------------------------

class AcademicTranslator:
    """Academic text translator with terminology consistency."""

    # Simple sentence templates for rule-based fallback
    _ZH_EN_PATTERNS = [
        (r"本文研究(.+?)。", r"This study examines \1."),
        (r"结果表明，(.+?)。", r"The results show that \1."),
        (r"基于(.+?)，", r"Based on \1, "),
        (r"通过(.+?)，", r"Through \1, "),
        (r"(.+?)对(.+?)有显著影响。", r"\1 has a significant effect on \2."),
        (r"(.+?)与(.+?)之间存在正相关。", r"There is a positive correlation between \1 and \2."),
    ]

    def __init__(self, glossary: Optional[GlossaryManager] = None):
        self.glossary = glossary or GlossaryManager()

    def translate(self, args: dict) -> Dict[str, Any]:
        """Translate academic text.

        Args:
            text: str
            source_lang: str - "zh" | "en"
            target_lang: str - "zh" | "en"
            discipline: str (optional)
        """
        text = args.get("text", "")
        source_lang = args.get("source_lang", "auto")
        target_lang = args.get("target_lang", "")
        discipline = args.get("discipline", "")

        if not text:
            return {"error": "No text provided"}

        if source_lang == "auto":
            source_lang = self._detect_language(text)

        if not target_lang:
            target_lang = "en" if source_lang == "zh" else "zh"

        # Terminology replacement (rule-based)
        translated, replacements = self._translate_with_glossary(
            text, source_lang, target_lang, discipline
        )

        # Flag terms not in glossary
        warnings = self._check_unknown_terms(text, source_lang, discipline)

        return {
            "original_text": text,
            "translated_text": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "replacements": replacements,
            "warnings": warnings,
            "note": "此为规则翻译结果，建议使用LLM进行润色以确保学术表达的地道性。",
        }

    def translate_abstract(self, args: dict) -> Dict[str, Any]:
        """Translate abstract from small languages (ja/de/fr/ru/ko).

        Args:
            text: str
            source_lang: str - "ja" | "de" | "fr" | "ru" | "ko"
        """
        text = args.get("text", "")
        source_lang = args.get("source_lang", "")

        if not text:
            return {"error": "No text provided"}

        supported = ("ja", "de", "fr", "ru", "ko")
        if source_lang not in supported:
            return {
                "error": f"Unsupported source language '{source_lang}'",
                "supported": list(supported),
            }

        lang_names = {
            "ja": "日语", "de": "德语", "fr": "法语", "ru": "俄语", "ko": "韩语",
        }

        # Without LLM, we can only provide a placeholder
        return {
            "original_text": text,
            "source_lang": source_lang,
            "source_lang_name": lang_names.get(source_lang, source_lang),
            "chinese_translation": "（需要LLM支持进行小语种翻译）",
            "english_translation": "（LLM assistance required for small-language translation）",
            "note": "小语种翻译完全依赖LLM能力。请在有LLM的环境下使用此功能。",
            "fallback_available": False,
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character ratios."""
        import re
        zh_chars = len(re.findall(r"[一-鿿]", text))
        total_chars = len(re.sub(r"\s", "", text))
        if total_chars == 0:
            return "en"
        ratio = zh_chars / total_chars
        return "zh" if ratio > 0.3 else "en"

    def _translate_with_glossary(
        self, text: str, source: str, target: str, discipline: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Replace known terms using glossary."""
        replacements = []
        translated = text

        # Sort terms by length (descending) to avoid partial replacements
        terms = sorted(self.glossary._terms.items(), key=lambda x: len(x[0]), reverse=True)

        if source == "zh" and target == "en":
            for cn, data in terms:
                if cn in translated:
                    if not discipline or data.get("discipline") == discipline or not data.get("discipline"):
                        translated = translated.replace(cn, data["en"])
                        replacements.append({"from": cn, "to": data["en"]})
        elif source == "en" and target == "zh":
            for cn, data in terms:
                en = data["en"]
                if en.lower() in translated.lower():
                    # Simple case-insensitive replacement
                    import re
                    translated = re.sub(rf"\b{re.escape(en)}\b", cn, translated, flags=re.IGNORECASE)
                    replacements.append({"from": en, "to": cn})

        return translated, replacements

    def _check_unknown_terms(self, text: str, source: str, discipline: str) -> List[str]:
        """Check for potentially untranslated academic terms."""
        warnings = []
        import re

        if source == "zh":
            # Look for 4+ character phrases that might be academic terms
            phrases = re.findall(r"[一-鿿]{4,8}", text)
            for phrase in set(phrases):
                if not self.glossary.lookup(phrase, discipline):
                    warnings.append(f"术语未收录：'{phrase}'，建议确认翻译准确性")
        else:
            # Look for multi-word phrases
            phrases = re.findall(r"[a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,}){1,3}", text)
            for phrase in set(phrases):
                if not self.glossary.lookup(phrase, discipline):
                    warnings.append(f"Term not in glossary: '{phrase}'")

        # Limit warnings
        return warnings[:10]
