"""Requirement clarification gate for academic paper generation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

PAPER_WRITE_TERMS = [
    "写",
    "撰写",
    "生成",
    "起草",
    "成文",
    "write",
    "draft",
    "generate",
]

PAPER_OBJECT_TERMS = [
    "论文",
    "文章",
    "研究报告",
    "综述",
    "paper",
    "article",
    "manuscript",
    "review",
]

EMPIRICAL_TERMS = [
    "实证",
    "回归",
    "因果",
    "数据",
    "样本",
    "变量",
    "稳健性",
    "empirical",
    "regression",
    "causal",
    "data",
]

THEORY_TERMS = [
    "理论",
    "规范研究",
    "思辨",
    "概念",
    "框架",
    "文献综述",
    "综述",
    "theoretical",
    "conceptual",
    "literature review",
]

SKIP_CLARIFICATION_TERMS = [
    "直接写",
    "直接生成",
    "自行决定",
    "按默认",
    "不用确认",
    "不要问",
    "你来定",
    "don't ask",
    "do not ask",
    "use defaults",
]


@dataclass
class PaperRequirementCheck:
    is_paper_request: bool
    requires_clarification: bool
    missing: List[str] = field(default_factory=list)
    message: str = ""


def check_paper_requirements(user_message: str) -> PaperRequirementCheck:
    """Return whether a paper request is underspecified enough to ask first."""
    if not is_paper_request(user_message):
        return PaperRequirementCheck(False, False)

    lowered = user_message.lower()
    if any(term in lowered for term in SKIP_CLARIFICATION_TERMS):
        return PaperRequirementCheck(True, False)

    missing: List[str] = []
    if not _has_paper_type(lowered):
        missing.append("论文类型：理论论文、实证论文、文献综述，或混合型")
    if not _has_word_count(lowered):
        missing.append("目标正文字数：例如 6500、8000、10000 字，不含参考文献")

    if not missing:
        return PaperRequirementCheck(True, False)

    return PaperRequirementCheck(
        True,
        True,
        missing,
        _build_clarification_message(missing),
    )


def is_paper_request(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in PAPER_WRITE_TERMS) and any(
        term in lowered for term in PAPER_OBJECT_TERMS
    )


def _has_paper_type(text: str) -> bool:
    return any(term in text for term in EMPIRICAL_TERMS + THEORY_TERMS)


def _has_word_count(text: str) -> bool:
    if re.search(r"\d{4,6}\s*(字|字符|words|word|characters|chars)", text):
        return True
    if re.search(r"不少于\s*\d{4,6}", text):
        return True
    if re.search(r"至少\s*\d{4,6}", text):
        return True
    return False


def _build_clarification_message(missing: List[str]) -> str:
    missing_text = "\n".join(f"- {item}" for item in missing)
    return (
        "我先确认几个会直接影响成稿质量的关键参数，确认后再开始写，避免生成不符合要求的稿件。\n\n"
        f"还缺：\n{missing_text}\n\n"
        "建议你直接回复类似：\n"
        "“理论论文，正文不少于 8000 字，Word 文档，优先使用工作空间参考文献。”\n\n"
        "如果你想让我自行决定，也可以回复：\n"
        "“按默认直接写。”\n"
        "默认将按理论论文、正文不少于 6500 字、20 篇以上真实参考文献、5 个表格、8 个图示执行。"
    )
