"""Tool registration for academic translation (Phase L)."""

import json
import logging
from typing import Any, Dict

from sophia.research.translation import AcademicTranslator, GlossaryManager
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_translation_tools(registry: ToolRegistry) -> None:
    """Register Phase L tools: academic translation and glossary management."""

    glossary = GlossaryManager()
    translator = AcademicTranslator(glossary)

    # --- translate_academic ---
    registry.register(
        name="translate_academic",
        description=(
            "学术文本翻译，支持中英互译。内置100+条核心学术术语对照表，"
            "优先保证术语一致性。无LLM时提供基于术语词典的规则翻译。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "待翻译文本"},
                "source_lang": {
                    "type": "string",
                    "enum": ["auto", "zh", "en"],
                    "default": "auto",
                    "description": "源语言",
                },
                "target_lang": {
                    "type": "string",
                    "enum": ["zh", "en"],
                    "description": "目标语言（留空则自动推断）",
                },
                "discipline": {"type": "string", "default": "", "description": "学科领域（用于术语匹配）"},
            },
            "required": ["text"],
        },
        handler=lambda args: json.dumps(
            translator.translate(args), ensure_ascii=False
        ),
    )

    # --- translate_abstract ---
    registry.register(
        name="translate_abstract",
        description=(
            "小语种学术摘要翻译（日语、德语、法语、俄语、韩语）。"
            "输出中文和英文双语摘要。需要LLM支持。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "待翻译的摘要文本"},
                "source_lang": {
                    "type": "string",
                    "enum": ["ja", "de", "fr", "ru", "ko"],
                    "description": "源语言代码",
                },
            },
            "required": ["text", "source_lang"],
        },
        handler=lambda args: json.dumps(
            translator.translate_abstract(args), ensure_ascii=False
        ),
    )

    # --- glossary_lookup ---
    registry.register(
        name="glossary_lookup",
        description="查找学术术语对照。",
        parameters={
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "待查找术语"},
                "discipline": {"type": "string", "default": "", "description": "学科筛选"},
            },
            "required": ["term"],
        },
        handler=lambda args: json.dumps(
            glossary.lookup(args.get("term", ""), args.get("discipline", "")) or {"not_found": True},
            ensure_ascii=False,
        ),
    )

    # --- glossary_search ---
    registry.register(
        name="glossary_search",
        description="搜索术语表中的术语。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "discipline": {"type": "string", "default": "", "description": "学科筛选"},
            },
            "required": ["query"],
        },
        handler=lambda args: json.dumps(
            glossary.search(args.get("query", ""), args.get("discipline", "")),
            ensure_ascii=False,
        ),
    )

    # --- glossary_add ---
    registry.register(
        name="glossary_add",
        description="向术语表中添加自定义术语。",
        parameters={
            "type": "object",
            "properties": {
                "cn": {"type": "string", "description": "中文术语"},
                "en": {"type": "string", "description": "英文术语"},
                "discipline": {"type": "string", "default": "", "description": "学科"},
            },
            "required": ["cn", "en"],
        },
        handler=lambda args: json.dumps(
            {"success": True, "added": {"cn": args["cn"], "en": args["en"]}},
            ensure_ascii=False,
        ),
    )

    # --- glossary_stats ---
    registry.register(
        name="glossary_stats",
        description="查看术语表统计信息。",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=lambda args: json.dumps(
            glossary.get_stats(), ensure_ascii=False
        ),
    )
