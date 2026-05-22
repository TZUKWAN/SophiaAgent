"""Tool registration for journal matching and submission guide (Phase J)."""

import json
import logging
from typing import Any, Dict

from sophia.research.journal_db import JournalDatabase
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_journal_tools(registry: ToolRegistry) -> None:
    """Register Phase J tools: journal search, match, and submission guide."""

    db = JournalDatabase()

    # --- journal_search ---
    registry.register(
        name="journal_search",
        description="根据名称、学科或收稿范围搜索期刊。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "default": 20, "description": "返回结果数量上限"},
            },
            "required": ["query"],
        },
        handler=lambda args: json.dumps(
            db.search(args.get("query", ""), limit=args.get("limit", 20)),
            ensure_ascii=False,
        ),
    )

    # --- journal_match ---
    registry.register(
        name="journal_match",
        description=(
            "根据论文标题、摘要和关键词匹配推荐期刊。"
            "返回按匹配度排序的期刊列表。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "abstract": {"type": "string", "description": "论文摘要"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表",
                },
                "discipline": {"type": "string", "description": "学科领域"},
                "method_type": {"type": "string", "description": "研究方法类型（可选）"},
                "top_n": {"type": "integer", "default": 10, "description": "推荐数量"},
            },
            "required": ["title", "abstract"],
        },
        handler=lambda args: json.dumps(
            db.match(args), ensure_ascii=False
        ),
    )

    # --- journal_guide ---
    registry.register(
        name="journal_guide",
        description="获取目标期刊的投稿指南，包括格式要求、常见拒稿原因和写作建议。",
        parameters={
            "type": "object",
            "properties": {
                "journal_id": {"type": "string", "description": "期刊ID"},
                "journal_name": {"type": "string", "description": "期刊名称（与ID二选一）"},
            },
        },
        handler=lambda args: json.dumps(
            db.get_submission_guide(args), ensure_ascii=False
        ),
    )

    # --- journal_list_disciplines ---
    registry.register(
        name="journal_list_disciplines",
        description="列出期刊数据库中所有学科分类。",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=lambda args: json.dumps(
            db.list_disciplines(), ensure_ascii=False
        ),
    )

    # --- journal_list ---
    registry.register(
        name="journal_list",
        description="列出期刊数据库中的期刊，可按学科筛选。",
        parameters={
            "type": "object",
            "properties": {
                "discipline": {"type": "string", "default": "", "description": "学科筛选（空表示全部）"},
                "limit": {"type": "integer", "default": 100, "description": "数量上限"},
            },
        },
        handler=lambda args: json.dumps(
            db.list_journals(
                discipline=args.get("discipline", ""),
                limit=args.get("limit", 100),
            ),
            ensure_ascii=False,
        ),
    )

    # --- journal_find_by_issn ---
    registry.register(
        name="journal_find_by_issn",
        description="按 ISSN 精确查找期刊，支持 ISSN 和 eISSN。",
        parameters={
            "type": "object",
            "properties": {
                "issn": {"type": "string", "description": "期刊 ISSN 或 eISSN，例如 0028-0836"},
            },
            "required": ["issn"],
        },
        handler=lambda args: json.dumps(
            db.find_by_issn(args.get("issn", "")),
            ensure_ascii=False,
        ),
    )

    # --- journal_cas_zone ---
    registry.register(
        name="journal_cas_zone",
        description="查询期刊的中科院分区（CAS 分区），支持按期刊名称或 ISSN 查询。",
        parameters={
            "type": "object",
            "properties": {
                "name_or_issn": {
                    "type": "string",
                    "description": "期刊名称（如 Nature）或 ISSN（如 0028-0836）",
                },
            },
            "required": ["name_or_issn"],
        },
        handler=lambda args: json.dumps(
            db.get_cas_zone(args.get("name_or_issn", "")),
            ensure_ascii=False,
        ),
    )
