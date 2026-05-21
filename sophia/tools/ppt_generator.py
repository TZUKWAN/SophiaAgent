"""Tool registration for academic presentation slide generation (Phase K)."""

import json
import logging
from typing import Any, Dict

from sophia.research.ppt_generator import HTMLSlideRenderer, SlidePlanner
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_ppt_tools(registry: ToolRegistry) -> None:
    """Register Phase K tools: PPT structure planning and HTML export."""

    planner = SlidePlanner()
    renderer = HTMLSlideRenderer()

    # --- ppt_structure ---
    registry.register(
        name="ppt_structure",
        description=(
            "生成学术汇报PPT的结构规划。支持学术会议汇报和学位论文答辩两种模式，"
            "输出每页的标题、内容要点、演讲备注和建议图表。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_title": {"type": "string", "description": "论文标题"},
                "paper_abstract": {"type": "string", "default": "", "description": "论文摘要"},
                "mode": {
                    "type": "string",
                    "enum": ["conference", "defense"],
                    "default": "conference",
                    "description": "汇报模式：conference=会议汇报(15页), defense=学位答辩(25页)",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "核心发现列表（用于自动填充发现页）",
                },
                "duration_minutes": {
                    "type": "integer",
                    "default": 0,
                    "description": "汇报时长（分钟），0表示自动推断",
                },
            },
            "required": ["paper_title"],
        },
        handler=lambda args: json.dumps(
            planner.generate_structure(args), ensure_ascii=False
        ),
    )

    # --- ppt_export_html ---
    registry.register(
        name="ppt_export_html",
        description=(
            "将PPT结构导出为可演示的HTML文件。生成的HTML可直接在浏览器中全屏演示，"
            "支持方向键翻页和进度条显示。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "slides": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "幻灯片结构数据（来自ppt_structure的输出）",
                },
                "title": {"type": "string", "default": "学术汇报", "description": "演示文稿标题"},
            },
            "required": ["slides"],
        },
        handler=lambda args: json.dumps(
            renderer.render(args), ensure_ascii=False
        ),
    )
