"""Template tools for SophiaAgent.

Registers template_list, template_get, and template_recommend tools
that wrap the TemplateRegistry.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sophia.prompts.templates.registry import TemplateRegistry
    from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_template_tools(tools: "ToolRegistry", registry: "TemplateRegistry") -> None:
    """Register template-related tools on the given ToolRegistry."""

    # ------------------------------------------------------------------
    # template_list
    # ------------------------------------------------------------------
    def _template_list(args: Dict[str, Any]) -> str:
        discipline = args.get("discipline")
        results = registry.list_templates(discipline=discipline)
        return json.dumps(
            {
                "count": len(results),
                "disciplines": registry.list_disciplines(),
                "templates": results,
            },
            ensure_ascii=False,
        )

    tools.register(
        "template_list",
        "List available discipline-specific writing templates. Optionally filter by discipline.",
        {
            "type": "object",
            "properties": {
                "discipline": {
                    "type": "string",
                    "description": "Filter by discipline (e.g. 'history', 'psychology'). Omit to list all.",
                },
            },
        },
        _template_list,
    )

    # ------------------------------------------------------------------
    # template_get
    # ------------------------------------------------------------------
    def _template_get(args: Dict[str, Any]) -> str:
        template_id = args.get("template_id", "").strip()
        if not template_id:
            return json.dumps({"error": "template_id is required"}, ensure_ascii=False)

        tmpl = registry.get_template(template_id)
        if tmpl is None:
            return json.dumps(
                {"error": f"Template not found: {template_id}"},
                ensure_ascii=False,
            )

        # Return full template plus computed helpers
        result = dict(tmpl)
        result["template_id"] = template_id
        result["outline"] = registry.get_outline(template_id)
        result["checklist"] = registry.get_checklist(template_id)
        return json.dumps(result, ensure_ascii=False)

    tools.register(
        "template_get",
        "Get full details of a template by its ID (format: 'discipline/filename').",
        {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "Template ID, e.g. 'history/outline_history_paper'",
                },
            },
            "required": ["template_id"],
        },
        _template_get,
    )

    # ------------------------------------------------------------------
    # template_recommend
    # ------------------------------------------------------------------
    def _template_recommend(args: Dict[str, Any]) -> str:
        research_question = args.get("research_question", "").strip()
        if not research_question:
            return json.dumps(
                {"error": "research_question is required"},
                ensure_ascii=False,
            )
        discipline = args.get("discipline")
        top_n = args.get("top_n", 5)

        recommendations = registry.recommend_templates(
            research_question=research_question,
            discipline=discipline,
        )
        if isinstance(top_n, int) and top_n > 0:
            recommendations = recommendations[:top_n]

        return json.dumps(
            {
                "research_question": research_question,
                "discipline_filter": discipline,
                "count": len(recommendations),
                "recommendations": recommendations,
            },
            ensure_ascii=False,
        )

    tools.register(
        "template_recommend",
        "Recommend writing templates based on a research question or topic.",
        {
            "type": "object",
            "properties": {
                "research_question": {
                    "type": "string",
                    "description": "The user's research question or paper topic",
                },
                "discipline": {
                    "type": "string",
                    "description": "Optional discipline filter (e.g. 'literature')",
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of recommendations to return",
                },
            },
            "required": ["research_question"],
        },
        _template_recommend,
    )
