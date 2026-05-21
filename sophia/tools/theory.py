"""Theory mapping tools for SophiaAgent.

Registers: theory_map, concept_trace, compare_schools
"""

import json
import logging
from typing import Any, Dict

from sophia.research.theory import TheoryMapper, _compare_schools, _concept_trace, _theory_map

logger = logging.getLogger(__name__)


def register_theory_tools(registry, provider=None):
    """Register theory mapping tools.

    Args:
        registry: ToolRegistry instance.
        provider: Optional LLM provider for LLM-augmented analysis.
    """
    mapper = TheoryMapper(provider=provider)

    registry.register(
        name="theory_map",
        description=(
            "Map a research topic to relevant social science theories. "
            "Returns matching theories with relevance scores, relations between them, "
            "and recommended theories to use."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic or question to map theories to",
                },
                "discipline": {
                    "type": "string",
                    "description": "Optional discipline filter (sociology, education, politics, psychology, communication)",
                },
            },
            "required": ["topic"],
        },
        handler=lambda args: _theory_map(args, mapper),
    )

    registry.register(
        name="concept_trace",
        description=(
            "Trace the historical evolution of a social science concept. "
            "Returns evolution stages, current debates, and cross-disciplinary usage. "
            "Pre-computed for: 社会资本, 内卷, 数字劳动, 文化资本, 治理, 全球化, 身份认同, 后真相."
        ),
        parameters={
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "Concept to trace (e.g. '社会资本', '内卷')",
                },
                "language": {
                    "type": "string",
                    "description": "Output language: 'zh' or 'en'",
                    "default": "zh",
                },
            },
            "required": ["concept"],
        },
        handler=lambda args: _concept_trace(args, mapper),
    )

    registry.register(
        name="compare_schools",
        description=(
            "Compare multiple theories or schools across standard dimensions. "
            "Returns a comparison table and markdown-formatted output. "
            "Dimensions: 本体论假设, 认识论立场, 方法论偏好, 核心概念, 代表学者, 经典文献, 主要批评, 适用场景, 局限性."
        ),
        parameters={
            "type": "object",
            "properties": {
                "theory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of theory IDs to compare (e.g. ['social_capital', 'field_theory'])",
                },
            },
            "required": ["theory_ids"],
        },
        handler=lambda args: _compare_schools(args, mapper),
    )

    logger.debug("Registered theory tools: theory_map, concept_trace, compare_schools")
