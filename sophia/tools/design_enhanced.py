"""Tool registration for enhanced research design (Phase H)."""

import json
import logging
from typing import Any, Dict

from sophia.research.design_enhanced import EnhancedDesignEngine
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_design_enhanced_tools(registry: ToolRegistry) -> None:
    """Register Phase H tools: design templates, mixed methods, quality assessment."""

    engine = EnhancedDesignEngine()

    # --- design_template ---
    registry.register(
        name="design_template",
        description=(
            "获取研究设计模板。支持实验、准实验、调查、质性、混合方法、纵向、"
            "案例研究和行动研究8种设计类型，包含各部分结构、常用方法和质量评估标准。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "design_type": {
                    "type": "string",
                    "enum": [
                        "experimental", "quasi_experimental", "survey",
                        "qualitative", "mixed_methods", "longitudinal",
                        "case_study", "action_research",
                    ],
                    "description": "研究设计类型",
                },
                "research_question": {
                    "type": "string",
                    "default": "",
                    "description": "研究问题（可选，用于对齐检查）",
                },
                "discipline": {
                    "type": "string",
                    "default": "",
                    "description": "学科（可选，用于学科专属建议）",
                },
            },
            "required": ["design_type"],
        },
        handler=lambda args: json.dumps(
            engine.get_design_template(args), ensure_ascii=False
        ),
    )

    # --- design_list_types ---
    registry.register(
        name="design_list_types",
        description="列出所有可用的研究设计类型及其简介。",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=lambda args: json.dumps(
            engine.list_design_types(), ensure_ascii=False
        ),
    )

    # --- mixed_methods_design ---
    registry.register(
        name="mixed_methods_design",
        description=(
            "生成混合方法研究设计。支持聚敛式并行、解释性顺序、探索性顺序、"
            "嵌入式和变革性5种设计类型，包含设计图、整合策略和质量检查清单。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "design_subtype": {
                    "type": "string",
                    "default": "convergent_parallel",
                    "enum": [
                        "convergent_parallel", "explanatory_sequential",
                        "exploratory_sequential", "embedded", "transformative",
                    ],
                    "description": "混合方法设计子类型",
                },
                "research_question": {"type": "string", "description": "研究问题"},
                "quantitative_focus": {"type": "string", "description": "定量部分聚焦什么"},
                "qualitative_focus": {"type": "string", "description": "定性部分聚焦什么"},
                "integration_strategy": {
                    "type": "string",
                    "default": "数据三角验证",
                    "description": "整合策略",
                },
            },
            "required": ["research_question", "quantitative_focus", "qualitative_focus"],
        },
        handler=lambda args: json.dumps(
            engine.mixed_methods_design(args), ensure_ascii=False
        ),
    )

    # --- design_quality_assessment ---
    registry.register(
        name="design_quality_assessment",
        description=(
            "评估研究设计质量。从研究问题清晰度、方法匹配度、内部效度、"
            "统计严谨性和伦理合规5个维度评分（每项0-20，总分100），给出等级和改进建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "design_type": {"type": "string", "description": "研究设计类型"},
                "research_question": {"type": "string", "description": "研究问题"},
                "sections": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "设计各部分（可选）",
                },
                "methods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "拟采用的方法列表",
                },
                "sample_size": {"type": "integer", "default": 0},
                "has_control_group": {"type": "boolean", "default": False},
                "has_randomization": {"type": "boolean", "default": False},
                "has_pretest": {"type": "boolean", "default": False},
                "has_ethics_approval": {"type": "boolean", "default": False},
                "has_power_analysis": {"type": "boolean", "default": False},
                "has_pilot": {"type": "boolean", "default": False},
            },
            "required": ["design_type", "research_question"],
        },
        handler=lambda args: json.dumps(
            engine.assess_design_quality(args), ensure_ascii=False
        ),
    )

    # --- method_fit_check ---
    registry.register(
        name="method_fit_check",
        description=(
            "检查特定方法与研究设计的匹配度。输出匹配分数、匹配等级、"
            "匹配理由和潜在担忧。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "research_question": {"type": "string", "description": "研究问题"},
                "design_type": {"type": "string", "description": "研究设计类型"},
                "proposed_method": {"type": "string", "description": "拟采用的方法"},
                "data_description": {
                    "type": "object",
                    "description": "数据描述，如{'N': 100, 'type': 'panel'}",
                },
            },
            "required": ["research_question", "design_type", "proposed_method"],
        },
        handler=lambda args: json.dumps(
            engine.check_method_fit(args), ensure_ascii=False
        ),
    )
