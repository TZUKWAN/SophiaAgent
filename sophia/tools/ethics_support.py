"""Tool registration for research ethics and IRB support (Phase I)."""

import json
import logging
from typing import Any, Dict

from sophia.research.ethics_support import EthicsSupportEngine
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_ethics_support_tools(registry: ToolRegistry) -> None:
    """Register Phase I tools: ethics checklist, consent generation, risk assessment."""

    engine = EthicsSupportEngine()

    # --- ethics_checklist ---
    registry.register(
        name="ethics_checklist",
        description=(
            "生成研究伦理审查清单。覆盖知情同意、隐私保密、风险受益、弱势群体保护、"
            "数据完整性和研究者行为6个维度，自动标记需关注项并给出IRB建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "study_type": {
                    "type": "string",
                    "enum": ["survey", "interview", "experiment", "observation", "secondary_data"],
                    "default": "survey",
                    "description": "研究类型",
                },
                "involves_vulnerable": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及弱势群体",
                },
                "involves_deception": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及欺骗",
                },
                "involves_sensitive_topics": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及敏感话题",
                },
                "data_linkable": {
                    "type": "boolean",
                    "default": False,
                    "description": "数据是否可关联到个人身份",
                },
                "cross_border": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及跨境数据传输",
                },
                "has_funding_conflict": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否存在利益冲突",
                },
            },
        },
        handler=lambda args: json.dumps(
            engine.ethics_checklist(args), ensure_ascii=False
        ),
    )

    # --- ethics_consent_generate ---
    registry.register(
        name="ethics_consent_generate",
        description=(
            "生成知情同意书。支持成人调查、成人访谈、未成年人和实验研究4种模板，"
            "自动填充研究信息并支持自定义章节。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "template_type": {
                    "type": "string",
                    "enum": ["adult_survey", "adult_interview", "minor_survey", "experiment"],
                    "default": "adult_survey",
                    "description": "同意书模板类型",
                },
                "study_title": {"type": "string", "description": "研究标题"},
                "researcher_name": {"type": "string", "description": "研究者姓名"},
                "researcher_contact": {"type": "string", "description": "研究者联系方式"},
                "estimated_duration": {"type": "string", "description": "预计耗时"},
                "study_purpose": {"type": "string", "description": "研究目的"},
                "study_significance": {"type": "string", "default": "", "description": "研究意义"},
                "discipline": {"type": "string", "default": "", "description": "学科领域"},
                "compensation": {"type": "string", "default": "", "description": "补偿方式"},
                "risks": {"type": "string", "default": "", "description": "风险描述"},
                "benefits": {"type": "string", "default": "", "description": "受益描述"},
                "age_range": {"type": "string", "default": "", "description": "年龄范围（未成年人模板用）"},
                "activity_description": {"type": "string", "default": "", "description": "活动描述（未成年人模板用）"},
                "retention_period": {"type": "string", "default": "", "description": "数据保存期限"},
                "recording_consent": {"type": "string", "default": "", "description": "录音同意（访谈模板用）"},
                "custom_sections": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [],
                    "description": "自定义章节",
                },
            },
            "required": ["study_title", "researcher_name", "researcher_contact", "study_purpose"],
        },
        handler=lambda args: json.dumps(
            engine.generate_consent(args), ensure_ascii=False
        ),
    )

    # --- ethics_consent_templates ---
    registry.register(
        name="ethics_consent_templates",
        description="列出所有可用的知情同意书模板类型。",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=lambda args: json.dumps(
            engine.list_consent_templates(), ensure_ascii=False
        ),
    )

    # --- ethics_risk_assessment ---
    registry.register(
        name="ethics_risk_assessment",
        description=(
            "评估研究风险等级。根据研究类型、弱势群体、欺骗、敏感话题、身体干预、"
            "数据关联性和跨境传输等因素，输出风险分数、等级、IRB审查路径和风险缓解建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "study_type": {
                    "type": "string",
                    "enum": ["survey", "interview", "experiment", "observation", "secondary_data"],
                    "default": "survey",
                    "description": "研究类型",
                },
                "involves_vulnerable": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及弱势群体",
                },
                "involves_deception": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及欺骗",
                },
                "involves_sensitive_topics": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及敏感话题",
                },
                "physical_intervention": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及身体干预",
                },
                "data_linkable": {
                    "type": "boolean",
                    "default": False,
                    "description": "数据是否可关联到个人身份",
                },
                "cross_border": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及跨境数据传输",
                },
                "data_sharing": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否涉及数据共享/公开",
                },
            },
        },
        handler=lambda args: json.dumps(
            engine.assess_risk(args), ensure_ascii=False
        ),
    )
