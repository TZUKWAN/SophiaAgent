"""Tool registration for interview and questionnaire pipeline (Phase G)."""

import json
import logging
from typing import Any, Dict

from sophia.research.interview_questionnaire import (
    DataCollectionTracker,
    InterviewEngine,
    PilotAnalyzer,
    QuestionnaireEngine,
    ScaleLibrary,
)
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_interview_questionnaire_tools(registry: ToolRegistry) -> None:
    """Register Phase G tools: interview, questionnaire, scale library, tracker, pilot."""

    q_engine = QuestionnaireEngine()
    i_engine = InterviewEngine()
    tracker = DataCollectionTracker()
    scale_lib = ScaleLibrary()
    pilot = PilotAnalyzer()

    # --- questionnaire_design ---
    registry.register(
        name="questionnaire_design",
        description=(
            "设计问卷调查问卷。根据研究主题、目标人群和研究问题自动生成问卷草稿，"
            "包含人口统计学题项、李克特量表题、开放题等。同时提供问卷设计质量检查报告和推荐量表。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "研究主题/标题"},
                "target_population": {"type": "string", "description": "目标人群，如'大学生'、'中小学教师'"},
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "研究问题列表",
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要测量的变量列表（可选）",
                },
                "n_questions_estimate": {
                    "type": "integer",
                    "default": 20,
                    "description": "预估题目数量",
                },
                "scale_suggestions": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否推荐已验证量表",
                },
                "include_demographics": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否包含人口统计学部分",
                },
            },
            "required": ["topic", "target_population"],
        },
        handler=lambda args: json.dumps(
            q_engine.design_questionnaire(args), ensure_ascii=False
        ),
    )

    # --- questionnaire_validate ---
    registry.register(
        name="questionnaire_validate",
        description=(
            "验证已有问卷的设计质量。检查双重问题、引导性措辞、量表一致性、"
            "题目数量合理性、反向计分提醒等常见问题。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "题目列表，每个题目包含question_id、text、type、options等字段",
                },
            },
            "required": ["questions"],
        },
        handler=lambda args: json.dumps(
            q_engine.validate_questionnaire(args), ensure_ascii=False
        ),
    )

    # --- interview_protocol ---
    registry.register(
        name="interview_protocol",
        description=(
            "生成半结构化访谈提纲。支持生命史访谈、现象学访谈、扎根理论访谈、"
            "叙事访谈和焦点访谈五种类型，包含追问策略和伦理提醒。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "研究主题"},
                "interview_type": {
                    "type": "string",
                    "default": "focused",
                    "enum": ["life_history", "phenomenological", "grounded", "narrative", "focused"],
                    "description": "访谈类型",
                },
                "target_population": {"type": "string", "description": "目标受访者群体"},
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "研究问题列表",
                },
                "n_questions_per_section": {
                    "type": "integer",
                    "default": 3,
                    "description": "每个部分的问题数量",
                },
                "estimated_duration": {
                    "type": "integer",
                    "default": 60,
                    "description": "预计访谈时长（分钟）",
                },
            },
            "required": ["topic", "target_population"],
        },
        handler=lambda args: json.dumps(
            i_engine.generate_protocol(args), ensure_ascii=False
        ),
    )

    # --- scale_search ---
    registry.register(
        name="scale_search",
        description=(
            "搜索常用学术量表。包含心理健康、自我效能、学习投入、职业倦怠、"
            "人格、生活满意度、社会支持等10+个已验证量表。支持按关键词、领域、人群筛选。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": "", "description": "搜索关键词"},
                "domain": {"type": "string", "default": "", "description": "领域筛选"},
                "population": {"type": "string", "default": "", "description": "适用人群筛选"},
                "language": {"type": "string", "default": "zh", "description": "语言筛选"},
            },
            "required": [],
        },
        handler=lambda args: json.dumps(
            scale_lib.search(
                query=args.get("query", ""),
                domain=args.get("domain", ""),
                population=args.get("population", ""),
                language=args.get("language", ""),
            ),
            ensure_ascii=False
        ),
    )

    # --- scale_get ---
    registry.register(
        name="scale_get",
        description="获取特定量表的完整信息，包括题项数量、量程、子维度、信度指标和来源。",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "量表名称"},
            },
            "required": ["name"],
        },
        handler=lambda args: json.dumps(
            scale_lib.get(args.get("name", "")), ensure_ascii=False
        ),
    )

    # --- data_collection_create ---
    registry.register(
        name="data_collection_create",
        description="创建数据采集追踪项目，用于记录问卷/访谈的回收进度。",
        parameters={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "项目ID（可选，自动分配）"},
                "project_name": {"type": "string", "description": "项目名称"},
                "target_sample_size": {"type": "integer", "default": 100},
                "collection_method": {
                    "type": "string",
                    "default": "survey",
                    "enum": ["survey", "interview", "mixed"],
                },
                "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
            },
            "required": ["project_name"],
        },
        handler=lambda args: json.dumps(
            tracker.create_project(args), ensure_ascii=False
        ),
    )

    # --- data_collection_add ---
    registry.register(
        name="data_collection_add",
        description="向数据采集项目添加一条记录（完成/部分完成/拒答/无效）。",
        parameters={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "项目ID"},
                "record_id": {"type": "string", "description": "记录ID（可选）"},
                "status": {
                    "type": "string",
                    "enum": ["completed", "partial", "refused", "invalid"],
                    "description": "记录状态",
                },
                "duration_minutes": {"type": "number", "default": 0},
                "quality_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "质量标记，如['too_fast', 'straight_line']",
                },
                "notes": {"type": "string", "default": ""},
            },
            "required": ["project_id", "status"],
        },
        handler=lambda args: json.dumps(
            tracker.add_record(args), ensure_ascii=False
        ),
    )

    # --- data_collection_report ---
    registry.register(
        name="data_collection_report",
        description="获取数据采集项目的进度报告，包括响应率、完成率、平均时长和质量标记统计。",
        parameters={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "项目ID"},
            },
            "required": ["project_id"],
        },
        handler=lambda args: json.dumps(
            tracker.get_report(args), ensure_ascii=False
        ),
    )

    # --- pilot_analysis ---
    registry.register(
        name="pilot_analysis",
        description=(
            "分析问卷预测试数据。输出每道题的均值、标准差、难度、天花板/地板效应、"
            "缺失率、题总相关，以及整体Cronbach's α信度系数和修改建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "预测试数据，每行一个受访者的字典",
                },
                "item_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "量表题项对应的列名列表",
                },
                "scale_min": {"type": "integer", "default": 1},
                "scale_max": {"type": "integer", "default": 5},
                "reverse_items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "反向计分题项的序号（1-based）",
                },
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "题项名称（可选）",
                },
            },
            "required": ["data", "item_cols"],
        },
        handler=lambda args: json.dumps(
            pilot.analyze(args), ensure_ascii=False
        ),
    )
