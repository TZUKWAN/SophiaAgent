"""Interview and questionnaire data collection pipeline (Phase G).

G-1 问卷设计助手
G-2 访谈提纲生成器
G-3 数据采集状态追踪
G-4 量表推荐库
G-5 预测试(pilot)分析
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# G-4: Scale Library
# ===========================================================================

@dataclass
class ScaleTemplate:
    name: str
    name_en: str
    description: str
    domain: str
    n_items: int
    scale_range: Tuple[int, int]
    scale_labels: List[str]
    subscales: List[Dict[str, Any]] = field(default_factory=list)
    reliability: Dict[str, float] = field(default_factory=dict)
    source: str = ""
    population: str = ""
    language: str = "zh"


_SCALE_LIBRARY: List[ScaleTemplate] = [
    ScaleTemplate(
        name="抑郁自评量表",
        name_en="SDS (Self-Rating Depression Scale)",
        description="Zung编制的自评抑郁量表，用于评估抑郁症状的严重程度。",
        domain="心理健康",
        n_items=20,
        scale_range=(1, 4),
        scale_labels=["从无或偶尔", "有时", "经常", "总是"],
        subscales=[],
        reliability={"cronbach_alpha": 0.84},
        source="Zung, W.W. (1965). A self-rating depression scale. Arch Gen Psychiatry.",
        population="成人",
        language="zh",
    ),
    ScaleTemplate(
        name="焦虑自评量表",
        name_en="SAS (Self-Rating Anxiety Scale)",
        description="Zung编制的自评焦虑量表。",
        domain="心理健康",
        n_items=20,
        scale_range=(1, 4),
        scale_labels=["从无或偶尔", "有时", "经常", "总是"],
        subscales=[],
        reliability={"cronbach_alpha": 0.82},
        source="Zung, W.W. (1971). A rating instrument for anxiety disorders. Psychosomatics.",
        population="成人",
        language="zh",
    ),
    ScaleTemplate(
        name="一般自我效能感量表",
        name_en="GSES (General Self-Efficacy Scale)",
        description="Schwarzer编制的一般自我效能感量表中文版，测量个体面对困难时的信心。",
        domain="自我效能",
        n_items=10,
        scale_range=(1, 4),
        scale_labels=["完全不正确", "多数不正确", "多数正确", "完全正确"],
        subscales=[],
        reliability={"cronbach_alpha": 0.87},
        source="Schwarzer, R., et al. (1997). The General Self-Efficacy Scale.\n王才康等 (2001). 中文修订版.",
        population="成人/大学生",
        language="zh",
    ),
    ScaleTemplate(
        name="学习投入量表",
        name_en="UWES-S (Utrecht Work Engagement Scale-Students)",
        description="Schaufeli等编制的学习投入量表中文版，测量活力、奉献和专注三个维度。",
        domain="教育心理",
        n_items=17,
        scale_range=(0, 6),
        scale_labels=["从不", "几乎不", "很少", "有时", "经常", "非常频繁", "总是"],
        subscales=[
            {"name": "活力", "items": [1, 4, 8, 12, 15]},
            {"name": "奉献", "items": [2, 5, 9, 13, 16]},
            {"name": "专注", "items": [3, 6, 10, 14, 17]},
        ],
        reliability={"cronbach_alpha": 0.91, "活力": 0.83, "奉献": 0.88, "专注": 0.85},
        source="Schaufeli, W.B., et al. (2002).\n方来坛等 (2008). 中文版修订.",
        population="大学生",
        language="zh",
    ),
    ScaleTemplate(
        name="教师职业倦怠量表",
        name_en="MBI-ES (Maslach Burnout Inventory-Educators Survey)",
        description="Maslach职业倦怠量表教育版中文版，测量情绪衰竭、去人格化和低个人成就感。",
        domain="职业健康",
        n_items=22,
        scale_range=(0, 6),
        scale_labels=["从不", "极少", "偶尔", "经常", "非常频繁", "每天"],
        subscales=[
            {"name": "情绪衰竭", "items": list(range(1, 9))},
            {"name": "去人格化", "items": list(range(9, 15))},
            {"name": "低个人成就感", "items": list(range(15, 23)), "reverse": True},
        ],
        reliability={"cronbach_alpha": 0.88, "情绪衰竭": 0.90, "去人格化": 0.78, "低个人成就感": 0.82},
        source="Maslach, C., et al. (1996). MBI Manual.\n李超平等 (2003). 中文版修订.",
        population="教师",
        language="zh",
    ),
    ScaleTemplate(
        name="大五人格简式量表",
        name_en="NEO-FFI (NEO Five-Factor Inventory)",
        description="Costa & McCrae大五人格简式量表中文版，测量神经质、外向性、开放性、宜人性、尽责性。",
        domain="人格",
        n_items=60,
        scale_range=(1, 5),
        scale_labels=["完全不同意", "不同意", "不确定", "同意", "完全同意"],
        subscales=[
            {"name": "神经质", "items": list(range(1, 13))},
            {"name": "外向性", "items": list(range(13, 25))},
            {"name": "开放性", "items": list(range(25, 37))},
            {"name": "宜人性", "items": list(range(37, 49))},
            {"name": "尽责性", "items": list(range(49, 61))},
        ],
        reliability={"cronbach_alpha": 0.90},
        source="Costa, P.T. & McCrae, R.R. (1992).\n张建平等 (2005). 中文版修订.",
        population="成人",
        language="zh",
    ),
    ScaleTemplate(
        name="生活满意度量表",
        name_en="SWLS (Satisfaction with Life Scale)",
        description="Diener等编制的生活满意度量表中文版。",
        domain="主观幸福感",
        n_items=5,
        scale_range=(1, 7),
        scale_labels=["非常不同意", "不同意", "轻微不同意", "中立", "轻微同意", "同意", "非常同意"],
        subscales=[],
        reliability={"cronbach_alpha": 0.87},
        source="Diener, E., et al. (1985).\n邢占军 (2005). 中文版修订.",
        population="成人",
        language="zh",
    ),
    ScaleTemplate(
        name="社会支持评定量表",
        name_en="SSRS (Social Support Rating Scale)",
        description="肖水源编制的社会支持评定量表，测量客观支持、主观支持和支持利用度。",
        domain="社会心理",
        n_items=10,
        scale_range=(1, 4),
        scale_labels=["无", "轻度", "中度", "重度"],
        subscales=[
            {"name": "客观支持", "items": [2, 6, 7]},
            {"name": "主观支持", "items": [1, 3, 4, 5]},
            {"name": "支持利用度", "items": [8, 9, 10]},
        ],
        reliability={"cronbach_alpha": 0.89},
        source="肖水源 (1994). 社会支持评定量表. 临床精神医学杂志.",
        population="成人",
        language="zh",
    ),
    ScaleTemplate(
        name="学业自我效能感量表",
        name_en="Academic Self-Efficacy Scale",
        description="梁宇颂等编制的学业自我效能感量表，测量学习能力自我效能和学习行为自我效能。",
        domain="教育心理",
        n_items=22,
        scale_range=(1, 5),
        scale_labels=["完全不符合", "不太符合", "不确定", "比较符合", "完全符合"],
        subscales=[
            {"name": "学习能力自我效能", "items": list(range(1, 12))},
            {"name": "学习行为自我效能", "items": list(range(12, 23))},
        ],
        reliability={"cronbach_alpha": 0.91, "学习能力": 0.88, "学习行为": 0.86},
        source="梁宇颂, 周宗奎 (2000). 学业自我效能感量表. 行为科学杂志.",
        population="中学生/大学生",
        language="zh",
    ),
    ScaleTemplate(
        name="心理韧性量表",
        name_en="CD-RISC (Connor-Davidson Resilience Scale)",
        description="Connor & Davidson编制的心理韧性量表中文版。",
        domain="心理健康",
        n_items=25,
        scale_range=(0, 4),
        scale_labels=["从不", "很少", "有时", "经常", "几乎总是"],
        subscales=[
            {"name": "坚韧", "items": list(range(1, 14))},
            {"name": "力量", "items": list(range(14, 22))},
            {"name": "乐观", "items": list(range(22, 26))},
        ],
        reliability={"cronbach_alpha": 0.89},
        source="Connor, K.M. & Davidson, J.R. (2003).\nYu, X. & Zhang, J. (2007). 中文版.",
        population="成人",
        language="zh",
    ),
]


class ScaleLibrary:
    """Library of commonly used academic scales."""

    def __init__(self):
        self._scales = {s.name: s for s in _SCALE_LIBRARY}

    def search(self, query: str = "", domain: str = "", population: str = "", language: str = "") -> List[Dict[str, Any]]:
        """Search scales by keywords, domain, population, or language."""
        results = []
        q = query.lower()
        for s in self._scales.values():
            if domain and s.domain != domain:
                continue
            if population and population not in s.population:
                continue
            if language and s.language != language:
                continue
            if q and q not in s.name.lower() and q not in s.description.lower() and q not in s.domain.lower():
                continue
            results.append(self._scale_to_dict(s))
        return results

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        s = self._scales.get(name)
        if s:
            return self._scale_to_dict(s)
        return None

    def list_domains(self) -> List[str]:
        return sorted({s.domain for s in self._scales.values()})

    def list_all(self) -> List[Dict[str, Any]]:
        return [self._scale_to_dict(s) for s in self._scales.values()]

    @staticmethod
    def _scale_to_dict(s: ScaleTemplate) -> Dict[str, Any]:
        return {
            "name": s.name,
            "name_en": s.name_en,
            "description": s.description,
            "domain": s.domain,
            "n_items": s.n_items,
            "scale_range": s.scale_range,
            "scale_labels": s.scale_labels,
            "subscales": s.subscales,
            "reliability": s.reliability,
            "source": s.source,
            "population": s.population,
            "language": s.language,
        }


# ===========================================================================
# G-1: Questionnaire Design Engine
# ===========================================================================

@dataclass
class QuestionItem:
    question_id: str
    text: str
    type: str  # "single_choice", "multiple_choice", "likert", "open_ended", "ranking", "matrix"
    options: List[Dict[str, Any]] = field(default_factory=list)
    required: bool = True
    condition: Optional[str] = None  # e.g. "Q1 == 'A'"
    tags: List[str] = field(default_factory=list)


class QuestionnaireEngine:
    """Design and validate questionnaires."""

    _QUESTION_TYPE_TEMPLATES = {
        "single_choice": {
            "description": "单选题",
            "option_format": "A/B/C/D",
        },
        "multiple_choice": {
            "description": "多选题",
            "option_format": "可多选",
        },
        "likert": {
            "description": "李克特量表题",
            "option_format": "1-5分/1-7分",
        },
        "open_ended": {
            "description": "开放题",
            "option_format": "文本框",
        },
        "ranking": {
            "description": "排序题",
            "option_format": "按优先级排序",
        },
        "matrix": {
            "description": "矩阵题",
            "option_format": "行×列",
        },
    }

    _COMMON_PITFALLS = [
        (r".*(非常|特别|极其|绝对).*", "避免使用极端程度副词，可能导致天花板效应"),
        (r".*(所有|全部|每个人|总是).*", "避免绝对化表述，受访者可能找到反例"),
        (r".*(和|与|及).*", "注意一题多问（double-barreled），建议拆分为多题"),
        (r".*(为什么|怎么).*", "开放题提示：考虑是否需要编码方案"),
        (r".*(是否).*", "是非题提示：二分法可能丢失信息，考虑改用程度量表"),
        (r".*不.*", "否定表述可能造成理解困难，建议改为肯定表述"),
    ]

    def __init__(self, scale_library: Optional[ScaleLibrary] = None):
        self.scale_library = scale_library or ScaleLibrary()

    def design_questionnaire(self, args: dict) -> Dict[str, Any]:
        """Generate a questionnaire draft from research topic and requirements.

        Args:
            topic: Research topic/title.
            target_population: e.g. "大学生", "中小学教师", "农民工".
            research_questions: List of research questions.
            variables: List of variables to measure (optional).
            n_questions_estimate: Estimated number of questions (default 20).
            scale_suggestions: Whether to suggest validated scales (default True).
            include_demographics: Whether to include demographic section (default True).

        Returns:
            Dictionary with sections, questions, validation notes, and scale suggestions.
        """
        topic = args.get("topic", "")
        target = args.get("target_population", "")
        rqs = args.get("research_questions", [])
        variables = args.get("variables", [])
        n_est = int(args.get("n_questions_estimate", 20))
        scale_suggestions = args.get("scale_suggestions", True)
        include_demo = args.get("include_demographics", True)

        sections = []
        all_questions = []
        q_counter = 1

        # Section 1: Demographics
        if include_demo:
            demo_questions = self._generate_demographics(target)
            for dq in demo_questions:
                dq.question_id = f"Q{q_counter}"
                all_questions.append(dq)
                q_counter += 1
            sections.append({
                "section_id": "S1",
                "title": "基本信息",
                "description": "本部分用于了解受访者的基本背景信息。",
                "questions": [self._q_to_dict(q) for q in demo_questions],
            })

        # Section 2+: Research questions mapped to sections
        for i, rq in enumerate(rqs, start=2):
            sec_questions = self._generate_questions_for_rq(
                rq, variables, n_est // max(len(rqs), 1), q_counter
            )
            for sq in sec_questions:
                all_questions.append(sq)
                q_counter += 1
            sections.append({
                "section_id": f"S{i}",
                "title": f"研究问题 {i-1}: {rq[:30]}...",
                "description": f"围绕'{rq}'设计的测量题项。",
                "questions": [self._q_to_dict(q) for q in sec_questions],
            })

        # Validation
        validation = self._validate_questionnaire(all_questions)

        # Scale suggestions
        scales = []
        if scale_suggestions:
            scales = self._suggest_scales(topic, variables, target)

        return {
            "title": f"《{topic}》调查问卷",
            "target_population": target,
            "estimated_time": self._estimate_time(all_questions),
            "sections": sections,
            "validation": validation,
            "suggested_scales": scales,
            "total_questions": len(all_questions),
        }

    def validate_questionnaire(self, args: dict) -> Dict[str, Any]:
        """Validate an existing questionnaire for common issues."""
        questions_data = args.get("questions", [])
        questions = []
        for i, qd in enumerate(questions_data, 1):
            questions.append(QuestionItem(
                question_id=qd.get("question_id", f"Q{i}"),
                text=qd.get("text", ""),
                type=qd.get("type", "open_ended"),
                options=qd.get("options", []),
                required=qd.get("required", True),
                condition=qd.get("condition"),
                tags=qd.get("tags", []),
            ))
        return self._validate_questionnaire(questions)

    def _generate_demographics(self, target: str) -> List[QuestionItem]:
        """Generate standard demographic questions based on target population."""
        common = [
            QuestionItem("", "您的性别是？", "single_choice", [
                {"value": "A", "label": "男"},
                {"value": "B", "label": "女"},
                {"value": "C", "label": "其他/不愿透露"},
            ]),
            QuestionItem("", "您的年龄是？", "single_choice", [
                {"value": "A", "label": "18岁以下"},
                {"value": "B", "label": "18-25岁"},
                {"value": "C", "label": "26-35岁"},
                {"value": "D", "label": "36-45岁"},
                {"value": "E", "label": "46-60岁"},
                {"value": "F", "label": "60岁以上"},
            ]),
        ]

        if "学生" in target or "大学生" in target:
            common.extend([
                QuestionItem("", "您的年级是？", "single_choice", [
                    {"value": "A", "label": "大一"},
                    {"value": "B", "label": "大二"},
                    {"value": "C", "label": "大三"},
                    {"value": "D", "label": "大四"},
                    {"value": "E", "label": "研究生及以上"},
                ]),
                QuestionItem("", "您的专业类别是？", "single_choice", [
                    {"value": "A", "label": "人文社科"},
                    {"value": "B", "label": "理工科"},
                    {"value": "C", "label": "医科"},
                    {"value": "D", "label": "艺术体育"},
                    {"value": "E", "label": "其他"},
                ]),
            ])
        elif "教师" in target:
            common.extend([
                QuestionItem("", "您的教龄是？", "single_choice", [
                    {"value": "A", "label": "5年以下"},
                    {"value": "B", "label": "5-10年"},
                    {"value": "C", "label": "10-20年"},
                    {"value": "D", "label": "20年以上"},
                ]),
                QuestionItem("", "您任教的学段是？", "single_choice", [
                    {"value": "A", "label": "小学"},
                    {"value": "B", "label": "初中"},
                    {"value": "C", "label": "高中"},
                    {"value": "D", "label": "高校"},
                ]),
            ])
        elif "农民工" in target or "工人" in target:
            common.extend([
                QuestionItem("", "您目前的工作年限是？", "single_choice", [
                    {"value": "A", "label": "1年以下"},
                    {"value": "B", "label": "1-3年"},
                    {"value": "C", "label": "3-5年"},
                    {"value": "D", "label": "5-10年"},
                    {"value": "E", "label": "10年以上"},
                ]),
                QuestionItem("", "您的月收入水平（元）是？", "single_choice", [
                    {"value": "A", "label": "3000以下"},
                    {"value": "B", "label": "3000-5000"},
                    {"value": "C", "label": "5000-8000"},
                    {"value": "D", "label": "8000-12000"},
                    {"value": "E", "label": "12000以上"},
                ]),
            ])

        return common

    def _generate_questions_for_rq(
        self, rq: str, variables: List[str], n_questions: int, start_idx: int
    ) -> List[QuestionItem]:
        """Generate questions for a research question."""
        questions = []
        # Simple heuristic: create Likert-scale items for each variable
        for vi, var in enumerate(variables[:n_questions]):
            if vi >= n_questions:
                break
            qid = f"Q{start_idx + vi}"
            questions.append(QuestionItem(
                question_id=qid,
                text=f"{var}。",
                type="likert",
                options=[
                    {"value": 1, "label": "非常不同意"},
                    {"value": 2, "label": "不同意"},
                    {"value": 3, "label": "一般"},
                    {"value": 4, "label": "同意"},
                    {"value": 5, "label": "非常同意"},
                ],
                tags=[var, "likert_5"],
            ))

        # If fewer variables than n_questions, add some open-ended
        remaining = n_questions - len(questions)
        if remaining > 0:
            questions.append(QuestionItem(
                question_id=f"Q{start_idx + len(variables)}",
                text=f"关于'{rq}'，您还有什么补充意见吗？",
                type="open_ended",
                required=False,
                tags=["open_ended", "supplement"],
            ))

        return questions

    def _validate_questionnaire(self, questions: List[QuestionItem]) -> Dict[str, Any]:
        """Check for common questionnaire design issues."""
        issues = []
        warnings = []
        info = []

        # Check for double-barreled questions
        for q in questions:
            if "和" in q.text or "与" in q.text or "及" in q.text:
                if q.type != "open_ended":
                    issues.append({
                        "question_id": q.question_id,
                        "type": "double_barreled",
                        "detail": f"题目'{q.text[:30]}...'可能包含多个问题，建议拆分。",
                    })

        # Check for leading/loaded questions
        for q in questions:
            for pattern, advice in self._COMMON_PITFALLS:
                if re.search(pattern, q.text) and q.type != "open_ended":
                    warnings.append({
                        "question_id": q.question_id,
                        "type": "loaded_language",
                        "detail": f"题目'{q.text[:30]}...': {advice}",
                    })
                    break

        # Check scale consistency
        likert_scales = set()
        for q in questions:
            if q.type == "likert" and q.options:
                likert_scales.add(len(q.options))
        if len(likert_scales) > 1:
            warnings.append({
                "type": "inconsistent_scales",
                "detail": f"问卷中使用了多种李克特量表点数：{sorted(likert_scales)}，建议统一为5点或7点量表。",
            })

        # Check logical jump coverage
        conditions = [q.condition for q in questions if q.condition]
        if conditions:
            info.append({
                "type": "conditional_logic",
                "detail": f"问卷包含{len(conditions)}道条件逻辑题，请确保逻辑跳转无死循环。",
            })

        # Check question count
        n = len(questions)
        if n > 50:
            warnings.append({
                "type": "long_questionnaire",
                "detail": f"题目数量({n})较多，预计完成时间超过15分钟，可能导致中途弃答率上升。",
            })
        elif n < 5:
            warnings.append({
                "type": "short_questionnaire",
                "detail": f"题目数量({n})过少，可能无法充分测量研究变量。",
            })

        # Check for missing reverse-coded items in Likert scales
        likert_count = sum(1 for q in questions if q.type == "likert")
        if likert_count > 5:
            info.append({
                "type": "reverse_coding_reminder",
                "detail": f"共有{likert_count}道李克特量表题，建议考虑设置反向计分题以控制默认偏差。",
            })

        return {
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "info_count": len(info),
            "issues": issues,
            "warnings": warnings,
            "info": info,
            "pass": len(issues) == 0 and len(warnings) <= 2,
        }

    def _suggest_scales(self, topic: str, variables: List[str], target: str) -> List[Dict[str, Any]]:
        """Suggest validated scales based on topic and variables."""
        suggestions = []
        topic_lower = topic.lower()

        # Keyword matching
        keyword_map = {
            "抑郁": ["抑郁自评量表"],
            "焦虑": ["焦虑自评量表"],
            "自我效能": ["一般自我效能感量表", "学业自我效能感量表"],
            "学习": ["学习投入量表", "学业自我效能感量表"],
            "倦怠": ["教师职业倦怠量表"],
            "人格": ["大五人格简式量表"],
            "满意": ["生活满意度量表"],
            "社会支持": ["社会支持评定量表"],
            "韧性": ["心理韧性量表"],
            "心理": ["抑郁自评量表", "焦虑自评量表", "心理韧性量表"],
        }

        matched = set()
        for kw, scales in keyword_map.items():
            if kw in topic_lower or any(kw in v for v in variables):
                matched.update(scales)

        for name in matched:
            scale = self.scale_library.get(name)
            if scale:
                suggestions.append(scale)

        return suggestions

    @staticmethod
    def _estimate_time(questions: List[QuestionItem]) -> str:
        seconds = 0
        for q in questions:
            if q.type == "open_ended":
                seconds += 45
            elif q.type in ("single_choice", "multiple_choice"):
                seconds += 10
            elif q.type == "likert":
                seconds += 15
            elif q.type == "ranking":
                seconds += 30
            elif q.type == "matrix":
                seconds += 20
        minutes = max(1, round(seconds / 60))
        return f"约{minutes}分钟"

    @staticmethod
    def _q_to_dict(q: QuestionItem) -> Dict[str, Any]:
        return {
            "question_id": q.question_id,
            "text": q.text,
            "type": q.type,
            "type_description": QuestionnaireEngine._QUESTION_TYPE_TEMPLATES.get(q.type, {}).get("description", ""),
            "options": q.options,
            "required": q.required,
            "condition": q.condition,
            "tags": q.tags,
        }


# ===========================================================================
# G-2: Interview Engine
# ===========================================================================

@dataclass
class InterviewSection:
    section_id: str
    title: str
    description: str
    questions: List[str]
    probing_strategies: List[str] = field(default_factory=list)
    estimated_time: int = 5  # minutes


class InterviewEngine:
    """Generate semi-structured interview protocols."""

    _PROBING_STRATEGIES = [
        "能具体说说吗？",
        "当时是什么情况？",
        "您当时的感受是怎样的？",
        "这件事对您有什么影响？",
        "还有其他相关的经历吗？",
        "您是如何看待这个问题的？",
        "能否举一个具体的例子？",
        "之后发生了什么？",
    ]

    _INTERVIEW_TYPES = {
        "life_history": {
            "name": "生命史访谈",
            "description": "关注个人生命历程中的关键事件和转折点。",
            "sections": ["成长背景", "教育经历", "职业历程", "关键事件", "人生反思"],
        },
        "phenomenological": {
            "name": "现象学访谈",
            "description": "深入探索特定现象在受访者主观经验中的呈现。",
            "sections": ["经验描述", "意义建构", "情感体验", "影响与改变"],
        },
        "grounded": {
            "name": "扎根理论访谈",
            "description": "通过迭代访谈收集数据，逐步构建理论。",
            "sections": ["开放性问题", "概念澄清", "关系探索", "理论饱和检验"],
        },
        "narrative": {
            "name": "叙事访谈",
            "description": "邀请受访者讲述故事，关注叙事结构和身份建构。",
            "sections": ["故事开场", "情节展开", "转折点", "结局与反思"],
        },
        "focused": {
            "name": "焦点访谈",
            "description": "围绕特定研究问题展开结构化但灵活的访谈。",
            "sections": ["背景了解", "核心问题探索", "深入追问", "总结与确认"],
        },
    }

    def generate_protocol(self, args: dict) -> Dict[str, Any]:
        """Generate an interview protocol.

        Args:
            topic: Research topic.
            interview_type: "life_history" | "phenomenological" | "grounded" | "narrative" | "focused"
            target_population: e.g. "大学生", "退休教师".
            research_questions: List of RQs.
            n_questions_per_section: int (default 3).
            estimated_duration: int in minutes (default 60).

        Returns:
            Protocol with sections, questions, probing strategies, and ethics reminders.
        """
        topic = args.get("topic", "")
        itype = args.get("interview_type", "focused")
        target = args.get("target_population", "")
        rqs = args.get("research_questions", [])
        n_per_sec = int(args.get("n_questions_per_section", 3))
        duration = int(args.get("estimated_duration", 60))

        type_info = self._INTERVIEW_TYPES.get(itype, self._INTERVIEW_TYPES["focused"])

        sections = []
        for i, sec_title in enumerate(type_info["sections"], 1):
            questions = self._generate_questions(
                sec_title, topic, target, rqs, n_per_sec
            )
            probing = self._select_probing_strategies(sec_title)
            sections.append({
                "section_id": f"I{i}",
                "title": sec_title,
                "description": f"围绕{sec_title}展开讨论。",
                "questions": questions,
                "probing_strategies": probing,
                "estimated_time": max(5, duration // len(type_info["sections"])),
            })

        return {
            "title": f"《{topic}》{type_info['name']}提纲",
            "interview_type": itype,
            "interview_type_name": type_info["name"],
            "target_population": target,
            "estimated_duration": f"{duration}分钟",
            "sections": sections,
            "general_probing_strategies": self._PROBING_STRATEGIES,
            "ethics_reminders": self._ethics_reminders(),
            "recording_suggestions": self._recording_suggestions(),
        }

    def _generate_questions(
        self, section: str, topic: str, target: str, rqs: List[str], n: int
    ) -> List[str]:
        """Generate questions for a section."""
        templates = {
            "成长背景": [
                "请您简单介绍一下您的成长背景。",
                "在您的成长过程中，有哪些人或事对您产生了重要影响？",
                "您成长的家庭环境是怎样的？",
            ],
            "教育经历": [
                "能谈谈您的受教育经历吗？",
                "在学习过程中，您遇到过哪些挑战？",
                "您觉得教育对您的价值观产生了怎样的影响？",
            ],
            "职业历程": [
                "您是如何进入目前这个职业领域的？",
                "在您的职业生涯中，有哪些重要的转折点？",
                "您对自己的职业发展有什么规划？",
            ],
            "关键事件": [
                "在您的人生中，有没有哪个事件让您印象特别深刻？",
                "能描述一下那个事件的具体经过吗？",
                "那个事件对您产生了怎样的影响？",
            ],
            "人生反思": [
                "回顾过去，您觉得自己最大的变化是什么？",
                "如果可以重来，您会做出不同的选择吗？",
                "您对年轻人有什么建议？",
            ],
            "经验描述": [
                f"关于{topic}，您能否描述一下您的亲身体验？",
                "在那种情境下，您注意到了什么？",
                "您能描述一下当时的环境吗？",
            ],
            "意义建构": [
                "您是如何理解这段经历的？",
                "这件事对您来说意味着什么？",
                "您从中学到了什么？",
            ],
            "情感体验": [
                "当您回忆起这段经历时，您有什么感受？",
                "在那种情况下，您最强烈的情绪是什么？",
                "这种感受对您后来的行为有什么影响？",
            ],
            "影响与改变": [
                "这段经历对您的生活产生了哪些具体的影响？",
                "您因为这个经历而改变了自己的哪些观念或行为？",
                "如果有机会，您希望怎样改变现状？",
            ],
            "开放性问题": [
                f"关于{topic}，您能谈谈您的看法吗？",
                "在这方面，您有过哪些经历？",
                "您认为关键因素是什么？",
            ],
            "概念澄清": [
                "您刚才提到的'XX'，能否再详细解释一下？",
                "在您看来，这个概念包含哪些方面？",
                "不同的人可能有不同的理解，您是怎么看的？",
            ],
            "关系探索": [
                "您觉得这两个因素之间有什么联系？",
                "在您的经验中，它们是如何相互影响的？",
                "有没有出现过矛盾或冲突的情况？",
            ],
            "理论饱和检验": [
                "还有没有我们没有谈到的相关方面？",
                "您还能想到其他例子吗？",
                "您觉得这个解释是否适用于所有情况？",
            ],
            "故事开场": [
                f"请您讲一个关于{topic}的故事。",
                "您最早接触到这件事是什么时候？",
                "能从头开始讲起吗？",
            ],
            "情节展开": [
                "接下来发生了什么？",
                "当时还有谁在场？",
                "您是如何应对的？",
            ],
            "转折点": [
                "有没有哪个时刻让您改变了想法？",
                "事情是在什么时候开始变化的？",
                "那个转折对后续产生了什么影响？",
            ],
            "结局与反思": [
                "这件事最终的结果是什么？",
                "您现在怎么看这件事？",
                "如果重来一次，您会怎么做？",
            ],
            "背景了解": [
                f"在开始之前，能请您简单介绍一下您与{topic}相关的背景吗？",
                "您是如何开始关注这个问题的？",
            ],
            "核心问题探索": [
                f"关于{topic}，您最关心的问题是什么？",
                "在您看来，目前存在哪些主要困难？",
            ],
            "深入追问": [
                "能具体说说当时的情况吗？",
                "您做出这个决定的原因是什么？",
                "其他人对此有什么看法？",
            ],
            "总结与确认": [
                "让我总结一下您刚才说的，您看理解得对吗？",
                "还有什么是您觉得重要但还没有谈到的吗？",
            ],
        }

        sec_questions = templates.get(section, [f"请谈谈您对{topic}的看法。"] * n)
        return sec_questions[:n]

    def _select_probing_strategies(self, section: str) -> List[str]:
        """Select relevant probing strategies for a section."""
        mapping = {
            "成长背景": ["能具体说说吗？", "当时是什么情况？"],
            "经验描述": ["在那种情境下，您注意到了什么？", "您能描述一下当时的环境吗？"],
            "情感体验": ["您当时的感受是怎样的？", "当您回忆起这段经历时，您有什么感受？"],
            "关键事件": ["能否举一个具体的例子？", "之后发生了什么？"],
            "深入追问": ["能具体说说当时的情况吗？", "您做出这个决定的原因是什么？"],
        }
        return mapping.get(section, self._PROBING_STRATEGIES[:3])

    @staticmethod
    def _ethics_reminders() -> List[str]:
        return [
            "访谈前须获得受访者的知情同意。",
            "告知受访者访谈目的、时长、录音/录像安排及数据使用方式。",
            "强调受访者有权随时退出，无需说明理由。",
            "对敏感话题保持敏感，受访者拒绝回答时需尊重。",
            "访谈录音须匿名化处理，妥善保管。",
            "访谈结束后提供致谢或小礼品（如适用）。",
        ]

    @staticmethod
    def _recording_suggestions() -> List[str]:
        return [
            "建议使用高质量录音设备，确保音质清晰。",
            "访谈环境应安静、私密、不受打扰。",
            "准备纸笔做简要笔记，标记重点时段。",
            "每次访谈后24小时内完成转录，确保记忆鲜活。",
            "转录稿须标注非语言信息（停顿、笑声、叹息等）。",
        ]


# ===========================================================================
# G-3: Data Collection Tracker
# ===========================================================================

class DataCollectionTracker:
    """Track data collection progress, response rate, and quality control."""

    def __init__(self):
        self._projects: Dict[str, Dict[str, Any]] = {}

    def create_project(self, args: dict) -> Dict[str, Any]:
        """Create a new data collection tracking project.

        Args:
            project_id: Unique identifier.
            project_name: Human-readable name.
            target_sample_size: int.
            collection_method: "survey" | "interview" | "mixed".
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD.
        """
        pid = args.get("project_id", f"proj_{len(self._projects)+1}")
        project = {
            "project_id": pid,
            "project_name": args.get("project_name", "未命名项目"),
            "target_sample_size": int(args.get("target_sample_size", 100)),
            "collection_method": args.get("collection_method", "survey"),
            "start_date": args.get("start_date", ""),
            "end_date": args.get("end_date", ""),
            "records": [],
            "created_at": "",  # Would use datetime in production
        }
        self._projects[pid] = project
        return {"status": "created", "project": project}

    def add_record(self, args: dict) -> Dict[str, Any]:
        """Add a data collection record.

        Args:
            project_id: Project ID.
            record_id: Unique record ID.
            status: "completed" | "partial" | "refused" | "invalid".
            duration_minutes: float.
            quality_flags: List[str] (e.g. "too_fast", "straight_line").
            notes: str.
        """
        pid = args.get("project_id", "")
        if pid not in self._projects:
            return {"error": f"Project '{pid}' not found."}

        record = {
            "record_id": args.get("record_id", f"rec_{len(self._projects[pid]['records'])+1}"),
            "status": args.get("status", "completed"),
            "duration_minutes": float(args.get("duration_minutes", 0)),
            "quality_flags": args.get("quality_flags", []),
            "notes": args.get("notes", ""),
        }
        self._projects[pid]["records"].append(record)
        return {"status": "record_added", "project_id": pid}

    def get_report(self, args: dict) -> Dict[str, Any]:
        """Get data collection progress report.

        Args:
            project_id: Project ID.
        """
        pid = args.get("project_id", "")
        if pid not in self._projects:
            return {"error": f"Project '{pid}' not found."}

        project = self._projects[pid]
        records = project["records"]
        target = project["target_sample_size"]

        completed = sum(1 for r in records if r["status"] == "completed")
        partial = sum(1 for r in records if r["status"] == "partial")
        refused = sum(1 for r in records if r["status"] == "refused")
        invalid = sum(1 for r in records if r["status"] == "invalid")
        valid = completed + partial

        response_rate = valid / (valid + refused) * 100 if (valid + refused) > 0 else 0
        completion_rate = completed / target * 100 if target > 0 else 0

        # Quality flags
        flag_counts = {}
        for r in records:
            for flag in r.get("quality_flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # Duration stats
        durations = [r["duration_minutes"] for r in records if r["duration_minutes"] > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        too_fast = sum(1 for d in durations if d < 3)

        return {
            "project_id": pid,
            "project_name": project["project_name"],
            "target_sample_size": target,
            "current_valid_n": valid,
            "completed": completed,
            "partial": partial,
            "refused": refused,
            "invalid": invalid,
            "response_rate_pct": round(response_rate, 1),
            "completion_rate_pct": round(completion_rate, 1),
            "quality_flags": flag_counts,
            "avg_duration_minutes": round(avg_duration, 1),
            "too_fast_count": too_fast,
            "progress_bar": f"{valid}/{target}",
            "status": "completed" if valid >= target else "in_progress",
        }

    def list_projects(self) -> List[Dict[str, Any]]:
        return [
            {
                "project_id": p["project_id"],
                "project_name": p["project_name"],
                "target_sample_size": p["target_sample_size"],
                "current_n": len(p["records"]),
            }
            for p in self._projects.values()
        ]


# ===========================================================================
# G-5: Pilot Analyzer
# ===========================================================================

class PilotAnalyzer:
    """Analyze pilot test data for questionnaire quality."""

    def analyze(self, args: dict) -> Dict[str, Any]:
        """Analyze pilot test results.

        Args:
            data: list of dicts (rows=respondents) or result_id.
            item_cols: List of column names for scale items.
            scale_min: int (default 1).
            scale_max: int (default 5).
            reverse_items: List of item indices (1-based) that are reverse-coded.
            item_names: List of str (optional).

        Returns:
            Comprehensive pilot analysis with reliability, difficulty, discrimination,
            and modification suggestions.
        """
        import numpy as np

        raw_data = args.get("data", [])
        item_cols = args.get("item_cols", [])
        scale_min = int(args.get("scale_min", 1))
        scale_max = int(args.get("scale_max", 5))
        reverse_items = args.get("reverse_items", [])
        item_names = args.get("item_names", [])

        if not raw_data or not item_cols:
            return {"error": "data and item_cols are required."}

        # Convert to numpy array
        try:
            if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
                arr = np.array([[row.get(c, np.nan) for c in item_cols] for row in raw_data], dtype=float)
            else:
                arr = np.asarray(raw_data, dtype=float)
        except (TypeError, ValueError):
            return {"error": "data must be numeric."}

        if arr.ndim != 2:
            return {"error": "data must be 2-D (respondents x items)."}

        n_resp, n_items = arr.shape
        if n_resp < 3:
            return {"error": "Pilot needs at least 3 respondents."}

        if not item_names:
            item_names = [f"item_{i+1}" for i in range(n_items)]

        # Apply reverse coding
        for ri in reverse_items:
            idx = ri - 1
            if 0 <= idx < n_items:
                arr[:, idx] = scale_min + scale_max - arr[:, idx]

        # Item-level analysis
        item_stats = []
        for i in range(n_items):
            col = arr[:, i]
            valid = col[~np.isnan(col)]
            mean_val = float(np.nanmean(col))
            std_val = float(np.nanstd(col, ddof=1)) if len(valid) > 1 else 0.0
            # Difficulty (mean / max)
            difficulty = mean_val / scale_max if scale_max != 0 else 0.0
            # Ceiling/floor effects
            ceiling_pct = float(np.sum(valid == scale_max) / len(valid) * 100) if len(valid) > 0 else 0.0
            floor_pct = float(np.sum(valid == scale_min) / len(valid) * 100) if len(valid) > 0 else 0.0

            item_stats.append({
                "item": item_names[i],
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "difficulty": round(difficulty, 2),
                "ceiling_pct": round(ceiling_pct, 1),
                "floor_pct": round(floor_pct, 1),
                "missing_pct": round(np.sum(np.isnan(col)) / n_resp * 100, 1),
            })

        # Total score and reliability
        total_scores = np.nansum(arr, axis=1)
        var_items = np.nanvar(arr, axis=1, ddof=1)
        var_total = np.nanvar(total_scores, ddof=1)

        if var_total > 0 and n_items > 1:
            alpha = float((n_items / (n_items - 1)) * (1 - np.nansum(var_items) / var_total))
        else:
            alpha = None

        # Item-total correlation (corrected)
        item_total_corr = []
        for i in range(n_items):
            remaining_total = total_scores - arr[:, i]
            valid_mask = ~np.isnan(arr[:, i]) & ~np.isnan(remaining_total)
            if np.sum(valid_mask) > 2:
                x = arr[:, i][valid_mask]
                y = remaining_total[valid_mask]
                xm = x - np.mean(x)
                ym = y - np.mean(y)
                num = np.sum(xm * ym)
                den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
                r = float(num / den) if den != 0 else 0.0
            else:
                r = None
            item_total_corr.append(r)

        # Flag problematic items
        suggestions = []
        for i, stat in enumerate(item_stats):
            if stat["ceiling_pct"] > 80:
                suggestions.append({
                    "item": item_names[i],
                    "issue": "ceiling_effect",
                    "detail": f"天花板效应({stat['ceiling_pct']}%选最高分)，建议修改措辞降低极端倾向。",
                })
            if stat["floor_pct"] > 80:
                suggestions.append({
                    "item": item_names[i],
                    "issue": "floor_effect",
                    "detail": f"地板效应({stat['floor_pct']}%选最低分)，建议检查题目难度。",
                })
            if stat["missing_pct"] > 20:
                suggestions.append({
                    "item": item_names[i],
                    "issue": "high_missing",
                    "detail": f"缺失率过高({stat['missing_pct']}%)，建议简化或删除该题。",
                })
            corr = item_total_corr[i]
            if corr is not None and corr < 0.2:
                suggestions.append({
                    "item": item_names[i],
                    "issue": "low_discrimination",
                    "detail": f"题总相关过低(r={corr:.2f})，该题可能与量表测量内容不一致。",
                })

        if alpha is not None and alpha < 0.7:
            suggestions.append({
                "item": "整体",
                "issue": "low_reliability",
                "detail": f"Cronbach's α = {alpha:.2f}，低于0.70的可接受标准，建议检查题目同质性。",
            })

        return {
            "n_respondents": n_resp,
            "n_items": n_items,
            "scale_range": {"min": scale_min, "max": scale_max},
            "cronbach_alpha": round(alpha, 3) if alpha is not None else None,
            "item_statistics": item_stats,
            "item_total_correlation": dict(zip(item_names, [
                round(r, 3) if r is not None else None for r in item_total_corr
            ])),
            "suggestions": suggestions,
            "overall_assessment": self._assess_pilot(alpha, item_stats, suggestions),
        }

    @staticmethod
    def _assess_pilot(alpha, item_stats, suggestions) -> str:
        if alpha is None:
            return "数据不足以计算信度，请增加预测试样本量。"
        if alpha >= 0.9:
            reliability = "优秀"
        elif alpha >= 0.8:
            reliability = "良好"
        elif alpha >= 0.7:
            reliability = "可接受"
        elif alpha >= 0.6:
            reliability = "边缘"
        else:
            reliability = "不足"

        n_issues = len(suggestions)
        if n_issues == 0:
            return f"预测试质量良好。Cronbach's α = {alpha:.2f}，信度{reliability}，无明显问题项。"
        else:
            return f"预测试发现{n_issues}个问题。Cronbach's α = {alpha:.2f}（{reliability}），建议按修改建议调整后再进行正式调查。"
