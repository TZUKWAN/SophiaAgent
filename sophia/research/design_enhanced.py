"""Enhanced research design support (Phase H): templates, mixed methods, quality assessment."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# H-1: Research Design Templates
# ---------------------------------------------------------------------------

_DESIGN_TEMPLATES = {
    "experimental": {
        "name": "实验研究设计",
        "description": "通过操纵自变量、控制无关变量来检验因果关系。",
        "sections": [
            {"title": "研究问题与假设", "content": "明确自变量、因变量和假设关系。"},
            {"title": "实验设计类型", "content": "选择：前实验/真实验/准实验；被试间/被试内/混合设计。"},
            {"title": "被试招募与分组", "content": "抽样方法、随机化方案、分组逻辑。"},
            {"title": "变量操作化", "content": "自变量操纵方式、因变量测量指标、控制变量清单。"},
            {"title": "实验流程", "content": "前测-干预-后测的时间线和具体步骤。"},
            {"title": "数据分析计划", "content": "统计方法、效应量指标、检验力分析。"},
            {"title": "伦理考量", "content": "知情同意、退出机制、风险最小化。"},
        ],
        "common_methods": ["t-test", "ANOVA", "MANOVA", "回归分析"],
        "quality_criteria": ["随机化质量", "组间基线可比性", "干预保真度", " attrition率", "效应量报告"],
    },
    "quasi_experimental": {
        "name": "准实验研究设计",
        "description": "在无法完全随机化的情况下，利用自然发生的分组或时间变化推断因果。",
        "sections": [
            {"title": "研究问题", "content": "明确因果推断的假设和识别策略。"},
            {"title": "识别策略", "content": "选择：双重差分(DiD)、断点回归(RDD)、合成控制(SCM)、工具变量(IV)。"},
            {"title": "数据来源与样本", "content": "面板数据/截面数据、样本量、时间跨度。"},
            {"title": "处理变量定义", "content": "处理组与对照组的划分标准、处理时点。"},
            {"title": "平行趋势/前提假设检验", "content": "根据识别策略选择对应的假设检验。"},
            {"title": "稳健性检验", "content": "安慰剂检验、置换检验、改变样本/模型设定。"},
            {"title": "局限与推广性", "content": "内部有效性和外部有效性的权衡。"},
        ],
        "common_methods": ["DiD", "RDD", "SCM", "IV", "PSM"],
        "quality_criteria": ["识别假设可检验性", "对照组可比性", "稳健性检验充分性", "异质性分析"],
    },
    "survey": {
        "name": "调查研究设计",
        "description": "通过问卷或结构化访谈收集数据，描述现状或检验变量间关系。",
        "sections": [
            {"title": "研究问题", "content": "描述性、相关性或因果推断（截面/纵向）。"},
            {"title": "抽样设计", "content": "概率抽样/非概率抽样、样本量计算、抽样框。"},
            {"title": "测量工具", "content": "自编问卷/成熟量表、信效度检验计划。"},
            {"title": "数据收集", "content": "线上/线下、匿名性、质量控制。"},
            {"title": "数据分析", "content": "描述统计、推断统计、结构方程模型等。"},
        ],
        "common_methods": ["描述统计", "相关分析", "回归分析", "SEM", "因子分析"],
        "quality_criteria": ["抽样代表性", "问卷信效度", "无应答率控制", "共同方法偏差检验"],
    },
    "qualitative": {
        "name": "质性研究设计",
        "description": "通过深度访谈、参与观察、文本分析等方法理解社会现象的意义。",
        "sections": [
            {"title": "研究问题", "content": "探索性、描述性或解释性问题。"},
            {"title": "研究范式", "content": "选择：现象学、扎根理论、民族志、案例研究、叙事研究。"},
            {"title": "参与者选择", "content": "目的性抽样策略、样本量（理论饱和）。"},
            {"title": "数据收集", "content": "访谈/观察/文献、半结构化工具、伦理审批。"},
            {"title": "数据分析", "content": "编码策略、主题分析/扎根理论流程。"},
            {"title": "可信度策略", "content": "三角验证、成员检验、反思日志、同行审核。"},
        ],
        "common_methods": ["深度访谈", "参与观察", "焦点小组", "文本分析", "主题分析", "扎根理论"],
        "quality_criteria": ["可信度", "可转移性", "可靠性", "可确认性"],
    },
    "mixed_methods": {
        "name": "混合方法研究设计",
        "description": "在同一研究中整合定量和定性方法，实现方法三角验证。",
        "sections": [
            {"title": "研究问题", "content": "明确需要混合方法才能回答的复合问题。"},
            {"title": "设计类型", "content": "选择：解释性顺序、探索性顺序、聚敛式并行。"},
            {"title": "定量部分", "content": "设计类型、假设、样本、测量、分析计划。"},
            {"title": "定性部分", "content": "范式、参与者、数据收集、分析策略。"},
            {"title": "整合策略", "content": "何时/如何整合两种数据、整合点、权重分配。"},
            {"title": "元推论", "content": "如何从定量+定性结果形成整体结论。"},
        ],
        "common_methods": ["问卷+访谈", "实验+焦点小组", "量化民族志"],
        "quality_criteria": ["设计整合质量", "两种方法各自质量", "元推论合理性", "方法三角验证"],
    },
    "longitudinal": {
        "name": "纵向研究设计",
        "description": "在不同时间点对同一组被试进行重复测量，追踪变化趋势。",
        "sections": [
            {"title": "研究问题", "content": "变化轨迹、因果关系时序、发展规律。"},
            {"title": "设计类型", "content": "趋势研究/队列研究/面板研究。"},
            {"title": "时间规划", "content": "波次数量、间隔、总跨度。"},
            {"title": "样本维持", "content": " attrition 应对策略、追踪机制。"},
            {"title": "分析方法", "content": "增长曲线模型、交叉滞后面板模型、潜在转变分析。"},
        ],
        "common_methods": ["LGM", "CLPM", "LTA", "GEE"],
        "quality_criteria": ["attrition率控制", "测量不变性", "时间间隔合理性", "样本量充足性"],
    },
    "case_study": {
        "name": "案例研究设计",
        "description": "对单个或多个案例进行深度分析，理解复杂现象的情境性特征。",
        "sections": [
            {"title": "研究问题", "content": "'怎么样'和'为什么'类问题。"},
            {"title": "案例选择", "content": "单案例/多案例、典型/极端/关键/便捷案例。"},
            {"title": "理论角色", "content": "理论构建/理论检验/理论扩展。"},
            {"title": "数据来源", "content": "档案、访谈、观察、实物等多源证据。"},
            {"title": "分析策略", "content": "模式匹配、解释构建、时序分析。"},
        ],
        "common_methods": ["Yin案例研究", "Stake案例研究", "过程追踪"],
        "quality_criteria": ["构念效度", "内部效度", "外部效度", "可靠性"],
    },
    "action_research": {
        "name": "行动研究设计",
        "description": "研究者与实践者合作，在实践中发现问题、实施干预、反思改进。",
        "sections": [
            {"title": "研究情境", "content": "实践场所、参与者角色、合作关系。"},
            {"title": "问题诊断", "content": "通过观察/访谈/数据收集识别实践问题。"},
            {"title": "行动规划", "content": "干预方案设计、预期效果、实施步骤。"},
            {"title": "行动实施", "content": "干预执行过程、调整记录。"},
            {"title": "效果评估", "content": "数据收集、结果分析、效果判断。"},
            {"title": "反思与推广", "content": "经验总结、理论提炼、情境适用性讨论。"},
        ],
        "common_methods": ["参与式行动研究", "变革性行动研究", "实践性行动研究"],
        "quality_criteria": ["民主性", "实践导向", "协作质量", "反思深度", "可持续性"],
    },
}


# ---------------------------------------------------------------------------
# H-3: Mixed Methods Design Types
# ---------------------------------------------------------------------------

_MIXED_METHODS_DESIGNS = {
    "convergent_parallel": {
        "name": "聚敛式并行设计",
        "name_en": "Convergent Parallel Design",
        "description": "定量与定性数据同时收集、独立分析，然后在解释阶段整合。",
        "strengths": ["效率较高", "可相互验证", "互补性强"],
        "weaknesses": ["需要足够资源同时开展两部分", "整合时可能出现矛盾", "样本可能不同"],
        "integration_points": ["数据收集后", "分析阶段", "结果解释阶段"],
        "priority": "equal",
        "timing": "concurrent",
    },
    "explanatory_sequential": {
        "name": "解释性顺序设计",
        "name_en": "Explanatory Sequential Design",
        "description": "先定量后定性，用定性数据解释或深化定量结果。",
        "strengths": ["定量结果指导定性抽样", "解释机制清晰", "适合验证后探索"],
        "weaknesses": ["总耗时较长", "两阶段样本衔接可能困难", "定量阶段无法修正"],
        "integration_points": ["定性抽样依据", "结果解释阶段"],
        "priority": "quantitative",
        "timing": "sequential",
    },
    "exploratory_sequential": {
        "name": "探索性顺序设计",
        "name_en": "Exploratory Sequential Design",
        "description": "先定性后定量，用定性发现构建测量工具或假设，再定量检验。",
        "strengths": ["避免定量工具缺乏本土效度", "适合新领域探索", "工具开发有据可依"],
        "weaknesses": ["总耗时较长", "定性阶段可能遗漏重要变量", "推广性不确定"],
        "integration_points": ["工具开发阶段", "假设构建阶段", "结果验证阶段"],
        "priority": "qualitative",
        "timing": "sequential",
    },
    "embedded": {
        "name": "嵌入式设计",
        "name_en": "Embedded Design",
        "description": "一种方法为主、另一种为辅，辅助数据嵌入主要框架中。",
        "strengths": ["资源集中", "主次分明", "可在主要设计中补充细节"],
        "weaknesses": ["辅助数据可能分析不充分", "整合深度受限", "可能沦为点缀"],
        "integration_points": ["数据收集嵌入", "结果报告嵌入"],
        "priority": "depends",
        "timing": "concurrent_or_sequential",
    },
    "transformative": {
        "name": "变革性设计",
        "name_en": "Transformative Design",
        "description": "以社会正义或弱势群体赋权为核心框架，整合定量和定性方法。",
        "strengths": ["关注边缘群体", "具有行动导向", "理论框架明确"],
        "weaknesses": ["政治敏感性高", "研究者立场需要反思", "推广性争议"],
        "integration_points": ["理论框架贯穿", "数据收集与分析", "行动与传播"],
        "priority": "equal",
        "timing": "flexible",
    },
}


class EnhancedDesignEngine:
    """Research design templates, mixed methods support, and quality assessment."""

    def get_design_template(self, args: dict) -> Dict[str, Any]:
        """H-1: Get a research design template.

        Args:
            design_type: str — "experimental" | "quasi_experimental" | "survey" | "qualitative" | "mixed_methods" | "longitudinal" | "case_study" | "action_research"
            research_question: str (optional)
            discipline: str (optional)
        """
        dtype = args.get("design_type", "survey")
        template = _DESIGN_TEMPLATES.get(dtype)
        if not template:
            return {
                "error": f"Unknown design type '{dtype}'.",
                "available": list(_DESIGN_TEMPLATES.keys()),
            }

        result = dict(template)
        result["design_type"] = dtype

        # Add discipline-specific notes
        discipline = args.get("discipline", "")
        if discipline:
            result["discipline_notes"] = self._discipline_notes(dtype, discipline)

        # Add research question alignment check
        rq = args.get("research_question", "")
        if rq:
            result["rq_alignment"] = self._check_rq_alignment(dtype, rq)

        return result

    def list_design_types(self) -> List[Dict[str, Any]]:
        """List all available design types."""
        return [
            {"design_type": k, "name": v["name"], "description": v["description"]}
            for k, v in _DESIGN_TEMPLATES.items()
        ]

    def mixed_methods_design(self, args: dict) -> Dict[str, Any]:
        """H-3: Generate a mixed methods research design.

        Args:
            design_subtype: str — "convergent_parallel" | "explanatory_sequential" | "exploratory_sequential" | "embedded" | "transformative"
            research_question: str
            quantitative_focus: str
            qualitative_focus: str
            integration_strategy: str (optional)
        """
        subtype = args.get("design_subtype", "convergent_parallel")
        design_info = _MIXED_METHODS_DESIGNS.get(subtype)
        if not design_info:
            return {
                "error": f"Unknown mixed methods subtype '{subtype}'.",
                "available": list(_MIXED_METHODS_DESIGNS.keys()),
            }

        result = {
            "design_subtype": subtype,
            **design_info,
            "research_question": args.get("research_question", ""),
            "quantitative_focus": args.get("quantitative_focus", ""),
            "qualitative_focus": args.get("qualitative_focus", ""),
            "integration_strategy": args.get("integration_strategy", "数据三角验证"),
            "visual_diagram": self._mixed_methods_diagram(subtype),
            "quality_checklist": self._mixed_methods_quality_checklist(subtype),
        }

        # Add quantitative and qualitative design templates
        result["quantitative_component"] = self._get_component_design("quantitative", args)
        result["qualitative_component"] = self._get_component_design("qualitative", args)

        return result

    def assess_design_quality(self, args: dict) -> Dict[str, Any]:
        """H-4: Assess research design quality.

        Args:
            design_type: str
            research_question: str
            sections: list of dict with 'title' and 'content'
            methods: list of str
            sample_size: int
            has_control_group: bool
            has_randomization: bool
            has_pretest: bool
            has_ethics_approval: bool
            has_power_analysis: bool
            has_pilot: bool
        """
        dtype = args.get("design_type", "")
        rq = args.get("research_question", "")
        sections = args.get("sections", [])
        methods = args.get("methods", [])
        n = int(args.get("sample_size", 0))

        scores = {}
        findings = []

        # 1. 研究问题清晰度
        rq_score = 20 if len(rq) > 20 else 10 if rq else 0
        if "怎么样" in rq or "为什么" in rq or "什么" in rq:
            rq_score = min(20, rq_score + 5)
        scores["research_question_clarity"] = rq_score

        # 2. 方法匹配度
        template = _DESIGN_TEMPLATES.get(dtype)
        method_score = 0
        if template:
            common_methods = template.get("common_methods", [])
            matched = sum(1 for m in methods if any(cm in m for cm in common_methods))
            method_score = min(20, matched * 10)
            if method_score < 10:
                findings.append({
                    "dimension": "方法匹配",
                    "issue": "所选方法与推荐方法差异较大",
                    "suggestion": f"建议考虑：{', '.join(common_methods[:3])}",
                })
        scores["method_match"] = method_score

        # 3. 内部效度
        internal_score = 0
        if args.get("has_control_group", False):
            internal_score += 8
        if args.get("has_randomization", False):
            internal_score += 8
        if args.get("has_pretest", False):
            internal_score += 4
        scores["internal_validity"] = min(20, internal_score)
        if internal_score < 10 and dtype in ("experimental", "quasi_experimental"):
            findings.append({
                "dimension": "内部效度",
                "issue": f"内部效度控制不足（控制组/随机化/前测缺失）",
                "suggestion": "实验设计应至少包含对照组或前测，以增强因果推断可信度。",
            })

        # 4. 统计严谨性
        stat_score = 0
        if args.get("has_power_analysis", False):
            stat_score += 10
        if args.get("has_pilot", False):
            stat_score += 5
        if n >= 30:
            stat_score += 5
        scores["statistical_rigor"] = min(20, stat_score)
        if not args.get("has_power_analysis", False) and n > 0:
            findings.append({
                "dimension": "统计严谨性",
                "issue": "缺少检验力分析",
                "suggestion": "建议报告先验检验力分析，确保样本量足以检测预期效应。",
            })

        # 5. 伦理合规
        ethics_score = 20 if args.get("has_ethics_approval", False) else 10
        scores["ethics_compliance"] = ethics_score
        if not args.get("has_ethics_approval", False):
            findings.append({
                "dimension": "伦理合规",
                "issue": "未明确伦理审批",
                "suggestion": "涉及人类被试的研究须获得伦理委员会批准（IRB/Ethics Review）。",
            })

        total = sum(scores.values())
        if total >= 85:
            grade = "A"
        elif total >= 65:
            grade = "B"
        elif total >= 45:
            grade = "C"
        else:
            grade = "D"

        return {
            "total_score": total,
            "grade": grade,
            "max_score": 100,
            "breakdown": scores,
            "findings": findings,
            "recommendations": self._design_recommendations(scores, findings),
        }

    def check_method_fit(self, args: dict) -> Dict[str, Any]:
        """H-2: Check if a specific method fits the research design.

        Args:
            research_question: str
            design_type: str
            proposed_method: str
            data_description: dict
        """
        rq = args.get("research_question", "").lower()
        dtype = args.get("design_type", "")
        method = args.get("proposed_method", "").lower()
        dd = args.get("data_description", {})

        fit_score = 0.5
        reasons = []
        concerns = []

        # Design-method match
        template = _DESIGN_TEMPLATES.get(dtype)
        if template:
            common = [m.lower() for m in template.get("common_methods", [])]
            if any(method in cm or cm in method for cm in common):
                fit_score += 0.3
                reasons.append(f"{method}是{dtype}设计的常用方法")
            else:
                concerns.append(f"{method}通常不用于{dtype}设计，常用方法包括：{', '.join(common[:3])}")

        # RQ-method match
        causal_keywords = ["effect", "impact", "因果", "影响", "效应"]
        desc_keywords = ["describe", "分布", "现状", "描述"]
        rel_keywords = ["relationship", "correlation", "相关", "关系"]

        if any(kw in rq for kw in causal_keywords):
            if method in ("did", "rdd", "iv", "scm", "实验"):
                fit_score += 0.2
                reasons.append("研究问题涉及因果推断，所选方法适合")
            else:
                concerns.append("研究问题涉及因果推断，但所选方法可能无法有效识别因果")
        elif any(kw in rq for kw in desc_keywords):
            if method in ("描述统计", "调查", "频次分析"):
                fit_score += 0.2
                reasons.append("研究问题为描述性，所选方法适合")

        # Data-method match
        n = dd.get("N", dd.get("n", 0))
        if isinstance(n, str):
            try:
                n = int(n)
            except ValueError:
                n = 0
        if n > 0 and n < 30 and any(m in method for m in ("回归", "anova", "结构方程")):
            concerns.append(f"样本量(n={n})较小，所选方法可能检验力不足")
            fit_score -= 0.15

        fit_score = max(0.0, min(1.0, fit_score))

        if fit_score >= 0.8:
            fit_level = "高度匹配"
        elif fit_score >= 0.6:
            fit_level = "基本匹配"
        elif fit_score >= 0.4:
            fit_level = "部分匹配"
        else:
            fit_level = "匹配度低"

        return {
            "proposed_method": method,
            "design_type": dtype,
            "fit_score": round(fit_score, 2),
            "fit_level": fit_level,
            "reasons": reasons,
            "concerns": concerns,
            "suggestions": concerns,  # Alias for downstream use
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discipline_notes(design_type: str, discipline: str) -> str:
        notes = {
            ("experimental", "教育学"): "教育实验常面临随机化困难，可考虑准实验设计或整群随机化。",
            ("experimental", "心理学"): "心理学实验注意控制 demand characteristics 和 experimenter bias。",
            ("qualitative", "社会学"): "社会学质性研究注意研究者位置性(positionality)的反思。",
            ("qualitative", "人类学"): "民族志研究需要长期田野工作和文化沉浸。",
            ("survey", "政治学"): "政治调查注意敏感问题的措辞和匿名性保障。",
            ("mixed_methods", "教育学"): "教育混合方法研究常用解释性顺序设计，先用量化发现问题再用质性解释。",
        }
        return notes.get((design_type, discipline), "")

    @staticmethod
    def _check_rq_alignment(design_type: str, rq: str) -> Dict[str, Any]:
        rq_lower = rq.lower()
        issues = []
        suggestions = []

        if design_type == "experimental":
            if not any(k in rq_lower for k in ("effect", "影响", "效应", "作用")):
                issues.append("实验设计通常用于检验因果效应，但研究问题未明确体现因果假设")
                suggestions.append("建议明确自变量和因变量，或使用'...对...的影响'句式")
        elif design_type == "qualitative":
            if any(k in rq_lower for k in ("多少", "比例", "显著", "p值")):
                issues.append("质性研究不适合回答量化问题")
                suggestions.append("建议改为探索性、解释性或意义建构类问题")
        elif design_type == "case_study":
            if any(k in rq_lower for k in ("普遍", "总体", "一般")):
                issues.append("案例研究不以统计推广为目标")
                suggestions.append("建议聚焦特定情境的深入理解，或明确理论推广逻辑")

        return {
            "aligned": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
        }

    @staticmethod
    def _mixed_methods_diagram(subtype: str) -> str:
        diagrams = {
            "convergent_parallel": """
[定量数据收集] ──→ [定量分析] ──┐
                                ├──→ [整合解释]
[定性数据收集] ──→ [定性分析] ──┘
""",
            "explanatory_sequential": """
[定量数据收集] ──→ [定量分析] ──→ [定性数据收集] ──→ [定性分析] ──→ [整合解释]
""",
            "exploratory_sequential": """
[定性数据收集] ──→ [定性分析] ──→ [工具开发/假设构建] ──→ [定量数据收集] ──→ [定量分析]
""",
        }
        return diagrams.get(subtype, "混合方法设计图")

    @staticmethod
    def _mixed_methods_quality_checklist(subtype: str) -> List[str]:
        common = [
            "定量部分和定性部分各自满足该范式的质量标准",
            "整合策略在设计阶段就已明确",
            "两种数据在分析或解释阶段确实进行了整合",
            "元推论（meta-inference）合理，未超出数据支撑范围",
        ]
        specific = {
            "convergent_parallel": [
                "两种数据的样本是否匹配或具有可比性",
                "如何处理定量与定性结果不一致的情况",
            ],
            "explanatory_sequential": [
                "定性抽样是否基于定量结果进行目的性选择",
                "定性部分是否有效解释了定量发现的异常值",
            ],
            "exploratory_sequential": [
                "从质性发现到量化工具的转换是否保留了原意",
                "量化检验的假设是否充分反映了质性发现",
            ],
        }
        return common + specific.get(subtype, [])

    @staticmethod
    def _get_component_design(component: str, args: dict) -> Dict[str, Any]:
        if component == "quantitative":
            return {
                "design_type": args.get("quantitative_design", "survey"),
                "sample_size": args.get("quantitative_n", "待确定"),
                "measurement": args.get("quantitative_measurement", "量表/问卷"),
                "analysis": args.get("quantitative_analysis", "描述统计+推断统计"),
            }
        else:
            return {
                "design_type": args.get("qualitative_design", "phenomenological"),
                "sample_size": args.get("qualitative_n", "待确定"),
                "data_collection": args.get("qualitative_data_collection", "半结构化访谈"),
                "analysis": args.get("qualitative_analysis", "主题分析"),
            }

    @staticmethod
    def _design_recommendations(scores: Dict[str, int], findings: List[Dict]) -> List[str]:
        recs = []
        if scores.get("internal_validity", 0) < 15:
            recs.append("考虑增加对照组、随机化或前测以提升内部效度")
        if scores.get("statistical_rigor", 0) < 15:
            recs.append("补充检验力分析或预测试，确保统计分析的可靠性")
        if scores.get("ethics_compliance", 0) < 20:
            recs.append("申请伦理审批，完善知情同意流程")
        if len(findings) > 3:
            recs.append("研究设计存在多个薄弱环节，建议重新评估核心设计决策")
        if not recs:
            recs.append("研究设计整体良好，可按计划推进")
        return recs
