"""Role templates for SophiaAgent's swarm system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class RoleTemplate:
    role_id: str
    name: str
    description: str
    system_prompt: str
    allowed_tools: List[str] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    needs_tools: bool = True


def _role(
    role_id: str,
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[List[str]] = None,
    expertise: Optional[List[str]] = None,
    needs_tools: bool = True,
) -> RoleTemplate:
    return RoleTemplate(
        role_id=role_id,
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=tools or [],
        expertise=expertise or [],
        needs_tools=needs_tools,
    )


_ROLE_TEMPLATES = [
    _role(
        "literature_searcher",
        "文献检索专家",
        "负责关键词拆解、数据库检索、文献筛选和引用线索整理。",
        "你是文献检索专家。优先给出真实可核验的检索策略、关键词组合、核心文献线索、证据等级和不足。不得编造文献。",
        ["literature_search", "ref_search", "web_search", "data_news"],
        ["文献", "综述", "研究", "引用", "参考文献", "检索", "数据库"],
    ),
    _role(
        "data_analyst",
        "数据分析专家",
        "负责数据读取、描述统计、建模、因果推断和可视化建议。",
        "你是数据分析专家。必须区分真实数据、缺失数据和无法验证的数据；报告样本量、变量、方法、限制和可复现步骤。",
        [
            "data_load",
            "data_describe",
            "data_visualize",
            "code_execute",
            "research_regression",
            "research_did",
            "research_plot",
        ],
        ["数据", "统计", "回归", "因果", "可视化", "模型", "样本"],
    ),
    _role(
        "writer",
        "学术写作专家",
        "负责大纲、章节组织、学术表达和最终文本草拟。",
        "你是学术写作专家。输出必须结构清晰、论证连贯、语言正式；不能把未经验证的信息写成事实。",
        [
            "doc_create",
            "doc_outline",
            "doc_write_section",
            "doc_export_markdown",
            "doc_export_docx",
            "doc_export_latex",
            "doc_export_pdf",
            "doc_pipeline_status",
        ],
        ["写", "撰写", "论文", "报告", "专著", "大纲", "章节"],
    ),
    _role(
        "reviewer",
        "学术评审专家",
        "负责真实性、逻辑、引用、语言、统计和伦理六维审查。",
        "你是学术评审专家。优先发现风险、漏洞、缺证据处和需要返工的地方，并给出可执行修改建议。",
        ["doc_auto_review", "doc_review_dimension", "doc_revise_from_review"],
        ["评审", "审查", "质量", "修改", "审核", "问题"],
    ),
    _role(
        "methodologist",
        "方法论专家",
        "负责研究设计、方法选择、适用条件和替代方案。",
        "你是方法论专家。必须说明方法选择依据、前提假设、适用边界、局限和替代方法。",
        ["methodology_advise", "research_design", "research_power_analysis"],
        ["方法", "研究设计", "实验", "调查", "访谈", "混合研究", "方案"],
    ),
    _role(
        "critic",
        "逻辑批判专家",
        "负责发现论证漏洞、矛盾、弱证据和过度推断。",
        "你是逻辑批判专家。你的职责是指出不成立、不充分、不一致或证据不足的地方，并解释原因。",
        [],
        ["逻辑", "批判", "矛盾", "漏洞", "证据", "风险"],
        needs_tools=False,
    ),
    _role(
        "synthesizer",
        "综合汇总专家",
        "负责整合多方结论、解决冲突并形成统一答复。",
        "你是综合汇总专家。必须保留各专家的可靠贡献，标明冲突和不确定性，形成简洁但完整的最终意见。",
        [],
        ["综合", "汇总", "整合", "总结", "结论"],
        needs_tools=False,
    ),
    _role(
        "citation_manager",
        "引用管理专家",
        "负责参考文献格式、文内引用和引用一致性检查。",
        "你是引用管理专家。不得伪造 DOI、作者、期刊或年份；无法核验时必须标注需要检索确认。",
        ["ref_add", "ref_list", "ref_format", "ref_search", "ref_add_relation"],
        ["引用", "参考文献", "APA", "GB/T", "BibTeX"],
    ),
]


class RoleTemplateBank:
    """Registry and matcher for static and dynamic swarm roles."""

    def __init__(self, templates: Optional[Iterable[RoleTemplate]] = None):
        source = list(templates) if templates is not None else _ROLE_TEMPLATES
        self._templates: Dict[str, RoleTemplate] = {role.role_id: role for role in source}

    def get(self, role_id: str) -> Optional[RoleTemplate]:
        return self._templates.get(role_id)

    def list_all(self) -> List[RoleTemplate]:
        return list(self._templates.values())

    def list_ids(self) -> List[str]:
        return sorted(self._templates)

    def register(self, template: RoleTemplate) -> None:
        self._templates[template.role_id] = template

    def create_dynamic_role(
        self,
        role_id: str,
        name: str,
        description: str,
        system_prompt: str,
        allowed_tools: Optional[List[str]] = None,
        expertise: Optional[List[str]] = None,
    ) -> RoleTemplate:
        template = RoleTemplate(
            role_id=role_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools or [],
            expertise=expertise or [],
        )
        self.register(template)
        return template

    def match_for_task(self, task_description: str, limit: Optional[int] = None) -> List[RoleTemplate]:
        text = task_description.lower()
        scored: List[tuple[int, RoleTemplate]] = []
        for role in self._templates.values():
            score = 0
            for keyword in role.expertise:
                if keyword.lower() in text:
                    score += 2
            if role.role_id in text:
                score += 3
            if score:
                scored.append((score, role))
        scored.sort(key=lambda item: item[0], reverse=True)
        roles = [role for _, role in scored]
        return roles[:limit] if limit else roles
