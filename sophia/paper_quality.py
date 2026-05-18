"""Paper generation quality contracts and checks."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

MIN_BODY_CHARS = 6500
MIN_REFERENCES = 20
MIN_TABLES = 5
MIN_FIGURES = 8


PAPER_GENERATION_CONTRACT = """\
[Sophia paper generation hard contract]
The user is asking for an academic paper or article. You must write the full
paper body in this answer, not a short outline and not a placeholder.

Reference priority:
1. If the user provided references, uploaded papers, or workspace literature,
   prioritize those sources and cite them first.
2. If the user did not provide references, first remind the user that Sophia can
   use their preferred references, then continue with workspace evidence or
   verifiable search when the current task already asks Sophia to complete the
   paper now. Do not stop and push the work back to the user.
3. Only when the user has no references available, or explicitly asks Sophia to
   search independently, use literature_search, web_search, citation tools, or
   other verifiable sources to supplement references.
4. Never replace user-provided references with searched references unless the
   user-provided source is unusable, duplicated, or unverifiable. Explain that
   limitation clearly.

Minimum deliverables:
1. Body length must be at least 6500 Chinese characters or equivalent academic
   prose, excluding the reference list.
2. The reference list must contain at least 20 real references. Use only
   references from workspace evidence, citation tools, literature tools, or
   otherwise verifiable sources. Never invent authors, years, journal names,
   DOI values, page numbers, or statistical facts.
3. The paper must contain at least 5 tables and 8 figures/diagrams for empirical
   papers. Purely theoretical papers may reduce or omit tables and figures,
   replacing them with conceptual diagrams, argument flowcharts, or literature
   matrices as needed.
4. Tables must be meaningful for the argument, such as concept comparison,
   literature matrix, mechanism table, challenge table, path table, variable
   table, evidence table, or policy table.
5. Figures must be substantive. If an actual image file cannot be generated,
   provide a clearly titled figure block with a Mermaid diagram or a precise
   image-generation/data-visualization instruction that can be rendered later.
   Do not count decorative text as a figure. Framework or architecture diagrams
   must be clear with large readable labels, simple node names, and high contrast.
6. Data visualizations must be one chart per figure. Do not combine many
   unrelated visualizations into one image. Each chart needs its own title,
   data source note, and interpretation.

Body text style rules:
7. Write in continuous paragraphs. Do NOT use bullet points, numbered lists,
   or sub-headings within the body text. Section-level headings (e.g., "引言",
   "文献综述") are allowed; mini-headings inside paragraphs are not.
8. The first sentence of each paragraph must be a concise summary of that
   paragraph's core argument. Keep opening sentences roughly consistent in
   length across paragraphs for visual rhythm. Maintain consistent grammatical
   structures across opening sentences where possible.
9. Academic prose must be formal, rigorous, and progressive. Each section
   should advance the argument step by step. Prefer short direct sentences.
10. High theoretical depth with elegant expression. Use precise professional
    terminology, but avoid obscure neologisms or coined concepts.

Banned patterns (NEVER use in body text):
11. Mechanical connectors: "首先", "其次", "再次", "最后", "第一", "第二",
    "第三", "其一", "其二".
12. Hype words: "重构", "重建", "填补空白", "颠覆", "开创性", "里程碑",
    "划时代", "前所未有", "重大突破".
13. Rhetorical questions: "如何", "何以", "为何", "为什么", "怎能", "岂能",
    "何尝".
14. Forced contrast patterns: "不是...而是...", "并非...而是...",
    "与其说...不如说...".
15. Unnecessary quotation marks, colons, dashes, and ellipsis strung together.
    Avoid decorative punctuation.
16. Do not add heading levels the user forbids. If the user asks for no third
    level headings, use only first-level and second-level headings.
17. Before the final answer ends, internally verify the minimum length,
    reference count, table count, figure count, citation consistency, and
    language style. If any item is below the threshold, continue expanding,
    adding verified references, and adding meaningful tables or figures before
    finalizing. Only report an irreducible evidence gap after attempting
    self-remediation with all available workspace and tool evidence.
18. If the user asks for Word/DOCX, the final deliverable must be exported as
    Word/DOCX. Do not treat Markdown as the final requested deliverable.
"""


@dataclass
class PaperQualityReport:
    body_chars: int
    reference_count: int
    table_count: int
    figure_count: int
    passed: bool
    issues: List[str]


def is_paper_generation_request(text: str) -> bool:
    lowered = text.lower()
    write_terms = [
        "写",
        "撰写",
        "生成",
        "起草",
        "成文",
        "write",
        "draft",
        "generate",
    ]
    paper_terms = [
        "论文",
        "文章",
        "研究报告",
        "综述",
        "paper",
        "article",
        "manuscript",
        "review",
    ]
    return any(term in lowered for term in write_terms) and any(
        term in lowered for term in paper_terms
    )


def build_paper_generation_contract(user_message: str) -> str:
    if not is_paper_generation_request(user_message):
        return ""
    return PAPER_GENERATION_CONTRACT


def has_user_supplied_references(text: str) -> bool:
    lowered = text.lower()
    reference_headings = [
        "参考文献如下",
        "参考文献：",
        "参考文献:",
        "references:",
        "bibliography",
    ]
    if any(term in lowered for term in reference_headings):
        return True
    numbered_refs = re.findall(r"(?m)^\s*(\[\d+\]|\d+[\.\)]|-\s+).{12,}", text)
    if len(numbered_refs) >= 2:
        return True
    return bool(re.search(r"(?m).{2,}(\(\d{4}[a-z]?\)|\d{4}).{8,}", text))


def build_reference_priority_notice(
    user_message: str,
    *,
    workspace_has_evidence: bool = False,
) -> str:
    if not is_paper_generation_request(user_message):
        return ""
    if has_user_supplied_references(user_message):
        return (
            "[Reference priority notice]\n"
            "The user has supplied references in the request. Prioritize these references. "
            "Use search only to verify, complete metadata, or supplement gaps after the "
            "user-provided references have been used."
        )
    if workspace_has_evidence:
        return (
            "[Reference priority notice]\n"
            "Workspace literature has been read. Prioritize the workspace papers and "
            "documents. Use search only to verify metadata or supplement unavoidable gaps."
        )
    return (
        "[Reference priority notice]\n"
        "Before independently searching for references, tell the user they may provide "
        "preferred references. "
        "If the current request asks Sophia to complete the paper now and no references were "
        "provided, continue using workspace evidence first and verifiable search second. "
        "Do not fabricate references and do not stop merely to ask the user to do the work."
    )


# Style rule check patterns
_BANNED_WORDS = [
    ("首先", "机械连接词"),
    ("其次", "机械连接词"),
    ("再次", "机械连接词"),
    ("最后", "机械连接词"),
    ("第一", "机械连接词"),
    ("第二", "机械连接词"),
    ("第三", "机械连接词"),
    ("其一", "机械连接词"),
    ("其二", "机械连接词"),
    ("重构", "夸张吹嘘词"),
    ("重建", "夸张吹嘘词"),
    ("填补空白", "夸张吹嘘词"),
    ("颠覆", "夸张吹嘘词"),
    ("开创性", "夸张吹嘘词"),
    ("里程碑", "夸张吹嘘词"),
    ("划时代", "夸张吹嘘词"),
    ("前所未有", "夸张吹嘘词"),
    ("重大突破", "夸张吹嘘词"),
    ("如何", "反问词"),
    ("何以", "反问词"),
    ("为何", "反问词"),
    ("为什么", "反问词"),
    ("怎能", "反问词"),
    ("岂能", "反问词"),
    ("何尝", "反问词"),
]

_BULLET_LIST_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[\.、)）]\s+")
_FORCED_CONTRAST_RE = re.compile(r"不是.{0,20}而是|并非.{0,20}而是|与其说.{0,20}不如说")
_RHETORICAL_QUESTION_RE = re.compile(r"[如何何以为何为什么怎能岂能何尝].{0,15}[?？]")


def _is_theoretical_paper(body: str) -> bool:
    """Heuristic: detect if paper is purely theoretical (no empirical analysis)."""
    empirical_keywords = [
        "实证", "回归", "did", "difference-in-differences", "双重差分",
        "面板数据", "问卷", "调查", "实验", "t检验", "方差分析", "anova",
        "回归分析", "描述统计", "中介效应", "调节效应", "结构方程",
        "数据", "样本", "变量", "系数", "显著性", "p值",
    ]
    lower_body = body.lower()
    empirical_hits = sum(1 for kw in empirical_keywords if kw in lower_body)
    return empirical_hits < 3


def _detect_banned_words(body: str) -> List[str]:
    issues = []
    for word, category in _BANNED_WORDS:
        if word in body:
            issues.append(f"检测到禁用词「{word}」（{category}），请删除或替换。")
    return issues


def _detect_forced_contrast(body: str) -> List[str]:
    issues = []
    for m in _FORCED_CONTRAST_RE.finditer(body):
        issues.append(
            f"检测到强制转折句式「{m.group()}」，请改为直接判断。"
        )
    return issues


def _detect_bullet_lists(body: str) -> List[str]:
    issues = []
    bullet_count = len(_BULLET_LIST_RE.findall(body))
    numbered_count = len(_NUMBERED_LIST_RE.findall(body))
    if bullet_count > 0:
        issues.append(f"检测到 {bullet_count} 处无序列表（项目符号），正文应使用段落化文本。")
    if numbered_count > 0:
        issues.append(f"检测到 {numbered_count} 处有序列表（编号），正文应使用段落化文本。")
    return issues


def _detect_rhetorical_questions(body: str) -> List[str]:
    issues = []
    for m in _RHETORICAL_QUESTION_RE.finditer(body):
        issues.append(f"检测到反问句「{m.group()}」，请改为直接陈述。")
    return issues[:3]  # Limit to avoid flooding


def inspect_generated_paper(content: str) -> PaperQualityReport:
    body, refs = _split_body_and_references(content)
    body_chars = _count_body_chars(body)
    reference_count = _count_references(refs)
    table_count = _count_tables(body)
    figure_count = _count_figures(body)
    is_theoretical = _is_theoretical_paper(body)

    issues: List[str] = []
    if body_chars < MIN_BODY_CHARS:
        issues.append(f"正文长度不足：{body_chars} 字，最低要求 {MIN_BODY_CHARS} 字。")
    if reference_count < MIN_REFERENCES:
        issues.append(f"参考文献不足：{reference_count} 条，最低要求 {MIN_REFERENCES} 条。")

    # Table/figure requirements: relaxed for purely theoretical papers
    if is_theoretical:
        if table_count < 2:
            issues.append(f"纯理论论文表格不足：{table_count} 个，建议至少 2 个（如概念比较表、文献矩阵）。")
        if figure_count < 2:
            issues.append(f"纯理论论文图示不足：{figure_count} 张，建议至少 2 个（如论证流程图、概念框架图）。")
    else:
        if table_count < MIN_TABLES:
            issues.append(f"表格不足：{table_count} 个，最低要求 {MIN_TABLES} 个。")
        if figure_count < MIN_FIGURES:
            issues.append(f"图片或图示不足：{figure_count} 张，最低要求 {MIN_FIGURES} 张。")

    # Style checks
    issues.extend(_detect_banned_words(body))
    issues.extend(_detect_forced_contrast(body))
    issues.extend(_detect_bullet_lists(body))
    issues.extend(_detect_rhetorical_questions(body))

    return PaperQualityReport(
        body_chars=body_chars,
        reference_count=reference_count,
        table_count=table_count,
        figure_count=figure_count,
        passed=not issues,
        issues=issues,
    )


def append_quality_report_if_needed(user_message: str, content: str) -> str:
    if not is_paper_generation_request(user_message) or not content.strip():
        return content
    report = inspect_generated_paper(content)
    if report.passed:
        return content
    lines = [
        "\n\n---",
        "论文质量自检：未达标",
        f"- 正文字数：{report.body_chars}，最低要求 {MIN_BODY_CHARS}",
        f"- 参考文献：{report.reference_count}，最低要求 {MIN_REFERENCES}",
        f"- 表格数量：{report.table_count}，最低要求 {MIN_TABLES}",
        f"- 图片或图示数量：{report.figure_count}，最低要求 {MIN_FIGURES}",
    ]
    lines.extend(f"- {issue}" for issue in report.issues)
    lines.append("Sophia 应继续扩写、补足真实参考文献、表格和图示后再视为完成。")
    return content.rstrip() + "\n" + "\n".join(lines)


def quality_report_dict(content: str) -> Dict:
    return asdict(inspect_generated_paper(content))


def _split_body_and_references(content: str) -> tuple[str, str]:
    pattern = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(参考文献|references)\s*$")
    match = pattern.search(content)
    if not match:
        return content, ""
    return content[: match.start()], content[match.end() :]


def _count_body_chars(text: str) -> int:
    cleaned = re.sub(r"```.*?```", "", text, flags=re.S)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\|.*\|", "", cleaned)
    cjk = re.findall(r"[\u4e00-\u9fff]", cleaned)
    latin_words = re.findall(r"[A-Za-z]+", cleaned)
    return len(cjk) + sum(max(1, len(word) // 5) for word in latin_words)


def _count_references(refs: str) -> int:
    if not refs.strip():
        return 0
    lines = [line.strip() for line in refs.splitlines() if line.strip()]
    count = 0
    for line in lines:
        if re.match(r"^(\[\d+\]|\d+[\.\)]|-\s+)", line):
            count += 1
        elif re.search(r"\(\d{4}[a-z]?\)|\d{4}", line) and len(line) >= 18:
            count += 1
    return count


def _count_tables(text: str) -> int:
    table_captions = len(re.findall(r"(?im)^\s*(表|table)\s*\d+", text))
    markdown_tables = len(re.findall(r"(?m)^\s*\|.+\|\s*$\n^\s*\|[\s:\-\|]+\|\s*$", text))
    return max(table_captions, markdown_tables)


def _count_figures(text: str) -> int:
    markdown_images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))
    figure_captions = len(re.findall(r"(?im)^\s*(图|figure)\s*\d+", text))
    mermaid_blocks = len(re.findall(r"```mermaid", text, flags=re.I))
    return max(figure_captions, markdown_images + mermaid_blocks)
