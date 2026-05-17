"""Paper generation quality contracts and checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
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
2. If the user did not provide references, first prompt the user to provide
   their preferred reference list or workspace papers.
3. Only when the user has no references available, or explicitly asks Sophia
   to search independently, use literature_search, web_search, citation tools,
   or other verifiable sources to supplement references.
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
3. The paper must contain at least 5 tables. Tables must be meaningful for the
   argument, such as concept comparison, literature matrix, mechanism table,
   challenge table, path table, variable table, evidence table, or policy table.
4. The paper must contain at least 8 figures. If an actual image file cannot be
   generated, provide a clearly titled figure block with a Mermaid diagram or a
   precise image-generation/data-visualization instruction that can be rendered
   later. Do not count decorative text as a figure.
5. Framework or architecture diagrams must be clear. Labels must be large and
   readable. Prefer simple node names, high contrast, and no dense tiny text.
6. Data visualizations must be one chart per figure. Do not combine many
   unrelated visualizations into one image. Each chart needs its own title,
   data source note, and interpretation.
7. Academic prose must be formal, rigorous, and progressive. Each section
   should advance the argument step by step.
8. Avoid the phrase pattern "不是...而是...". Give direct judgments.
9. Avoid unnecessary quotation marks, colons, dashes, and long sentences.
   Prefer shorter sentences with clear subject, predicate, and object.
10. Do not add heading levels the user forbids. If the user asks for no third
    level headings, use only first-level and second-level headings.
11. Before the final answer ends, internally verify the minimum length,
    reference count, table count, figure count, citation consistency, and
    language style. If any item cannot be satisfied because evidence is missing,
    say exactly what is missing instead of fabricating it.
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
    write_terms = ["写", "生成", "撰写", "成文", "write", "draft", "generate"]
    paper_terms = ["论文", "文章", "paper", "article", "文稿"]
    return any(term in lowered for term in write_terms) and any(
        term in lowered for term in paper_terms
    )


def build_paper_generation_contract(user_message: str) -> str:
    if not is_paper_generation_request(user_message):
        return ""
    return PAPER_GENERATION_CONTRACT


def has_user_supplied_references(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in ["参考文献如下", "参考文献：", "references:", "bibliography"]):
        return True
    numbered_refs = re.findall(r"(?m)^\s*(\[\d+\]|\d+[\.\)]|\-\s+).{12,}", text)
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
            "Use search only to verify, complete metadata, or supplement gaps after the user-provided "
            "references have been used."
        )
    if workspace_has_evidence:
        return (
            "[Reference priority notice]\n"
            "Workspace literature has been read. Prioritize the workspace papers and documents. "
            "Use search only to verify metadata or supplement unavoidable gaps."
        )
    return (
        "[Reference priority notice]\n"
        "Before independently searching for references, ask the user to provide their preferred "
        "reference list or workspace papers. If the user has no references available or explicitly "
        "asks Sophia to search, then use literature_search, web_search, citation tools, or other "
        "verifiable sources. Do not fabricate references."
    )


def inspect_generated_paper(content: str) -> PaperQualityReport:
    body, refs = _split_body_and_references(content)
    body_chars = _count_body_chars(body)
    reference_count = _count_references(refs)
    table_count = _count_tables(body)
    figure_count = _count_figures(body)

    issues: List[str] = []
    if body_chars < MIN_BODY_CHARS:
        issues.append(f"正文长度不足：{body_chars} 字，最低要求 {MIN_BODY_CHARS} 字。")
    if reference_count < MIN_REFERENCES:
        issues.append(f"参考文献不足：{reference_count} 条，最低要求 {MIN_REFERENCES} 条。")
    if table_count < MIN_TABLES:
        issues.append(f"表格不足：{table_count} 个，最低要求 {MIN_TABLES} 个。")
    if figure_count < MIN_FIGURES:
        issues.append(f"图片或图示不足：{figure_count} 张，最低要求 {MIN_FIGURES} 张。")
    if re.search(r"不是.{0,20}而是", body):
        issues.append("正文仍包含“不是...而是...”式转折，需要改为直接判断。")

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
    lines.append("请继续扩写、补足真实参考文献、补足表格和图示后再视为完成。")
    return content.rstrip() + "\n" + "\n".join(lines)


def quality_report_dict(content: str) -> Dict:
    return asdict(inspect_generated_paper(content))


def _split_body_and_references(content: str) -> tuple[str, str]:
    pattern = re.compile(r"(?im)^#{1,6}\s*(参考文献|references)\s*$|^(参考文献|References)\s*$")
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
        if re.match(r"^(\[\d+\]|\d+[\.\)]|\-\s+)", line):
            count += 1
        elif re.search(r"\(\d{4}[a-z]?\)|\d{4}", line) and len(line) >= 18:
            count += 1
    return count


def _count_tables(text: str) -> int:
    table_captions = len(re.findall(r"(?im)^(表\s*\d+|Table\s*\d+)", text))
    markdown_tables = len(re.findall(r"(?m)^\s*\|.+\|\s*$\n^\s*\|[\s:\-\|]+\|\s*$", text))
    return max(table_captions, markdown_tables)


def _count_figures(text: str) -> int:
    markdown_images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))
    figure_captions = len(re.findall(r"(?im)^(图\s*\d+|Figure\s*\d+)", text))
    mermaid_blocks = len(re.findall(r"```mermaid", text, flags=re.I))
    return max(figure_captions, markdown_images + mermaid_blocks)
