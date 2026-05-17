from sophia.paper_quality import (
    MIN_BODY_CHARS,
    append_quality_report_if_needed,
    build_paper_generation_contract,
    build_reference_priority_notice,
    has_user_supplied_references,
    inspect_generated_paper,
    is_paper_generation_request,
)


def test_contract_injected_for_paper_request():
    contract = build_paper_generation_contract("Please write a full paper")
    assert "6500" in contract
    assert "20 real references" in contract
    assert "5 tables" in contract
    assert "8 figures" in contract
    assert "Reference priority" in contract


def test_detects_chinese_paper_request():
    assert is_paper_generation_request("请基于工作空间论文，写一篇生成式人工智能综述论文")


def test_short_paper_fails_quality_gate():
    content = """# Paper

This body is very short.

Table 1 Example

| A | B |
|---|---|
| 1 | 2 |

Figure 1 Example

## References
[1] Smith, J. (2024). Generative AI and culture. Journal of Culture.
"""
    report = inspect_generated_paper(content)
    assert not report.passed
    assert report.body_chars < MIN_BODY_CHARS
    assert report.reference_count == 1
    assert report.table_count == 1
    assert report.figure_count == 1


def test_append_quality_report_if_needed_marks_failure():
    content = "# Paper\n\nThis body is very short."
    out = append_quality_report_if_needed("write a paper", content)
    assert out != content
    assert "6500" in out
    assert "20" in out


def test_detects_user_supplied_references():
    text = """Please write a paper.

References:
1. Smith, J. (2024). Generative AI and culture. Journal of Culture.
2. Wang, L. (2023). International communication. Communication Review.
"""
    assert has_user_supplied_references(text)


def test_reference_notice_prioritizes_supplied_references():
    notice = build_reference_priority_notice(
        "write a paper. References: Smith (2024). Culture and AI.",
    )
    assert "user has supplied references" in notice
    assert "Prioritize these references" in notice


def test_reference_notice_asks_before_search_when_missing_references():
    notice = build_reference_priority_notice("write a paper about culture and AI")
    assert "Before independently searching for references" in notice
    assert "Do not fabricate references" in notice


def test_reference_notice_prioritizes_workspace_literature():
    notice = build_reference_priority_notice(
        "write a paper about culture and AI",
        workspace_has_evidence=True,
    )
    assert "Workspace literature has been read" in notice
    assert "Prioritize the workspace papers" in notice


def test_counts_chinese_reference_heading_tables_and_figures():
    content = "# 论文\n\n"
    content += "正文" * 4000
    content += "\n\n" + "\n\n".join(f"表 {idx} 表格标题" for idx in range(1, 6))
    content += "\n\n" + "\n\n".join(f"图 {idx} 图示标题" for idx in range(1, 9))
    content += "\n\n参考文献\n\n"
    content += "\n".join(
        f"{idx}. Author, A. ({2000 + idx}). Real article title. Journal Name."
        for idx in range(1, 21)
    )

    report = inspect_generated_paper(content)

    assert report.body_chars >= MIN_BODY_CHARS
    assert report.reference_count == 20
    assert report.table_count == 5
    assert report.figure_count == 8
    assert report.passed
