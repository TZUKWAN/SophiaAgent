from sophia.paper_quality import (
    MIN_BODY_CHARS,
    append_quality_report_if_needed,
    build_paper_generation_contract,
    build_reference_priority_notice,
    has_user_supplied_references,
    inspect_generated_paper,
)


def test_contract_injected_for_paper_request():
    contract = build_paper_generation_contract("Please write a full paper")
    assert "6500" in contract
    assert "20 real references" in contract
    assert "5 tables" in contract
    assert "8 figures" in contract
    assert "Reference priority" in contract


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
