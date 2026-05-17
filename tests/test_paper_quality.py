from sophia.paper_quality import (
    MIN_BODY_CHARS,
    append_quality_report_if_needed,
    build_paper_generation_contract,
    inspect_generated_paper,
)


def test_contract_injected_for_paper_request():
    contract = build_paper_generation_contract("帮我写一篇论文")
    assert "6500" in contract
    assert "20 real references" in contract
    assert "5 tables" in contract
    assert "8 figures" in contract


def test_short_paper_fails_quality_gate():
    content = """# 论文

正文很短。

表 1 示例

| A | B |
|---|---|
| 1 | 2 |

图 1 示例

## 参考文献

[1] 张三. 生成式人工智能研究. 学术期刊, 2024.
"""
    report = inspect_generated_paper(content)
    assert not report.passed
    assert report.body_chars < MIN_BODY_CHARS
    assert report.reference_count == 1
    assert report.table_count == 1
    assert report.figure_count == 1


def test_append_quality_report_if_needed_marks_failure():
    content = "# 论文\n\n正文很短。"
    out = append_quality_report_if_needed("写一篇论文", content)
    assert "论文质量自检：未达标" in out
    assert "参考文献" in out
