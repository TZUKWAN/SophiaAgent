from pathlib import Path

from sophia.workspace_context import (
    asks_for_paper_document,
    collect_workspace_context,
    needs_workspace_context,
    save_generated_markdown,
)


def test_detects_workspace_paper_request():
    text = "基于工作空间中的论文，仔细阅读后写论文"

    assert needs_workspace_context(text)
    assert asks_for_paper_document(text)


def test_collect_workspace_context_reads_local_text(tmp_path):
    paper = tmp_path / "生成式人工智能与中华文化国际传播.md"
    paper.write_text("作者：张三\n生成式人工智能提升中华文化传播效率。", encoding="utf-8")

    context = collect_workspace_context(str(tmp_path), "基于工作空间中的论文写一篇文章")

    assert context.requested is True
    assert context.has_evidence
    assert "生成式人工智能提升中华文化传播效率" in context.to_prompt_block()


def test_save_generated_markdown_for_paper_request(tmp_path):
    path = save_generated_markdown(
        str(tmp_path),
        "生成式人工智能语境下中华文化国际传播的机遇、挑战与路径这个论文",
        "# 论文标题\n正文",
    )

    assert path is not None
    saved = Path(path)
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "# 论文标题\n正文"
