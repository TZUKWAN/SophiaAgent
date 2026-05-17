from pathlib import Path

from sophia.workspace_context import (
    asks_for_paper_document,
    collect_workspace_context,
    iter_workspace_context_events,
    needs_workspace_context,
    save_generated_markdown,
)


def test_detects_workspace_paper_request():
    text = "基于工作空间中的论文，仔细阅读后写论文"

    assert needs_workspace_context(text)
    assert asks_for_paper_document(text)


def test_collect_workspace_context_reads_local_text(tmp_path):
    paper = tmp_path / "generative_ai_chinese_culture.md"
    paper.write_text(
        "Author: Zhang San\nGenerative AI improves Chinese culture communication.",
        encoding="utf-8",
    )

    context = collect_workspace_context(str(tmp_path), "基于工作空间中的论文写一篇文章")

    assert context.requested is True
    assert context.has_evidence
    assert "Generative AI improves Chinese culture communication." in context.to_prompt_block()


def test_collect_workspace_context_reads_all_supported_files(tmp_path):
    for idx in range(12):
        paper = tmp_path / f"paper_{idx:02d}.md"
        paper.write_text(
            f"Real workspace paper {idx}. Generative AI and cultural communication.",
            encoding="utf-8",
        )

    context = collect_workspace_context(
        str(tmp_path),
        "基于工作空间中的论文写一篇文章",
    )

    assert context.total_candidates == 12
    assert len(context.evidences) == 12
    assert context.skipped == []


def test_workspace_context_streams_per_file_events(tmp_path):
    for idx in range(3):
        paper = tmp_path / f"paper_{idx:02d}.md"
        paper.write_text(
            f"Real workspace paper {idx}. Generative AI and cultural communication.",
            encoding="utf-8",
        )

    events = list(iter_workspace_context_events(str(tmp_path), "基于工作空间中的论文写一篇文章"))

    assert events[0]["type"] == "workspace_scan_start"
    assert events[0]["total_files"] == 3
    assert [event["type"] for event in events].count("workspace_file_start") == 3
    assert [event["type"] for event in events].count("workspace_file_done") == 3
    assert events[-1]["type"] == "workspace_context_complete"
    assert len(events[-1]["context"].evidences) == 3


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
