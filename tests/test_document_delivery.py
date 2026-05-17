from pathlib import Path

from docx import Document

from sophia.document_delivery import requested_output_format, save_generated_docx


def test_requested_output_format_detects_word_chinese_and_english():
    assert requested_output_format("请生成 Word 文档") == "docx"
    assert requested_output_format("export as .docx") == "docx"
    assert requested_output_format("请生成 markdown") == "markdown"


def test_save_generated_docx_creates_word_file_for_paper_request(tmp_path):
    content = """# 生成式人工智能论文

第一章 引言

这是一段正文。

表 1 文献矩阵

| 维度 | 内容 |
|---|---|
| 主题 | 文化传播 |

图 1 研究框架

```mermaid
graph TD
  A[技术] --> B[传播]
```

参考文献

1. Wang, L. (2024). Generative AI and culture. Journal of Communication.
"""
    path = save_generated_docx(str(tmp_path), "请写一篇论文并生成 Word 文档", content)

    assert path is not None
    saved = Path(path)
    assert saved.exists()
    assert saved.suffix == ".docx"
    doc = Document(str(saved))
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    assert "生成式人工智能论文" in text
    assert "第一章 引言" in text
