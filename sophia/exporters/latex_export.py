"""LaTeX export for SophiaAgent.

Renders document projects into .tex files using Jinja2 templates.
Optionally compiles to PDF via xelatex/latexmk.
"""

import logging
import os
import subprocess
from datetime import datetime
from typing import Dict

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

LATEX_SPECIAL = {
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in user-provided text."""
    if not text:
        return text
    for char, replacement in LATEX_SPECIAL.items():
        text = text.replace(char, replacement)
    return text


TEMPLATE_MAP = {
    "paper": "paper.tex.j2",
    "report": "report.tex.j2",
    "monograph": "monograph.tex.j2",
    "grant-nsfc": "grant_nsfc.tex.j2",
    "grant-nssfc": "grant_nssfc.tex.j2",
    "grant-moe": "grant_moe.tex.j2",
}


def _prepare_context(doc: Dict) -> Dict:
    """Prepare Jinja2 context from document data."""
    sections = []
    for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x)):
        section = doc["sections"][key]
        sections.append({
            "title": _escape_latex(section.get("title", "")),
            "content": section.get("content", ""),
        })

    return {
        "title": _escape_latex(doc.get("title", "")),
        "authors": _escape_latex(doc.get("authors", "")),
        "abstract": doc.get("abstract", ""),
        "keywords": doc.get("keywords", []),
        "date": doc.get("updated_at", datetime.now().strftime("%Y年%m月")),
        "sections": sections,
        "references": doc.get("references", []),
    }


def export_latex(
    doc: Dict, output_dir: str,
    compile_pdf: bool = False, engine: str = "xelatex",
) -> Dict:
    """Export a document to LaTeX (.tex) and optionally compile to PDF.

    Args:
        doc: Document dict from _load_doc()
        output_dir: Directory to write output files
        compile_pdf: Whether to compile to PDF
        engine: LaTeX engine (xelatex, pdflatex, lualatex)

    Returns:
        Dict with output paths and status.
    """
    doc_type = doc.get("doc_type", "paper")
    template_name = TEMPLATE_MAP.get(doc_type, "paper.tex.j2")

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    env.filters["escape_latex"] = _escape_latex
    template = env.get_template(template_name)

    context = _prepare_context(doc)
    tex_content = template.render(**context)

    os.makedirs(output_dir, exist_ok=True)

    doc_id = doc.get("id", "doc")
    tex_path = os.path.join(output_dir, f"{doc_id}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    result = {
        "format": "latex",
        "tex_path": tex_path,
        "chars": len(tex_content),
    }

    if compile_pdf:
        pdf_result = _compile_pdf(tex_path, output_dir, engine)
        result.update(pdf_result)

    return result


def _compile_pdf(tex_path: str, output_dir: str, engine: str = "xelatex") -> Dict:
    """Compile a .tex file to PDF.

    Returns:
        Dict with pdf_path or error.
    """
    tex_filename = os.path.basename(tex_path)

    try:
        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output_dir}",
            tex_filename,
        ]

        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        pdf_filename = tex_filename.replace(".tex", ".pdf")
        pdf_path = os.path.join(output_dir, pdf_filename)

        if os.path.exists(pdf_path):
            return {
                "pdf_path": pdf_path,
                "compiled": True,
                "log": result.stdout[-500:] if result.stdout else "",
            }
        else:
            return {
                "compiled": False,
                "error": "PDF not generated",
                "log": result.stdout[-1000:] if result.stdout else result.stderr[-1000:],
            }
    except FileNotFoundError:
        return {
            "compiled": False,
            "error": f"LaTeX engine '{engine}' not found. Install TeX Live or MiKTeX.",
        }
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "error": "LaTeX compilation timed out (120s)",
        }
    except Exception as e:
        return {
            "compiled": False,
            "error": str(e),
        }
