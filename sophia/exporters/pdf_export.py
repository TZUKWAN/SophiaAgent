"""PDF export dispatcher for SophiaAgent.

Routes to LaTeX compilation (xelatex) or falls back to DOCX-to-PDF.
"""

import logging
from typing import Dict

from sophia.exporters.latex_export import export_latex

logger = logging.getLogger(__name__)


def export_pdf(doc: Dict, output_dir: str, engine: str = "xelatex") -> Dict:
    """Export a document to PDF.

    Strategy:
    1. Render .tex via Jinja2 template
    2. Compile with xelatex (supports Chinese)

    Args:
        doc: Document dict
        output_dir: Directory for output files
        engine: LaTeX engine

    Returns:
        Dict with pdf_path or error.
    """
    result = export_latex(doc, output_dir, compile_pdf=True, engine=engine)

    if result.get("compiled"):
        return {
            "format": "pdf",
            "pdf_path": result["pdf_path"],
            "tex_path": result["tex_path"],
        }

    # Compilation failed - return error with details
    return {
        "format": "pdf",
        "tex_path": result.get("tex_path", ""),
        "compiled": False,
        "error": result.get("error", "Unknown compilation error"),
        "log": result.get("log", ""),
        "suggestion": "Install TeX Live or MiKTeX with xelatex support for PDF export.",
    }
