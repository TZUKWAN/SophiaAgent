"""DOCX export for SophiaAgent.

Exports document projects to Word (.docx) format.
Uses the production-grade DOCXEngine for high-fidelity output.
"""

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def export_docx(doc: Dict, output_path: str, result_store=None, citation_style: str = "apa7") -> Dict:
    """Export a document to DOCX format.

    Args:
        doc: Document dict from _load_doc()
        output_path: Path to write the .docx file
        result_store: Optional ResultStore for embedding research results
        citation_style: "apa7" or "gb-t-7714-2015"

    Returns:
        Dict with output path and status.
    """
    try:
        from sophia.exporters.docx_engine import DOCXEngine
    except ImportError as e:
        return {
            "format": "docx",
            "error": f"DOCX export requires python-docx. {e}",
        }

    engine = DOCXEngine(result_store=result_store)
    return engine.export_paper(
        doc=doc,
        output_path=output_path,
        citation_style=citation_style,
        include_results=True,
    )
