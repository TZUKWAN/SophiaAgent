"""PaperReader: Extract key elements, annotations, and compare papers.

Supports LLM-based extraction (if provider available) with keyword-based
rule fallback. Uses pymupdf for PDF annotation parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based rule fallback
# ---------------------------------------------------------------------------

_KEYWORD_PATTERNS = {
    "research_question": [
        r"research question[s]?",
        r"研究问题",
        r"研究目的",
        r"研究目标",
        r"本文旨在",
        r"本研究旨在",
        r"we ask",
        r"we investigate",
        r"this paper examines",
        r"this study explores",
    ],
    "theoretical_framework": [
        r"theoretical framework",
        r"理论框架",
        r"理论基础",
        r"based on",
        r"drawing on",
        r"informed by",
        r" grounded in ",
        r"借鉴",
        r"基于",
    ],
    "methods": [
        r"method[s]?",
        r"methodology",
        r"研究方法",
        r"研究设计",
        r"实证方法",
        r"qualitative",
        r"quantitative",
        r"mixed method",
        r"regression",
        r"interview",
        r"survey",
        r"experiment",
        r"case study",
        r"ethnography",
        r"content analysis",
        r"discourse analysis",
    ],
    "data_sources": [
        r"data",
        r"dataset",
        r"sample",
        r"数据来源",
        r"数据",
        r"样本",
        r"调查对象",
        r"we collected",
        r"we used",
        r"drawn from",
        r"来源于",
        r"采集",
    ],
    "main_findings": [
        r"finding[s]?",
        r"result[s]?",
        r"主要发现",
        r"研究发现",
        r"实证结果",
        r"we find",
        r"we found",
        r"our results show",
        r"the results indicate",
        r"结果表明",
        r"结果显示",
    ],
    "limitations": [
        r"limitation[s]?",
        r"局限",
        r"不足之处",
        r"研究局限",
        r"future research",
        r"further research",
        r"不足",
    ],
}


def _extract_by_keywords(text: str) -> Dict[str, Any]:
    """Extract key elements using keyword heuristics."""
    text_lower = text.lower()
    elements: Dict[str, Any] = {}

    for key, patterns in _KEYWORD_PATTERNS.items():
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, text_lower, re.IGNORECASE):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 200)
                snippet = text[start:end].replace("\n", " ").strip()
                matches.append(snippet)
        elements[key] = matches[:3] if matches else []

    # Try to infer sample size
    sample_match = re.search(
        r"(\d{2,6})\s*(?:participants?|subjects?|respondents?|samples?|cases?|observations?|样本|被试|受访者)",
        text_lower,
    )
    elements["sample_size"] = sample_match.group(1) if sample_match else None

    # Try to infer core argument from abstract-like first paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        first = paragraphs[0]
        elements["core_argument"] = first[:500]
    else:
        elements["core_argument"] = ""

    return elements


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are an academic research assistant. Extract the following elements from the paper text below. Return ONLY a valid JSON object with these keys:

- research_question: list of research questions (strings)
- core_arguments: list of core arguments or hypotheses (strings)
- methods: list of methods used (strings)
- data_sources: list of data sources or sample descriptions (strings)
- main_findings: list of main findings or results (strings)
- limitations: list of limitations or future research directions (strings)
- theoretical_framework: list of theories or frameworks referenced (strings)
- sample_size: inferred sample size as a string, or null

Paper text:
{text}

Return JSON only, no markdown fences."""


def _call_llm_for_extraction(provider, text: str) -> Optional[Dict[str, Any]]:
    """Call LLM to extract key elements."""
    if provider is None:
        return None
    try:
        prompt = _EXTRACTION_PROMPT.format(text=text[:8000])
        response = provider.chat(
            [{"role": "user", "content": prompt}],
            tools=None,
        )
        content = response.content or ""
        # Strip markdown fences
        content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        content = re.sub(r"\s*```$", "", content.strip(), flags=re.MULTILINE)
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# PaperReader class
# ---------------------------------------------------------------------------

class PaperReader:
    """Extract key elements, annotations, and compare academic papers."""

    def __init__(self, provider=None):
        self.provider = provider

    # -- internal helpers --------------------------------------------------

    def _json(self, data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, default=str)

    # -- B-1: extract_key_elements -----------------------------------------

    def extract_key_elements(self, text: str) -> Dict[str, Any]:
        """Extract research questions, arguments, methods, findings, etc.

        Uses LLM if provider is available, otherwise keyword-based fallback.
        """
        if not text or not text.strip():
            return {
                "research_question": [],
                "core_arguments": [],
                "methods": [],
                "data_sources": [],
                "main_findings": [],
                "limitations": [],
                "theoretical_framework": [],
                "sample_size": None,
            }

        # Try LLM first
        llm_result = _call_llm_for_extraction(self.provider, text)
        if llm_result is not None:
            # Normalize keys
            normalized = {
                "research_question": llm_result.get("research_question", []),
                "core_arguments": llm_result.get("core_arguments", llm_result.get("core_argument", [])),
                "methods": llm_result.get("methods", []),
                "data_sources": llm_result.get("data_sources", []),
                "main_findings": llm_result.get("main_findings", []),
                "limitations": llm_result.get("limitations", []),
                "theoretical_framework": llm_result.get("theoretical_framework", []),
                "sample_size": llm_result.get("sample_size", None),
            }
            return normalized

        # Fallback: keyword-based extraction
        kw = _extract_by_keywords(text)
        return {
            "research_question": kw.get("research_question", []),
            "core_arguments": [kw.get("core_argument", "")] if kw.get("core_argument") else [],
            "methods": kw.get("methods", []),
            "data_sources": kw.get("data_sources", []),
            "main_findings": kw.get("main_findings", []),
            "limitations": kw.get("limitations", []),
            "theoretical_framework": kw.get("theoretical_framework", []),
            "sample_size": kw.get("sample_size", None),
        }

    # -- B-2: extract_annotations -------------------------------------------

    def extract_annotations(self, pdf_path: str) -> Dict[str, Any]:
        """Extract highlights, underlines, text annotations from a PDF.

        Returns dict with:
            - annotations: list of {page, rect, content, type, surrounding_context, color}
            - color_groups: dict mapping color -> list of annotation indices
            - markdown: str (export)
        """
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}

        try:
            import fitz  # pymupdf
        except ImportError:
            return {"error": "pymupdf (fitz) is required for PDF annotation extraction"}

        annotations: List[Dict[str, Any]] = []
        color_groups: Dict[str, List[int]] = {}

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return {"error": f"Failed to open PDF: {e}"}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            for annot in page.annots() or []:
                annot_type = annot.type[1] if annot.type else "Unknown"
                rect = annot.rect
                color = annot.colors.get("stroke") or annot.colors.get("fill") or annot.colors.get("interior") or None
                color_str = ""
                if color and len(color) >= 3:
                    color_str = "#{:02x}{:02x}{:02x}".format(
                        int(color[0] * 255),
                        int(color[1] * 255),
                        int(color[2] * 255),
                    )

                # Extract text content
                content = ""
                if hasattr(annot, "info") and annot.info:
                    content = annot.info.get("content", "") or ""

                # For highlight/underline, extract highlighted text
                if annot_type in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
                    try:
                        quad_points = annot.vertices
                        if quad_points:
                            highlighted_parts = []
                            for i in range(0, len(quad_points), 4):
                                if i + 3 < len(quad_points):
                                    q = fitz.Quad(quad_points[i:i+4])
                                    highlighted_parts.append(page.get_textbox(q.rect))
                            highlighted_text = " ".join(highlighted_parts).strip()
                            if highlighted_text:
                                content = highlighted_text
                    except Exception:
                        pass

                # surrounding_context: +/-100 chars around annotation rect center
                surrounding = ""
                if page_text:
                    cx = int((rect.x0 + rect.x1) / 2)
                    cy = int((rect.y0 + rect.y1) / 2)
                    # Approximate char position in page text
                    idx = (page_num * 1000) + cx  # rough approximation
                    text_len = len(page_text)
                    start = max(0, idx - 100)
                    end = min(text_len, idx + 100)
                    surrounding = page_text[start:end].strip()

                annot_entry = {
                    "page": page_num + 1,
                    "rect": {
                        "x0": round(rect.x0, 2),
                        "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2),
                        "y1": round(rect.y1, 2),
                    },
                    "content": content,
                    "type": annot_type,
                    "surrounding_context": surrounding,
                    "color": color_str,
                }
                idx = len(annotations)
                annotations.append(annot_entry)

                if color_str:
                    color_groups.setdefault(color_str, []).append(idx)

        doc.close()

        # Build markdown export
        md_lines = [f"# PDF Annotations: {os.path.basename(pdf_path)}", ""]
        md_lines.append(f"Total annotations: {len(annotations)}")
        md_lines.append("")

        if color_groups:
            md_lines.append("## Color Groups")
            for color, indices in sorted(color_groups.items()):
                md_lines.append(f"- {color}: {len(indices)} annotations")
            md_lines.append("")

        for i, a in enumerate(annotations):
            md_lines.append(f"## Annotation {i + 1}")
            md_lines.append(f"- **Page**: {a['page']}")
            md_lines.append(f"- **Type**: {a['type']}")
            md_lines.append(f"- **Color**: {a['color'] or 'N/A'}")
            md_lines.append(f"- **Content**: {a['content']}")
            md_lines.append(f"- **Context**: ...{a['surrounding_context']}...")
            md_lines.append("")

        return {
            "annotations": annotations,
            "color_groups": color_groups,
            "markdown": "\n".join(md_lines),
            "total": len(annotations),
        }

    # -- B-1/B-5: compare_papers -------------------------------------------

    def compare_papers(self, elements_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-paper comparison matrix.

        Dimensions: research_question, theoretical_framework, methods,
        data_sources, sample_size, main_findings, limitations.

        Auto-detects consensus and controversies.
        """
        if not elements_list:
            return {"error": "elements_list is empty"}

        dimensions = [
            "research_question",
            "theoretical_framework",
            "methods",
            "data_sources",
            "sample_size",
            "main_findings",
            "limitations",
        ]

        n = len(elements_list)
        matrix: Dict[str, List[Any]] = {d: [] for d in dimensions}

        for i, elem in enumerate(elements_list):
            for d in dimensions:
                val = elem.get(d)
                if val is None:
                    val = "N/A"
                elif isinstance(val, list):
                    val = "; ".join(str(v) for v in val[:3])
                matrix[d].append({
                    "paper_index": i,
                    "value": val,
                })

        # Consensus detection: simple string overlap heuristic
        consensus: Dict[str, Any] = {}
        controversies: Dict[str, Any] = {}

        for d in dimensions:
            values = [m["value"] for m in matrix[d] if m["value"] != "N/A"]
            if len(values) < 2:
                continue

            # Token overlap for consensus
            token_sets = []
            for v in values:
                tokens = set(re.findall(r"\w+", str(v).lower()))
                token_sets.append(tokens)

            if token_sets:
                intersection = set.intersection(*token_sets)
                union = set.union(*token_sets)
                if union:
                    jaccard = len(intersection) / len(union)
                else:
                    jaccard = 0.0

                if jaccard > 0.3:
                    consensus[d] = {
                        "jaccard": round(jaccard, 3),
                        "shared_terms": list(intersection)[:10],
                        "description": f"Papers show consensus on {d}",
                    }
                elif jaccard < 0.1 and len(values) >= 2:
                    controversies[d] = {
                        "jaccard": round(jaccard, 3),
                        "description": f"Papers diverge significantly on {d}",
                        "values": values,
                    }

        return {
            "matrix": matrix,
            "paper_count": n,
            "dimensions": dimensions,
            "consensus": consensus,
            "controversies": controversies,
        }
