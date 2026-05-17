"""Peer review pipeline for SophiaAgent.

Two review modes:
1. Automated six-dimension review (authenticity, logic, citations, language, statistics, ethics)
   — runs rule-based checkers, no LLM required, instant.
2. LLM peer review (positive / critical / balanced)
   — deep qualitative critique using 5-dimension scoring matrix.

Also supports:
- Four-level feedback classification (Fatal / Major / Minor / Suggestion)
- Chair synthesis
- Review -> Revise loop
"""

import json
import logging
from typing import Any, Dict

from sophia.review.engine import ReviewEngine

logger = logging.getLogger(__name__)

SCORING_DIMENSIONS = {
    "problem_validity": {
        "label": "问题有效性", "weight": 0.20,
        "criteria": "研究问题是否明确、有价值、有理论或现实意义",
    },
    "method_soundness": {
        "label": "方法合理性", "weight": 0.25,
        "criteria": "研究方法是否适合研究问题，设计是否严谨",
    },
    "evidence_adequacy": {
        "label": "证据充分性", "weight": 0.20,
        "criteria": "数据或证据是否充分支撑结论",
    },
    "novelty": {
        "label": "创新性", "weight": 0.20,
        "criteria": "研究是否提供新的理论视角、方法或发现",
    },
    "reproducibility": {
        "label": "可复现性", "weight": 0.15,
        "criteria": "研究过程和结果是否可以被独立验证",
    },
}

SEVERITY_LEVELS = ["fatal", "major", "minor", "suggestion"]

SEVERITY_LABELS = {
    "fatal": "致命问题",
    "major": "重要问题",
    "minor": "次要问题",
    "suggestion": "改进建议",
}


def doc_review(args: Dict[str, Any], workspace: str) -> str:
    """Generate a structured peer review for a document.

    Args: {id: str, perspective: "positive"|"critical"|"balanced"}
    """
    from sophia.tools.writing import _load_doc

    doc_id = args.get("id", "")
    perspective = args.get("perspective", "balanced")

    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    # Extract full text
    sections = doc.get("sections", {})
    full_text_parts = []
    for key in sorted(sections.keys(), key=lambda x: int(x)):
        s = sections[key]
        title = s.get("title", "")
        content = s.get("content", "")
        if content:
            full_text_parts.append(f"## {title}\n{content}")

    if not full_text_parts:
        return json.dumps(
            {"error": "Document has no written content to review"},
            ensure_ascii=False,
        )

    full_text = "\n\n".join(full_text_parts)

    # Structure the review output (to be filled by LLM via exec mode)
    review_template = {
        "document_id": doc_id,
        "document_title": doc.get("title", ""),
        "perspective": perspective,
        "dimensions": {},
        "overall_score": 0,
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "detailed_comments": [],
        "recommendation": "",
    }

    for dim_key, dim_info in SCORING_DIMENSIONS.items():
        review_template["dimensions"][dim_key] = {
            "label": dim_info["label"],
            "weight": dim_info["weight"],
            "criteria": dim_info["criteria"],
            "score": None,
            "comment": "",
        }

    return json.dumps({
        "action": "review_template",
        "review": review_template,
        "full_text_length": len(full_text),
        "sections_reviewed": len(full_text_parts),
        "instructions": (
            f"Use this template to review the document '{doc.get('title', '')}'. "
            f"Perspective: {perspective}. "
            "Score each dimension 1-10, add comments per dimension, "
            "list strengths/weaknesses with severity levels (fatal/major/minor/suggestion), "
            "provide overall recommendation (accept/revise/reject). "
            "Full text follows below for your analysis:\n\n" + full_text[:8000]
        ),
    }, ensure_ascii=False)


def doc_review_summary(args: Dict[str, Any], workspace: str) -> str:
    """Save a completed review to the document metadata.

    Args: {id: str, review: dict}
    """
    from sophia.tools.writing import _load_doc, _save_doc

    doc_id = args.get("id", "")
    review = args.get("review", {})

    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    reviews = doc.get("reviews", [])
    reviews.append(review)
    doc["reviews"] = reviews
    _save_doc(workspace, doc)

    return json.dumps({
        "action": "review_saved",
        "id": doc_id,
        "total_reviews": len(reviews),
    }, ensure_ascii=False)


def register_review_tools(registry, workspace: str):
    """Register peer review tools."""
    from functools import partial

    registry.register(
        name="doc_review",
        description=(
            "Generate a structured peer review template for an academic document. "
            "The review uses a five-dimension scoring matrix: "
            "problem validity (20%), method soundness (25%), evidence adequacy (20%), "
            "novelty (20%), reproducibility (15%). "
            "Feedback is classified as fatal/major/minor/suggestion. "
            "Perspectives: 'positive', 'critical', or 'balanced'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID to review"},
                "perspective": {
                    "type": "string",
                    "default": "balanced",
                    "enum": ["positive", "critical", "balanced"],
                    "description": "Review perspective",
                },
            },
            "required": ["id"],
        },
        handler=partial(doc_review, workspace=workspace),
    )

    registry.register(
        name="doc_review_save",
        description="Save a completed review to a document's metadata.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "review": {
                    "type": "object",
                    "description": "Completed review object with scores and comments",
                },
            },
            "required": ["id", "review"],
        },
        handler=partial(doc_review_summary, workspace=workspace),
    )

    def _systematic_review(args: Dict[str, Any], workspace: str) -> str:
        """Start a PRISMA 2020 systematic review."""
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"}, ensure_ascii=False)

        databases = args.get("databases", ["semantic_scholar", "crossref"])
        year_from = args.get("year_from")
        year_to = args.get("year_to")

        inclusion = args.get("inclusion_criteria", [])
        exclusion = args.get("exclusion_criteria", [])

        review_data = {
            "action": "systematic_review_started",
            "query": query,
            "databases": databases,
            "year_range": [year_from, year_to],
            "inclusion_criteria": inclusion,
            "exclusion_criteria": exclusion,
            "prisma_stages": {
                "identification": {"status": "pending", "count": None,
                    "description": "Records identified through database searching"},
                "screening": {"status": "pending", "count": None,
                    "description": "Records screened (title/abstract)"},
                "full_text": {"status": "pending", "count": None,
                    "description": "Full-text articles assessed for eligibility"},
                "included": {"status": "pending", "count": None,
                    "description": "Studies included in synthesis"},
                "excluded_duplicates": 0,
                "excluded_screening": 0,
                "excluded_full_text": 0,
            },
            "instructions": (
                f"Conduct a systematic review following PRISMA 2020 guidelines. "
                f"Search query: '{query}'. "
                f"Databases: {databases}. "
                f"Year range: {year_from}-{year_to}. "
                f"Inclusion criteria: {inclusion}. "
                f"Exclusion criteria: {exclusion}. "
                "Step 1: Use literature_search to find records. "
                "Step 2: Screen titles/abstracts against inclusion criteria. "
                "Step 3: Assess full-text eligibility. "
                "Step 4: Report PRISMA flow numbers at each stage."
            ),
        }

        return json.dumps(review_data, ensure_ascii=False)

    registry.register(
        name="systematic_review",
        description=(
            "Start a PRISMA 2020 systematic review. "
            "Defines search strategy, inclusion/exclusion criteria, "
            "and generates PRISMA flow diagram data. "
            "The LLM should follow the 4-stage PRISMA workflow: "
            "identification -> screening -> eligibility -> inclusion."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Systematic review search query"},
                "databases": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["semantic_scholar", "arxiv", "crossref"]},
                    "default": ["semantic_scholar", "crossref"],
                    "description": "Databases to search",
                },
                "year_from": {"type": "integer", "description": "Start year"},
                "year_to": {"type": "integer", "description": "End year"},
                "inclusion_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Inclusion criteria",
                },
                "exclusion_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclusion criteria",
                },
            },
            "required": ["query"],
        },
        handler=partial(_systematic_review, workspace=workspace),
    )

    # ------------------------------------------------------------------
    # Automated six-dimension review engine
    # ------------------------------------------------------------------

    def _doc_auto_review(args: Dict[str, Any], workspace: str) -> str:
        """Run automated six-dimension review on a document.

        Args: {id: str, citation_style: "apa7"|"gb-t-7714-2015", dimensions: list}
        """
        from sophia.tools.writing import _load_doc

        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)

        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

        citation_style = args.get("citation_style", "apa7")
        dimensions = args.get("dimensions", None)

        engine = ReviewEngine()
        report = engine.review(doc, citation_style=citation_style, dimensions=dimensions)

        # Attach report to document metadata
        reviews = doc.get("reviews", [])
        reviews.append({
            "type": "automated_six_dimension",
            "report": report,
        })
        doc["reviews"] = reviews
        from sophia.tools.writing import _save_doc
        _save_doc(workspace, doc)

        return json.dumps(report, ensure_ascii=False)

    def _doc_review_dimension(args: Dict[str, Any], workspace: str) -> str:
        """Run a single dimension check on a document.

        Args: {id: str, dimension: str, citation_style: str}
        """
        from sophia.tools.writing import _load_doc

        doc_id = args.get("id", "")
        dimension = args.get("dimension", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        if not dimension:
            return json.dumps({"error": "dimension is required"}, ensure_ascii=False)

        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

        citation_style = args.get("citation_style", "apa7")
        engine = ReviewEngine()
        result = engine.review_dimension(doc, dimension, citation_style=citation_style)
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        name="doc_auto_review",
        description=(
            "Run automated six-dimension review on a document. "
            "Dimensions: authenticity (data/citation verification), "
            "logic (methodology-evidence chain), citations (format/matching), "
            "language (style/tone), statistics (p-value consistency/effect sizes), "
            "ethics (fabrication indicators). "
            "Returns weighted overall score + recommendation (accept/minor_revision/major_revision/reject)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID to review"},
                "citation_style": {
                    "type": "string",
                    "default": "apa7",
                    "enum": ["apa7", "gb-t-7714-2015"],
                    "description": "Citation style to validate against",
                },
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["authenticity", "logic", "citations", "language", "statistics", "ethics"],
                    },
                    "description": "Subset of dimensions to check (default: all)",
                },
            },
            "required": ["id"],
        },
        handler=partial(_doc_auto_review, workspace=workspace),
    )

    registry.register(
        name="doc_review_dimension",
        description="Run a single dimension check on a document (e.g. 'statistics').",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "dimension": {
                    "type": "string",
                    "enum": ["authenticity", "logic", "citations", "language", "statistics", "ethics"],
                    "description": "Dimension to check",
                },
                "citation_style": {
                    "type": "string",
                    "default": "apa7",
                    "enum": ["apa7", "gb-t-7714-2015"],
                },
            },
            "required": ["id", "dimension"],
        },
        handler=partial(_doc_review_dimension, workspace=workspace),
    )

    def _doc_revise_from_review(args: Dict[str, Any], workspace: str) -> str:
        """Apply automated fixes based on review findings.

        Args: {id: str, apply: bool}
        """
        from sophia.tools.writing import _load_doc, _save_doc

        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)

        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

        reviews = doc.get("reviews", [])
        auto_reviews = [r for r in reviews if r.get("type") == "automated_six_dimension"]
        if not auto_reviews:
            return json.dumps({"error": "No automated review found. Run doc_auto_review first."}, ensure_ascii=False)

        latest = auto_reviews[-1]["report"]
        findings = latest.get("all_findings", [])

        applied = []
        for f in findings:
            fix = _apply_fix(doc, f)
            if fix:
                applied.append(fix)

        _save_doc(workspace, doc)

        return json.dumps({
            "action": "revision_applied",
            "id": doc_id,
            "findings_total": len(findings),
            "fixes_applied": len(applied),
            "fixes": applied,
            "remaining_issues": len(findings) - len(applied),
        }, ensure_ascii=False)

    registry.register(
        name="doc_revise_from_review",
        description=(
            "Apply automated fixes to a document based on the latest automated review findings. "
            "Fixes weak phrases, informal language, placeholder references, etc. "
            "Returns list of applied fixes and remaining issues."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(_doc_revise_from_review, workspace=workspace),
    )


def _apply_fix(doc: Dict[str, Any], finding: Dict[str, Any]) -> Dict[str, Any]:
    """Try to automatically fix a single finding. Returns fix description or None."""
    ftype = finding.get("type", "")
    fix = None

    if ftype == "placeholder_reference":
        refs = doc.get("references", [])
        for i, ref in enumerate(refs):
            placeholders = ["placeholder", "xxx", "待补充", "待定", "unknown", "???"]
            if any(p.lower() in ref.lower() for p in placeholders):
                refs[i] = "[TODO: Complete this reference]"
                fix = {"type": "placeholder_replaced", "index": i}
        doc["references"] = refs

    elif ftype == "informal_language":
        for key, sec in doc.get("sections", {}).items():
            content = sec.get("content", "")
            replacements = {
                "really ": "", "very ": "", "pretty ": "", "quite ": "",
                "fairly ": "", "stuff": "material", "things": "factors",
                "get ": "obtain ", "got ": "obtained ", "getting ": "obtaining ",
            }
            for old, new in replacements.items():
                if old in content.lower():
                    content = content.replace(old, new)
                    content = content.replace(old.capitalize(), new.capitalize())
            sec["content"] = content
        fix = {"type": "informal_words_removed"}

    elif ftype == "weak_phrase":
        for key, sec in doc.get("sections", {}).items():
            content = sec.get("content", "")
            weak = [
                "it is interesting to note that", "it should be noted that",
                "it is important to mention", "in this day and age",
                "due to the fact that", "in order to",
                "for the purpose of", "in spite of the fact that",
                "at this point in time", "in the event that",
            ]
            for phrase in weak:
                content = content.replace(phrase, "")
                content = content.replace(phrase.capitalize(), "")
            sec["content"] = content
        fix = {"type": "weak_phrases_removed"}

    elif ftype == "p_value_rounding":
        for key, sec in doc.get("sections", {}).items():
            content = sec.get("content", "")
            content = content.replace("p = 0.000", "p < .001")
            content = content.replace("p = .000", "p < .001")
            sec["content"] = content
        fix = {"type": "p_value_rounding_fixed"}

    return fix
