"""Review → Revise → Export loop for SophiaAgent.

Automated paper quality assurance loop:
1. Run six-dimension auto-review
2. Apply automated fixes
3. If not accept, flag for human/LLM revision
4. Export to DOCX when accept or max iterations reached
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from sophia.review.engine import ReviewEngine
from sophia.pipeline.assembler import PaperAssembler

logger = logging.getLogger(__name__)


class PaperPipeline:
    """End-to-end pipeline: assemble -> review -> revise -> export."""

    MAX_ITERATIONS = 3

    def __init__(self, workspace: str, result_store=None, doc_store=None):
        self.workspace = workspace
        self.store = result_store
        self.doc_store = doc_store
        self.review_engine = ReviewEngine(result_store=result_store)
        self.assembler = PaperAssembler(result_store=result_store)

    def run(self, doc_id: str, citation_style: str = "apa7") -> Dict[str, Any]:
        """Run the full pipeline on a document.

        Returns final report with iteration history.
        """
        doc = self._load_doc(doc_id)
        if not doc:
            return {"error": f"Document '{doc_id}' not found"}

        history = []

        # Stage 1: Assemble from ResultStore
        doc = self.assembler.assemble(doc)
        self._save_doc(doc)
        history.append({"stage": "assemble", "status": "ok"})

        # Stage 2-4: Review → Revise loop
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            report = self.review_engine.review(doc, citation_style=citation_style)
            history.append({
                "stage": "review",
                "iteration": iteration,
                "overall_score": report["overall_score"],
                "recommendation": report["recommendation"],
                "critical_issues": len(report["critical_issues"]),
            })

            if report["recommendation"] == "accept":
                history.append({"stage": "decision", "result": "accept"})
                break

            # Apply automated fixes
            fixes = self._apply_fixes(doc, report["all_findings"])
            history.append({
                "stage": "revise",
                "iteration": iteration,
                "fixes_applied": len(fixes),
                "fixes": fixes,
            })
            self._save_doc(doc)

            # If major issues remain after max iterations, stop anyway
            if iteration == self.MAX_ITERATIONS:
                history.append({
                    "stage": "decision",
                    "result": "max_iterations_reached",
                    "final_recommendation": report["recommendation"],
                })

        # Stage 5: Export to DOCX
        export_result = self._export_docx(doc, citation_style)
        history.append({"stage": "export", **export_result})

        return {
            "document_id": doc_id,
            "final_score": report["overall_score"],
            "final_recommendation": report["recommendation"],
            "iterations": len([h for h in history if h.get("stage") == "review"]),
            "history": history,
            "export_path": export_result.get("path"),
        }

    def _load_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if self.doc_store:
            return self.doc_store.get(doc_id)
        # Fallback to writing tool's load
        try:
            from sophia.tools.writing import _load_doc
            return _load_doc(self.workspace, doc_id)
        except Exception:
            return None

    def _save_doc(self, doc: Dict[str, Any]) -> None:
        try:
            from sophia.tools.writing import _save_doc
            _save_doc(self.workspace, doc)
        except Exception:
            logger.warning("Failed to save doc")

    def _apply_fixes(self, doc: Dict[str, Any], findings: List[Dict]) -> List[Dict]:
        """Apply automated fixes to document based on findings."""
        applied = []
        for f in findings:
            fix = self._apply_single_fix(doc, f)
            if fix:
                applied.append(fix)
        return applied

    def _apply_single_fix(self, doc: Dict[str, Any], finding: Dict[str, Any]) -> Optional[Dict]:
        ftype = finding.get("type", "")

        if ftype == "placeholder_reference":
            refs = doc.get("references", [])
            changed = False
            for i, ref in enumerate(refs):
                placeholders = ["placeholder", "xxx", "待补充", "待定", "unknown", "???"]
                if any(p.lower() in ref.lower() for p in placeholders):
                    refs[i] = "[TODO: Complete this reference]"
                    changed = True
            if changed:
                doc["references"] = refs
                return {"type": "placeholder_replaced"}

        if ftype == "truncated_reference":
            refs = doc.get("references", [])
            for i, ref in enumerate(refs):
                if len(ref) < 20 and "[TODO" not in ref:
                    refs[i] = ref + " [TODO: Complete]"
                    doc["references"] = refs
                    return {"type": "truncation_flagged", "index": i}

        if ftype == "informal_language":
            detail = finding.get("detail", "")
            word_match = None
            m = re.search(r"Informal word '(\w+)'", detail)
            if m:
                word_match = m.group(1)
            if word_match:
                for sec in doc.get("sections", {}).values():
                    content = sec.get("content", "")
                    pattern = re.compile(re.escape(word_match), re.IGNORECASE)
                    sec["content"] = pattern.sub(f"[{word_match}]", content)
                return {"type": "informal_flagged", "word": word_match}

        if ftype == "weak_phrase":
            detail = finding.get("detail", "")
            m = re.search(r"Weak phrase '([^']+)'", detail)
            if m:
                phrase = m.group(1)
                for sec in doc.get("sections", {}).values():
                    content = sec.get("content", "")
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    sec["content"] = pattern.sub("", content)
                return {"type": "weak_phrase_removed", "phrase": phrase}

        if ftype == "p_value_rounding":
            for sec in doc.get("sections", {}).values():
                content = sec.get("content", "")
                content = content.replace("p = 0.000", "p < .001")
                content = content.replace("p = .000", "p < .001")
                sec["content"] = content
            return {"type": "p_value_rounding_fixed"}

        if ftype == "p_value_inconsistency":
            return {"type": "p_value_inconsistency_flagged", "note": "Requires human verification"}

        if ftype == "missing_effect_sizes":
            return {"type": "missing_effect_sizes_flagged", "note": "Add Cohen's d or eta-squared"}

        if ftype == "missing_confidence_intervals":
            return {"type": "missing_ci_flagged", "note": "Add 95% CI"}

        if ftype == "citation_not_in_list":
            detail = finding.get("detail", "")
            m = re.search(r"'([^']+)'", detail)
            if m:
                cite = m.group(1)
                refs = doc.get("references", [])
                refs.append(f"[TODO: Add full reference for {cite}]")
                doc["references"] = refs
                return {"type": "missing_reference_added", "citation": cite}

        return None

    def _export_docx(self, doc: Dict[str, Any], citation_style: str) -> Dict[str, Any]:
        """Export document to DOCX."""
        try:
            import os
            from sophia.exporters.docx_export import export_docx
            from sophia.tools.writing import _docs_dir

            output_dir = os.path.join(_docs_dir(self.workspace), "export")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{doc['id']}_final.docx")

            result = export_docx(doc, output_path, result_store=self.store, citation_style=citation_style)
            return result
        except Exception as e:
            logger.exception("DOCX export failed")
            return {"error": str(e)}
