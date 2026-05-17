"""Tests for PaperAssembler and PaperPipeline."""

import json
import os
import pytest

from sophia.pipeline.assembler import PaperAssembler
from sophia.pipeline.loop import PaperPipeline
from sophia.tools.writing import _save_doc, _load_doc, _docs_dir


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, ".sophia", "documents"), exist_ok=True)
    return ws


# ---------------------------------------------------------------------------
# PaperAssembler
# ---------------------------------------------------------------------------

def test_assembler_extract_result_ids():
    assembler = PaperAssembler()
    doc = {
        "sections": {
            "1": {"content": "We found res_a1b2c3d and res_e4f5a6b in our analysis."},
            "2": {"content": "See also res_a1b2c3d for comparison."},
        }
    }
    ids = assembler._extract_result_ids(doc)
    assert sorted(ids) == ["res_a1b2c3d", "res_e4f5a6b"]


def test_assembler_build_methods_paragraph():
    assembler = PaperAssembler()
    para = assembler._build_methods_paragraph("research_did", {"treatment_col": "treat"}, {})
    assert "difference-in-differences" in para
    assert "TWFE" in para


def test_assembler_build_results_paragraph_with_apa():
    assembler = PaperAssembler()
    payload = {"apa": "t(98) = 2.50, p = .014, d = 0.50."}
    para = assembler._build_results_paragraph("research_ttest", payload)
    assert para == "t(98) = 2.50, p = .014, d = 0.50."


def test_assembler_build_results_paragraph_ttest():
    assembler = PaperAssembler()
    payload = {"t_statistic": 2.5, "p_value": 0.014, "cohens_d": 0.5}
    para = assembler._build_results_paragraph("research_ttest", payload)
    assert "t = 2.50" in para
    assert "p = 0.014" in para


def test_assembler_assemble_empty_doc():
    assembler = PaperAssembler()
    doc = {"sections": {}}
    result = assembler.assemble(doc)
    assert result == doc


# ---------------------------------------------------------------------------
# PaperPipeline (mocked, no real ResultStore)
# ---------------------------------------------------------------------------

def test_pipeline_load_missing_doc(tmp_workspace):
    pipeline = PaperPipeline(workspace=tmp_workspace)
    result = pipeline.run("nonexistent")
    assert "error" in result


def test_pipeline_apply_fix_placeholder(tmp_workspace):
    pipeline = PaperPipeline(workspace=tmp_workspace)
    doc = {
        "id": "test-001",
        "title": "Test",
        "references": ["Smith (2020). Title. Journal.", "placeholder"],
        "sections": {},
    }
    fix = pipeline._apply_single_fix(doc, {"type": "placeholder_reference"})
    assert fix is not None
    assert "TODO" in doc["references"][1]


def test_pipeline_apply_fix_p_zero(tmp_workspace):
    pipeline = PaperPipeline(workspace=tmp_workspace)
    doc = {
        "id": "test-002",
        "title": "Test",
        "references": [],
        "sections": {
            "1": {"title": "Results", "content": "The result was significant, p = 0.000."}
        },
    }
    fix = pipeline._apply_single_fix(doc, {"type": "p_value_rounding"})
    assert fix is not None
    assert "p < .001" in doc["sections"]["1"]["content"]


def test_pipeline_apply_fix_weak_phrase(tmp_workspace):
    pipeline = PaperPipeline(workspace=tmp_workspace)
    doc = {
        "id": "test-003",
        "title": "Test",
        "references": [],
        "sections": {
            "1": {"title": "Intro", "content": "It is interesting to note that this works."}
        },
    }
    fix = pipeline._apply_single_fix(doc, {"type": "weak_phrase", "detail": "Weak phrase 'it is interesting to note that'"})
    assert fix is not None
    assert "It is interesting to note that" not in doc["sections"]["1"]["content"]


def test_pipeline_full_run_no_results(tmp_workspace):
    """Pipeline runs on a minimal doc without result_ids."""
    doc = {
        "id": "pipe-001",
        "title": "Minimal Paper",
        "abstract": "Abstract text.",
        "authors": ["Alice"],
        "sections": {
            "1": {"title": "Introduction", "content": "Intro."},
            "2": {"title": "Methods", "content": ""},
            "3": {"title": "Results", "content": ""},
            "4": {"title": "Discussion", "content": "Discuss."},
        },
        "references": ["Smith (2020). Title. Journal, 1(1), 1-10."],
    }
    _save_doc(tmp_workspace, doc)

    pipeline = PaperPipeline(workspace=tmp_workspace)
    result = pipeline.run("pipe-001")

    assert result["document_id"] == "pipe-001"
    assert result["iterations"] >= 1
    assert "history" in result
    assert "export_path" in result
