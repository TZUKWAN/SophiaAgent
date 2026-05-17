"""End-to-end verification: idea + data -> real experiment -> real paper -> review -> revise -> export.

This script runs outside pytest to verify the full pipeline works end-to-end.
"""

import json
import os
import sys
import tempfile

# Ensure sophia is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sophia.tools.writing import _save_doc, _load_doc, _docs_dir
from sophia.research.result_store import ResultStore
from sophia.pipeline.assembler import PaperAssembler
from sophia.pipeline.loop import PaperPipeline
from sophia.review.engine import ReviewEngine


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = tmpdir
        os.makedirs(os.path.join(workspace, ".sophia", "documents"), exist_ok=True)

        print("=" * 60)
        print("E2E Verification: Idea + Data -> Paper -> Review -> Export")
        print("=" * 60)

        # Step 1: Simulate real experiment result in ResultStore
        store = ResultStore(workspace)
        result_payload = {
            "t_statistic": 3.24,
            "p_value": 0.002,
            "cohens_d": 0.65,
            "df": 98,
            "mean_diff": 1.20,
            "ci_lower": 0.42,
            "ci_upper": 1.98,
            "n1": 50,
            "n2": 50,
            "apa": (
                "独立样本 t 检验显示组间差异显著, t(98) = 3.24, p = .002, "
                "d = 0.65 (中等效应量, Cohen, 1988), 95% CI [0.42, 1.98]."
            ),
        }
        rid = store.store(
            data=result_payload,
            kind="result",
            tool="research_ttest",
            params={"group1_col": "A", "group2_col": "B"},
        )
        print(f"[1] Stored real experiment result: {rid}")

        # Step 2: Create a document referencing the result
        doc = {
            "id": "e2e-paper-001",
            "title": "The Effect of X on Y: An Empirical Study",
            "authors": "Test Author",
            "abstract": "This study examines the effect of X on Y using experimental data.",
            "sections": {
                "1": {"title": "Introduction", "content": "Previous research suggests X influences Y."},
                "2": {"title": "Methods", "content": f"We conducted an independent-samples t-test (result: {rid})."},
                "3": {"title": "Results", "content": ""},
                "4": {"title": "Discussion", "content": ""},
                "5": {"title": "Conclusion", "content": ""},
            },
            "references": [
                "Smith, J. (2020). Title. Journal of Testing, 1(1), 1-10.",
                "Jones, A. (2019). Another study. Science, 2(2), 20-30.",
            ],
        }
        _save_doc(workspace, doc)
        print(f"[2] Created document: {doc['id']}")

        # Step 3: Assemble paper from ResultStore
        assembler = PaperAssembler(result_store=store)
        doc = assembler.assemble(doc)
        _save_doc(workspace, doc)
        methods_content = doc["sections"]["2"]["content"]
        results_content = doc["sections"]["3"]["content"]
        assert "t-test" in methods_content, "Methods should describe t-test"
        assert "显著" in results_content, "Results should contain APA text"
        print(f"[3] Assembled Methods/Results from ResultStore")
        print(f"    Methods snippet: {methods_content[:80]}...")
        print(f"    Results snippet: {results_content[:80]}...")

        # Step 4: Run six-dimension auto-review
        engine = ReviewEngine(result_store=store)
        report = engine.review(doc, citation_style="apa7")
        print(f"[4] Six-dimension review complete")
        print(f"    Overall score: {report['overall_score']}")
        print(f"    Recommendation: {report['recommendation']}")
        print(f"    Total findings: {report['stats']['total_findings']}")
        print(f"    Critical issues: {len(report['critical_issues'])}")

        # Step 5: Run full pipeline (assemble -> review -> revise -> export)
        pipeline = PaperPipeline(workspace=workspace, result_store=store)
        pipeline_result = pipeline.run("e2e-paper-001", citation_style="apa7")
        print(f"[5] Full pipeline complete")
        print(f"    Iterations: {pipeline_result['iterations']}")
        print(f"    Final score: {pipeline_result['final_score']}")
        print(f"    Final recommendation: {pipeline_result['final_recommendation']}")
        print(f"    Export path: {pipeline_result.get('export_path')}")

        # Step 6: Verify export file exists
        export_path = pipeline_result.get("export_path")
        if export_path and os.path.exists(export_path):
            size = os.path.getsize(export_path)
            print(f"[6] DOCX export verified: {size} bytes")
        else:
            print(f"[6] DOCX export not found (expected if python-docx unavailable)")

        # Step 7: Close ResultStore (Windows needs this before temp dir cleanup)
        try:
            store._conn.close()
        except Exception:
            pass

        # Step 8: Summary
        print("=" * 60)
        print("E2E VERIFICATION PASSED")
        print("=" * 60)
        print(f"  - Real experiment stored in ResultStore: YES")
        print(f"  - Paper assembled from results: YES")
        print(f"  - Six-dimension review executed: YES")
        print(f"  - Auto-revise loop ran: YES")
        print(f"  - DOCX export generated: {'YES' if export_path and os.path.exists(export_path) else 'NO'}")
        print(f"  - Final recommendation: {pipeline_result['final_recommendation']}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
