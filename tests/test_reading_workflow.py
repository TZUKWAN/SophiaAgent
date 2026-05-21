"""End-to-end workflow tests for reading, notes, and literature graph."""

import json
import os

import pytest

from sophia.research.literature_graph import LiteratureGraph
from sophia.research.notes import ZettelkastenStore
from sophia.research.reader import PaperReader
from sophia.tools.reading import (
    literature_graph_build,
    literature_graph_clusters,
    literature_graph_visualize,
    note_create,
    note_from_paper,
    note_graph,
    note_link,
    note_search,
    paper_compare,
    paper_extract_elements,
    paper_extract_annotations,
)


class TestPaperReadingWorkflow:
    def test_extract_and_create_note(self, tmp_workspace):
        text = (
            "研究问题：社会资本如何影响居民幸福感？\n"
            "理论框架：基于社会资本理论（Putnam, 2000）\n"
            "方法：问卷调查与OLS回归分析\n"
            "数据来源：某市三个社区的1250名居民\n"
            "主要发现：社会资本对幸福感有显著正向影响（β=0.35, p<0.001）\n"
            "局限：横截面数据，无法推断因果关系"
        )
        result = paper_extract_elements({"text": text})
        parsed = json.loads(result)
        assert "research_question" in parsed

        # Create note from elements
        note_result = note_from_paper(
            {"elements": parsed, "paper_title": "社会资本与幸福感", "paper_id": "test001"},
            tmp_workspace,
        )
        note_parsed = json.loads(note_result)
        assert note_parsed["success"] is True
        assert note_parsed["note"]["note_type"] == "evidence"

    def test_compare_papers_workflow(self):
        paper1 = {
            "research_question": ["Does social capital affect happiness?"],
            "theoretical_framework": ["Social capital theory"],
            "methods": ["Survey, OLS regression"],
            "data_sources": ["Urban residents"],
            "sample_size": "1250",
            "main_findings": ["Positive significant effect"],
            "limitations": ["Cross-sectional data"],
        }
        paper2 = {
            "research_question": ["How does trust influence well-being?"],
            "theoretical_framework": ["Social capital theory"],
            "methods": ["Experiment"],
            "data_sources": ["University students"],
            "sample_size": "300",
            "main_findings": ["Mixed results"],
            "limitations": ["Small sample, student population"],
        }
        result = paper_compare({"elements_list": [paper1, paper2]})
        parsed = json.loads(result)
        assert parsed["paper_count"] == 2
        assert "matrix" in parsed
        assert "consensus" in parsed
        assert "controversies" in parsed


class TestNotesWorkflow:
    def test_create_search_link_graph(self, tmp_workspace):
        # Create notes
        r1 = note_create({
            "title": "社会资本理论",
            "content": "Putnam (2000) 提出的社会资本理论强调社会网络的价值。",
            "note_type": "concept",
            "tags": ["理论", "社会学"],
        }, tmp_workspace)
        n1 = json.loads(r1)["note"]["id"]

        r2 = note_create({
            "title": "幸福感测量",
            "content": "主观幸福感通常通过生活满意度量表测量。",
            "note_type": "concept",
            "tags": ["测量", "心理学"],
        }, tmp_workspace)
        n2 = json.loads(r2)["note"]["id"]

        # Link notes
        link_result = note_link({"note_id": n1, "links": [n2]}, tmp_workspace)
        assert json.loads(link_result)["success"] is True

        # Search
        search_result = note_search({"query": "社会资本"}, tmp_workspace)
        search_parsed = json.loads(search_result)
        assert search_parsed["count"] == 1

        # Graph
        graph_result = note_graph({}, tmp_workspace)
        graph_parsed = json.loads(graph_result)
        assert graph_parsed["node_count"] == 2
        assert graph_parsed["edge_count"] == 2  # forward + back

    def test_note_types(self, tmp_workspace):
        for nt in ["concept", "evidence", "comment"]:
            result = note_create({
                "title": f"Test {nt}",
                "content": f"Content for {nt}",
                "note_type": nt,
            }, tmp_workspace)
            parsed = json.loads(result)
            assert parsed["success"] is True
            assert parsed["note"]["note_type"] == nt


class TestLiteratureGraphWorkflow:
    def test_full_workflow(self, tmp_workspace):
        # Setup bib and relations
        bib_path = os.path.join(tmp_workspace, ".sophia", "references.bib")
        os.makedirs(os.path.dirname(bib_path), exist_ok=True)
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("""
@article{paperA,
  author = {Author A},
  title = {Title A},
  journal = {Journal A},
  year = {2024},
}
@article{paperB,
  author = {Author B},
  title = {Title B},
  journal = {Journal B},
  year = {2023},
}
@article{paperC,
  author = {Author C},
  title = {Title C},
  journal = {Journal C},
  year = {2022},
}
""")

        rel_path = os.path.join(tmp_workspace, ".sophia", "citation_relations.json")
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump([
                {"from": "paperA", "to": "paperB", "type": "cites"},
                {"from": "paperB", "to": "paperC", "type": "extends"},
                {"from": "paperA", "to": "paperC", "type": "theory_similar"},
            ], f)

        # Build graph
        build_result = literature_graph_build({}, tmp_workspace)
        build_parsed = json.loads(build_result)
        assert "nodes" in build_parsed or "error" not in build_parsed

        # Visualize
        viz_result = literature_graph_visualize({"format": "mermaid"}, tmp_workspace)
        viz_parsed = json.loads(viz_result)
        assert "visualization" in viz_parsed
        assert "graph TD" in viz_parsed["visualization"]

        # Clusters and key papers
        cluster_result = literature_graph_clusters({}, tmp_workspace)
        cluster_parsed = json.loads(cluster_result)
        assert "clusters" in cluster_parsed
        assert "key_papers" in cluster_parsed

    def test_visualize_formats(self, tmp_workspace):
        bib_path = os.path.join(tmp_workspace, ".sophia", "references.bib")
        os.makedirs(os.path.dirname(bib_path), exist_ok=True)
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("""
@article{x,
  author = {X},
  title = {X},
  journal = {J},
  year = {2024},
}
""")

        for fmt in ["mermaid", "tikz", "dot"]:
            result = literature_graph_visualize({"format": fmt}, tmp_workspace)
            parsed = json.loads(result)
            assert "visualization" in parsed
            assert "error" not in parsed


class TestIntegration:
    def test_paper_to_note_to_graph(self, tmp_workspace):
        # Step 1: Extract elements from paper text
        text = (
            "研究问题：数字鸿沟对老年人健康的影响\n"
            "理论框架：数字不平等理论\n"
            "方法：倾向得分匹配（PSM）\n"
            "数据来源：中国健康与养老追踪调查（CHARLS）\n"
            "样本：4560名60岁以上老年人\n"
            "主要发现：数字鸿沟显著负向影响老年人自评健康\n"
            "局限：遗漏变量可能仍然存在"
        )
        extract_result = json.loads(paper_extract_elements({"text": text}))
        assert extract_result["sample_size"] == "4560" or "4560" in str(extract_result)

        # Step 2: Create evidence note
        note_result = json.loads(note_from_paper({
            "elements": extract_result,
            "paper_title": "数字鸿沟与老年人健康",
            "paper_id": "charls2024",
        }, tmp_workspace))
        assert note_result["success"] is True
        note_id = note_result["note"]["id"]

        # Step 3: Create a concept note and link
        concept_result = json.loads(note_create({
            "title": "数字鸿沟",
            "content": "数字鸿沟是指不同群体在信息技术接入和使用上的不平等。",
            "note_type": "concept",
            "tags": ["数字不平等", "老龄化"],
        }, tmp_workspace))
        concept_id = concept_result["note"]["id"]

        # Step 4: Link evidence to concept
        link_result = json.loads(note_link({
            "note_id": note_id,
            "links": [concept_id],
        }, tmp_workspace))
        assert link_result["success"] is True

        # Step 5: Verify graph
        graph_result = json.loads(note_graph({}, tmp_workspace))
        assert graph_result["node_count"] == 2
        assert graph_result["edge_count"] >= 1

        # Step 6: Search
        search_result = json.loads(note_search({"query": "数字鸿沟"}, tmp_workspace))
        assert search_result["count"] >= 1
