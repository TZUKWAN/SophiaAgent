"""Tests for core tools: files, citation, writing, analysis."""
import json
import os


class TestFileTools:
    def test_file_write_and_read(self, tmp_workspace):
        from sophia.tools.files import register_file_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_file_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("file_write", {
            "path": "test.txt", "content": "hello world"
        }))
        assert "error" not in result
        assert result["bytes_written"] == 11

        result = json.loads(reg.dispatch("file_read", {"path": "test.txt"}))
        assert "hello world" in result["content"]

    def test_file_list(self, tmp_workspace):
        from sophia.tools.files import register_file_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_file_tools(reg, tmp_workspace)

        reg.dispatch("file_write", {"path": "a.txt", "content": "a"})
        reg.dispatch("file_write", {"path": "b.txt", "content": "b"})

        result = json.loads(reg.dispatch("file_list", {}))
        assert len(result["entries"]) >= 2

    def test_path_traversal_blocked(self, tmp_workspace):
        from sophia.tools.files import register_file_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_file_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("file_read", {"path": "../../etc/passwd"}))
        assert "error" in result


class TestCitationTools:
    def test_add_and_list(self, tmp_workspace):
        from sophia.tools.citation import register_citation_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_citation_tools(reg, tmp_workspace)

        reg.dispatch("ref_add", {
            "key": "Test2024", "type": "article",
            "fields": {
                "title": "Test Paper",
                "author": "A. Author",
                "year": "2024",
            },
        })
        result = json.loads(reg.dispatch("ref_list", {}))
        assert result["total"] == 1
        assert result["references"][0]["key"] == "Test2024"

    def test_format_gbt7714(self, tmp_workspace):
        from sophia.tools.citation import register_citation_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_citation_tools(reg, tmp_workspace)

        reg.dispatch("ref_add", {
            "key": "Test2024", "type": "article",
            "fields": {
                "title": "Test", "author": "Author",
                "year": "2024", "journal": "J",
                "volume": "1", "pages": "1-10",
            },
        })
        result = json.loads(
            reg.dispatch("ref_format", {"style": "gb-t-7714-2015"})
        )
        assert len(result["citations"]) == 1
        assert "Author" in result["citations"][0]["formatted"]

    def test_search(self, tmp_workspace):
        from sophia.tools.citation import register_citation_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_citation_tools(reg, tmp_workspace)

        reg.dispatch("ref_add", {
            "key": "Smith2024", "type": "article",
            "fields": {
                "title": "Network Analysis",
                "author": "Smith",
            },
        })
        result = json.loads(
            reg.dispatch("ref_search", {"query": "network"})
        )
        assert result["found"] == 1


class TestWritingTools:
    def test_create_and_get(self, tmp_workspace):
        from sophia.tools.writing import register_writing_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_writing_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("doc_create", {
            "title": "Test Paper", "doc_type": "paper"
        }))
        assert result["action"] == "created"
        doc_id = result["id"]

        result = json.loads(reg.dispatch("doc_get", {"id": doc_id}))
        assert result["title"] == "Test Paper"
        assert len(result["sections"]) == 8

    def test_write_section_by_title(self, tmp_workspace):
        from sophia.tools.writing import register_writing_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_writing_tools(reg, tmp_workspace)

        reg.dispatch("doc_create", {
            "title": "T", "doc_type": "paper", "id": "testdoc",
        })
        result = json.loads(reg.dispatch("doc_write_section", {
            "id": "testdoc", "section": "引言",
            "content": "This is the intro.",
        }))
        assert result["action"] == "section_written"
        assert result["section_title"] == "引言"

    def test_export_markdown(self, tmp_workspace):
        from sophia.tools.writing import register_writing_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_writing_tools(reg, tmp_workspace)

        reg.dispatch("doc_create", {
            "title": "MD Test", "doc_type": "paper", "id": "mdtest",
        })
        reg.dispatch("doc_write_section", {
            "id": "mdtest", "section": "2", "content": "Intro text",
        })
        result = json.loads(
            reg.dispatch("doc_export_markdown", {"id": "mdtest"})
        )
        assert result["format"] == "markdown"
        assert os.path.exists(result["path"])


class TestAnalysisSandbox:
    def test_safe_code(self, tmp_workspace):
        from sophia.tools.analysis import register_analysis_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_analysis_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("code_execute", {
            "code": "x = 2 + 2\nprint(x)"
        }))
        assert "error" not in result
        assert "4" in result["stdout"]

    def test_os_import_blocked(self, tmp_workspace):
        from sophia.tools.analysis import register_analysis_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_analysis_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("code_execute", {
            "code": "import os\nos.system('echo hacked')"
        }))
        assert "error" in result
        assert "not available" in result["error"]

    def test_exec_blocked(self, tmp_workspace):
        from sophia.tools.analysis import register_analysis_tools
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        register_analysis_tools(reg, tmp_workspace)

        result = json.loads(reg.dispatch("code_execute", {
            "code": "exec('print(42)')"
        }))
        assert "error" in result
