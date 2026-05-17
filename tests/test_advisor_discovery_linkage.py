"""Tests for Advisor-Discovery-SkillFactory deep linkage (S4)."""
import json
import pytest

from sophia.research.advisor import MethodologyAdvisor
from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.skills import SkillManager
from sophia.skills.factory import SkillFactory


@pytest.fixture
def catalog(tmp_path):
    return MethodCatalog(str(tmp_path / "catalog.db"))


@pytest.fixture
def advisor_with_catalog(catalog):
    return MethodologyAdvisor(catalog=catalog)


class TestAdvisorCatalogFilter:
    def test_filters_uninstalled_methods(self, catalog):
        """Advisor linked to catalog should only recommend installed methods."""
        # Mark did as installed (default seed does this)
        # Explicitly mark iv as experimental (not installed)
        catalog.update("iv", status="experimental")

        advisor = MethodologyAdvisor(catalog=catalog)
        result = json.loads(advisor.advise({
            "research_question": "policy effect on employment",
            "data_description": {"N": 5000, "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))

        methods = [r["method_id"] for r in result["recommended_methods"]]
        # iv was marked experimental, should not appear
        assert "iv" not in methods
        # did is installed, should appear
        assert "did" in methods

    def test_enriches_from_catalog(self, catalog):
        """Advisor should enrich recommendations with catalog metadata."""
        advisor = MethodologyAdvisor(catalog=catalog)
        result = json.loads(advisor.advise({
            "research_question": "treatment effect",
            "data_description": {"N": 100, "units": 2, "periods": 1},
            "design": "observational",
            "outcome_type": "continuous",
            "constraints": ["no instrument"],
        }))

        for rec in result["recommended_methods"]:
            assert "catalog_status" in rec
            assert "catalog_verified" in rec
            assert rec["catalog_status"] == "installed"

    def test_skill_linked_still_recommended(self, catalog):
        """Methods linked to skills should still be recommended."""
        # Mark did as skill_linked (e.g., after a skill registered it)
        catalog.update("did", status="skill_linked")

        advisor = MethodologyAdvisor(catalog=catalog)
        result = json.loads(advisor.advise({
            "research_question": "policy effect on employment",
            "data_description": {"N": 5000, "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))

        methods = [r["method_id"] for r in result["recommended_methods"]]
        assert "did" in methods
        # Verify catalog_status reflects skill_linked
        did_rec = next(r for r in result["recommended_methods"] if r["method_id"] == "did")
        assert did_rec["catalog_status"] == "skill_linked"

    def test_backward_compat_no_catalog(self):
        """Advisor without catalog should still work (backward compatible)."""
        advisor = MethodologyAdvisor()
        result = json.loads(advisor.advise({
            "research_question": "treatment effect",
            "data_description": {"N": 100},
            "design": "observational",
            "outcome_type": "continuous",
        }))
        assert len(result["recommended_methods"]) > 0
        # No catalog metadata when catalog is None
        for rec in result["recommended_methods"]:
            assert "catalog_status" not in rec


class TestCatalogSkillRegistration:
    def test_skill_registers_tools_in_catalog(self, catalog, tmp_path):
        """Creating a skill should register its tools in the method catalog."""
        skill_mgr = SkillManager(str(tmp_path / "skills.db"))
        factory = SkillFactory(skill_manager=skill_mgr, catalog=catalog)

        sid = factory.create_skill(
            name="Test Skill",
            workflow=[
                {"tool": "research_ttest", "params": {"group1": [1, 2], "group2": [3, 4]}},
                {"tool": "research_regression", "params": {"x": [1, 2, 3], "y": [2, 4, 6]}},
            ],
        )

        # Both tools should be linked in catalog
        ttest_entry = catalog.get_by_tool("research_ttest")
        assert ttest_entry is not None
        assert ttest_entry["status"] == "skill_linked"
        assert "skill_id" in ttest_entry.get("discovery_context", "")

        reg_entry = catalog.get_by_tool("research_regression")
        assert reg_entry is not None
        assert reg_entry["status"] == "skill_linked"

    def test_auto_generated_skill_registers_in_catalog(self, catalog, tmp_path):
        """Auto-generated skills should also register in catalog."""
        from sophia.learning import LearningManager
        from sophia.skills.pattern_miner import ExecutionPatternMiner

        skill_mgr = SkillManager(str(tmp_path / "skills.db"))
        learning_mgr = LearningManager()
        factory = SkillFactory(
            skill_manager=skill_mgr,
            learning_manager=learning_mgr,
            catalog=catalog,
        )

        # Seed log with repeated sequence
        for i in range(3):
            t = 1000 + i * 2
            learning_mgr.record_execution("tool.post_dispatch", {
                "tool": "research_load_data",
                "args": {},
                "result": json.dumps({"status": "success"}),
                "timestamp": t,
            })
            learning_mgr.record_execution("tool.post_dispatch", {
                "tool": "research_ttest",
                "args": {},
                "result": json.dumps({"status": "success"}),
                "timestamp": t + 1,
            })

        installed = factory.auto_generate_from_logs(top_n=1)
        assert len(installed) > 0

        # The tools should be linked in catalog
        load_entry = catalog.get_by_tool("research_load_data")
        assert load_entry is not None
        assert load_entry["status"] == "skill_linked"

    def test_catalog_stats_include_skills(self, catalog, tmp_path):
        """Catalog stats should reflect skill-linked methods."""
        skill_mgr = SkillManager(str(tmp_path / "skills.db"))
        factory = SkillFactory(skill_manager=skill_mgr, catalog=catalog)

        factory.create_skill(
            name="Stats Skill",
            workflow=[{"tool": "research_ttest", "params": {}}],
        )

        stats = catalog.get_stats()
        assert stats["by_status"].get("skill_linked", 0) >= 1


class TestAgentIntegration:
    def test_agent_links_advisor_and_factory_to_catalog(self, tmp_path):
        """SophiaAgent should link advisor and skill_factory to the same catalog."""
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        assert agent.advisor.catalog is agent.method_catalog
        assert agent.skill_factory.catalog is agent.method_catalog

    def test_advisor_via_agent_tools(self, tmp_path):
        """Agent's methodology_advise tool should use catalog-linked advisor."""
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        result = json.loads(agent.tools.dispatch("methodology_advise", {
            "research_question": "policy effect on employment",
            "data_description": {"N": 5000, "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))

        assert "recommended_methods" in result
        assert len(result["recommended_methods"]) > 0
        # With catalog linked, all recommendations should be installed
        for rec in result["recommended_methods"]:
            assert rec.get("catalog_status") == "installed"
