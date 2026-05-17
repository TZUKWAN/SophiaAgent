from sophia.swarm.roles import RoleTemplate, RoleTemplateBank


def test_default_roles_include_required_core_roles():
    bank = RoleTemplateBank()
    ids = bank.list_ids()
    for role_id in ["literature_searcher", "data_analyst", "writer", "reviewer", "methodologist", "critic", "synthesizer"]:
        assert role_id in ids


def test_pure_analysis_roles_have_no_tools():
    bank = RoleTemplateBank()
    assert bank.get("critic").needs_tools is False
    assert bank.get("critic").allowed_tools == []


def test_dynamic_role_registration_and_matching():
    bank = RoleTemplateBank()
    role = RoleTemplate("policy_expert", "政策专家", "政策分析", "分析政策", expertise=["政策"])
    bank.register(role)
    assert bank.get("policy_expert") is role
    assert bank.match_for_task("请分析政策影响")[0].role_id == "policy_expert"
