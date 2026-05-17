import json

from sophia.config import Config
from sophia.doctor import render_report, run_doctor


def test_doctor_core_checks_pass_for_local_tools(tmp_path):
    config = Config()
    config.session.workspace = str(tmp_path)
    config.session.db_path = str(tmp_path / "sessions.db")
    config.model.base_url = "http://localhost:9999/v1"
    config.model.api_key = "test-key"

    report = run_doctor(config, network=False)
    data = report.to_dict()

    assert any(check["name"] == "python" and check["status"] in {"pass", "warn"} for check in data["checks"])
    assert any(check["name"] == "workspace" and check["status"] == "pass" for check in data["checks"])
    assert any(check["name"] == "tools" and check["status"] == "pass" for check in data["checks"])
    assert "SophiaAgent Doctor" in render_report(report)


def test_doctor_warns_when_model_is_not_configured(tmp_path):
    config = Config()
    config.session.workspace = str(tmp_path)
    config.model.base_url = ""
    config.model.api_key = ""

    report = run_doctor(config, network=False)
    model_check = next(check for check in report.checks if check.name == "model")

    assert model_check.status == "warn"
    assert "No model endpoint" in model_check.message


def test_doctor_json_is_serializable(tmp_path):
    config = Config()
    config.session.workspace = str(tmp_path)
    payload = json.dumps(run_doctor(config).to_dict(), ensure_ascii=False)

    assert "checks" in payload
