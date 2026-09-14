"""Tests for the Topic 27 RV32-bench feature case report.

The report drives the real ``rv32_bench.main`` on a tiny deterministic model
and is the CI artifact that proves the honest-report contract: schema v2,
labeled truncation, null ratio with a reason, and a non-vacuous provenance
audit.  Dynamic tests only run the toy case model with explicit budgets.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import run_topic27_rv32_bench_case as case
from scratchv.standalone import bench_report, rv32_bench


class _UnavailableProfiledMachine:
    available = False

    def __init__(self, mem_size=0):
        self.mem_size = mem_size


@pytest.fixture(scope="module")
def case_model(tmp_path_factory):
    pytest.importorskip("onnx")
    path = (
        tmp_path_factory.mktemp("topic27_case")
        / "topic27_rv32_bench_feature.onnx"
    )
    case.build_case_model(path)
    return path


@pytest.fixture(scope="module")
def base_report(case_model, tmp_path_factory):
    work = tmp_path_factory.mktemp("topic27_case_run")
    return case.evaluate(case_model, work, case_builder=case.CASE_BUILDER)


def test_report_passes_all_hard_checks(base_report):
    assert base_report["schema_version"] == case.SCHEMA_VERSION
    assert base_report["rv32_report"]["schema_version"] == (
        case.RV32_SCHEMA_VERSION)
    assert base_report["schema_errors"] == []
    assert base_report["audit_violations"] == []
    assert base_report["hard_failures"] == []
    assert all(base_report["hard_checks"].values())
    assert base_report["honesty"]


def test_completion_and_ratio_are_honest(base_report):
    assert base_report["completion"] in case.ALLOWED_COMPLETIONS
    comparison = base_report["comparison"]
    if comparison["dynamic_instruction_ratio"] is None:
        assert comparison["incomparable_reason"]
    else:
        for side in ("scratchv", "llvm"):
            dyn = base_report["rv32_report"][side]["dynamic"]
            assert dyn["source"] == "simulated"
            assert dyn["completion"] == "halted"

    rv_report = base_report["rv32_report"]
    assert bench_report.validate_report_schema(rv_report) == []
    dyn = rv_report["scratchv"]["dynamic"]
    if base_report["config"]["simulator_available"]:
        assert dyn["source"] == "simulated"
        assert dyn["ops"]["total"] == dyn["executed"]
    else:
        assert dyn["source"] == "unavailable"
        assert dyn["completion"] == "not_run"
        assert dyn["ops"] is None
        assert dyn["reason"]


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "rv32_bench_case_report.json"
    md_path = tmp_path / "rv32_bench_case_report.md"
    exit_code = case.main([
        "--json", str(json_path),
        "--markdown", str(md_path),
    ])
    assert exit_code == 0

    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["schema_version"] == case.SCHEMA_VERSION
    assert data["rv32_report"]["schema_version"] == case.RV32_SCHEMA_VERSION
    assert data["github_summary"].startswith("# RV32 Benchmark Summary")
    assert "## " in data["report_markdown"]
    assert data["config"]["llvm_skipped"] is True

    markdown = md_path.read_text()
    for section in (
        "## RV32 report schema",
        "## Completion / ratio status",
        "## Rendered report summary",
        "## Probes",
        "## Hard checks",
        "## Honesty",
    ):
        assert section in markdown, section
    assert "Topic 27 RV32-Bench Feature Case" in capsys.readouterr().out


def test_main_rejects_missing_model(tmp_path, capsys):
    json_path = tmp_path / "missing.json"
    exit_code = case.main([
        "--model", str(tmp_path / "nope.onnx"),
        "--json", str(json_path),
        "--markdown", str(tmp_path / "missing.md"),
    ])
    assert exit_code == 2
    assert not json_path.exists()
    assert "case model not found" in capsys.readouterr().err


def test_audit_probe_rejects_forgery(base_report):
    probe = base_report["audit_probe"]
    assert probe["rejected"] is True
    assert len(probe["tampers"]) >= 2
    assert all(t["rejected"] and t["violations"] for t in probe["tampers"])
    fields = {t["field"] for t in probe["tampers"]}
    assert "comparison.dynamic_instruction_ratio=0.31" in fields
    assert "schema_version=rv32-bench/1" in fields

    clean = base_report["rv32_report"]
    assert rv32_bench.audit_provenance(clean) == []
    forged = json.loads(json.dumps(clean))
    forged["comparison"] = {
        "dynamic_instruction_ratio": 0.31, "incomparable_reason": None,
    }
    violations = rv32_bench.audit_provenance(forged)
    assert any("incomplete/non-simulated" in v for v in violations)


def test_degraded_mode_is_honest_without_simulator(
        case_model, tmp_path, monkeypatch):
    monkeypatch.setattr(
        rv32_bench, "ProfiledMachine", _UnavailableProfiledMachine)
    report = case.evaluate(case_model, tmp_path / "degraded")

    assert report["config"]["simulator_available"] is False
    assert report["completion"] == "not_run"
    assert report["hard_failures"] == []
    assert report["comparison"]["dynamic_instruction_ratio"] is None
    assert "scratchv" in report["comparison"]["incomparable_reason"]

    dyn = report["rv32_report"]["scratchv"]["dynamic"]
    assert dyn["source"] == "unavailable"
    assert dyn["ops"] is None
    assert report["budget_probe"]["status"] == "not_run"
    assert report["timeout_probe"]["status"] == "not_run"
    assert "TinyFive is unavailable" in report["honesty"]

    markdown = case.render_markdown(report)
    assert "not_run" in markdown
    assert "[unavailable]" in report["report_markdown"]


def test_probes_match_environment(base_report):
    budget = base_report["budget_probe"]
    timeout = base_report["timeout_probe"]
    if base_report["config"]["simulator_available"]:
        assert budget["status"] == "confirmed"
        assert budget["completion"] == "budget_exhausted"
        assert budget["limit"] == budget["executed"] == case.BUDGET_LIMIT
        assert budget["ops_total"] == budget["executed"]
        assert budget["ratio"] is None and budget["incomparable_reason"]
        assert budget["schema_errors"] == []
        assert budget["audit_violations"] == []

        assert timeout["status"] == "confirmed"
        assert timeout["completion"] == "timeout"
        assert timeout["partial"] is True
        assert timeout["ratio"] is None and timeout["incomparable_reason"]
        assert timeout["schema_errors"] == []
    else:
        assert budget["status"] == "not_run" and budget["reason"]
        assert timeout["status"] == "not_run" and timeout["reason"]


def test_hard_check_gate_is_not_vacuous(case_model, tmp_path, monkeypatch):
    def silent_driver(_argv):
        return rv32_bench.EXIT_OK

    monkeypatch.setattr(rv32_bench, "main", silent_driver)
    report = case.evaluate(case_model, tmp_path / "silent")

    assert report["run"]["exit_code"] == 0
    assert report["rv32_report"] is None
    for name in (
        "rv32_json_artifact_written",
        "rv32_schema_valid",
        "provenance_audit_clean",
        "completion_is_legal",
        "ratio_is_honest",
        "audit_gate_rejects_forgery",
        "budget_probe_honest",
        "timeout_probe_honest",
    ):
        assert name in report["hard_failures"], report["hard_failures"]
