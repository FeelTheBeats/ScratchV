"""Tests for the Topic 15 inliner feature case report.

The report is the CI artifact that proves the inliner removes eligible CALL
sites with independent clones, keeps refused CALLs untouched, and produces
identical IR fingerprints across runs.  The DSL/ONNX frontends never emit
CALL, so the case itself is programmatic IR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import run_topic15_inline_case as runner
from benchmarks.cases.topic15_inline_feature import (
    build_program,
    build_rejected_program,
)
from scratchv.analysis.ir_verifier import ErrorLevel, IRVerifier
from scratchv.ir.types import OpCode
from scratchv.optimizer.inliner import Inliner

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic15_inline_feature.py"
)


def _calls(program):
    return [
        ins
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
        if ins.opcode is OpCode.CALL
    ]


def _error_messages(program):
    return [
        err.message
        for err in IRVerifier(program).verify()
        if err.level is ErrorLevel.ERROR
    ]


def test_case_program_shape_and_verifier_clean():
    program = build_program()
    calls = _calls(program)
    assert [call.target for call in calls] == ["inc", "inc"]
    assert all(call.attrs["argc"] == 1 for call in calls)

    # Known verifier defect: a residual CALL's function-name target is
    # checked as a block label, so the uninlined program reports exactly one
    # spurious ERROR per CALL.  After inlining there must be none.
    uninlined = _error_messages(program)
    assert len(uninlined) == 2
    assert all("jump target 'inc' does not exist" in msg
               for msg in uninlined)

    inliner = Inliner(program, runner.default_inliner_config())
    assert inliner.run() == 2
    assert _error_messages(program) == []


def test_measure_ab_counts():
    off = runner._measure_uninlined(build_program)
    on = runner._measure_inlined(build_program, repeats=1)

    assert off["call_count"] == 2
    assert off["clones"] == 0 and off["clone_count"] == 0
    assert on["call_count"] == 0
    assert on["clones"] == 2 and on["clone_count"] == 2
    assert on["rejected"] == 0 and on["warnings"] == []
    assert on["pass_time_ms"] is not None and on["pass_time_ms"] >= 0
    assert on["ir_instructions"] - off["ir_instructions"] == (
        on["clones"] * off["callee_body_size"])


def test_clone_namespaces_and_returns_are_independent():
    on = runner._measure_inlined(build_program, repeats=1)
    details = {detail["index"]: detail for detail in on["clones_detail"]}
    assert set(details) == {0, 1}

    dests0 = set(details[0]["dest_names"])
    dests1 = set(details[1]["dest_names"])
    assert dests0 and dests1 and not (dests0 & dests1)
    assert all(name.endswith("_inl0") for name in dests0)
    assert all(name.endswith("_inl1") for name in dests1)
    assert details[0]["returnless"] and details[1]["returnless"]
    assert on["clone_returns"] == 0
    assert on["returns_in_callee"] == 1
    assert not any(on["duplicate_defined_names"].values())

    for detail in details.values():
        assert len(detail["return_redirects"]) == len(detail["blocks"])
        for target in detail["return_redirects"].values():
            assert target in on["block_names"]
            assert target.endswith("_cont")


def test_rejected_branch_keeps_calls_and_records_warnings():
    rejected = runner._measure_rejected(build_rejected_program)

    assert rejected["clones"] == 0
    assert rejected["rejected"] == 2
    assert rejected["call_count"] == 2
    assert rejected["dump_unchanged"] is True
    assert len(rejected["warnings"]) == 2
    assert any("body_too_large (9 > 4)" in w for w in rejected["warnings"])
    assert any("loop_body_unsupported" in w for w in rejected["warnings"])
    assert all(w.startswith("inliner: skip") for w in rejected["warnings"])


def test_evaluate_passes_all_hard_checks_and_is_deterministic():
    report = runner.evaluate(CASE, repeats=1)
    assert report["schema_version"] == runner.SCHEMA_VERSION
    assert report["topic"] == "topic15-function-inline"
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["honesty"]

    on = report["inlined"]
    assert set(on["fingerprints"]) == {on["fingerprint"]}
    second = runner.evaluate(CASE, repeats=1)
    assert second["inlined"]["fingerprint"] == on["fingerprint"]
    assert (
        second["rejected"]["fingerprint"]
        == report["rejected"]["fingerprint"]
    )


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    exit_code = runner.main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
        "--repeats", "1",
    ])
    assert exit_code == 0

    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic15-function-inline"
    assert data["inlined"]["call_count"] == 0
    assert data["rejected"]["rejected"] == 2

    markdown = md_path.read_text()
    assert "# Topic 15 Function-Inline Feature Case" in markdown
    assert "## A/B summary" in markdown
    assert "## Clone detail" in markdown
    assert "## Rejection detail" in markdown
    assert "## Hard checks" in markdown
    assert "## Honesty" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--case", str(tmp_path / "missing.py")])
    assert exc.value.code == 2


def test_hard_check_gate_is_not_vacuous(monkeypatch, tmp_path):
    """A no-op inliner must surface as hard failures and exit code 1."""

    class NoopInliner:
        def __init__(self, program, config=None):
            self.program = program
            self.stats = {"inlined": 0, "rejected": 0, "rounds": 0}
            self.warnings = []

        def run(self):
            return 0

    monkeypatch.setattr(runner, "Inliner", NoopInliner)
    report = runner.evaluate(CASE, repeats=1)
    assert "inlined_removes_all_calls" in report["hard_failures"]
    assert "inliner_clones_each_site" in report["hard_failures"]
    assert "clone_returns_rewritten_to_branches" in report["hard_failures"]
    assert "ir_grows_by_cloned_bodies" in report["hard_failures"]

    exit_code = runner.main([
        "--case", str(CASE),
        "--json", str(tmp_path / "report.json"),
        "--markdown", str(tmp_path / "report.md"),
        "--repeats", "1",
    ])
    assert exit_code == 1
