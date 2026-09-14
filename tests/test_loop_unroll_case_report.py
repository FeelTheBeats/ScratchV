"""Tests for the Topic 10 loop-unroll feature case report.

The report is the CI artifact that proves the unroll pass is wired through
the configured compiler pipeline, actually transforms the case, and keeps
architectural results identical under the RV32 emulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic10_unroll_case import (
    EXPECTED_RESULT,
    SCHEMA_VERSION,
    _measure_side,
    evaluate,
    main,
    measure_wiring,
)

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic10_unroll_feature.dsl"
)


def test_case_program_runs_expected_result():
    off = _measure_side(unroll=False, repeats=1)
    assert off["execution"]["a0"] == EXPECTED_RESULT
    assert off["execution"]["dynamic_instructions"] > 0
    assert off["loops_unrolled"] == 0


def test_measure_side_reports_unroll_metrics():
    on = _measure_side(unroll=True, repeats=1)
    assert on["loops_unrolled"] == 1
    assert on["unroll_stats"]["full_unrolls"] == 1
    assert on["unroll_stats"]["instructions_after"] > (
        on["unroll_stats"]["instructions_before"])
    assert on["pass_time_ms"] >= 0


def test_measure_is_deterministic():
    first = _measure_side(unroll=True, repeats=1)
    second = _measure_side(unroll=True, repeats=1)
    assert first["execution"]["registers"] == second["execution"]["registers"]
    assert (
        first["execution"]["dynamic_instructions"]
        == second["execution"]["dynamic_instructions"]
    )
    assert first["asm_instructions"] == second["asm_instructions"]


def test_wiring_reports_pass_presence():
    wiring = measure_wiring(CASE)
    assert wiring["on_pass_present"] and not wiring["off_pass_present"]
    assert wiring["on_pass_stats"]["loops_unrolled"] == 1


def test_evaluate_passes_all_hard_checks():
    report = evaluate(CASE, repeats=1)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["honesty"]


def test_hard_check_gate_is_not_vacuous(monkeypatch):
    """A disabled pipeline must be reported as a hard failure."""
    def fake_wiring(_case):
        return {
            "off_has_loop_markers": True,
            "on_has_loop_markers": True,
            "off_pass_present": False,
            "on_pass_present": False,
            "on_pass_stats": None,
        }

    monkeypatch.setattr(
        "benchmarks.run_topic10_unroll_case.measure_wiring", fake_wiring)
    report = evaluate(CASE, repeats=1)
    assert "pipeline_runs_pass_when_enabled" in report["hard_failures"]
    assert "loop_markers_removed_when_enabled" in report["hard_failures"]


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    exit_code = main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
        "--repeats", "1",
    ])
    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic10-loop-unroll"
    assert data["observed_registers"] == ["x10"]
    markdown = md_path.read_text()
    assert "Topic 10 Loop-Unroll Feature Case" in markdown
    assert "Dynamic instructions" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2
