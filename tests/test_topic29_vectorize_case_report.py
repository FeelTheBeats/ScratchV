"""Tests for the Topic 29 SIMD-vectorize feature case report.

The report is the CI artifact that proves the vectorize opt-in is wired
through the configured compiler pipeline, actually rewrites the canonical
element-addressing loop, and keeps architectural results identical under
the RV32 emulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic29_vectorize_case import (
    EXPECTED_OUTPUTS,
    EXPECTED_RESULT,
    SCHEMA_VERSION,
    _measure_side,
    evaluate,
    main,
    measure_dsl_wiring,
    measure_failure_matrix,
)

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic29_vectorize_feature.dsl"
)


def test_ab_sides_compile_and_on_emits_vector_ops():
    off = _measure_side(vectorize=False, repeats=1, case_path=CASE)
    on = _measure_side(vectorize=True, repeats=1, case_path=CASE)

    assert off["success"] and on["success"]
    assert off["vector_ops"] == 0
    assert on["vector_ops"] == 4
    assert on["vector_ops_by_opcode"] == {
        "vadd": 1, "vload": 1, "vrelu": 1, "vstore": 1,
    }
    assert on["assembles"] and on["binary_bytes"] > 0
    assert not on["vector_mnemonics_in_asm"]
    assert "vectorized 1/1 loop(s), width=4" in on["opt_message"]


def test_execution_is_equivalent():
    off = _measure_side(vectorize=False, repeats=1, case_path=CASE)
    on = _measure_side(vectorize=True, repeats=1, case_path=CASE)
    exec_off = off["execution"]
    exec_on = on["execution"]

    assert exec_off["a0"] == EXPECTED_RESULT == exec_on["a0"]
    assert exec_off["outputs"] == EXPECTED_OUTPUTS
    assert exec_on["outputs"] == EXPECTED_OUTPUTS
    assert exec_off["outputs"] == exec_on["outputs"]
    assert exec_off["registers"]["x10"] == exec_on["registers"]["x10"]
    assert exec_off["dynamic_instructions"] > 0
    assert exec_on["dynamic_instructions"] > 0


def test_failure_matrix_rejects_llvm_and_bad_widths():
    matrix = measure_failure_matrix(CASE)

    assert matrix["llvm_rejected"]
    assert matrix["invalid_widths_rejected"]
    assert matrix["vector_isa_rejected"]
    assert matrix["no_artifact_written_on_failure"]
    assert not matrix["llvm"]["success"]
    assert not matrix["vector_isa"]["success"]
    assert all(not row["success"] for row in matrix["invalid_widths"])


def test_compilation_is_deterministic():
    first = _measure_side(vectorize=True, repeats=2, case_path=CASE)
    second = _measure_side(vectorize=True, repeats=2, case_path=CASE)

    assert first["artifact_deterministic"]
    assert second["artifact_deterministic"]
    assert first["runs"] >= 2
    assert first["asm_sha256"] == second["asm_sha256"]


def test_dsl_case_wiring_and_evaluate_pass_all_hard_checks():
    wiring = measure_dsl_wiring(CASE)
    assert wiring["off_success"] and wiring["on_success"]
    assert wiring["on_vectorizer_ran"] and not wiring["off_vectorizer_ran"]

    report = evaluate(CASE, repeats=1)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["topic"] == "topic29-simd-vectorize"
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["honesty"]


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
    assert data["topic"] == "topic29-simd-vectorize"
    assert data["observed_registers"] == ["x10"]
    assert data["vectorize_off"]["vector_ops"] == 0
    assert data["vectorize_on"]["vector_ops"] > 0
    markdown = md_path.read_text()
    assert "Topic 29 SIMD Vectorize Feature Case" in markdown
    assert "Vector op statistics" in markdown
    assert "Execution equivalence" in markdown
    assert "Honesty" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2


def test_hard_check_gate_is_not_vacuous(monkeypatch):
    """A case without vector ops must be reported as a hard failure."""
    def fake_side(*, vectorize, repeats, case_path=None):
        side = _measure_side(vectorize=vectorize, repeats=1, case_path=CASE)
        side["vector_ops"] = 0
        side["assembles"] = False
        return side

    monkeypatch.setattr(
        "benchmarks.run_topic29_vectorize_case._measure_side", fake_side)
    report = evaluate(CASE, repeats=1)

    assert "vector_ops_present_when_enabled" in report["hard_failures"]
    assert "on_artifact_assembles" in report["hard_failures"]
