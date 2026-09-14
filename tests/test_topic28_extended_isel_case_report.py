"""Tests for the Topic 28 extended instruction-selection feature case report.

The report is the CI artifact that proves the ``--extended-isel`` opt-in is
wired through ``CompilerDriver``, that the extended-only opcodes select FP
mnemonics under ``extended_isel=True``, that the RV32IM encoder gate fails
loud on F/D assembly while accepting FP-mnemonic-free extended assembly, and
that the counterexample flag combinations are recorded explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic28_extended_isel_case import (
    EXPECTED_DSL_RESULT,
    REQUIRED_FP_MNEMONICS,
    SCHEMA_VERSION,
    build_fp_feature_program,
    build_integer_extended_program,
    compile_ir_program,
    evaluate,
    main,
    measure_dsl_ab,
)

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic28_extended_isel_feature.dsl"
)


def test_dsl_case_compiles_in_both_configurations():
    """Both A/B configurations compile and execute to the expected result."""
    ab = measure_dsl_ab(CASE)
    assert ab["off"]["success"], ab["off"]["errors"]
    assert ab["on"]["success"], ab["on"]["errors"]
    assert ab["off"]["asm_instructions"] > 0
    assert ab["on"]["asm_instructions"] > 0
    assert ab["off"]["fp_mnemonics"] == []
    assert ab["on"]["fp_mnemonics"] == []
    assert ab["off"]["execution"]["a0"] == EXPECTED_DSL_RESULT
    assert ab["on"]["execution"]["a0"] == EXPECTED_DSL_RESULT
    assert ab["off"]["execution"]["dynamic_instructions"] > 0
    assert (
        ab["off"]["execution"]["registers"]
        == ab["on"]["execution"]["registers"]
    )


def test_extended_probe_reports_fp_mnemonics():
    """Extended-only IR shapes must select the expected FP mnemonics."""
    probe = compile_ir_program(build_fp_feature_program())
    for mnemonic in REQUIRED_FP_MNEMONICS:
        assert mnemonic in probe["fp_mnemonics"]


def test_encoder_gate_rejects_fp_and_accepts_integer_extended():
    """F/D assembly is rejected fail-loud; F/D-free extended asm encodes."""
    fp = compile_ir_program(build_fp_feature_program())
    assert fp["encoder"]["encoded"] is False
    assert fp["encoder"]["error_type"] == "UnsupportedInstructionError"

    integer = compile_ir_program(build_integer_extended_program())
    assert integer["fp_mnemonics"] == []
    assert integer["encoder"]["encoded"] is True
    assert integer["encoder"]["words"] > 0


def test_evaluate_passes_hard_checks_and_records_matrix():
    report = evaluate(CASE)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["honesty"]

    rows = {row["id"]: row for row in report["behavior_matrix"]}
    assert rows["extended_isel_off"]["status"] == "error"
    assert rows["extended_isel_no_fp64"]["status"] == "error"
    assert "enable_fp64" in rows["extended_isel_no_fp64"]["observed"]
    assert rows["llvm_backend"]["status"] == "warning"
    assert rows["dag_isel_precedence"]["status"] == "warning"
    assert rows["fp64_flag_without_extended"]["status"] == "warning"
    assert rows["hardware_sqrt_without_extended"]["status"] == "warning"

    fp_execution = report["execution"]["fp_probe"]
    assert fp_execution["status"] == "skipped"
    assert fp_execution["reason"]


def test_measurements_are_deterministic():
    first_ab = measure_dsl_ab(CASE)
    second_ab = measure_dsl_ab(CASE)
    assert first_ab["off"]["deterministic"] and first_ab["on"]["deterministic"]
    assert (
        first_ab["off"]["asm_sha256"] == second_ab["off"]["asm_sha256"]
    )
    assert (
        first_ab["on"]["asm_sha256"] == second_ab["on"]["asm_sha256"]
    )
    assert first_ab["identical_asm"] is True

    first = compile_ir_program(build_fp_feature_program())
    second = compile_ir_program(build_fp_feature_program())
    assert first["asm_sha256"] == second["asm_sha256"]


def test_hard_check_gate_is_not_vacuous(monkeypatch):
    """A degenerate extended probe must be reported as hard failures."""
    degenerate_asm = "  add a0, x1, x2\n"

    def fake_compile_ir_program(_program, **_kwargs):
        return {
            "success": True,
            "asm": degenerate_asm,
            "asm_instructions": 1,
            "fp_mnemonics": [],
            "asm_sha256": "0" * 64,
            "encoder": {
                "encoded": True,
                "bytes": 4,
                "words": 1,
                "error_type": None,
                "error": None,
            },
        }

    monkeypatch.setattr(
        "benchmarks.run_topic28_extended_isel_case.compile_ir_program",
        fake_compile_ir_program,
    )
    report = evaluate(CASE)
    assert report["hard_failures"]
    assert "extended_probe_has_fp_mnemonics" in report["hard_failures"]
    assert "fp_asm_rejected_by_encoder" in report["hard_failures"]
    assert "base_selector_rejects_extended_ops" in report["hard_failures"]


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    exit_code = main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
    ])
    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic28-extended-isel"
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["honesty"]
    markdown = md_path.read_text()
    assert "Topic 28 Extended Instruction-Selection Feature Case" in markdown
    assert "Failure / degradation matrix" in markdown
    assert "Hard checks" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2
