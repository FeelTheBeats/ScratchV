"""Tests for the Topic 16 LLVM-codegen feature case report.

The report is the CI artifact that proves the real ``CompilerDriver`` with
``backend="llvm"`` compiles the deterministic DSL case, that the emitted
module is structurally legal text with canonical NN-operator lowering, and
that the optional LLVM toolchain degrades gracefully when absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic16_llvm_case import (
    SCHEMA_VERSION,
    TARGET_TRIPLE,
    analyze_module,
    check_lowering,
    check_target_triple,
    compile_case,
    evaluate,
    main,
    run_toolchain,
)

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic16_llvm_feature.dsl"
)


def test_case_compiles_and_ir_is_nonempty():
    info = compile_case(CASE)
    assert info["success"], info["errors"]
    assert info["ir_text"].strip()
    assert "define float @main(" in info["ir_text"]
    assert info["elapsed_ms"] >= 0


def test_module_structure_ssa_unique_blocks_terminated_metrics_positive():
    stats = analyze_module(compile_case(CASE)["ir_text"])
    assert stats["function_count"] == 1
    assert stats["basic_block_count"] >= 12
    assert stats["instruction_count"] > 0
    assert stats["ssa_definition_count"] > 0
    assert stats["load_count"] > 0
    assert stats["store_count"] > 0
    assert stats["alloca_count"] > 0
    assert stats["gep_count"] > 0
    assert stats["ssa_definitions_unique"]
    assert stats["labels_unique"]
    assert stats["all_blocks_terminated"]
    assert stats["duplicate_ssa_definitions"] == []
    assert stats["duplicate_labels"] == []
    assert stats["unterminated_blocks"] == []


def test_nn_operator_lowering_markers_are_present():
    lowering = check_lowering(compile_case(CASE)["ir_text"])
    assert all(lowering["markers"].values()), lowering["markers"]
    counts = lowering["counts"]
    assert counts["fmul"] > 0 and counts["fadd"] > 0
    assert counts["getelementptr"] > 0
    assert counts["expf_calls"] > 0
    assert counts["tanhf_calls"] > 0
    for prefix in (
        "dot_i_hdr", "mm_i_hdr", "mm_j_hdr", "mm_k_hdr",
        "sm_max_i_hdr", "sm_sum_i_hdr", "sm_div_i_hdr",
    ):
        assert any(
            label.startswith(prefix)
            for label in lowering["loop_header_labels"]
        ), prefix


def test_target_triple_override_and_body_stability():
    triple = check_target_triple(CASE.read_text(encoding="utf-8"))
    assert triple["explicit_triple"] == TARGET_TRIPLE
    assert not triple["default_has_triple"]
    assert triple["explicit_has_triple"]
    assert triple["body_identical_without_triple"]


def test_evaluate_passes_all_hard_checks():
    report = evaluate(CASE)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["topic"] == "topic16-llvm-codegen"
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["honesty"]

    tools = report["toolchain"]
    asm = tools["llvm-as"]
    if asm["available"]:
        assert asm["status"] == "assembled"
        assert asm["returncode"] == 0
    else:
        assert asm["status"] == "skipped"
        assert asm["skip_reason"]
    opt = tools["opt"]
    if opt["available"]:
        assert opt["status"] == "optimized"
        assert opt["returncode"] == 0
    else:
        assert opt["status"] == "skipped"
        assert opt["skip_reason"]
    lli = tools["lli"]
    assert lli["status"] in ("executed", "skipped")
    if lli["status"] == "skipped":
        assert lli["skip_reason"]


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "llvm_codegen_report.json"
    md_path = tmp_path / "llvm_codegen_report.md"
    exit_code = main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
    ])
    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic16-llvm-codegen"
    assert data["module"]["function_count"] == 1
    markdown = md_path.read_text()
    assert "Topic 16 LLVM Codegen Feature Case" in markdown
    assert "Toolchain matrix" in markdown
    assert "Honesty" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2


def test_degradation_paths_are_not_vacuous(monkeypatch, tmp_path):
    """Missing tools and unlowerable cases must be reported explicitly."""
    monkeypatch.setattr(
        "benchmarks.run_topic16_llvm_case.shutil.which", lambda _name: None)
    tools = run_toolchain(
        "define float @main() {\nentry:\n  ret float 0.0\n}\n"
    )
    assert set(tools) == {"llvm-as", "opt", "lli"}
    for name, row in tools.items():
        assert row["available"] is False
        assert row["status"] == "skipped"
        assert row["skip_reason"] == f"{name} not installed"
    assert run_toolchain("")["llvm-as"]["skip_reason"] == (
        "no IR module was produced")

    bad_case = tmp_path / "unsupported_shape.dsl"
    bad_case.write_text("c = matmul(x, w, m:2, n:2, k:2)\nreturn c\n")
    report = evaluate(bad_case)
    assert "case_compiles_via_llvm_driver" in report["hard_failures"]
    assert "target_triple_is_configurable" in report["hard_failures"]
    assert report["target_triple"]["status"] == "error"

    def fake_compile(_case):
        return {
            "success": False,
            "errors": ["codegen failed"],
            "warnings": [],
            "elapsed_ms": 0.0,
            "opt_message": "",
            "ir_text": "",
        }

    monkeypatch.setattr(
        "benchmarks.run_topic16_llvm_case.compile_case", fake_compile)
    report = evaluate(CASE)
    assert "case_compiles_via_llvm_driver" in report["hard_failures"]
    assert "ir_module_is_nonempty" in report["hard_failures"]
    assert "nn_operator_lowering_markers_present" in report["hard_failures"]
