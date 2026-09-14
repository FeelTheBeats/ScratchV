"""Tests for the Topic 17 register-allocation feature case report.

The report is the CI artifact that proves both allocator modes compile the
same deterministic DSL case, the linear path emits a real frame, and the RV32
emulator executes both products to identical architectural state.  The case
and its checks deliberately avoid the high-pressure shapes (register-pool
exhaustion, reload-time eviction) so the tests stay stable on this branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic17_regalloc_case import (
    EXPECTED_RESULT,
    SCHEMA_VERSION,
    evaluate,
    main,
    measure_allocator,
)
from scratchv.simulator.rv32_emulator import RV32Emulator

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic17_regalloc_feature.dsl"
)


def test_case_compiles_and_runs_for_both_allocators():
    for mode in ("greedy", "linear"):
        measured = measure_allocator(CASE, mode, repeats=2)
        assert measured["compile_success"], measured["errors"]
        assert measured["execution"]["a0"] == EXPECTED_RESULT
        assert measured["execution"]["sp"] == RV32Emulator.STACK_TOP
        assert measured["execution"]["dynamic_instructions"] > 0
        assert measured["asm_instructions"] > 0


def test_linear_hygiene_and_frame_evidence():
    linear = measure_allocator(CASE, "linear", repeats=2)
    assert linear["hygiene"]["clean"], linear["hygiene"]["issues"]
    assert linear["hygiene"]["assembles"]

    assert linear["spill_accesses"] > 0
    assert linear["frame_size"] > 0
    assert linear["frame"]["prologue_offsets"], "no prologue frame adjustment"
    assert (
        sum(linear["frame"]["prologue_offsets"])
        + sum(linear["frame"]["epilogue_offsets"])
        == 0
    )
    assert all(
        0 <= offset and offset + 4 <= linear["frame_size"]
        for offset in linear["spill_offsets"]
    )
    # The low-pressure case must not depend on reload-time eviction (F2).
    assert linear["eviction_count"] == 0


def test_execution_equivalent_between_allocators():
    greedy = measure_allocator(CASE, "greedy", repeats=2)
    linear = measure_allocator(CASE, "linear", repeats=2)
    for reg in ("x2", "x10"):
        assert (
            greedy["execution"]["registers"][reg]
            == linear["execution"]["registers"][reg]
        )
    assert greedy["execution"]["a0"] == EXPECTED_RESULT
    assert linear["execution"]["a0"] == EXPECTED_RESULT


def test_linear_allocation_is_deterministic():
    first = measure_allocator(CASE, "linear", repeats=3)
    second = measure_allocator(CASE, "linear", repeats=3)
    assert first["deterministic"]
    assert first["distinct_asm"] == 1
    assert second["deterministic"]
    assert first["asm_sha256"] == second["asm_sha256"]
    assert (
        first["execution"]["dynamic_instructions"]
        == second["execution"]["dynamic_instructions"]
    )


def test_evaluate_passes_all_hard_checks():
    report = evaluate(CASE, repeats=2)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["topic"] == "topic17-regalloc"
    assert report["runs"] >= 2
    assert report["hard_failures"] == []
    assert report["hard_checks"] and all(report["hard_checks"].values())
    assert report["honesty"]


def test_hard_check_gate_is_not_vacuous(monkeypatch):
    """An injected linear failure must surface in ``hard_failures``."""
    from benchmarks import run_topic17_regalloc_case as module

    real_measure = module.measure_allocator

    def fake_measure(case_path, mode, repeats):
        if mode == "linear":
            return module._failed_measurement(
                mode, ["injected allocator failure"])
        return real_measure(case_path, mode, repeats)

    monkeypatch.setattr(module, "measure_allocator", fake_measure)
    report = module.evaluate(CASE, repeats=2)
    assert "linear_compile_succeeds" in report["hard_failures"]
    assert "execution_matches_expected" in report["hard_failures"]
    assert "linear_allocation_deterministic" in report["hard_failures"]


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    exit_code = main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
        "--repeats", "2",
    ])
    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic17-regalloc"
    assert data["observed_registers"] == ["x2", "x10"]
    markdown = md_path.read_text()
    assert "Topic 17 Register-Allocation Feature Case" in markdown
    assert "A/B summary" in markdown
    assert "Hard checks" in markdown
    assert "Honesty" in markdown
    assert capsys.readouterr().out


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2
