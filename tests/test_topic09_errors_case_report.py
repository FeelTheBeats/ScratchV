"""Tests for the Topic 09 DSL-error feature case report.

The report is the CI artifact that proves the configured compiler pipeline
rejects an intentionally invalid DSL case with positioned, categorized and
rendered diagnostics, while collector capacity and strict fail-fast
semantics stay correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic09_errors_case import (
    EXPECTED_ARITY_SUGGESTION,
    EXPECTED_COLUMN_MARKER,
    EXPECTED_COMPILER_DIAGNOSTICS,
    EXPECTED_RICH_DIAGNOSTICS,
    EXPECTED_SPELLING_SUGGESTION,
    LIMIT_CASE_LINES,
    LIMIT_CASE_MAX_ERRORS,
    SCHEMA_VERSION,
    check_collector_limit,
    check_strict_mode,
    collect_rich,
    compile_case,
    evaluate,
    main,
    render_via_api,
)
from scratchv.frontend.dsl_errors import DSLSyntaxError, ErrorCode
from scratchv.frontend.dsl_extended import ExtendedDSLParser

CASE = (
    Path(__file__).resolve().parents[1]
    / "tests" / "data" / "topic09_dsl_errors_feature.dsl"
)


def test_case_compilation_fails_with_at_least_three_diagnostics():
    compiler = compile_case(CASE, CASE.read_text())
    assert compiler["success"] is False
    assert compiler["output_written"] is False
    assert compiler["diagnostics"]
    assert len(compiler["diagnostics"]) == 3
    assert len({d["error_code"] for d in compiler["diagnostics"]}) == 3


def test_compiler_diagnostic_positions_match_case():
    source = CASE.read_text()
    lines = source.split("\n")
    compiler = compile_case(CASE, source)
    assert tuple(
        (d["line"], d["col"], d["error_code"])
        for d in compiler["diagnostics"]
    ) == EXPECTED_COMPILER_DIAGNOSTICS
    for diag in compiler["diagnostics"]:
        assert diag["line"] >= 1 and diag["col"] >= 1
        assert diag["source_line"] == lines[diag["line"] - 1]
        assert diag["source_line_matches_case"] is True


def test_rich_collector_suggestions_and_render_alignment():
    source = CASE.read_text()
    collector = collect_rich(source, str(CASE))
    assert collector.error_count == 4
    assert tuple(
        (e.line, e.col, e.error_code) for e in collector.errors
    ) == EXPECTED_RICH_DIAGNOSTICS

    spelling = collector.errors[0]
    assert spelling.error_code == ErrorCode.SEM_UNKNOWN_OP
    assert spelling.fix_hint == EXPECTED_SPELLING_SUGGESTION
    arity = next(
        e for e in collector.errors if e.error_code == ErrorCode.SEM_ARITY
    )
    assert arity.fix_hint == EXPECTED_ARITY_SUGGESTION

    render = render_via_api(spelling)
    assert render["api"].endswith("render_error")
    assert render["stream_isatty"] is False
    assert render["contains_ansi"] is False
    assert EXPECTED_COLUMN_MARKER in render["text"]
    assert render["caret_aligned_with_token"] is True
    assert render["caret_col"] == render["token_col"]
    assert render["note_line"] == f"note: {EXPECTED_SPELLING_SUGGESTION}"


def test_collector_suppression_accounting():
    limit = check_collector_limit()
    assert limit["error_count"] == LIMIT_CASE_MAX_ERRORS
    assert limit["suppressed_count"] == (
        LIMIT_CASE_LINES - LIMIT_CASE_MAX_ERRORS
    )
    assert limit["suppressed_count"] == 7
    assert limit["limit_reached"] is True
    assert f"error limit ({LIMIT_CASE_MAX_ERRORS})" in limit["report_note"]
    assert "7 further errors suppressed" in limit["report_note"]


def test_strict_mode_raises_first_error():
    source = CASE.read_text()
    strict = check_strict_mode(source, str(CASE))
    assert strict["raised"] is True
    assert strict["is_dsl_syntax_error"] is True
    assert strict["exception_type"] == "DSLSyntaxError"
    expected_line, expected_col, expected_code = (
        EXPECTED_COMPILER_DIAGNOSTICS[0]
    )
    assert (
        strict["error_code"], strict["line"], strict["col"],
    ) == (expected_code, expected_line, expected_col)

    with pytest.raises(DSLSyntaxError) as excinfo:
        ExtendedDSLParser().parse(source, filename=str(CASE))
    assert excinfo.value.error_code == expected_code


def test_evaluate_passes_all_hard_checks():
    report = evaluate(CASE)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert len(report["hard_checks"]) == 15
    assert report["compiler"]["success"] is False
    assert report["collector_limit"]["suppressed_count"] == 7
    assert report["honesty"]


def test_hard_check_gate_is_not_vacuous(monkeypatch):
    """A pipeline that accepts the invalid case must fail the gate."""

    def fake_compile(_case_path, _source):
        return {
            "success": True,
            "errors": [],
            "diagnostics": [],
            "diagnostic_limit_reached": False,
            "diagnostic_limit": 20,
            "output_written": True,
        }

    monkeypatch.setattr(
        "benchmarks.run_topic09_errors_case.compile_case", fake_compile)
    report = evaluate(CASE)
    assert "compile_fails" in report["hard_failures"]
    assert "compile_writes_no_output" in report["hard_failures"]
    assert "at_least_three_diagnostics" in report["hard_failures"]
    assert "compiler_positions_match_case" in report["hard_failures"]
    # Unrelated invariants must still hold: the gate is targeted, not global.
    assert "strict_mode_raises_first_error" not in report["hard_failures"]
    assert "collector_limit_accounting" not in report["hard_failures"]


def test_main_writes_reports_and_rejects_missing_case(tmp_path, capsys):
    json_path = tmp_path / "dsl_error_report.json"
    md_path = tmp_path / "dsl_error_report.md"
    exit_code = main([
        "--case", str(CASE),
        "--json", str(json_path),
        "--markdown", str(md_path),
    ])
    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["hard_failures"] == []
    assert data["topic"] == "topic09-dsl-errors"
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["compiler"]["success"] is False
    markdown = md_path.read_text()
    assert "Topic 09 DSL-Error Feature Case" in markdown
    assert "Render sample" in markdown
    assert EXPECTED_COLUMN_MARKER in markdown
    assert "## Hard checks" in markdown
    assert "## Honesty" in markdown
    assert capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])
    assert exc.value.code == 2
