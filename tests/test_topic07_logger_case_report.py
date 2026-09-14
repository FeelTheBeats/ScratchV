"""Tests for the Topic 07 structured-logging feature case report.

The report is the CI artifact that proves the compiler logger produces a
complete staged DEBUG record, keeps the generated output byte-identical to
a run without logging, records ERROR on a failing compile, and does not
leak handlers across init/shutdown cycles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import benchmarks.run_topic07_logger_case as case_report
from benchmarks.run_topic07_logger_case import (
    SCHEMA_VERSION,
    evaluate,
    main,
    measure_failure_path,
    measure_handler_lifecycle,
    measure_logged_run,
    measure_plain_run,
)
from scratchv.utils.logger import init_logger, shutdown

CASE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "cases" / "topic07_logger_feature.dsl"
)


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    shutdown()


def test_logged_compile_records_all_phases_with_debug(tmp_path):
    logged = measure_logged_run(CASE, repeats=1, workdir=tmp_path / "logged")

    assert logged["success"]
    assert logged["log_file_exists"] and logged["log_file_bytes"] > 0
    assert logged["levels"]["DEBUG"] > 0
    assert logged["levels"]["INFO"] > 0
    assert all(logged["phases"].values()), logged["phases"]
    assert all(logged["phase_records"].values()), logged["phase_records"]
    assert all(logged["debug_markers"].values()), logged["debug_markers"]


def test_ab_outputs_are_byte_identical(tmp_path):
    logged = measure_logged_run(CASE, repeats=1, workdir=tmp_path / "logged")
    plain = measure_plain_run(CASE, repeats=1, workdir=tmp_path / "plain")

    assert logged["success"] and plain["success"]
    assert logged["output_sha256"] == plain["output_sha256"]
    assert logged["output_file_sha256"] == plain["output_file_sha256"]
    assert logged["output_bytes"] == plain["output_bytes"]
    assert (Path(logged["output_file"]).read_bytes()
            == Path(plain["output_file"]).read_bytes())


def test_failure_path_records_error_and_exc_info(tmp_path):
    failure = measure_failure_path(tmp_path / "failure")

    assert failure["success"] is False
    assert failure["error_count"] >= 1
    assert failure["log_file_exists"]
    assert failure["has_error_record"]
    assert any("compilation failed" in record
               for record in failure["error_records"])
    assert failure["output_written"] is False
    # The exc_info contract is probed separately from the compiler log.
    assert failure["exc_info_has_traceback"]
    assert failure["exc_info_has_runtime_error"]


def test_hard_check_gate_is_not_vacuous(monkeypatch, tmp_path):
    """Broken logging/output must be reported as hard failures."""
    real = measure_logged_run(CASE, repeats=1, workdir=tmp_path / "real")
    broken = dict(real)
    broken["success"] = False
    broken["log_file_exists"] = False
    broken["log_file_bytes"] = 0
    broken["levels"] = {name: 0 for name in real["levels"]}
    broken["debug_markers"] = {name: False for name in real["debug_markers"]}
    broken["output_sha256"] = "0" * 64
    broken["output_file_sha256"] = "0" * 64

    monkeypatch.setattr(
        case_report, "measure_logged_run", lambda *args, **kwargs: broken)

    report = evaluate(CASE, repeats=1)
    assert "case_compiles_with_logging" in report["hard_failures"]
    assert "log_file_created" in report["hard_failures"]
    assert "log_file_contains_debug_records" in report["hard_failures"]
    assert "outputs_byte_identical" in report["hard_failures"]


def test_evaluate_passes_all_hard_checks(tmp_path):
    report = evaluate(CASE, repeats=2)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["topic"] == "topic07-logger"
    assert report["hard_failures"] == []
    assert all(report["hard_checks"].values())
    assert report["outputs_equal"] is True
    assert report["honesty"]
    assert report["runs"] == 2
    assert report["logged"]["handler_counts"] == [2, 2]
    assert report["logged"]["handler_count_after_shutdown"] == 0
    assert report["overhead_ms"] == pytest.approx(
        report["logged"]["compile_ms_median"]
        - report["plain"]["compile_ms_median"],
        abs=1e-3,
    )


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
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["hard_failures"] == []
    assert data["outputs_equal"] is True
    markdown = md_path.read_text()
    assert "Topic 07 Structured-Logging Feature Case" in markdown
    assert "A/B summary" in markdown
    assert "Failure path" in markdown
    assert "Hard checks" in markdown
    assert "Honesty" in markdown
    assert "Topic 07 Structured-Logging Feature Case" in (
        capsys.readouterr().out
    )


def test_main_rejects_missing_case(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--case", str(tmp_path / "missing.dsl")])

    assert exc.value.code == 2


def test_repeated_init_shutdown_does_not_leak_handlers(tmp_path):
    lifecycle = measure_handler_lifecycle(tmp_path / "lifecycle")

    assert lifecycle["handler_counts"] == [1, 1, 1, 2, 2, 2]
    assert lifecycle["max_handlers"] == 2
    assert lifecycle["after_shutdown"] == 0
    assert lifecycle["leaked"] is False

    # Direct double-init check: the old handlers must be released first.
    init_logger(level="INFO", use_color=False)
    assert len(logging.getLogger("scratchv").handlers) == 1
    init_logger(level="INFO", use_color=False)
    assert len(logging.getLogger("scratchv").handlers) == 1
    shutdown()
    assert len(logging.getLogger("scratchv").handlers) == 0
