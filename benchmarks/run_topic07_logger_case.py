#!/usr/bin/env python3
"""Run one Topic 07 structured-logging feature case and emit auditable reports.

The report proves five separate facts:

1. compiling the deterministic case with structured logging enabled
   (``CompilerConfig(use_logger=True, log_level="INFO", log_file=...)``)
   succeeds and writes a staged log record covering every compiler phase;
2. the log file stays a complete DEBUG record while the console level is
   INFO, and the compiler output is byte-for-byte identical to a run with
   logging disabled (A/B output equality);
3. a syntax-error compile with logging enabled returns ``success=False``
   and still records an ERROR line without the process crashing;
4. an ``exc_info=True`` ERROR record renders its traceback into the log
   file (the D1/D2 formatter contract, probed separately so it never
   contaminates the compiler failure log);
5. repeated ``init_logger`` / ``shutdown`` cycles do not accumulate
   handlers, and repeated compiles keep the handler count stable.

This is a deterministic feature/integration case, not a real-workload
performance claim.  The timing A/B is recorded for information only; no
threshold is enforced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.utils.logger import (
    get_logger,
    init_logger,
    shutdown,
)

SCHEMA_VERSION = "topic07-logger-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic07_logger_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/logger_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/logger_report.md")

#: Console level used by the logged run.  The file handler must stay at
#: DEBUG regardless, which is what hard check 4 verifies.
LOG_LEVEL = "INFO"
#: Common pipeline configuration for both A/B sides (CLI defaults).
BASE_CONFIG: dict[str, Any] = {
    "optimize_level": "all",
    "reg_alloc": "greedy",
}
OUTPUT_NAME = "feature_case.s"
#: Phase names this branch actually emits through ``log_phase`` /
#: ``log_progress``.  ``compiler.passes`` is the PassManager progress logger
#: (``scratchv.compiler.passes``).
EXPECTED_PHASES = (
    "compiler.parse",
    "compiler.optimize",
    "compiler.passes",
    "compiler.codegen",
    "compiler.asm",
    "compiler.emit",
)
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
#: Deterministic syntax error: the DSL validator reports E101 before any
#: phase starts, so the driver must still log an ERROR summary.
FAILURE_SOURCE = "add(a, b\n"


def _one_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handler_count() -> int:
    return len(logging.getLogger("scratchv").handlers)


def _count_levels(text: str) -> dict[str, int]:
    """Count records per level in a ``_PlainFormatter`` log file."""
    counts = {name: 0 for name in LEVELS}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in counts:
            counts[parts[2]] += 1
    return counts


def _phase_coverage(text: str) -> dict[str, bool]:
    return {
        phase: f"[scratchv.{phase}]" in text for phase in EXPECTED_PHASES
    }


def _phase_records(text: str) -> dict[str, int]:
    return {
        phase: sum(
            1 for line in text.splitlines()
            if f"[scratchv.{phase}]" in line
        )
        for phase in EXPECTED_PHASES
    }


def _debug_markers(text: str) -> dict[str, bool]:
    """DEBUG-only lines that INFO console filtering must not remove."""
    return {
        "config_line": "config: backend=" in text,
        "parser_detail": "parser: extended-dsl" in text,
        "pass_detail": "pass constant-folding:" in text,
        "codegen_step": "-> instruction selection" in text,
    }


def measure_logged_run(
    case_path: Path, repeats: int, workdir: Path,
) -> dict[str, Any]:
    """Compile *case_path* *repeats* times with logging enabled.

    A fresh driver per repeat reproduces one-compile-per-process usage and
    exercises ``init_logger`` re-initialisation each time; the handler
    count after every compile must stay at two (console + file).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "compiler.log"
    output_path = workdir / OUTPUT_NAME
    shutdown()

    result = None
    times: list[float] = []
    handler_counts: list[int] = []
    for _ in range(repeats):
        driver = CompilerDriver(CompilerConfig(
            use_logger=True,
            log_level=LOG_LEVEL,
            log_file=str(log_path),
            log_color=False,
            **BASE_CONFIG,
        ))
        started = time.perf_counter()
        result = driver.compile(str(case_path), str(output_path))
        times.append((time.perf_counter() - started) * 1000.0)
        handler_counts.append(_handler_count())

    log_text = (
        log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    )
    data: dict[str, Any] = {
        "enabled": True,
        "success": bool(result and result.success),
        "errors": [_one_line(e) for e in (result.errors[:3] if result else [])],
        "runs": repeats,
        "compile_ms_samples": times,
        "compile_ms_median": statistics.median(times),
        "output_bytes": len(result.output_text.encode()) if result else 0,
        "output_sha256": _sha256_text(result.output_text) if result else "",
        "output_file": str(output_path),
        "output_file_sha256": _sha256_file(output_path),
        "handler_counts": handler_counts,
        "handler_count_after_compile": (
            handler_counts[-1] if handler_counts else 0
        ),
        "log_file": str(log_path),
        "log_file_exists": log_path.is_file(),
        "log_file_bytes": len(log_text.encode("utf-8")),
        "log_file_sha256": _sha256_text(log_text),
        "levels": _count_levels(log_text),
        "phases": _phase_coverage(log_text),
        "phase_records": _phase_records(log_text),
        "debug_markers": _debug_markers(log_text),
    }
    shutdown()
    data["handler_count_after_shutdown"] = _handler_count()
    return data


def measure_plain_run(
    case_path: Path, repeats: int, workdir: Path,
) -> dict[str, Any]:
    """Compile *case_path* *repeats* times without any logging configured."""
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / OUTPUT_NAME
    shutdown()

    result = None
    times: list[float] = []
    handler_counts: list[int] = []
    for _ in range(repeats):
        driver = CompilerDriver(CompilerConfig(**BASE_CONFIG))
        started = time.perf_counter()
        result = driver.compile(str(case_path), str(output_path))
        times.append((time.perf_counter() - started) * 1000.0)
        handler_counts.append(_handler_count())
    shutdown()

    return {
        "enabled": False,
        "success": bool(result and result.success),
        "errors": [_one_line(e) for e in (result.errors[:3] if result else [])],
        "runs": repeats,
        "compile_ms_samples": times,
        "compile_ms_median": statistics.median(times),
        "output_bytes": len(result.output_text.encode()) if result else 0,
        "output_sha256": _sha256_text(result.output_text) if result else "",
        "output_file": str(output_path),
        "output_file_sha256": _sha256_file(output_path),
        "handler_counts": handler_counts,
        "handler_count_after_compile": (
            handler_counts[-1] if handler_counts else 0
        ),
    }


def measure_failure_path(workdir: Path) -> dict[str, Any]:
    """Compile a syntax-error DSL with logging enabled and inspect the log.

    The validator reports the syntax error before any phase starts, so the
    driver returns ``success=False`` with an ERROR summary line.  The
    ``exc_info`` formatter contract is probed afterwards in a separate log
    file so the compiler failure record stays uncontaminated.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    bad_path = workdir / "bad.dsl"
    bad_path.write_text(FAILURE_SOURCE, encoding="utf-8")
    log_path = workdir / "failure.log"
    probe_log_path = workdir / "exc_info_probe.log"
    output_path = workdir / "bad.s"
    shutdown()

    driver = CompilerDriver(CompilerConfig(
        use_logger=True,
        log_level=LOG_LEVEL,
        log_file=str(log_path),
        log_color=False,
        **BASE_CONFIG,
    ))
    result = driver.compile(str(bad_path), str(output_path))
    shutdown()

    log_text = (
        log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    )
    levels = _count_levels(log_text)

    init_logger(level=LOG_LEVEL, log_file=str(probe_log_path),
                use_color=False)
    try:
        raise RuntimeError("exc_info probe")
    except RuntimeError:
        get_logger("probe.exc_info").error(
            "exception formatting probe", exc_info=True,
        )
    shutdown()
    probe_text = (
        probe_log_path.read_text(encoding="utf-8")
        if probe_log_path.is_file() else ""
    )

    return {
        "success": bool(result.success),
        "error_count": len(result.errors),
        "first_error": _one_line(result.errors[0]) if result.errors else "",
        "output_written": output_path.is_file(),
        "log_file": str(log_path),
        "log_file_exists": log_path.is_file(),
        "log_file_bytes": len(log_text.encode("utf-8")),
        "levels": levels,
        "error_records": [
            line for line in log_text.splitlines() if " ERROR " in line
        ],
        "has_error_record": levels.get("ERROR", 0) > 0,
        "exc_info_log_file": str(probe_log_path),
        "exc_info_has_traceback": (
            "Traceback (most recent call last)" in probe_text
        ),
        "exc_info_has_runtime_error": "RuntimeError: exc_info probe" in (
            probe_text
        ),
    }


def measure_handler_lifecycle(workdir: Path) -> dict[str, Any]:
    """Re-initialise the logger repeatedly and watch the handler count.

    Three console-only cycles give one handler each; three file cycles give
    two (console + file) each.  ``init_logger`` releases the previous
    handlers, so the count must never grow and ``shutdown`` must reach 0.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shutdown()
    counts: list[int] = []
    for _ in range(3):
        init_logger(level=LOG_LEVEL, use_color=False)
        counts.append(_handler_count())
        shutdown()
    for i in range(3):
        init_logger(
            level=LOG_LEVEL,
            log_file=str(workdir / f"cycle_{i}.log"),
            use_color=False,
        )
        counts.append(_handler_count())
        get_logger("lifecycle").debug("handler lifecycle probe %d", i)
        shutdown()
    after = _handler_count()
    return {
        "cycles": len(counts),
        "handler_counts": counts,
        "expected_handlers": {"console_only": 1, "console_and_file": 2},
        "max_handlers": max(counts) if counts else 0,
        "after_shutdown": after,
        "leaked": after != 0 or (bool(counts) and max(counts) > 2),
    }


def evaluate(case_path: Path, repeats: int) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        logged = measure_logged_run(case_path, repeats, workdir / "logged")
        plain = measure_plain_run(case_path, repeats, workdir / "plain")
        failure = measure_failure_path(workdir / "failure")
        lifecycle = measure_handler_lifecycle(workdir / "lifecycle")

    outputs_equal = (
        logged["success"]
        and plain["success"]
        and logged["output_sha256"] == plain["output_sha256"]
        and logged["output_file_sha256"] == plain["output_file_sha256"]
        and logged["output_bytes"] == plain["output_bytes"]
    )
    overhead_ms = logged["compile_ms_median"] - plain["compile_ms_median"]
    overhead_pct = (
        overhead_ms / plain["compile_ms_median"] * 100.0
        if plain["compile_ms_median"] else 0.0
    )

    hard_checks = {
        "case_compiles_with_logging": logged["success"],
        "case_compiles_without_logging": plain["success"],
        "log_file_created": (
            logged["log_file_exists"] and logged["log_file_bytes"] > 0
        ),
        "log_file_contains_debug_records": (
            logged["levels"]["DEBUG"] > 0
            and all(logged["debug_markers"].values())
        ),
        "log_file_records_all_phases": all(logged["phases"].values()),
        "outputs_byte_identical": outputs_equal,
        "failure_returns_unsuccessful": (
            not failure["success"] and failure["error_count"] > 0
        ),
        "failure_log_has_error_record": failure["has_error_record"],
        "exc_info_rendered_to_file": (
            failure["exc_info_has_traceback"]
            and failure["exc_info_has_runtime_error"]
        ),
        "no_handler_leak_after_reinit": (
            not lifecycle["leaked"]
            and logged["handler_count_after_shutdown"] == 0
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic07-logger",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "config": {
            "use_logger": True,
            "log_level": LOG_LEVEL,
            "log_color": False,
            **BASE_CONFIG,
        },
        "runs": repeats,
        "logged": logged,
        "plain": plain,
        "outputs_equal": outputs_equal,
        "failure": failure,
        "handler_lifecycle": lifecycle,
        "overhead_ms": round(overhead_ms, 4),
        "overhead_pct": round(overhead_pct, 2),
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": (
            "Deterministic feature case compiled by the repository's own "
            "CompilerDriver.  The timing A/B is wall-clock on a tiny case "
            "and is recorded for information only; no threshold is "
            "enforced.  Output equality is byte-level (sha256) between the "
            "logged and plain runs.  The failure path is a DSL syntax "
            "error reported by the validator, so its ERROR record carries "
            "no traceback by design; the exc_info formatter contract is "
            "probed separately through the public logger API.  Handler "
            "counts prove init_logger/shutdown do not accumulate handlers."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    logged = report["logged"]
    plain = report["plain"]
    failure = report["failure"]
    lifecycle = report["handler_lifecycle"]
    checks = report["hard_checks"]
    passed = len(checks) - len(report["hard_failures"])
    lines = [
        "# Topic 07 Structured-Logging Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}`",
        f"- Generated: {report['generated_at']}",
        f"- Config: log_level={report['config']['log_level']}, "
        f"log_color={report['config']['log_color']}, "
        f"optimize_level={report['config']['optimize_level']}, "
        f"reg_alloc={report['config']['reg_alloc']}",
        f"- Hard checks: "
        f"{'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({passed}/{len(checks)})",
        "",
        "## A/B summary",
        "",
        "| Metric | logging off | logging on | delta |",
        "|--------|------------:|-----------:|------:|",
        f"| Compile success | {plain['success']} | {logged['success']} | "
        f"same |",
        f"| Output bytes | {plain['output_bytes']} | "
        f"{logged['output_bytes']} | "
        f"{logged['output_bytes'] - plain['output_bytes']:+d} |",
        f"| Output sha256 (first 12) | `{plain['output_sha256'][:12]}` | "
        f"`{logged['output_sha256'][:12]}` | "
        f"{'identical' if report['outputs_equal'] else 'DIFFERS'} |",
        f"| Compile time (ms, median of {report['runs']}) | "
        f"{plain['compile_ms_median']:.4f} | "
        f"{logged['compile_ms_median']:.4f} | "
        f"{report['overhead_ms']:+.4f} "
        f"({report['overhead_pct']:+.1f}%) |",
        f"| Handler count after compile | "
        f"{plain['handler_count_after_compile']} | "
        f"{logged['handler_count_after_compile']} | - |",
        f"| Log file bytes | n/a | {logged['log_file_bytes']} | - |",
        f"| DEBUG records in log | n/a | {logged['levels']['DEBUG']} | - |",
        "",
        "> Timing is wall-clock on a tiny deterministic case; it is "
        "reported for information only (no pass/fail threshold).",
        "",
        "## Log phase coverage (logging on)",
        "",
        "| Phase | Present | Records |",
        "|-------|---------|--------:|",
    ]
    for phase, present in logged["phases"].items():
        lines.append(
            f"| `{phase}` | {'yes' if present else 'NO'} | "
            f"{logged['phase_records'].get(phase, 0)} |"
        )
    lines += [
        "",
        "- Level counts: " + ", ".join(
            f"{name}={count}" for name, count in logged["levels"].items()
        ),
        "- DEBUG-only markers: " + ", ".join(
            f"{name}={'yes' if ok else 'NO'}"
            for name, ok in logged["debug_markers"].items()
        ),
        "",
        "## Failure path (syntax error, logging on)",
        "",
        f"- Compile success: {failure['success']} (expected False), "
        f"errors: {failure['error_count']}",
        f"- First error: `{failure['first_error']}`",
        f"- ERROR records in log: {failure['levels']['ERROR']}",
        f"- Log file: {failure['log_file_bytes']} bytes, "
        f"exists={failure['log_file_exists']}",
        f"- Output written: {failure['output_written']} (expected False)",
        f"- exc_info formatter probe: "
        f"traceback rendered={failure['exc_info_has_traceback']}, "
        f"marker rendered={failure['exc_info_has_runtime_error']}",
        "",
        "## Handler lifecycle",
        "",
        f"- handler counts across re-inits: {lifecycle['handler_counts']}",
        f"- max handlers: {lifecycle['max_handlers']} "
        f"(console + file = 2)",
        f"- handlers after shutdown: {lifecycle['after_shutdown']}",
        f"- leaked: {lifecycle['leaked']}",
        "",
        "## Hard checks",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    lines += [
        "",
        "## Honesty",
        "",
        report["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.case.is_file():
        parser.error(f"feature case not found: {args.case}")

    report = evaluate(args.case, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report) + "\n")
    print(render_markdown(report))
    if report["hard_failures"]:
        print("HARD FAILURES: " + ", ".join(report["hard_failures"]))
        return 1
    print(f"reports written: {args.json}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
