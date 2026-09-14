#!/usr/bin/env python3
"""Run one Topic 09 DSL-error feature case and emit auditable CI reports.

The checked-in case is *intentionally invalid*.  The report proves four
separate facts:

1. the configured compiler pipeline rejects the case through its
   pre-validation pass (``CompilerDriver.compile(dsl_source=...)`` returns
   ``success=False`` with multiple structured diagnostics) and writes no
   assembly output;
2. the rich collector mode reports the same failures with richer error
   codes (E3xx), spelling/arity suggestions and end positions;
3. rendered diagnostics use gcc/clang-style gutters and caret spans aligned
   with the offending token (embedded verbatim in the Markdown report);
4. collector capacity semantics (``suppressed_count`` / ``limit_reached``)
   and strict fail-fast semantics (first ``DSLSyntaxError``) are correct.

This is a deterministic error-path feature/integration case, not a
real-workload performance claim.  Real benchmark numbers remain separate in
``run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_errors import (
    DSLSyntaxError,
    ErrorCode,
    ErrorCollector,
    format_error,
    render_error,
)
from scratchv.frontend.dsl_extended import ExtendedDSLParser

SCHEMA_VERSION = "topic09-dsl-error-case/1"
#: Intentionally invalid DSL fixture.  Kept under ``tests/data`` (not
#: ``benchmarks/cases``) so the CI bench runner never treats a fixture that
#: must fail to parse as a benchmark case.
DEFAULT_CASE = (
    Path(__file__).resolve().parents[1]
    / "tests" / "data" / "topic09_dsl_errors_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/dsl_error_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/dsl_error_report.md")

#: Program-generated overflow input for collector capacity semantics.
LIMIT_CASE_LINES = 12
LIMIT_CASE_MAX_ERRORS = 5

#: Exact diagnostics the compiler pre-validation path must emit for the
#: checked-in case, as ``(line, column, error_code)`` in the validator code
#: space (E1xx lexical / E2xx syntax).  Any drift fails a hard check.
EXPECTED_COMPILER_DIAGNOSTICS: tuple[tuple[int, int, str], ...] = (
    (5, 5, "E200"),   # unsupported operation 'retrun'
    (6, 9, "E201"),   # operation 'add' expects 2 positional argument(s)
    (9, 1, "E110"),   # 'endwhile' without matching 'while'
)

#: Exact diagnostics the rich collector path must emit, as
#: ``(line, column, error_code)`` in the rich code space (E3xx etc.).
EXPECTED_RICH_DIAGNOSTICS: tuple[tuple[int, int, str], ...] = (
    (5, 5, ErrorCode.SEM_UNKNOWN_OP),          # E301 unknown operation
    (6, 5, ErrorCode.SEM_ARITY),               # E302 arity mismatch
    (7, 1, ErrorCode.SYN_MISSING_TERMINATOR),  # E203 missing 'endif'
    (9, 1, ErrorCode.SYN_STRAY_TERMINATOR),    # E204 stray 'endwhile'
)

EXPECTED_SPELLING_SUGGESTION = "did you mean 'return'?"
EXPECTED_ARITY_SUGGESTION = "add() requires exactly 2 arguments"
EXPECTED_COLUMN_MARKER = "^~~~~~"


def source_lines(source: str) -> list[str]:
    """Split source into physical lines with CRLF/CR normalized."""
    return source.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def diagnostic_payload(err: DSLSyntaxError, source: str) -> dict[str, Any]:
    """Convert an error into a JSON-friendly, auditable payload."""
    rendered = format_error(err, use_color=False)
    rendered_lines = rendered.splitlines()
    marker = rendered_lines[2] if len(rendered_lines) > 2 else ""
    lines = source_lines(source)
    return {
        "error_code": err.error_code,
        "line": err.line,
        "col": err.col,
        "end_col": err.end_col,
        "message": err.message,
        "suggestion": err.fix_hint,
        "source_line": err.source_line,
        "rendered": rendered,
        "marker": marker,
        "has_marker": "^" in marker and "|" in marker,
        "source_line_matches_case": (
            0 < err.line <= len(lines) and err.source_line == lines[err.line - 1]
        ),
    }


def compile_case(case_path: Path, source: str) -> dict[str, Any]:
    """Compile inline source and record the failure diagnostics.

    The case path is passed for diagnostic filenames only; the source text is
    supplied through ``dsl_source`` as required.
    """
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "topic09_dsl_error_case.s"
        result = CompilerDriver(CompilerConfig()).compile(
            str(case_path), str(output_path), dsl_source=source,
        )
        output_written = output_path.exists()
    return {
        "success": result.success,
        "errors": list(result.errors),
        "diagnostics": [
            diagnostic_payload(err, source) for err in result.diagnostics
        ],
        "diagnostic_limit_reached": result.diagnostic_limit_reached,
        "diagnostic_limit": result.diagnostic_limit,
        "output_written": output_written,
    }


def collect_rich(source: str, filename: str) -> ErrorCollector:
    """Parse in collector mode and return the populated collector."""
    collector = ErrorCollector(
        filename=filename, use_color=False, source=source,
    )
    ExtendedDSLParser().parse(source, filename=filename, collector=collector)
    return collector


def check_collector_limit() -> dict[str, Any]:
    """Feed a generated input larger than ``max_errors`` and record limits."""
    generated = "\n".join(
        f"v{i} = retrun(x, {i})" for i in range(LIMIT_CASE_LINES)
    ) + "\n"
    collector = ErrorCollector(
        filename="<generated-limit-case>",
        use_color=False,
        max_errors=LIMIT_CASE_MAX_ERRORS,
    )
    ExtendedDSLParser().parse(
        generated, filename="<generated-limit-case>", collector=collector,
    )
    return {
        "generated_lines": LIMIT_CASE_LINES,
        "max_errors": LIMIT_CASE_MAX_ERRORS,
        "error_count": collector.error_count,
        "suppressed_count": collector.suppressed_count,
        "limit_reached": collector.limit_reached,
        "expected_suppressed": LIMIT_CASE_LINES - LIMIT_CASE_MAX_ERRORS,
        "report_note": collector.report().splitlines()[-1],
    }


def check_strict_mode(source: str, filename: str) -> dict[str, Any]:
    """Parse with ``collector=None``; the first error must raise."""
    try:
        ExtendedDSLParser().parse(source, filename=filename)
    except DSLSyntaxError as err:
        return {
            "raised": True,
            "is_dsl_syntax_error": True,
            "exception_type": type(err).__name__,
            "error_code": err.error_code,
            "line": err.line,
            "col": err.col,
            "message": err.message,
            "suggestion": err.fix_hint,
            "rendered": format_error(err, use_color=False),
        }
    except Exception as err:  # pragma: no cover - defensive
        return {
            "raised": True,
            "is_dsl_syntax_error": False,
            "exception_type": type(err).__name__,
            "error_code": None,
            "line": None,
            "col": None,
            "message": str(err),
            "suggestion": None,
            "rendered": "",
        }
    return {
        "raised": False,
        "is_dsl_syntax_error": False,
        "exception_type": None,
        "error_code": None,
        "line": None,
        "col": None,
        "message": "",
        "suggestion": None,
        "rendered": "",
    }


def render_via_api(err: DSLSyntaxError) -> dict[str, Any]:
    """Render one diagnostic through the stream-aware public API.

    ``render_error`` picks the color mode from the destination stream; a
    plain ``io.StringIO`` is not a TTY, so the output must stay ANSI-free.
    The caret is compared against the token visible in the rendered source
    line, which is exactly the gcc/clang column-alignment contract.
    """
    stream = io.StringIO()
    text = render_error(err, stream=stream, use_color=False)
    rendered_lines = text.splitlines()
    source_display = rendered_lines[1] if len(rendered_lines) > 1 else ""
    marker = rendered_lines[2] if len(rendered_lines) > 2 else ""
    caret_col = marker.index("^") if "^" in marker else None
    token_col = source_display.find("retrun")
    return {
        "api": "scratchv.frontend.dsl_errors.render_error",
        "stream_isatty": bool(getattr(stream, "isatty", lambda: False)()),
        "contains_ansi": "\033[" in text,
        "text": text,
        "source_display": source_display,
        "marker": marker,
        "caret_col": caret_col,
        "token_col": token_col,
        "has_column_marker": EXPECTED_COLUMN_MARKER in marker,
        "caret_aligned_with_token": (
            caret_col is not None and caret_col == token_col
        ),
        "note_line": rendered_lines[3] if len(rendered_lines) > 3 else "",
    }


def evaluate(case_path: Path) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    source = case_path.read_text()
    filename = str(case_path)

    compiler = compile_case(case_path, source)
    collector = collect_rich(source, filename)
    rich_diagnostics = [
        diagnostic_payload(err, source) for err in collector.errors
    ]
    limit = check_collector_limit()
    strict = check_strict_mode(source, filename)

    spelling_err = next(
        (err for err in collector.errors
         if err.error_code == ErrorCode.SEM_UNKNOWN_OP),
        None,
    )
    render = (
        render_via_api(spelling_err) if spelling_err is not None
        else {
            "api": "scratchv.frontend.dsl_errors.render_error",
            "stream_isatty": False,
            "contains_ansi": False,
            "text": "",
            "source_display": "",
            "marker": "",
            "caret_col": None,
            "token_col": None,
            "has_column_marker": False,
            "caret_aligned_with_token": False,
            "note_line": "",
        }
    )

    compiler_positions = tuple(
        (d["line"], d["col"], d["error_code"])
        for d in compiler["diagnostics"]
    )
    rich_positions = tuple(
        (d["line"], d["col"], d["error_code"]) for d in rich_diagnostics
    )
    first_expected_line, first_expected_col, first_expected_code = (
        EXPECTED_COMPILER_DIAGNOSTICS[0]
    )

    hard_checks = {
        "compile_fails": compiler["success"] is False,
        "compile_writes_no_output": (
            compiler["success"] is False and not compiler["output_written"]
        ),
        "at_least_three_diagnostics": (
            len(compiler["diagnostics"]) >= 3
        ),
        "at_least_two_error_codes": (
            len({d["error_code"] for d in compiler["diagnostics"]}) >= 2
        ),
        "all_diagnostics_have_positions": all(
            d["line"] >= 1 and d["col"] >= 1
            for d in compiler["diagnostics"]
        ),
        "compiler_positions_match_case": (
            compiler_positions == EXPECTED_COMPILER_DIAGNOSTICS
        ),
        "rich_positions_match_case": (
            rich_positions == EXPECTED_RICH_DIAGNOSTICS
        ),
        "diagnostic_source_lines_match_case": all(
            d["source_line_matches_case"]
            for d in compiler["diagnostics"] + rich_diagnostics
        ),
        "spelling_suggestion_present": bool(
            spelling_err is not None
            and any(
                d["error_code"] == ErrorCode.SEM_UNKNOWN_OP
                and d["suggestion"] == EXPECTED_SPELLING_SUGGESTION
                for d in rich_diagnostics
            )
            and any(
                d["error_code"] == "E200"
                and EXPECTED_SPELLING_SUGGESTION in d["rendered"]
                for d in compiler["diagnostics"]
            )
        ),
        "arity_suggestion_present": any(
            d["error_code"] == ErrorCode.SEM_ARITY
            and d["suggestion"] == EXPECTED_ARITY_SUGGESTION
            for d in rich_diagnostics
        ),
        "render_gutter_and_caret_aligned": bool(
            render["has_column_marker"]
            and render["caret_aligned_with_token"]
        ),
        "render_uses_stream_api_without_ansi": bool(
            render["api"] == "scratchv.frontend.dsl_errors.render_error"
            and not render["contains_ansi"]
            and EXPECTED_COLUMN_MARKER in render["text"]
        ),
        "all_rich_diagnostics_render_markers": all(
            d["has_marker"] for d in rich_diagnostics
        ),
        "collector_limit_accounting": (
            limit["error_count"] == limit["max_errors"]
            and limit["suppressed_count"] == limit["expected_suppressed"]
            and limit["limit_reached"] is True
        ),
        "strict_mode_raises_first_error": bool(
            strict["raised"]
            and strict["is_dsl_syntax_error"]
            and (
                strict["error_code"], strict["line"], strict["col"],
            ) == (
                first_expected_code, first_expected_line, first_expected_col,
            )
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic09-dsl-errors",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "expected_compiler_diagnostics": [
            list(item) for item in EXPECTED_COMPILER_DIAGNOSTICS
        ],
        "expected_rich_diagnostics": [
            list(item) for item in EXPECTED_RICH_DIAGNOSTICS
        ],
        "compiler": compiler,
        "rich_collector": {
            "error_count": collector.error_count,
            "suppressed_count": collector.suppressed_count,
            "limit_reached": collector.limit_reached,
            "max_errors": collector.max_errors,
            "diagnostics": rich_diagnostics,
        },
        "render": render,
        "collector_limit": limit,
        "strict": strict,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": (
            "Deterministic error-path feature case: the checked-in DSL is "
            "intentionally invalid, so the audited facts are diagnostic "
            "counts, positions, suggestions and rendering -- not runtime or "
            "performance data.  Strict pre-validation uses the E1xx/E2xx "
            "code space while rich collection uses E2xx/E3xx; both are "
            "recorded so the two code spaces stay distinguishable."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    rich = report["rich_collector"]
    limit = report["collector_limit"]
    strict = report["strict"]
    render = report["render"]
    lines = [
        "# Topic 09 DSL-Error Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}`",
        f"- Generated: {report['generated_at']}",
        f"- Hard checks: "
        f"{'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({len(report['hard_checks']) - len(report['hard_failures'])}"
        f"/{len(report['hard_checks'])})",
        "",
        "## Compiler diagnostics (pre-validation path)",
        "",
        f"- `CompilerDriver.compile(dsl_source=...)`: "
        f"success={compiler['success']}, "
        f"output_written={compiler['output_written']}, "
        f"diagnostics={len(compiler['diagnostics'])}",
        "",
        "| code | line | col | end_col | suggestion | message |",
        "|------|-----:|----:|--------:|------------|---------|",
    ]
    for d in compiler["diagnostics"]:
        end_col = d["end_col"] if d["end_col"] is not None else "-"
        lines.append(
            f"| `{d['error_code']}` | {d['line']} | {d['col']} | "
            f"{end_col} | {d['suggestion'] or '-'} | {d['message']} |"
        )
    lines += [
        "",
        "## Rich collector diagnostics",
        "",
        "| code | line | col | end_col | suggestion | message |",
        "|------|-----:|----:|--------:|------------|---------|",
    ]
    for d in rich["diagnostics"]:
        end_col = d["end_col"] if d["end_col"] is not None else "-"
        lines.append(
            f"| `{d['error_code']}` | {d['line']} | {d['col']} | "
            f"{end_col} | {d['suggestion'] or '-'} | {d['message']} |"
        )
    lines += [
        "",
        f"- errors={rich['error_count']}, "
        f"suppressed={rich['suppressed_count']}, "
        f"limit_reached={rich['limit_reached']}",
        "",
        "## Render sample (spelling error via render_error)",
        "",
        "```text",
        render["text"],
        "```",
        "",
        f"- marker=`{render['marker']}` "
        f"caret_col={render['caret_col']} token_col={render['token_col']} "
        f"aligned={render['caret_aligned_with_token']} "
        f"ansi={render['contains_ansi']}",
        "",
        "## Collector capacity (generated overflow case)",
        "",
        f"- generated_lines={limit['generated_lines']}, "
        f"max_errors={limit['max_errors']}, "
        f"errors={limit['error_count']}, "
        f"suppressed={limit['suppressed_count']} "
        f"(expected {limit['expected_suppressed']}), "
        f"limit_reached={limit['limit_reached']}",
        f"- {limit['report_note']}",
        "",
        "## Strict mode (collector=None)",
        "",
        f"- raised={strict['raised']} type={strict['exception_type']} "
        f"code={strict['error_code']} line={strict['line']} "
        f"col={strict['col']}",
        f"- message: {strict['message']}",
        f"- suggestion: {strict['suggestion'] or '-'}",
        "",
        "## Hard checks",
        "",
    ]
    for name, ok in report["hard_checks"].items():
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
    args = parser.parse_args(argv)
    if not args.case.is_file():
        parser.error(f"feature case not found: {args.case}")

    report = evaluate(args.case)
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
