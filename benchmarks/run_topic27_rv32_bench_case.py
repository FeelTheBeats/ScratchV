#!/usr/bin/env python3
"""Topic 27 RV32-bench feature case: drive ``rv32_bench`` and audit its report.

The case runs the real ``rv32_bench.main`` driver on one tiny deterministic
ONNX model (3x3 Conv, 8x8 -> 6x6, fixed seed) and proves the honest-report
contract end to end:

1. ``rv32_bench.{json,md,html}`` artifacts are written and the JSON passes
   ``bench_report.validate_report_schema``;
2. ``completion`` stays inside the documented enumeration and a null
   comparison ratio always carries an ``incomparable_reason``;
3. ``audit_provenance`` rejects forged reports, so the honesty gate is not
   vacuous;
4. budget truncation (``--max-instructions``) and wall-clock timeout are
   labeled honestly, or explicitly reported as ``not_run`` when the
   simulator is missing.

CI has no LLVM toolchain (and may lack TinyFive), so every run uses
``--allow-missing-simulator --skip-llvm`` and degrades to a static-only
report instead of inventing dynamic numbers.  This is a deterministic
feature/integration case, not a performance claim: the embedded dynamic
counts are TinyFive emulator instruction counts on a toy model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.standalone import bench_report, rv32_bench

SCHEMA_VERSION = "topic27-rv32-bench-case/1"
RV32_SCHEMA_VERSION = "rv32-bench/2"
DEFAULT_JSON = Path("benchmark_reports/rv32_bench_case_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/rv32_bench_case_report.md")

ALLOWED_COMPLETIONS = frozenset({
    "halted", "budget_exhausted", "timeout", "error", "not_run",
})

CASE_BUILDER = "tiny_conv_8x8_k3"
BUDGET_LIMIT = 1000
TIMEOUT_PROBE_S = 0.01
TIMEOUT_PROBE_CHUNK = 1
RUN_CHUNK = 4096
RUN_TIMEOUT_S = 120.0


# ═══════════════════════════════════════════════════════════════════════════
# Case model
# ═══════════════════════════════════════════════════════════════════════════

def build_case_model(path: str | Path) -> Path:
    """Write the deterministic tiny Conv ONNX model used by the case."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = Path(path)
    rng = np.random.RandomState(0)
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 8, 8])
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 6, 6])
    weight = numpy_helper.from_array(
        (rng.randn(1, 1, 3, 3).astype(np.float32) * 0.1), "W")
    bias = numpy_helper.from_array(np.zeros(1, np.float32), "B")
    node = helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "topic27_rv32_bench_case", [inp], [out], [weight, bias])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Driver invocation
# ═══════════════════════════════════════════════════════════════════════════

def _run_rv32(model_path: Path, out_dir: Path, extra_args: list[str], *,
              timeout_s: float = RUN_TIMEOUT_S,
              chunk: int = RUN_CHUNK) -> dict[str, Any]:
    """Invoke the real ``rv32_bench.main`` in-process and collect artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(model_path),
        "--quiet",
        "--output-dir", str(out_dir),
        "--allow-missing-simulator",
        "--skip-llvm",
        "--timeout", str(timeout_s),
        "--chunk-instructions", str(chunk),
    ] + [str(arg) for arg in extra_args]
    started = time.perf_counter()
    status = "ok"
    error: str | None = None
    exit_code: int | None = None
    try:
        exit_code = rv32_bench.main(argv)
    except Exception as exc:
        status = "exception"
        error = f"{type(exc).__name__}: {exc}"[:300]
    elapsed = time.perf_counter() - started

    json_path = out_dir / "rv32_bench.json"
    md_path = out_dir / "rv32_bench.md"
    html_path = out_dir / "rv32_bench.html"
    rv_report: dict | None = None
    if json_path.is_file():
        try:
            rv_report = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            status = "invalid_json"
            error = str(exc)[:300]

    return {
        "argv": argv,
        "exit_code": exit_code,
        "status": status,
        "error": error,
        "elapsed_s": round(elapsed, 4),
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "html": str(html_path),
            "json_written": json_path.is_file(),
            "markdown_written": md_path.is_file(),
            "html_written": html_path.is_file(),
            "json_bytes": (
                json_path.stat().st_size if json_path.is_file() else None),
            "json_sha256": (
                _sha256(json_path) if json_path.is_file() else None),
        },
        "report": rv_report,
    }


def _probe(model_path: Path, out_dir: Path, extra_args: list[str],
           expected: str, *, timeout_s: float = RUN_TIMEOUT_S,
           chunk: int = RUN_CHUNK) -> dict[str, Any]:
    """Run one truncation probe and summarize its honesty evidence."""
    run = _run_rv32(
        model_path, out_dir, extra_args, timeout_s=timeout_s, chunk=chunk)
    rv = run.pop("report")
    probe: dict[str, Any] = {"expected": expected, **run}
    if rv is None:
        probe.update({
            "status": "no_report",
            "note": run.get("error") or "rv32 report not written",
            "completion": None, "source": None, "limit": None,
            "executed": None, "ops_total": None, "partial": None,
            "ratio": None, "incomparable_reason": None,
            "schema_errors": ["rv32 report not written"],
            "audit_violations": ["rv32 report not written"],
        })
        return probe

    dyn = (rv.get("scratchv") or {}).get("dynamic") or {}
    cmp_ = rv.get("comparison") or {}
    output = (rv.get("scratchv") or {}).get("output") or {}
    probe.update({
        "completion": dyn.get("completion"),
        "source": dyn.get("source"),
        "reason": dyn.get("reason"),
        "limit": dyn.get("limit"),
        "executed": dyn.get("executed"),
        "ops_total": (dyn.get("ops") or {}).get("total"),
        "partial": output.get("partial"),
        "ratio": cmp_.get("dynamic_instruction_ratio"),
        "incomparable_reason": cmp_.get("incomparable_reason"),
        "schema_errors": bench_report.validate_report_schema(rv),
        "audit_violations": rv32_bench.audit_provenance(rv),
    })
    if dyn.get("source") == "unavailable" and \
            dyn.get("completion") == "not_run":
        probe["status"] = "not_run"
        probe["note"] = dyn.get("reason") or "simulator unavailable"
    elif dyn.get("completion") == expected:
        probe["status"] = "confirmed"
        probe["note"] = None
    else:
        probe["status"] = "unexpected"
        probe["note"] = (
            f"expected completion={expected!r}, got {dyn.get('completion')!r}")
    return probe


# ═══════════════════════════════════════════════════════════════════════════
# Audit probe
# ═══════════════════════════════════════════════════════════════════════════

def _tamper(field: str, violations: list[str]) -> dict[str, Any]:
    return {
        "field": field,
        "violations": violations,
        "rejected": bool(violations),
    }


def _audit_probe(rv_report: dict | None) -> dict[str, Any]:
    """Forge a copied report and prove ``audit_provenance`` rejects it."""
    if not rv_report:
        return {
            "tampers": [],
            "rejected": False,
            "note": "rv32 report not written; audit gate cannot be probed",
        }
    tampers = []

    ratio_forgery = deepcopy(rv_report)
    ratio_forgery["comparison"] = {
        "dynamic_instruction_ratio": 0.31, "incomparable_reason": None,
    }
    tampers.append(_tamper(
        "comparison.dynamic_instruction_ratio=0.31",
        rv32_bench.audit_provenance(ratio_forgery),
    ))

    schema_forgery = deepcopy(rv_report)
    schema_forgery["schema_version"] = "rv32-bench/1"
    tampers.append(_tamper(
        "schema_version=rv32-bench/1",
        rv32_bench.audit_provenance(schema_forgery),
    ))

    dyn = (rv_report.get("scratchv") or {}).get("dynamic") or {}
    if dyn.get("source") == "simulated" and \
            isinstance(dyn.get("executed"), int):
        counters_forgery = deepcopy(rv_report)
        counters_forgery["scratchv"]["dynamic"]["executed"] = (
            dyn["executed"] + 999)
        tampers.append(_tamper(
            "scratchv.dynamic.executed+=999",
            rv32_bench.audit_provenance(counters_forgery),
        ))

    return {
        "tampers": tampers,
        "rejected": all(tamper["rejected"] for tamper in tampers),
        "note": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report assembly and hard checks
# ═══════════════════════════════════════════════════════════════════════════

def _simulator_state(rv_report: dict | None):
    if not rv_report:
        return None
    dyn = (rv_report.get("scratchv") or {}).get("dynamic") or {}
    source = dyn.get("source")
    if source == "simulated":
        return True
    if source == "unavailable":
        return False
    return None


def _ratio_is_honest(rv_report: dict | None) -> bool:
    if not rv_report:
        return False
    comparison = rv_report.get("comparison") or {}
    ratio = comparison.get("dynamic_instruction_ratio")
    if ratio is None:
        return bool(comparison.get("incomparable_reason"))
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        return False
    for side in ("scratchv", "llvm"):
        dyn = (rv_report.get(side) or {}).get("dynamic") or {}
        if dyn.get("source") != "simulated" or \
                dyn.get("completion") != "halted":
            return False
    return True


def _probe_honest(probe: dict | None, expected: str) -> bool:
    if not probe:
        return False
    if probe.get("status") == "confirmed":
        consistent = (
            probe.get("completion") == expected
            and probe.get("executed") == probe.get("ops_total")
            and probe.get("ratio") is None
            and bool(probe.get("incomparable_reason"))
            and probe.get("schema_errors") == []
            and probe.get("audit_violations") == []
        )
        if expected == "budget_exhausted":
            consistent = consistent and (
                probe.get("limit") == probe.get("executed"))
        elif expected == "timeout":
            consistent = consistent and probe.get("limit") is None
        return consistent
    if probe.get("status") == "not_run":
        return bool(probe.get("reason"))
    return False


def _hard_checks(report: dict[str, Any]) -> dict[str, bool]:
    rv_report = report.get("rv32_report")
    artifacts = (report.get("run") or {}).get("artifacts") or {}
    return {
        "rv32_json_artifact_written": bool(artifacts.get("json_written")),
        "rv32_markdown_artifact_written": bool(
            artifacts.get("markdown_written")),
        "rv32_schema_version_v2": (
            bool(rv_report)
            and rv_report.get("schema_version") == RV32_SCHEMA_VERSION
        ),
        "rv32_schema_valid": report.get("schema_errors") == [],
        "provenance_audit_clean": report.get("audit_violations") == [],
        "completion_is_legal": (
            report.get("completion") in ALLOWED_COMPLETIONS),
        "ratio_is_honest": _ratio_is_honest(rv_report),
        "github_summary_rendered": (
            "# RV32 Benchmark Summary" in (
                report.get("github_summary") or "")),
        "report_markdown_rendered": (
            "## " in (report.get("report_markdown") or "")),
        "audit_gate_rejects_forgery": bool(
            (report.get("audit_probe") or {}).get("rejected")),
        "budget_probe_honest": _probe_honest(
            report.get("budget_probe"), "budget_exhausted"),
        "timeout_probe_honest": _probe_honest(
            report.get("timeout_probe"), "timeout"),
    }


def _honesty(rv_report: dict | None, simulator_available) -> str:
    if not rv_report:
        return (
            "The rv32_bench driver did not produce a report; no dynamic or "
            "static number is claimed."
        )
    dyn = (rv_report.get("scratchv") or {}).get("dynamic") or {}
    parts = [
        "Deterministic feature case executed through the real rv32_bench CLI "
        "on a generated tiny Conv model; any dynamic counts are TinyFive "
        "emulator instruction counts, not hardware cycles, and this case "
        "makes no speedup or performance claim.",
        "LLVM compilation is always skipped (--skip-llvm) because CI has no "
        "LLVM/clang toolchain; the LLVM side is unavailable by design, so "
        "dynamic_instruction_ratio is null with an explicit "
        "incomparable_reason.",
    ]
    if simulator_available is True:
        parts.append(
            f"TinyFive executed the model "
            f"(completion={dyn.get('completion')}); the budget and wall-clock "
            "probes confirm that truncated runs are labeled "
            "budget_exhausted/timeout and excluded from comparison."
        )
    elif simulator_available is False:
        parts.append(
            "TinyFive is unavailable in this environment, so the dynamic "
            "section is source=unavailable with ops=null and the "
            "budget/timeout probes are reported as not_run instead of "
            "being fabricated."
        )
    else:
        parts.append(
            "Simulator availability could not be determined because the "
            "driver failed before simulation; dynamic data is absent and "
            "unclaimed."
        )
    return " ".join(parts)


def evaluate(model_path: str | Path, work_root: str | Path, *,
             case_builder: str | None = None) -> dict[str, Any]:
    """Run the case (full + probes) and build the auditable report payload."""
    model_path = Path(model_path).resolve()
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
            prefix="t27_case_", dir=str(work_root)) as tmp:
        root = Path(tmp)
        run = _run_rv32(model_path, root / "primary", ["--full"])
        rv_report = run.pop("report")

        completion = None
        comparison: dict[str, Any] = {
            "dynamic_instruction_ratio": None, "incomparable_reason": None,
        }
        schema_errors = ["rv32 report not written"]
        audit_violations = ["rv32 report not written"]
        github_summary = ""
        report_markdown = ""
        if rv_report is not None:
            completion = (
                (rv_report.get("scratchv") or {}).get("dynamic") or {}
            ).get("completion")
            comparison = dict(rv_report.get("comparison") or {})
            schema_errors = bench_report.validate_report_schema(rv_report)
            audit_violations = rv32_bench.audit_provenance(rv_report)
            github_summary = bench_report.render_github_summary(rv_report)
            report_markdown = bench_report.render_markdown(rv_report)

        simulator_available = _simulator_state(rv_report)

        budget_probe = _probe(
            model_path, root / "budget_probe",
            ["--max-instructions", str(BUDGET_LIMIT)], "budget_exhausted",
        )
        timeout_probe = _probe(
            model_path, root / "timeout_probe", [], "timeout",
            timeout_s=TIMEOUT_PROBE_S, chunk=TIMEOUT_PROBE_CHUNK,
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic27-rv32-bench",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": {
            "model": str(model_path),
            "sha256": _sha256(model_path),
            "bytes": model_path.stat().st_size,
            "builder": case_builder,
            "description": (
                "deterministic 3x3 Conv, input [1,1,8,8] -> output [1,1,6,6], "
                "seed 0 weights"
            ),
        },
        "config": {
            "flags": ["--allow-missing-simulator", "--skip-llvm"],
            "rv32_schema_version": RV32_SCHEMA_VERSION,
            "budget_limit": BUDGET_LIMIT,
            "timeout_probe_s": TIMEOUT_PROBE_S,
            "timeout_probe_chunk": TIMEOUT_PROBE_CHUNK,
            "simulator_available": simulator_available,
            "llvm_skipped": True,
        },
        "completion": completion,
        "comparison": comparison,
        "schema_errors": schema_errors,
        "audit_violations": audit_violations,
        "run": run,
        "github_summary": github_summary,
        "report_markdown": report_markdown,
        "budget_probe": budget_probe,
        "timeout_probe": timeout_probe,
        "audit_probe": _audit_probe(rv_report),
        "rv32_report": rv_report,
        "honesty": _honesty(rv_report, simulator_available),
    }
    checks = _hard_checks(report)
    report["hard_checks"] = checks
    report["hard_failures"] = sorted(
        name for name, ok in checks.items() if not ok)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Markdown rendering
# ═══════════════════════════════════════════════════════════════════════════

def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the case report as Markdown for stdout and job summaries."""
    rv_report = report.get("rv32_report") or {}
    model = rv_report.get("model") or {}
    env = rv_report.get("environment") or {}
    targets = rv_report.get("targets") or {}
    scratchv = rv_report.get("scratchv") or {}
    sv_compile = scratchv.get("compile") or {}
    sv_dyn = scratchv.get("dynamic") or {}
    llvm = rv_report.get("llvm") or {}
    ll_compile = llvm.get("compile") or {}
    ll_dyn = llvm.get("dynamic") or {}
    comparison = rv_report.get("comparison") or {}
    case = report.get("case") or {}
    run = report.get("run") or {}
    ratio = comparison.get("dynamic_instruction_ratio")
    tag = "PASS" if not report["hard_failures"] else "FAIL"

    lines = [
        "# Topic 27 RV32-Bench Feature Case",
        "",
        f"- Schema: `{report.get('schema_version')}` "
        f"(embedded report: `{_cell(rv_report.get('schema_version'))}`)",
        f"- Case: `{_cell(case.get('model'))}` "
        f"(sha256={str(case.get('sha256') or '')[:12]}, "
        f"{_cell(case.get('bytes'))} bytes, "
        f"builder={_cell(case.get('builder'))})",
        f"- Generated: {report.get('generated_at')}",
        f"- Hard checks: {tag} "
        f"({len(report['hard_checks']) - len(report['hard_failures'])}"
        f"/{len(report['hard_checks'])})",
        f"- Driver: rv32_bench.main exit={_cell(run.get('exit_code'))}, "
        f"status={_cell(run.get('status'))}, "
        f"elapsed={_cell(run.get('elapsed_s'))}s",
        f"- render_markdown bytes: {len(report.get('report_markdown') or '')}",
        "",
        "## RV32 report schema",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| schema_version | {_cell(rv_report.get('schema_version'))} |",
        f"| generated_at | {_cell(rv_report.get('generated_at'))} |",
        f"| model.path | {_cell(model.get('path'))} |",
        f"| model.sha256 | {_cell(model.get('sha256'))} |",
        f"| environment.python | {_cell(env.get('python'))} |",
        f"| environment.tinyfive | {_cell(env.get('tinyfive'))} |",
        f"| environment.llvmlite | {_cell(env.get('llvmlite'))} |",
        f"| targets.scratchv.isa | "
        f"{_cell((targets.get('scratchv') or {}).get('isa'))} |",
        f"| targets.llvm.isa | "
        f"{_cell((targets.get('llvm') or {}).get('isa'))} |",
        f"| scratchv.compile.status | {_cell(sv_compile.get('status'))} |",
        f"| scratchv.compile.static_insns | "
        f"{_cell(sv_compile.get('static_insns'))} |",
        f"| llvm.compile.status | {_cell(ll_compile.get('status'))} |",
        f"| scratchv.dynamic.source | {_cell(sv_dyn.get('source'))} |",
        f"| comparison.dynamic_instruction_ratio | {_cell(ratio)} |",
        "",
        "## Completion / ratio status",
        "",
        "| Side | source | completion | executed | limit |",
        "|------|--------|------------|---------:|------:|",
        f"| ScratchV | {_cell(sv_dyn.get('source'))} | "
        f"{_cell(sv_dyn.get('completion'))} | "
        f"{_cell(sv_dyn.get('executed'))} | {_cell(sv_dyn.get('limit'))} |",
        f"| LLVM | {_cell(ll_dyn.get('source'))} | "
        f"{_cell(ll_dyn.get('completion'))} | "
        f"{_cell(ll_dyn.get('executed'))} | {_cell(ll_dyn.get('limit'))} |",
        "",
    ]
    if ratio is None:
        lines.append(
            f"- ratio: **null** — "
            f"{_cell(comparison.get('incomparable_reason'))}")
    else:
        lines.append(
            f"- ratio: **{ratio:g}** (both sides simulated and halted)")
    lines += [
        "",
        "## Rendered report summary",
        "",
        report.get("github_summary") or "_rv32 report not written_",
        "",
        "## Probes",
        "",
        "| Probe | Expected | Status | completion | executed | limit | Note |",
        "|-------|----------|--------|------------|---------:|------:|------|",
    ]
    for label, probe in (
        ("budget truncation", report.get("budget_probe")),
        ("wall-clock timeout", report.get("timeout_probe")),
    ):
        probe = probe or {}
        lines.append(
            f"| {label} | {_cell(probe.get('expected'))} | "
            f"{_cell(probe.get('status'))} | "
            f"{_cell(probe.get('completion'))} | "
            f"{_cell(probe.get('executed'))} | {_cell(probe.get('limit'))} | "
            f"{_cell(probe.get('note'))} |"
        )
    audit_probe = report.get("audit_probe") or {}
    forged = ", ".join(
        tamper.get("field", "?")
        for tamper in audit_probe.get("tampers") or []
    )
    lines.append(
        f"| provenance forgery | rejected | "
        f"{'rejected' if audit_probe.get('rejected') else 'NOT rejected'} | "
        f"— | — | — | forged: {_cell(forged)} |"
    )
    lines += [
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
        report.get("honesty") or "",
        "",
        "> Exit codes: 0 = all hard checks pass, 1 = hard-check failure, "
        "2 = usage error (missing case model).",
        "",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _execute(args: argparse.Namespace, work_root: Path) -> int:
    work_root.mkdir(parents=True, exist_ok=True)
    builder: str | None = None
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        model_path = args.json.parent / "topic27_rv32_bench_feature.onnx"
        try:
            build_case_model(model_path)
        except Exception as exc:
            print(
                f"error: case_model_generation_failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        builder = CASE_BUILDER

    report = evaluate(model_path, work_root, case_builder=builder)
    markdown = render_markdown(report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    if report["hard_failures"]:
        print(
            "HARD FAILURES: " + ", ".join(report["hard_failures"]),
            file=sys.stderr,
        )
        return 1
    print(f"reports written: {args.json}, {args.markdown}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Existing ONNX model; default generates the tiny Conv case model",
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Directory for intermediate artifacts (default: temporary dir)",
    )
    args = parser.parse_args(argv)

    if args.model is not None and not args.model.is_file():
        print(f"error: case model not found: {args.model}", file=sys.stderr)
        return 2

    if args.work_dir is not None:
        return _execute(args, Path(args.work_dir))
    with tempfile.TemporaryDirectory(prefix="topic27_rv32_case_") as tmp:
        return _execute(args, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
