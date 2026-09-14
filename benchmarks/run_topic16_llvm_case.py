#!/usr/bin/env python3
"""Run one Topic 16 LLVM-codegen feature case and emit auditable CI reports.

The report proves three separate facts:

1. the real ``CompilerDriver`` configured with ``backend="llvm"`` compiles the
   deterministic DSL case into a non-empty LLVM IR module;
2. the emitted module satisfies text-level legality invariants (unique SSA
   definitions and labels, every basic block terminated, positive metrics) and
   still contains the canonical lowering of each NN operator in the case
   (loop skeleton, ``getelementptr``, MAC ``fmul``/``fadd``, softmax three
   passes, gelu ``tanhf``);
3. ``LLVMCodegen(target_triple=...)`` can stamp a target triple while leaving
   the module body otherwise identical, and — when the LLVM tools are
   installed — ``llvm-as``/``opt`` actually accept the module.

Graceful degradation: when a tool is missing its row is reported as
``skipped`` with a ``skip_reason`` and it never counts as a hard failure and
no assemblability/execution claim is made.  ``lli`` execution is skipped
unless the module exposes a runnable ``i32 @main()`` entry, so this report
makes no execution or performance claim.  Real ONNX workload numbers remain
separate in ``run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_extended import ExtendedDSLParser

SCHEMA_VERSION = "topic16-llvm-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic16_llvm_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/llvm_codegen_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/llvm_codegen_report.md")
#: Triple used only to prove ``LLVMCodegen(target_triple=...)`` overridability;
#: the default driver path intentionally emits a triple-less module.
TARGET_TRIPLE = "riscv32-unknown-elf"
TOOL_NAMES = ("llvm-as", "opt", "lli")
TOOL_TIMEOUT_S = 60
LLI_TIMEOUT_S = 30

_DEFINE_RE = re.compile(r"^define\s+([\w.*]+)\s+@([\w.$]+)\(")
_DEF_RE = re.compile(r"^\s*(%[\w.$]+)\s*=")
_LABEL_RE = re.compile(r"^([A-Za-z0-9_.$]+):")
_TERMINATOR_RE = re.compile(r"^(ret|br|unreachable)\b")
_EXECUTABLE_ENTRY_RE = re.compile(r"^define\s+i32\s+@main\(\)\s*\{", re.M)

HONESTY = (
    "Deterministic structural feature case: the module is produced by the "
    "real CompilerDriver with backend='llvm'. Text-level invariants (unique "
    "SSA definitions, unique labels, terminated basic blocks, positive "
    "metrics) and NN-operator lowering markers are verified without any "
    "external tool. llvm-as/opt/lli results are claimed only when the "
    "corresponding binary is present; a missing tool is reported as "
    "'skipped' with a skip_reason and is not a hard failure. lli execution is "
    "skipped unless the module exposes a runnable i32 @main() entry, so no "
    "execution, timing or speedup claim is made."
)


# ---------------------------------------------------------------------------
# IR text analysis (no external tools)
# ---------------------------------------------------------------------------

def _split_functions(ir: str) -> list[dict[str, Any]]:
    """Split module text into ``define`` bodies."""
    functions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in ir.splitlines():
        match = _DEFINE_RE.match(raw.strip())
        if match:
            current = {
                "name": match.group(2),
                "return_type": match.group(1),
                "lines": [],
            }
            functions.append(current)
            continue
        if current is None:
            continue
        if raw.strip() == "}":
            current = None
            continue
        current["lines"].append(raw)
    return functions


def _split_blocks(lines: list[str]) -> list[dict[str, Any]]:
    """Split a function body into basic blocks (implicit entry included)."""
    blocks: list[dict[str, Any]] = []
    current = {"label": "<entry>", "instructions": []}
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith(";"):
            continue
        match = _LABEL_RE.match(stripped)
        if match:
            if current["instructions"] or blocks:
                blocks.append(current)
            current = {"label": match.group(1), "instructions": []}
            continue
        current["instructions"].append(stripped)
    if current["instructions"] or blocks:
        blocks.append(current)
    if (blocks and blocks[0]["label"] == "<entry>"
            and not blocks[0]["instructions"]):
        blocks.pop(0)
    return blocks


def analyze_module(ir: str) -> dict[str, Any]:
    """Compute module/function/block metrics and legality indicators."""
    duplicate_ssa: list[str] = []
    duplicate_labels: list[str] = []
    unterminated: list[str] = []
    empty_blocks: list[str] = []
    function_stats: list[dict[str, Any]] = []
    totals = {
        "basic_block_count": 0,
        "instruction_count": 0,
        "ssa_definition_count": 0,
        "load_count": 0,
        "store_count": 0,
        "alloca_count": 0,
        "call_count": 0,
        "gep_count": 0,
        "terminator_count": 0,
    }

    for func in _split_functions(ir):
        blocks = _split_blocks(func["lines"])
        defs: list[str] = []
        counts = {key: 0 for key in totals}
        counts["basic_block_count"] = len(blocks)
        for block in blocks:
            instructions = block["instructions"]
            if not instructions and block["label"] != "<entry>":
                empty_blocks.append(f"{func['name']}:{block['label']}")
            if instructions and not _TERMINATOR_RE.match(instructions[-1]):
                unterminated.append(f"{func['name']}:{block['label']}")
            counts["terminator_count"] += sum(
                1 for line in instructions if _TERMINATOR_RE.match(line)
            )
            for line in instructions:
                counts["instruction_count"] += 1
                match = _DEF_RE.match(line)
                if match:
                    defs.append(match.group(1))
                    counts["ssa_definition_count"] += 1
                    if "= load" in line:
                        counts["load_count"] += 1
                    elif "= alloca" in line:
                        counts["alloca_count"] += 1
                    elif "= call" in line:
                        counts["call_count"] += 1
                    elif "= getelementptr" in line:
                        counts["gep_count"] += 1
                elif line.startswith("store "):
                    counts["store_count"] += 1
        seen: set[str] = set()
        for name in defs:
            if name in seen:
                duplicate_ssa.append(f"{func['name']}:{name}")
            seen.add(name)
        labels: set[str] = set()
        for block in blocks:
            if block["label"] in labels:
                duplicate_labels.append(f"{func['name']}:{block['label']}")
            labels.add(block["label"])
        for key in totals:
            totals[key] += counts[key]
        function_stats.append({
            "name": func["name"],
            "return_type": func["return_type"],
            "basic_blocks": len(blocks),
            "instructions": counts["instruction_count"],
            "ssa_definitions": counts["ssa_definition_count"],
            "loads": counts["load_count"],
            "stores": counts["store_count"],
            "allocas": counts["alloca_count"],
            "calls": counts["call_count"],
            "geps": counts["gep_count"],
            "terminators": counts["terminator_count"],
        })

    return {
        "ir_lines": len(ir.splitlines()),
        "function_count": len(function_stats),
        "function_names": [func["name"] for func in function_stats],
        **totals,
        "ssa_definitions_unique": not duplicate_ssa,
        "labels_unique": not duplicate_labels,
        "all_blocks_terminated": not unterminated,
        "duplicate_ssa_definitions": duplicate_ssa,
        "duplicate_labels": duplicate_labels,
        "unterminated_blocks": unterminated,
        "empty_blocks": empty_blocks,
        "functions": function_stats,
    }


# ---------------------------------------------------------------------------
# Lowering markers
# ---------------------------------------------------------------------------

def _count(ir: str, pattern: str) -> int:
    return len(re.findall(pattern, ir))


def check_lowering(ir: str) -> dict[str, Any]:
    """Check the canonical lowering of every operator used by the case."""
    counts = {
        "getelementptr": _count(ir, r"\bgetelementptr\b"),
        "fmul": _count(ir, r"=\s*fmul\b"),
        "fadd": _count(ir, r"=\s*fadd\b"),
        "fcmp_ogt": _count(ir, r"\bfcmp ogt\b"),
        "select_i1": _count(ir, r"\bselect i1\b"),
        "icmp_slt_i32": _count(ir, r"\bicmp slt i32\b"),
        "br_i1": _count(ir, r"\bbr i1\b"),
        "expf_calls": _count(ir, r"=\s*call float @expf\("),
        "tanhf_calls": _count(ir, r"=\s*call float @tanhf\("),
    }
    header_labels = [
        label
        for label in re.findall(r"^([A-Za-z0-9_.$]+):", ir, re.M)
        if "_hdr" in label
    ]

    def has_header(prefix: str) -> bool:
        return any(label.startswith(prefix) for label in header_labels)

    markers = {
        "relu_uses_fcmp_and_select": (
            counts["fcmp_ogt"] > 0 and counts["select_i1"] > 0
        ),
        "dot_uses_loop_gep_and_mac": (
            has_header("dot_i_hdr")
            and counts["getelementptr"] > 0
            and counts["fmul"] > 0
            and counts["fadd"] > 0
        ),
        "matmul_uses_nested_loops_and_gep": (
            has_header("mm_i_hdr")
            and has_header("mm_j_hdr")
            and has_header("mm_k_hdr")
            and counts["getelementptr"] > 0
            and counts["fmul"] > 0
        ),
        "softmax_uses_three_passes_and_expf": (
            has_header("sm_max_i_hdr")
            and has_header("sm_sum_i_hdr")
            and has_header("sm_div_i_hdr")
            and counts["expf_calls"] > 0
        ),
        "gelu_uses_tanh": counts["tanhf_calls"] > 0,
        "loops_have_icmp_and_conditional_branch": (
            counts["icmp_slt_i32"] >= 6 and counts["br_i1"] >= 6
        ),
    }
    return {
        "counts": counts,
        "loop_header_labels": sorted(header_labels),
        "markers": markers,
    }


# ---------------------------------------------------------------------------
# Real CompilerDriver compilation
# ---------------------------------------------------------------------------

def compile_case(case_path: Path) -> dict[str, Any]:
    """Compile the DSL case through ``CompilerDriver(backend="llvm")``."""
    source = case_path.read_text(encoding="utf-8")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "topic16_case.ll"
        result = CompilerDriver(CompilerConfig(
            backend="llvm",
            optimize_level="all",
            dump_ir=True,
        )).compile("", str(output), dsl_source=source)
        elapsed_ms = (time.perf_counter() - started) * 1000
        ir_text = result.output_text
        if output.is_file():
            ir_text = output.read_text(encoding="utf-8")
    return {
        "success": result.success,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "elapsed_ms": round(elapsed_ms, 3),
        "opt_message": result.stats.get("opt_message", ""),
        "ir_text": ir_text,
    }


def check_target_triple(source: str) -> dict[str, Any]:
    """Prove ``target_triple`` is optional and overridable."""
    try:
        default_ir = LLVMCodegen(ExtendedDSLParser().parse(source)).emit()
        explicit_ir = LLVMCodegen(
            ExtendedDSLParser().parse(source), TARGET_TRIPLE
        ).emit()
    except Exception as exc:  # pragma: no cover - exercised via bad cases
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "default_has_triple": None,
            "explicit_triple": TARGET_TRIPLE,
            "explicit_has_triple": None,
            "body_identical_without_triple": False,
        }

    def without_triple(text: str) -> list[str]:
        return [
            line for line in text.splitlines() if "target triple" not in line
        ]

    return {
        "status": "ok",
        "error": None,
        "default_has_triple": "target triple" in default_ir,
        "explicit_triple": TARGET_TRIPLE,
        "explicit_has_triple": (
            f'target triple = "{TARGET_TRIPLE}"' in explicit_ir
        ),
        "body_identical_without_triple": (
            without_triple(default_ir) == without_triple(explicit_ir)
        ),
    }


# ---------------------------------------------------------------------------
# External LLVM toolchain (gracefully optional)
# ---------------------------------------------------------------------------

def _tool_row(name: str, path: str | None) -> dict[str, Any]:
    return {
        "tool": name,
        "available": path is not None,
        "path": path,
        "status": "skipped",
        "skip_reason": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "detail": "",
    }


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
    )


def run_toolchain(ir: str) -> dict[str, dict[str, Any]]:
    """Assemble/optimise the module with whatever LLVM tools exist locally."""
    paths = {name: shutil.which(name) for name in TOOL_NAMES}
    tools = {name: _tool_row(name, paths[name]) for name in TOOL_NAMES}
    for name, row in tools.items():
        if not row["available"]:
            row["skip_reason"] = f"{name} not installed"
    if not ir.strip():
        for row in tools.values():
            row["skip_reason"] = "no IR module was produced"
        return tools

    with tempfile.TemporaryDirectory() as tmp:
        ll_path = Path(tmp) / "module.ll"
        ll_path.write_text(ir, encoding="utf-8")
        bc_path = Path(tmp) / "module.bc"
        opt_path = Path(tmp) / "module.opt.ll"

        asm = tools["llvm-as"]
        if asm["available"]:
            try:
                proc = _run(
                    [paths["llvm-as"], str(ll_path), "-o", str(bc_path)],
                    TOOL_TIMEOUT_S,
                )
                asm["returncode"] = proc.returncode
                asm["stderr"] = proc.stderr.strip()
                if proc.returncode == 0:
                    asm["status"] = "assembled"
                    asm["detail"] = f"bitcode {bc_path.stat().st_size} bytes"
                    asm["skip_reason"] = None
                else:
                    asm["status"] = "failed"
            except subprocess.TimeoutExpired:
                asm["status"] = "timeout"
                asm["skip_reason"] = "llvm-as exceeded the time limit"
            except OSError as exc:
                asm["status"] = "error"
                asm["skip_reason"] = str(exc)

        opt = tools["opt"]
        if opt["available"]:
            try:
                proc = _run(
                    [
                        paths["opt"], "-O3", "-S",
                        str(ll_path), "-o", str(opt_path),
                    ],
                    TOOL_TIMEOUT_S,
                )
                opt["returncode"] = proc.returncode
                opt["stderr"] = proc.stderr.strip()
                if proc.returncode == 0:
                    opt["status"] = "optimized"
                    opt_lines = len(
                        opt_path.read_text(encoding="utf-8").splitlines()
                    )
                    opt["detail"] = f"{opt_lines} lines after -O3"
                    if asm["available"]:
                        reparsed = _run(
                            [paths["llvm-as"], str(opt_path),
                             "-o", str(Path(tmp) / "module.opt.bc")],
                            TOOL_TIMEOUT_S,
                        )
                        opt["detail"] += (
                            f"; reassembly rc={reparsed.returncode}"
                        )
                else:
                    opt["status"] = "failed"
            except subprocess.TimeoutExpired:
                opt["status"] = "timeout"
                opt["skip_reason"] = "opt exceeded the time limit"
            except OSError as exc:
                opt["status"] = "error"
                opt["skip_reason"] = str(exc)

        lli = tools["lli"]
        if lli["available"]:
            if not _EXECUTABLE_ENTRY_RE.search(ir):
                lli["skip_reason"] = (
                    "module exposes no runnable i32 @main() entry "
                    "(kernel-only module); IR validity is covered by "
                    "llvm-as/opt"
                )
            elif bc_path.is_file():
                try:
                    proc = _run([paths["lli"], str(bc_path)], LLI_TIMEOUT_S)
                    lli["returncode"] = proc.returncode
                    lli["stdout"] = proc.stdout.strip()
                    lli["stderr"] = proc.stderr.strip()
                    lli["status"] = (
                        "executed" if proc.returncode == 0 else "failed"
                    )
                except subprocess.TimeoutExpired:
                    lli["status"] = "timeout"
                    lli["skip_reason"] = "lli exceeded the time limit"
                except OSError as exc:
                    lli["status"] = "error"
                    lli["skip_reason"] = str(exc)
            else:
                lli["skip_reason"] = "no bitcode available (llvm-as missing)"
        return tools


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def evaluate(case_path: Path) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    source = case_path.read_text(encoding="utf-8")
    compile_info = compile_case(case_path)
    ir_text = compile_info.pop("ir_text")
    module = analyze_module(ir_text)
    lowering = check_lowering(ir_text)
    triple = check_target_triple(source)
    toolchain = run_toolchain(ir_text)

    ir_lines = len(ir_text.splitlines())
    compile_info["ir_lines"] = ir_lines
    compile_info["ir_head"] = ir_text.splitlines()[:40]

    asm, opt, lli = toolchain["llvm-as"], toolchain["opt"], toolchain["lli"]
    hard_checks = {
        "case_compiles_via_llvm_driver": bool(compile_info["success"]),
        "ir_module_is_nonempty": ir_lines > 0,
        "ssa_definitions_unique_within_functions": (
            module["ssa_definitions_unique"]),
        "block_labels_unique_within_functions": module["labels_unique"],
        "every_basic_block_terminated": module["all_blocks_terminated"],
        "module_metrics_are_positive": (
            module["function_count"] >= 1
            and module["instruction_count"] > 0
            and module["ssa_definition_count"] > 0
            and module["load_count"] > 0
            and module["store_count"] > 0
            and module["alloca_count"] > 0
            and module["gep_count"] > 0
        ),
        "nn_operator_lowering_markers_present": all(
            lowering["markers"].values()),
        "target_triple_is_configurable": (
            triple["default_has_triple"] is False
            and triple["explicit_has_triple"] is True
            and triple["body_identical_without_triple"]
        ),
        "llvm_toolchain_accepts_module_when_available": (
            bool(compile_info["success"])
            and (not asm["available"] or asm["returncode"] == 0)
            and (not opt["available"] or opt["returncode"] == 0)
            and (not lli["available"] or lli["status"] != "failed")
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic16-llvm-codegen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "config": {
            "backend": "llvm",
            "optimize_level": "all",
            "dump_ir": True,
        },
        "compile": compile_info,
        "module": module,
        "lowering": lowering,
        "target_triple": triple,
        "toolchain": toolchain,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": HONESTY,
    }


def render_markdown(report: dict[str, Any]) -> str:
    compile_info = report["compile"]
    module = report["module"]
    lowering = report["lowering"]
    triple = report["target_triple"]
    passed = not report["hard_failures"]
    lines = [
        "# Topic 16 LLVM Codegen Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}`",
        f"- Backend: `{report['config']['backend']}` "
        f"(optimize_level=`{report['config']['optimize_level']}`)",
        f"- Generated: {report['generated_at']}",
        f"- Hard checks: {'PASS' if passed else 'FAIL'} "
        f"({len(report['hard_checks']) - len(report['hard_failures'])}"
        f"/{len(report['hard_checks'])})",
        "",
        "## Compile metrics",
        "",
        "| Metric | Value |",
        "|--------|---:|",
        f"| Compilation success | {compile_info['success']} |",
        f"| IR lines | {module['ir_lines']} |",
        f"| Functions | {module['function_count']} |",
        f"| Basic blocks | {module['basic_block_count']} |",
        f"| Instructions | {module['instruction_count']} |",
        f"| SSA definitions | {module['ssa_definition_count']} |",
        f"| load / store / alloca | {module['load_count']} / "
        f"{module['store_count']} / {module['alloca_count']} |",
        f"| calls / getelementptr | {module['call_count']} / "
        f"{module['gep_count']} |",
        f"| Terminators (br/ret/unreachable) | {module['terminator_count']} |",
        f"| Compile time (ms) | {compile_info['elapsed_ms']:.3f} |",
        "",
        "## Module structure",
        "",
        "| Function | Return | Blocks | Instructions | SSA defs | "
        "Terminated |",
        "|----------|--------|-------:|-------------:|---------:|"
        "-----------:|",
    ]
    for func in module["functions"]:
        terminated = all(
            not block.startswith(f"{func['name']}:")
            for block in module["unterminated_blocks"]
        )
        lines.append(
            f"| `{func['name']}` | `{func['return_type']}` | "
            f"{func['basic_blocks']} | {func['instructions']} | "
            f"{func['ssa_definitions']} | {'yes' if terminated else 'no'} |"
        )
    duplicate_ssa = module["duplicate_ssa_definitions"] or "none"
    duplicate_labels = module["duplicate_labels"] or "none"
    unterminated = module["unterminated_blocks"] or "none"

    def mark(name: str) -> str:
        return "yes" if lowering["markers"][name] else "no"

    lines += [
        "",
        f"- Duplicate SSA definitions: {duplicate_ssa}",
        f"- Duplicate block labels: {duplicate_labels}",
        f"- Unterminated blocks: {unterminated}",
        f"- Optimizer pipeline: `{compile_info['opt_message'] or 'n/a'}`",
        f"- Compilation errors: {compile_info['errors'] or 'none'}",
        "",
        "## NN operator lowering markers",
        "",
        "| Marker | Present | Evidence count |",
        "|--------|---------|----------------|",
        f"| relu: fcmp+select | {mark('relu_uses_fcmp_and_select')} "
        f"| fcmp ogt={lowering['counts']['fcmp_ogt']}, "
        f"select i1={lowering['counts']['select_i1']} |",
        f"| dot: loop+GEP+MAC | {mark('dot_uses_loop_gep_and_mac')} "
        f"| gep={lowering['counts']['getelementptr']}, "
        f"fmul={lowering['counts']['fmul']}, "
        f"fadd={lowering['counts']['fadd']} |",
        f"| matmul: nested loops+GEP | "
        f"{mark('matmul_uses_nested_loops_and_gep')} "
        f"| icmp slt i32={lowering['counts']['icmp_slt_i32']}, "
        f"br i1={lowering['counts']['br_i1']} |",
        f"| softmax: 3 passes+expf | "
        f"{mark('softmax_uses_three_passes_and_expf')} "
        f"| expf calls={lowering['counts']['expf_calls']} |",
        f"| gelu: tanhf | {mark('gelu_uses_tanh')} "
        f"| tanhf calls={lowering['counts']['tanhf_calls']} |",
        "",
        f"- Loop header labels: "
        f"{', '.join(lowering['loop_header_labels']) or 'none'}",
        "",
        "## Target triple",
        "",
        f"- Default module carries a triple: `{triple['default_has_triple']}`",
        f"- Explicit `LLVMCodegen(target_triple="
        f"\"{triple['explicit_triple']}\")` emits it: "
        f"`{triple['explicit_has_triple']}`",
        f"- Module body identical apart from the triple line: "
        f"`{triple['body_identical_without_triple']}`",
        "",
        "## Toolchain matrix",
        "",
        "| Tool | Available | Status | Return code | Detail |",
        "|------|-----------|--------|------------:|--------|",
    ]
    for name in TOOL_NAMES:
        row = report["toolchain"][name]
        detail = row["detail"] or row["skip_reason"] or "-"
        rc = row["returncode"] if row["returncode"] is not None else "-"
        lines.append(
            f"| {name} | {'yes' if row['available'] else 'no'} | "
            f"{row['status']} | {rc} | {detail} |"
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
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(render_markdown(report))
    if report["hard_failures"]:
        print("HARD FAILURES: " + ", ".join(report["hard_failures"]))
        return 1
    print(f"reports written: {args.json}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
