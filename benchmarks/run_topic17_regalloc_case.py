#!/usr/bin/env python3
"""Run one Topic 17 register-allocation feature case and emit auditable CI reports.

The report proves three separate facts:

1. the configured compiler pipeline accepts both ``reg_alloc="greedy"`` and
   ``reg_alloc="linear"`` for the same deterministic low-pressure DSL case and
   both products are assemblable RISC-V;
2. the linear path emits a real frame: spill/reload code for the loop-carried
   values, a balanced prologue/epilogue adjustment, and no frame-relative
   spill access outside the allocated frame;
3. the RV32 emulator executes both products to identical architectural state
   on the observed registers.

The case is deliberately low-pressure: it validates the frame/ABI plumbing
without exercising register-pool exhaustion, reload-time eviction or
high-pressure spilling.  This is a deterministic feature/integration case, not
a real-workload speedup claim; ``linear`` remains Stage 1 opt-in and its
block-local force-spill strategy is expected to trade dynamic instructions for
frame traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator

SCHEMA_VERSION = "topic17-regalloc-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic17_regalloc_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/regalloc_case_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/regalloc_case_report.md")
#: Expected ``a0`` of the deterministic case: i = 0..5, k = i * i + i -> 30.
EXPECTED_RESULT = 30
#: Registers that must be identical in both products: the return value and the
#: stack pointer (the linear path's frame must be balanced at retirement).
OBSERVED_REGISTERS = ("x2", "x10")

_BARE_VREG = re.compile(r"(?<![A-Za-z0-9_.])v[0-9]+(?![A-Za-z0-9_])")
_NEG_OFFSET = re.compile(r"-\d+\(sp\)")
_SP_OPERAND = re.compile(r"^(-?\d+)\(sp\)$")
_REG_NAME = re.compile(r"^(?:x\d+|[a-z][a-z0-9]*)$")
_REG_ONLY_OPS = frozenset({
    "add", "sub", "mul", "and", "or", "xor", "sll", "srl", "sra", "slt",
})
_INSERTED_MARKERS = {
    "reload": "reload_count",
    "store redefined": "writeback_count",
    "evict": "eviction_count",
}


def count_asm(asm: str) -> int:
    """Number of real instructions (labels and directives excluded)."""
    return sum(
        1
        for line in parse_asm(asm)
        if line.opcode is not None and not line.is_directive
    )


def _sp_adjustments(parsed) -> tuple[list[int], list[int]]:
    """``(prologue, epilogue)`` ``addi sp, sp, ±N`` immediate lists."""
    prologue: list[int] = []
    epilogue: list[int] = []
    for line in parsed:
        if line.opcode != "addi" or len(line.operands) != 3:
            continue
        dst, src, imm = (operand.strip() for operand in line.operands)
        if dst != "sp" or src != "sp":
            continue
        try:
            value = int(imm)
        except ValueError:
            continue
        (prologue if value < 0 else epilogue).append(value)
    return prologue, epilogue


def _spill_offsets(parsed) -> list[int]:
    """Offsets of every ``lw``/``sw`` access through ``sp``."""
    offsets: list[int] = []
    for line in parsed:
        if line.opcode not in ("lw", "sw"):
            continue
        for operand in line.operands:
            match = _SP_OPERAND.match(operand.strip())
            if match is not None:
                offsets.append(int(match.group(1)))
    return offsets


def _comment_counts(parsed) -> dict[str, int]:
    counts = {name: 0 for name in _INSERTED_MARKERS.values()}
    for line in parsed:
        comment = line.comment or ""
        for prefix, name in _INSERTED_MARKERS.items():
            if comment.startswith(prefix):
                counts[name] += 1
    return counts


def check_hygiene(asm: str) -> dict[str, Any]:
    """Lightweight assembly hygiene: no vreg/SPILL_ leftovers, valid operands,
    and the product must be accepted by the repository encoder."""
    issues: list[str] = []
    body = "\n".join(line.split("#", 1)[0] for line in asm.splitlines())
    if "SPILL_" in body:
        issues.append("SPILL_ marker in assembly")
    if _BARE_VREG.search(body):
        issues.append("bare virtual register in assembly")
    if _NEG_OFFSET.search(body):
        issues.append("negative sp offset in assembly")

    for line in parse_asm(asm):
        if line.opcode not in _REG_ONLY_OPS:
            continue
        for operand in (operand.strip() for operand in line.operands):
            if not _REG_NAME.match(operand):
                issues.append(
                    f"{line.opcode}: non-register operand {operand!r}")
                break

    assembles = False
    assemble_error: Optional[str] = None
    try:
        assemble_to_binary(asm)
        assembles = True
    except Exception as exc:  # fail-loudly contract: record, do not crash
        assemble_error = f"{type(exc).__name__}: {exc}"
        issues.append(f"assemble_to_binary rejected the product: {exc}")
    return {
        "clean": not issues,
        "issues": issues,
        "assembles": assembles,
        "assemble_error": assemble_error,
    }


def run_asm(asm: str) -> dict[str, Any]:
    """Assemble and execute *asm*; return register state and counters."""
    try:
        binary = assemble_to_binary(asm)
        emulator = RV32Emulator()
        emulator.load_code(bytes(binary))
        dynamic = emulator.run(max_instr=10000)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "backend": "rv32-emulator",
        "registers": {f"x{i}": emulator.regs[i] for i in range(32)},
        "a0": emulator.regs[REG_ID["a0"]],
        "sp": emulator.regs[REG_ID["sp"]],
        "dynamic_instructions": dynamic,
    }


def _failed_measurement(mode: str, errors: list[str]) -> dict[str, Any]:
    """Uniform measurement shape for a failed compilation."""
    return {
        "mode": mode,
        "compile_success": False,
        "errors": list(errors),
        "compile_time_ms": 0.0,
        "runs": 0,
        "distinct_asm": 0,
        "deterministic": False,
        "asm_sha256": None,
        "asm_instructions": 0,
        "asm_lines": 0,
        "spill_accesses": 0,
        "spill_offsets": [],
        "reload_count": 0,
        "writeback_count": 0,
        "eviction_count": 0,
        "frame": {"prologue_offsets": [], "epilogue_offsets": []},
        "frame_size": 0,
        "hygiene": {
            "clean": False,
            "issues": list(errors),
            "assembles": False,
            "assemble_error": None,
        },
        "execution": None,
        "asm_head": [],
    }


def measure_allocator(
        case_path: Path, mode: str, repeats: int) -> dict[str, Any]:
    """Compile the case *repeats* times with *mode* and collect A/B metrics."""
    source = case_path.read_text()
    texts: list[str] = []
    times: list[float] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(repeats):
            driver = CompilerDriver(CompilerConfig(reg_alloc=mode))
            started = time.perf_counter()
            result = driver.compile(
                f"topic17-regalloc-{mode}.dsl",
                str(Path(tmp) / f"{mode}_{i}.s"),
                dsl_source=source,
            )
            times.append((time.perf_counter() - started) * 1000)
            if not result.success:
                errors.extend(result.errors or ["compilation failed"])
                break
            texts.append(result.output_text)

    if not texts:
        return _failed_measurement(
            mode, errors or ["compilation produced no output"])

    asm = texts[0]
    parsed = parse_asm(asm)
    prologue, epilogue = _sp_adjustments(parsed)
    spill_offsets = _spill_offsets(parsed)
    comment_counts = _comment_counts(parsed)
    hygiene = check_hygiene(asm)
    execution = run_asm(asm) if hygiene["assembles"] else {
        "error": "assembly rejected before execution",
    }
    return {
        "mode": mode,
        "compile_success": True,
        "errors": errors,
        "compile_time_ms": round(statistics.median(times), 4),
        "runs": len(texts),
        "distinct_asm": len(set(texts)),
        "deterministic": len(set(texts)) == 1,
        "asm_sha256": hashlib.sha256(asm.encode("utf-8")).hexdigest(),
        "asm_instructions": count_asm(asm),
        "asm_lines": len(asm.splitlines()),
        "spill_accesses": len(spill_offsets),
        "spill_offsets": spill_offsets,
        "reload_count": comment_counts["reload_count"],
        "writeback_count": comment_counts["writeback_count"],
        "eviction_count": comment_counts["eviction_count"],
        "frame": {
            "prologue_offsets": prologue,
            "epilogue_offsets": epilogue,
        },
        "frame_size": -sum(prologue) if prologue else 0,
        "hygiene": hygiene,
        "execution": execution,
        "asm_head": asm.splitlines()[:12],
    }


def evaluate(case_path: Path, repeats: int) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    runs = max(repeats, 2)
    greedy = measure_allocator(case_path, "greedy", runs)
    linear = measure_allocator(case_path, "linear", runs)

    def a0(measured: dict[str, Any]) -> Optional[int]:
        execution = measured.get("execution")
        return execution.get("a0") if isinstance(execution, dict) else None

    def sp(measured: dict[str, Any]) -> Optional[int]:
        execution = measured.get("execution")
        return execution.get("sp") if isinstance(execution, dict) else None

    registers_greedy = (
        (greedy.get("execution") or {}).get("registers") or {})
    registers_linear = (
        (linear.get("execution") or {}).get("registers") or {})
    prologue = linear["frame"]["prologue_offsets"]
    epilogue = linear["frame"]["epilogue_offsets"]
    frame_size = linear["frame_size"]

    hard_checks = {
        "greedy_compile_succeeds": bool(greedy["compile_success"]),
        "linear_compile_succeeds": bool(linear["compile_success"]),
        "allocators_produce_different_code": (
            bool(greedy["compile_success"])
            and bool(linear["compile_success"])
            and greedy["asm_sha256"] != linear["asm_sha256"]
        ),
        "linear_emits_spill_code": linear["spill_accesses"] > 0,
        "linear_frame_balanced": (
            bool(prologue) and sum(prologue) + sum(epilogue) == 0
        ),
        "linear_spill_offsets_inside_frame": (
            frame_size > 0
            and all(
                0 <= offset and offset + 4 <= frame_size
                for offset in linear["spill_offsets"]
            )
        ),
        "linear_hygiene_clean": bool(linear["hygiene"]["clean"]),
        "linear_assembles_to_binary": bool(linear["hygiene"]["assembles"]),
        "linear_allocation_deterministic": bool(linear["deterministic"]),
        "execution_matches_expected": (
            a0(greedy) == EXPECTED_RESULT and a0(linear) == EXPECTED_RESULT
        ),
        "observed_registers_identical": (
            bool(registers_greedy)
            and bool(registers_linear)
            and all(
                registers_greedy.get(reg) == registers_linear.get(reg)
                for reg in OBSERVED_REGISTERS
            )
        ),
        "stack_pointer_balanced": (
            sp(greedy) == RV32Emulator.STACK_TOP
            and sp(linear) == RV32Emulator.STACK_TOP
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic17-regalloc",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "expected_result": EXPECTED_RESULT,
        "observed_registers": list(OBSERVED_REGISTERS),
        "config": {"optimize_level": "none", "modes": ["greedy", "linear"]},
        "runs": runs,
        "greedy": greedy,
        "linear": linear,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": (
            "Deterministic feature case executed by the repository RV32 "
            "emulator.  The linear-scan allocator is still Stage 1 opt-in "
            "(`--reg-alloc linear`; default `greedy`) and force-spills "
            "cross-block values, so its product is expected to contain more "
            "dynamic instructions and frame traffic; no performance claim is "
            "made.  Dynamic-instruction numbers are emulator counts, not "
            "hardware cycles.  The case is deliberately low-pressure (no "
            "register-pool exhaustion, no reload-time eviction) so it "
            "validates frame/ABI plumbing, not the high-pressure spill paths."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    greedy, linear = report["greedy"], report["linear"]
    dyn_greedy = (greedy.get("execution") or {}).get(
        "dynamic_instructions", "n/a")
    dyn_linear = (linear.get("execution") or {}).get(
        "dynamic_instructions", "n/a")
    a0_greedy = (greedy.get("execution") or {}).get("a0", "n/a")
    a0_linear = (linear.get("execution") or {}).get("a0", "n/a")
    prologue = linear["frame"]["prologue_offsets"]
    epilogue = linear["frame"]["epilogue_offsets"]
    frame_note = (
        f"prologue addi sp, sp, {min(prologue)}, epilogue addi sp, sp, "
        f"{max(epilogue)}" if prologue else "no frame adjustment"
    )
    lines = [
        "# Topic 17 Register-Allocation Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}` (expected `a0 == "
        f"{report['expected_result']}`)",
        f"- Generated: {report['generated_at']}",
        f"- Hard checks: "
        f"{'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({len(report['hard_checks']) - len(report['hard_failures'])}"
        f"/{len(report['hard_checks'])})",
        "",
        "## A/B summary (greedy vs linear)",
        "",
        "| Metric | greedy | linear | note |",
        "|--------|-------:|-------:|------|",
        f"| ASM instructions | {greedy['asm_instructions']} | "
        f"{linear['asm_instructions']} | linear force-spills cross-block "
        f"values |",
        f"| `sp(...)` load/store accesses | {greedy['spill_accesses']} | "
        f"{linear['spill_accesses']} | frame traffic |",
        f"| reload / writeback / evict | {greedy['reload_count']} / "
        f"{greedy['writeback_count']} / {greedy['eviction_count']} | "
        f"{linear['reload_count']} / {linear['writeback_count']} / "
        f"{linear['eviction_count']} | allocator-inserted code |",
        f"| Frame size (bytes) | 0 | {linear['frame_size']} | "
        f"{frame_note} |",
        f"| Compile time (ms, median) | {greedy['compile_time_ms']:.4f} | "
        f"{linear['compile_time_ms']:.4f} | wall clock, not a claim |",
        f"| Dynamic instructions (emulator) | {dyn_greedy} | {dyn_linear} | "
        f"emulator counts, not cycles |",
        f"| `a0` result | {a0_greedy} | {a0_linear} | equal |",
        "",
        "## Frame evidence (linear)",
        "",
        f"- Prologue/epilogue: {prologue} / {epilogue} -> "
        f"frame_size={linear['frame_size']}",
        f"- Spill offsets: {linear['spill_offsets']} inside "
        f"[0, {linear['frame_size']})",
        f"- Hygiene: {'clean' if linear['hygiene']['clean'] else 'issues'} "
        f"(assembles_to_binary={linear['hygiene']['assembles']})",
        "- First lines:",
        "",
        "```asm",
        *linear["asm_head"],
        "```",
        "",
        "## Execution (RV32 emulator)",
        "",
        f"- Observed registers compared: "
        f"{', '.join(report['observed_registers'])}",
        f"- greedy: a0={a0_greedy}, sp="
        f"{(greedy.get('execution') or {}).get('sp', 'n/a')}",
        f"- linear: a0={a0_linear}, sp="
        f"{(linear.get('execution') or {}).get('sp', 'n/a')}",
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


def main(argv: Optional[list[str]] = None) -> int:
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
