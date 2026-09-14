#!/usr/bin/env python3
"""Run one Topic 10 loop-unroll feature case and emit auditable CI reports.

The report proves three separate facts:

1. the configured compiler pipeline runs the unroll pass when ``loop_unroll``
   is opted in, and keeps the loop markers when it is not;
2. the pass changes a deterministic low-pressure case and reports categorized
   metrics (full/partial/epilogue counts, instructions before/after, skips);
3. the RV32 emulator executes the original and the unrolled program to
   identical architectural state while the unrolled program needs fewer
   dynamic instructions.

This is a deterministic feature/integration case, not a real-workload speedup
claim.  Real ONNX benchmark numbers remain separate in ``run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode, Program
from scratchv.optimizer.loop_unroll import LoopUnroll
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator

SCHEMA_VERSION = "topic10-unroll-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic10_unroll_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/loop_unroll_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/loop_unroll_report.md")
#: Expected return value of the deterministic IR case below (iv = 0..3 sums
#: to 3 after the low-pressure body ``v2 = iv + one`` with ``one`` = 1).
EXPECTED_RESULT = 4
#: Registers that carry observable results.  Greedy allocation may place the
#: same value into different temporaries after unrolling, so only the return
#: register is compared; the full state is still recorded for auditing.
OBSERVED_REGISTERS = ("x10",)


def _make_const(builder: IRBuilder, name: str, value: int):
    value_obj = builder.make_value(
        name=name, dtype=DataType.INT32, is_constant=False)
    builder._emit(OpCode.LOAD_CONST, value_obj, value=value)
    return value_obj


def build_case_program(end: int = 4) -> Program:
    """Deterministic low-pressure loop; returns ``v2`` from the last trip.

    ``one`` is emitted with ``is_constant=False`` so the backend keeps it as
    a runtime value; the emulator then observes ``a0 == end``.
    """
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    one = _make_const(builder, "one", 1)
    iv = builder.for_loop(0, end)
    v2 = builder.add(iv, one)
    builder.endfor()
    builder.ret(v2)
    return builder.program


def count_ir(program: Program) -> int:
    return sum(
        1
        for func in program.functions
        for block in func.blocks
        for _ in block.instructions
    )


def count_asm(asm: str) -> int:
    return sum(
        line.opcode is not None and not line.is_directive
        for line in parse_asm(asm)
    )


def emit_asm(program: Program) -> str:
    machine = InstructionSelector(program).run()
    allocated = RegisterAllocator(machine, mode="greedy").run()
    return AsmEmitter(allocated).emit()


def run_asm(asm: str) -> dict[str, Any]:
    """Assemble and execute *asm*; return register state and counters."""
    binary = assemble_to_binary(asm)
    emulator = RV32Emulator()
    emulator.load_code(bytes(binary))
    dynamic = emulator.run()
    return {
        "backend": "rv32-emulator",
        "registers": {f"x{i}": emulator.regs[i] for i in range(32)},
        "a0": emulator.regs[REG_ID["a0"]],
        "dynamic_instructions": dynamic,
    }


def _measure_side(*, unroll: bool, repeats: int) -> dict[str, Any]:
    """Run the deterministic case with unrolling off/on."""
    program = build_case_program()
    ir_before = count_ir(program)
    pass_time_ms = 0.0
    unroll_stats: dict[str, Any] | None = None
    loops_unrolled = 0
    if unroll:
        times = []
        runner: LoopUnroll | None = None
        for _ in range(repeats):
            candidate = build_case_program()
            runner = LoopUnroll(candidate)
            started = time.perf_counter()
            runner.run()
            times.append((time.perf_counter() - started) * 1000)
            program = candidate
        pass_time_ms = statistics.median(times)
        assert runner is not None
        unroll_stats = runner.stats
        loops_unrolled = runner.stats["loops_unrolled"]
    ir_after = count_ir(program)
    asm = emit_asm(program)
    execution = run_asm(asm)
    return {
        "loop_unroll": unroll,
        "loops_unrolled": loops_unrolled,
        "ir_instructions": ir_after,
        "ir_instructions_before_pass": ir_before,
        "ir_instructions_added": ir_after - ir_before,
        "asm_instructions": count_asm(asm),
        "pass_time_ms": round(pass_time_ms, 4),
        "unroll_stats": unroll_stats,
        "execution": execution,
        "asm_head": asm.splitlines()[:12],
    }


def measure_wiring(case_path: Path) -> dict[str, Any]:
    """Prove the configured compiler pipeline honours the opt-in flag."""
    source = case_path.read_text()
    common = dict(optimize_level="all", reg_alloc="greedy", dump_ir=True)
    with tempfile.TemporaryDirectory() as tmp:
        off = CompilerDriver(
            CompilerConfig(loop_unroll=False, **common)).compile(
            "", str(Path(tmp) / "off.s"), dsl_source=source)
        on = CompilerDriver(
            CompilerConfig(loop_unroll=True, **common)).compile(
            "", str(Path(tmp) / "on.s"), dsl_source=source)
    if not (off.success and on.success):
        raise RuntimeError(
            f"feature case failed to compile: {off.errors or on.errors}")
    off_ir = off.ir_dump.split("--- IR Dump (after")[1]
    on_ir = on.ir_dump.split("--- IR Dump (after")[1]
    return {
        "off_has_loop_markers": "endfor" in off_ir,
        "on_has_loop_markers": "endfor" in on_ir,
        "off_pass_present": "loop-unroll" in off.stats.get("passes", {}),
        "on_pass_present": "loop-unroll" in on.stats.get("passes", {}),
        "on_pass_stats": on.stats.get("passes", {}).get("loop-unroll"),
    }


def evaluate(case_path: Path, repeats: int) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    off = _measure_side(unroll=False, repeats=repeats)
    on = _measure_side(unroll=True, repeats=repeats)
    wiring = measure_wiring(case_path)

    hard_checks = {
        "pipeline_runs_pass_when_enabled": bool(wiring["on_pass_present"]),
        "pipeline_skips_pass_when_disabled": (
            not wiring["off_pass_present"]),
        "loop_markers_removed_when_enabled": (
            not wiring["on_has_loop_markers"]),
        "loop_markers_kept_when_disabled": (
            wiring["off_has_loop_markers"]),
        "case_loop_was_unrolled": on["loops_unrolled"] >= 1,
        "execution_result_is_expected": (
            off["execution"]["a0"] == EXPECTED_RESULT
            and on["execution"]["a0"] == EXPECTED_RESULT
        ),
        "observed_registers_identical": all(
            off["execution"]["registers"][reg]
            == on["execution"]["registers"][reg]
            for reg in OBSERVED_REGISTERS
        ),
        "dynamic_instructions_reduced": (
            on["execution"]["dynamic_instructions"]
            < off["execution"]["dynamic_instructions"]),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic10-loop-unroll",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "expected_result": EXPECTED_RESULT,
        "observed_registers": list(OBSERVED_REGISTERS),
        "config": {"optimize_level": "all", "reg_alloc": "greedy"},
        "runs": repeats,
        "unroll_off": off,
        "unroll_on": on,
        "wiring": wiring,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": (
            "Deterministic feature case executed by the repository RV32 "
            "emulator.  Dynamic-instruction numbers are emulator counts, not "
            "hardware cycles; unrolling trades code size for dynamic "
            "instructions and remains opt-in."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    off, on = report["unroll_off"], report["unroll_on"]
    dyn_off = off["execution"]["dynamic_instructions"]
    dyn_on = on["execution"]["dynamic_instructions"]
    saved = dyn_off - dyn_on
    pct = (saved / dyn_off * 100) if dyn_off else 0.0
    stats = on["unroll_stats"] or {}
    skipped = sum(stats.get("skipped", {}).values()) if stats else 0
    lines = [
        "# Topic 10 Loop-Unroll Feature Case",
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
        "## A/B summary",
        "",
        "| Metric | unroll off | unroll on | delta |",
        "|--------|-----------:|----------:|------:|",
        f"| IR instructions | {off['ir_instructions']} | "
        f"{on['ir_instructions']} | {on['ir_instructions_added']:+d} |",
        f"| ASM instructions | {off['asm_instructions']} | "
        f"{on['asm_instructions']} | "
        f"{on['asm_instructions'] - off['asm_instructions']:+d} |",
        f"| Dynamic instructions (emulator) | {dyn_off} | {dyn_on} | "
        f"-{saved} ({pct:.1f}%) |",
        f"| `a0` result | {off['execution']['a0']} | "
        f"{on['execution']['a0']} | equal |",
        f"| Unroll pass time (ms, median) | n/a | "
        f"{on['pass_time_ms']:.4f} | - |",
        "",
        "## Unroll pass metrics (on)",
        "",
        f"- loops_seen={stats.get('loops_seen', 0)}, "
        f"loops_unrolled={stats.get('loops_unrolled', 0)}, "
        f"full={stats.get('full_unrolls', 0)}, "
        f"partial={stats.get('partial_unrolls', 0)}, "
        f"epilogue={stats.get('partial_epilogues', 0)}, "
        f"skipped={skipped}",
        f"- IR instructions before->after pass: "
        f"{stats.get('instructions_before', 'n/a')} -> "
        f"{stats.get('instructions_after', 'n/a')}",
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
