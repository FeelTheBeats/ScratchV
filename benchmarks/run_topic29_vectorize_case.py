#!/usr/bin/env python3
"""Run one Topic 29 SIMD-vectorize feature case and emit auditable CI reports.

The report proves four separate facts:

1. the configured compiler pipeline honours the ``vectorize`` opt-in on a
   real DSL input (both configurations compile; the pass only runs when
   enabled);
2. the canonical element-addressing loop (design doc appendix 5.1,
   ``out[i] = relu(a[i] + a[i])``, ``n=16``, ``W=4``, ``rem=0``) is
   rewritten to vector IR ops when enabled and left scalar when disabled;
3. the vectorized artifact assembles to RV32IM and the RV32 emulator
   produces the same architectural state (``a0``, observed register,
   output memory) for the scalar and vectorized programs;
4. unsupported driver configurations fail loudly: ``vectorize`` plus the
   LLVM backend, illegal ``vector_width`` values and a non-scalar
   ``vector_isa``.

This is a deterministic feature/integration case, not a real-workload
speedup claim.  Real ONNX benchmark numbers remain separate in
``run_benchmark.py``.
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
from typing import Any

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Program
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator

SCHEMA_VERSION = "topic29-vectorize-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic29_vectorize_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/vectorize_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/vectorize_report.md")

#: Fixed phase-1 strip width for both A/B sides.
VECTOR_WIDTH = 4
#: Trip count of the feature case; ``n % W == 0`` on purpose (review F1).
TRIP_COUNT = 16
#: Element base address of the input array written by the report.
INPUT_BASE = 0x400000
#: Element base address of the output array produced by the case.
OUTPUT_BASE = 0x420000
#: Deterministic inputs (negative values included); the case must not use
#: division, whose negative semantics the RV32 emulator stores unsigned.
CASE_INPUTS = [7, -3, 0, 100000, -1, 2, -2048, 2047,
               123456, -654321, 0, -5, 9, -9, 42, -42]
#: ``out[i] = relu(a[i] + a[i])`` computed in Python for the same inputs.
EXPECTED_OUTPUTS = [max(value + value, 0) for value in CASE_INPUTS]
#: Expected return value: the case returns ``out[0]``.
EXPECTED_RESULT = EXPECTED_OUTPUTS[0]
#: Registers that carry observable results.  Loop counters and strip
#: induction variables necessarily differ between the scalar and the
#: vectorized program, so only the return register is compared; the full
#: state is still recorded for auditing, as is the output memory.
OBSERVED_REGISTERS = ("x10",)
#: Illegal ``vector_width`` values for the pre-validation matrix.
INVALID_WIDTHS = (0, 1, -2, True, "4")

_VECTOR_MNEMONIC_RE = re.compile(r"^\s*v[a-z]", re.MULTILINE)

HONESTY = (
    "Deterministic phase-1 feature case, not a workload speedup claim.  "
    "The program is the canonical element-addressing loop of design doc "
    "appendix 5.1 (out[i] = relu(a[i] + a[i]), n=16, W=4, rem=0), built "
    "with IRBuilder because the phase-1 DSL grammar has no array "
    "load/store syntax; CompilerDriver still reads and validates the DSL "
    "case file on every run and only the final parse step is replaced by "
    "the equivalent IR program.  Known review findings are avoided by "
    "shape: n % W == 0 leaves the remainder clone untouched (F1), the "
    "original induction variable is not used after ENDFOR (F2), and the "
    "single element base plus a provably disjoint constant output base "
    "cannot alias (F3); the branch's regression tests for those defect "
    "shapes live in tests/test_vectorize.py and "
    "tests/test_vector_lowering.py.  The case also stays at W=4/n=16 "
    "because the pre-existing greedy allocator miscompiles vectorized "
    "loops that exceed its 19-register window (e.g. the two-load mul map "
    "at W=4).  Instruction counts are RV32 emulator dynamic counts, not "
    "hardware cycles; phase 1 lowers vector ops to per-lane scalar "
    "RV32IM, so the assembly contains no vector mnemonics and the "
    "observed dynamic-instruction drop is loop-overhead amortization on "
    "this case, not a general speedup.  The report writes the input "
    "arrays itself, so no external workload is run."
)


class _FeatureCaseDriver(CompilerDriver):
    """CompilerDriver that compiles the pre-built IR feature case.

    The driver, its pre-validation, the configured passes, the vectorizer,
    the backend and the output writer are used unmodified; only the final
    ``_parse`` step returns the IRBuilder-built equivalent of the DSL case
    because the phase-1 DSL grammar cannot express array element chains.
    """

    def __init__(self, config: CompilerConfig, program: Program) -> None:
        super().__init__(config)
        self.case_program = program

    def _parse(self, input_path: str, dsl_source: str | None = None):
        return self.case_program


def build_case_program(n: int = TRIP_COUNT) -> Program:
    """Build the deterministic vectorizable feature case.

    Shape: ``out[i] = relu(a[i] + a[i])`` for ``i in [0, n)``, followed by
    ``return out[0]``.  The shape is chosen inside the phase-1 verified
    subset: ``n % W == 0`` (no remainder clone, review F1), the induction
    variable is not used after ``endfor`` (review F2) and the element base
    plus the disjoint constant output base cannot alias (review F3).
    """
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    base = builder.load_const(INPUT_BASE, dtype=DataType.INT32)
    out = builder.load_const(OUTPUT_BASE, dtype=DataType.INT32)
    iv = builder.for_loop(0, n)
    c4 = builder.load_const(4, dtype=DataType.INT32)
    offset = builder.mul(iv, c4)
    va = builder.load(builder.add(base, offset))
    r = builder.relu(builder.add(va, va))
    builder.store(builder.add(out, offset), r)
    builder.endfor()
    last = builder.load(out)
    builder.ret(last)
    return builder.program


def count_ir(program: Program) -> int:
    return sum(
        1
        for func in program.functions
        for block in func.blocks
        for _ in block.instructions
    )


def vector_op_counts(program: Program) -> dict[str, int]:
    counts: dict[str, int] = {}
    for func in program.functions:
        for block in func.blocks:
            for instr in block.instructions:
                if instr.opcode.is_vector():
                    key = instr.opcode.value
                    counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_asm(asm: str) -> int:
    return sum(
        line.opcode is not None and not line.is_directive
        for line in parse_asm(asm)
    )


def execute_binary(binary: bytes) -> dict[str, Any]:
    """Execute *binary* with the case inputs; return state and counters."""
    emulator = RV32Emulator()
    emulator.load_code(binary)
    for index, value in enumerate(CASE_INPUTS):
        emulator.write_i32(INPUT_BASE + 4 * index, value)
    dynamic = emulator.run(max_instr=200000)
    outputs = [
        emulator.read_i32(OUTPUT_BASE + 4 * index)
        for index in range(TRIP_COUNT)
    ]
    return {
        "backend": "rv32-emulator",
        "registers": {f"x{i}": emulator.regs[i] for i in range(32)},
        "a0": emulator.regs[REG_ID["a0"]],
        "outputs": outputs,
        "dynamic_instructions": dynamic,
    }


def _empty_side(vectorize: bool,
                errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "vectorize": vectorize,
        "success": False,
        "errors": list(errors or []),
        "warnings": [],
        "opt_message": "",
        "vector_ops": 0,
        "vector_ops_by_opcode": {},
        "ir_instructions": 0,
        "asm_instructions": 0,
        "asm_bytes": 0,
        "binary_bytes": 0,
        "assembles": False,
        "vector_mnemonics_in_asm": False,
        "compile_time_ms": 0.0,
        "runs": 0,
        "artifact_deterministic": False,
        "asm_sha256": "",
        "execution": None,
        "asm_head": [],
    }


def _measure_side(*, vectorize: bool, repeats: int,
                  case_path: Path = DEFAULT_CASE) -> dict[str, Any]:
    """Compile and execute the feature case with vectorization off/on.

    Runs at least twice so ``artifact_deterministic`` is never vacuous.
    """
    runs = max(repeats, 2)
    times: list[float] = []
    asm_texts: list[str] = []
    program: Program | None = None
    result = None
    for _ in range(runs):
        program = build_case_program()
        driver = _FeatureCaseDriver(
            CompilerConfig(
                vectorize=vectorize,
                vector_width=VECTOR_WIDTH,
                reg_alloc="greedy",
                optimize_level="none",
            ),
            program,
        )
        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            result = driver.compile(str(case_path), str(Path(tmp) / "case.s"))
            times.append((time.perf_counter() - started) * 1000.0)
        if not result.success:
            return _empty_side(vectorize, result.errors)
        asm_texts.append(result.output_text)

    assert program is not None and result is not None
    asm = asm_texts[-1]
    counts = vector_op_counts(program)
    side: dict[str, Any] = {
        "vectorize": vectorize,
        "success": True,
        "errors": [],
        "warnings": list(result.warnings),
        "opt_message": result.stats.get("opt_message", ""),
        "vector_ops": sum(counts.values()),
        "vector_ops_by_opcode": counts,
        "ir_instructions": count_ir(program),
        "asm_instructions": count_asm(asm),
        "asm_bytes": len(asm),
        "binary_bytes": 0,
        "assembles": False,
        "vector_mnemonics_in_asm": bool(_VECTOR_MNEMONIC_RE.search(asm)),
        "compile_time_ms": round(statistics.median(times), 4),
        "runs": runs,
        "artifact_deterministic": len(set(asm_texts)) == 1,
        "asm_sha256": hashlib.sha256(asm.encode("utf-8")).hexdigest(),
        "execution": None,
        "asm_head": asm.splitlines()[:12],
    }
    try:
        binary = bytes(assemble_to_binary(asm))
    except Exception as exc:  # pragma: no cover - report, never crash
        side["assemble_error"] = str(exc)
        return side
    side["assembles"] = len(binary) > 0
    side["binary_bytes"] = len(binary)
    try:
        side["execution"] = execute_binary(binary)
    except Exception as exc:  # pragma: no cover - report, never crash
        side["execution_error"] = str(exc)
    return side


def measure_dsl_wiring(case_path: Path) -> dict[str, Any]:
    """Prove the opt-in flag is wired through the real DSL driver path."""
    source = case_path.read_text(encoding="utf-8")
    common = dict(optimize_level="none", reg_alloc="greedy")
    with tempfile.TemporaryDirectory() as tmp:
        off = CompilerDriver(
            CompilerConfig(vectorize=False, **common)
        ).compile("", str(Path(tmp) / "off.s"), dsl_source=source)
        on = CompilerDriver(
            CompilerConfig(
                vectorize=True, vector_width=VECTOR_WIDTH, **common,
            )
        ).compile("", str(Path(tmp) / "on.s"), dsl_source=source)
    off_message = off.stats.get("opt_message", "")
    on_message = on.stats.get("opt_message", "")
    return {
        "off_success": off.success,
        "on_success": on.success,
        "off_vectorizer_ran": "vectorized" in off_message,
        "on_vectorizer_ran": "vectorized" in on_message,
        "on_opt_message": on_message,
        "on_rejections": [
            warning for warning in on.warnings if "rejected" in warning
        ],
    }


def _attempt_failure(case_path: Path, label: str,
                     **overrides: Any) -> dict[str, Any]:
    config = CompilerConfig(
        vectorize=True,
        vector_width=VECTOR_WIDTH,
        optimize_level="none",
        reg_alloc="greedy",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    driver = _FeatureCaseDriver(config, build_case_program())
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / (
            "out.ll" if config.backend == "llvm" else "out.s"
        )
        result = driver.compile(str(case_path), str(output))
        artifact_written = output.exists()
    return {
        "label": label,
        "config": {
            "backend": config.backend,
            "vectorize": config.vectorize,
            "vector_width": config.vector_width,
            "vector_isa": config.vector_isa,
        },
        "success": result.success,
        "errors": list(result.errors),
        "artifact_written": artifact_written,
    }


def measure_failure_matrix(case_path: Path) -> dict[str, Any]:
    """Run the driver pre-validation matrix for unsupported configurations."""
    llvm = _attempt_failure(case_path, "backend=llvm", backend="llvm")
    widths = [
        _attempt_failure(case_path, f"vector_width={value!r}",
                         vector_width=value)
        for value in INVALID_WIDTHS
    ]
    vector_isa = _attempt_failure(case_path, "vector_isa=v", vector_isa="v")
    all_rows = [llvm, vector_isa, *widths]
    return {
        "llvm": llvm,
        "vector_isa": vector_isa,
        "invalid_widths": widths,
        "llvm_rejected": (
            not llvm["success"]
            and any("llvm" in error.lower() for error in llvm["errors"])
        ),
        "invalid_widths_rejected": all(
            (not row["success"])
            and any("width" in error for error in row["errors"])
            for row in widths
        ),
        "vector_isa_rejected": (
            not vector_isa["success"]
            and any("phase 2" in error for error in vector_isa["errors"])
        ),
        "no_artifact_written_on_failure": all(
            not row["artifact_written"] for row in all_rows
        ),
    }


def evaluate(case_path: Path, repeats: int) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    off = _measure_side(vectorize=False, repeats=repeats, case_path=case_path)
    on = _measure_side(vectorize=True, repeats=repeats, case_path=case_path)
    wiring = measure_dsl_wiring(case_path)
    matrix = measure_failure_matrix(case_path)

    exec_off = off.get("execution") or {}
    exec_on = on.get("execution") or {}
    hard_checks = {
        "dsl_case_compiles_with_and_without_opt_in": (
            wiring["off_success"] and wiring["on_success"]
        ),
        "dsl_opt_in_runs_vectorizer_only_when_enabled": (
            wiring["on_vectorizer_ran"] and not wiring["off_vectorizer_ran"]
        ),
        "both_configs_compile": off["success"] and on["success"],
        "vector_ops_present_when_enabled": on["vector_ops"] > 0,
        "vector_ops_absent_when_disabled": off["vector_ops"] == 0,
        "opt_in_reports_vectorized_loop": (
            "vectorized 1/1" in on["opt_message"]
        ),
        "on_artifact_assembles": on["assembles"],
        "output_is_rv32im_only": (
            not on["vector_mnemonics_in_asm"]
            and not off["vector_mnemonics_in_asm"]
        ),
        "execution_result_is_expected": bool(
            exec_off and exec_on
            and exec_off["a0"] == EXPECTED_RESULT
            and exec_on["a0"] == EXPECTED_RESULT
            and exec_off["outputs"] == EXPECTED_OUTPUTS
            and exec_on["outputs"] == EXPECTED_OUTPUTS
        ),
        "observed_registers_identical": bool(
            exec_off and exec_on
            and all(
                exec_off["registers"][reg] == exec_on["registers"][reg]
                for reg in OBSERVED_REGISTERS
            )
        ),
        "output_memory_identical": bool(
            exec_off and exec_on
            and exec_off["outputs"] == exec_on["outputs"]
        ),
        "llvm_backend_rejected": matrix["llvm_rejected"],
        "invalid_widths_rejected": matrix["invalid_widths_rejected"],
        "vector_isa_rejected": matrix["vector_isa_rejected"],
        "failed_configs_write_no_artifact": matrix[
            "no_artifact_written_on_failure"
        ],
        "artifacts_deterministic": (
            off["artifact_deterministic"] and on["artifact_deterministic"]
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic29-simd-vectorize",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "vector_width": VECTOR_WIDTH,
        "trip_count": TRIP_COUNT,
        "strips": TRIP_COUNT // VECTOR_WIDTH,
        "remainder": TRIP_COUNT % VECTOR_WIDTH,
        "expected_result": EXPECTED_RESULT,
        "expected_outputs": EXPECTED_OUTPUTS,
        "observed_registers": list(OBSERVED_REGISTERS),
        "case_inputs": CASE_INPUTS,
        "config": {
            "optimize_level": "none",
            "reg_alloc": "greedy",
            "vector_width": VECTOR_WIDTH,
        },
        "runs": repeats,
        "dsl_wiring": wiring,
        "vectorize_off": off,
        "vectorize_on": on,
        "failure_matrix": matrix,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": HONESTY,
    }


def render_markdown(report: dict[str, Any]) -> str:
    off, on = report["vectorize_off"], report["vectorize_on"]
    exec_off = off.get("execution") or {}
    exec_on = on.get("execution") or {}
    dyn_off = exec_off.get("dynamic_instructions", 0)
    dyn_on = exec_on.get("dynamic_instructions", 0)
    saved = dyn_off - dyn_on
    pct = (saved / dyn_off * 100) if dyn_off else 0.0
    opcodes = sorted(
        set(off["vector_ops_by_opcode"]) | set(on["vector_ops_by_opcode"])
    )
    lines = [
        "# Topic 29 SIMD Vectorize Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}` "
        f"(IR shape `out[i] = relu(a[i] + a[i])`, n="
        f"{report['trip_count']}, W={report['vector_width']}, rem="
        f"{report['remainder']})",
        f"- Generated: {report['generated_at']}",
        f"- Hard checks: "
        f"{'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({len(report['hard_checks']) - len(report['hard_failures'])}"
        f"/{len(report['hard_checks'])})",
        "",
        "## A/B summary",
        "",
        "| Metric | vectorize off | vectorize on | delta |",
        "|--------|--------------:|-------------:|------:|",
        f"| Compilation success | {'yes' if off['success'] else 'no'} | "
        f"{'yes' if on['success'] else 'no'} | - |",
        f"| IR instructions | {off['ir_instructions']} | "
        f"{on['ir_instructions']} | "
        f"{on['ir_instructions'] - off['ir_instructions']:+d} |",
        f"| Vector ops (IR) | {off['vector_ops']} | {on['vector_ops']} | "
        f"{on['vector_ops'] - off['vector_ops']:+d} |",
        f"| ASM instructions | {off['asm_instructions']} | "
        f"{on['asm_instructions']} | "
        f"{on['asm_instructions'] - off['asm_instructions']:+d} |",
        f"| Binary bytes | {off['binary_bytes']} | {on['binary_bytes']} | "
        f"{on['binary_bytes'] - off['binary_bytes']:+d} |",
        f"| Dynamic instructions (emulator) | {dyn_off} | {dyn_on} | "
        f"-{saved} ({pct:.1f}%) |",
        f"| `a0` result | {exec_off.get('a0', 'n/a')} | "
        f"{exec_on.get('a0', 'n/a')} | expected "
        f"{report['expected_result']} |",
        f"| Compile time (ms, median of {on['runs']} runs) | "
        f"{off['compile_time_ms']:.4f} | {on['compile_time_ms']:.4f} | - |",
        f"| Artifact SHA-256 | `{off['asm_sha256'][:16]}` | "
        f"`{on['asm_sha256'][:16]}` | - |",
        "",
        "## Vector op statistics (on)",
        "",
        "| Opcode | off | on |",
        "|--------|----:|---:|",
    ]
    for opcode in opcodes:
        lines.append(
            f"| `{opcode}` | {off['vector_ops_by_opcode'].get(opcode, 0)} | "
            f"{on['vector_ops_by_opcode'].get(opcode, 0)} |"
        )
    residual = (
        "YES" if on["vector_mnemonics_in_asm"]
        else "none (lowered per-lane to RV32IM)"
    )
    lines += [
        "",
        f"- Vectorizer message: `{on['opt_message']}`",
        f"- Residual vector mnemonics in on-side assembly: {residual}",
        f"- Deterministic artifacts across {on['runs']} compiles: "
        f"{'yes' if on['artifact_deterministic'] else 'no'}",
        "",
        "## Execution equivalence (RV32 emulator)",
        "",
        f"- Inputs written at `0x{INPUT_BASE:x}`: {report['case_inputs']}",
        f"- Expected outputs: {report['expected_outputs']}",
        f"- off outputs: {exec_off.get('outputs', 'n/a')}",
        f"- on outputs:  {exec_on.get('outputs', 'n/a')}",
        f"- `a0`: off={exec_off.get('a0', 'n/a')}, "
        f"on={exec_on.get('a0', 'n/a')}, "
        f"expected={report['expected_result']}",
        f"- Observed registers {report['observed_registers']}:",
    ]
    for side_name, execution in (("off", exec_off), ("on", exec_on)):
        observed = [
            execution.get("registers", {}).get(reg)
            for reg in report["observed_registers"]
        ]
        lines.append(f"  - {side_name}: {observed}")
    lines += [
        "",
        "## DSL wiring (real CompilerDriver path)",
        "",
        f"- off compile success: "
        f"{report['dsl_wiring']['off_success']}, vectorizer ran: "
        f"{report['dsl_wiring']['off_vectorizer_ran']}",
        f"- on compile success: "
        f"{report['dsl_wiring']['on_success']}, vectorizer ran: "
        f"{report['dsl_wiring']['on_vectorizer_ran']} "
        f"(`{report['dsl_wiring']['on_opt_message']}`)",
        f"- on-side phase-1 rejections: "
        f"{report['dsl_wiring']['on_rejections'] or 'none'}",
        "",
        "## Rejection matrix (driver pre-validation)",
        "",
        "| Config | success | error |",
        "|--------|---------|-------|",
    ]
    matrix = report["failure_matrix"]
    for row in [matrix["llvm"], matrix["vector_isa"],
                *matrix["invalid_widths"]]:
        error = row["errors"][0] if row["errors"] else ""
        if len(error) > 90:
            error = error[:87] + "..."
        lines.append(
            f"| {row['label']} | {row['success']} | {error} |"
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
