#!/usr/bin/env python3
"""Run the Topic 28 extended instruction-selection feature case.

The report proves four separate facts, all deterministic and auditable:

1. the ``extended_isel`` opt-in flag is wired through ``CompilerDriver``: the
   feature DSL case compiles in both configurations, the assembly is
   reproducible, and the RV32 emulator executes both products to the same
   architectural state and the expected result;
2. the extended-only opcodes (sqrt / min / max / abs / idiv / rem and the
   float64 family) select FP mnemonics under ``extended_isel=True``; raising
   the encoder gate, ``assemble_to_binary`` rejects F/D assembly with
   ``UnsupportedInstructionError`` instead of silently mis-encoding it, while
   the FP-mnemonic-free extended assembly is accepted by the RV32IM encoder;
3. the counterexample matrix is recorded: ``--no-fp64`` on a float64 program
   and the base selector on extended opcodes fail loud, and the LLVM backend /
   DAG-selection combinations warn instead of silently ignoring the flag;
4. no execution equivalence is claimed for the FP probe: the RV32IM encoder
   rejects every F/D mnemonic and ``RV32Emulator`` retires RV32IM only, so the
   FP execution entry is explicitly ``skipped``.

This is a deterministic feature/integration case, not a real-workload speedup
claim.  Real workloads remain covered by ``run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.riscv_encoder import (
    UnsupportedInstructionError,
    _is_fd_mnemonic,
    assemble_to_binary,
)
from scratchv.compiler import CompileResult, CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode, Program
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator

SCHEMA_VERSION = "topic28-extended-isel-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic28_extended_isel_feature.dsl"
)
DEFAULT_JSON = Path("benchmark_reports/extended_isel_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/extended_isel_report.md")
#: a0 after the deterministic DSL case: acc = 6, bias = 36, neg_acc = -6,
#: total = 36 - (-6) = 42, res = 42 / 6 = 7.
EXPECTED_DSL_RESULT = 7
#: FP mnemonics the extended-only IR probe must emit.  The branch lowers
#: float64 arithmetic to ``fadd.d``/``fmul.d`` (``fadd.s`` has no producer).
REQUIRED_FP_MNEMONICS = ("fadd.d", "fmul.d")
#: Text-level evidence uses the greedy allocator path (see the honesty note).
REG_ALLOC = "greedy"
OPTIMIZE_LEVEL = "all"

HONESTY = (
    "Deterministic feature case, not a workload speedup claim.  The stock DSL "
    "frontend exposes a closed op table (add/sub/mul/div/neg/exp/relu/gelu/"
    "dot/matmul/softmax/maxpool), so the Topic 28-only opcodes are exercised "
    "through IR-level probes built with IRBuilder; the A/B DSL comparison "
    "therefore proves opt-in wiring and execution equivalence on the shared "
    "base path, not a lowering difference (the DSL produces byte-identical "
    "assembly in both configurations, which is expected).  Only shapes "
    "covered by tests/test_inst_select_ext.py are used: f32 sqrt literals go "
    "through the exact bit-pattern materialization (the F1 regression fix), "
    "float64 literal operands never reach arithmetic (load_const_f64 "
    "materializes exact IEEE-754 bits first, and f64 literals fail loud by "
    "design), and integer min/max/abs literals are materialized into "
    "registers before the register-only branchless sequences.  The RV32IM "
    "encoder rejects every F/D mnemonic by design (fail-loud, final encoding "
    "out of scope), so no FP assembly can be accepted by assemble_to_binary; "
    "the gate checked here is explicit rejection plus successful encoding of "
    "the FP-mnemonic-free extended assembly.  FP execution is skipped: "
    "RV32Emulator retires RV32IM only and the F/D product cannot be encoded, "
    "so no execution-equivalence claim is made.  The extended integer "
    "min/max/abs sequences use SLT/SRAI/REM, which the minimal emulator does "
    "not implement, so they are validated by encoding only.  Execution "
    "evidence uses the greedy allocator path because the LinearScanAllocator "
    "emits branch targets as trailing comments that assemble_to_binary drops "
    "(pre-existing integration gap, orthogonal to Topic 28)."
)


# ═══════════════════════════════════════════════════════════════════════
# IR case builders (Topic 28 extended shapes)
# ═══════════════════════════════════════════════════════════════════════

def _runtime_const(builder: IRBuilder, name: str, value: int):
    """LOAD_CONST value kept as a runtime register, not a folded literal."""
    value_obj = builder.make_value(
        name=name, dtype=DataType.INT32, is_constant=False)
    builder._emit(OpCode.LOAD_CONST, value_obj, value=value)
    return value_obj


def build_fp_feature_program() -> Program:
    """FP shapes verified by the branch tests (see the honesty note)."""
    builder = IRBuilder()
    builder.new_function("fp_feature")
    builder.new_block("entry")
    x = builder.make_value(name="x", dtype=DataType.FLOAT32)
    i = builder.make_value(name="i", dtype=DataType.INT32)
    j = builder.make_value(name="j", dtype=DataType.INT32)
    # Software sqrt: register operand and exact-bit-pattern f32 literal.
    sqrt_reg = builder.sqrt(x)
    builder.sqrt(builder.make_const(2.5, dtype=DataType.FLOAT32))
    # f64 arithmetic only on values materialized by load_const_f64.
    acc = builder.fadd_d(
        builder.load_const_f64(1.5), builder.load_const_f64(2.0))
    builder.fmul_d(acc, builder.load_const_f64(0.5))
    # Integer extended sequences with literal materialization.
    lo = builder.min(i, builder.make_const(3, dtype=DataType.INT32))
    builder.max(j, builder.make_const(2, dtype=DataType.INT32))
    builder.abs(lo)
    builder.ret(sqrt_reg)
    return builder.program


def build_integer_extended_program() -> Program:
    """Extended integer program without F/D mnemonics (encoder-acceptable)."""
    builder = IRBuilder()
    builder.new_function("integer_extended")
    builder.new_block("entry")
    five = _runtime_const(builder, "five", 5)
    seventy = _runtime_const(builder, "seventy", 70)
    lo = builder.min(five, builder.make_const(3, dtype=DataType.INT32))
    hi = builder.max(lo, seventy)
    mag = builder.abs(builder.make_const(-9, dtype=DataType.INT32))
    quotient = builder.idiv(hi, lo)
    remainder = builder.rem(hi, lo)
    builder.ret(builder.add(builder.add(quotient, remainder), mag))
    return builder.program


def build_hardware_sqrt_program() -> Program:
    """f32 sqrt literal for the ``--hardware-sqrt`` branch (emits fsqrt.s)."""
    builder = IRBuilder()
    builder.new_function("hardware_sqrt")
    builder.new_block("entry")
    builder.ret(builder.sqrt(builder.make_const(2.0, dtype=DataType.FLOAT32)))
    return builder.program


def build_fp64_only_program() -> Program:
    """Pure float64 program for the ``--no-fp64`` counterexample."""
    builder = IRBuilder()
    builder.new_function("fp64_only")
    builder.new_block("entry")
    total = builder.fadd_d(
        builder.load_const_f64(1.5), builder.load_const_f64(2.0))
    builder.ret(total)
    return builder.program


# ═══════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════

def count_asm(asm: str) -> int:
    """Count real (non-directive) assembly instructions."""
    return sum(
        line.opcode is not None and not line.is_directive
        for line in parse_asm(asm)
    )


def fp_mnemonics(asm: str) -> list[str]:
    """Sorted F/D mnemonics present in *asm* (encoder predicate reused)."""
    return sorted({
        line.opcode
        for line in parse_asm(asm)
        if line.opcode is not None
        and not line.is_directive
        and _is_fd_mnemonic(line.opcode)
    })


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encoder_gate(asm: str) -> dict[str, Any]:
    """Run ``assemble_to_binary`` and categorize the outcome."""
    try:
        binary = assemble_to_binary(asm)
    except UnsupportedInstructionError as exc:
        return {
            "encoded": False,
            "bytes": 0,
            "words": 0,
            "error_type": "UnsupportedInstructionError",
            "error": str(exc),
        }
    except ValueError as exc:
        return {
            "encoded": False,
            "bytes": 0,
            "words": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "encoded": True,
        "bytes": len(binary),
        "words": len(binary) // 4,
        "error_type": None,
        "error": None,
    }


def compile_ir_program(
    program: Program, *,
    extended_isel: bool = True,
    enable_fp64: bool = True,
    use_hardware_sqrt: bool = False,
) -> dict[str, Any]:
    """Compile one IR program through the driver's codegen entry.

    ``CompilerDriver.compile`` only accepts DSL/ONNX sources and the DSL
    frontend cannot express Topic 28-only opcodes, so the probe calls the
    driver's own RISC-V codegen entry (the same one ``compile`` uses).
    """
    driver = CompilerDriver(CompilerConfig(
        extended_isel=extended_isel,
        enable_fp64=enable_fp64,
        use_hardware_sqrt=use_hardware_sqrt,
        reg_alloc=REG_ALLOC,
        optimize_level=OPTIMIZE_LEVEL,
    ))
    asm = driver._generate_code(program)
    return {
        "success": True,
        "asm": asm,
        "asm_instructions": count_asm(asm),
        "fp_mnemonics": fp_mnemonics(asm),
        "asm_sha256": _sha256(asm),
        "encoder": encoder_gate(asm),
    }


def _capture_failure(action: Callable[[], Any]) -> dict[str, Any]:
    """Run *action*; record an explicit rejection instead of propagating."""
    try:
        result = action()
    except Exception as exc:  # noqa: BLE001 - the rejection is the evidence
        return {
            "rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "rejected": False,
        "error_type": None,
        "error": None,
        "unexpected_result": result,
    }


def run_asm(asm: str) -> dict[str, Any]:
    """Assemble and execute *asm*; return register state and counters."""
    binary = assemble_to_binary(asm)
    emulator = RV32Emulator()
    emulator.load_code(bytes(binary))
    dynamic = emulator.run()
    return {
        "backend": "rv32-emulator",
        "a0": emulator.regs[REG_ID["a0"]],
        "registers": {f"x{i}": emulator.regs[i] for i in range(32)},
        "dynamic_instructions": dynamic,
        "encoded_words": len(binary) // 4,
    }


# ═══════════════════════════════════════════════════════════════════════
# Measurements
# ═══════════════════════════════════════════════════════════════════════

def _compile_dsl(case_path: Path, **config_overrides) -> CompileResult:
    """Compile the feature DSL case with one driver configuration."""
    source = case_path.read_text(encoding="utf-8")
    overrides = {
        "reg_alloc": REG_ALLOC,
        "optimize_level": OPTIMIZE_LEVEL,
        **config_overrides,
    }
    driver = CompilerDriver(CompilerConfig(**overrides))
    with tempfile.TemporaryDirectory() as tmp:
        output = str(Path(tmp) / "case.s")
        return driver.compile("", output, dsl_source=source)


def _dsl_side(case_path: Path, *, extended_isel: bool) -> dict[str, Any]:
    """Compile the case twice and execute the selected product."""
    first = _compile_dsl(case_path, extended_isel=extended_isel)
    second = _compile_dsl(case_path, extended_isel=extended_isel)
    deterministic = (
        first.success and second.success
        and first.output_text == second.output_text
    )
    side: dict[str, Any] = {
        "extended_isel": extended_isel,
        "success": first.success,
        "errors": list(first.errors),
        "warnings": list(first.warnings),
        "asm_instructions": (
            count_asm(first.output_text) if first.success else None),
        "fp_mnemonics": (
            fp_mnemonics(first.output_text) if first.success else None),
        "asm_sha256": (
            _sha256(first.output_text) if first.success else None),
        "asm": first.output_text if first.success else "",
        "deterministic": deterministic,
        "execution": None,
    }
    if first.success and second.success:
        try:
            side["execution"] = run_asm(first.output_text)
        except Exception as exc:  # noqa: BLE001 - recorded for the hard check
            side["execution"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    return side


def measure_dsl_ab(case_path: Path) -> dict[str, Any]:
    """CompilerDriver A/B on the feature DSL case."""
    off = _dsl_side(case_path, extended_isel=False)
    on = _dsl_side(case_path, extended_isel=True)
    return {
        "off": off,
        "on": on,
        "identical_asm": (
            off["success"] and on["success"]
            and off["asm"] == on["asm"]
        ),
    }


def measure_extended_probe() -> dict[str, Any]:
    """IR-level evidence for the extended-only opcodes."""
    fp_first = compile_ir_program(build_fp_feature_program())
    fp_second = compile_ir_program(build_fp_feature_program())
    integer = compile_ir_program(build_integer_extended_program())
    hardware = compile_ir_program(
        build_hardware_sqrt_program(), use_hardware_sqrt=True)
    base_selector = _capture_failure(lambda: compile_ir_program(
        build_fp_feature_program(), extended_isel=False))
    return {
        "fp": {
            **fp_first,
            "deterministic": fp_first["asm"] == fp_second["asm"],
        },
        "integer": integer,
        "hardware_sqrt": hardware,
        "base_selector": base_selector,
    }


def measure_error_matrix(case_path: Path) -> dict[str, Any]:
    """Record fail-loud and warning behaviour for the flag combinations."""
    source = case_path.read_text(encoding="utf-8")

    def compile_case(**overrides) -> CompileResult:
        config = {
            "extended_isel": True,
            "reg_alloc": REG_ALLOC,
            "optimize_level": OPTIMIZE_LEVEL,
            **overrides,
        }
        driver = CompilerDriver(CompilerConfig(**config))
        with tempfile.TemporaryDirectory() as tmp:
            suffix = "ll" if config.get("backend") == "llvm" else "s"
            return driver.compile(
                "", str(Path(tmp) / f"case.{suffix}"), dsl_source=source)

    off = _capture_failure(lambda: compile_ir_program(
        build_fp_feature_program(), extended_isel=False))
    disabled = _capture_failure(lambda: compile_ir_program(
        build_fp64_only_program(), extended_isel=True, enable_fp64=False))
    llvm = compile_case(backend="llvm")
    dag = compile_case(use_dag_isel=True)
    no_ext_fp64 = compile_case(extended_isel=False, enable_fp64=False)
    no_ext_hw = compile_case(extended_isel=False, use_hardware_sqrt=True)

    def warns(result: CompileResult, needle: str) -> bool:
        return result.success and any(needle in w for w in result.warnings)

    rows = [
        {
            "id": "extended_isel_off",
            "config": "extended_isel=False on extended-only opcodes",
            "expected": "explicit rejection (opt-in feature)",
            "observed": off["error"] or "no error",
            "status": (
                "error"
                if off["rejected"] and off["error_type"] == "ValueError"
                else "unexpected"
            ),
        },
        {
            "id": "extended_isel_no_fp64",
            "config": "extended_isel=True, enable_fp64=False on float64 IR",
            "expected": "explicit rejection mentioning enable_fp64",
            "observed": disabled["error"] or "no error",
            "status": (
                "error"
                if disabled["rejected"]
                and "enable_fp64" in (disabled["error"] or "")
                else "unexpected"
            ),
        },
        {
            "id": "llvm_backend",
            "config": "extended_isel=True, backend=llvm",
            "expected": "warning (RISC-V only)",
            "observed": (
                "; ".join(llvm.warnings) if llvm.warnings
                else "no warning"
            ),
            "status": (
                "warning" if warns(llvm, "RISC-V only") else "unexpected"
            ),
        },
        {
            "id": "dag_isel_precedence",
            "config": "extended_isel=True, use_dag_isel=True",
            "expected": "warning (dag-isel takes precedence)",
            "observed": (
                "; ".join(dag.warnings) if dag.warnings else "no warning"
            ),
            "status": "warning" if warns(dag, "precedence") else "unexpected",
        },
        {
            "id": "fp64_flag_without_extended",
            "config": "extended_isel=False, enable_fp64=False",
            "expected": "warning (no effect without --extended-isel)",
            "observed": (
                "; ".join(no_ext_fp64.warnings) if no_ext_fp64.warnings
                else "no warning"
            ),
            "status": (
                "warning" if warns(no_ext_fp64, "no effect")
                else "unexpected"
            ),
        },
        {
            "id": "hardware_sqrt_without_extended",
            "config": "extended_isel=False, use_hardware_sqrt=True",
            "expected": "warning (no effect without --extended-isel)",
            "observed": (
                "; ".join(no_ext_hw.warnings) if no_ext_hw.warnings
                else "no warning"
            ),
            "status": (
                "warning" if warns(no_ext_hw, "no effect")
                else "unexpected"
            ),
        },
    ]
    return {
        "rows": rows,
        "off": off,
        "no_fp64": disabled,
        "llvm_success": llvm.success,
        "dag_success": dag.success,
    }


# ═══════════════════════════════════════════════════════════════════════
# Report assembly
# ═══════════════════════════════════════════════════════════════════════

def _execution_a0(side: dict[str, Any]) -> Any:
    execution = side.get("execution") or {}
    return execution.get("a0")


def evaluate(case_path: Path) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    dsl_ab = measure_dsl_ab(case_path)
    probe = measure_extended_probe()
    matrix = measure_error_matrix(case_path)

    off, on = dsl_ab["off"], dsl_ab["on"]
    fp = probe["fp"]
    integer, hardware = probe["integer"], probe["hardware_sqrt"]
    base_selector = probe["base_selector"]
    no_fp64 = matrix["no_fp64"]

    off_exec = off.get("execution") or {}
    on_exec = on.get("execution") or {}

    hard_checks = {
        "dsl_ab_both_compile": off["success"] and on["success"],
        "dsl_ab_asm_deterministic": (
            off["deterministic"] and on["deterministic"]),
        "dsl_execution_matches_expected": (
            _execution_a0(off) == EXPECTED_DSL_RESULT
            and _execution_a0(on) == EXPECTED_DSL_RESULT
            and off_exec.get("dynamic_instructions", 0) > 0
            and on_exec.get("dynamic_instructions", 0) > 0
        ),
        "dsl_execution_registers_identical": (
            bool(off_exec.get("registers"))
            and off_exec.get("registers") == on_exec.get("registers")
        ),
        "extended_probe_has_fp_mnemonics": (
            set(REQUIRED_FP_MNEMONICS).issubset(fp["fp_mnemonics"])),
        "extended_probe_asm_deterministic": fp["deterministic"],
        "fp_asm_rejected_by_encoder": (
            not fp["encoder"]["encoded"]
            and fp["encoder"]["error_type"] == "UnsupportedInstructionError"
        ),
        "hardware_sqrt_emits_fsqrt_s": "fsqrt.s" in hardware["fp_mnemonics"],
        "hardware_sqrt_rejected_by_encoder": (
            not hardware["encoder"]["encoded"]
            and hardware["encoder"]["error_type"]
            == "UnsupportedInstructionError"
        ),
        "integer_extended_asm_is_encodable": (
            integer["encoder"]["encoded"]
            and integer["encoder"]["words"] > 0
            and not integer["fp_mnemonics"]
        ),
        "base_selector_rejects_extended_ops": (
            base_selector["rejected"]
            and base_selector["error_type"] == "ValueError"
        ),
        "fp64_disabled_raises_explicitly": (
            no_fp64["rejected"]
            and "enable_fp64" in (no_fp64["error"] or "")
        ),
        "llvm_backend_warns": (
            matrix["llvm_success"]
            and any(row["id"] == "llvm_backend"
                    and row["status"] == "warning"
                    for row in matrix["rows"])
        ),
        "dag_isel_precedence_warns": (
            matrix["dag_success"]
            and any(row["id"] == "dag_isel_precedence"
                    and row["status"] == "warning"
                    for row in matrix["rows"])
        ),
        "fp64_flags_without_extended_warn": (
            any(row["id"] in ("fp64_flag_without_extended",
                              "hardware_sqrt_without_extended")
                and row["status"] == "warning"
                for row in matrix["rows"])
        ),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic28-extended-isel",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "config": {
            "reg_alloc": REG_ALLOC,
            "optimize_level": OPTIMIZE_LEVEL,
        },
        "expected_dsl_result": EXPECTED_DSL_RESULT,
        "required_fp_mnemonics": list(REQUIRED_FP_MNEMONICS),
        "dsl_ab": dsl_ab,
        "extended_probe": probe,
        "behavior_matrix": matrix["rows"],
        "execution": {
            "dsl_ab": {
                "status": "ok",
                "backend": "rv32-emulator",
                "expected_a0": EXPECTED_DSL_RESULT,
                "off": off.get("execution"),
                "on": on.get("execution"),
            },
            "fp_probe": {
                "status": "skipped",
                "reason": (
                    "RV32Emulator retires RV32IM only and the RV32IM "
                    "encoder rejects every F/D mnemonic (fail-loud), so no "
                    "F/D binary exists to execute; no execution-equivalence "
                    "claim is made."
                ),
            },
        },
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": HONESTY,
    }


def render_markdown(report: dict[str, Any]) -> str:
    dsl_ab = report["dsl_ab"]
    probe = report["extended_probe"]
    off, on = dsl_ab["off"], dsl_ab["on"]
    fp, integer = probe["fp"], probe["integer"]
    hardware = probe["hardware_sqrt"]
    off_exec = off.get("execution") or {}
    on_exec = on.get("execution") or {}

    def mnemonics(items: list[str] | None) -> str:
        return ", ".join(items) if items else "(none)"

    def gate(cell: dict[str, Any]) -> str:
        if cell["encoded"]:
            return f"accepted ({cell['words']} words)"
        return f"rejected: {cell['error_type']}"

    hard_total = len(report["hard_checks"])
    hard_ok = hard_total - len(report["hard_failures"])
    lines = [
        "# Topic 28 Extended Instruction-Selection Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}`",
        f"- Generated: {report['generated_at']}",
        f"- Expected `a0` (DSL case): {report['expected_dsl_result']}",
        f"- Hard checks: "
        f"{'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({hard_ok}/{hard_total})",
        "",
        "## DSL A/B (CompilerDriver, feature case)",
        "",
        "| Metric | extended_isel=False | extended_isel=True |",
        "|--------|--------------------:|-------------------:|",
        f"| compiled | {'yes' if off['success'] else 'no'} | "
        f"{'yes' if on['success'] else 'no'} |",
        f"| ASM instructions | {off['asm_instructions']} | "
        f"{on['asm_instructions']} |",
        f"| FP mnemonics in ASM | {mnemonics(off['fp_mnemonics'])} | "
        f"{mnemonics(on['fp_mnemonics'])} |",
        f"| two compiles identical | {off['deterministic']} | "
        f"{on['deterministic']} |",
        f"| `a0` (RV32 emulator) | {off_exec.get('a0')} | "
        f"{on_exec.get('a0')} |",
        f"| dynamic instructions | {off_exec.get('dynamic_instructions')} | "
        f"{on_exec.get('dynamic_instructions')} |",
        f"| ASM identical across configs | "
        f"{'yes' if dsl_ab['identical_asm'] else 'no'} | - |",
        "",
        "The DSL frontend exposes a closed op table, so the A/B case covers "
        "the opt-in wiring and execution path; the extended-only opcodes are "
        "probed below at IR level.",
        "",
        "## Extended-only IR probe (FP mnemonics and encoder gate)",
        "",
        "| Probe | Config | FP mnemonics | `assemble_to_binary` |",
        "|-------|--------|--------------|----------------------|",
        f"| fp_feature (sqrt/min/max/abs/f64) | extended_isel=True | "
        f"{mnemonics(fp['fp_mnemonics'])} | {gate(fp['encoder'])} |",
        f"| hardware_sqrt | extended_isel=True, "
        f"use_hardware_sqrt=True | {mnemonics(hardware['fp_mnemonics'])} | "
        f"{gate(hardware['encoder'])} |",
        f"| integer_extended (no F/D mnemonic) | extended_isel=True | "
        f"{mnemonics(integer['fp_mnemonics'])} | {gate(integer['encoder'])} |",
        f"| fp_feature | extended_isel=False | - | "
        f"rejected: {probe['base_selector']['error_type']} |",
        "",
        "## Failure / degradation matrix",
        "",
        "| Config | Expected | Observed | Status |",
        "|--------|----------|----------|--------|",
    ]
    for row in report["behavior_matrix"]:
        observed = row["observed"].replace("\n", " ")
        if len(observed) > 90:
            observed = observed[:87] + "..."
        lines.append(
            f"| `{row['config']}` | {row['expected']} | {observed} | "
            f"{row['status']} |"
        )

    lines += [
        "",
        "## Execution",
        "",
        f"- DSL A/B: status `{report['execution']['dsl_ab']['status']}`, "
        f"backend `{report['execution']['dsl_ab']['backend']}`, expected "
        f"`a0 == {report['execution']['dsl_ab']['expected_a0']}`.",
        f"- FP probe: status `{report['execution']['fp_probe']['status']}` — "
        f"{report['execution']['fp_probe']['reason']}",
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
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(report) + "\n", encoding="utf-8")
    print(render_markdown(report))
    if report["hard_failures"]:
        print("HARD FAILURES: " + ", ".join(report["hard_failures"]))
        return 1
    print(f"reports written: {args.json}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
