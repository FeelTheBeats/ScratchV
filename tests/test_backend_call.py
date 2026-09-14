"""Tests for minimal CALL lowering in the RISC-V backend (Topic 15)."""

import os
import subprocess
import sys
import textwrap

import pytest

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import (
    InstructionSelector, UnsupportedCallError,
)
from scratchv.backend.machine_types import MachineOp, MachineOperand
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode


def _program_with_call(n_args: int = 2, is_tail: bool = False,
                       has_ret: bool = True):
    b = IRBuilder()
    params = [b.make_value(name=f"p{i}") for i in range(n_args)]
    b.new_function("inc", params=params)
    b.new_block("entry")
    if params:
        total = params[0]
        for p in params[1:]:
            total = b.add(total, p)
        b.ret(total)
    else:
        b.ret()

    b.new_function("main")
    b.new_block("entry")
    args = [b.make_value(name=f"x{i}") for i in range(n_args)]
    r = b.call("inc", args, has_ret=has_ret, is_tail=is_tail)
    b.ret(r)
    return b.program


def _program_with_8_arg_call():
    b = IRBuilder()
    params = [b.make_value(name=f"p{i}") for i in range(8)]
    b.new_function("callee", params=params)
    b.new_block("entry")
    total = params[0]
    for p in params[1:]:
        total = b.add(total, p)
    b.ret(total)

    b.new_function("main")
    b.new_block("entry")
    args = [b.make_value(name=f"x{i}") for i in range(8)]
    r = b.call("callee", args)
    b.ret(r)
    return b.program


def _build_inc_callee(b: IRBuilder) -> None:
    a = b.make_value(name="a")
    b_val = b.make_value(name="b")
    b.new_function("inc", params=[a, b_val])
    b.new_block("entry")
    t = b.add(a, b_val)
    b.ret(t)


def _extract_arg_moves(asm: str):
    """Split the staged CALL argument moves into (stage1, stage2) pairs."""
    stage1: list[tuple[str, str]] = []
    stage2: list[tuple[str, str]] = []
    for line in asm.splitlines():
        line = line.strip()
        if not line.startswith("mv "):
            continue
        body, _, comment = line[3:].partition("#")
        dst, src = (part.strip() for part in body.split(","))
        comment = comment.strip()
        if comment.startswith("arg") and comment.endswith("-> tmp"):
            stage1.append((dst, src))
        elif comment.startswith("tmp -> a"):
            stage2.append((dst, src))
    return stage1, stage2


def _simulate_moves(moves):
    """Symbolically execute moves; returns final register -> value mapping."""
    values: dict[str, str] = {}
    for dst, src in moves:
        values[dst] = values.get(src, src)
    return values


def _assert_args_preserved(asm: str, n_args: int = 8) -> None:
    """Every argument must reach its a{i} register unclobbered."""
    stage1, stage2 = _extract_arg_moves(asm)
    assert len(stage1) == n_args
    assert len(stage2) == n_args
    assert [dst for dst, _ in stage2] == [f"a{i}" for i in range(n_args)]
    values = _simulate_moves(stage1 + stage2)
    assert [values[f"a{i}"] for i in range(n_args)] == [
        src for _, src in stage1]


def _call_instr(program):
    return next(
        ins
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
        if ins.opcode is OpCode.CALL
    )


def test_default_raises_unsupported_call_error():
    program = _program_with_call()
    sel = InstructionSelector(program)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    msg = str(excinfo.value)
    assert msg.startswith("CALL ")
    assert "ABI" in msg
    assert "inc" in msg
    assert "'main'" in msg


def test_minimal_call_asm():
    program = _program_with_call()
    call = _call_instr(program)

    sel = InstructionSelector(program, allow_uninlined_calls=True)
    instrs = sel.run()
    ops = [i.op.value for i in instrs]
    assert "jal" in ops
    assert "mv" in ops

    jal_index = next(
        idx for idx, i in enumerate(instrs) if i.op is MachineOp.JAL)
    jal = instrs[jal_index]
    assert jal.dst == MachineOperand.reg("ra")
    assert jal.comment == "inc"

    moves = [
        i for i in instrs[:jal_index]
        if i.op is MachineOp.MV and (
            i.comment.startswith("arg") or i.comment.startswith("tmp -> a"))
    ]
    assert len(moves) == 4  # 2 staging + 2 argument registers
    stage1, stage2 = moves[:2], moves[2:]

    # Stage 1 reads the arguments into fresh temporaries ...
    assert stage1[0].dst.kind == "vreg"
    assert stage1[0].dst == MachineOperand.vreg("_call_arg0_1")
    assert stage1[0].src1 == MachineOperand.vreg("x0")
    assert stage1[1].dst == MachineOperand.vreg("_call_arg1_1")
    assert stage1[1].src1 == MachineOperand.vreg("x1")
    assert stage1[0].dst != stage1[1].dst

    # ... stage 2 only then writes the physical a-registers.
    assert stage2[0].dst == MachineOperand.reg("a0")
    assert stage2[0].src1 == stage1[0].dst
    assert stage2[1].dst == MachineOperand.reg("a1")
    assert stage2[1].src1 == stage1[1].dst

    ret_move = instrs[jal_index + 1]
    assert ret_move.op is MachineOp.MV
    assert ret_move.dst == MachineOperand.vreg(call.dest.name)
    assert ret_move.src1 == MachineOperand.reg("a0")

    asm = AsmEmitter(instrs).emit()
    assert "jal ra, inc" in asm


def test_minimal_call_arg_moves_are_parallel_safe():
    """Stage-1 must read every argument before any a-register is written."""
    program = _program_with_8_arg_call()
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    instrs = sel.run()
    jal_index = next(
        idx for idx, i in enumerate(instrs) if i.op is MachineOp.JAL)

    moves = [
        i for i in instrs[:jal_index]
        if i.op is MachineOp.MV and (
            i.comment.startswith("arg") or i.comment.startswith("tmp -> a"))
    ]
    assert len(moves) == 16
    for instr in moves[:8]:
        assert instr.dst.kind == "vreg"  # staging reads only
    for instr in moves[8:]:
        assert instr.dst.kind == "reg"
        assert instr.dst.value.startswith("a")
        assert instr.src1.kind == "vreg"  # a-regs written from temps only


@pytest.mark.parametrize("seed", ["0", "1", "4"])
def test_minimal_call_asm_preserves_args_under_fixed_hash_seed(seed, tmp_path):
    """End-to-end asm must stage arguments without clobbering sources.

    The linear allocator's decisions depend on the Python hash seed, so the
    reviewed failing seeds are replayed in a subprocess.
    """
    script = tmp_path / "emit_call_asm.py"
    script.write_text(textwrap.dedent(
        """
        import sys
        from scratchv.compiler import CompilerConfig, CompilerDriver
        from scratchv.ir.builder import IRBuilder

        b = IRBuilder()
        params = [b.make_value(name=f"p{i}") for i in range(8)]
        b.new_function("callee", params=params)
        b.new_block("entry")
        total = params[0]
        for p in params[1:]:
            total = b.add(total, p)
        b.ret(total)

        b.new_function("main")
        b.new_block("entry")
        args = [b.make_value(name=f"x{i}") for i in range(8)]
        r = b.call("callee", args)
        b.ret(r)

        driver = CompilerDriver(CompilerConfig(
            minimal_call_codegen=True, reg_alloc="linear"))
        sys.stdout.write(driver._generate_riscv_linear(b.program))
        """
    ))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root)
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        env=env, cwd=root, check=True,
    )
    _assert_args_preserved(proc.stdout)


def test_minimal_void_call_has_no_return_move():
    program = _program_with_call(n_args=1, has_ret=False)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    instrs = sel.run()
    jal_index = next(
        idx for idx, i in enumerate(instrs) if i.op is MachineOp.JAL)
    assert not any(
        i.op is MachineOp.MV for i in instrs[jal_index + 1:])


def test_more_than_8_args_always_raises():
    program = _program_with_call(n_args=9)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    assert "args > 8" in str(excinfo.value)


def test_tail_call_always_raises():
    program = _program_with_call(is_tail=True)
    sel = InstructionSelector(program, allow_uninlined_calls=True)
    with pytest.raises(UnsupportedCallError) as excinfo:
        sel.run()
    assert "tail" in str(excinfo.value)


def test_driver_default_rejects_residual_call():
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    driver = CompilerDriver(CompilerConfig())
    with pytest.raises(UnsupportedCallError):
        driver._generate_riscv_linear(program)


def test_driver_minimal_call_codegen_emits_jal():
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    driver = CompilerDriver(CompilerConfig(
        minimal_call_codegen=True, reg_alloc="greedy"))
    asm = driver._generate_riscv_linear(program)
    assert "jal ra, inc" in asm


def test_compile_reports_codegen_error_and_writes_nothing(tmp_path):
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    output = tmp_path / "out.s"
    driver = CompilerDriver(CompilerConfig())
    driver._parse = lambda *args, **kwargs: program

    result = driver.compile("dummy.onnx", str(output))

    assert result.success is False
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Codegen error:")
    assert "ABI" in result.errors[0]
    assert "inc" in result.errors[0]
    assert not output.exists()


def test_compile_failure_carries_inliner_warnings(tmp_path):
    """Rejected inlining reasons survive into the failed CompileResult."""
    from scratchv.compiler import CompilerConfig, CompilerDriver

    b = IRBuilder()
    a = b.make_value(name="a")
    b.new_function("f", params=[a])
    b.new_block("entry")
    r = b.call("f", [a])  # direct recursion: inliner rejects
    b.ret(r)
    b.new_function("main")
    b.new_block("entry")
    x = b.make_const(1.0)
    rm = b.call("f", [x])
    b.ret(rm)

    output = tmp_path / "out.s"
    driver = CompilerDriver(CompilerConfig(
        optimize_level="basic", inline=True))
    driver._parse = lambda *args, **kwargs: b.program

    result = driver.compile("dummy.onnx", str(output))

    assert result.success is False
    assert any("recursive_callee" in w for w in result.warnings)
    assert any("inliner: skip" in w for w in result.warnings)
    assert "Codegen error" in result.errors[0]
    assert not output.exists()


def test_llvm_backend_rejects_residual_call():
    from scratchv.backend.llvm_codegen import LLVMCodegen

    program = _program_with_call()
    with pytest.raises(UnsupportedCallError) as excinfo:
        LLVMCodegen(program).emit()
    msg = str(excinfo.value)
    assert "CALL inc" in msg
    assert "LLVM" in msg


def test_llvm_backend_inlined_program_still_emits():
    from scratchv.backend.llvm_codegen import LLVMCodegen

    b = IRBuilder()
    _build_inc_callee(b)
    b.new_function("main")
    b.new_block("entry")
    x = b.make_const(1.0)
    y = b.make_const(2.0)
    r = b.call("inc", [x, y])
    b.ret(r)

    from scratchv.optimizer.inliner import Inliner, InlinerConfig
    assert Inliner(b.program, InlinerConfig()).run() == 1
    text = LLVMCodegen(b.program).emit()
    assert "UNSUPPORTED" not in text


def test_dag_isel_call_reports_error():
    """DAG ISel has no CALL builder; the driver surfaces a clear failure."""
    from scratchv.compiler import CompilerConfig, CompilerDriver

    program = _program_with_call()
    driver = CompilerDriver(CompilerConfig(use_dag_isel=True))
    with pytest.raises(ValueError, match="call"):
        driver._generate_riscv_dag(program)
