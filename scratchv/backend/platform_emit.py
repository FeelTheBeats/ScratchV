"""Platform (``.s``) emission for the main pipeline (Phase 3).

Lowers a single size-independent operator program to the backend's
``MachineInstr`` representation and renders it with :class:`AsmEmitter`, with no
dependency on the standalone string generators. The kernels are hand-scheduled
onto physical registers (like the standalone ones they replace), so register
allocation is not involved; the point is to express them through the shared
machine layer instead of copying strings.

ABI (FP32 / rv32imf, self-describing input): ``a0`` input, ``a1`` output,
``a2`` size scalar; entry ``cnn_entry``; ``sp`` preset by the caller.
"""

from __future__ import annotations

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import MachineInstr, MachineOp, MachineOperand
from scratchv.ir.types import OpCode, Program


class PlatformEmitError(ValueError):
    """A graph cannot be emitted as a single size-independent platform kernel."""


_ENTRY = "cnn_entry"


def _reg(name: str) -> MachineOperand:
    return MachineOperand.reg(name)


def _imm(value: int) -> MachineOperand:
    return MachineOperand.immediate(value)


def _i(op: MachineOp, dst=None, src1=None, src2=None, comment="", target=None):
    return MachineInstr(op, dst, src1, src2, comment, target)


def _label(name: str) -> MachineInstr:
    return MachineInstr(MachineOp.LABEL, target=name)


# ── FWHT (a0=in, a1=out, a2=N; forward Hadamard) ───────────────────────────

def _lower_fwht() -> list[MachineInstr]:
    L = _label
    code = [
        L(_ENTRY),
        _i(MachineOp.MV, _reg("t0"), _reg("a0"), comment="src = in"),
        _i(MachineOp.MV, _reg("t1"), _reg("a1"), comment="dst = out"),
        _i(MachineOp.MV, _reg("t2"), _reg("a2"), comment="N"),
        _i(MachineOp.MV, _reg("t3"), _reg("zero"), comment="i = 0"),
        L(".Lp_fwht_copy"),
        _i(MachineOp.FLW, _reg("ft0"), _reg("0(t0)")),
        _i(MachineOp.FSW, _reg("ft0"), _reg("0(t1)")),
        _i(MachineOp.ADDI, _reg("t0"), _reg("t0"), _imm(4)),
        _i(MachineOp.ADDI, _reg("t1"), _reg("t1"), _imm(4)),
        _i(MachineOp.ADDI, _reg("t3"), _reg("t3"), _imm(1)),
        _i(MachineOp.BLT, _reg("t3"), _reg("t2"), target=".Lp_fwht_copy"),
        _i(MachineOp.LI, _reg("t5"), _imm(1), comment="length = 1"),
        L(".Lp_fwht_len"),
        _i(MachineOp.BGE, _reg("t5"), _reg("t2"), target=".Lp_fwht_done"),
        _i(MachineOp.SLLI, _reg("t6"), _reg("t5"), _imm(1), comment="step = 2*length"),
        _i(MachineOp.LI, _reg("s2"), _imm(0), comment="i = 0"),
        L(".Lp_fwht_i"),
        _i(MachineOp.BGE, _reg("s2"), _reg("t2"), target=".Lp_fwht_next_len"),
        _i(MachineOp.LI, _reg("s3"), _imm(0), comment="j = 0"),
        L(".Lp_fwht_j"),
        _i(MachineOp.BGE, _reg("s3"), _reg("t5"), target=".Lp_fwht_next_block"),
        _i(MachineOp.ADD, _reg("s4"), _reg("s2"), _reg("s3"), comment="idx = i+j"),
        _i(MachineOp.SLLI, _reg("s5"), _reg("s4"), _imm(2)),
        _i(MachineOp.ADD, _reg("s6"), _reg("a1"), _reg("s5"), comment="&a[idx]"),
        _i(MachineOp.ADD, _reg("s7"), _reg("s4"), _reg("t5"), comment="idx+length"),
        _i(MachineOp.SLLI, _reg("s7"), _reg("s7"), _imm(2)),
        _i(MachineOp.ADD, _reg("s8"), _reg("a1"), _reg("s7"), comment="&a[idx+length]"),
        _i(MachineOp.FLW, _reg("ft0"), _reg("0(s6)"), comment="u"),
        _i(MachineOp.FLW, _reg("ft1"), _reg("0(s8)"), comment="v"),
        _i(MachineOp.FADD_S, _reg("ft2"), _reg("ft0"), _reg("ft1"), comment="u+v"),
        _i(MachineOp.FSUB_S, _reg("ft3"), _reg("ft0"), _reg("ft1"), comment="u-v"),
        _i(MachineOp.FSW, _reg("ft2"), _reg("0(s6)")),
        _i(MachineOp.FSW, _reg("ft3"), _reg("0(s8)")),
        _i(MachineOp.ADDI, _reg("s3"), _reg("s3"), _imm(1)),
        _i(MachineOp.J, target=".Lp_fwht_j"),
        L(".Lp_fwht_next_block"),
        _i(MachineOp.ADD, _reg("s2"), _reg("s2"), _reg("t6")),
        _i(MachineOp.J, target=".Lp_fwht_i"),
        L(".Lp_fwht_next_len"),
        _i(MachineOp.SLLI, _reg("t5"), _reg("t5"), _imm(1)),
        _i(MachineOp.J, target=".Lp_fwht_len"),
        L(".Lp_fwht_done"),
        _i(MachineOp.RET),
    ]
    return code


# ── CSR SpMM (a0 = self-describing block [M,K,N,nnz] + row/col/values/B) ────

def _lower_spmm() -> list[MachineInstr]:
    L = _label
    code = [
        L(_ENTRY),
        _i(MachineOp.LW, _reg("s0"), _reg("0(a0)"), comment="M"),
        _i(MachineOp.LW, _reg("s1"), _reg("8(a0)"), comment="N"),
        _i(MachineOp.LW, _reg("s11"), _reg("12(a0)"), comment="nnz"),
        _i(MachineOp.ADDI, _reg("s2"), _reg("a0"), _imm(16), comment="row_ptr"),
        _i(MachineOp.SLLI, _reg("t0"), _reg("s0"), _imm(2)),
        _i(MachineOp.ADDI, _reg("t0"), _reg("t0"), _imm(4), comment="(M+1)*4"),
        _i(MachineOp.ADD, _reg("s3"), _reg("s2"), _reg("t0"), comment="col_idx"),
        _i(MachineOp.SLLI, _reg("t0"), _reg("s11"), _imm(2)),
        _i(MachineOp.ADD, _reg("s4"), _reg("s3"), _reg("t0"), comment="values"),
        _i(MachineOp.ADD, _reg("s5"), _reg("s4"), _reg("t0"), comment="B"),
        _i(MachineOp.LI, _reg("s6"), _imm(0), comment="i = 0"),
        L(".Lp_spmm_i"),
        _i(MachineOp.BGE, _reg("s6"), _reg("s0"), target=".Lp_spmm_done"),
        _i(MachineOp.SLLI, _reg("t0"), _reg("s6"), _imm(2)),
        _i(MachineOp.ADD, _reg("t0"), _reg("s2"), _reg("t0")),
        _i(MachineOp.LW, _reg("s7"), _reg("0(t0)"), comment="lo = row_ptr[i]"),
        _i(MachineOp.LW, _reg("s8"), _reg("4(t0)"), comment="hi = row_ptr[i+1]"),
        _i(MachineOp.LI, _reg("s9"), _imm(0), comment="j = 0"),
        L(".Lp_spmm_j"),
        _i(MachineOp.BGE, _reg("s9"), _reg("s1"), target=".Lp_spmm_next_i"),
        _i(MachineOp.FMV_W_X, _reg("ft0"), _reg("zero"), comment="acc = 0.0"),
        _i(MachineOp.MV, _reg("s10"), _reg("s7"), comment="p = lo"),
        L(".Lp_spmm_p"),
        _i(MachineOp.BGE, _reg("s10"), _reg("s8"), target=".Lp_spmm_store"),
        _i(MachineOp.SLLI, _reg("t1"), _reg("s10"), _imm(2)),
        _i(MachineOp.ADD, _reg("t1"), _reg("s4"), _reg("t1")),
        _i(MachineOp.FLW, _reg("ft1"), _reg("0(t1)"), comment="values[p]"),
        _i(MachineOp.SLLI, _reg("t2"), _reg("s10"), _imm(2)),
        _i(MachineOp.ADD, _reg("t2"), _reg("s3"), _reg("t2")),
        _i(MachineOp.LW, _reg("t2"), _reg("0(t2)"), comment="k = col_idx[p]"),
        _i(MachineOp.MUL, _reg("t3"), _reg("t2"), _reg("s1")),
        _i(MachineOp.ADD, _reg("t3"), _reg("t3"), _reg("s9")),
        _i(MachineOp.SLLI, _reg("t3"), _reg("t3"), _imm(2)),
        _i(MachineOp.ADD, _reg("t3"), _reg("s5"), _reg("t3")),
        _i(MachineOp.FLW, _reg("ft2"), _reg("0(t3)"), comment="B[k*N+j]"),
        _i(MachineOp.FMUL_S, _reg("ft1"), _reg("ft1"), _reg("ft2")),
        _i(MachineOp.FADD_S, _reg("ft0"), _reg("ft0"), _reg("ft1"), comment="acc += term"),
        _i(MachineOp.ADDI, _reg("s10"), _reg("s10"), _imm(1)),
        _i(MachineOp.J, target=".Lp_spmm_p"),
        L(".Lp_spmm_store"),
        _i(MachineOp.MUL, _reg("t3"), _reg("s6"), _reg("s1")),
        _i(MachineOp.ADD, _reg("t3"), _reg("t3"), _reg("s9")),
        _i(MachineOp.SLLI, _reg("t3"), _reg("t3"), _imm(2)),
        _i(MachineOp.ADD, _reg("t3"), _reg("a1"), _reg("t3")),
        _i(MachineOp.FSW, _reg("ft0"), _reg("0(t3)")),
        _i(MachineOp.ADDI, _reg("s9"), _reg("s9"), _imm(1)),
        _i(MachineOp.J, target=".Lp_spmm_j"),
        L(".Lp_spmm_next_i"),
        _i(MachineOp.ADDI, _reg("s6"), _reg("s6"), _imm(1)),
        _i(MachineOp.J, target=".Lp_spmm_i"),
        L(".Lp_spmm_done"),
        _i(MachineOp.RET),
    ]
    return code


# ── Direct Conv2D / Winograd (a0 = [batch,H,W,Cin,Cout,K] + NHWC + OIHW) ────

def _lower_conv() -> list[MachineInstr]:
    L = _label
    code = [
        L(_ENTRY),
        _i(MachineOp.LW, _reg("s0"), _reg("0(a0)"), comment="batch"),
        _i(MachineOp.LW, _reg("s1"), _reg("4(a0)"), comment="H"),
        _i(MachineOp.LW, _reg("s2"), _reg("8(a0)"), comment="W"),
        _i(MachineOp.LW, _reg("s3"), _reg("12(a0)"), comment="Cin"),
        _i(MachineOp.LW, _reg("s4"), _reg("16(a0)"), comment="Cout"),
        _i(MachineOp.LW, _reg("s5"), _reg("20(a0)"), comment="K"),
        _i(MachineOp.SRLI, _reg("s6"), _reg("s5"), _imm(1), comment="pad = K/2"),
        _i(MachineOp.ADDI, _reg("s7"), _reg("a0"), _imm(24), comment="feat base"),
        _i(MachineOp.MUL, _reg("t0"), _reg("s0"), _reg("s1")),
        _i(MachineOp.MUL, _reg("t0"), _reg("t0"), _reg("s2")),
        _i(MachineOp.MUL, _reg("t0"), _reg("t0"), _reg("s3"), comment="batch*H*W*Cin"),
        _i(MachineOp.SLLI, _reg("t0"), _reg("t0"), _imm(2)),
        _i(MachineOp.ADD, _reg("s8"), _reg("s7"), _reg("t0"), comment="wt base"),
        _i(MachineOp.LI, _reg("s9"), _imm(0), comment="b = 0"),
        L(".Lp_conv_b"),
        _i(MachineOp.BGE, _reg("s9"), _reg("s0"), target=".Lp_conv_done"),
        _i(MachineOp.LI, _reg("s10"), _imm(0), comment="oc = 0"),
        L(".Lp_conv_oc"),
        _i(MachineOp.BGE, _reg("s10"), _reg("s4"), target=".Lp_conv_next_b"),
        _i(MachineOp.LI, _reg("s11"), _imm(0), comment="oh = 0"),
        L(".Lp_conv_oh"),
        _i(MachineOp.BGE, _reg("s11"), _reg("s1"), target=".Lp_conv_next_oc"),
        _i(MachineOp.LI, _reg("t0"), _imm(0), comment="ow = 0"),
        L(".Lp_conv_ow"),
        _i(MachineOp.BGE, _reg("t0"), _reg("s2"), target=".Lp_conv_next_oh"),
        _i(MachineOp.FMV_W_X, _reg("ft0"), _reg("zero"), comment="acc = 0.0"),
        _i(MachineOp.LI, _reg("t2"), _imm(0), comment="c = 0"),
        L(".Lp_conv_c"),
        _i(MachineOp.BGE, _reg("t2"), _reg("s3"), target=".Lp_conv_store"),
        _i(MachineOp.LI, _reg("t3"), _imm(0), comment="kh = 0"),
        L(".Lp_conv_kh"),
        _i(MachineOp.BGE, _reg("t3"), _reg("s5"), target=".Lp_conv_next_c"),
        _i(MachineOp.ADD, _reg("t5"), _reg("s11"), _reg("t3")),
        _i(MachineOp.SUB, _reg("t5"), _reg("t5"), _reg("s6"), comment="ih = oh-pad+kh"),
        _i(MachineOp.BLT, _reg("t5"), _reg("zero"), target=".Lp_conv_next_kh"),
        _i(MachineOp.BGE, _reg("t5"), _reg("s1"), target=".Lp_conv_next_kh"),
        _i(MachineOp.LI, _reg("t4"), _imm(0), comment="kw = 0"),
        L(".Lp_conv_kw"),
        _i(MachineOp.BGE, _reg("t4"), _reg("s5"), target=".Lp_conv_next_kh"),
        _i(MachineOp.ADD, _reg("t6"), _reg("t0"), _reg("t4")),
        _i(MachineOp.SUB, _reg("t6"), _reg("t6"), _reg("s6"), comment="iw = ow-pad+kw"),
        _i(MachineOp.BLT, _reg("t6"), _reg("zero"), target=".Lp_conv_next_kw"),
        _i(MachineOp.BGE, _reg("t6"), _reg("s2"), target=".Lp_conv_next_kw"),
        _i(MachineOp.MUL, _reg("a3"), _reg("s9"), _reg("s1")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("t5")),
        _i(MachineOp.MUL, _reg("a3"), _reg("a3"), _reg("s2")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("t6")),
        _i(MachineOp.MUL, _reg("a3"), _reg("a3"), _reg("s3")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("t2")),
        _i(MachineOp.SLLI, _reg("a3"), _reg("a3"), _imm(2)),
        _i(MachineOp.ADD, _reg("a3"), _reg("s7"), _reg("a3")),
        _i(MachineOp.FLW, _reg("ft1"), _reg("0(a3)"), comment="x"),
        _i(MachineOp.MUL, _reg("a4"), _reg("s10"), _reg("s3")),
        _i(MachineOp.ADD, _reg("a4"), _reg("a4"), _reg("t2")),
        _i(MachineOp.MUL, _reg("a4"), _reg("a4"), _reg("s5")),
        _i(MachineOp.ADD, _reg("a4"), _reg("a4"), _reg("t3")),
        _i(MachineOp.MUL, _reg("a4"), _reg("a4"), _reg("s5")),
        _i(MachineOp.ADD, _reg("a4"), _reg("a4"), _reg("t4")),
        _i(MachineOp.SLLI, _reg("a4"), _reg("a4"), _imm(2)),
        _i(MachineOp.ADD, _reg("a4"), _reg("s8"), _reg("a4")),
        _i(MachineOp.FLW, _reg("ft2"), _reg("0(a4)"), comment="w"),
        _i(MachineOp.FMUL_S, _reg("ft3"), _reg("ft1"), _reg("ft2")),
        _i(MachineOp.FADD_S, _reg("ft0"), _reg("ft0"), _reg("ft3"), comment="acc += x*w"),
        L(".Lp_conv_next_kw"),
        _i(MachineOp.ADDI, _reg("t4"), _reg("t4"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_kw"),
        L(".Lp_conv_next_kh"),
        _i(MachineOp.ADDI, _reg("t3"), _reg("t3"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_kh"),
        L(".Lp_conv_next_c"),
        _i(MachineOp.ADDI, _reg("t2"), _reg("t2"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_c"),
        L(".Lp_conv_store"),
        _i(MachineOp.MUL, _reg("a3"), _reg("s9"), _reg("s4")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("s10")),
        _i(MachineOp.MUL, _reg("a3"), _reg("a3"), _reg("s1")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("s11")),
        _i(MachineOp.MUL, _reg("a3"), _reg("a3"), _reg("s2")),
        _i(MachineOp.ADD, _reg("a3"), _reg("a3"), _reg("t0")),
        _i(MachineOp.SLLI, _reg("a3"), _reg("a3"), _imm(2)),
        _i(MachineOp.ADD, _reg("a3"), _reg("a1"), _reg("a3")),
        _i(MachineOp.FSW, _reg("ft0"), _reg("0(a3)")),
        _i(MachineOp.ADDI, _reg("t0"), _reg("t0"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_ow"),
        L(".Lp_conv_next_oh"),
        _i(MachineOp.ADDI, _reg("s11"), _reg("s11"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_oh"),
        L(".Lp_conv_next_oc"),
        _i(MachineOp.ADDI, _reg("s10"), _reg("s10"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_oc"),
        L(".Lp_conv_next_b"),
        _i(MachineOp.ADDI, _reg("s9"), _reg("s9"), _imm(1)),
        _i(MachineOp.J, target=".Lp_conv_b"),
        L(".Lp_conv_done"),
        _i(MachineOp.RET),
    ]
    return code


_LOWERERS = {
    OpCode.FWHT: _lower_fwht,
    OpCode.SPMM_CSR: _lower_spmm,
    OpCode.CONV: _lower_conv,
    OpCode.WINOGRAD_CONV: _lower_conv,
}


def render_platform(instructions: list[MachineInstr]) -> str:
    """Render machine instructions as a platform listing."""
    text = AsmEmitter(instructions).emit()
    if ".size cnn_entry" not in text:
        text = text.rstrip("\n") + f"\n  .size {_ENTRY}, .-{_ENTRY}\n"
    return ".option norelax\n" + text


def generate_platform_asm(program: Program) -> str:
    """Emit the FP32/rv32imf platform listing for a single-operator program."""
    operators = [
        instruction.opcode
        for function in program.functions
        for block in function.blocks
        for instruction in block.instructions
        if instruction.opcode in _LOWERERS
    ]
    if len(operators) != 1:
        raise PlatformEmitError(
            "platform emission requires exactly one Fwht/Conv/WinogradConv/SpmmCsr "
            f"operator, found {len(operators)}"
        )
    return render_platform(_LOWERERS[operators[0]]())
