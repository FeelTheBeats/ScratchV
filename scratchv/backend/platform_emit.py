"""Platform (``.s``) emission for the main pipeline (Phase 3, route B).

Rather than lowering IR through the fragile scalar linear-allocator / ABI-frame
path, this reuses the size-independent, 30/30-verified standalone platform
kernels. Supported graphs are exactly one size-independent operator
(Fwht / Conv|WinogradConv / CSR Spmm); anything else is rejected explicitly.
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Program
from scratchv.standalone.onnx_to_riscv_standalone import (
    _emit_platform_conv,
    _emit_platform_fwht,
    _emit_platform_spmm,
)


class PlatformEmitError(ValueError):
    """A graph cannot be emitted as a single size-independent platform kernel."""


_PLATFORM_KERNELS = {
    OpCode.FWHT: _emit_platform_fwht,
    OpCode.CONV: _emit_platform_conv,
    OpCode.WINOGRAD_CONV: _emit_platform_conv,
    OpCode.SPMM_CSR: _emit_platform_spmm,
}


def generate_platform_asm(program: Program) -> str:
    """Emit the FP32/rv32imf platform listing for a single-operator program."""
    operators = [
        instruction.opcode
        for function in program.functions
        for block in function.blocks
        for instruction in block.instructions
        if instruction.opcode in _PLATFORM_KERNELS
    ]
    if len(operators) != 1:
        raise PlatformEmitError(
            "platform emission requires exactly one Fwht/Conv/WinogradConv/SpmmCsr "
            f"operator, found {len(operators)}"
        )
    return _PLATFORM_KERNELS[operators[0]]()
