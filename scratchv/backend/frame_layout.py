"""Function-level stack frame layout and callee-saved handling (topic 17 W9).

Splits a machine instruction program into functions and basic blocks, runs
the linear-scan allocator per block with cross-block values forced to memory,
then computes a 16-byte aligned frame and inserts ``ra``/``s-reg``
save/restore code around the function body.

Frame layout (low to high addresses)::

    sp + 0                    spill area (block 0, block 1, ...)
    sp + spill_bytes          saved s-regs (in CALLEE_SAVED order)
    sp + frame_size - 4       saved ra (when the function contains a call)

Usage::

    from scratchv.backend.frame_layout import FunctionFrameAllocator
    allocated = FunctionFrameAllocator().allocate_program(machine_instrs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from scratchv.backend.machine_types import (
    CALLEE_SAVED,
    REG_NUMS,
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.regalloc_linear import LinearScanAllocator, RegAllocError

_MEM_OPERAND_RE = re.compile(r"^(-?\d+)\((\w+)\)$")

# Comments the linear-scan allocator attaches to inserted spill/reload code.
_SPILL_COMMENT_MARKERS = ("reload ", "evict ", "store redefined ")


def align16(size: int) -> int:
    """Round *size* up to a multiple of 16 bytes."""
    return ((size + 15) // 16) * 16


@dataclass
class FrameInfo:
    """Layout of one function's stack frame."""

    frame_size: int
    ra_offset: Optional[int]
    saved_offsets: dict[str, int] = field(default_factory=dict)
    spill_base: int = 0


@dataclass
class FunctionChunk:
    """A function split out of the flat program stream."""

    name: Optional[str]
    label: Optional[MachineInstr]
    instrs: list[MachineInstr] = field(default_factory=list)


def _is_function_label(instr: MachineInstr) -> bool:
    return (
        instr.op == MachineOp.LABEL
        and bool(instr.comment)
        and not instr.comment.startswith(".")
    )


def _is_block_label(instr: MachineInstr) -> bool:
    return (
        instr.op == MachineOp.LABEL
        and bool(instr.comment)
        and instr.comment.startswith(".")
    )


def _operands_of(instr: MachineInstr):
    return (instr.dst, instr.src1, instr.src2)


def _vregs_of(instr: MachineInstr) -> set[str]:
    out: set[str] = set()
    for op in _operands_of(instr):
        if op is not None and op.kind == "vreg":
            name = str(op.value)
            if name not in REG_NUMS:  # physical names never allocate
                out.add(name)
    return out


def split_functions(
        instrs: list[MachineInstr],
) -> tuple[list[MachineInstr], list[FunctionChunk]]:
    """Split a program into preamble instructions and function chunks.

    A ``LABEL`` whose name does not start with ``.`` starts a new function.
    Instructions before the first function label form the preamble (returned
    unchanged by the frame allocator).
    """
    preamble: list[MachineInstr] = []
    functions: list[FunctionChunk] = []
    current: Optional[FunctionChunk] = None

    for instr in instrs:
        if _is_function_label(instr):
            current = FunctionChunk(
                name=instr.comment, label=instr, instrs=[])
            functions.append(current)
        elif current is None:
            preamble.append(instr)
        else:
            current.instrs.append(instr)

    return preamble, functions


def split_blocks(instrs: list[MachineInstr]) -> list[list[MachineInstr]]:
    """Split function body instructions into basic blocks at ``.`` labels."""
    blocks: list[list[MachineInstr]] = []
    current: list[MachineInstr] = []

    for instr in instrs:
        if _is_block_label(instr):
            if current:
                blocks.append(current)
            current = [instr]
        else:
            current.append(instr)
    if current:
        blocks.append(current)
    return blocks


def cross_block_vregs(blocks: list[list[MachineInstr]]) -> set[str]:
    """Vregs that appear in more than one basic block (forced to memory)."""
    counts: dict[str, int] = {}
    for block in blocks:
        seen = set()
        for instr in block:
            seen |= _vregs_of(instr)
        for v in seen:
            counts[v] = counts.get(v, 0) + 1
    return {v for v, c in counts.items() if c >= 2}


def _is_ret(instr: MachineInstr) -> bool:
    if instr.op != MachineOp.JALR:
        return False
    for op in _operands_of(instr):
        if op is not None and op.kind == "reg" and str(op.value) == "ra":
            return True
    return False


def _has_call(instrs: list[MachineInstr]) -> bool:
    for instr in instrs:
        if instr.op == MachineOp.CALL:
            return True
        if (instr.op == MachineOp.JAL and instr.comment
                and not instr.comment.startswith(".")):
            return True
    return False


def _sw(reg: str, offset: int, comment: str = "") -> MachineInstr:
    return MachineInstr(
        MachineOp.SW, MachineOperand.reg(reg),
        MachineOperand.mem(offset), comment=comment,
    )


def _lw(reg: str, offset: int, comment: str = "") -> MachineInstr:
    return MachineInstr(
        MachineOp.LW, MachineOperand.reg(reg),
        MachineOperand.mem(offset), comment=comment,
    )


def emit_prologue(info: FrameInfo) -> list[MachineInstr]:
    """``addi sp`` + ``sw ra`` / ``sw s-reg`` for a non-empty frame."""
    if info.frame_size == 0:
        return []
    out = [MachineInstr(
        MachineOp.ADDI, MachineOperand.reg("sp"),
        MachineOperand.reg("sp"),
        MachineOperand.immediate(-info.frame_size),
    )]
    if info.ra_offset is not None:
        out.append(_sw("ra", info.ra_offset, comment="save ra"))
    for reg in CALLEE_SAVED:
        if reg in info.saved_offsets:
            out.append(_sw(reg, info.saved_offsets[reg],
                           comment=f"save {reg}"))
    return out


def emit_epilogue(info: FrameInfo) -> list[MachineInstr]:
    """Inverse of :func:`emit_prologue` (inserted before every ret)."""
    if info.frame_size == 0:
        return []
    out: list[MachineInstr] = []
    for reg in reversed(CALLEE_SAVED):
        if reg in info.saved_offsets:
            out.append(_lw(reg, info.saved_offsets[reg],
                           comment=f"restore {reg}"))
    if info.ra_offset is not None:
        out.append(_lw("ra", info.ra_offset, comment="restore ra"))
    out.append(MachineInstr(
        MachineOp.ADDI, MachineOperand.reg("sp"),
        MachineOperand.reg("sp"),
        MachineOperand.immediate(info.frame_size),
    ))
    return out


class FunctionFrameAllocator:
    """Allocate registers per block and lay out a frame per function.

    Parameters
    ----------
    alloc_factory:
        Optional factory producing a ``LinearScanAllocator`` for
        ``stack_base=<int>`` and ``pre_spilled=<collection>``.  Used by
        tests and pressure experiments to constrain the register pool.
    cross_block_policy:
        Only ``"force-spill"`` is supported (values live across blocks are
        kept in their spill slots).
    """

    def __init__(
        self,
        alloc_factory: Optional[Callable[..., LinearScanAllocator]] = None,
        cross_block_policy: str = "force-spill",
    ) -> None:
        if cross_block_policy != "force-spill":
            raise ValueError(
                f"unsupported cross_block_policy: {cross_block_policy!r}")
        self._alloc_factory = alloc_factory
        self.cross_block_policy = cross_block_policy
        self.last_frame_info: dict[str, FrameInfo] = {}

    def _make_allocator(
        self, stack_base: int, pre_spilled: set[str],
        slot_hints: dict[str, int],
    ) -> LinearScanAllocator:
        if self._alloc_factory is None:
            return LinearScanAllocator(
                stack_base=stack_base, pre_spilled=pre_spilled,
                slot_hints=slot_hints)
        return self._alloc_factory(
            stack_base=stack_base, pre_spilled=pre_spilled,
            slot_hints=slot_hints)

    def allocate_program(
        self, instrs: list[MachineInstr],
    ) -> list[MachineInstr]:
        """Allocate a whole program and insert prologue/epilogue code."""
        preamble, functions = split_functions(instrs)
        out: list[MachineInstr] = list(preamble)
        for func in functions:
            out.extend(self._allocate_function(func))
        return out

    def _allocate_function(
        self, func: FunctionChunk,
    ) -> list[MachineInstr]:
        blocks = split_blocks(func.instrs)
        cross = cross_block_vregs(blocks)

        # Function-wide slots for cross-block values: every block must agree
        # on where a forced-spilled vreg lives.
        global_slots = {v: 4 * i for i, v in enumerate(sorted(cross))}
        global_bytes = 4 * len(global_slots)

        # Emit every block once.  A block's spill area starts after the
        # areas actually consumed by the previous blocks (including the
        # slots that reload-time eviction adds during emission), so the
        # frame size below is the upper bound of the emitted code instead of
        # a pass-1 estimate that pass 2 can outgrow (F1).
        block_outputs: list[list[MachineInstr]] = []
        total_local = 0
        for block in blocks:
            base = global_bytes + total_local
            local = cross & {v for i in block for v in _vregs_of(i)}
            alloc = self._make_allocator(
                stack_base=base, pre_spilled=local,
                slot_hints=global_slots)
            block_out = alloc.emit_machine_instrs(block)
            block_outputs.append(block_out)
            local_slots = {
                slot for v, slot in alloc.spill_slots.items()
                if v not in global_slots
            }
            expected = {base + 4 * i for i in range(len(local_slots))}
            if local_slots != expected:
                raise RegAllocError(
                    f"frame layout for {func.name!r}: block spill slots "
                    f"{sorted(local_slots)} do not tile its region "
                    f"[{base}, {base + 4 * len(local_slots)})")
            total_local += 4 * len(local_slots)

        spill_bytes = global_bytes + total_local
        # The saved set is collected from the emitted instructions (not from
        # the static allocation map): reload targets and scratch registers
        # are picked dynamically during emission and may land on a
        # callee-saved register that pass 1 never assigned (F3).
        used_callee = self._callee_saved_regs(block_outputs)
        has_call = _has_call(func.instrs)
        saved = ["ra"] if has_call else []
        saved += [r for r in CALLEE_SAVED if r in used_callee]
        frame_size = align16(spill_bytes + 4 * len(saved))

        save_area_start = frame_size - 4 * len(saved)
        if spill_bytes > save_area_start:
            raise RegAllocError(
                f"frame layout for {func.name!r}: spill area "
                f"[0, {spill_bytes}) overlaps save area "
                f"[{save_area_start}, {frame_size})")
        self._check_spill_bounds(block_outputs, spill_bytes)

        info = FrameInfo(
            frame_size=frame_size,
            ra_offset=(frame_size - 4) if has_call else None,
            saved_offsets={
                r: frame_size - 4 * (i + 1)
                for i, r in enumerate(saved) if r != "ra"
            },
        )
        if func.name:
            self.last_frame_info[func.name] = info

        if func.label is not None:
            out: list[MachineInstr] = [func.label]
        else:
            out = []
        out.extend(emit_prologue(info))

        for block_out in block_outputs:
            for instr in block_out:
                if _is_ret(instr):
                    out.extend(emit_epilogue(info))
                out.append(instr)
        return out

    @staticmethod
    def _callee_saved_regs(
        block_outputs: list[list[MachineInstr]],
    ) -> set[str]:
        """Callee-saved registers mentioned by the emitted function body.

        Conservative (a register only read is saved too): the prologue store
        is harmless in that case, while missing one for a dynamically chosen
        reload/scratch target corrupts the caller's state.
        """
        used: set[str] = set()
        for block_out in block_outputs:
            for instr in block_out:
                for op in _operands_of(instr):
                    if (op is not None and op.kind == "reg"
                            and str(op.value) in CALLEE_SAVED):
                        used.add(str(op.value))
        return used

    @staticmethod
    def _check_spill_bounds(
        block_outputs: list[list[MachineInstr]],
        spill_bytes: int,
    ) -> None:
        """Every allocator-inserted spill/reload access must stay in the
        spill area ``[0, spill_bytes)`` (fail loudly on layout drift)."""
        for block_out in block_outputs:
            for instr in block_out:
                if not instr.comment.startswith(_SPILL_COMMENT_MARKERS):
                    continue
                for op in _operands_of(instr):
                    if op is None or op.kind != "mem":
                        continue
                    match = _MEM_OPERAND_RE.match(str(op.value))
                    if match is None or match.group(2) != "sp":
                        continue
                    offset = int(match.group(1))
                    if offset < 0 or offset + 4 > spill_bytes:
                        raise RegAllocError(
                            f"spill access {op.value} outside spill area "
                            f"[0, {spill_bytes}) in {instr!r}")
