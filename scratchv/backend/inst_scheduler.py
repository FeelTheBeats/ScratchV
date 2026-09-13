"""Instruction Scheduler for RISC-V (intra-block list scheduling).

Reorders RISC-V instructions inside a single basic block to hide
pipeline stalls, without adding, removing, or replacing any
instruction.  The scheduler is self-contained: it only relies on
``_asm_parser`` for line structure and integer-register
canonicalisation.

Public entry point for compiler integration::

    from scratchv.backend.inst_scheduler import schedule_asm

    result = schedule_asm(asm_text)
    asm_text = result.asm_text

The legacy API (``InstructionScheduler``, ``parse_instructions`` and
``machine_instrs_from_scheduled``) is preserved for existing callers.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from ._asm_parser import (
    ParsedAsmLine, canonical_reg, is_integer_reg, parse_asm,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Opcode scheduling semantics
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeKind(Enum):
    """Dependency edge kinds."""

    RAW = "raw"
    WAR = "war"
    WAW = "waw"
    MEMORY = "memory"
    CONTROL = "control"


@dataclass(frozen=True)
class OpcodeInfo:
    """Scheduling semantics for one opcode.

    Attributes:
        exec_latency: Cycles the instruction occupies the EX stage.
        resource: Execution resource (alu/mul/div/lsu/fpu/branch).
        memory: ``none``, ``load`` or ``store``.
        pipelined: ``False`` for non-pipelined units (div/rem/fsqrt).
        is_barrier: Anchored instruction that splits scheduling regions.
        is_terminator: Pinned to the end of its basic block.
    """

    exec_latency: int = 1
    resource: str = "alu"
    memory: str = "none"
    pipelined: bool = True
    is_barrier: bool = False
    is_terminator: bool = False

    @property
    def result_latency(self) -> int:
        """Cycles before the result can be read (loads need one extra)."""
        return self.exec_latency + (1 if self.memory == "load" else 0)


_LOADS = frozenset({"lw", "lh", "lb", "lbu", "lhu", "flw", "fld"})
_STORES = frozenset({"sw", "sh", "sb", "fsw", "fsd"})
_MULS = frozenset({"mul", "mulh", "mulhsu", "mulhu"})
_DIVS = frozenset({"div", "divu", "rem", "remu"})
_BRANCHES = frozenset(
    {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}
)
_JUMPS = frozenset({"j", "jal", "jalr", "ret", "jr"})
_BARRIER_OPS = frozenset({"call", "tail"})
_ALU_OPS = frozenset({
    "add", "addi", "sub", "sll", "srl", "sra", "xor", "or", "and",
    "xori", "ori", "andi", "slli", "srli", "srai",
    "slt", "sltu", "slti", "sltiu", "lui", "auipc",
    "li", "mv", "max", "nop",
})
_FP_MISC = frozenset({
    "fmv.s", "fmv.s.x", "fcvt.s.d", "fcvt.d.s",
    "fabs.d", "fneg.d", "li.d",
})
_FP_PIPELINED: dict[str, int] = {
    "fadd.s": 3, "fsub.s": 3, "fmul.s": 4, "fmax.s": 2, "fmin.s": 2,
    "fle.s": 2, "flt.s": 2, "feq.s": 2,
    "fadd.d": 4, "fsub.d": 4, "fmul.d": 5, "fmax.d": 3, "fmin.d": 3,
    "flt.d": 3, "feq.d": 3,
}
_FP_NON_PIPELINED: dict[str, int] = {
    "fdiv.s": 12, "fsqrt.s": 12, "fdiv.d": 16, "fsqrt.d": 16,
}


def _build_opcode_table() -> dict[str, OpcodeInfo]:
    table: dict[str, OpcodeInfo] = {}
    for opcode in _ALU_OPS:
        table[opcode] = OpcodeInfo()
    for opcode in _LOADS:
        table[opcode] = OpcodeInfo(resource="lsu", memory="load")
    for opcode in _STORES:
        table[opcode] = OpcodeInfo(resource="lsu", memory="store")
    for opcode in _FP_MISC:
        table[opcode] = OpcodeInfo(resource="fpu")
    for opcode in _MULS:
        table[opcode] = OpcodeInfo(exec_latency=3, resource="mul")
    for opcode in _DIVS:
        table[opcode] = OpcodeInfo(
            exec_latency=16, resource="div", pipelined=False)
    for opcode in _BRANCHES | _JUMPS:
        table[opcode] = OpcodeInfo(resource="branch", is_terminator=True)
    for opcode, latency in _FP_PIPELINED.items():
        table[opcode] = OpcodeInfo(exec_latency=latency, resource="fpu")
    for opcode, latency in _FP_NON_PIPELINED.items():
        table[opcode] = OpcodeInfo(
            exec_latency=latency, resource="fpu", pipelined=False)
    for opcode in _BARRIER_OPS:
        table[opcode] = OpcodeInfo(is_barrier=True)
    return table


_OPCODE_INFO: dict[str, OpcodeInfo] = _build_opcode_table()
_BARRIER_INFO = OpcodeInfo(is_barrier=True)


def _info(opcode: str) -> OpcodeInfo:
    """Return scheduling semantics; unknown opcodes are barriers."""
    return _OPCODE_INFO.get(opcode, _BARRIER_INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# Register canonicalisation and operand classification
# ═══════════════════════════════════════════════════════════════════════════════

# psABI float-register aliases (``_asm_parser`` only knows integer ones).
_FLOAT_REG_ALIASES: dict[str, str] = {
    "ft0": "f0", "ft1": "f1", "ft2": "f2", "ft3": "f3",
    "ft4": "f4", "ft5": "f5", "ft6": "f6", "ft7": "f7",
    "fs0": "f8", "fs1": "f9",
    "fa0": "f10", "fa1": "f11", "fa2": "f12", "fa3": "f13",
    "fa4": "f14", "fa5": "f15", "fa6": "f16", "fa7": "f17",
    "fs2": "f18", "fs3": "f19", "fs4": "f20", "fs5": "f21",
    "fs6": "f22", "fs7": "f23", "fs8": "f24", "fs9": "f25",
    "fs10": "f26", "fs11": "f27",
    "ft8": "f28", "ft9": "f29", "ft10": "f30", "ft11": "f31",
}
_FLOAT_REG_RE = re.compile(r"f(?:[0-9]|[12][0-9]|3[01])$")
_MEM_OPERAND_RE = re.compile(r"\((\w+)\)\s*$")


def _canon_reg(name: str) -> Optional[str]:
    """Canonicalise an integer (``xN``) or float (``fN``) register name."""
    text = (name or "").strip().lower()
    if not text:
        return None
    if is_integer_reg(text):
        return canonical_reg(text)
    if _FLOAT_REG_RE.match(text):
        return text
    return _FLOAT_REG_ALIASES.get(text)


def _base_reg(operand: str) -> Optional[str]:
    """Extract the base register of ``offset(base)``; None if absent."""
    match = _MEM_OPERAND_RE.search(operand)
    if match is None:
        return None
    return _canon_reg(match.group(1))


def _is_int_literal(text: str) -> bool:
    try:
        int(text.strip(), 0)
    except ValueError:
        return False
    return True


def _classify_terminator(
    opcode: str, operands: list[str]
) -> tuple[set[str], set[str], bool]:
    """Classify a branch/jump/return instruction."""
    defines: set[str] = set()
    uses: set[str] = set()

    def add_use(text: str) -> bool:
        reg = _canon_reg(text)
        if reg is None:
            return False
        if reg != "x0":
            uses.add(reg)
        return True

    if opcode == "ret":
        uses.add("x1")
        return defines, uses, True

    if opcode == "jr":
        if not operands or not add_use(operands[0]):
            return defines, uses, False
        return defines, uses, True

    if opcode == "jalr":
        if len(operands) == 1:          # jalr rs1 -> implicit ra
            if not add_use(operands[0]):
                return defines, uses, False
            defines.add("x1")
            return defines, uses, True
        if len(operands) < 2:
            return defines, uses, False
        dst = _canon_reg(operands[0])
        base = _base_reg(operands[1])
        if base is None:
            base = _canon_reg(operands[1])
        if dst is None or base is None:
            return defines, uses, False
        if dst != "x0":
            defines.add(dst)
        if base != "x0":
            uses.add(base)
        return defines, uses, True

    if opcode == "jal":
        if not operands:                # jal label -> implicit ra
            defines.add("x1")
            return defines, uses, True
        dst = _canon_reg(operands[0])
        if dst is None:
            defines.add("x1")
        elif dst != "x0":
            defines.add(dst)
        return defines, uses, True

    if opcode in _BRANCHES:
        for operand in operands:
            reg = _canon_reg(operand)   # label targets yield None
            if reg is not None and reg != "x0":
                uses.add(reg)
        return defines, uses, True

    # "j": no register operands
    return defines, uses, True


def classify_operands(
    opcode: str, operands: list[str], info: OpcodeInfo
) -> tuple[set[str], set[str], bool]:
    """Classify one instruction's operands into defines and uses.

    Returns ``(defines, uses, ok)``.  ``ok=False`` means the instruction
    cannot be classified safely and must be treated as a barrier.
    """
    defines: set[str] = set()
    uses: set[str] = set()

    if info.is_barrier:
        return defines, uses, True

    if info.memory == "store":
        if len(operands) < 2:
            return defines, uses, False
        value = _canon_reg(operands[0])
        base = _base_reg(operands[1])
        if value is None or base is None:
            return defines, uses, False
        if value != "x0":
            uses.add(value)
        if base != "x0":
            uses.add(base)
        return defines, uses, True

    if info.memory == "load":
        if len(operands) < 2:
            return defines, uses, False
        dst = _canon_reg(operands[0])
        base = _base_reg(operands[1])
        if dst is None or base is None:
            return defines, uses, False
        if dst != "x0":
            defines.add(dst)
        if base != "x0":
            uses.add(base)
        return defines, uses, True

    if info.is_terminator:
        return _classify_terminator(opcode, operands)

    if opcode == "nop":
        return defines, uses, True

    if not operands:
        return defines, uses, False
    dst = _canon_reg(operands[0])
    if dst is None:
        return defines, uses, False
    if dst != "x0":
        defines.add(dst)
    for operand in operands[1:]:
        reg = _canon_reg(operand)
        if reg is not None:
            if reg != "x0":
                uses.add(reg)
        elif not _is_int_literal(operand):
            return defines, uses, False  # symbolic operand: stay safe
    return defines, uses, True


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchedInst:
    """An instruction node for the scheduler.

    Attributes:
        id: Unique index in the input list.
        opcode: Instruction mnemonic.
        operands: List of operand strings.
        defines: Registers written (canonical names).
        uses: Registers read (canonical names).
        raw_line: Original assembly text line.
        info: Scheduling semantics (inferred from opcode when None).
        parsed: Parsed assembly line (for lossless rewrite).
        pinned: Terminator pinned to the end of its block.
    """

    id: int
    opcode: str
    operands: list[str] = field(default_factory=list)
    defines: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    raw_line: str = ""
    info: Optional[OpcodeInfo] = None
    parsed: Optional[ParsedAsmLine] = None
    pinned: bool = False

    def __repr__(self) -> str:
        return (f"SchedInst({self.id}, {self.opcode}, "
                f"def={self.defines}, use={self.uses})")


@dataclass(frozen=True)
class SchedEdge:
    """A dependency edge, kept separately for verification."""

    pred: int
    succ: int
    kinds: frozenset[EdgeKind]
    min_separation: int


@dataclass
class DAGNode:
    """A node in the instruction dependency DAG.

    ``predecessors``/``successors`` keep the legacy ``(node, latency)``
    tuple shape; ``latency`` is the edge's minimum separation.
    """

    inst: SchedInst
    predecessors: list[tuple["DAGNode", int]] = field(default_factory=list)
    successors: list[tuple["DAGNode", int]] = field(default_factory=list)
    scheduled: bool = False
    ready_time: int = 0
    priority: int = 0
    unscheduled_preds: int = 0
    earliest_issue: int = 0
    issue_cycle: Optional[int] = None

    def __repr__(self) -> str:
        return (f"DAGNode(id={self.inst.id}, {self.inst.opcode}, "
                f"prio={self.priority}, pred={len(self.predecessors)})")


@dataclass
class Block:
    """A schedulable basic block: movable instructions (+ terminator)."""

    id: str
    insts: list[SchedInst]
    start_idx: int = 0


@dataclass
class BlockStats:
    """Per-block scheduling statistics."""

    block_id: str
    n_inst: int
    n_moved: int
    orig_cycles: int
    sched_cycles: int
    orig_stalls: int
    sched_stalls: int
    status: str


@dataclass
class ScheduleConfig:
    """Scheduling configuration.

    Attributes:
        issue_width: Instructions issued per cycle (V1 fixed to 1).
        max_block_size: Blocks larger than this keep their order.
        strict: Reserved for the standalone CLI exit code.
    """

    issue_width: int = 1
    max_block_size: int = 1024
    strict: bool = False


@dataclass
class ScheduleResult:
    """Result of scheduling an assembly text."""

    asm_text: str
    blocks_seen: int
    blocks_applied: int
    blocks_fallback: int
    warnings: list[str]
    stats: dict


class DependencyCycleError(RuntimeError):
    """Raised when the dependency graph unexpectedly contains a cycle."""


# ═══════════════════════════════════════════════════════════════════════════════
# Basic-block splitting (lossless)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_dotlabel(line: ParsedAsmLine) -> bool:
    """True for the linear-regalloc pseudo label ``.label X  # X``."""
    return line.is_directive and line.opcode == "label"


def _is_anchor(line: ParsedAsmLine) -> bool:
    """True for lines that never move and split scheduling regions."""
    if line.is_empty or line.is_comment_only:
        return True
    if line.is_directive:
        return not _is_dotlabel(line)
    if line.label is not None:
        return True
    return line.opcode is None


def _make_schedinst(
    line: ParsedAsmLine, uid: int, info: OpcodeInfo
) -> Optional[SchedInst]:
    """Build a schedulable instruction; None when it must be a barrier."""
    opcode = line.opcode or ""
    defines, uses, ok = classify_operands(opcode, line.operands, info)
    if not ok:
        return None
    return SchedInst(
        id=uid,
        opcode=opcode,
        operands=list(line.operands),
        defines=defines,
        uses=uses,
        raw_line=line.raw,
        info=info,
        parsed=line,
        pinned=info.is_terminator,
    )


def split_blocks(asm_text: str) -> tuple[list[Any], bool]:
    """Split assembly text into anchored lines and basic blocks.

    Returns ``(pieces, trailing_newline)`` where each piece is either a
    ``ParsedAsmLine`` (anchored, never moves) or a ``Block``.
    """
    trailing_newline = asm_text.endswith("\n")
    lines = parse_asm(asm_text)
    if trailing_newline and lines and lines[-1].is_empty:
        lines.pop()                     # phantom line from the final "\n"

    pieces: list[Any] = []
    current: list[SchedInst] = []
    anchor_id = "<anonymous>"
    uid = 0

    def flush() -> None:
        nonlocal current
        if current:
            first = current[0]
            line_no = (first.parsed.lineno + 1) if first.parsed else 0
            pieces.append(Block(
                id=f"{anchor_id}@L{line_no}",
                insts=current,
                start_idx=first.id,
            ))
            current = []

    for line in lines:
        if _is_anchor(line) or _is_dotlabel(line):
            flush()
            pieces.append(line)
            if line.label:
                anchor_id = line.label
            elif _is_dotlabel(line) and line.operands:
                anchor_id = line.operands[0]
            continue

        info = _info(line.opcode or "")
        if info.is_barrier:
            flush()
            pieces.append(line)         # barrier: anchored + region split
            continue

        inst = _make_schedinst(line, uid, info)
        if inst is None:
            flush()
            pieces.append(line)         # unclassifiable: anchored
            continue

        uid += 1
        current.append(inst)
        if inst.pinned:
            flush()                     # terminators end their block

    flush()
    return pieces, trailing_newline


# ═══════════════════════════════════════════════════════════════════════════════
# InstructionScheduler
# ═══════════════════════════════════════════════════════════════════════════════

class InstructionScheduler:
    """List scheduler for RISC-V basic blocks.

    Parameters
    ----------
    latency_model:
        Optional opcode -> result-latency override (legacy API).
    """

    def __init__(self, latency_model: Optional[dict[str, int]] = None):
        self.latency_model: Optional[dict[str, int]] = latency_model
        self._nodes: list[DAGNode] = []
        self.edges: list[SchedEdge] = []
        self._edge_kinds: dict[tuple[int, int], set[EdgeKind]] = {}
        self._edge_sep: dict[tuple[int, int], int] = {}

    # ── helpers ─────────────────────────────────────────────────────────────

    def _info_of(self, inst: SchedInst) -> OpcodeInfo:
        return inst.info if inst.info is not None else _info(inst.opcode)

    def _lat(self, inst: SchedInst) -> int:
        if self.latency_model is not None:
            override = self.latency_model.get(inst.opcode)
            if override is not None:
                return int(override)
        return self._info_of(inst).result_latency

    def _add_edge(
        self, pred: DAGNode, succ: DAGNode,
        kind: EdgeKind, separation: int,
    ) -> None:
        if pred.inst.id == succ.inst.id:
            return
        key = (pred.inst.id, succ.inst.id)
        self._edge_kinds.setdefault(key, set()).add(kind)
        previous = self._edge_sep.get(key)
        if previous is None or separation > previous:
            self._edge_sep[key] = separation

    # ── DAG construction ────────────────────────────────────────────────────

    def build_dag(self, instructions: list[SchedInst]) -> list[DAGNode]:
        """Build the dependency DAG for one basic block."""
        self._nodes = [DAGNode(inst=inst) for inst in instructions]
        by_id = {node.inst.id: node for node in self._nodes}
        self._edge_kinds = {}
        self._edge_sep = {}
        self.edges = []

        last_def: dict[str, DAGNode] = {}
        readers: dict[str, list[DAGNode]] = {}
        last_store: Optional[DAGNode] = None
        loads_since_store: list[DAGNode] = []

        for node in self._nodes:
            inst = node.inst

            # RAW: every use depends on the most recent definition
            for reg in inst.uses:
                pred = last_def.get(reg)
                if pred is not None:
                    self._add_edge(
                        pred, node, EdgeKind.RAW, self._lat(pred.inst))

            # WAW / WAR: later definitions must not pass prior ones
            for reg in inst.defines:
                pred = last_def.get(reg)
                if pred is not None:
                    self._add_edge(pred, node, EdgeKind.WAW, 0)
                for reader in readers.get(reg, []):
                    self._add_edge(reader, node, EdgeKind.WAR, 0)
                readers[reg] = []
                last_def[reg] = node

            for reg in inst.uses:
                if reg not in inst.defines:
                    readers.setdefault(reg, []).append(node)

            # Memory: conservative total order (no alias analysis)
            info = self._info_of(inst)
            if info.memory == "load":
                if last_store is not None:
                    self._add_edge(last_store, node, EdgeKind.MEMORY, 0)
                loads_since_store.append(node)
            elif info.memory == "store":
                if last_store is not None:
                    self._add_edge(last_store, node, EdgeKind.MEMORY, 0)
                for load_node in loads_since_store:
                    self._add_edge(load_node, node, EdgeKind.MEMORY, 0)
                last_store = node
                loads_since_store = []

        # Control: every body node precedes the pinned terminator
        terminators = [n for n in self._nodes if n.inst.pinned]
        if terminators:
            for node in self._nodes:
                if node.inst.pinned:
                    continue
                for terminator in terminators:
                    self._add_edge(node, terminator, EdgeKind.CONTROL, 0)

        self._materialize(by_id)
        if not self._acyclic():
            raise DependencyCycleError("dependency graph contains a cycle")
        self._compute_priorities()
        return self._nodes

    def _materialize(self, by_id: dict[int, DAGNode]) -> None:
        for node in self._nodes:
            node.predecessors = []
            node.successors = []
            node.unscheduled_preds = 0
        for key in sorted(self._edge_kinds):
            pred = by_id[key[0]]
            succ = by_id[key[1]]
            separation = self._edge_sep[key]
            pred.successors.append((succ, separation))
            succ.predecessors.append((pred, separation))
            succ.unscheduled_preds += 1
            self.edges.append(SchedEdge(
                pred=key[0],
                succ=key[1],
                kinds=frozenset(self._edge_kinds[key]),
                min_separation=separation,
            ))

    def _acyclic(self) -> bool:
        indegree = {n.inst.id: n.unscheduled_preds for n in self._nodes}
        stack = [n for n in self._nodes if indegree[n.inst.id] == 0]
        seen = 0
        while stack:
            node = stack.pop()
            seen += 1
            for succ, _ in node.successors:
                indegree[succ.inst.id] -= 1
                if indegree[succ.inst.id] == 0:
                    stack.append(succ)
        return seen == len(self._nodes)

    def _topo_order(self) -> list[DAGNode]:
        indegree = {n.inst.id: n.unscheduled_preds for n in self._nodes}
        ready = [n for n in self._nodes if indegree[n.inst.id] == 0]
        order: list[DAGNode] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for succ, _ in node.successors:
                indegree[succ.inst.id] -= 1
                if indegree[succ.inst.id] == 0:
                    ready.append(succ)
        return order

    def _compute_priorities(self) -> None:
        """Critical-path height, back-propagated in reverse topological order."""
        for node in reversed(self._topo_order()):
            best = self._lat(node.inst)
            for succ, separation in node.successors:
                candidate = separation + succ.priority
                if candidate > best:
                    best = candidate
            node.priority = best

    # ── list scheduling ─────────────────────────────────────────────────────

    def schedule(self, dag: list[DAGNode]) -> list[SchedInst]:
        """List-schedule the DAG; returns instructions in issue order."""
        self._nodes = dag
        for node in dag:
            node.scheduled = False
            node.ready_time = 0
            node.earliest_issue = 0
            node.issue_cycle = None
            node.unscheduled_preds = len(node.predecessors)

        remaining = list(dag)
        result: list[SchedInst] = []
        clock = 0

        while remaining:
            ready = [
                n for n in remaining
                if n.unscheduled_preds == 0 and n.earliest_issue <= clock
            ]
            if not ready:
                dep_ready = [n for n in remaining if n.unscheduled_preds == 0]
                if not dep_ready:       # pragma: no cover - guarded by DAG check
                    raise DependencyCycleError("no issuable instruction")
                clock = min(n.earliest_issue for n in dep_ready)
                continue

            ready.sort(key=lambda n: (-n.priority, n.earliest_issue, n.inst.id))
            node = ready[0]
            node.scheduled = True
            node.issue_cycle = clock
            node.ready_time = clock
            result.append(node.inst)
            remaining.remove(node)

            for succ, separation in node.successors:
                expected = clock + separation
                if succ.earliest_issue < expected:
                    succ.earliest_issue = expected
                succ.unscheduled_preds -= 1
            clock += 1

        return result

    # ── cost model ──────────────────────────────────────────────────────────

    def estimate_sequence(self, insts: list[SchedInst]) -> tuple[int, int]:
        """Order-sensitive single-issue scoreboard estimate.

        Returns ``(total_cycles, stall_cycles)``.
        """
        reg_ready: dict[str, int] = {}
        ex_free = 0
        clock = 0
        stalls = 0
        for inst in insts:
            info = self._info_of(inst)
            need = 0
            for reg in inst.uses:
                ready = reg_ready.get(reg, 0)
                if ready > need:
                    need = ready
            issue = max(clock, ex_free, need)
            stalls += issue - clock
            latency = self._lat(inst)
            for reg in inst.defines:
                reg_ready[reg] = issue + latency
            ex_free = issue + (
                info.exec_latency if not info.pipelined else 1)
            clock = issue + 1
        return clock, stalls

    def estimate_cycles(self, instructions: list[SchedInst]) -> int:
        """Legacy API: total estimated cycles for a sequence."""
        cycles, _ = self.estimate_sequence(instructions)
        return cycles

    def report(self, original: list[SchedInst],
               scheduled: list[SchedInst]) -> str:
        """Return a comparison report between original and scheduled order."""
        orig_cycles = self.estimate_cycles(original)
        sched_cycles = self.estimate_cycles(scheduled)
        improvement = orig_cycles - sched_cycles
        pct = (improvement / orig_cycles * 100) if orig_cycles > 0 else 0.0

        lines = ["Instruction Scheduling Report",
                 f"  Original instructions: {len(original)}",
                 f"  Estimated cycles (original): {orig_cycles}",
                 f"  Estimated cycles (scheduled): {sched_cycles}",
                 f"  Improvement: {improvement} cycles ({pct:.1f}%)",
                 "  Scheduled order:"]
        for index, inst in enumerate(scheduled):
            ops = ", ".join(inst.operands) if inst.operands else ""
            lines.append(f"    {index}: {inst.opcode} {ops}".rstrip())
        return "\n".join(lines)

    # ── verification ────────────────────────────────────────────────────────

    def verify(self, original: list[SchedInst], scheduled: list[SchedInst],
               edges: Optional[list[SchedEdge]] = None) -> list[str]:
        """Verify a candidate order; returns a list of error messages."""
        errors: list[str] = []
        if sorted(i.id for i in original) != sorted(i.id for i in scheduled):
            return ["instruction set changed (not a permutation)"]

        order = {inst.id: index for index, inst in enumerate(scheduled)}
        for edge in (edges if edges is not None else self.edges):
            if order.get(edge.pred, -1) >= order.get(edge.succ, -1):
                errors.append(
                    f"dependency edge {edge.pred}->{edge.succ} violated")
        if original and original[-1].pinned:
            if not scheduled or not scheduled[-1].pinned:
                errors.append("terminator is not the last instruction")
        return errors


# ═══════════════════════════════════════════════════════════════════════════════
# schedule_asm — compiler integration entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild(pieces: list[Any], applied: dict[int, list[SchedInst]],
             trailing_newline: bool) -> str:
    """Rebuild assembly text losslessly from pieces."""
    lines: list[str] = []
    for piece in pieces:
        if isinstance(piece, Block):
            for inst in applied.get(id(piece), piece.insts):
                lines.append(inst.raw_line)
        else:
            lines.append(piece.raw)
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    return text


def _anchor_diagnostics(pieces: list[Any]) -> list[str]:
    """Warnings for anchored instructions that cannot be scheduled."""
    warnings: list[str] = []
    for piece in pieces:
        if isinstance(piece, Block):
            continue
        line = piece
        if line.opcode is None or line.is_directive or line.label is not None:
            continue
        info = _info(line.opcode)
        line_no = line.lineno + 1
        if info.is_barrier:
            if line.opcode not in _BARRIER_OPS:
                warnings.append(
                    f"schedule: unknown opcode {line.opcode!r} at line "
                    f"{line_no}; anchored")
        elif not classify_operands(line.opcode, line.operands, info)[2]:
            warnings.append(
                f"schedule: unsupported operands for {line.opcode!r} at "
                f"line {line_no}; anchored")
    return warnings


def schedule_asm(asm_text: str,
                 config: Optional[ScheduleConfig] = None) -> ScheduleResult:
    """Schedule every basic block of an assembly text.

    Blocks are kept in their original order unless the scoreboard model
    predicts strictly fewer cycles; any failure keeps that block as-is.
    """
    config = config or ScheduleConfig()
    scheduler = InstructionScheduler()
    pieces, trailing_newline = split_blocks(asm_text)

    applied: dict[int, list[SchedInst]] = {}
    block_stats: list[BlockStats] = []
    warnings: list[str] = _anchor_diagnostics(pieces)
    seen = applied_count = fallback_count = 0

    for piece in pieces:
        if not isinstance(piece, Block):
            continue
        seen += 1
        insts = piece.insts

        if len(insts) > config.max_block_size:
            fallback_count += 1
            warnings.append(
                f"schedule: {piece.id} skipped: block too large "
                f"({len(insts)} instructions)")
            block_stats.append(BlockStats(
                piece.id, len(insts), 0, 0, 0, 0, 0,
                "fallback: block too large"))
            continue

        try:
            dag = scheduler.build_dag(insts)
            scheduled = scheduler.schedule(dag)
            orig_cycles, orig_stalls = scheduler.estimate_sequence(insts)
            sched_cycles, sched_stalls = scheduler.estimate_sequence(scheduled)
            errors = (scheduler.verify(insts, scheduled)
                      if sched_cycles < orig_cycles else [])
        except DependencyCycleError as exc:
            fallback_count += 1
            warnings.append(f"schedule: {piece.id} fallback: {exc}")
            block_stats.append(BlockStats(
                piece.id, len(insts), 0, 0, 0, 0, 0,
                f"fallback: {exc}"))
            continue
        except Exception as exc:        # optional pass: never break a build
            fallback_count += 1
            warnings.append(
                f"schedule: {piece.id} fallback: unexpected error: {exc}")
            block_stats.append(BlockStats(
                piece.id, len(insts), 0, 0, 0, 0, 0,
                "fallback: unexpected error"))
            continue

        moved = sum(1 for a, b in zip(insts, scheduled) if a.id != b.id)
        status = "unchanged"

        if errors:
            fallback_count += 1
            status = "fallback: verification failed"
            warnings.append(
                f"schedule: {piece.id} fallback: {'; '.join(errors)}")
        elif sched_cycles < orig_cycles:
            applied_count += 1
            status = "applied"
            applied[id(piece)] = scheduled

        block_stats.append(BlockStats(
            piece.id, len(insts), moved, orig_cycles, sched_cycles,
            orig_stalls, sched_stalls, status))

    stats = {
        "blocks_seen": seen,
        "blocks_applied": applied_count,
        "blocks_fallback": fallback_count,
        "blocks": [asdict(item) for item in block_stats],
    }
    return ScheduleResult(
        asm_text=_rebuild(pieces, applied, trailing_newline),
        blocks_seen=seen,
        blocks_applied=applied_count,
        blocks_fallback=fallback_count,
        warnings=warnings,
        stats=stats,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_instructions(asm_text: str) -> list[SchedInst]:
    """Parse assembly text into a list of SchedInst (legacy API).

    Labels, directives and comment-only lines are skipped, as before.
    """
    result: list[SchedInst] = []
    for line in parse_asm(asm_text):
        if line.is_empty or line.is_comment_only or line.is_directive:
            continue
        if line.opcode is None:
            continue
        info = _info(line.opcode)
        defines, uses, _ = classify_operands(line.opcode, line.operands, info)
        result.append(SchedInst(
            id=len(result),
            opcode=line.opcode,
            operands=list(line.operands),
            defines=defines,
            uses=uses,
            raw_line=line.raw,
            info=info,
            parsed=line,
            pinned=info.is_terminator,
        ))
    return result


def _to_machine_operand(text: str) -> Any:
    from scratchv.backend.machine_types import MachineOperand

    if _canon_reg(text) is not None:
        return MachineOperand.reg(text)
    try:
        return MachineOperand.immediate(int(text, 0))
    except ValueError:
        return MachineOperand.vreg(text)


def machine_instrs_from_scheduled(scheduled: list[SchedInst]) -> list:
    """Convert scheduled SchedInst back to MachineInstr (legacy API)."""
    from scratchv.backend.machine_types import MachineInstr, MachineOp

    result = []
    for inst in scheduled:
        if inst.opcode == ".label":
            result.append(MachineInstr(MachineOp.LABEL, comment=""))
            continue
        try:
            mop = MachineOp(inst.opcode)
        except ValueError as exc:
            raise ValueError(
                f"unknown opcode {inst.opcode!r} in scheduled instruction"
            ) from exc
        ops = [_to_machine_operand(operand) for operand in inst.operands]
        dst = ops[0] if len(ops) >= 1 else None
        src1 = ops[1] if len(ops) >= 2 else None
        src2 = ops[2] if len(ops) >= 3 else None
        result.append(MachineInstr(mop, dst, src1, src2, inst.raw_line))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point: python -m scratchv.backend.inst_scheduler in.s."""
    parser = argparse.ArgumentParser(
        description="RISC-V Instruction Scheduler (List Scheduling)",
    )
    parser.add_argument("input", type=str, help="Input assembly file (.s)")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)")
    parser.add_argument(
        "--report", action="store_true",
        help="Print scheduling report to stderr")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 2 when any block falls back")

    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        asm_text = handle.read()

    result = schedule_asm(asm_text, ScheduleConfig(strict=args.strict))

    if args.report:
        print(
            f"Instruction scheduling: {result.blocks_seen} block(s), "
            f"{result.blocks_applied} applied, "
            f"{result.blocks_fallback} fallback",
            file=sys.stderr)
        for warning in result.warnings:
            print(warning, file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(result.asm_text)
    else:
        print(result.asm_text, end="")

    if args.strict and result.blocks_fallback:
        sys.exit(2)


if __name__ == "__main__":
    main()
