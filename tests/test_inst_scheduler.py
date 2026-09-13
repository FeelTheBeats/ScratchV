"""Precision tests for the intra-block RISC-V instruction scheduler.

The suite pins down the observable behaviour of the rewritten
``scratchv/backend/inst_scheduler.py``: opcode semantics, register
canonicalisation, operand classification, basic-block splitting,
dependency edges, critical-path priorities, list scheduling, the
scoreboard cost model, the verifier, the lossless ``schedule_asm``
entry point and the legacy API.  Assertions are exact values, orders
and byte strings, not smoke checks.
"""

from __future__ import annotations

import random

import pytest

from scratchv.backend._asm_parser import ParsedAsmLine
from scratchv.backend.inst_scheduler import (
    Block,
    DAGNode,
    EdgeKind,
    InstructionScheduler,
    SchedInst,
    _canon_reg,
    _info,
    classify_operands,
    parse_instructions,
    schedule_asm,
    split_blocks,
)

LOAD_USE_ASM = (
    "  lw t0, 0(a0)\n"
    "  add t1, t0, t2\n"
    "  lw t3, 4(a0)\n"
    "  add t4, t3, t5\n"
)
LOAD_USE_SCHEDULED = (
    "  lw t0, 0(a0)\n"
    "  lw t3, 4(a0)\n"
    "  add t1, t0, t2\n"
    "  add t4, t3, t5\n"
)


def _blocks(pieces: list) -> list[Block]:
    return [piece for piece in pieces if isinstance(piece, Block)]


def _anchors(pieces: list) -> list[ParsedAsmLine]:
    return [piece for piece in pieces if isinstance(piece, ParsedAsmLine)]


def _edge(scheduler: InstructionScheduler, pred: int, succ: int):
    matches = [e for e in scheduler.edges if e.pred == pred and e.succ == succ]
    assert len(matches) == 1, matches
    return matches[0]


def _build(text: str, latency_model=None):
    insts = parse_instructions(text)
    scheduler = InstructionScheduler(latency_model=latency_model)
    dag = scheduler.build_dag(insts)
    return insts, scheduler, dag


class TestOpcodeSemantics:
    """OpcodeInfo result latencies, barrier/terminator flags and pipelining."""

    def test_result_latency_values(self):
        assert _info("lw").result_latency == 2
        assert _info("lh").result_latency == 2
        assert _info("flw").result_latency == 2
        assert _info("fld").result_latency == 2
        assert _info("add").result_latency == 1
        assert _info("addi").result_latency == 1
        assert _info("lui").result_latency == 1
        assert _info("mul").result_latency == 3
        assert _info("mulh").result_latency == 3
        assert _info("div").result_latency == 16
        assert _info("remu").result_latency == 16

    def test_barrier_and_terminator_flags(self):
        assert _info("call").is_barrier is True
        assert _info("tail").is_barrier is True
        assert _info("no_such_opcode_xyz").is_barrier is True
        assert _info("add").is_barrier is False
        for opcode in ("beq", "bne", "jal", "jalr", "ret", "jr"):
            assert _info(opcode).is_terminator is True
        assert _info("add").is_terminator is False

    def test_pipelined_flags(self):
        for opcode in ("div", "divu", "rem", "remu",
                       "fdiv.s", "fsqrt.s", "fdiv.d", "fsqrt.d"):
            assert _info(opcode).pipelined is False
        for opcode in ("add", "addi", "mul", "lw", "sw", "fadd.d"):
            assert _info(opcode).pipelined is True
        assert _info("fdiv.d").result_latency == 16
        assert _info("fadd.d").result_latency == 4
        assert _info("fmul.d").result_latency == 5


class TestRegisterCanonicalisation:
    """_canon_reg maps psABI aliases to xN/fN and rejects non-registers."""

    def test_integer_aliases(self):
        assert _canon_reg("ra") == "x1"
        assert _canon_reg("sp") == "x2"
        assert _canon_reg("fp") == "x8"
        assert _canon_reg("s0") == "x8"
        assert _canon_reg("t0") == "x5"
        assert _canon_reg("a0") == "x10"
        assert _canon_reg("x5") == "x5"
        assert _canon_reg("zero") == "x0"

    def test_float_aliases_and_unknown_names(self):
        assert _canon_reg("ft0") == "f0"
        assert _canon_reg("ft11") == "f31"
        assert _canon_reg("fs0") == "f8"
        assert _canon_reg("fs11") == "f27"
        assert _canon_reg("fa0") == "f10"
        assert _canon_reg("fa7") == "f17"
        for index in range(32):
            assert _canon_reg(f"f{index}") == f"f{index}"
        assert _canon_reg("f32") is None
        assert _canon_reg("x32") is None
        assert _canon_reg("notareg") is None
        assert _canon_reg("") is None
        assert _canon_reg("%lo(foo)") is None


class TestOperandClassification:
    """classify_operands roles: store value, load base, terminator implicit regs."""

    def test_memory_operand_roles(self):
        assert classify_operands(
            "sw", ["t0", "0(sp)"], _info("sw")) == (set(), {"x5", "x2"}, True)
        assert classify_operands(
            "lw", ["t0", "0(sp)"], _info("lw")) == ({"x5"}, {"x2"}, True)

    def test_symbolic_offset_memory_operand(self):
        assert classify_operands(
            "lw", ["t0", "%lo(foo)(sp)"],
            _info("lw")) == ({"x5"}, {"x2"}, True)
        assert classify_operands(
            "sw", ["t0", "%lo(foo)(sp)"],
            _info("sw")) == (set(), {"x5", "x2"}, True)

    def test_branch_ignores_label_target(self):
        assert classify_operands(
            "beq", ["t0", "t1", ".Lend"],
            _info("beq")) == (set(), {"x5", "x6"}, True)

    def test_implicit_ra_handling(self):
        assert classify_operands("jal", ["foo"], _info("jal")) == (
            {"x1"}, set(), True)
        assert classify_operands("jal", ["x0", "foo"], _info("jal")) == (
            set(), set(), True)
        assert classify_operands("jal", ["ra", "foo"], _info("jal")) == (
            {"x1"}, set(), True)
        assert classify_operands("jalr", ["zero", "ra"], _info("jalr")) == (
            set(), {"x1"}, True)
        assert classify_operands("ret", [], _info("ret")) == (
            set(), {"x1"}, True)

    def test_x0_and_float_registers(self):
        assert classify_operands("add", ["zero", "t0", "t1"], _info("add")) == (
            set(), {"x5", "x6"}, True)
        assert classify_operands("add", ["t0", "zero", "t1"], _info("add")) == (
            {"x5"}, {"x6"}, True)
        assert classify_operands(
            "fadd.d", ["fa0", "fa1", "fa2"],
            _info("fadd.d")) == ({"f10"}, {"f11", "f12"}, True)
        assert classify_operands(
            "fmul.d", ["ft0", "fa1", "fs2"],
            _info("fmul.d")) == ({"f0"}, {"f11", "f18"}, True)

    def test_unclassifiable_symbolic_operand(self):
        defines, uses, ok = classify_operands(
            "add", ["t0", "t1", "%lo(bar)"], _info("add"))
        assert ok is False
        assert defines == {"x5"}
        assert uses == {"x6"}


class TestSplitBlocks:
    """Lossless block splitting: anchors, labels, terminators and barriers."""

    def test_gas_label_and_inline_label_anchor(self):
        pieces, trailing = split_blocks("main:\n  add t0, t1, t2\n")
        assert trailing is True
        anchors = _anchors(pieces)
        assert len(anchors) == 1 and anchors[0].label == "main"
        blocks = _blocks(pieces)
        assert len(blocks) == 1
        assert [i.opcode for i in blocks[0].insts] == ["add"]
        assert blocks[0].id == "main@L2"

        pieces, _ = split_blocks("main: add t0, t1, t2\n")
        assert _blocks(pieces) == []
        anchors = _anchors(pieces)
        assert len(anchors) == 1
        assert anchors[0].label == "main"
        assert anchors[0].opcode == "add"

    def test_dotlabel_is_boundary_anchor(self):
        pieces, _ = split_blocks(".label main  # main\n  add t0, t1, t2\n")
        dotlabels = [p for p in _anchors(pieces)
                     if p.is_directive and p.opcode == "label"]
        assert len(dotlabels) == 1
        blocks = _blocks(pieces)
        assert len(blocks) == 1
        assert blocks[0].id == "main@L2"
        assert all(i.opcode != "label" for b in blocks for i in b.insts)

        pieces, _ = split_blocks(
            "  add t0, t1, t2\n.label main\n  sub t2, t3, t4\n")
        blocks = _blocks(pieces)
        assert len(blocks) == 2
        assert [i.opcode for i in blocks[0].insts] == ["add"]
        assert [i.opcode for i in blocks[1].insts] == ["sub"]

    def test_terminator_ends_block(self):
        pieces, _ = split_blocks(
            "  add t0, t1, t2\n  beq t0, t1, .Lend\n  add t2, t3, t4\n")
        blocks = _blocks(pieces)
        assert len(blocks) == 2
        assert [i.opcode for i in blocks[0].insts] == ["add", "beq"]
        assert blocks[0].insts[-1].pinned is True
        assert [i.opcode for i in blocks[1].insts] == ["add"]

    def test_call_barrier_splits_region(self):
        pieces, _ = split_blocks(
            "  add t0, t1, t2\n  call foo\n  add t2, t3, t4\n")
        blocks = _blocks(pieces)
        assert len(blocks) == 2
        assert [p.opcode for p in _anchors(pieces)] == ["call"]
        assert all(i.opcode != "call" for b in blocks for i in b.insts)

    def test_directive_empty_and_comment_anchors(self):
        pieces, _ = split_blocks(".text\n\n# standalone\n  add t0, t1, t2\n")
        blocks = _blocks(pieces)
        assert len(blocks) == 1
        assert [p.opcode for p in _anchors(pieces)] == ["text", None, None]

        pieces, _ = split_blocks(
            "  add t0, t1, t2\n.word 7\n  add t2, t3, t4\n")
        assert len(_blocks(pieces)) == 2
        assert [p.opcode for p in _anchors(pieces)] == ["word"]

    def test_trailing_newline_and_empty_text(self):
        pieces, trailing = split_blocks("  add t0, t1, t2\n")
        assert trailing is True
        assert not any(p.is_empty for p in _anchors(pieces))
        assert len(_blocks(pieces)) == 1

        pieces, trailing = split_blocks("  add t0, t1, t2")
        assert trailing is False
        assert len(_blocks(pieces)) == 1

        pieces, trailing = split_blocks("")
        assert trailing is False
        assert _blocks(pieces) == []

        pieces, trailing = split_blocks("\n")
        assert trailing is True
        assert _blocks(pieces) == []


class TestDependencyDAG:
    """DAG edge kinds, separations, memory order and edge merging."""

    def test_raw_edge_separation(self):
        _, scheduler, dag = _build("  lw t0, 0(a0)\n  add t1, t0, t2\n")
        edge = _edge(scheduler, 0, 1)
        assert edge.kinds == frozenset({EdgeKind.RAW})
        assert edge.min_separation == 2
        assert dag[0].successors == [(dag[1], 2)]
        assert dag[1].predecessors == [(dag[0], 2)]

    def test_war_and_waw_edges(self):
        _, scheduler, _ = _build("  sub t3, t0, t4\n  add t0, t1, t2\n")
        edge = _edge(scheduler, 0, 1)
        assert edge.kinds == frozenset({EdgeKind.WAR})
        assert edge.min_separation == 0

        _, scheduler, _ = _build("  add t0, t1, t2\n  sub t0, t3, t4\n")
        edge = _edge(scheduler, 0, 1)
        assert edge.kinds == frozenset({EdgeKind.WAW})
        assert edge.min_separation == 0

    def test_rmw_has_no_self_loop(self):
        insts, scheduler, dag = _build("  addi t0, t0, 1\n")
        assert insts[0].defines == {"x5"}
        assert insts[0].uses == {"x5"}
        assert scheduler.edges == []
        assert dag[0].predecessors == []
        assert dag[0].successors == []

    def test_memory_edges(self):
        _, scheduler, _ = _build("  sw t0, 0(a0)\n  lw t1, 0(a0)\n")
        assert _edge(scheduler, 0, 1).kinds == frozenset({EdgeKind.MEMORY})

        _, scheduler, _ = _build("  lw t0, 0(a0)\n  sw t1, 0(a0)\n")
        assert _edge(scheduler, 0, 1).kinds == frozenset({EdgeKind.MEMORY})

        _, scheduler, _ = _build("  sw t0, 0(a0)\n  sw t1, 4(a0)\n")
        assert _edge(scheduler, 0, 1).kinds == frozenset({EdgeKind.MEMORY})

        _, scheduler, _ = _build("  lw t0, 0(a0)\n  lw t1, 4(a0)\n")
        assert scheduler.edges == []

    def test_control_edge_to_terminator(self):
        insts, scheduler, _ = _build("  add t0, t1, t2\n  beq t3, t4, .Lend\n")
        assert insts[-1].pinned is True
        edge = _edge(scheduler, 0, 1)
        assert edge.kinds == frozenset({EdgeKind.CONTROL})
        assert edge.min_separation == 0

    def test_edge_merging(self):
        _, scheduler, dag = _build("  lw t0, 0(a0)\n  sw t0, 0(a0)\n")
        assert len(scheduler.edges) == 1
        edge = scheduler.edges[0]
        assert edge.kinds == frozenset({EdgeKind.RAW, EdgeKind.MEMORY})
        assert edge.min_separation == 2
        assert dag[0].successors == [(dag[1], 2)]

        _, scheduler, _ = _build("  add t0, a0, a1\n  add a0, t1, t0\n")
        assert len(scheduler.edges) == 1
        assert scheduler.edges[0].kinds == frozenset(
            {EdgeKind.RAW, EdgeKind.WAR})
        assert scheduler.edges[0].min_separation == 1


class TestPriorities:
    """Critical-path height exactly matches the hand-computed values."""

    def test_hand_computed_priorities(self):
        """lw->add (sep 2) ->mul (sep 1), plus an independent lw:
        priorities are 6, 4, 3 and 2 respectively.
        """
        _, scheduler, dag = _build(
            "  lw t0, 0(a0)\n"
            "  add t1, t0, t2\n"
            "  mul t3, t1, t4\n"
            "  lw t5, 4(a0)\n")
        assert _edge(scheduler, 0, 1).min_separation == 2
        assert _edge(scheduler, 1, 2).min_separation == 1
        assert [n.priority for n in dag] == [6, 4, 3, 2]


class TestListScheduling:
    """Issue order, stall estimates and terminator pinning."""

    def test_load_use_schedule_and_estimates(self):
        insts, scheduler, dag = _build(LOAD_USE_ASM)
        scheduled = scheduler.schedule(dag)
        assert [i.id for i in scheduled] == [0, 2, 1, 3]
        assert [i.opcode for i in scheduled] == ["lw", "lw", "add", "add"]
        assert scheduler.estimate_sequence(insts) == (6, 2)
        assert scheduler.estimate_sequence(scheduled) == (4, 0)

    def test_terminator_is_scheduled_last(self):
        _, scheduler, dag = _build(
            "  add t0, t1, t2\n  addi t3, t0, 1\n  beq a0, a1, .Lend\n")
        scheduled = scheduler.schedule(dag)
        assert [i.opcode for i in scheduled] == ["add", "addi", "beq"]
        assert scheduled[-1].pinned is True

    def test_no_dependencies_keep_order(self):
        insts, scheduler, dag = _build(
            "  add t0, t1, t2\n  add t3, t4, t5\n  add t6, s0, s1\n")
        scheduled = scheduler.schedule(dag)
        assert [i.id for i in scheduled] == [0, 1, 2]
        assert scheduler.estimate_sequence(scheduled) == \
            scheduler.estimate_sequence(insts)


class TestVerifier:
    """verify() flags edge violations, permutations and misplaced terminators."""

    def test_violated_order_is_reported(self):
        insts, scheduler, _ = _build(LOAD_USE_ASM)
        errors = scheduler.verify(insts, list(reversed(insts)))
        assert errors
        assert any("dependency edge" in e for e in errors)
        assert any("0->1" in e for e in errors)

    def test_legal_order_and_boundary_checks(self):
        insts, scheduler, _ = _build(LOAD_USE_ASM)
        assert scheduler.verify(insts, insts) == []
        assert scheduler.verify(insts, insts[1:]) == [
            "instruction set changed (not a permutation)"]

        t_insts, t_scheduler, _ = _build(
            "  add t0, t1, t2\n  beq t3, t4, .Lend\n")
        errors = t_scheduler.verify(t_insts, [t_insts[-1], t_insts[0]])
        assert "terminator is not the last instruction" in errors


class TestScheduleAsmEntryPoint:
    """End-to-end lossless scheduling: exact text, stats and determinism."""

    def test_load_use_block_applied_with_exact_output_and_stats(self):
        result = schedule_asm(LOAD_USE_ASM)
        assert result.asm_text == LOAD_USE_SCHEDULED
        assert (result.blocks_seen, result.blocks_applied,
                result.blocks_fallback) == (1, 1, 0)
        assert result.warnings == []
        assert result.stats["blocks_seen"] == 1
        assert result.stats["blocks_applied"] == 1
        assert result.stats["blocks_fallback"] == 0
        assert result.stats["blocks"] == [{
            "block_id": "<anonymous>@L1",
            "n_inst": 4,
            "n_moved": 2,
            "orig_cycles": 6,
            "sched_cycles": 4,
            "orig_stalls": 2,
            "sched_stalls": 0,
            "status": "applied",
        }]

    def test_no_benefit_input_is_byte_identical(self):
        text = (".text\n"
                ".globl main\n"
                "main:\n"
                "  # comment\n"
                "  add t0, t1, t2\n"
                "  addi t0, t0, 1\n"
                "  ret\n")
        result = schedule_asm(text)
        assert result.asm_text == text
        assert result.blocks_applied == 0
        assert result.stats["blocks"][0]["status"] == "unchanged"
        assert result.stats["blocks"][0]["block_id"] == "main@L5"
        assert result.stats["blocks"][0]["n_moved"] == 0

    def test_linear_dotlabel_style_reorders_in_place(self):
        text = (".label main  # main\n"
                "  lw t0, 0(a0)\n"
                "  add t1, t0, t2\n"
                "  lw t3, 4(a0)\n"
                "  add t4, t3, t5")
        result = schedule_asm(text)
        expected = (".label main  # main\n"
                    "  lw t0, 0(a0)\n"
                    "  lw t3, 4(a0)\n"
                    "  add t1, t0, t2\n"
                    "  add t4, t3, t5")
        assert result.asm_text == expected
        assert result.asm_text.startswith(".label main  # main\n")
        assert not result.asm_text.endswith("\n")
        assert result.blocks_applied == 1
        assert result.stats["blocks"][0]["block_id"] == "main@L2"

    def test_trailing_newline_preserved(self):
        with_newline = schedule_asm(LOAD_USE_ASM)
        assert with_newline.asm_text.endswith("\n")
        without = schedule_asm(LOAD_USE_ASM.rstrip("\n"))
        assert without.asm_text == LOAD_USE_SCHEDULED.rstrip("\n")
        assert not without.asm_text.endswith("\n")

    def test_deterministic_across_100_runs(self):
        text = (".text\n"
                "main:\n"
                "  lw t0, 0(a0)\n"
                "  add t1, t0, t2\n"
                "  lw t3, 4(a0)\n"
                "  add t4, t3, t5\n"
                "  ret\n"
                "# trailing comment\n")
        outputs = {schedule_asm(text).asm_text for _ in range(100)}
        assert len(outputs) == 1
        result = schedule_asm(text)
        assert result.blocks_applied == 1
        assert [line.strip() for line in result.asm_text.splitlines()][:2] == \
            [".text", "main:"]

    def test_empty_and_blank_input(self):
        empty = schedule_asm("")
        assert empty.asm_text == ""
        assert empty.blocks_seen == 0
        assert empty.stats["blocks"] == []
        blank = schedule_asm("\n")
        assert blank.asm_text == "\n"
        assert blank.blocks_seen == 0


class TestLegacyCompatibility:
    """Old constructor signatures, latency model, report and parse semantics."""

    def test_legacy_constructor_signatures(self):
        inst = SchedInst(0, "add", ["t0", "t1", "t2"],
                         defines={"t0"}, uses={"t1", "t2"})
        assert (inst.id, inst.opcode, inst.operands) == (
            0, "add", ["t0", "t1", "t2"])
        node = DAGNode(inst=inst)
        other = DAGNode(inst=inst)
        node.predecessors.append((other, 2))
        other.successors.append((node, 2))
        assert node.predecessors == [(other, 2)]
        assert other.successors == [(node, 2)]
        assert node.priority == 0 and node.scheduled is False

    def test_latency_model_override(self):
        scheduler = InstructionScheduler(latency_model={"add": 5})
        assert scheduler.latency_model == {"add": 5}
        insts = parse_instructions(
            "  add t0, t1, t2\n  add t1, t0, t3\n")
        cycles = scheduler.estimate_cycles(insts)
        assert isinstance(cycles, int)
        assert cycles == 6
        assert InstructionScheduler().estimate_cycles(insts) == 2

    def test_report_mentions_scheduling_and_cycles(self):
        insts = parse_instructions(LOAD_USE_ASM)
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        scheduled = scheduler.schedule(dag)
        report = scheduler.report(insts, scheduled)
        assert "Scheduling" in report
        assert "cycles" in report.lower()

    def test_parse_instructions_skips_non_instructions(self):
        insts = parse_instructions(
            "main:\n  add t0, t1, t2\n  # comment\n  .word 1\n  ret\n")
        assert [i.opcode for i in insts] == ["add", "ret"]
        assert insts[-1].pinned is True
        store = parse_instructions("  sw t0, 0(sp)\n")[0]
        assert store.defines == set()
        assert store.uses == {"x5", "x2"}


_REGISTERS = ("t0", "t1", "t2", "t3", "a0", "a1", "s0")


def _random_block(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(2, 8)):
        kind = rng.randrange(6)
        rd = rng.choice(_REGISTERS)
        rs1 = rng.choice(_REGISTERS)
        rs2 = rng.choice(_REGISTERS)
        if kind == 0:
            lines.append(f"  lw {rd}, {rng.choice((0, 4, 8))}(sp)")
        elif kind == 1:
            lines.append(f"  sw {rd}, {rng.choice((0, 4, 8))}(sp)")
        elif kind == 2:
            lines.append(f"  add {rd}, {rs1}, {rs2}")
        elif kind == 3:
            lines.append(f"  addi {rd}, {rs1}, {rng.choice((1, 2, 4))}")
        elif kind == 4:
            lines.append(f"  mul {rd}, {rs1}, {rs2}")
        else:
            lines.append(f"  sub {rd}, {rs1}, {rs2}")
    if rng.random() < 0.5:
        lines.append(
            f"  beq {rng.choice(_REGISTERS)}, {rng.choice(_REGISTERS)}, .Lend")
    else:
        lines.append("  ret")
    return "\n".join(lines)


class TestProperties:
    """Randomised legal blocks: permutation, topological order, pinned tail."""

    def test_random_blocks_preserve_dependencies_and_terminator(self):
        for seed in range(50):
            insts = parse_instructions(_random_block(random.Random(seed)))
            scheduler = InstructionScheduler()
            dag = scheduler.build_dag(insts)
            scheduled = scheduler.schedule(dag)
            assert sorted(i.id for i in scheduled) == sorted(i.id for i in insts)
            position = {i.id: k for k, i in enumerate(scheduled)}
            for edge in scheduler.edges:
                assert position[edge.pred] < position[edge.succ], (seed, edge)
            assert scheduled[-1].pinned is True

    def test_random_blocks_schedule_asm_returns_permutation(self):
        for seed in range(50):
            text = _random_block(random.Random(1000 + seed))
            result = schedule_asm(text + "\n")
            assert result.asm_text.endswith("\n")
            assert result.blocks_seen == 1
            assert result.blocks_fallback == 0
            assert sorted(result.asm_text.splitlines()) == \
                sorted(text.splitlines())
            assert result.asm_text.splitlines()[-1] == text.splitlines()[-1]


class TestAnchorDiagnostics:
    """Diagnostics for anchored instructions (post-review hardening)."""

    def test_unknown_opcode_warns_and_is_anchored(self):
        result = schedule_asm("  addi t0, t1, 1\n  frobnicate x1, x2\n")
        assert any("frobnicate" in warning for warning in result.warnings)
        assert "frobnicate" in result.asm_text

    def test_barrier_opcode_does_not_warn(self):
        result = schedule_asm("  call helper\n  addi t0, t1, 1\n")
        assert result.warnings == []

    def test_unclassifiable_operands_warn(self):
        result = schedule_asm("  addi t0, t1, 1\n  add t0, t1, %lo(bar)\n")
        assert any("unsupported operands" in warning
                   for warning in result.warnings)

    def test_nop_is_schedulable(self):
        result = schedule_asm("  addi t0, t1, 1\n  nop\n  add t2, t1, t1\n")
        assert result.blocks_seen == 1
        assert result.blocks_fallback == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
