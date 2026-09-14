"""Topic 15 function-inline feature case, built programmatically.

The DSL and ONNX frontends never emit ``OpCode.CALL``, so this case is
constructed with :class:`IRBuilder` instead of a frontend source file.

``build_program()`` (the eligible A/B case)::

    inc(x):                     # 3 instructions
        c = 1
        t = x + c
        return t

    main():
        a = 2 ; b = 3
        r1 = inc(a)             # call site 0 -> clone namespace _inl0
        r2 = inc(b)             # call site 1 -> clone namespace _inl1
        s = r1 + r2
        return s

Expected after ``Inliner`` with the default fixed config: both CALLs are
gone, ``main`` gains two independent clones (``inc_entry_inl0`` /
``inc_entry_inl1``) with distinct renamed definitions in the ``_inl0`` /
``_inl1`` namespaces, every clone RETURN becomes ``br <caller>_inl{k}_cont``
to the continuation block, and the callee ``inc`` still keeps its own
RETURN.  Not running the pass must leave both CALLs in place.

``build_rejected_program()`` (the conservative-rejection case)::

    heavy(p, v):                # side-effect STORE + 9 instructions
        store(p, v)
        ... 6 constants ...
        t = p + v
        return t

    loopy(a):                   # body contains FOR/ENDFOR
        for 0..4 ...
        endfor
        return a

    main():
        r1 = heavy(x, y)
        r2 = loopy(x)
        return r1 + r2

With ``InlinerConfig(max_instrs=4)`` both sites are refused
(``body_too_large`` for the oversized side-effecting callee,
``loop_body_unsupported`` for the loop-bodied one).  The CALLs must stay in
place, one warning per site must be recorded, and the IR must be identical
afterwards.

No claim is made about executing residual CALLs: the RISC-V CALL ABI is not
implemented in this branch.
"""

from __future__ import annotations

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Program


def build_program() -> Program:
    """Eligible case: one callee, two rename-verified call sites."""
    b = IRBuilder()
    x = b.make_value(name="x", dtype=DataType.FLOAT32)
    b.new_function("inc", params=[x])
    b.new_block("entry")
    one = b.load_const(1)
    t = b.add(x, one)
    b.ret(t)

    b.new_function("main")
    b.new_block("entry")
    a = b.make_const(2)
    c = b.make_const(3)
    r1 = b.call("inc", [a])
    r2 = b.call("inc", [c])
    s = b.add(r1, r2)
    b.ret(s)
    return b.program


def build_rejected_program() -> Program:
    """Rejected case: oversized side-effecting callee + loop-bodied callee."""
    b = IRBuilder()
    p = b.make_value(name="p", dtype=DataType.FLOAT32)
    v = b.make_value(name="v", dtype=DataType.FLOAT32)
    b.new_function("heavy", params=[p, v])
    b.new_block("entry")
    b.store(p, v)
    for _ in range(6):
        b.load_const(1)
    t = b.add(p, v)
    b.ret(t)

    a = b.make_value(name="a", dtype=DataType.FLOAT32)
    b.new_function("loopy", params=[a])
    b.new_block("entry")
    b.for_loop(0, 4)
    b.endfor()
    b.ret(a)

    b.new_function("main")
    b.new_block("entry")
    x = b.make_const(1)
    y = b.make_const(2)
    r_heavy = b.call("heavy", [x, y])
    r_loopy = b.call("loopy", [x])
    s = b.add(r_heavy, r_loopy)
    b.ret(s)
    return b.program
