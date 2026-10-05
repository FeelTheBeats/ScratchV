"""Size-independent platform kernels: FWHT and CSR SpMM.

The generated ``.s`` is executed by a tiny in-test RV32IM interpreter (only the
instruction subset the kernels use), so the kernels' logic is verified without
a RISC-V toolchain.
"""

import io
import contextlib
import re
import struct

import numpy as np
import pytest


# ── Minimal RV32IMF interpreter (subset used by the kernels) ───────────────
_RN = {"zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6,
       "t2": 7, "s0": 8, "s1": 9, "a0": 10, "a1": 11, "a2": 12, "a3": 13,
       "a4": 14, "a5": 15, "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20,
       "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
       "t3": 28, "t4": 29, "t5": 30, "t6": 31}
_FN = {f"ft{i}": i for i in range(12)}
_FN.update({f"fs{i}": 12 + i for i in range(12)})
_FN.update({f"fa{i}": 24 + i for i in range(8)})


def _parse(asm):
    prog, labels = [], {}
    for raw in asm.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("."):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(prog)
            continue
        prog.append(line)
    return prog, labels


def _s32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _run(prog, labels, regs, mem, max_steps=10_000_000):
    pc = steps = 0
    fregs = [0.0] * 32
    while pc < len(prog) and steps < max_steps:
        steps += 1
        p = prog[pc].replace(",", " ").split()
        op, r = p[0], p[1:]
        nxt = pc + 1
        g = lambda i: regs[i]
        st = lambda i, v: regs.__setitem__(i, v & 0xFFFFFFFF)
        gf = lambda i: fregs[i]
        sf = lambda i, v: fregs.__setitem__(i, _f32(v))

        if op == "li":
            st(_RN[r[0]], int(r[1]))
        elif op == "mv":
            st(_RN[r[0]], g(_RN[r[1]]))
        elif op in ("lw", "sw"):
            off = int(r[1][:r[1].index("(")])
            bs = r[1][r[1].index("(") + 1:-1]
            addr = (g(_RN[bs]) + off) & 0xFFFFFFFF
            if op == "lw":
                st(_RN[r[0]], int.from_bytes(mem[addr:addr + 4], "little"))
            else:
                mem[addr:addr + 4] = int(g(_RN[r[0]])).to_bytes(4, "little")
        elif op in ("flw", "fsw"):
            off = int(r[1][:r[1].index("(")])
            bs = r[1][r[1].index("(") + 1:-1]
            addr = (g(_RN[bs]) + off) & 0xFFFFFFFF
            if op == "flw":
                fregs[_FN[r[0]]] = struct.unpack("<f", bytes(mem[addr:addr + 4]))[0]
            else:
                mem[addr:addr + 4] = struct.pack("<f", fregs[_FN[r[0]]])
        elif op == "fmv.w.x":
            bits = int(g(_RN[r[1]])) & 0xFFFFFFFF
            fregs[_FN[r[0]]] = struct.unpack("<f", bits.to_bytes(4, "little"))[0]
        elif op in ("fadd.s", "fsub.s", "fmul.s"):
            a, b = gf(_FN[r[1]]), gf(_FN[r[2]])
            sf(_FN[r[0]], {"fadd.s": a + b, "fsub.s": a - b, "fmul.s": a * b}[op])
        elif op == "add":
            st(_RN[r[0]], g(_RN[r[1]]) + g(_RN[r[2]]))
        elif op == "sub":
            st(_RN[r[0]], g(_RN[r[1]]) - g(_RN[r[2]]))
        elif op == "mul":
            st(_RN[r[0]], g(_RN[r[1]]) * g(_RN[r[2]]))
        elif op == "srai":
            st(_RN[r[0]], _s32(g(_RN[r[1]])) >> int(r[2]))
        elif op == "srli":
            st(_RN[r[0]], (g(_RN[r[1]]) & 0xFFFFFFFF) >> int(r[2]))
        elif op == "slli":
            st(_RN[r[0]], g(_RN[r[1]]) << int(r[2]))
        elif op == "addi":
            st(_RN[r[0]], g(_RN[r[1]]) + int(r[2]))
        elif op in ("blt", "bge", "beq", "bne"):
            a, b = _s32(g(_RN[r[0]])), _s32(g(_RN[r[1]]))
            cond = {"blt": a < b, "bge": a >= b,
                    "beq": a == b, "bne": a != b}[op]
            nxt = labels[r[2]] if cond else nxt
        elif op == "j":
            nxt = labels[r[0]]
        elif op == "ret":
            return steps
        else:
            raise AssertionError(f"unknown op {op}")
        pc = nxt
    return steps


def _asm_for(model_builder, tmp_path):
    import onnx
    from scratchv.standalone.onnx_to_riscv_standalone import (
        ONNXModel, CNNRISCVGenerator, MemoryPlan, emit_platform_asm,
    )
    path = tmp_path / "m.onnx"
    onnx.save(model_builder(), str(path))
    model = ONNXModel.from_file(str(path))
    mem = MemoryPlan()
    for vi in model.inputs:
        n = 1
        for d in model.get_shape(vi.name):
            n *= d
        mem.alloc_workspace(vi.name, n)
    wd = mem.layout_weights(model.initializers)
    gen = CNNRISCVGenerator(model, mem)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return emit_platform_asm(gen, wd, model)


def _fwht_model():
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv")
    return helper.make_model(
        helper.make_graph([node], "f", [x], [y]),
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)])


def _spmm_model():
    from onnx import TensorProto, helper
    # empty weight arrays are fine; the platform kernel reads runtime memory
    x = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 2])
    y = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 2])
    vals = helper.make_tensor("values", TensorProto.FLOAT, [4], [0.0] * 4)
    col = helper.make_tensor("col", TensorProto.INT32, [4], [0, 1, 2, 3])
    row = helper.make_tensor("rowptr", TensorProto.INT32, [5], [0, 1, 2, 3, 4])
    node = helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"])
    return helper.make_model(
        helper.make_graph([node], "s", [x], [y], [vals, col, row]),
        opset_imports=[helper.make_opsetid("", 13)])


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def test_platform_fwht_kernel_matches_reference(tmp_path, n):
    asm = _asm_for(_fwht_model, tmp_path)
    prog, labels = _parse(asm)

    rng = np.random.default_rng(0)
    x = [float(v) for v in rng.uniform(-0.9, 0.9, n)]
    mem = bytearray(0x200000)
    inp, out = 0x10000, 0x80000
    for i, v in enumerate(x):
        mem[inp + 4 * i:inp + 4 * i + 4] = struct.pack("<f", v)
    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, n
    _run(prog, labels, regs, mem)

    got = [struct.unpack("<f", bytes(mem[out + 4 * i:out + 4 * i + 4]))[0]
           for i in range(n)]
    a = list(x)
    length = 1
    while length < n:
        for i in range(0, n, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = _f32(u + v), _f32(u - v)
        length <<= 1
    assert np.allclose(got, a, rtol=1e-4, atol=1e-3)


def test_platform_spmm_kernel_matches_reference(tmp_path):
    asm = _asm_for(_spmm_model, tmp_path)
    prog, labels = _parse(asm)

    M, K, N = 4, 4, 2
    values = [0.25, -0.5, 0.75, 0.125]
    col = [0, 2, 1, 3]
    row = [0, 2, 2, 3, 4]
    nnz = row[-1]
    rng = np.random.default_rng(1)
    B = rng.uniform(-0.9, 0.9, (K, N)).astype(np.float32)

    mem = bytearray(0x40000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4),
                                     int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    wf = lambda a, v: mem.__setitem__(slice(a, a + 4), struct.pack("<f", float(v)))
    inp, out = 0x1000, 0x10000

    off = 0
    for v in [M, K, N, nnz]:
        w(inp + off, v); off += 4
    for v in row:
        w(inp + off, v); off += 4
    for v in col:
        w(inp + off, v); off += 4
    for v in values:
        wf(inp + off, v); off += 4
    for v in B.reshape(-1):
        wf(inp + off, v); off += 4

    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, nnz
    _run(prog, labels, regs, mem)

    got = [struct.unpack("<f", bytes(mem[out + 4 * (i * N + j):
                                           out + 4 * (i * N + j) + 4]))[0]
           for i in range(M) for j in range(N)]
    ref = []
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for p in range(row[i], row[i + 1]):
                acc = _f32(acc + _f32(values[p] * float(B[col[p]][j])))
            ref.append(_f32(acc))
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4)


def _conv_model(cin=3, h=8, w=8, k=3, cout=8):
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, h, w])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, h, w])
    wt = helper.make_tensor("w", TensorProto.FLOAT, [cout, cin, k, k],
                            [0.0] * (cout * cin * k * k))
    b = helper.make_tensor("b", TensorProto.FLOAT, [cout], [0.0] * cout)
    pad = k // 2
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"],
                            pads=[pad, pad, pad, pad])
    return helper.make_model(
        helper.make_graph([node], "c", [x], [y], [wt, b]),
        opset_imports=[helper.make_opsetid("", 13)])


@pytest.mark.parametrize("cin,h,w,k,cout", [(3, 8, 8, 3, 8), (3, 6, 6, 5, 8),
                                             (4, 5, 5, 3, 6)])
def test_platform_conv_kernel_matches_reference(tmp_path, cin, h, w, k, cout):
    asm = _asm_for(lambda: _conv_model(cin, h, w, k, cout), tmp_path)
    prog, labels = _parse(asm)

    batch = 1
    rng = np.random.default_rng(0)
    Nn, Cin, H, W = batch, cin, h, w
    Cout, K, pad = cout, k, k // 2
    Hout = Wout = H
    feat = rng.uniform(-0.9, 0.9, (batch, H, W, Cin)).astype(np.float32)   # NHWC
    wt = rng.uniform(-0.9, 0.9, (Cout, Cin, K, K)).astype(np.float32)      # OIHW

    mem = bytearray(0x800000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4),
                                     int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    wf = lambda a, v: mem.__setitem__(slice(a, a + 4), struct.pack("<f", float(v)))
    r32f = lambda a: struct.unpack("<f", bytes(mem[a:a + 4]))[0]
    inp, out = 0x10000, 0x80000

    off = 0
    for v in [batch, H, W, Cin, Cout, K]:
        w(inp + off, v); off += 4
    for v in feat.reshape(-1):
        wf(inp + off, v); off += 4
    for v in wt.reshape(-1):
        wf(inp + off, v); off += 4

    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, batch * H * W * Cin
    _run(prog, labels, regs, mem)

    got = [r32f(out + 4 * i) for i in range(batch * Cout * Hout * Wout)]
    ref = []
    for b in range(batch):
        for oc in range(Cout):
            for oh in range(Hout):
                for ow in range(Wout):
                    acc = 0.0
                    for c in range(Cin):
                        for kh in range(K):
                            ih = oh - pad + kh
                            if ih < 0 or ih >= H:
                                continue
                            for kw in range(K):
                                iw = ow - pad + kw
                                if iw < 0 or iw >= W:
                                    continue
                                x = float(feat[b, ih, iw, c])
                                wv = float(wt[oc, c, kh, kw])
                                acc = _f32(acc + _f32(x * wv))
                    ref.append(_f32(acc))
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4)
