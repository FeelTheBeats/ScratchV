"""Regression for the CSR SpmmCsr operator (task 3.4).

Compiles a small CSR sparse × dense model, runs it on the RV32 emulator and
compares against an independent Q16.16 reference. Covers SpMV (N=1), SpMM
(N>1), a non-square matrix and an all-zero (empty) row.
"""

import contextlib
import io
import struct
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from scratchv.standalone.onnx_to_riscv_standalone import convert_onnx_to_riscv
from scratchv.standalone.benchmark import PROFILES, RV32EmulatorFast

Q = 65536


def _q16(a):
    return np.trunc(np.asarray(a, dtype=np.float64) * Q).astype(np.int64)


def _wrap32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _ref_csr(values_q, col, row, b_q, m, n):
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(int(row[i]), int(row[i + 1])):
            a = int(values_q[j])
            k = int(col[j])
            for c in range(n):
                prod = _wrap32(a * int(b_q[k][c])) >> 16
                out[i][c] = _wrap32(out[i][c] + prod)
    return out


def _spmm_model(values, col, row, m, k, n):
    x = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    y = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
    nnz = len(values)
    vals = helper.make_tensor("values", TensorProto.FLOAT, [nnz],
                              [float(v) for v in values])
    coli = helper.make_tensor("col", TensorProto.INT32, [nnz], [int(c) for c in col])
    rowi = helper.make_tensor("rowptr", TensorProto.INT32, [m + 1], [int(r) for r in row])
    node = helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"])
    graph = helper.make_graph([node], "spmm", [x], [y], [vals, coli, rowi])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _run(model, b_q):
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.onnx"
        onnx.save(model, path)
        meta = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rc = convert_onnx_to_riscv(path, f"{td}/m.bin", f"{td}/m.s",
                                       benchmark=False, metadata=meta)
        assert rc == 0, "compile failed"
        with open(f"{td}/m.bin", "rb") as f:
            binary = f.read()

    flat = [int(v) for v in np.asarray(b_q).reshape(-1)]
    in_bytes = struct.pack(f"<{len(flat)}i", *flat)
    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, int(meta["code_bytes"]))
    emu.regs[10] = 0x04000000
    emu.regs[11] = 0x05000000
    for i, byte in enumerate(in_bytes):
        emu.mem[0x04000000 + i] = byte
    emu.run(max_instr=50_000_000, uarch=PROFILES["basic"])
    out_el = int(meta["output_elements"])
    return [_wrap32(emu.read_mem_i32(0x05000000 + 4 * i)) for i in range(out_el)]


# A = [[v,v,0,0],[0,0,0,0],[0,v,0,0],[0,0,0,v]]  (row 1 empty)
_ROW = [0, 2, 2, 3, 4]
_COL = [0, 2, 1, 3]
_VALUES = [0.25, -0.5, 0.75, 0.125]


def _case(m, k, n):
    rng = np.random.default_rng(3)
    b = _q16(rng.uniform(-0.4, 0.4, (k, n)))
    model = _spmm_model(_VALUES, _COL, _ROW, m, k, n)
    got = _run(model, b)
    ref = _ref_csr(_q16(_VALUES), _COL, _ROW, b, m, n)
    return got, [v for row in ref for v in row]


def test_spmv_n1():
    got, ref = _case(4, 4, 1)
    assert got == ref


def test_spmm_n2_with_empty_row():
    got, ref = _case(4, 4, 2)
    assert got == ref


def test_non_square():
    # uses only the first 4 rows/cols; extend B to k=6 to exercise non-square
    rng = np.random.default_rng(5)
    b = _q16(rng.uniform(-0.4, 0.4, (6, 3)))
    model = _spmm_model(_VALUES, _COL, _ROW, 4, 6, 3)
    got = _run(model, b)
    ref = _ref_csr(_q16(_VALUES), _COL, _ROW, b, 4, 3)
    assert got == [v for row in ref for v in row]
