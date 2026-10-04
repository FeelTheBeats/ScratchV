"""Winograd F(2,3) convolution regression (tasks 2.3 / 2.5).

Verifies the generated WinogradConv against an independent direct-convolution
Q16.16 reference (stride 1, SAME). Winograd trades a little precision for
fewer multiplies, so the tolerance scales with the reduction length Cin.
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


def _ref_conv_direct(x_q, w_q, pads):
    """Q16.16 direct convolution (stride 1, arbitrary symmetric pads)."""
    N, C, H, W = x_q.shape
    Cout = w_q.shape[0]
    ph, pw = pads[0], pads[1]
    xp = np.pad(x_q, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((N, Cout, H, W), dtype=np.int64)
    for n in range(N):
        for oc in range(Cout):
            for oh in range(H):
                for ow in range(W):
                    acc = 0
                    for ic in range(C):
                        for kh in range(3):
                            for kw in range(3):
                                prod = _wrap32(int(xp[n, ic, oh + kh, ow + kw])
                                               * int(w_q[oc, ic, kh, kw])) >> 16
                                acc = _wrap32(acc + prod)
                    out[n, oc, oh, ow] = acc
    return out


def _model(weight_f32, cin, h, w):
    cout = weight_f32.shape[0]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, h, w])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, h, w])
    wt = helper.make_tensor("w", TensorProto.FLOAT, list(weight_f32.shape),
                            weight_f32.ravel().tolist())
    node = helper.make_node("WinogradConv", ["x", "w"], ["y"],
                            domain="org.scratchv", pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "wino", [x], [y], [wt])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)],
    )


def _run(model, x_q):
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

    flat = [int(v) for v in np.asarray(x_q).reshape(-1)]
    in_bytes = struct.pack(f"<{len(flat)}i", *flat)
    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, int(meta["code_bytes"]))
    emu.regs[10] = 0x04000000
    emu.regs[11] = 0x05000000
    for i, byte in enumerate(in_bytes):
        emu.mem[0x04000000 + i] = byte
    emu.run(max_instr=100_000_000, uarch=PROFILES["basic"])
    out_el = int(meta["output_elements"])
    return np.array([_wrap32(emu.read_mem_i32(0x05000000 + 4 * i))
                     for i in range(out_el)], dtype=np.int64)


@pytest.mark.parametrize("cin,cout,h,w", [(3, 4, 8, 8), (4, 2, 7, 7), (8, 3, 16, 16)])
def test_winograd_matches_direct_within_tolerance(cin, cout, h, w):
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.3, (cout, cin, 3, 3)).astype(np.float32)
    x_q = _q16(rng.uniform(-0.25, 0.25, (1, cin, h, w)))

    got = _run(_model(weight, cin, h, w), x_q)
    ref = _ref_conv_direct(x_q, _q16(weight), [1, 1]).reshape(-1)

    assert got.size == ref.size
    max_lsb = int(np.abs(got - ref).max())
    tol = 16 * cin + 8  # Winograd Q16 truncation grows with the reduction
    assert max_lsb <= tol, f"max_lsb={max_lsb} > tol={tol}"
