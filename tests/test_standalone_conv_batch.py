"""Conv batch (N>1) regression (task A).

_gen_conv emits the computation once per sample (compile-time unrolled) with
an incremental input/output slice offset. Verify N=1/2/3/4 against an
independent Q16.16 direct-conv reference, including K=5 and odd sizes.
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
    N, C, H, W = x_q.shape
    Cout, _, KH, KW = w_q.shape
    ph, pw = pads
    xp = np.pad(x_q, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((N, Cout, H, W), dtype=np.int64)
    for n in range(N):
        for oc in range(Cout):
            for oh in range(H):
                for ow in range(W):
                    acc = 0
                    for ic in range(C):
                        for kh in range(KH):
                            for kw in range(KW):
                                prod = _wrap32(int(xp[n, ic, oh + kh, ow + kw])
                                               * int(w_q[oc, ic, kh, kw])) >> 16
                                acc = _wrap32(acc + prod)
                    out[n, oc, oh, ow] = acc
    return out


def _model(weight, n, cin, h, w, pad):
    cout, _, k, _ = weight.shape
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, cin, h, w])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [n, cout, h, w])
    wt = helper.make_tensor("w", TensorProto.FLOAT, list(weight.shape),
                            weight.ravel().tolist())
    node = helper.make_node("Conv", ["x", "w"], ["y"],
                            pads=[pad, pad, pad, pad])
    graph = helper.make_graph([node], "c", [x], [y], [wt])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _run(model, x_q):
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.onnx"
        onnx.save(model, path)
        meta = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rc = convert_onnx_to_riscv(path, f"{td}/m.bin", f"{td}/m.s",
                                       benchmark=False, metadata=meta)
        assert rc == 0
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


@pytest.mark.parametrize("n,cin,cout,h,w,k", [
    (1, 3, 4, 8, 8, 3),
    (2, 3, 5, 9, 9, 3),
    (3, 3, 4, 10, 10, 5),
    (4, 4, 8, 8, 8, 3),
])
def test_conv_batch_matches_reference(n, cin, cout, h, w, k):
    rng = np.random.default_rng(n)
    weight = rng.normal(0, 0.3, (cout, cin, k, k)).astype(np.float32)
    x_q = _q16(rng.uniform(-0.25, 0.25, (n, cin, h, w)))
    pad = k // 2

    got = _run(_model(weight, n, cin, h, w, pad), x_q)
    ref = _ref_conv_direct(x_q, _q16(weight), [pad, pad]).reshape(-1)
    assert got.size == ref.size
    assert int(np.abs(got - ref).max()) == 0
