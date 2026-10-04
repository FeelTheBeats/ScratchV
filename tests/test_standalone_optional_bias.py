"""Regression: Conv/Gemm must accept a missing bias input (treated as zero).

Before the fix, `_gen_conv` / `_gen_gemm` hard-indexed `node.inputs[2]`, so a
bias-less model raised IndexError at codegen time. The no-bias output must be
identical to the same model with an explicit all-zero bias.
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


def _conv_model(bias):
    rng = np.random.default_rng(0)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])
    wt = helper.make_tensor("w", TensorProto.FLOAT, [8, 3, 3, 3],
                            rng.normal(0, 0.3, size=8 * 3 * 9).astype(np.float32).ravel().tolist())
    inputs = ["x", "w"]
    inits = [wt]
    if bias is not None:
        bs = helper.make_tensor("b", TensorProto.FLOAT, [8], bias)
        inputs.append("b")
        inits.append(bs)
    node = helper.make_node("Conv", inputs, ["y"], pads=[1, 1, 1, 1])
    g = helper.make_graph([node], "conv", [x], [y], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])


def _gemm_model(bias):
    rng = np.random.default_rng(1)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    wt = helper.make_tensor("w", TensorProto.FLOAT, [8, 4],
                            rng.normal(0, 0.3, size=32).astype(np.float32).ravel().tolist())
    inputs = ["x", "w"]
    inits = [wt]
    if bias is not None:
        bs = helper.make_tensor("b", TensorProto.FLOAT, [8], bias)
        inputs.append("b")
        inits.append(bs)
    node = helper.make_node("Gemm", inputs, ["y"], transB=1)
    g = helper.make_graph([node], "gemm", [x], [y], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])


def _run(model, in_el):
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

    rng = np.random.default_rng(7)
    q = np.trunc(rng.uniform(-0.25, 0.25, size=in_el) * Q).astype(np.int64)
    input_bytes = struct.pack(f"<{in_el}i", *q.tolist())

    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, int(meta["code_bytes"]))
    emu.regs[10] = 0x04000000
    emu.regs[11] = 0x05000000
    for i, b in enumerate(input_bytes):
        emu.mem[0x04000000 + i] = b
    emu.run(max_instr=5_000_000, uarch=PROFILES["basic"])
    out_el = int(meta["output_elements"])
    out = [emu.read_mem_i32(0x05000000 + 4 * i) for i in range(out_el)]
    return [v - 0x100000000 if v >= 0x80000000 else v for v in out]


def test_conv_without_bias_matches_zero_bias():
    no_bias = _run(_conv_model(None), 1 * 3 * 8 * 8)
    zero_bias = _run(_conv_model([0.0] * 8), 1 * 3 * 8 * 8)
    assert no_bias == zero_bias


def test_gemm_without_bias_matches_zero_bias():
    no_bias = _run(_gemm_model(None), 4)
    zero_bias = _run(_gemm_model([0.0] * 8), 4)
    assert no_bias == zero_bias
