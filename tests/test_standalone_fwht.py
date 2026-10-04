"""Regression for the scalar Fwht operator (task 1.3).

Uses an analytically-known property instead of a duplicated reference:
FWHT of the impulse [1, 0, 0, ...] is the all-ones vector (first Hadamard
column), and the inverse additionally scales by 1/N.
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


def _fwht_model(n, direction="forward"):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv",
                            direction=direction)
    graph = helper.make_graph([node], "fwht", [x], [y])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)],
    )


def _signed(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _run(model, input_q16):
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

    in_bytes = struct.pack(f"<{len(input_q16)}i", *input_q16)
    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, int(meta["code_bytes"]))
    emu.regs[10] = 0x04000000
    emu.regs[11] = 0x05000000
    for i, b in enumerate(in_bytes):
        emu.mem[0x04000000 + i] = b
    emu.run(max_instr=5_000_000, uarch=PROFILES["basic"])
    out_el = int(meta["output_elements"])
    return [_signed(emu.read_mem_i32(0x05000000 + 4 * i)) for i in range(out_el)]


@pytest.mark.parametrize("n", [2, 16, 64, 256])
def test_forward_impulse_is_all_ones(n):
    impulse = [Q] + [0] * (n - 1)
    assert _run(_fwht_model(n, "forward"), impulse) == [Q] * n


@pytest.mark.parametrize("n", [16, 64, 256])
def test_inverse_impulse_is_scaled_ones(n):
    impulse = [Q] + [0] * (n - 1)
    expected = Q >> (n.bit_length() - 1)  # 1/N in Q16.16
    assert _run(_fwht_model(n, "inverse"), impulse) == [expected] * n


def test_non_power_of_two_raises():
    with pytest.raises(Exception) as exc:
        _run(_fwht_model(48), [0] * 48)
    assert "power-of-two" in str(exc.value)


def test_unknown_direction_raises():
    with pytest.raises(Exception) as exc:
        _run(_fwht_model(16, "sideways"), [0] * 16)
    assert "direction" in str(exc.value)
