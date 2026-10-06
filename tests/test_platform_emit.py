"""Platform ``.s`` emission from the main pipeline via real lowering.

The size-independent kernels are lowered to the backend's MachineInstr model and
rendered with AsmEmitter (no standalone string generators). Correctness is
checked by executing the emitted listing in the shared in-test RV32IMF
interpreter.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from scratchv.backend.platform_emit import PlatformEmitError, generate_platform_asm
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, Value
from tests._op_ref import conv_direct_ref_f32, fwht_forward_f32, spmm_ref_f32
from tests.test_platform_kernels import _RN, _f32, _run


def _parse_platform(asm):
    """Collect labels including ``.L`` locals; skip directives."""
    prog, labels = [], {}
    for raw in asm.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(prog)
            continue
        if line.startswith("."):
            continue
        prog.append(line)
    return prog, labels


def _program(build):
    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block()
    build(builder)
    return builder.program


def _input(builder, shape):
    value = Value(builder._fresh("arg"), D.FLOAT32, shape=shape)
    builder.current_func.params.append(value)
    return value


def _save(model, tmp_path, name):
    path = tmp_path / name
    onnx.save(model, path)
    return str(path)


def _fwht_model(n=64, direction="forward"):
    return helper.make_model(helper.make_graph(
        [helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv", direction=direction)],
        "fwht",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])],
    ), opset_imports=[helper.make_opsetid("org.scratchv", 1), helper.make_opsetid("", 13)])


# ── contract ───────────────────────────────────────────────────────────────

def test_emit_fwht_contract():
    asm = generate_platform_asm(_program(lambda b: b.ret(b.fwht(_input(b, (1, 64))))))
    assert ".globl cnn_entry" in asm
    assert ".option norelax" in asm
    assert re.search(r"^\s*ret\s*$", asm, re.M)
    assert ".size cnn_entry, .-cnn_entry" in asm
    assert "_start:" not in asm
    assert ".incbin" not in asm and ".include" not in asm
    # size-independent: N comes from a2, no compile-time bound.
    assert re.search(r"mv\s+t2,\s+a2", asm)
    assert ".Lp_fwht_copy" in asm
    assert "li t2, 64" not in asm


def test_emit_uses_no_standalone():
    # The lowering module must not import the standalone generators.
    import scratchv.backend.platform_emit as module

    source = open(module.__file__).read()
    assert "scratchv.standalone" not in source
    assert "onnx_to_riscv_standalone" not in source


def test_emit_rejects_multi_operator_programs():
    def build(b):
        x = _input(b, (1, 16))
        b.ret(b.fwht(b.fwht(x)))

    with pytest.raises(PlatformEmitError):
        generate_platform_asm(_program(build))


# ── numeric execution (in-test RV32IMF interpreter) ─────────────────────────

@pytest.mark.parametrize("n", [8, 16, 64, 256, 1024])
def test_lowered_fwht_matches_reference(n):
    asm = generate_platform_asm(_program(lambda b: b.ret(b.fwht(_input(b, (1, n))))))
    prog, labels = _parse_platform(asm)

    rng = np.random.default_rng(n)
    x = rng.uniform(-0.9, 0.9, n).astype(np.float32)
    mem = bytearray(0x200000)
    inp, out = 0x10000, 0x80000
    for i, v in enumerate(x):
        mem[inp + 4 * i:inp + 4 * i + 4] = struct.pack("<f", float(v))
    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, n
    _run(prog, labels, regs, mem)
    got = np.array([struct.unpack("<f", bytes(mem[out + 4 * i:out + 4 * i + 4]))[0]
                    for i in range(n)], dtype=np.float32)
    assert np.allclose(got, fwht_forward_f32(x), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("m,k,n", [(4, 4, 1), (4, 4, 2), (4, 6, 3)])
def test_lowered_spmm_matches_reference(m, k, n):
    row = [0, 2, 2, 3, 4]
    col = [0, 2, 1, 3]
    values = np.array([0.25, -0.5, 0.75, 0.125], np.float32)
    asm = generate_platform_asm(_program(lambda b: b.ret(b.spmm_csr(
        _input(b, (4,)), _input(b, (4,)), _input(b, (4,)), _input(b, (k, n))))))
    prog, labels = _parse_platform(asm)

    rng = np.random.default_rng(1)
    B = rng.uniform(-0.9, 0.9, (k, n)).astype(np.float32)
    mem = bytearray(0x40000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4), int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    wf = lambda a, v: mem.__setitem__(slice(a, a + 4), struct.pack("<f", float(v)))
    inp, out = 0x1000, 0x10000
    off = 0
    for v in [m, k, n, row[-1]]:
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
    regs[10], regs[11], regs[12] = inp, out, row[-1]
    _run(prog, labels, regs, mem)
    got = np.array([struct.unpack("<f", bytes(mem[out + 4 * (i * n + j):
                                                  out + 4 * (i * n + j) + 4]))[0]
                    for i in range(m) for j in range(n)], dtype=np.float32).reshape(m, n)
    assert np.allclose(got, spmm_ref_f32(values, col, row, B, m, n), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch,cin,cout,h,w,k", [(1, 3, 4, 8, 8, 3), (2, 2, 3, 5, 5, 5)])
def test_lowered_conv_matches_reference(batch, cin, cout, h, w, k):
    def build(b):
        x = _input(b, (batch, cin, h, w))
        wt = _input(b, (cout, cin, k, k))
        bias = _input(b, (cout,))
        b.ret(b.conv(x, wt, bias, out_channels=cout, kernel_size=k, stride=1, padding=k // 2))

    asm = generate_platform_asm(_program(build))
    prog, labels = _parse_platform(asm)

    rng = np.random.default_rng(2)
    x = rng.uniform(-0.5, 0.5, (batch, cin, h, w)).astype(np.float32)
    weight = rng.normal(0, 0.3, (cout, cin, k, k)).astype(np.float32)

    mem = bytearray(0x400000)
    w32 = lambda a, v: mem.__setitem__(slice(a, a + 4), int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    wf = lambda a, v: mem.__setitem__(slice(a, a + 4), struct.pack("<f", float(v)))
    inp, out = 0x1000, 0x200000
    off = 0
    for v in [batch, h, w, cin, cout, k]:
        w32(inp + off, v); off += 4
    for v in np.transpose(x, (0, 2, 3, 1)).reshape(-1):   # NHWC
        wf(inp + off, v); off += 4
    for v in weight.reshape(-1):                          # OIHW
        wf(inp + off, v); off += 4
    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, batch * h * w * cin
    _run(prog, labels, regs, mem)
    got = np.array([struct.unpack("<f", bytes(mem[out + 4 * i:out + 4 * i + 4]))[0]
                    for i in range(batch * cout * h * w)],
                   dtype=np.float32).reshape(batch, cout, h, w)
    assert np.allclose(got, conv_direct_ref_f32(x, weight, k // 2), rtol=1e-4, atol=1e-4)


# ── driver / CLI ───────────────────────────────────────────────────────────

def test_driver_platform_asm_end_to_end(tmp_path):
    driver = CompilerDriver(CompilerConfig(platform_asm=True))
    output = tmp_path / "platform.s"
    result = driver.compile(_save(_fwht_model(), tmp_path, "fwht.onnx"), str(output))
    assert result.success, result.errors
    assert result.output_path == str(output)
    text = output.read_text()
    assert ".globl cnn_entry" in text and "ret" in text


def test_driver_platform_asm_rejects_unsupported_model(tmp_path):
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Relu", ["X"], ["Y"])], "relu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
    ), opset_imports=[helper.make_opsetid("", 13)])
    result = CompilerDriver(CompilerConfig(platform_asm=True)).compile(
        _save(model, tmp_path, "relu.onnx"), str(tmp_path / "out.s"))
    assert not result.success
    assert "platform emission requires" in result.errors[0]


def test_cli_flag_wires_platform_asm():
    from scratchv.main import args_to_config, build_arg_parser

    args = build_arg_parser().parse_args(["model.onnx", "--platform-asm"])
    assert args_to_config(args).platform_asm is True


@pytest.mark.parametrize("name", ["fwht", "winograd", "spmm"])
def test_emitted_listing_assembles_to_rv32imf(tmp_path, name):
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        pytest.skip("no host assembler")
    if name == "fwht":
        asm = generate_platform_asm(_program(lambda b: b.ret(b.fwht(_input(b, (1, 64))))))
    elif name == "winograd":
        def build(b):
            x = _input(b, (1, 3, 8, 8))
            wt = _input(b, (4, 3, 3, 3))
            b.ret(b.winograd_conv(x, wt, cout=4, cin=3))
        asm = generate_platform_asm(_program(build))
    else:
        def build(b):
            b.ret(b.spmm_csr(_input(b, (4,)), _input(b, (4,)), _input(b, (4,)),
                             _input(b, (4, 2))))
        asm = generate_platform_asm(_program(build))
    source = tmp_path / f"{name}.s"
    source.write_text(asm)
    result = subprocess.run(
        [compiler, "--target=riscv32-unknown-elf", "-march=rv32imf", "-mabi=ilp32",
         "-nostdlib", "-c", str(source), "-o", str(tmp_path / f"{name}.o")],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"host {compiler} cannot assemble rv32imf: "
                    f"{(result.stderr or '').strip()[:120]}")
    assert result.returncode == 0, result.stderr
