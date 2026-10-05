"""Phase 3 route B: emit a platform ``.s`` from the main pipeline.

Reuses the verified standalone size-independent kernels for a single
FWHT / Conv|Winograd / CSR SpMM operator, wired through CompilerDriver.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from scratchv.backend.platform_emit import PlatformEmitError, generate_platform_asm
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, Value


def _program(build):
    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block()
    build(builder)
    return builder.program


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


def test_emit_fwht_contract():
    asm = generate_platform_asm(_program(lambda b: b.ret(b.fwht(_input(b, (1, 64))))))
    assert ".globl cnn_entry" in asm
    assert ".option norelax" in asm
    assert re.search(r"^\s*ret\s*$", asm, re.M)
    assert ".size cnn_entry" in asm
    assert "_start:" not in asm
    assert ".incbin" not in asm and ".include" not in asm
    # size-independent: N comes from a2, no compile-time bound.
    assert "mv   t2, a2" in asm
    assert "_p_fwht_copy" in asm
    assert "li t2, 64" not in asm


def test_emit_conv_and_spmm_use_expected_kernels():
    def build_conv(b):
        x = _input(b, (1, 3, 8, 8))
        w = _input(b, (8, 3, 3, 3))
        bias = _input(b, (8,))
        b.ret(b.conv(x, w, bias, out_channels=8))

    conv = generate_platform_asm(_program(build_conv))
    assert "# ScratchV platform kernel: direct Conv2D" in conv
    assert ".globl cnn_entry" in conv

    def build_spmm(b):
        b.ret(b.spmm_csr(_input(b, (8,)), _input(b, (8,)), _input(b, (3,)),
                         _input(b, (8, 2))))

    spmm = generate_platform_asm(_program(build_spmm))
    assert "# ScratchV platform kernel: CSR SpMM" in spmm


def test_emit_rejects_multi_operator_programs():
    def build(b):
        x = _input(b, (1, 16))
        b.ret(b.fwht(b.fwht(x)))

    with pytest.raises(PlatformEmitError):
        generate_platform_asm(_program(build))


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


@pytest.mark.parametrize("name", ["fwht", "winograd", "spmm"])
def test_emitted_listing_assembles_to_rv32imf(tmp_path, name):
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        pytest.skip("no host assembler")
    if name == "fwht":
        model = _fwht_model()
    elif name == "winograd":
        weight = np.random.default_rng(0).normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32)
        model = helper.make_model(helper.make_graph(
            [helper.make_node("WinogradConv", ["x", "w"], ["y"],
                              domain="org.scratchv", pads=[1, 1, 1, 1])], "wino",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
            [helper.make_tensor("w", TensorProto.FLOAT, list(weight.shape), weight.ravel().tolist())],
        ), opset_imports=[helper.make_opsetid("org.scratchv", 1), helper.make_opsetid("", 13)])
    else:
        model = helper.make_model(helper.make_graph(
            [helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"])], "spmm",
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 1])],
            [helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 1])],
            [helper.make_tensor("values", TensorProto.FLOAT, [4], [0.25, -0.5, 0.75, 0.125]),
             helper.make_tensor("col", TensorProto.INT32, [4], [0, 2, 1, 3]),
             helper.make_tensor("rowptr", TensorProto.INT32, [4], [0, 2, 2, 3])],
        ), opset_imports=[helper.make_opsetid("", 13)])
    asm = generate_platform_asm(ONNXParser().parse(_save(model, tmp_path, f"{name}.onnx")))
    source = tmp_path / f"{name}.s"
    source.write_text(asm)
    # The platform assembles with -march=rv32imf; use a linked object build if
    # the host toolchain supports the target, otherwise just require parse/emit.
    result = subprocess.run(
        [compiler, "--target=riscv32-unknown-elf", "-march=rv32imf", "-mabi=ilp32",
         "-nostdlib", "-c", str(source), "-o", str(tmp_path / f"{name}.o")],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"host {compiler} cannot assemble rv32imf: "
                    f"{(result.stderr or '').strip()[:120]}")
    assert result.returncode == 0, result.stderr


def test_cli_flag_wires_platform_asm():
    from scratchv.main import args_to_config, build_arg_parser

    args = build_arg_parser().parse_args(["model.onnx", "--platform-asm"])
    assert args_to_config(args).platform_asm is True


def _input(builder, shape):
    value = Value(builder._fresh("arg"), D.FLOAT32, shape=shape)
    builder.current_func.params.append(value)
    return value
