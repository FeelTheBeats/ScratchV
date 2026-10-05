"""--platform-asm output shape (platform contract; no RISC-V toolchain needed)."""

import contextlib
import io
import tempfile

import onnx
from onnx import TensorProto, helper
import pytest

from scratchv.standalone.onnx_to_riscv_standalone import (
    ONNXModel,
    CNNRISCVGenerator,
    MemoryPlan,
    emit_platform_asm,
)


def _conv_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])
    import numpy as np
    rng = np.random.default_rng(0)
    w = helper.make_tensor("w", TensorProto.FLOAT, [8, 3, 3, 3],
                           rng.normal(0, 0.3, size=8 * 3 * 9).astype(np.float32).ravel().tolist())
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "c", [x], [y], [w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _build_platform_asm(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(_conv_model(), str(path))
    model = ONNXModel.from_file(str(path))
    mem = MemoryPlan()
    for vi in model.inputs:
        n = 1
        for d in model.get_shape(vi.name):
            n *= d
        mem.alloc_workspace(vi.name, n)
    weight_data = mem.layout_weights(model.initializers)
    gen = CNNRISCVGenerator(model, mem)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return emit_platform_asm(gen, weight_data)


def _fwht_model():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv")
    graph = helper.make_graph([node], "fwht", [x], [y])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)],
    )


def test_platform_asm_is_size_independent_for_fwht(tmp_path):
    from scratchv.standalone.onnx_to_riscv_standalone import emit_platform_asm

    path = tmp_path / "fwht.onnx"
    onnx.save(_fwht_model(), str(path))
    model = ONNXModel.from_file(str(path))
    mem = MemoryPlan()
    for vi in model.inputs:
        n = 1
        for d in model.get_shape(vi.name):
            n *= d
        mem.alloc_workspace(vi.name, n)
    weight_data = mem.layout_weights(model.initializers)
    gen = CNNRISCVGenerator(model, mem)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()

    asm = emit_platform_asm(gen, weight_data, model)
    assert ".globl cnn_entry" in asm
    # runtime-sized: bounds come from a2, not a compile-time constant
    assert "mv   t2, a2" in asm
    assert "blt  t3, t2, _p_fwht_copy" in asm  # copy uses runtime N
    # no hard-coded 64 anywhere as a loop bound
    assert "li   t2, 64" not in asm


def test_platform_asm_contract(tmp_path):
    asm = _build_platform_asm(tmp_path)

    # entry symbol is global cnn_entry, and old _start is gone
    assert ".globl cnn_entry" in asm
    assert "\ncnn_entry:" in asm
    assert not any(l.strip() == "_start:" for l in asm.splitlines())

    # no forbidden directives
    assert ".incbin" not in asm
    assert ".include" not in asm

    # no unresolved placeholder / symbolic temp labels leak
    assert "auipc gp, 0x0" not in asm
    assert ".Lsv_" not in asm
    assert "la gp, __data_start" in asm

    # branch/jump targets use labels, not numeric offsets
    import re
    for line in asm.splitlines():
        code = line.split("#", 1)[0]
        if re.match(r"\s*(bne|beq|blt|bge|j|jal)\b", code):
            assert not re.search(r",\s*-?\d+\s*$", code), f"numeric target: {line}"

    # inline data section with .word, no incbin
    assert "__data_start:" in asm
    assert ".word" in asm
    assert asm.count(".word") > 10
