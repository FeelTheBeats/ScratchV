"""F2: INT32 initializer support in the standalone ONNX parser / memory plan.

CSR sparse operators (SpmmCsr) need integer index arrays (`col_indices`,
`row_ptr`) laid out verbatim as one 32-bit word per element — they must NOT be
run through the Q16.16 float conversion used for weights.
"""

import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from scratchv.standalone.onnx_to_riscv_standalone import (
    MemoryPlan,
    ONNXModel,
    ONNX_INT32,
)


def _relu_model(initializers):
    """Minimal valid graph carrying the given initializers."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "int32_test", [x], [y], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _roundtrip(model, tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    return ONNXModel.from_file(str(path))


def test_parser_reads_int32_raw_data(tmp_path):
    """numpy_helper stores raw_data; parser must report dtype INT32 and shape."""
    values = np.array([10, -20, 30, -40], dtype=np.int32)
    init = numpy_helper.from_array(values, name="idx")
    model = _relu_model([init])

    parsed = _roundtrip(model, tmp_path)
    tensor = parsed.initializers["idx"]

    assert tensor.data_type == ONNX_INT32
    assert tensor.shape == (4,)
    assert struct.unpack("<4i", tensor.data) == (10, -20, 30, -40)


def test_parser_reads_int32_typed_field(tmp_path):
    """helper.make_tensor(raw=False) stores int32_data (field 5), incl. negatives."""
    init = helper.make_tensor(
        "idx", TensorProto.INT32, [4], [100, -200, 300, -400]
    )
    model = _relu_model([init])

    parsed = _roundtrip(model, tmp_path)
    tensor = parsed.initializers["idx"]

    assert tensor.data_type == ONNX_INT32
    assert tensor.shape == (4,)
    assert struct.unpack("<4i", tensor.data) == (100, -200, 300, -400)


def test_layout_weights_int32_is_verbatim_one_word_per_element():
    """INT32 tensors must be laid out as raw int32 words, not Q16.16."""
    model = ONNXModel()
    from scratchv.standalone.onnx_to_riscv_standalone import TensorInfo

    t = TensorInfo()
    t.name = "row_ptr"
    t.shape = (3,)
    t.data_type = ONNX_INT32
    t.data = struct.pack("<3i", 0, 2, 3)
    model.initializers["row_ptr"] = t

    mem = MemoryPlan()
    blob = mem.layout_weights(model.initializers)

    assert mem.get_weight_offset("row_ptr") == 0
    assert blob == struct.pack("<3i", 0, 2, 3)
    # A Q16.16 conversion would inflate 2 -> 131072; ensure it did not happen.
    assert struct.unpack("<3i", blob) == (0, 2, 3)


def test_layout_weights_mixes_float_and_int32():
    """Float weights still become Q16.16; INT32 arrays stay verbatim."""
    model = ONNXModel()
    from scratchv.standalone.onnx_to_riscv_standalone import TensorInfo

    w = TensorInfo()
    w.name = "w"
    w.shape = (2,)
    w.data_type = 1  # ONNX_FLOAT
    w.data = struct.pack("<2f", 0.5, -0.25)

    idx = TensorInfo()
    idx.name = "idx"
    idx.shape = (2,)
    idx.data_type = ONNX_INT32
    idx.data = struct.pack("<2i", 7, -7)
    model.initializers["w"] = w
    model.initializers["idx"] = idx

    mem = MemoryPlan()
    blob = mem.layout_weights(model.initializers)

    w_off = mem.get_weight_offset("w")
    i_off = mem.get_weight_offset("idx")
    assert w_off == 0
    assert i_off == 8  # 2 float words

    assert struct.unpack("<2i", blob[w_off : w_off + 8]) == (0x8000, -0x4000)
    assert struct.unpack("<2i", blob[i_off : i_off + 8]) == (7, -7)
