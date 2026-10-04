"""Winograd F(2,3) compile-time kernel transform (task 2.2)."""

import struct
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from scratchv.standalone.onnx_to_riscv_standalone import (
    ONNXModel,
    _prepare_winograd_kernels,
    winograd_f23_kernel_transform,
)

G = np.array([[1.0, 0.0, 0.0],
              [0.5, 0.5, 0.5],
              [0.5, -0.5, 0.5],
              [0.0, 0.0, 1.0]])


def test_kernel_transform_matches_numpy():
    rng = np.random.default_rng(0)
    g = rng.normal(0, 0.5, (3, 3))
    got = np.array(winograd_f23_kernel_transform(g.tolist()))
    want = G @ g @ G.T
    assert np.allclose(got, want, atol=1e-6)


def _winograd_model(weight):
    cout, cin = weight.shape[0], weight.shape[1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, 8, 8])
    wt = helper.make_tensor("w", TensorProto.FLOAT, list(weight.shape),
                            weight.astype(np.float32).ravel().tolist())
    node = helper.make_node("WinogradConv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "wino", [x], [y], [wt])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_prepare_winograd_kernels_registers_u():
    rng = np.random.default_rng(1)
    weight = rng.normal(0, 0.3, (2, 3, 3, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.onnx"
        onnx.save(_winograd_model(weight), path)
        model = ONNXModel.from_file(path)
        _prepare_winograd_kernels(model)

    assert "w__wino23" in model.initializers
    u = model.initializers["w__wino23"]
    assert u.shape == (2, 3, 4, 4)
    assert u.data_type == onnx.TensorProto.FLOAT

    got = np.array(struct.unpack(f"<{2 * 3 * 16}f", u.data)).reshape(2, 3, 4, 4)
    want = np.stack([[G @ weight[oc, ic] @ G.T for ic in range(3)]
                     for oc in range(2)])
    assert np.allclose(got, want, atol=1e-5)

    # node records the transformed tensor name
    node = model.nodes[0]
    assert node.attrs.get("_winograd_u") == "w__wino23"
