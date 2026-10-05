"""Frontend wiring for Fwht / WinogradConv / SpmmCsr (docs/normal 01).

Parses ONNX models with the custom operators, checks domain/op gating and
shape inference, folds Winograd kernels, and cross-checks the parsed IR against
the standalone references through the IR interpreter. Float models exercise the
native FP32 path; INT32 models exercise the Q16.16 path (same opcodes).
"""

from __future__ import annotations

import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError
from scratchv.verification.ir_interpreter import IRInterpreter
from tests._op_ref import (
    conv_direct_ref_f32,
    conv_direct_ref_q16,
    fwht_forward_f32,
    fwht_forward_q16,
    q16,
    spmm_ref_f32,
    spmm_ref_q16,
)

_ROW = [0, 2, 2, 3, 4]
_COL = [0, 2, 1, 3]
_VALUES = [0.25, -0.5, 0.75, 0.125]


def _save(model):
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return tmp.name


def _parse(model):
    parser = ONNXParser()
    program = parser.parse(_save(model))
    ok, issues = verify_ir(program)
    assert ok, issues
    return parser, program


def _run(model, feed):
    parser, program = _parse(model)
    return IRInterpreter(program).run(feed, initializers=parser.initializers).return_value


def _fwht_model(n, direction="forward", domain="org.scratchv", elem_type=TensorProto.FLOAT):
    x = helper.make_tensor_value_info("X", elem_type, [1, n])
    y = helper.make_tensor_value_info("Y", elem_type, [1, n])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain=domain, direction=direction)
    graph = helper.make_graph([node], "fwht", [x], [y])
    imports = [helper.make_opsetid("", 13)]
    if domain:
        imports.insert(0, helper.make_opsetid(domain, 1))
    return helper.make_model(graph, opset_imports=imports)


def _spmm_model(values, col, row, m, k, n, elem_type=TensorProto.FLOAT, domain=""):
    x = helper.make_tensor_value_info("B", elem_type, [k, n])
    y = helper.make_tensor_value_info("C", elem_type, [m, n])
    nnz = len(values)
    if elem_type == TensorProto.INT32:
        vals = helper.make_tensor("values", elem_type, [nnz], [int(v) for v in values])
    else:
        vals = helper.make_tensor("values", elem_type, [nnz], [float(v) for v in values])
    coli = helper.make_tensor("col", TensorProto.INT32, [nnz], [int(c) for c in col])
    rowi = helper.make_tensor("rowptr", TensorProto.INT32, [m + 1], [int(r) for r in row])
    node = helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"], domain=domain)
    graph = helper.make_graph([node], "spmm", [x], [y], [vals, coli, rowi])
    imports = [helper.make_opsetid("", 13)]
    if domain:
        imports.insert(0, helper.make_opsetid(domain, 1))
    return helper.make_model(graph, opset_imports=imports)


def _winograd_model(weight, bias=None):
    cout, cin, _, _ = weight.shape
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, 8, 8])
    wt = helper.make_tensor("w", TensorProto.FLOAT, list(weight.shape),
                            weight.astype(np.float32).ravel().tolist())
    inits, ins = [wt], ["x", "w"]
    if bias is not None:
        inits.append(helper.make_tensor("b", TensorProto.FLOAT, [cout],
                                        bias.astype(np.float32).tolist()))
        ins.append("b")
    node = helper.make_node("WinogradConv", ins, ["y"], domain="org.scratchv", pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "wino", [x], [y], inits)
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("org.scratchv", 1), helper.make_opsetid("", 13)])


# ── domain / op gating ─────────────────────────────────────────────────────

def test_unknown_domain_is_rejected():
    model = _fwht_model(16, domain="com.example")
    with pytest.raises(ONNXParseError) as exc:
        ONNXParser().parse(_save(model))
    assert "Unsupported ONNX domains" in str(exc.value)


def test_unknown_op_in_supported_domain_is_rejected():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])
    node = helper.make_node("Bogus", ["X"], ["Y"], domain="org.scratchv")
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]), opset_imports=[
        helper.make_opsetid("org.scratchv", 1), helper.make_opsetid("", 13)])
    with pytest.raises(ONNXParseError) as exc:
        ONNXParser().parse(_save(model))
    assert "Unsupported ONNX op types" in str(exc.value)


# ── Fwht ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [16, 64])
def test_fwht_float_parses_and_runs(n):
    rng = np.random.default_rng(n)
    x = rng.uniform(-1.0, 1.0, (1, n)).astype(np.float32)
    parser, program = _parse(_fwht_model(n))
    dest = program.functions[0].blocks[0].instructions[0].dest
    assert dest.shape == (1, n)
    got = IRInterpreter(program).run({"X": x}, initializers=parser.initializers).return_value
    assert np.allclose(got.reshape(-1), fwht_forward_f32(x.reshape(-1)))


def test_fwht_int32_runs_q16():
    n = 64
    rng = np.random.default_rng(n)
    x = q16(rng.uniform(-1.0, 1.0, (1, n)))
    parser, program = _parse(_fwht_model(n, elem_type=TensorProto.INT32))
    got = IRInterpreter(program).run({"X": x}, initializers=parser.initializers).return_value
    assert got.reshape(-1).tolist() == fwht_forward_q16(x.reshape(-1))


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ONNXParseError) as exc:
        ONNXParser().parse(_save(_fwht_model(48)))
    assert "power-of-two" in str(exc.value)


def test_fwht_rejects_bad_direction():
    with pytest.raises(ONNXParseError) as exc:
        ONNXParser().parse(_save(_fwht_model(16, direction="sideways")))
    assert "direction" in str(exc.value)


# ── SpmmCsr ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,k,n", [(4, 4, 1), (4, 4, 2), (4, 6, 3)])
def test_spmm_float_parses_and_runs(m, k, n):
    rng = np.random.default_rng(3)
    b = rng.uniform(-0.4, 0.4, (k, n)).astype(np.float32)
    parser, program = _parse(_spmm_model(_VALUES, _COL, _ROW, m, k, n))
    dest = program.functions[0].blocks[0].instructions[0].dest
    assert dest.shape == (m, n)
    got = IRInterpreter(program).run({"B": b}, initializers=parser.initializers).return_value
    ref = spmm_ref_f32(np.array(_VALUES, np.float32), _COL, _ROW, b, m, n)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_spmm_int32_runs_q16():
    m, k, n = 4, 6, 3
    rng = np.random.default_rng(5)
    b = q16(rng.uniform(-0.4, 0.4, (k, n)))
    parser, program = _parse(_spmm_model(q16(_VALUES).tolist(), _COL, _ROW, m, k, n,
                                         elem_type=TensorProto.INT32))
    got = IRInterpreter(program).run({"B": b}, initializers=parser.initializers).return_value
    ref = np.array(spmm_ref_q16(q16(_VALUES), _COL, _ROW, b, m, n), np.int32)
    assert (got == ref).all()


# ── WinogradConv ───────────────────────────────────────────────────────────

def test_winograd_folds_u_and_runs():
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32)
    x = rng.uniform(-0.25, 0.25, (1, 3, 8, 8)).astype(np.float32)
    parser, program = _parse(_winograd_model(weight))
    assert "w__wino23" in parser.initializers
    assert parser.initializers["w__wino23"].shape == (4, 3, 4, 4)
    got = IRInterpreter(program).run({"x": x}, initializers=parser.initializers).return_value
    assert np.allclose(got, conv_direct_ref_f32(x, weight, 1), rtol=1e-4, atol=1e-4)


def test_winograd_optional_bias_is_consumed():
    rng = np.random.default_rng(1)
    weight = rng.normal(0, 0.3, (2, 2, 3, 3)).astype(np.float32)
    bias = rng.normal(0, 0.1, (2,)).astype(np.float32)
    parser, program = _parse(_winograd_model(weight, bias=bias))
    op = program.functions[0].blocks[0].instructions[0]
    assert len(op.operands) == 3  # x, U, bias


def test_winograd_q16_kernel_still_matches():
    # A WinogradConv produced directly in IR (weight folded then quantized) must
    # match the Q16 direct-convolution reference within the reduction tolerance.
    from scratchv.standalone.onnx_to_riscv_standalone import winograd_f23_kernel_transform

    rng = np.random.default_rng(4)
    weight = rng.normal(0, 0.3, (2, 3, 3, 3)).astype(np.float32)
    x_q = q16(rng.uniform(-0.25, 0.25, (1, 3, 8, 8)))
    u = np.empty((2, 3, 4, 4), np.float32)
    for oc in range(2):
        for ic in range(3):
            u[oc, ic] = np.asarray(
                winograd_f23_kernel_transform(weight[oc, ic].tolist()), np.float32)
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import DataType as D, Value

    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block()
    xv = builder.make_value(name="x", dtype=D.INT32)
    xv.shape = (1, 3, 8, 8)
    uv = builder.make_value(name="u", dtype=D.INT32)
    uv.shape = (2, 3, 4, 4)
    builder.program.global_values.extend([xv, uv])
    result = builder.winograd_conv(xv, uv, cout=2, cin=3)
    result.shape = (1, 2, 8, 8)
    builder.ret(result)
    got = IRInterpreter(builder.program).run(
        {}, initializers={"x": x_q, "u": q16(u)}).return_value
    ref = conv_direct_ref_q16(x_q, q16(weight), 1)
    assert int(np.abs(got.astype(np.int64) - ref).max()) <= 16 * 3 + 8
