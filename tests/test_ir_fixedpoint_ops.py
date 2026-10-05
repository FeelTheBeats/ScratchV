"""IR kernels for Fwht / SpmmCsr / WinogradConv (docs/normal 03).

INT32/Q16.16 is checked bit-for-bit against the standalone references; FLOAT32
is checked against native numpy. Both share the opcode and dispatch on dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Instruction, OpCode as O, Value
from scratchv.standalone.onnx_to_riscv_standalone import winograd_f23_kernel_transform
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter
from tests._op_ref import (
    Q,
    conv_direct_ref_f32,
    conv_direct_ref_q16,
    fwht_forward_f32,
    fwht_forward_q16,
    fwht_inverse_q16,
    q16,
    spmm_ref_q16,
)

DTYPES = {
    np.dtype("float32"): D.FLOAT32,
    np.dtype("float64"): D.FLOAT64,
    np.dtype("int32"): D.INT32,
    np.dtype("int64"): D.INT64,
}


def _program(op, arrays, dtype=None, attrs=None, shape=None):
    params = [Value(f"x{i}", DTYPES[a.dtype], shape=a.shape) for i, a in enumerate(arrays)]
    builder = IRBuilder()
    builder.new_function("main", params)
    builder.new_block()
    dest = Value("result", dtype or params[0].dtype, shape=shape or ())
    builder.current_block.add(Instruction(op, dest, params, attrs or {}))
    builder.ret(dest)
    return builder.program, params


def _run(op, arrays, dtype=None, attrs=None, shape=None):
    program, params = _program(op, arrays, dtype, attrs, shape)
    ok, issues = verify_ir(program)
    assert ok, issues
    inputs = {p.name: a for p, a in zip(params, arrays)}
    return IRInterpreter(program).run(inputs).return_value


def _winograd_u(weight: np.ndarray) -> np.ndarray:
    cout, cin = weight.shape[0], weight.shape[1]
    folded = np.empty((cout, cin, 4, 4), np.float32)
    for oc in range(cout):
        for ic in range(cin):
            folded[oc, ic] = np.asarray(
                winograd_f23_kernel_transform(weight[oc, ic].tolist()), dtype=np.float32)
    return folded


# ── FWHT ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 16, 64, 256])
def test_fwht_forward_matches_reference(n):
    rng = np.random.default_rng(n)
    x = q16(rng.uniform(-1.0, 1.0, n))
    got = _run(O.FWHT, [x], D.INT32, {"direction": "forward"}, shape=(n,))
    assert got.reshape(-1).tolist() == fwht_forward_q16(x)


@pytest.mark.parametrize("n", [2, 16, 64, 256])
def test_fwht_impulse_is_all_ones(n):
    x = np.array([Q] + [0] * (n - 1), np.int32)
    got = _run(O.FWHT, [x], D.INT32, {"direction": "forward"}, shape=(n,))
    assert (got == Q).all()


@pytest.mark.parametrize("n", [16, 64, 256])
def test_fwht_inverse_matches_reference(n):
    rng = np.random.default_rng(n + 1)
    x = q16(rng.uniform(-0.5, 0.5, n))
    got = _run(O.FWHT, [x], D.INT32, {"direction": "inverse"}, shape=(n,))
    assert got.reshape(-1).tolist() == fwht_inverse_q16(x)


def test_fwht_float_matches_reference():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, 64).astype(np.float32)
    got = _run(O.FWHT, [x], D.FLOAT32, {"direction": "forward"}, shape=(64,))
    assert np.allclose(got.reshape(-1), fwht_forward_f32(x))


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(IRExecutionError) as exc:
        _run(O.FWHT, [np.zeros(48, np.int32)], D.INT32, {"direction": "forward"}, shape=(48,))
    assert exc.value.code == "ShapeError"


def test_fwht_rejects_unknown_direction():
    with pytest.raises(IRExecutionError) as exc:
        _run(O.FWHT, [np.ones(16, np.int32)], D.INT32, {"direction": "sideways"}, shape=(16,))
    assert exc.value.code == "AttributeError"


# ── SPMM_CSR ───────────────────────────────────────────────────────────────

_ROW = [0, 2, 2, 3, 4]
_COL = [0, 2, 1, 3]
_VALUES = [0.25, -0.5, 0.75, 0.125]


@pytest.mark.parametrize("m,k,n", [(4, 4, 1), (4, 4, 2), (4, 6, 3)])
def test_spmm_matches_reference(m, k, n):
    rng = np.random.default_rng(3)
    b = q16(rng.uniform(-0.4, 0.4, (k, n)))
    values = q16(_VALUES)
    got = _run(O.SPMM_CSR,
               [values, np.array(_COL, np.int32), np.array(_ROW, np.int32), b],
               D.INT32, shape=(m, n))
    ref = np.array(spmm_ref_q16(values, _COL, _ROW, b, m, n), np.int32)
    assert (got == ref).all()


def test_spmm_low32_multiply_wraps():
    # a * B = 2**20 * 2**20 = 2**40 -> low 32 bits are zero -> >>16 == 0.
    values = np.array([2 ** 20], np.int32)
    col = np.array([0], np.int32)
    rowptr = np.array([0, 1], np.int32)
    b = np.array([[2 ** 20]], np.int32)
    got = _run(O.SPMM_CSR, [values, col, rowptr, b], D.INT32, shape=(1, 1))
    assert got[0, 0] == 0


def test_spmm_rejects_out_of_range_column():
    values = np.array([1], np.int32)
    col = np.array([5], np.int32)
    rowptr = np.array([0, 1], np.int32)
    b = np.zeros((2, 1), np.int32)
    with pytest.raises(IRExecutionError) as exc:
        _run(O.SPMM_CSR, [values, col, rowptr, b], D.INT32, shape=(1, 1))
    assert exc.value.code == "IndexError"


def test_spmm_signature_rejects_mixed_value_dtype():
    values = Value("values", D.FLOAT32, shape=(1,))
    col = Value("col", D.INT32, shape=(1,))
    rowptr = Value("rowptr", D.INT32, shape=(2,))
    b = Value("b", D.INT32, shape=(2, 1))
    dest = Value("out", D.FLOAT32, shape=(1, 1))
    program = _manual(O.SPMM_CSR, dest, [values, col, rowptr, b])
    _, issues = verify_ir(program)
    assert any(i.rule == "type-consistency" and "must share a dtype" in i.message for i in issues)


def test_spmm_signature_rejects_float_indices():
    values = Value("values", D.FLOAT32, shape=(1,))
    col = Value("col", D.FLOAT32, shape=(1,))
    rowptr = Value("rowptr", D.INT32, shape=(2,))
    b = Value("b", D.FLOAT32, shape=(2, 1))
    dest = Value("out", D.FLOAT32, shape=(1, 1))
    program = _manual(O.SPMM_CSR, dest, [values, col, rowptr, b])
    _, issues = verify_ir(program)
    assert any(i.rule == "type-consistency" and "col must be i32 or i64" in i.message for i in issues)


# ── WINOGRAD_CONV ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cin,cout,h,w", [(3, 4, 8, 8), (4, 2, 7, 7), (8, 3, 16, 16)])
def test_winograd_q16_matches_direct_within_tolerance(cin, cout, h, w):
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.3, (cout, cin, 3, 3)).astype(np.float32)
    x_q = q16(rng.uniform(-0.25, 0.25, (1, cin, h, w)))
    u_q = q16(_winograd_u(weight))
    got = _run(O.WINOGRAD_CONV, [x_q, u_q], D.INT32,
               {"cout": cout, "cin": cin}, shape=(1, cout, h, w))
    ref = conv_direct_ref_q16(x_q, q16(weight), 1)
    max_lsb = int(np.abs(got.astype(np.int64) - ref).max())
    assert max_lsb <= 16 * cin + 8, f"max_lsb={max_lsb}"


@pytest.mark.parametrize("cin,cout,h,w", [(3, 4, 8, 8), (4, 2, 7, 7)])
def test_winograd_fp32_matches_direct(cin, cout, h, w):
    rng = np.random.default_rng(2)
    weight = rng.normal(0, 0.3, (cout, cin, 3, 3)).astype(np.float32)
    x = rng.uniform(-0.25, 0.25, (1, cin, h, w)).astype(np.float32)
    u = _winograd_u(weight)
    got = _run(O.WINOGRAD_CONV, [x, u], D.FLOAT32,
               {"cout": cout, "cin": cin}, shape=(1, cout, h, w))
    ref = conv_direct_ref_f32(x, weight, 1)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_winograd_signature_rejects_bad_bias_shape():
    x = Value("x", D.FLOAT32, shape=(1, 2, 8, 8))
    u = Value("u", D.FLOAT32, shape=(4, 2, 4, 4))
    bias = Value("bias", D.FLOAT32, shape=(3,))
    dest = Value("out", D.FLOAT32, shape=(1, 4, 8, 8))
    program = _manual(O.WINOGRAD_CONV, dest, [x, u, bias], attrs={"cout": 4, "cin": 2})
    _, issues = verify_ir(program)
    assert any(i.rule == "type-consistency" and "bias shape must match" in i.message for i in issues)


def _manual(op, dest, operands, attrs=None):
    builder = IRBuilder()
    builder.new_function("main", list(operands))
    builder.new_block()
    builder.current_block.add(Instruction(op, dest, operands, attrs or {}))
    builder.ret(dest)
    return builder.program
