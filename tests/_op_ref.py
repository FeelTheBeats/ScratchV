"""Shared reference kernels for the standalone-verified operators.

Two numeric contracts (see docs/normal/03):
  * INT32/Q16.16 -- bit-exact against the standalone ``_gen_*`` generator;
  * FLOAT32      -- native IEEE-754 (tensor-c / platform kernels).

Used by the IR kernel tests and by the tensor-c / frontend cross-checks.
"""

from __future__ import annotations

import numpy as np

Q = 1 << 16


def wrap32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def srai16(value: int) -> int:
    # RV32 SRAI after a low-32 multiply: sign-extend then arithmetic shift.
    return wrap32(value) >> 16


def q16(array) -> np.ndarray:
    """float -> Q16.16 int32 (truncate, clamp), matching float32_to_q16."""
    scaled = np.trunc(np.asarray(array, dtype=np.float64) * Q)
    scaled = np.clip(scaled, -(2 ** 31), 2 ** 31 - 1)
    return scaled.astype(np.int32)


def fwht_forward_q16(values) -> list[int]:
    a = [int(v) for v in values]
    count = len(a)
    assert count > 0 and (count & (count - 1)) == 0
    length = 1
    while length < count:
        for i in range(0, count, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = wrap32(u + v), wrap32(u - v)
        length <<= 1
    return a


def fwht_inverse_q16(values) -> list[int]:
    a = fwht_forward_q16(values)
    return [wrap32(v) >> (len(a).bit_length() - 1) for v in a]


def fwht_forward_f32(values) -> np.ndarray:
    a = np.array(values, dtype=np.float32).reshape(-1).copy()
    count = a.size
    length = 1
    while length < count:
        for i in range(0, count, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = np.float32(u + v), np.float32(u - v)
        length <<= 1
    return a


def spmm_ref_q16(values, col, rowptr, b, m, n) -> list[list[int]]:
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(int(rowptr[i]), int(rowptr[i + 1])):
            a, k = int(values[j]), int(col[j])
            for c in range(n):
                out[i][c] = wrap32(out[i][c] + srai16(a * int(b[k][c])))
    return out


def spmm_ref_f32(values, col, rowptr, b, m, n) -> np.ndarray:
    out = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(int(rowptr[i]), int(rowptr[i + 1])):
            out[i, :] = np.float32(out[i, :] + np.float32(values[j] * b[int(col[j]), :]))
    return out


def conv_direct_ref_q16(x_q, w_q, pad: int) -> np.ndarray:
    batch, channels, height, width = x_q.shape
    cout = w_q.shape[0]
    padded = np.pad(x_q, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((batch, cout, height, width), dtype=np.int64)
    for nb in range(batch):
        for oc in range(cout):
            for oh in range(height):
                for ow in range(width):
                    acc = 0
                    for ic in range(channels):
                        for kh in range(3):
                            for kw in range(3):
                                prod = srai16(
                                    int(padded[nb, ic, oh + kh, ow + kw])
                                    * int(w_q[oc, ic, kh, kw]))
                                acc = wrap32(acc + prod)
                    out[nb, oc, oh, ow] = acc
    return out


def conv_direct_ref_f32(x, w, pad: int) -> np.ndarray:
    batch, channels, height, width = x.shape
    cout = w.shape[0]
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((batch, cout, height, width), dtype=np.float32)
    for nb in range(batch):
        for oc in range(cout):
            for oh in range(height):
                for ow in range(width):
                    acc = np.float32(0.0)
                    for ic in range(channels):
                        for kh in range(3):
                            for kw in range(3):
                                acc = np.float32(
                                    acc + np.float32(padded[nb, ic, oh + kh, ow + kw]
                                                     * w[oc, ic, kh, kw]))
                    out[nb, oc, oh, ow] = acc
    return out
