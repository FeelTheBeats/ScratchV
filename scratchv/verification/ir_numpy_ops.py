"""Strict NumPy kernels for shared IR, independent of frontend and execution state.

Axes use Python-style negative indexing. REDUCE_MEAN with omitted/empty axes
reduces all dimensions. RESHAPE zeros copy input dimensions (allowzero=False).
SLICE uses ONNX bounds clamping; UNSQUEEZE axes refer to the output rank.
EXPAND computes the broadcast shape of the input and the requested shape.
GELU uses the tanh approximation. CONV is NCHW, group=1, dilation=1; MAXPOOL
accepts NCHW/CHW with no padding. Unsupported semantic attributes are rejected.
CAST preserves ONNX integer narrowing (discard high bits); undefined float-to-int
conversions and nonfinite numeric results fail. ABS and integer POW reject overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scratchv.ir.types import DataType, Instruction, OpCode

DTYPES = {
    DataType.FLOAT32: np.dtype("float32"),
    DataType.FLOAT64: np.dtype("float64"),
    DataType.INT32: np.dtype("int32"),
    DataType.INT64: np.dtype("int64"),
}


class OpError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Kernel:
    handler: Callable
    attributes: frozenset[str]


KERNELS: dict[OpCode, Kernel] = {}


def kernel(opcode, *attributes):
    def register(handler):
        KERNELS[opcode] = Kernel(handler, frozenset(attributes))
        return handler

    return register


def integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise OpError("AttributeError", f"{name} must be an integer")
    return int(value)


def integers(value, name):
    if not isinstance(value, (tuple, list)):
        raise OpError("AttributeError", f"{name} must be an integer sequence")
    return tuple(integer(v, name) for v in value)


def axis(value, rank):
    value = integer(value, "axis")
    if not -rank <= value < rank:
        raise OpError("ShapeError", f"axis {value} outside rank {rank}")
    return value % rank


def axes(values, rank):
    normalized = tuple(axis(a, rank) for a in values)
    if len(set(normalized)) != len(normalized):
        raise OpError("AttributeError", "axes must not contain duplicates")
    return normalized


def flag(value, name):
    if not isinstance(value, (bool, int, np.bool_, np.integer)) or value not in (0, 1):
        raise OpError("AttributeError", f"{name} must be boolean or 0/1")
    return bool(value)


def check_instruction(instr: Instruction):
    """Preflight attributes without evaluating tensors, including dead branches."""
    spec = KERNELS.get(instr.opcode)
    if spec is None:
        raise OpError("UnsupportedOpcode", f"unsupported opcode: {instr.opcode.value}")
    unknown = set(instr.attrs) - spec.attributes
    if unknown:
        raise OpError(
            "UnsupportedAttribute", f"unsupported attributes: {sorted(unknown)}"
        )
    required = {
        OpCode.RESHAPE: ("shape",),
        OpCode.CONCAT: ("axis",),
        OpCode.SLICE: ("starts", "ends"),
        OpCode.UNSQUEEZE: ("axes",),
        OpCode.EXPAND: ("shape",),
        OpCode.LOAD_CONST: ("value",),
        OpCode.DOT: ("length",),
        OpCode.MAXPOOL: ("kernel", "stride"),
    }.get(instr.opcode, ())
    for key in required:
        if key not in instr.attrs:
            raise OpError("AttributeError", f"missing attribute: {key}")
    for key in ("shape", "axes", "perm", "starts", "ends", "steps"):
        if key in instr.attrs:
            integers(instr.attrs[key], key)
    for key in (
        "axis",
        "m",
        "n",
        "k",
        "length",
        "kernel",
        "kernel_size",
        "stride",
        "padding",
        "out_channels",
        "cout",
        "cin",
    ):
        if key in instr.attrs:
            integer(instr.attrs[key], key)
    for key in ("keepdims", "trans_a", "trans_b"):
        if key in instr.attrs:
            flag(instr.attrs[key], key)
    if instr.opcode == OpCode.SLICE:
        count = len(instr.attrs["starts"])
        if len(instr.attrs["ends"]) != count or any(
            key in instr.attrs and len(instr.attrs[key]) != count
            for key in ("axes", "steps")
        ):
            raise OpError("AttributeError", "SLICE parameter lengths must match")
        if "steps" in instr.attrs and 0 in instr.attrs["steps"]:
            raise OpError("AttributeError", "SLICE step cannot be zero")
    if instr.opcode == OpCode.MATMUL:
        supplied = set(instr.attrs) & {"m", "n", "k"}
        if supplied and supplied != {"m", "n", "k"}:
            raise OpError("AttributeError", "MATMUL m/n/k must be supplied together")
        if any(instr.attrs[key] < 1 for key in supplied):
            raise OpError("AttributeError", "MATMUL m/n/k must be positive")
    if (
        instr.opcode in (OpCode.UNSQUEEZE, OpCode.REDUCE_MEAN)
        and "axes" in instr.attrs
        and len(set(instr.attrs["axes"])) != len(instr.attrs["axes"])
    ):
        raise OpError("AttributeError", "duplicate axes")
    if instr.opcode in (OpCode.EXPAND, OpCode.RESHAPE):
        shape = instr.attrs["shape"]
        if instr.opcode == OpCode.EXPAND and any(d < 0 for d in shape):
            raise OpError("AttributeError", "EXPAND dimensions must be nonnegative")
        if instr.opcode == OpCode.RESHAPE and (
            any(d < -1 for d in shape) or shape.count(-1) > 1
        ):
            raise OpError(
                "AttributeError",
                "RESHAPE permits at most one -1 and no smaller dimensions",
            )
    if instr.opcode in (
        OpCode.CONV,
        OpCode.GEMM,
        OpCode.MAXPOOL,
        OpCode.GELU,
        OpCode.SIGMOID,
        OpCode.SOFTMAX,
        OpCode.EXP,
        OpCode.SQRT,
        OpCode.COS,
        OpCode.SIN,
        OpCode.RECIPROCAL,
        OpCode.REDUCE_MEAN,
    ) and instr.dest.dtype not in (DataType.FLOAT32, DataType.FLOAT64):
        raise OpError(
            "UnsupportedDType", f"{instr.opcode.value} requires floating data"
        )
    for key in ("alpha", "beta"):
        if key in instr.attrs and (
            isinstance(instr.attrs[key], bool)
            or not isinstance(instr.attrs[key], (int, float))
            or not np.isfinite(instr.attrs[key])
        ):
            raise OpError("AttributeError", f"{key} must be finite numeric")


def compute(instr, operands):
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        result = np.asarray(
            KERNELS[instr.opcode].handler(
                operands, instr.attrs, DTYPES[instr.dest.dtype]
            )
        )
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise OpError("NumericError", "operation produced nonfinite output")
    return result


@kernel(OpCode.LOAD_CONST, "value")
def load_const(xs, attrs, dtype):
    return np.asarray(attrs["value"], dtype=dtype)


def binary(opcode, function):
    @kernel(opcode)
    def run(xs, attrs, dtype):
        return function(xs[0], xs[1])

    return run


binary(OpCode.ADD, np.add)
binary(OpCode.SUB, np.subtract)
binary(OpCode.MUL, np.multiply)


@kernel(OpCode.DIV)
def divide(xs, attrs, dtype):
    a, b = np.broadcast_arrays(*xs)
    if np.any(b == 0):
        raise OpError("NumericError", "division by zero")
    if np.issubdtype(dtype, np.integer):
        minimum = np.iinfo(dtype).min
        if np.any((a == minimum) & (b == -1)):
            raise OpError("NumericError", "integer division overflow")
        # Object integers retain all i64 bits; // on signed arrays rounds down.
        magnitude = np.abs(a.astype(object)) // np.abs(b.astype(object))
        return np.asarray(
            np.where((a < 0) ^ (b < 0), -magnitude, magnitude), dtype=dtype
        )
    return np.divide(a, b)


@kernel(OpCode.NEG)
def negate(xs, attrs, dtype):
    return np.negative(xs[0])


@kernel(OpCode.EXP)
def exponent(xs, attrs, dtype):
    return np.exp(xs[0])


@kernel(OpCode.ABS)
def absolute(xs, attrs, dtype):
    x = xs[0]
    if np.issubdtype(dtype, np.integer) and np.any(x == np.iinfo(dtype).min):
        raise OpError("NumericError", "ABS integer overflow")
    return np.abs(x)


@kernel(OpCode.COS)
def cosine(xs, attrs, dtype):
    return np.cos(xs[0])


@kernel(OpCode.SIN)
def sine(xs, attrs, dtype):
    return np.sin(xs[0])


@kernel(OpCode.RECIPROCAL)
def reciprocal(xs, attrs, dtype):
    if np.any(xs[0] == 0):
        raise OpError("NumericError", "RECIPROCAL division by zero")
    return np.reciprocal(xs[0])


def _float_to_integer(x, dtype):
    """Truncate, then check exact power-of-two bounds before NumPy casts.

    Comparing against i64.max converts it to 2**63 in float64 and would admit
    an out-of-range endpoint; the exclusive upper bound avoids that bug.
    """
    if not np.isfinite(x).all():
        raise OpError("NumericError", "cannot convert nonfinite data to integer")
    truncated = np.trunc(x)
    bound = 1 << (np.iinfo(dtype).bits - 1)
    if np.any(truncated < -bound) or np.any(truncated >= bound):
        raise OpError("NumericError", "float-to-integer conversion out of range")
    return truncated.astype(dtype)


@kernel(OpCode.CAST)
def cast(xs, attrs, dtype):
    x = xs[0]
    if np.issubdtype(x.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        return _float_to_integer(x, dtype)
    # ONNX defines integer narrowing by dropping high bits, not saturation.
    return x.astype(dtype)


def _bounded_integer_power(base, exponent, minimum, maximum):
    """Exact exponentiation with bounded intermediates, including i64 inputs."""
    negative = base < 0 and exponent % 2 != 0
    limit = -minimum if negative else maximum
    result, factor = 1, abs(base)
    while exponent:
        if exponent & 1:
            result *= factor
            if result > limit:
                raise OpError("NumericError", "POW integer overflow")
        exponent >>= 1
        if exponent:
            factor *= factor
            if factor > limit:
                raise OpError("NumericError", "POW integer overflow")
    return -result if negative else result


@kernel(OpCode.POW)
def power(xs, attrs, dtype):
    base, exponent = np.broadcast_arrays(*xs)
    if np.any((base == 0) & (exponent < 0)):
        raise OpError("NumericError", "POW zero base with negative exponent")
    if np.issubdtype(dtype, np.floating):
        # NumPy can promote f32 with i64/f64 exponent to f64. ONNX output
        # always retains the base dtype; do not cast the exponent prematurely.
        return np.power(base, exponent).astype(dtype)
    bounds = np.iinfo(dtype)
    result = np.empty(base.shape, dtype=dtype)
    for index in np.ndindex(base.shape):
        a, b = int(base[index]), exponent[index].item()
        if not np.isfinite(b):
            raise OpError("NumericError", "POW requires a finite exponent")
        if b >= 0 and int(b) == b:
            result[index] = _bounded_integer_power(a, int(b), bounds.min, bounds.max)
        else:
            # Fractional/negative powers use real arithmetic and truncate to
            # the integer base type, rejecting invalid domains and overflow.
            value = np.float_power(np.float64(a), b)
            result[index] = _float_to_integer(np.asarray(value), dtype)
    return result


@kernel(OpCode.SQRT)
def square_root(xs, attrs, dtype):
    if np.any(xs[0] < 0):
        raise OpError("NumericError", "SQRT requires nonnegative data")
    return np.sqrt(xs[0])


@kernel(OpCode.REDUCE_MEAN, "axes", "keepdims")
def reduce_mean(xs, attrs, dtype):
    x = xs[0]
    selected = attrs.get("axes")
    selected = tuple(range(x.ndim)) if not selected else axes(selected, x.ndim)
    if any(x.shape[a] == 0 for a in selected):
        raise OpError("NumericError", "cannot reduce an empty dimension")
    return np.mean(
        x,
        axis=selected,
        keepdims=flag(attrs.get("keepdims", True), "keepdims"),
        dtype=dtype,
    )


@kernel(OpCode.MATMUL, "m", "n", "k")
def matmul(xs, attrs, dtype):
    a, b = xs
    if "m" in attrs:
        m, n, k = (attrs[key] for key in ("m", "n", "k"))
        if a.ndim == b.ndim == 1:
            a, b = a.reshape(m, k), b.reshape(k, n)
        elif a.shape != (m, k) or b.shape != (k, n):
            raise OpError("ShapeError", "MATMUL m/n/k disagree with tensor shapes")
    return np.matmul(a, b)


@kernel(OpCode.DOT, "length")
def dot(xs, attrs, dtype):
    if any(x.ndim != 1 or x.size != attrs["length"] for x in xs):
        raise OpError("ShapeError", "DOT requires vectors matching length")
    return np.dot(*xs)


@kernel(OpCode.RESHAPE, "shape")
def reshape(xs, attrs, dtype):
    x = xs[0]
    shape = []
    for i, dimension in enumerate(attrs["shape"]):
        if dimension == 0:
            if i >= x.ndim:
                raise OpError(
                    "ShapeError", "RESHAPE zero has no corresponding input axis"
                )
            dimension = x.shape[i]
        shape.append(dimension)
    return np.reshape(x, tuple(shape))


@kernel(OpCode.TRANSPOSE, "perm")
def transpose(xs, attrs, dtype):
    x = xs[0]
    perm = attrs.get("perm", tuple(reversed(range(x.ndim))))
    if len(perm) != x.ndim or set(perm) != set(range(x.ndim)):
        raise OpError(
            "AttributeError", "perm must be a complete nonnegative axis permutation"
        )
    return np.transpose(x, perm)


@kernel(OpCode.CONCAT, "axis")
def concat(xs, attrs, dtype):
    return np.concatenate(xs, axis=axis(attrs["axis"], xs[0].ndim))


@kernel(OpCode.GATHER, "axis")
def gather(xs, attrs, dtype):
    data, indices = xs
    selected = axis(attrs.get("axis", 0), data.ndim)
    dimension = data.shape[selected]
    if np.any(indices < -dimension) or np.any(indices >= dimension):
        raise OpError("IndexError", "GATHER index out of bounds")
    return np.take(data, indices, axis=selected)


@kernel(OpCode.SLICE, "starts", "ends", "axes", "steps")
def slice_tensor(xs, attrs, dtype):
    x = xs[0]
    count = len(attrs["starts"])
    selected = axes(attrs.get("axes", tuple(range(count))), x.ndim)
    steps = attrs.get("steps", (1,) * count)
    slices = [slice(None)] * x.ndim
    for a, start, end, step in zip(selected, attrs["starts"], attrs["ends"], steps):
        dimension = x.shape[a]
        if dimension == 0:
            slices[a] = slice(0, 0)
            continue
        # ONNX clamps negative-step starts to [0, dim-1], whereas Python
        # permits a normalized start of -1 (which produces an empty slice).
        start = start + dimension if start < 0 else start
        end = end + dimension if end < 0 else end
        if step > 0:
            start = max(0, min(dimension, start))
            end = max(0, min(dimension, end))
        else:
            start = max(0, min(dimension - 1, start))
            end = max(-1, min(dimension - 1, end))
            # Python interprets literal -1 relative to the array again.
            # None represents the exclusive position before its first item.
            if end == -1:
                end = None
        slices[a] = slice(start, end, step)
    return x[tuple(slices)]


@kernel(OpCode.UNSQUEEZE, "axes")
def unsqueeze(xs, attrs, dtype):
    selected = axes(attrs["axes"], xs[0].ndim + len(attrs["axes"]))
    return np.expand_dims(xs[0], axis=selected)


@kernel(OpCode.EXPAND, "shape")
def expand(xs, attrs, dtype):
    shape = np.broadcast_shapes(xs[0].shape, tuple(attrs["shape"]))
    return np.broadcast_to(xs[0], shape)


@kernel(OpCode.RELU)
def relu(xs, attrs, dtype):
    return np.maximum(xs[0], np.asarray(0, dtype=dtype))


@kernel(OpCode.SIGMOID)
def sigmoid(xs, attrs, dtype):
    x = xs[0]
    result = np.empty_like(x)
    positive = x >= 0
    one = np.asarray(1, dtype=dtype)
    result[positive] = one / (one + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    result[~positive] = z / (one + z)
    return result


@kernel(OpCode.GELU)
def gelu(xs, attrs, dtype):
    x = xs[0]
    half, one, factor, cubic = (
        np.asarray(v, dtype=dtype) for v in (0.5, 1, np.sqrt(2 / np.pi), 0.044715)
    )
    return half * x * (one + np.tanh(factor * (x + cubic * x**3)))


@kernel(OpCode.SOFTMAX, "axis")
def softmax(xs, attrs, dtype):
    x = xs[0]
    selected = axis(attrs.get("axis", -1), x.ndim)
    if x.shape[selected] == 0:
        raise OpError("NumericError", "SOFTMAX cannot normalize an empty axis")
    maximum = np.max(x, axis=selected, keepdims=True)
    if not np.isfinite(maximum).all():
        raise OpError("NumericError", "SOFTMAX row is entirely masked")
    weights = np.exp(x - maximum)
    return weights / np.sum(weights, axis=selected, keepdims=True, dtype=dtype)


@kernel(OpCode.GEMM, "trans_a", "trans_b", "alpha", "beta")
def gemm(xs, attrs, dtype):
    a, b, bias = xs
    if a.ndim != 2 or b.ndim != 2:
        raise OpError("ShapeError", "GEMM inputs A/B must be matrices")
    if attrs.get("trans_a", False):
        a = a.T
    if attrs.get("trans_b", False):
        b = b.T
    product = np.matmul(a, b)
    bias = np.broadcast_to(bias, product.shape)
    return (
        np.asarray(attrs.get("alpha", 1), dtype=dtype) * product
        + np.asarray(attrs.get("beta", 1), dtype=dtype) * bias
    )


@kernel(OpCode.CONV, "out_channels", "kernel_size", "stride", "padding")
def conv(xs, attrs, dtype):
    x, w, bias = xs
    if x.ndim != 4 or w.ndim != 4 or x.shape[1] != w.shape[1]:
        raise OpError(
            "ShapeError", "CONV requires compatible NCHW data and OIHW weights"
        )
    channels, _, height, width = w.shape
    kernel_size = attrs.get("kernel_size", 3)
    stride, padding = attrs.get("stride", 1), attrs.get("padding", 1)
    if height != width or height != kernel_size or stride < 1 or padding < 0:
        raise OpError("AttributeError", "invalid CONV square kernel/stride/padding")
    if attrs.get("out_channels", channels) != channels or bias.shape != (channels,):
        raise OpError(
            "ShapeError", "CONV output channels or bias disagree with weights"
        )
    out_h = (x.shape[2] + 2 * padding - height) // stride + 1
    out_w = (x.shape[3] + 2 * padding - width) // stride + 1
    if out_h < 1 or out_w < 1:
        raise OpError("ShapeError", "CONV kernel exceeds input")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.empty((x.shape[0], channels, out_h, out_w), dtype=dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = padded[
                :, :, i * stride : i * stride + height, j * stride : j * stride + width
            ]
            output[:, :, i, j] = np.einsum("nchw,ochw->no", patch, w) + bias
    return output


@kernel(OpCode.MAXPOOL, "kernel", "stride")
def maxpool(xs, attrs, dtype):
    x = xs[0]
    if x.ndim not in (3, 4):
        raise OpError("ShapeError", "MAXPOOL requires CHW or NCHW")
    size, stride = attrs["kernel"], attrs["stride"]
    if size < 1 or stride < 1:
        raise OpError("AttributeError", "MAXPOOL kernel/stride must be positive")
    height, width = ((dimension - size) // stride + 1 for dimension in x.shape[-2:])
    if height < 1 or width < 1:
        raise OpError("ShapeError", "MAXPOOL kernel exceeds input")
    output = np.empty(x.shape[:-2] + (height, width), dtype=dtype)
    for i in range(height):
        for j in range(width):
            output[..., i, j] = np.max(
                x[..., i * stride : i * stride + size, j * stride : j * stride + size],
                axis=(-2, -1),
            )
    return output


# ── Standalone-verified operators: Q16.16 on INT32, native on FLOAT32 ──────
# One opcode dispatches on the destination dtype. INT32 carries Q16.16 and is
# bit-exact against the standalone ``_gen_*`` generator; FLOAT32 is native
# IEEE-754 and matches tensor-c / the platform kernels. Q16 is never inferred
# from ``dtype == INT32`` outside these kernels (CSR indices are plain INT32).

_WINO_F23_BT = ((1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1))
_WINO_F23_AT = ((1.0, 1.0, 1.0, 0.0), (0.0, 1.0, -1.0, -1.0))


def _wrap32(v):
    """Low-32-bit wrap, mirroring RV32 add/sub; returns int32."""
    return (((np.asarray(v, np.int64) + (1 << 31)) % (1 << 32)) - (1 << 31)).astype(np.int32)


def _srai16(v):
    """RV32 SRAI: sign-extend the low 32 bits, arithmetic shift right by 16."""
    return (np.asarray(v, np.int64).astype(np.int32) >> 16).astype(np.int32)


@kernel(OpCode.FWHT, "direction")
def fwht(xs, attrs, dtype):
    x = xs[0]
    count = int(x.size)
    if count <= 0 or (count & (count - 1)):
        raise OpError("ShapeError", "FWHT requires a power-of-two element count")
    direction = attrs.get("direction", "forward")
    if direction not in ("forward", "inverse"):
        raise OpError("AttributeError", "FWHT direction must be forward/inverse")
    shift = count.bit_length() - 1

    if np.issubdtype(dtype, np.integer):
        a = x.astype(np.int64).reshape(-1).copy()
        length = 1
        while length < count:
            for i in range(0, count, 2 * length):
                for j in range(length):
                    u, v = a[i + j], a[i + j + length]
                    a[i + j], a[i + j + length] = _wrap32(u + v), _wrap32(u - v)
            length <<= 1
        if direction == "inverse" and shift:
            a >>= shift
        return a.astype(dtype).reshape(x.shape)

    a = x.astype(np.float32).reshape(-1).copy()
    length = 1
    while length < count:
        for i in range(0, count, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = np.float32(u + v), np.float32(u - v)
        length <<= 1
    if direction == "inverse" and shift:
        a = (a / np.float32(count)).astype(np.float32)
    return a.reshape(x.shape)


@kernel(OpCode.SPMM_CSR)
def spmm_csr(xs, attrs, dtype):
    values, col, rowptr, b = xs
    if values.dtype != b.dtype:
        raise OpError("DTypeError", "SPMM_CSR values and B must share a dtype")
    if col.dtype.kind not in "iu" or rowptr.dtype.kind not in "iu":
        raise OpError("DTypeError", "SPMM_CSR col/rowptr must be integer")
    if b.ndim < 1:
        raise OpError("ShapeError", "SPMM_CSR B must be at least 1-D")
    k = int(b.shape[0])
    m = int(rowptr.size) - 1
    n = int(b.shape[1]) if b.ndim >= 2 else 1
    if values.size != col.size:
        raise OpError("ShapeError", "SPMM_CSR values/col length mismatch")
    if rowptr.size != m + 1 or m < 0:
        raise OpError("ShapeError", "SPMM_CSR rowptr must have M+1 entries")
    if np.any(col < 0) or np.any(col >= k):
        raise OpError("IndexError", "SPMM_CSR column index out of range")
    if np.any(rowptr < 0) or np.any(rowptr > values.size):
        raise OpError("IndexError", "SPMM_CSR rowptr out of range")
    b = b.reshape(k, n)
    out = np.zeros((m, n), dtype=dtype)
    integer = np.issubdtype(dtype, np.integer)
    for i in range(m):
        lo, hi = int(rowptr[i]), int(rowptr[i + 1])
        if hi < lo:
            raise OpError("IndexError", "SPMM_CSR rowptr must be nondecreasing")
        for j in range(lo, hi):
            kk = int(col[j])
            if integer:
                a = int(values[j])
                for c in range(n):
                    out[i, c] = _wrap32(int(out[i, c]) + int(_srai16(_wrap32(a * int(b[kk, c])))))
            else:
                out[i, :] += values[j] * b[kk, :]
    return out


@kernel(OpCode.WINOGRAD_CONV, "cout", "cin")
def winograd_conv(xs, attrs, dtype):
    x, u = xs[0], xs[1]
    bias = xs[2] if len(xs) == 3 else None
    if x.ndim != 4 or u.ndim != 4 or tuple(u.shape[2:]) != (4, 4):
        raise OpError("ShapeError", "WINOGRAD_CONV requires NCHW x and [Cout,Cin,4,4] U")
    if x.shape[1] != u.shape[1]:
        raise OpError("ShapeError", "WINOGRAD_CONV Cin disagrees between x and U")
    batch, cin, height, width = (int(v) for v in x.shape)
    cout = int(u.shape[0])
    if attrs.get("cout", 0) and attrs["cout"] != cout:
        raise OpError("AttributeError", "WINOGRAD_CONV cout disagrees with U")
    if attrs.get("cin", 0) and attrs["cin"] != cin:
        raise OpError("AttributeError", "WINOGRAD_CONV cin disagrees with x")
    if bias is not None and bias.shape not in ((), (cout,)):
        raise OpError("ShapeError", "WINOGRAD_CONV bias must be scalar or Cout")
    tiles_h, tiles_w = (height + 1) // 2, (width + 1) // 2
    integer = np.issubdtype(dtype, np.integer)
    numpy_dtype = np.int64 if integer else np.float32
    # The last tile starts at 2*(T-1) and reads four rows/cols, which for odd
    # spatial sizes overruns H+2 by one; a H+3/W+3 zero border covers it exactly.
    padded = np.zeros((batch, cin, height + 3, width + 3), numpy_dtype)
    padded[:, :, 1:1 + height, 1:1 + width] = x
    out = np.empty((batch, cout, height, width), dtype=dtype)
    for nb in range(batch):
        for oc in range(cout):
            for ti in range(tiles_h):
                for tj in range(tiles_w):
                    acc = np.zeros((4, 4), numpy_dtype)
                    for ic in range(cin):
                        d = padded[nb, ic, 2 * ti:2 * ti + 4, 2 * tj:2 * tj + 4]
                        v = np.zeros((4, 4), numpy_dtype)
                        for a in range(4):
                            for bb in range(4):
                                v[a, bb] = sum(
                                    _WINO_F23_BT[a][m] * _WINO_F23_BT[bb][j] * d[m, j]
                                    for m in range(4) for j in range(4))
                        for p in range(4):
                            for q in range(4):
                                if integer:
                                    acc[p, q] += (np.int64(u[oc, ic, p, q]) * np.int64(v[p, q])) >> 16
                                else:
                                    acc[p, q] += u[oc, ic, p, q] * v[p, q]
                    for oy in range(2):
                        for ox in range(2):
                            y = sum(_WINO_F23_AT[oy][a] * _WINO_F23_AT[ox][bb] * acc[a, bb]
                                    for a in range(4) for bb in range(4))
                            if bias is not None:
                                y = y + (bias if bias.ndim == 0 else bias[oc])
                            if 2 * ti + oy < height and 2 * tj + ox < width:
                                out[nb, oc, 2 * ti + oy, 2 * tj + ox] = (
                                    _wrap32(y) if integer else np.float32(y))
    return out
