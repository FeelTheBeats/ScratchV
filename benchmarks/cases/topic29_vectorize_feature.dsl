# SIMD vectorize feature case (Topic 29, phase 1).
#
# Scalar shape mirrored by the report's vectorizable IR case:
#   out[i] = relu(a[i] + a[i])  for i in [0, 16)
# The phase-1 DSL grammar has no array load/store syntax, so the report
# builds the canonical element-addressing IR (design doc appendix 5.1)
# and compiles it through CompilerDriver; this file is still read and
# validated by every driver call and compiled off/on in the wiring check.
for i = 0, 16
  t = add(x, x)
  y = relu(t)
endfor
return y
