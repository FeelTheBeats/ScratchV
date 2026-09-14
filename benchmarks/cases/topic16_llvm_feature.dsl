# Topic 16 LLVM-codegen feature case.
#
# Deterministic operator chain covering five NN operators (relu, dot,
# matmul, softmax, gelu).  The tensor operators are deliberately
# scalar-degenerate (length/all dims = 1): the DSL frontend does not
# materialize operand shapes yet, so larger extents would be rejected by the
# LLVM backend's element-count checks.  The canonical lowering structure
# (loop skeleton, getelementptr, MAC fmul/fadd, softmax three passes, gelu
# tanhf call) is emitted regardless of the trip count, and the case report
# asserts those markers.
#
# This case is compiled by the real CompilerDriver with backend="llvm"; it is
# a structural/feature case, not a performance workload.
y = relu(x)
d = dot(a, b, len:1)
c = matmul(m1, m2, m:1, n:1, k:1)
s = softmax(v)
g = gelu(z)
p1 = add(y, d)
p2 = add(c, s)
p3 = add(p2, g)
t = add(p1, p3)
return t
