# Topic 28 extended instruction-selection feature case (deterministic).
#
# The stock DSL frontend (scratchv/frontend/dsl_parser.py) exposes a closed op
# table: add/sub/mul/div/neg/exp/relu/gelu/dot/matmul/softmax/maxpool.  The
# Topic 28 opcodes (sqrt/min/max/abs/idiv/rem/mod and the float64 family) have
# no DSL syntax, so this file drives the CompilerDriver A/B wiring check while
# the extended-only instruction shapes are probed at IR level by
# benchmarks/run_topic28_extended_isel_case.py (see its honesty note).
#
# Loop trip count 0..3 leaves acc = 6, sq = 36, bias = relu(36) = 36.
# neg_acc = -6; total = 36 - (-6) = 42; res = 42 / 6 = 7.
for i = 0, 4
  acc = add(i, i)
  sq = mul(acc, acc)
  bias = relu(sq)
endfor
neg_acc = neg(acc)
total = sub(bias, neg_acc)
res = div(total, acc)
return res
