# Loop unrolling feature case (Topic 10).
# Low-pressure loop body: the greedy allocator is known-correct for this
# shape; the report additionally runs an equivalent IR-level program under
# the RV32 emulator to compare architectural state before and after unroll.
for i = 0, 4
  t = add(i, one)
  acc = add(acc, t)
endfor
return acc
