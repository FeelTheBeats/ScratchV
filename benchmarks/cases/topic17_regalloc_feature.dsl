# Topic 17 register-allocation feature case (linear scan v1.5 + frame layout).
# Deterministic short loop: i = 0..5, t = i*i, k = t + i -> k = 30.
# Deliberately low-pressure: the loop-carried values are force-spilled across
# basic blocks, but no block ever fills the physical register pool, so neither
# reload-time eviction nor high-pressure spilling is exercised.
for i = 0, 6
    t = mul(i, i)
    k = add(t, i)
endfor
return k
