# Structured-logging feature case (Topic 07).
# Deterministic loop + arithmetic chain.  The case is deliberately small:
# parse / optimize / codegen / asm / emit all run on it, which is all the
# report needs to prove the staged log record and the byte-identical output.
for i = 0, 4
  acc = add(acc, x)
endfor
t = mul(acc, x)
return t
