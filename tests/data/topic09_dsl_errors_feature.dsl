# Topic 09 DSL-error feature case (intentionally invalid).
# Used by benchmarks/run_topic09_errors_case.py; CI asserts the exact
# diagnostics below, so do not "fix" this file.
a = add(1, 2)
b = retrun(a, 1)
c = add(a)
if (a > b):
  d = mul(a, b)
endwhile
return d
