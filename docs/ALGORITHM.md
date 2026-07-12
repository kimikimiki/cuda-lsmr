# LSMR implementation notes

The library solves the damped least-squares problem

```text
minimize || [ b ] - [ A      ] x ||_2.
         || [ 0 ]   [ damp I ]   ||
```

LSMR applies MINRES-like rotations to the Golub-Kahan bidiagonalization. Each
iteration computes one forward and one transpose matrix-vector product:

```text
u_(k+1) = (A v_k - alpha_k u_k) / beta_(k+1)
v_(k+1) = (A^T u_(k+1) - beta_(k+1) v_k) / alpha_(k+1)
```

The two numerators are dispatched as scaled BLAS operations. For sparse A this
means non-transpose CSR(A) and non-transpose CSR(A^T) cuSPARSE calls. For dense A
they become normal and transposed cuBLAS GEMV calls.

## Numerical state

Device vectors and matrix values are FP32. Rotation and stopping-rule scalars
are evaluated as host doubles after FP32 norms are returned. Coefficients are
cast to FP32 when applied to device vectors. This matches the practical CuPy
FP32 path closely while preserving stable scalar `hypot`-style rotations.

The implementation returns the standard eight diagnostics:

- solution vector through caller-owned device storage;
- stop code and iteration count;
- estimates of `||r||`, `||A^T r||`, `||A||`, `cond(A)`, and `||x||`.

Stop codes follow SciPy/CuPy conventions. The default maximum iteration count is
`5 * min(m, n)` when the caller passes zero.

## Fused updates

The search-direction recurrences include

```text
hbar <- h - coefficient * hbar
x    <- x + step * hbar
h    <- v - theta_over_rho * h
```

The first two equations are fused because the new `x` uses the new `hbar` at the
same index. The final `h` update is a second independent kernel. Matrix products
reuse `u` and `v` in place through the BLAS beta parameter, eliminating another
pair of temporary vectors.

## Correctness policy

Tests cover exact small systems with CSC, CSR, dense A, dense b, sparse b, and
multiple RHS columns. Real-data validation compares iterations, stop codes,
solutions, and independently recomputed residuals with CuPy 14.0.1. CUDA memory
access is checked with Compute Sanitizer.

