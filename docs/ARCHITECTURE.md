# Architecture

The dependency direction is intentionally one-way:

```text
public API -> solver -> linear operator -> CUDA library/kernel backend
                         runtime context -> stream-ordered memory pool
```

## Invariants

1. LSMR only depends on `LinearOperator`, never on CSR, CSC, or cuSPARSE.
2. A sparse operator owns both CSR(A) and CSC(A) for non-transpose SpMV paths.
3. Temporary device storage is allocated through one context memory pool.
4. Public views state dimensions, pointer ownership, and memory location.
5. Algorithm state is FP32; scalar host decisions are synchronized explicitly.
6. Tests cover each backend through the same solver interface.

## Dispatch boundary

`SparseMatrix` and `DenseMatrix` implement the same two operations. Future
shape or architecture tuning remains private to those operator classes, so the
solver recurrence and public API do not change when a backend is replaced.

