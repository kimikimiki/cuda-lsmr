# Architecture

The project separates numerical policy, matrix storage, CUDA execution, and
language interoperability. Dependencies point toward lower layers only.

```text
include/cuda_lsmr
  Context, views, LinearOperator, Solver, options/results
                    |
src/solver          | knows dimensions and A*x / A^T*x only
                    v
src/operators       SparseMatrix dispatch        DenseMatrix dispatch
                    |                             |
                    v                             v
                 cuSPARSE                      cuBLAS GEMV
                    \                             /
                     src/runtime + src/kernels
                     stream, handles, pool, fused vectors
```

## Public API

The public layer contains non-owning views with explicit dimensions and memory
locations. Owning CUDA classes use PImpl so cuSPARSE descriptors, cuBLAS handles,
and allocation policy do not leak into user headers.

`LinearOperator` is the solver's only matrix dependency. It provides scaled
forms of `A*x` and `A^T*x`, allowing Golub-Kahan updates to use the output
vector's old value as the BLAS beta term.

## Sparse operator

`SparseMatrix::FromCsr` copies CSR(A), then calls `cusparseCsr2cscEx2` once to
construct CSC(A). `FromCsc` treats CSC(A) as CSR(A^T), converts that object, and
obtains CSR(A). Both paths finish with these descriptors:

- `m x n` CSR(A) for `A*v`.
- `n x m` CSR(A^T), backed by CSC(A), for `A^T*u`.

The operator caches dense-vector descriptors and the maximum cuSPARSE SpMV
workspace. It mutates descriptor pointers at each call, so one operator is not
concurrently callable from multiple host threads.

## Dense operator

`DenseMatrix` owns a column-major FP32 device copy. Normal and transpose paths
map directly to `cublasSgemv` with `CUBLAS_OP_N` and `CUBLAS_OP_T`.

## Runtime and memory

Each `Context` owns:

- one CUDA stream, unless it wraps a caller-owned stream;
- one cuBLAS handle and one cuSPARSE handle bound to that stream;
- one private stream-ordered CUDA memory pool.

`DeviceBuffer<T>` allocates through `cudaMallocFromPoolAsync` and frees through
`cudaFreeAsync`. Sparse storage, conversion/SpMV workspaces, LSMR vectors, and
sparse-RHS staging all use this path. A resize reuses capacity whenever possible.
There are no allocations in the iteration loop after shape warmup.

## Solver

The solver owns `u`, `v`, `h`, `hbar`, and RHS staging buffers. Scalars required
by stopping decisions return to the host through cuBLAS norm calls; matrix and
vector state remains on the device. `hbar` and `x` share one fused global-memory
pass because the latter depends on the newly computed former.

Sparse RHS values are scattered once into a zeroed reusable dense buffer. Dense
host RHS values are copied once. Multiple RHS columns reuse the same workspace
and run sequential independent solves.

## Invariants

1. `Context` outlives every matrix and solver created from it.
2. The LSMR recurrence never inspects a matrix storage format.
3. Sparse A always owns CSR(A) and CSC(A).
4. Public pointer views state their memory location.
5. All iterative data and accumulation are FP32.
6. Host synchronization occurs only when a scalar decision or public result
   requires it.

