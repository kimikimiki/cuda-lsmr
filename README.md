# cuda-lsmr

`cuda-lsmr` is an FP32 CUDA library for rectangular linear systems and
least-squares problems. It implements the Golub-Kahan/LSMR iteration in C++17
and CUDA, with cuSPARSE for sparse operators and cuBLAS for dense operators and
vector norms.

The library supports:

- CSC or CSR sparse matrices as `A`.
- Dense column-major matrices as `A`.
- Dense or sparse vectors as `b`.
- Multiple dense right-hand sides through `Solver::solve_many`.
- Optional damping, an initial guess, SciPy-compatible tolerances, stop codes,
  and diagnostics.
- Caller-owned CUDA streams and device output storage.
- Reusable, stream-ordered private CUDA memory pools.

## CUDA requirement

The minimum supported and tested version is **CUDA Toolkit 12.9**. This is a
deliberate compatibility baseline tied to the development and benchmark
environment; older CUDA releases have not been validated and are rejected by
CMake.

The initial validation platform is:

- CUDA Toolkit 12.9.86
- NVIDIA driver 581.57
- NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6
- CMake 3.22.1
- GCC 11.4.0

`sm_86` is the default architecture. Set `CMAKE_CUDA_ARCHITECTURES` explicitly
for another GPU supported by CUDA 12.9.

## Design

Every sparse input is materialized twice on the GPU:

- CSR(A) executes `A * v` with contiguous row traversal.
- CSC(A), interpreted as CSR(A^T), executes `A^T * u` as another non-transpose
  SpMV.

This intentionally spends about twice the matrix storage to avoid transpose
scatter behavior in the two bandwidth-bound SpMV calls per iteration. Small
vector recurrences are fused where dependencies allow, and all solver
workspaces grow through one private CUDA memory pool and are reused across
compatible solves.

```text
Public API
  -> Solver recurrence and stopping policy
    -> LinearOperator dispatch
      -> CSR(A)/CSR(A^T) cuSPARSE or dense cuBLAS backend
        -> fused vector kernels and stream-ordered memory runtime
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/ALGORITHM.md](docs/ALGORITHM.md) for the module boundaries and numerical
details.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build and install only the shared C++ library:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_LSMR_BUILD_TESTS=OFF \
  -DCUDA_LSMR_BUILD_EXAMPLES=OFF
cmake --build build -j
cmake --install build --prefix "$HOME/.local"
```

Consumers can then use the exported CMake target:

```cmake
find_package(cuda_lsmr CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE cuda_lsmr::cuda_lsmr)
```

## C++ usage

```cpp
#include <cuda_lsmr/lsmr.hpp>

cuda_lsmr::Context context;
auto A = cuda_lsmr::SparseMatrix::FromCsc(
    context,
    {rows, cols, nnz, csc_offsets, row_indices, values,
     cuda_lsmr::MemoryLocation::kHost});

cuda_lsmr::Solver solver(context);
cuda_lsmr::SolverOptions options;
options.atol = 1.0e-6F;
options.btol = 1.0e-6F;
options.max_iterations = 5000;

// d_x is caller-owned device memory. b may point to host or device FP32 data.
auto result = solver.solve(
    *A, {rows, b, cuda_lsmr::MemoryLocation::kHost}, {cols, d_x}, options);
context.synchronize();
```

The complete runnable version is
[examples/sparse_example.cu](examples/sparse_example.cu).

### Lifetime and data contracts

- `Context` must outlive matrices and solvers created from it.
- Sparse indices are zero-based signed 32-bit integers.
- Sparse-vector indices must be unique and in range.
- Dense matrices and multiple RHS matrices are column-major.
- The iterative state, inputs, accumulation, and output are FP32.
- One matrix/solver/context tuple is not concurrently callable from multiple
  host threads; use one context per concurrent stream.

## Python benchmark binding

The optional pybind11 module exists for CuPy interoperability and benchmark
validation. The public library itself does not depend on Python.

```bash
PYTHON=python3
cmake -S . -B build \
  -DCUDA_LSMR_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR="$($PYTHON -m pybind11 --cmakedir)"
cmake --build build -j
PYTHONPATH=build/python "$PYTHON" benchmarks/benchmark_real_data.py \
  --data-root /path/to/private/benchmark-inputs \
  --datasets dataset-a,dataset-b --frame 0 --repeats 7 --maxiter 5000
```

The binding receives CuPy device pointers for `b` and `x`; MAT loading, sparse
conversion, matrix upload, and warmup are excluded from timed solves.

## Verified FP32 benchmark

Seven synchronized repetitions on the platform above, using two private MAT
inputs, `atol=btol=1e-6`, `conlim=1e8`, and `maxiter=5000`:

| Dataset | Shape | nnz | cuda-lsmr median | CuPy 14.0.1 median | Ratio | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset A | 306136 x 6177 | 5,588,894 | 60.24 ms | 127.28 ms | 0.473 | 86 / 86 |
| Dataset B | 936063 x 23394 | 19,747,202 | 167.11 ms | 229.57 ms | 0.728 | 96 / 96 |

Both implementations returned stop code 2. Relative solution differences were
`1.84e-7` (Dataset A) and `6.26e-7` (Dataset B), while the independently recomputed
relative residuals matched. These numbers describe this GPU and these two
matrices; they are not a general performance claim. Full methodology and raw
samples are recorded in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Limitations

- FP32 real arithmetic and 32-bit sparse indices only.
- No preconditioner or distributed/multi-GPU path.
- Multiple RHS columns currently run independent LSMR recurrences; they do not
  yet use a block-LSMR/SpMM algorithm.
- Dual sparse storage roughly doubles matrix data memory and is unsuitable when
  that tradeoff exceeds available VRAM.
- cuSPARSE's fastest CSR algorithm is not guaranteed bitwise deterministic.

## References

- Fong and Saunders, [LSMR: An iterative algorithm for sparse least-squares problems](https://arxiv.org/abs/1006.0758)
- [CuPy 14.0.0 LSMR source](https://github.com/cupy/cupy/blob/v14.0.0/cupyx/scipy/sparse/linalg/_solve.py)
- [NVIDIA cuSPARSE 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.2/cusparse/index.html)
- [CUB layered reduction design walkthrough](https://dlog.com.cn/posts/cuda09/cub/)

## License

MIT. See [LICENSE](LICENSE).
