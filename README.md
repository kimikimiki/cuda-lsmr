# cuda-lsmr

`cuda-lsmr` is a structured FP32 CUDA library for solving rectangular linear
systems and least-squares problems with the LSMR algorithm. The implementation
keeps the iterative path on the GPU and uses CUDA, cuBLAS, and cuSPARSE directly.

## Compatibility baseline

The minimum supported and tested toolkit is **CUDA Toolkit 12.9**. This is the
toolkit installed in the development environment; older CUDA releases are not
claimed to work. The initial validation platform is:

- CUDA Toolkit 12.9.86
- NVIDIA driver 581.57
- NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6
- CMake 3.22.1
- GCC-compatible C++17 host compiler

The default CMake target is `sm_86`. Override `CMAKE_CUDA_ARCHITECTURES` when
building for another GPU supported by CUDA 12.9.

## Architecture

The project follows an API-to-kernel layering similar to mature CUDA libraries:

```text
include/cuda_lsmr/        Stable C++ API and non-owning data views
src/solver/               LSMR recurrence and stopping policy
src/operators/            Sparse and dense A*x / A^T*x backends
src/runtime/              CUDA stream, handles, memory pool, RAII buffers
src/kernels/              Small vector and sparse-RHS CUDA kernels
bindings/                 Optional Python benchmark/interoperability module
tests/                    C++ correctness and API tests
benchmarks/               Reproducible CuPy comparison harness
```

Sparse matrices are always materialized in both forms:

- CSR(A) serves `A * v` with contiguous row traversal.
- CSC(A), interpreted as CSR(A^T), serves `A^T * u` without a transpose SpMV.

This doubles matrix storage by design to improve useful memory bandwidth in the
two SpMV operations executed by every LSMR iteration.

## Status

The repository is being implemented in auditable stages. Build and usage
instructions will be completed with the solver and benchmark harness.

## References

- Fong and Saunders, [LSMR: An iterative algorithm for sparse least-squares problems](https://arxiv.org/abs/1006.0758)
- [CuPy 14 LSMR documentation](https://docs.cupy.dev/en/v14.0.0/reference/generated/cupyx.scipy.sparse.linalg.lsmr.html)
- [NVIDIA cuSPARSE 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.2/cusparse/index.html)
- [CUB layered reduction design walkthrough](https://dlog.com.cn/posts/cuda09/cub/)

