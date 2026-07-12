#include "internal/vector_ops.cuh"

#include <cuda_runtime.h>

#include <cstdint>

#include "internal/cuda_utils.hpp"

namespace cuda_lsmr::internal {
namespace {

constexpr int kBlockSize = 256;

/// Fuses the two dependent hbar/x recurrences into one global-memory pass.
__global__ void UpdateHbarAndXKernel(float* hbar, const float* h, float* x,
                                    float coefficient, float step,
                                    std::int64_t size) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    const float next_hbar = h[index] - coefficient * hbar[index];
    hbar[index] = next_hbar;
    x[index] = fmaf(step, next_hbar, x[index]);
  }
}

/// Updates the next search direction with one fused multiply-add.
__global__ void UpdateHKernel(float* h, const float* v, float coefficient,
                              std::int64_t size) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) h[index] = fmaf(-coefficient, h[index], v[index]);
}

/// Writes sparse RHS entries; callers guarantee unique valid indices.
__global__ void ScatterSparseKernel(const std::int32_t* indices,
                                    const float* values, std::int64_t nnz,
                                    float* dense, std::int64_t dense_size) {
  const std::int64_t item =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item < nnz) {
    const std::int32_t index = indices[item];
    if (index >= 0 && index < dense_size) dense[index] = values[item];
  }
}

/// Returns the one-dimensional launch size for a vector operation.
int GridSize(std::int64_t count) {
  return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

void FillZero(float* output, std::int64_t size, cudaStream_t stream) {
  if (size > 0) {
    CUDA_LSMR_CHECK_CUDA(cudaMemsetAsync(
        output, 0, static_cast<std::size_t>(size) * sizeof(float), stream));
  }
}

void UpdateHbarAndX(float* hbar, const float* h, float* x, float coefficient,
                    float step, std::int64_t size, cudaStream_t stream) {
  if (size <= 0) return;
  UpdateHbarAndXKernel<<<GridSize(size), kBlockSize, 0, stream>>>(
      hbar, h, x, coefficient, step, size);
  CUDA_LSMR_CHECK_CUDA(cudaGetLastError());
}

void UpdateH(float* h, const float* v, float coefficient, std::int64_t size,
             cudaStream_t stream) {
  if (size <= 0) return;
  UpdateHKernel<<<GridSize(size), kBlockSize, 0, stream>>>(h, v, coefficient,
                                                          size);
  CUDA_LSMR_CHECK_CUDA(cudaGetLastError());
}

void ScatterSparse(const std::int32_t* indices, const float* values,
                   std::int64_t nnz, float* dense, std::int64_t dense_size,
                   cudaStream_t stream) {
  if (nnz <= 0) return;
  ScatterSparseKernel<<<GridSize(nnz), kBlockSize, 0, stream>>>(
      indices, values, nnz, dense, dense_size);
  CUDA_LSMR_CHECK_CUDA(cudaGetLastError());
}

}  // namespace cuda_lsmr::internal
