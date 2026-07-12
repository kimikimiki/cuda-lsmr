#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cuda_lsmr::internal {

/// Sets a device vector to zero asynchronously.
void FillZero(float* output, std::int64_t size, cudaStream_t stream);

/// Updates `hbar = h - coefficient*hbar` and `x += step*hbar` in one pass.
void UpdateHbarAndX(float* hbar, const float* h, float* x, float coefficient,
                    float step, std::int64_t size, cudaStream_t stream);

/// Updates `h = v - coefficient*h` in one pass.
void UpdateH(float* h, const float* v, float coefficient, std::int64_t size,
             cudaStream_t stream);

/// Scatters unique sparse entries into a zeroed dense device vector.
void ScatterSparse(const std::int32_t* indices, const float* values,
                   std::int64_t nnz, float* dense, std::int64_t dense_size,
                   cudaStream_t stream);

}  // namespace cuda_lsmr::internal
