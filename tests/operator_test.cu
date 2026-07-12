#include "cuda_lsmr/lsmr.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

/// Verifies that two small host vectors agree within FP32 rounding error.
void ExpectNear(const float* actual, const float* expected, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (std::fabs(actual[i] - expected[i]) >= 1.0e-5F) {
      std::ostringstream message;
      message << "operator result mismatch at " << i << ": got " << actual[i]
              << ", expected " << expected[i];
      throw std::runtime_error(message.str());
    }
  }
}

/// Throws when a CUDA Runtime test operation fails.
void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/// Runs A*x and A^T*y checks through one operator implementation.
void CheckOperator(cuda_lsmr::Context& context,
                   const cuda_lsmr::LinearOperator& matrix) {
  const std::array<float, 2> x{2.0F, -1.0F};
  const std::array<float, 3> u{1.0F, 2.0F, -1.0F};
  float* d_x{};
  float* d_u{};
  float* d_ax{};
  float* d_atu{};
  CheckCuda(cudaMalloc(&d_x, sizeof(x)));
  CheckCuda(cudaMalloc(&d_u, sizeof(u)));
  CheckCuda(cudaMalloc(&d_ax, 3 * sizeof(float)));
  CheckCuda(cudaMalloc(&d_atu, 2 * sizeof(float)));
  CheckCuda(cudaMemcpy(d_x, x.data(), sizeof(x), cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(d_u, u.data(), sizeof(u), cudaMemcpyHostToDevice));

  matrix.apply(d_x, d_ax);
  matrix.apply_transpose(d_u, d_atu);
  context.synchronize();

  std::array<float, 3> ax{};
  std::array<float, 2> atu{};
  CheckCuda(cudaMemcpy(ax.data(), d_ax, sizeof(ax), cudaMemcpyDeviceToHost));
  CheckCuda(cudaMemcpy(atu.data(), d_atu, sizeof(atu), cudaMemcpyDeviceToHost));
  const std::array<float, 3> expected_ax{2.0F, 4.0F, 6.0F};
  const std::array<float, 2> expected_atu{-2.0F, -8.0F};
  ExpectNear(ax.data(), expected_ax.data(), ax.size());
  ExpectNear(atu.data(), expected_atu.data(), atu.size());

  CheckCuda(cudaFree(d_x));
  CheckCuda(cudaFree(d_u));
  CheckCuda(cudaFree(d_ax));
  CheckCuda(cudaFree(d_atu));
}

}  // namespace

int main() {
  cuda_lsmr::Context context;
  // A = [[1, 0], [0, -4], [3, 0]].
  const std::array<std::int32_t, 4> csr_offsets{0, 1, 2, 3};
  const std::array<std::int32_t, 3> csr_indices{0, 1, 0};
  const std::array<float, 3> csr_values{1.0F, -4.0F, 3.0F};
  const std::array<std::int32_t, 3> csc_offsets{0, 2, 3};
  const std::array<std::int32_t, 3> csc_indices{0, 2, 1};
  const std::array<float, 3> csc_values{1.0F, 3.0F, -4.0F};
  const std::array<float, 6> dense_values{1.0F, 0.0F, 3.0F,
                                          0.0F, -4.0F, 0.0F};

  auto from_csr = cuda_lsmr::SparseMatrix::FromCsr(
      context, {3, 2, 3, csr_offsets.data(), csr_indices.data(),
                csr_values.data(), cuda_lsmr::MemoryLocation::kHost});
  auto from_csc = cuda_lsmr::SparseMatrix::FromCsc(
      context, {3, 2, 3, csc_offsets.data(), csc_indices.data(),
                csc_values.data(), cuda_lsmr::MemoryLocation::kHost});
  auto dense = cuda_lsmr::DenseMatrix::Create(
      context, {3, 2, 3, dense_values.data(), cuda_lsmr::MemoryLocation::kHost});

  CheckOperator(context, *from_csr);
  CheckOperator(context, *from_csc);
  CheckOperator(context, *dense);
  if (from_csr->nnz() != 3 || context.pool_reserved_bytes() == 0) {
    throw std::runtime_error("operator metadata or memory pool check failed");
  }
  return 0;
}
