#include "cuda_lsmr/lsmr.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

/// Throws when a CUDA Runtime test operation fails.
void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/// Checks one two-element solution against its expected FP32 values.
void CheckSolution(const std::array<float, 2>& actual,
                   const std::array<float, 2>& expected) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 2.0e-4F) {
      std::ostringstream message;
      message << "solution mismatch at " << i << ": got " << actual[i]
              << ", expected " << expected[i];
      throw std::runtime_error(message.str());
    }
  }
}

/// Solves one RHS and copies its device output back for verification.
std::array<float, 2> SolveDense(cuda_lsmr::Context& context,
                                cuda_lsmr::Solver& solver,
                                const cuda_lsmr::LinearOperator& matrix,
                                const std::array<float, 3>& rhs) {
  float* output{};
  CheckCuda(cudaMalloc(&output, 2 * sizeof(float)));
  const auto result = solver.solve(
      matrix, {3, rhs.data(), cuda_lsmr::MemoryLocation::kHost}, {2, output});
  if (result.stop_code <= 0 || result.iterations <= 0) {
    throw std::runtime_error("LSMR did not report convergence");
  }
  context.synchronize();
  std::array<float, 2> host{};
  CheckCuda(cudaMemcpy(host.data(), output, sizeof(host), cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(output));
  return host;
}

}  // namespace

int main() {
  cuda_lsmr::Context context;
  cuda_lsmr::Solver solver(context);
  // A = [[1, 0], [0, -4], [3, 0]].
  const std::array<std::int32_t, 3> csc_offsets{0, 2, 3};
  const std::array<std::int32_t, 3> csc_indices{0, 2, 1};
  const std::array<float, 3> csc_values{1.0F, 3.0F, -4.0F};
  const std::array<float, 6> dense_values{1.0F, 0.0F, 3.0F,
                                          0.0F, -4.0F, 0.0F};
  auto sparse = cuda_lsmr::SparseMatrix::FromCsc(
      context, {3, 2, 3, csc_offsets.data(), csc_indices.data(),
                csc_values.data(), cuda_lsmr::MemoryLocation::kHost});
  auto dense = cuda_lsmr::DenseMatrix::Create(
      context, {3, 2, 3, dense_values.data(), cuda_lsmr::MemoryLocation::kHost});

  const std::array<float, 3> rhs{2.0F, 4.0F, 6.0F};
  const std::array<float, 2> expected{2.0F, -1.0F};
  CheckSolution(SolveDense(context, solver, *sparse, rhs), expected);
  CheckSolution(SolveDense(context, solver, *dense, rhs), expected);

  float* sparse_output{};
  CheckCuda(cudaMalloc(&sparse_output, 2 * sizeof(float)));
  const std::array<std::int32_t, 2> rhs_indices{0, 2};
  const std::array<float, 2> rhs_values{2.0F, 6.0F};
  const auto sparse_result = solver.solve(
      *sparse,
      {3, 2, rhs_indices.data(), rhs_values.data(),
       cuda_lsmr::MemoryLocation::kHost},
      {2, sparse_output});
  if (sparse_result.stop_code <= 0) {
    throw std::runtime_error("sparse RHS solve did not converge");
  }
  context.synchronize();
  std::array<float, 2> sparse_solution{};
  CheckCuda(cudaMemcpy(sparse_solution.data(), sparse_output,
                       sizeof(sparse_solution), cudaMemcpyDeviceToHost));
  CheckSolution(sparse_solution, {2.0F, 0.0F});
  CheckCuda(cudaFree(sparse_output));

  float* batch_output{};
  CheckCuda(cudaMalloc(&batch_output, 4 * sizeof(float)));
  const std::array<float, 6> batch_rhs{2.0F, 4.0F, 6.0F,
                                       4.0F, 8.0F, 12.0F};
  const auto batch_results = solver.solve_many(
      *sparse,
      {3, 2, 3, batch_rhs.data(), cuda_lsmr::MemoryLocation::kHost},
      {2, 2, 2, batch_output});
  if (batch_results.size() != 2) throw std::runtime_error("batch result count");
  context.synchronize();
  std::array<float, 4> batch_solution{};
  CheckCuda(cudaMemcpy(batch_solution.data(), batch_output,
                       sizeof(batch_solution), cudaMemcpyDeviceToHost));
  CheckSolution({batch_solution[0], batch_solution[1]}, expected);
  CheckSolution({batch_solution[2], batch_solution[3]}, {4.0F, -2.0F});
  CheckCuda(cudaFree(batch_output));
  return 0;
}
