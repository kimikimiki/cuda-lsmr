#include "cuda_lsmr/lsmr.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

/// Throws when an example CUDA Runtime operation fails.
void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace

int main() {
  // A = [[1, 0], [0, -4], [3, 0]] in zero-based CSC storage.
  const std::array<std::int32_t, 3> offsets{0, 2, 3};
  const std::array<std::int32_t, 3> indices{0, 2, 1};
  const std::array<float, 3> values{1.0F, 3.0F, -4.0F};
  const std::array<float, 3> rhs{2.0F, 4.0F, 6.0F};

  cuda_lsmr::Context context;
  auto matrix = cuda_lsmr::SparseMatrix::FromCsc(
      context, {3, 2, 3, offsets.data(), indices.data(), values.data(),
                cuda_lsmr::MemoryLocation::kHost});
  cuda_lsmr::Solver solver(context);

  float* device_solution{};
  CheckCuda(cudaMalloc(&device_solution, 2 * sizeof(float)));
  const cuda_lsmr::SolverResult result = solver.solve(
      *matrix, {3, rhs.data(), cuda_lsmr::MemoryLocation::kHost},
      {2, device_solution});
  context.synchronize();

  std::array<float, 2> solution{};
  CheckCuda(cudaMemcpy(solution.data(), device_solution, sizeof(solution),
                       cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(device_solution));
  std::cout << "x = [" << solution[0] << ", " << solution[1]
            << "], iterations = " << result.iterations
            << ", stop = " << result.stop_code << '\n';
  return 0;
}

