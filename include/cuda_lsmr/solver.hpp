#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_lsmr/types.hpp"

namespace cuda_lsmr {

class Context;
class LinearOperator;

/// Configures damping, stopping tolerances, and the iteration bound.
struct SolverOptions {
  float damp{0.0F};
  float atol{1.0e-6F};
  float btol{1.0e-6F};
  float conlim{1.0e8F};
  std::int32_t max_iterations{0};
};

/// Summarizes one LSMR solve using SciPy-compatible stop codes.
struct SolverResult {
  std::int32_t stop_code{};
  std::int32_t iterations{};
  float norm_r{};
  float norm_ar{};
  float norm_a{};
  float cond_a{};
  float norm_x{};
};

/// Reuses all LSMR vector workspaces across solves of compatible shapes.
class Solver {
 public:
  /// Binds a solver workspace to a CUDA context.
  explicit Solver(Context& context);
  ~Solver();
  Solver(Solver&&) noexcept;
  Solver& operator=(Solver&&) noexcept;
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  /// Solves with a dense host or device RHS into a device output vector.
  SolverResult solve(const LinearOperator& matrix, DenseVectorView rhs,
                     MutableDenseVectorView solution,
                     const SolverOptions& options = {},
                     const DenseVectorView* initial_guess = nullptr);

  /// Densifies a sparse RHS in pooled device memory and solves the system.
  SolverResult solve(const LinearOperator& matrix, SparseVectorView rhs,
                     MutableDenseVectorView solution,
                     const SolverOptions& options = {},
                     const DenseVectorView* initial_guess = nullptr);

  /// Solves independent dense right-hand sides stored as matrix columns.
  std::vector<SolverResult> solve_many(
      const LinearOperator& matrix, DenseMatrixView rhs,
      MutableDenseMatrixView solution, const SolverOptions& options = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuda_lsmr

