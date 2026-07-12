#include "cuda_lsmr/solver.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cuda_lsmr/context.hpp"
#include "cuda_lsmr/error.hpp"
#include "cuda_lsmr/matrix.hpp"
#include "internal/context_impl.hpp"
#include "internal/cuda_utils.hpp"
#include "internal/device_buffer.hpp"
#include "internal/vector_ops.cuh"

namespace cuda_lsmr {
namespace {

/// Computes a stable symmetric Givens rotation.
void SymOrtho(double a, double b, double& c, double& s, double& r) {
  if (b == 0.0) {
    c = std::copysign(1.0, a);
    s = 0.0;
    r = std::abs(a);
  } else if (a == 0.0) {
    c = 0.0;
    s = std::copysign(1.0, b);
    r = std::abs(b);
  } else if (std::abs(b) > std::abs(a)) {
    const double tau = a / b;
    s = std::copysign(1.0 / std::sqrt(1.0 + tau * tau), b);
    c = s * tau;
    r = b / s;
  } else {
    const double tau = b / a;
    c = std::copysign(1.0 / std::sqrt(1.0 + tau * tau), a);
    s = c * tau;
    r = a / c;
  }
}

/// Returns a CUDA copy kind for an input view.
cudaMemcpyKind CopyKind(MemoryLocation location) {
  return location == MemoryLocation::kHost ? cudaMemcpyHostToDevice
                                           : cudaMemcpyDeviceToDevice;
}

}  // namespace

struct Solver::Impl {
  explicit Impl(Context& owner) : context(owner) {}

  /// Grows reusable vectors for one `(m,n)` operator shape.
  void Reserve(std::int64_t m, std::int64_t n) {
    u.Resize(static_cast<std::size_t>(m), context.stream(), context.impl_->pool);
    dense_rhs.Resize(static_cast<std::size_t>(m), context.stream(),
                     context.impl_->pool);
    v.Resize(static_cast<std::size_t>(n), context.stream(), context.impl_->pool);
    h.Resize(static_cast<std::size_t>(n), context.stream(), context.impl_->pool);
    hbar.Resize(static_cast<std::size_t>(n), context.stream(), context.impl_->pool);
  }

  /// Copies one vector through the context cuBLAS handle.
  void Copy(std::int64_t size, const float* source, float* destination) const {
    CUDA_LSMR_CHECK_CUBLAS(cublasScopy(context.impl_->cublas,
                                       static_cast<int>(size), source, 1,
                                       destination, 1));
  }

  /// Returns the FP32 Euclidean norm and synchronizes the host scalar decision.
  float Norm(std::int64_t size, const float* values) const {
    float result{};
    CUDA_LSMR_CHECK_CUBLAS(cublasSnrm2(context.impl_->cublas,
                                       static_cast<int>(size), values, 1,
                                       &result));
    return result;
  }

  /// Scales one device vector in place.
  void Scale(std::int64_t size, float factor, float* values) const {
    CUDA_LSMR_CHECK_CUBLAS(cublasSscal(context.impl_->cublas,
                                       static_cast<int>(size), &factor, values,
                                       1));
  }

  /// Executes LSMR after the RHS has been made available on the device.
  SolverResult SolveDevice(const LinearOperator& matrix, const float* rhs,
                           MutableDenseVectorView solution,
                           const SolverOptions& options,
                           const DenseVectorView* initial_guess) {
    const std::int64_t m = matrix.rows();
    const std::int64_t n = matrix.cols();
    if (m <= 0 || n <= 0 || m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max()) {
      throw Error("LSMR requires positive dimensions supported by cuBLAS int32 APIs");
    }
    if (solution.size != n || solution.values == nullptr || rhs == nullptr) {
      throw Error("LSMR RHS or output dimensions are incompatible with A");
    }
    if (options.damp < 0.0F || options.atol < 0.0F || options.btol < 0.0F ||
        options.conlim < 0.0F || options.max_iterations < 0) {
      throw Error("LSMR options must be non-negative");
    }
    Reserve(m, n);

    Copy(m, rhs, u.data());
    const double norm_b = static_cast<double>(Norm(m, rhs));
    if (initial_guess == nullptr) {
      internal::FillZero(solution.values, n, context.stream());
    } else {
      if (initial_guess->size != n || initial_guess->values == nullptr) {
        throw Error("LSMR initial guess dimensions are incompatible with A");
      }
      CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
          solution.values, initial_guess->values,
          static_cast<std::size_t>(n) * sizeof(float),
          CopyKind(initial_guess->location), context.stream()));
      matrix.apply_scaled(-1.0F, solution.values, 1.0F, u.data());
    }

    double beta = static_cast<double>(Norm(m, u.data()));
    double alpha = 0.0;
    if (beta > 0.0) {
      Scale(m, static_cast<float>(1.0 / beta), u.data());
      matrix.apply_transpose(u.data(), v.data());
      alpha = static_cast<double>(Norm(n, v.data()));
    } else {
      internal::FillZero(v.data(), n, context.stream());
    }
    if (alpha > 0.0) Scale(n, static_cast<float>(1.0 / alpha), v.data());

    std::int32_t iterations = 0;
    double zetabar = alpha * beta;
    double alphabar = alpha;
    double rho = 1.0;
    double rhobar = 1.0;
    double cbar = 1.0;
    double sbar = 0.0;
    Copy(n, v.data(), h.data());
    internal::FillZero(hbar.data(), n, context.stream());

    double betadd = beta;
    double betad = 0.0;
    double rhodold = 1.0;
    double tautildeold = 0.0;
    double thetatilde = 0.0;
    double zeta = 0.0;
    double d = 0.0;
    double norm_a_squared = alpha * alpha;
    double max_rbar = 0.0;
    double min_rbar = std::numeric_limits<double>::max();
    double norm_a = alpha;
    double cond_a = 1.0;
    double norm_x = 0.0;
    std::int32_t stop_code = 0;
    const double ctol = options.conlim > 0.0F ? 1.0 / options.conlim : 0.0;
    double norm_r = beta;
    double norm_ar = alpha * beta;
    if (norm_ar == 0.0) {
      return {stop_code, iterations, static_cast<float>(norm_r),
              static_cast<float>(norm_ar), static_cast<float>(norm_a),
              static_cast<float>(cond_a), static_cast<float>(norm_x)};
    }

    const std::int64_t default_limit =
        std::min<std::int64_t>(5 * std::min(m, n),
                               std::numeric_limits<std::int32_t>::max());
    const std::int32_t max_iterations =
        options.max_iterations > 0 ? options.max_iterations
                                   : static_cast<std::int32_t>(default_limit);

    while (iterations < max_iterations) {
      ++iterations;
      matrix.apply_scaled(1.0F, v.data(), static_cast<float>(-alpha), u.data());
      beta = static_cast<double>(Norm(m, u.data()));
      if (beta > 0.0) {
        Scale(m, static_cast<float>(1.0 / beta), u.data());
        matrix.apply_transpose_scaled(1.0F, u.data(), static_cast<float>(-beta),
                                      v.data());
        alpha = static_cast<double>(Norm(n, v.data()));
        if (alpha > 0.0) Scale(n, static_cast<float>(1.0 / alpha), v.data());
      }

      double chat{}, shat{}, alphahat{};
      SymOrtho(alphabar, options.damp, chat, shat, alphahat);
      const double rho_old = rho;
      double c{}, s{};
      SymOrtho(alphahat, beta, c, s, rho);
      const double theta_new = s * alpha;
      alphabar = c * alpha;

      const double rhobar_old = rhobar;
      const double zeta_old = zeta;
      const double theta_bar = sbar * rho;
      const double rho_temp = cbar * rho;
      SymOrtho(cbar * rho, theta_new, cbar, sbar, rhobar);
      zeta = cbar * zetabar;
      zetabar = -sbar * zetabar;

      const double hbar_coefficient =
          theta_bar * rho / (rho_old * rhobar_old);
      const double x_step = zeta / (rho * rhobar);
      internal::UpdateHbarAndX(
          hbar.data(), h.data(), solution.values,
          static_cast<float>(hbar_coefficient), static_cast<float>(x_step), n,
          context.stream());
      internal::UpdateH(h.data(), v.data(), static_cast<float>(theta_new / rho),
                        n, context.stream());

      const double beta_acute = chat * betadd;
      const double beta_check = -shat * betadd;
      const double beta_hat = c * beta_acute;
      betadd = -s * beta_acute;
      const double theta_tilde_old = thetatilde;
      double c_tilde_old{}, s_tilde_old{}, rho_tilde_old{};
      SymOrtho(rhodold, theta_bar, c_tilde_old, s_tilde_old, rho_tilde_old);
      thetatilde = s_tilde_old * rhobar;
      rhodold = c_tilde_old * rhobar;
      betad = -s_tilde_old * betad + c_tilde_old * beta_hat;
      tautildeold =
          (zeta_old - theta_tilde_old * tautildeold) / rho_tilde_old;
      const double tau_d = (zeta - thetatilde * tautildeold) / rhodold;
      d += beta_check * beta_check;
      norm_r = std::sqrt(d + (betad - tau_d) * (betad - tau_d) +
                         betadd * betadd);

      norm_a_squared += beta * beta;
      norm_a = std::sqrt(norm_a_squared);
      norm_a_squared += alpha * alpha;
      max_rbar = std::max(max_rbar, rhobar_old);
      if (iterations > 1) min_rbar = std::min(min_rbar, rhobar_old);
      cond_a = std::max(max_rbar, rho_temp) / std::min(min_rbar, rho_temp);
      norm_ar = std::abs(zetabar);
      norm_x = static_cast<double>(Norm(n, solution.values));

      const double test1 = norm_b > 0.0 ? norm_r / norm_b : 0.0;
      const double test2 = norm_a * norm_r != 0.0
                               ? norm_ar / (norm_a * norm_r)
                               : std::numeric_limits<double>::infinity();
      const double test3 = 1.0 / cond_a;
      const double scaled_x = norm_b > 0.0 ? norm_a * norm_x / norm_b : 0.0;
      const double t1 = test1 / (1.0 + scaled_x);
      const double rtol = options.btol + options.atol * scaled_x;

      if (iterations >= max_iterations) stop_code = 7;
      if (1.0 + test3 <= 1.0) stop_code = 6;
      if (1.0 + test2 <= 1.0) stop_code = 5;
      if (1.0 + t1 <= 1.0) stop_code = 4;
      if (test3 <= ctol) stop_code = 3;
      if (test2 <= options.atol) stop_code = 2;
      if (test1 <= rtol) stop_code = 1;
      if (stop_code > 0) break;
    }

    return {stop_code, iterations, static_cast<float>(norm_r),
            static_cast<float>(norm_ar), static_cast<float>(norm_a),
            static_cast<float>(cond_a), static_cast<float>(norm_x)};
  }

  Context& context;
  internal::DeviceBuffer<float> u;
  internal::DeviceBuffer<float> v;
  internal::DeviceBuffer<float> h;
  internal::DeviceBuffer<float> hbar;
  internal::DeviceBuffer<float> dense_rhs;
  internal::DeviceBuffer<std::int32_t> sparse_indices;
  internal::DeviceBuffer<float> sparse_values;
};

Solver::Solver(Context& context) : impl_(std::make_unique<Impl>(context)) {}
Solver::~Solver() = default;
Solver::Solver(Solver&&) noexcept = default;
Solver& Solver::operator=(Solver&&) noexcept = default;

SolverResult Solver::solve(const LinearOperator& matrix, DenseVectorView rhs,
                           MutableDenseVectorView solution,
                           const SolverOptions& options,
                           const DenseVectorView* initial_guess) {
  if (rhs.size != matrix.rows() || rhs.values == nullptr) {
    throw Error("Dense RHS dimensions are incompatible with A");
  }
  const float* device_rhs = rhs.values;
  if (rhs.location == MemoryLocation::kHost) {
    impl_->dense_rhs.Resize(static_cast<std::size_t>(rhs.size),
                            impl_->context.stream(), impl_->context.impl_->pool);
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl_->dense_rhs.data(), rhs.values,
        static_cast<std::size_t>(rhs.size) * sizeof(float),
        cudaMemcpyHostToDevice, impl_->context.stream()));
    device_rhs = impl_->dense_rhs.data();
  }
  return impl_->SolveDevice(matrix, device_rhs, solution, options, initial_guess);
}

SolverResult Solver::solve(const LinearOperator& matrix, SparseVectorView rhs,
                           MutableDenseVectorView solution,
                           const SolverOptions& options,
                           const DenseVectorView* initial_guess) {
  if (rhs.size != matrix.rows() || rhs.nnz < 0 || rhs.nnz > rhs.size ||
      (rhs.nnz > 0 && (rhs.indices == nullptr || rhs.values == nullptr))) {
    throw Error("Sparse RHS dimensions or pointers are invalid");
  }
  impl_->dense_rhs.Resize(static_cast<std::size_t>(rhs.size),
                          impl_->context.stream(), impl_->context.impl_->pool);
  internal::FillZero(impl_->dense_rhs.data(), rhs.size, impl_->context.stream());
  const std::int32_t* indices = rhs.indices;
  const float* values = rhs.values;
  if (rhs.location == MemoryLocation::kHost && rhs.nnz > 0) {
    impl_->sparse_indices.Resize(static_cast<std::size_t>(rhs.nnz),
                                 impl_->context.stream(), impl_->context.impl_->pool);
    impl_->sparse_values.Resize(static_cast<std::size_t>(rhs.nnz),
                                impl_->context.stream(), impl_->context.impl_->pool);
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl_->sparse_indices.data(), rhs.indices,
        static_cast<std::size_t>(rhs.nnz) * sizeof(std::int32_t),
        cudaMemcpyHostToDevice, impl_->context.stream()));
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl_->sparse_values.data(), rhs.values,
        static_cast<std::size_t>(rhs.nnz) * sizeof(float),
        cudaMemcpyHostToDevice, impl_->context.stream()));
    indices = impl_->sparse_indices.data();
    values = impl_->sparse_values.data();
  }
  internal::ScatterSparse(indices, values, rhs.nnz, impl_->dense_rhs.data(),
                          rhs.size, impl_->context.stream());
  return impl_->SolveDevice(matrix, impl_->dense_rhs.data(), solution, options,
                            initial_guess);
}

std::vector<SolverResult> Solver::solve_many(
    const LinearOperator& matrix, DenseMatrixView rhs,
    MutableDenseMatrixView solution, const SolverOptions& options) {
  if (rhs.rows != matrix.rows() || solution.rows != matrix.cols() ||
      rhs.cols <= 0 || rhs.cols != solution.cols ||
      rhs.leading_dimension < rhs.rows ||
      solution.leading_dimension < solution.rows || rhs.values == nullptr ||
      solution.values == nullptr) {
    throw Error("Dense RHS matrix or output dimensions are incompatible with A");
  }
  std::vector<SolverResult> results;
  results.reserve(static_cast<std::size_t>(rhs.cols));
  for (std::int64_t column = 0; column < rhs.cols; ++column) {
    const float* b = rhs.values + column * rhs.leading_dimension;
    float* x = solution.values + column * solution.leading_dimension;
    results.push_back(solve(matrix, {rhs.rows, b, rhs.location},
                            {solution.rows, x}, options));
  }
  return results;
}

}  // namespace cuda_lsmr
