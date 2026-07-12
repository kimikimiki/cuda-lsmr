#pragma once

#include <cstdint>
#include <memory>

#include "cuda_lsmr/types.hpp"

namespace cuda_lsmr {

class Context;

/// Abstract device-side matrix operator used by the LSMR recurrence.
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  /// Returns the number of matrix rows.
  [[nodiscard]] virtual std::int64_t rows() const noexcept = 0;

  /// Returns the number of matrix columns.
  [[nodiscard]] virtual std::int64_t cols() const noexcept = 0;

  /// Computes `y = A * x` on the context stream.
  virtual void apply(const float* x, float* y) const;

  /// Computes `y = A^T * x` on the context stream.
  virtual void apply_transpose(const float* x, float* y) const;

  /// Computes `y = alpha * A * x + beta * y` on the context stream.
  virtual void apply_scaled(float alpha, const float* x, float beta,
                            float* y) const = 0;

  /// Computes `y = alpha * A^T * x + beta * y` on the context stream.
  virtual void apply_transpose_scaled(float alpha, const float* x, float beta,
                                      float* y) const = 0;
};

/// Owns CSR(A) and CSC(A) device copies for transpose-free SpMV dispatch.
class SparseMatrix final : public LinearOperator {
 public:
  /// Uploads CSR(A), then constructs and retains CSC(A) on the device.
  static std::unique_ptr<SparseMatrix> FromCsr(Context& context,
                                               CsrMatrixView matrix);

  /// Uploads CSC(A), then constructs and retains CSR(A) on the device.
  static std::unique_ptr<SparseMatrix> FromCsc(Context& context,
                                               CscMatrixView matrix);

  ~SparseMatrix() override;
  SparseMatrix(SparseMatrix&&) noexcept;
  SparseMatrix& operator=(SparseMatrix&&) noexcept;
  SparseMatrix(const SparseMatrix&) = delete;
  SparseMatrix& operator=(const SparseMatrix&) = delete;

  /// Returns the number of matrix rows.
  [[nodiscard]] std::int64_t rows() const noexcept override;

  /// Returns the number of matrix columns.
  [[nodiscard]] std::int64_t cols() const noexcept override;

  /// Returns the stored nonzero count.
  [[nodiscard]] std::int64_t nnz() const noexcept;

  /// Computes `y = A * x` through CSR(A).
  void apply(const float* x, float* y) const override;

  /// Computes `y = A^T * x` through CSC(A) interpreted as CSR(A^T).
  void apply_transpose(const float* x, float* y) const override;

  /// Computes a scaled CSR(A) SpMV.
  void apply_scaled(float alpha, const float* x, float beta,
                    float* y) const override;

  /// Computes a scaled CSR(A^T) SpMV.
  void apply_transpose_scaled(float alpha, const float* x, float beta,
                              float* y) const override;

 private:
  struct Impl;
  explicit SparseMatrix(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Owns a column-major FP32 dense matrix and uses cuBLAS GEMV.
class DenseMatrix final : public LinearOperator {
 public:
  /// Copies or adopts a dense matrix view into context-owned device storage.
  static std::unique_ptr<DenseMatrix> Create(Context& context,
                                             DenseMatrixView matrix);

  ~DenseMatrix() override;
  DenseMatrix(DenseMatrix&&) noexcept;
  DenseMatrix& operator=(DenseMatrix&&) noexcept;
  DenseMatrix(const DenseMatrix&) = delete;
  DenseMatrix& operator=(const DenseMatrix&) = delete;

  /// Returns the number of matrix rows.
  [[nodiscard]] std::int64_t rows() const noexcept override;

  /// Returns the number of matrix columns.
  [[nodiscard]] std::int64_t cols() const noexcept override;

  /// Computes `y = A * x` with cuBLAS.
  void apply(const float* x, float* y) const override;

  /// Computes `y = A^T * x` with cuBLAS.
  void apply_transpose(const float* x, float* y) const override;

  /// Computes a scaled cuBLAS GEMV.
  void apply_scaled(float alpha, const float* x, float beta,
                    float* y) const override;

  /// Computes a scaled transposed cuBLAS GEMV.
  void apply_transpose_scaled(float alpha, const float* x, float beta,
                              float* y) const override;

 private:
  struct Impl;
  explicit DenseMatrix(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuda_lsmr
