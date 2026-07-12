#include "cuda_lsmr/matrix.hpp"

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "cuda_lsmr/context.hpp"
#include "cuda_lsmr/error.hpp"
#include "internal/context_impl.hpp"
#include "internal/cuda_utils.hpp"
#include "internal/device_buffer.hpp"

namespace cuda_lsmr {

struct DenseMatrix::Impl {
  Impl(Context& owner, std::int64_t row_count, std::int64_t column_count,
       std::int64_t leading_dim)
      : context(owner),
        rows(row_count),
        cols(column_count),
        leading_dimension(leading_dim),
        values(static_cast<std::size_t>(leading_dim * column_count), owner.stream()) {}

  Context& context;
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t leading_dimension{};
  internal::DeviceBuffer<float> values;
};

std::unique_ptr<DenseMatrix> DenseMatrix::Create(Context& context,
                                                 DenseMatrixView matrix) {
  if (matrix.rows <= 0 || matrix.cols <= 0 ||
      matrix.leading_dimension < matrix.rows || matrix.values == nullptr) {
    throw Error("DenseMatrix requires positive dimensions, valid ld, and values");
  }
  if (matrix.rows > std::numeric_limits<int>::max() ||
      matrix.cols > std::numeric_limits<int>::max() ||
      matrix.leading_dimension > std::numeric_limits<int>::max()) {
    throw Error("DenseMatrix dimensions exceed the cuBLAS int32 API");
  }
  auto impl = std::make_unique<Impl>(context, matrix.rows, matrix.cols,
                                     matrix.leading_dimension);
  const cudaMemcpyKind kind = matrix.location == MemoryLocation::kHost
                                  ? cudaMemcpyHostToDevice
                                  : cudaMemcpyDeviceToDevice;
  CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
      impl->values.data(), matrix.values,
      static_cast<std::size_t>(matrix.leading_dimension * matrix.cols) *
          sizeof(float),
      kind, context.stream()));
  return std::unique_ptr<DenseMatrix>(new DenseMatrix(std::move(impl)));
}

DenseMatrix::DenseMatrix(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DenseMatrix::~DenseMatrix() = default;
DenseMatrix::DenseMatrix(DenseMatrix&&) noexcept = default;
DenseMatrix& DenseMatrix::operator=(DenseMatrix&&) noexcept = default;

std::int64_t DenseMatrix::rows() const noexcept { return impl_->rows; }
std::int64_t DenseMatrix::cols() const noexcept { return impl_->cols; }

void DenseMatrix::apply(const float* x, float* y) const {
  if (x == nullptr || y == nullptr) throw Error("DenseMatrix::apply received null");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  CUDA_LSMR_CHECK_CUBLAS(cublasSgemv(
      impl_->context.impl_->cublas, CUBLAS_OP_N, static_cast<int>(impl_->rows),
      static_cast<int>(impl_->cols), &alpha, impl_->values.data(),
      static_cast<int>(impl_->leading_dimension), x, 1, &beta, y, 1));
}

void DenseMatrix::apply_transpose(const float* x, float* y) const {
  if (x == nullptr || y == nullptr) {
    throw Error("DenseMatrix::apply_transpose received null");
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  CUDA_LSMR_CHECK_CUBLAS(cublasSgemv(
      impl_->context.impl_->cublas, CUBLAS_OP_T, static_cast<int>(impl_->rows),
      static_cast<int>(impl_->cols), &alpha, impl_->values.data(),
      static_cast<int>(impl_->leading_dimension), x, 1, &beta, y, 1));
}

}  // namespace cuda_lsmr

