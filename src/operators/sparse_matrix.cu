#include "cuda_lsmr/matrix.hpp"

#include <cusparse.h>

#include <algorithm>
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
namespace {

/// Validates dimensions and pointers for a compressed sparse matrix.
void ValidateCompressed(std::int64_t rows, std::int64_t cols, std::int64_t nnz,
                        const std::int32_t* offsets, const std::int32_t* indices,
                        const float* values) {
  if (rows <= 0 || cols <= 0 || nnz < 0) {
    throw Error("Sparse matrix dimensions must be positive and nnz non-negative");
  }
  if (rows > std::numeric_limits<std::int32_t>::max() ||
      cols > std::numeric_limits<std::int32_t>::max() ||
      nnz > std::numeric_limits<std::int32_t>::max()) {
    throw Error("SparseMatrix FP32 API currently requires 32-bit dimensions and nnz");
  }
  if (offsets == nullptr || (nnz > 0 && (indices == nullptr || values == nullptr))) {
    throw Error("Sparse matrix contains a null data pointer");
  }
}

/// Selects the CUDA copy kind for a caller-provided view.
cudaMemcpyKind CopyKind(MemoryLocation location) {
  return location == MemoryLocation::kHost ? cudaMemcpyHostToDevice
                                           : cudaMemcpyDeviceToDevice;
}

}  // namespace

struct SparseMatrix::Impl {
  explicit Impl(Context& owner, std::int64_t row_count, std::int64_t column_count,
                std::int64_t nonzeros)
      : context(owner),
        rows(row_count),
        cols(column_count),
        nnz(nonzeros),
        csr_offsets(static_cast<std::size_t>(rows + 1), owner.stream()),
        csr_indices(static_cast<std::size_t>(nnz), owner.stream()),
        csr_values(static_cast<std::size_t>(nnz), owner.stream()),
        csc_offsets(static_cast<std::size_t>(cols + 1), owner.stream()),
        csc_indices(static_cast<std::size_t>(nnz), owner.stream()),
        csc_values(static_cast<std::size_t>(nnz), owner.stream()) {}

  /// Creates matrix/vector descriptors and reserves one reusable SpMV buffer.
  void InitializeDescriptors() {
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateCsr(
        &csr_descriptor, rows, cols, nnz, csr_offsets.data(), csr_indices.data(),
        csr_values.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // CSC(A) is structurally CSR(A^T), so transpose multiplication stays a
    // non-transpose CSR operation and avoids scatter-style transpose kernels.
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateCsr(
        &transpose_descriptor, cols, rows, nnz, csc_offsets.data(),
        csc_indices.data(), csc_values.data(), CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float* dummy = nnz > 0 ? csr_values.data() : nullptr;
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateDnVec(&ax_x, cols, dummy, CUDA_R_32F));
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateDnVec(&ax_y, rows, dummy, CUDA_R_32F));
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateDnVec(&at_x, rows, dummy, CUDA_R_32F));
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCreateDnVec(&at_y, cols, dummy, CUDA_R_32F));

    const float alpha = 1.0F;
    const float beta = 0.0F;
    std::size_t ax_bytes{};
    std::size_t at_bytes{};
    CUDA_LSMR_CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        context.impl_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        csr_descriptor, ax_x, &beta, ax_y, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1, &ax_bytes));
    CUDA_LSMR_CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        context.impl_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        transpose_descriptor, at_x, &beta, at_y, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1, &at_bytes));
    spmv_buffer.Resize(std::max(ax_bytes, at_bytes), context.stream());
  }

  /// Converts CSR input to CSC output with cuSPARSE generic storage conversion.
  void ConvertCsrToCsc(std::int64_t conversion_rows,
                       std::int64_t conversion_cols, const float* input_values,
                       const std::int32_t* input_offsets,
                       const std::int32_t* input_indices, float* output_values,
                       std::int32_t* output_offsets,
                       std::int32_t* output_indices) {
    std::size_t bytes{};
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
        context.impl_->cusparse, conversion_rows, conversion_cols, nnz,
        input_values, input_offsets, input_indices, output_values,
        output_offsets, output_indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bytes));
    conversion_buffer.Resize(bytes, context.stream());
    CUDA_LSMR_CHECK_CUSPARSE(cusparseCsr2cscEx2(
        context.impl_->cusparse, conversion_rows, conversion_cols, nnz,
        input_values, input_offsets, input_indices, output_values,
        output_offsets, output_indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        conversion_buffer.data()));
  }

  /// Executes one cached-descriptor non-transpose SpMV.
  void Spmv(cusparseSpMatDescr_t matrix, cusparseDnVecDescr_t input,
            cusparseDnVecDescr_t output, const float* x, float* y) const {
    CUDA_LSMR_CHECK_CUSPARSE(cusparseDnVecSetValues(input, const_cast<float*>(x)));
    CUDA_LSMR_CHECK_CUSPARSE(cusparseDnVecSetValues(output, y));
    const float alpha = 1.0F;
    const float beta = 0.0F;
    CUDA_LSMR_CHECK_CUSPARSE(cusparseSpMV(
        context.impl_->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        matrix, input, &beta, output, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1, spmv_buffer.data()));
  }

  ~Impl() {
    if (ax_x != nullptr) static_cast<void>(cusparseDestroyDnVec(ax_x));
    if (ax_y != nullptr) static_cast<void>(cusparseDestroyDnVec(ax_y));
    if (at_x != nullptr) static_cast<void>(cusparseDestroyDnVec(at_x));
    if (at_y != nullptr) static_cast<void>(cusparseDestroyDnVec(at_y));
    if (csr_descriptor != nullptr) {
      static_cast<void>(cusparseDestroySpMat(csr_descriptor));
    }
    if (transpose_descriptor != nullptr) {
      static_cast<void>(cusparseDestroySpMat(transpose_descriptor));
    }
  }

  Context& context;
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t nnz{};
  internal::DeviceBuffer<std::int32_t> csr_offsets;
  internal::DeviceBuffer<std::int32_t> csr_indices;
  internal::DeviceBuffer<float> csr_values;
  internal::DeviceBuffer<std::int32_t> csc_offsets;
  internal::DeviceBuffer<std::int32_t> csc_indices;
  internal::DeviceBuffer<float> csc_values;
  internal::DeviceBuffer<std::byte> conversion_buffer;
  mutable internal::DeviceBuffer<std::byte> spmv_buffer;
  cusparseSpMatDescr_t csr_descriptor{};
  cusparseSpMatDescr_t transpose_descriptor{};
  cusparseDnVecDescr_t ax_x{};
  cusparseDnVecDescr_t ax_y{};
  cusparseDnVecDescr_t at_x{};
  cusparseDnVecDescr_t at_y{};
};

std::unique_ptr<SparseMatrix> SparseMatrix::FromCsr(Context& context,
                                                    CsrMatrixView matrix) {
  ValidateCompressed(matrix.rows, matrix.cols, matrix.nnz, matrix.row_offsets,
                     matrix.column_indices, matrix.values);
  auto impl = std::make_unique<Impl>(context, matrix.rows, matrix.cols, matrix.nnz);
  const cudaMemcpyKind kind = CopyKind(matrix.location);
  CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
      impl->csr_offsets.data(), matrix.row_offsets,
      static_cast<std::size_t>(matrix.rows + 1) * sizeof(std::int32_t), kind,
      context.stream()));
  if (matrix.nnz > 0) {
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl->csr_indices.data(), matrix.column_indices,
        static_cast<std::size_t>(matrix.nnz) * sizeof(std::int32_t), kind,
        context.stream()));
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl->csr_values.data(), matrix.values,
        static_cast<std::size_t>(matrix.nnz) * sizeof(float), kind,
        context.stream()));
  }
  impl->ConvertCsrToCsc(matrix.rows, matrix.cols, impl->csr_values.data(),
                        impl->csr_offsets.data(), impl->csr_indices.data(),
                        impl->csc_values.data(), impl->csc_offsets.data(),
                        impl->csc_indices.data());
  impl->InitializeDescriptors();
  return std::unique_ptr<SparseMatrix>(new SparseMatrix(std::move(impl)));
}

std::unique_ptr<SparseMatrix> SparseMatrix::FromCsc(Context& context,
                                                    CscMatrixView matrix) {
  ValidateCompressed(matrix.rows, matrix.cols, matrix.nnz, matrix.column_offsets,
                     matrix.row_indices, matrix.values);
  auto impl = std::make_unique<Impl>(context, matrix.rows, matrix.cols, matrix.nnz);
  const cudaMemcpyKind kind = CopyKind(matrix.location);
  CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
      impl->csc_offsets.data(), matrix.column_offsets,
      static_cast<std::size_t>(matrix.cols + 1) * sizeof(std::int32_t), kind,
      context.stream()));
  if (matrix.nnz > 0) {
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl->csc_indices.data(), matrix.row_indices,
        static_cast<std::size_t>(matrix.nnz) * sizeof(std::int32_t), kind,
        context.stream()));
    CUDA_LSMR_CHECK_CUDA(cudaMemcpyAsync(
        impl->csc_values.data(), matrix.values,
        static_cast<std::size_t>(matrix.nnz) * sizeof(float), kind,
        context.stream()));
  }
  // Converting CSR(A^T) to CSC(A^T) produces CSR(A) arrays directly.
  impl->ConvertCsrToCsc(matrix.cols, matrix.rows, impl->csc_values.data(),
                        impl->csc_offsets.data(), impl->csc_indices.data(),
                        impl->csr_values.data(), impl->csr_offsets.data(),
                        impl->csr_indices.data());
  impl->InitializeDescriptors();
  return std::unique_ptr<SparseMatrix>(new SparseMatrix(std::move(impl)));
}

SparseMatrix::SparseMatrix(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SparseMatrix::~SparseMatrix() = default;
SparseMatrix::SparseMatrix(SparseMatrix&&) noexcept = default;
SparseMatrix& SparseMatrix::operator=(SparseMatrix&&) noexcept = default;

std::int64_t SparseMatrix::rows() const noexcept { return impl_->rows; }
std::int64_t SparseMatrix::cols() const noexcept { return impl_->cols; }
std::int64_t SparseMatrix::nnz() const noexcept { return impl_->nnz; }

void SparseMatrix::apply(const float* x, float* y) const {
  if (x == nullptr || y == nullptr) throw Error("SparseMatrix::apply received null");
  impl_->Spmv(impl_->csr_descriptor, impl_->ax_x, impl_->ax_y, x, y);
}

void SparseMatrix::apply_transpose(const float* x, float* y) const {
  if (x == nullptr || y == nullptr) {
    throw Error("SparseMatrix::apply_transpose received null");
  }
  impl_->Spmv(impl_->transpose_descriptor, impl_->at_x, impl_->at_y, x, y);
}

}  // namespace cuda_lsmr

