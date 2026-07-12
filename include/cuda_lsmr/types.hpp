#pragma once

#include <cstdint>

namespace cuda_lsmr {

/// Identifies whether a non-owning view points to host or device memory.
enum class MemoryLocation { kHost, kDevice };

/// Describes a 32-bit-indexed FP32 CSR matrix.
struct CsrMatrixView {
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t nnz{};
  const std::int32_t* row_offsets{};
  const std::int32_t* column_indices{};
  const float* values{};
  MemoryLocation location{MemoryLocation::kHost};
};

/// Describes a 32-bit-indexed FP32 CSC matrix.
struct CscMatrixView {
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t nnz{};
  const std::int32_t* column_offsets{};
  const std::int32_t* row_indices{};
  const float* values{};
  MemoryLocation location{MemoryLocation::kHost};
};

/// Describes an FP32 column-major dense matrix.
struct DenseMatrixView {
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t leading_dimension{};
  const float* values{};
  MemoryLocation location{MemoryLocation::kHost};
};

/// Describes a mutable FP32 column-major dense matrix on the device.
struct MutableDenseMatrixView {
  std::int64_t rows{};
  std::int64_t cols{};
  std::int64_t leading_dimension{};
  float* values{};
};

/// Describes an FP32 dense vector.
struct DenseVectorView {
  std::int64_t size{};
  const float* values{};
  MemoryLocation location{MemoryLocation::kDevice};
};

/// Describes a mutable FP32 device vector.
struct MutableDenseVectorView {
  std::int64_t size{};
  float* values{};
};

/// Describes an FP32 sparse vector with unique zero-based indices.
struct SparseVectorView {
  std::int64_t size{};
  std::int64_t nnz{};
  const std::int32_t* indices{};
  const float* values{};
  MemoryLocation location{MemoryLocation::kDevice};
};

}  // namespace cuda_lsmr

