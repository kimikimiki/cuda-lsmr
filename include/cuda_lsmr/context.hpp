#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuda_lsmr {

/// Owns the CUDA stream, library handles, and stream-ordered memory pool.
class Context {
 public:
  /// Creates a context on the current CUDA device with an owned stream.
  Context();

  /// Wraps a caller-owned stream; the stream remains owned by the caller.
  explicit Context(cudaStream_t stream);

  /// Releases handles and pooled allocations after pending work completes.
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  /// Returns the stream used by every operation in this context.
  [[nodiscard]] cudaStream_t stream() const noexcept;

  /// Blocks until all work submitted through this context is complete.
  void synchronize() const;

  /// Keeps up to `bytes` reserved in the CUDA pool; zero releases eagerly.
  void set_pool_release_threshold(std::uint64_t bytes);

  /// Returns bytes currently reserved by the CUDA memory pool.
  [[nodiscard]] std::size_t pool_reserved_bytes() const;

 private:
  struct Impl;
  static std::unique_ptr<Impl> CreateImpl(cudaStream_t stream, bool owns_stream);
  std::unique_ptr<Impl> impl_;

  friend class LinearOperator;
  friend class SparseMatrix;
  friend class DenseMatrix;
  friend class Solver;
};

}  // namespace cuda_lsmr
