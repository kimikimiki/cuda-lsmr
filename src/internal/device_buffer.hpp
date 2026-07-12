#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>

#include "internal/cuda_utils.hpp"

namespace cuda_lsmr::internal {

/// Owns a stream-ordered allocation from the context CUDA memory pool.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  /// Allocates `count` elements in stream order.
  DeviceBuffer(std::size_t count, cudaStream_t stream) { Resize(count, stream); }

  /// Returns storage to the pool in stream order.
  ~DeviceBuffer() { ResetNoThrow(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { Swap(other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      ResetNoThrow();
      Swap(other);
    }
    return *this;
  }

  /// Reallocates storage when the requested capacity exceeds the current one.
  void Resize(std::size_t count, cudaStream_t stream) {
    if (count <= capacity_) {
      size_ = count;
      return;
    }
    Reset();
    stream_ = stream;
    if (count > 0) {
      CUDA_LSMR_CHECK_CUDA(
          cudaMallocAsync(reinterpret_cast<void**>(&data_), count * sizeof(T), stream_));
    }
    size_ = count;
    capacity_ = count;
  }

  /// Releases the current allocation in stream order.
  void Reset() {
    if (data_ != nullptr) {
      CUDA_LSMR_CHECK_CUDA(cudaFreeAsync(data_, stream_));
    }
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  /// Returns the mutable device pointer.
  [[nodiscard]] T* data() noexcept { return data_; }

  /// Returns the const device pointer.
  [[nodiscard]] const T* data() const noexcept { return data_; }

  /// Returns the logical element count.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  /// Returns the allocation capacity in elements.
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  void ResetNoThrow() noexcept {
    if (data_ != nullptr) {
      static_cast<void>(cudaFreeAsync(data_, stream_));
    }
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  void Swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(stream_, other.stream_);
  }

  T* data_{};
  std::size_t size_{};
  std::size_t capacity_{};
  cudaStream_t stream_{};
};

}  // namespace cuda_lsmr::internal

