#include "cuda_lsmr/context.hpp"

#include <limits>
#include <utility>

#include "internal/context_impl.hpp"
#include "internal/cuda_utils.hpp"

namespace cuda_lsmr {

std::unique_ptr<Context::Impl> Context::CreateImpl(cudaStream_t stream,
                                                   bool owns_stream) {
  auto impl = std::make_unique<Context::Impl>();
  impl->stream = stream;
  impl->owns_stream = owns_stream;
  CUDA_LSMR_CHECK_CUDA(cudaGetDevice(&impl->device));
  CUDA_LSMR_CHECK_CUDA(cudaDeviceGetDefaultMemPool(&impl->pool, impl->device));

  std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
  CUDA_LSMR_CHECK_CUDA(cudaMemPoolSetAttribute(
      impl->pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  CUDA_LSMR_CHECK_CUBLAS(cublasCreate(&impl->cublas));
  CUDA_LSMR_CHECK_CUSPARSE(cusparseCreate(&impl->cusparse));
  CUDA_LSMR_CHECK_CUBLAS(cublasSetStream(impl->cublas, stream));
  CUDA_LSMR_CHECK_CUSPARSE(cusparseSetStream(impl->cusparse, stream));
  return impl;
}

Context::Context() {
  cudaStream_t stream{};
  CUDA_LSMR_CHECK_CUDA(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  try {
    impl_ = CreateImpl(stream, true);
  } catch (...) {
    static_cast<void>(cudaStreamDestroy(stream));
    throw;
  }
}

Context::Context(cudaStream_t stream) {
  if (stream == nullptr) {
    throw Error("Context requires a non-null caller-owned CUDA stream");
  }
  impl_ = CreateImpl(stream, false);
}

Context::~Context() {
  if (!impl_) {
    return;
  }
  static_cast<void>(cudaStreamSynchronize(impl_->stream));
  if (impl_->cusparse != nullptr) {
    static_cast<void>(cusparseDestroy(impl_->cusparse));
  }
  if (impl_->cublas != nullptr) {
    static_cast<void>(cublasDestroy(impl_->cublas));
  }
  if (impl_->owns_stream && impl_->stream != nullptr) {
    static_cast<void>(cudaStreamDestroy(impl_->stream));
  }
}

Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

cudaStream_t Context::stream() const noexcept {
  return impl_ ? impl_->stream : nullptr;
}

void Context::synchronize() const {
  if (!impl_) {
    throw Error("Cannot synchronize a moved-from Context");
  }
  CUDA_LSMR_CHECK_CUDA(cudaStreamSynchronize(impl_->stream));
}

void Context::set_pool_release_threshold(std::uint64_t bytes) {
  if (!impl_) {
    throw Error("Cannot configure a moved-from Context");
  }
  CUDA_LSMR_CHECK_CUDA(cudaMemPoolSetAttribute(
      impl_->pool, cudaMemPoolAttrReleaseThreshold, &bytes));
}

std::size_t Context::pool_reserved_bytes() const {
  if (!impl_) {
    return 0;
  }
  std::uint64_t bytes{};
  CUDA_LSMR_CHECK_CUDA(
      cudaMemPoolGetAttribute(impl_->pool, cudaMemPoolAttrReservedMemCurrent, &bytes));
  return static_cast<std::size_t>(bytes);
}

}  // namespace cuda_lsmr
