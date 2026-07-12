#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include "cuda_lsmr/context.hpp"

namespace cuda_lsmr {

struct Context::Impl {
  cudaStream_t stream{};
  bool owns_stream{};
  int device{};
  cudaMemPool_t pool{};
  cublasHandle_t cublas{};
  cusparseHandle_t cusparse{};
};

}  // namespace cuda_lsmr

