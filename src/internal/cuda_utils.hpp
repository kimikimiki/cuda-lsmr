#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <sstream>
#include <string>

#include "cuda_lsmr/error.hpp"

namespace cuda_lsmr::internal {

/// Throws `Error` when a CUDA Runtime call fails.
inline void CheckCuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << expression << " failed: " << cudaGetErrorString(status);
  throw Error(message.str());
}

/// Throws `Error` when a cuBLAS call fails.
inline void CheckCublas(cublasStatus_t status, const char* expression) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream message;
  message << expression << " failed with cuBLAS status " << status;
  throw Error(message.str());
}

/// Throws `Error` when a cuSPARSE call fails.
inline void CheckCusparse(cusparseStatus_t status, const char* expression) {
  if (status == CUSPARSE_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream message;
  message << expression << " failed: " << cusparseGetErrorString(status);
  throw Error(message.str());
}

#define CUDA_LSMR_CHECK_CUDA(expression) \
  ::cuda_lsmr::internal::CheckCuda((expression), #expression)
#define CUDA_LSMR_CHECK_CUBLAS(expression) \
  ::cuda_lsmr::internal::CheckCublas((expression), #expression)
#define CUDA_LSMR_CHECK_CUSPARSE(expression) \
  ::cuda_lsmr::internal::CheckCusparse((expression), #expression)

}  // namespace cuda_lsmr::internal

