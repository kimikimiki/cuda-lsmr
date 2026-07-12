#include "cuda_lsmr/version.hpp"

namespace cuda_lsmr {

const char* version() noexcept { return "0.1.0"; }

int minimum_cuda_version() noexcept { return 12090; }

}  // namespace cuda_lsmr

