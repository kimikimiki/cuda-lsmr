#pragma once

namespace cuda_lsmr {

/// Returns the semantic version compiled into the library.
[[nodiscard]] const char* version() noexcept;

/// Returns the minimum supported CUDA Toolkit version as an integer.
[[nodiscard]] int minimum_cuda_version() noexcept;

}  // namespace cuda_lsmr

