#pragma once

#include <stdexcept>
#include <string>

namespace cuda_lsmr {

/// Reports invalid input or a CUDA-library failure.
class Error : public std::runtime_error {
 public:
  /// Constructs an error with a human-readable message.
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace cuda_lsmr

