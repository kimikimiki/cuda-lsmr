#include "cuda_lsmr/version.hpp"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_cuda_lsmr, module) {
  module.doc() = "Zero-copy benchmark bindings for cuda-lsmr";
  module.attr("__version__") = cuda_lsmr::version();
}

