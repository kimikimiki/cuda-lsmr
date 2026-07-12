#include "cuda_lsmr/version.hpp"

#include <cassert>
#include <string_view>

int main() {
  assert(std::string_view(cuda_lsmr::version()) == "0.1.0");
  assert(cuda_lsmr::minimum_cuda_version() == 12090);
  return 0;
}

