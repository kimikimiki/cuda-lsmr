#include "cuda_lsmr/version.hpp"

#include <string_view>

int main() {
  return std::string_view(cuda_lsmr::version()) == "0.1.0" &&
                 cuda_lsmr::minimum_cuda_version() == 12090
             ? 0
             : 1;
}
