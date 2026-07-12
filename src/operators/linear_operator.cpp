#include "cuda_lsmr/matrix.hpp"

namespace cuda_lsmr {

void LinearOperator::apply(const float* x, float* y) const {
  apply_scaled(1.0F, x, 0.0F, y);
}

void LinearOperator::apply_transpose(const float* x, float* y) const {
  apply_transpose_scaled(1.0F, x, 0.0F, y);
}

}  // namespace cuda_lsmr
