/* Copyright 2021 The TensorFlow NUFFT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Copyright 2017-2021 The Simons Foundation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_nufft/cc/kernels/nufft_util.h"

#include <cstdint>

#include "tensorflow_nufft/cc/kernels/legendre_rule_fast.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"
#include "tensorflow_nufft/cc/kernels/omp_api.h"


namespace tensorflow {
namespace nufft {

template<typename FloatType>
FloatType calculate_scale_factor(
    int rank, const SpreadParameters<FloatType> &opts) {

  int n = 100;
  FloatType h = 2.0 / n;
  FloatType x = -1.0;
  FloatType sum = 0.0;
  for(int i = 1; i < n; i++) {
    x += h;
    sum += exp(opts.kernel_beta * sqrt(1.0 - x * x));
  }
  sum += 1.0;
  sum *= h;
  sum *= sqrt(1.0 / opts.kernel_c);
  FloatType scale = sum;
  if (rank > 1) { scale *= sum; }
  if (rank > 2) { scale *= sum; }
  return 1.0 / scale;
}

template<typename FloatType>
void array_range(int64_t n, FloatType* a, FloatType *lo, FloatType *hi) {
  *lo = INFINITY; *hi = -INFINITY;
  for (int64_t m = 0; m < n; ++m) {
    if (a[m] < *lo) *lo = a[m];
    if (a[m] > *hi) *hi = a[m];
  }
}

template float calculate_scale_factor<float>(
    int, const SpreadParameters<float>&);
template double calculate_scale_factor<double>(
    int, const SpreadParameters<double>&);

template void array_range<float>(int64_t, float*, float*, float*);
template void array_range<double>(int64_t, double*, double*, double*);

}  // namespace nufft
}  // namespace tensorflow
