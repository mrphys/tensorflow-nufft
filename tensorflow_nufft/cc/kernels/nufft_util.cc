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
FloatType evaluate_kernel(FloatType x, const SpreadParameters<FloatType> &opts) {
  if (abs(x) >= opts.kernel_half_width)
    return 0.0;
  return exp(opts.kernel_beta * sqrt(1.0 - opts.kernel_c * x * x));
}

template<typename FloatType>
void kernel_fseries_1d(int grid_size,
                       const SpreadParameters<FloatType>& spread_params,
                       FloatType* fseries_coeffs) {

  FloatType kernel_half_width = spread_params.kernel_width / 2.0;

  // Number of quadrature nodes in z (from 0 to J/2, reflections will be added).
  int q = static_cast<int>(2 + 3.0 * kernel_half_width);
  FloatType f[kMaxQuadNodes];
  double z[2 * kMaxQuadNodes];
  double w[2 * kMaxQuadNodes];

  // Only half the nodes used, eg on (0, 1).
  legendre_compute_glr(2 * q, z, w);

  // Set up nodes z[n] and values f[n].
  std::complex<FloatType> a[kMaxQuadNodes];
  for (int n=0; n < q; ++n) {
    z[n] *= kernel_half_width;                         // rescale nodes
    f[n] = kernel_half_width * (FloatType)w[n] * evaluate_kernel((FloatType)z[n], spread_params); // vals & quadr wei
    a[n] = exp(2 * kPi<FloatType> * kImaginaryUnit<FloatType> * (FloatType)(grid_size / 2 - z[n]) / (FloatType)grid_size);  // phase winding rates
  }
  int nout = grid_size / 2 + 1;                   // how many values we're writing to
  int nt = std::min(nout, (int)spread_params.num_threads);         // how many chunks
  std::vector<int> brk(nt + 1);        // start indices for each thread
  for (int t = 0; t <= nt; ++t)             // split nout mode indices btw threads
    brk[t] = (int)(0.5 + nout * t / (double)nt);

  #pragma omp parallel num_threads(nt)
  {                                     // each thread gets own chunk to do
    int t = OMP_GET_THREAD_NUM();
    std::complex<FloatType> aj[kMaxQuadNodes];    // phase rotator for this thread

    for (int n = 0; n < q; ++n)
      aj[n] = pow(a[n], (FloatType)brk[t]);    // init phase factors for chunk

    for (int j = brk[t]; j < brk[t + 1]; ++j) {          // loop along output array
      FloatType x = 0.0;                      // accumulator for answer at this j
      for (int n = 0; n < q; ++n) {
        x += f[n] * 2 * real(aj[n]);      // include the negative freq
        aj[n] *= a[n];                  // wind the phases
      }
      fseries_coeffs[j] = x;
    }
  }
}

template<typename IntType>
IntType next_smooth_int(IntType n, IntType b) {
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;   // even
  IntType nplus = n - 2;    // to cancel out the +=2 at start of loop
  IntType numdiv = 2;       // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0)) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
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

template void kernel_fseries_1d<float>(
    int, const SpreadParameters<float>&, float*);
template void kernel_fseries_1d<double>(
    int, const SpreadParameters<double>&, double*);

template int next_smooth_int<int>(int, int);
template int64_t next_smooth_int<int64_t>(int64_t, int64_t);

template void array_range<float>(int64_t, float*, float*, float*);
template void array_range<double>(int64_t, double*, double*, double*);

}  // namespace nufft
}  // namespace tensorflow
