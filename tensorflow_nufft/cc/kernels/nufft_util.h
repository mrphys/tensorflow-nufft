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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_UTIL_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_UTIL_H_

#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


namespace tensorflow {
namespace nufft {

// Calculates the scaling factor needed to ensure that the interpolation and
// spreading do not scale the values.
template<typename FloatType>
FloatType calculate_scale_factor(int rank,
                                 const SpreadParameters<FloatType>& opts);

// Evaluates the exponential of semi-circle kernel at the specified point.
// Kernel is related to an asymptotic approximation to the Kaiser-Bessel kernel,
// itself an approximation to prolate spheroidal wavefunction (PSWF) of order 0.
template<typename FloatType>
FloatType evaluate_kernel(FloatType x, const SpreadParameters<FloatType> &opts);

// Approximates exact Fourier series coeffs of cnufftspread's real symmetric
// kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
// narrowness of kernel. Uses phase winding for cheap eval on the regular freq
// grid. Note that this is also the Fourier transform of the non-periodized
// kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
// overall prefactor of 1/h, which is needed anyway for the correction, and
// arises because the quadrature weights are scaled for grid units not x units.
template<typename FloatType>
void kernel_fseries_1d(int grid_size,
                       const SpreadParameters<FloatType>& spread_params,
                       FloatType* fseries_coeffs);

// Finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). If b is specified, the returned number must also be a
// multiple of b (b must be a number whose prime factors are no larger than 5).
template<typename IntType>
IntType next_smooth_int(IntType n, IntType b = 1);

// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
template<typename FloatType>
void array_range(int64_t n, FloatType* a, FloatType *lo, FloatType *hi);

}  // namespace nufft
}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_UTIL_H_
