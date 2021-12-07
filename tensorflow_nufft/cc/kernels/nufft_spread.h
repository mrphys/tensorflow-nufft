/* Copyright 2021 University College London. All Rights Reserved.

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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace nufft {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Direction of the spreading/interpolation.
enum class SpreadDirection {
  SPREAD, // non-uniform to uniform
  INTERP  // uniform to non-uniform
};

template<typename FloatType>
struct SpreadParameters {

  // The spread direction (U->NU or NU->U). See enum above.
  SpreadDirection spread_direction;

  // TODO: revise the following options.

  // This is the main documentation for these options...
  int nspread;            // w, the kernel width in grid pts

  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  bool check_bounds;      // 0: don't check NU pts in 3-period range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  bool pad_kernel;            // 0: no pad w to mult of 4, 1: do pad
                          // (this helps SIMD for kerevalmeth=0, eg on i7).
  int num_threads;        // # threads for spreadinterp (0: use max avail)
  int sort_threads;       // # threads for sort (0: auto-choice up to num_threads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int verbosity;          // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold;   // num threads before switching spreadSorted to using atomic ops
  double upsampling_factor;       // sigma, upsampling factor
  bool spread_only;   // 0: NUFFT, 1: spread or interpolation only
  // ES kernel specific consts used in fast eval, depend on precision FLT...
  FloatType ES_beta;
  FloatType ES_halfwidth;
  FloatType ES_c;
  FloatType ES_scale;           // used for spread/interp only
};

template<typename Device, typename FloatType>
class SpreaderBase {

 protected:

  SpreadParameters<FloatType> params_;
};

template<typename Device, typename FloatType>
class Spreader;

#if GOOGLE_CUDA
template<typename FloatType>
class Spreader<GPUDevice, FloatType> : public SpreaderBase<GPUDevice, FloatType> {

  Status initialize(const SpreadParameters<FloatType>& params);

  Status spread();

  Status interp();
};
#endif // GOOGLE_CUDA

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_
