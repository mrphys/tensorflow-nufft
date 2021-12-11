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

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

// Specifies the direction of spreading.
enum class SpreadDirection {
  SPREAD, // non-uniform to uniform
  INTERP  // uniform to non-uniform
};

// Specifies whether non-uniform points should be sorted.
enum class SortPoints {
  AUTO = -1,  // Choose automatically using a heuristic.
  NO = 0,     // Do not sort non-uniform points.
  YES = 1     // Sort non-uniform points.
};

// Specifies the spread method.
enum class SpreadMethod {
  AUTO = -1,
  NUPTS_DRIVEN = 0,
  SUBPROBLEM = 1,
  PAUL = 2,
  BLOCK_GATHER = 3
};

template<typename FloatType>
struct SpreadParameters {

  // The spread direction (U->NU or NU->U). See enum above.
  SpreadDirection spread_direction;

  // Whether to sort the non-uniform points.
  SortPoints sort_points = SortPoints::AUTO;

  // Specifies the spread method.
  SpreadMethod spread_method = SpreadMethod::AUTO;

  // TODO: revise the following options.

  // This is the main documentation for these options...
  int nspread;            // w, the kernel width in grid pts

  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  bool check_bounds;      // 0: don't check NU pts in 3-period range; 1: do
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

  #if GOOGLE_CUDA
  // Used for 3D subproblem method. 0 means automatic selection.
  dim3 gpu_bin_size = {0, 0, 0};

  // Used for 3D spread-block-gather method. 0 means automatic selection.
  dim3 gpu_obin_size = {0, 0, 0};
  #endif // GOOGLE_CUDA
};

template<typename Device, typename FloatType>
class SpreaderBase {

 public:
  SpreaderBase(Device& device)
      : device_(device),
        rank_(0),
        num_points_(0),
        grid_dims_{1, 1, 1},
        points_{nullptr, nullptr, nullptr},
        params_(),
        is_initialized_(false) { }

  ~SpreaderBase() { }

 protected:
  Device& device_;
  int rank_;
  int num_points_;
  int grid_dims_[3];
  FloatType* points_[3]; // not owned
  SpreadParameters<FloatType> params_;
  bool is_initialized_;
};

template<typename Device, typename FloatType>
class Spreader;

#if GOOGLE_CUDA
template<typename FloatType>
class Spreader<GPUDevice, FloatType> : public SpreaderBase<GPUDevice, FloatType> {

 public:
  Spreader(GPUDevice& device)
      : SpreaderBase<GPUDevice, FloatType>(device),
        indices_points_(nullptr),
        sort_indices_(nullptr) {}
  
  ~Spreader() {
    if (this->indices_points_) {
      this->device_.deallocate(this->indices_points_);
    }
    if (this->sort_indices_) {
      this->device_.deallocate(this->sort_indices_);
    }
  }

  Status initialize(
      int rank, int* grid_dims, int num_points,
      FloatType* points_x, FloatType* points_y, FloatType* points_z,
      const SpreadParameters<FloatType> params);

  Status spread();

  Status interp();

 private:
  int* indices_points_;
  int* sort_indices_;
  int* bin_sizes_;
  int* bin_start_pts_;
};
#endif // GOOGLE_CUDA

}  // namespace nufft
}  // namespace tensorflow

#endif // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_
