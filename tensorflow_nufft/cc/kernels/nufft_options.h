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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_OPTIONS_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_OPTIONS_H_

#include <fftw3.h>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/vector_types.h"
#endif // GOOGLE_CUDA

namespace tensorflow {
namespace nufft {

enum class ModeOrder {
  CMCL = 0, // CMCL-style mode order.
  FFT = 1   // FFT-style mode order.
};

enum class KernelEvaluationMethod {
  AUTO = 0,   // Select automatically.
  DIRECT = 1, // Direct evaluation of kernel.
  HORNER = 2  // Evaluate using Horner piecewise polynomial. Faster.
};

enum class SpreadThreading {
  AUTO = 0,                       // Choose automatically.
  SEQUENTIAL_MULTI_THREADED = 1,  // Use sequential multi-threaded spreading.
  PARALLEL_SINGLE_THREADED = 2    // Use parallel single-threaded spreading.
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

// Options for the NUFFT operations. This class is used for both the CPU and the
// GPU implementation, although some options are only used by one or the other.
struct Options {
  // The mode order to use. See enum above. Applies only to type 1 and type 2
  // transforms. Applies only to the CPU kernel.
  ModeOrder mode_order = ModeOrder::CMCL;

  // Whether to check if the NUFFT points are in the correct range
  // [-3*pi, 3*pi]. This check has a small performance penalty. Applies only to
  // the CPU kernel.
  bool check_bounds = true;

  // The verbosity level. 0 means silent, 1 means some timing/debug, and 2 means
  // more debug. Applies to the CPU and the GPU kernels.
  int verbosity = 0;

  // Whether to print warnings to stderr. Applies only to the CPU kernel.
  bool show_warnings = true;

  // The number of threads to use. A value of 0 means the system picks an
  // appropriate number. Applies only to the CPU kernel.
  int num_threads = 0;

  // FFTW flags. Applies only to the CPU kernel.
  int fftw_flags = FFTW_ESTIMATE;

  // Whether to sort the non-uniform points. See enum above. Used by CPU and GPU
  // kernels.
  SortPoints sort_points = SortPoints::AUTO;

  // The kernel evaluation method. See enum above. Applies to the CPU and the
  // GPU kernels.
  KernelEvaluationMethod kernel_evaluation_method = \
      KernelEvaluationMethod::AUTO;

  // Whether to pad the interpolation kernel to a multiple of 4. This helps SIMD
  // when using direct kernel evaluation. Applies only to the CPU kernel.
  bool pad_kernel = true;

  // The upsampling factor used to create the intermediate grid. A value of 0.0
  // means the upsampling factor is automatically chosen. Applies to the CPU and
  // the GPU kernels.
  double upsampling_factor = 0.0;

  // The spreader threading strategy. See enum above. Only relevant if the
  // number of threads is larger than 1. Applies only to the CPU kernel.
  SpreadThreading spread_threading = SpreadThreading::AUTO;

  // The maximum batch size for the vectorized NUFFT. A value of 0 means the
  // batch size is automatically chosen. Applies to CPU and GPU kernels.
  int max_batch_size = 0;

  // The number of threads above which the spreader OMP critical goes atomic.
  // Applies only to the CPU kernel.
  int num_threads_for_atomic_spread = -1;

  // The maximum spreader subproblem size. Applies only to the CPU kernel.
  int max_spread_subproblem_size = 0;

  // Do only spreading and/or interpolation (no FFT or deconvolution).
  bool spread_only = false;

  // The CUDA interpolation/spreading method.
  SpreadMethod spread_method = SpreadMethod::AUTO;

  #if GOOGLE_CUDA

  // Maximum subproblem size.
  int gpu_max_subproblem_size = 1024;

  // Used for 3D subproblem method. 0 means automatic selection.
  dim3 gpu_bin_size = {0, 0, 0};

  // Used for 3D spread-block-gather method. 0 means automatic selection.
  dim3 gpu_obin_size = {0, 0, 0};

  #endif // GOOGLE_CUDA
};

}  // namespace nufft
}  // namespace tensorflow

#endif // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_OPTIONS_H_
