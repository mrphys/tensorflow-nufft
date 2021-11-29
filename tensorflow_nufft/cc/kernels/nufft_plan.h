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

#ifndef TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H
#define TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H

#include <cstdint>
#include <fftw3.h>

#include "tensorflow_nufft/cc/kernels/nufft_options.h"


namespace tensorflow {
namespace nufft {

template<typename FloatType>
struct FftwPlanType;

template<>
struct FftwPlanType<float> {
  typedef fftwf_plan Type;
};

template<>
struct FftwPlanType<double> {
  typedef fftw_plan Type;
};

template<typename FloatType>
struct FftwComplexType;

template<>
struct FftwComplexType<float> {
  typedef fftwf_complex Type;
};

template<>
struct FftwComplexType<double> {
  typedef fftw_complex Type;
};

// Transform type naming by:
// Dutt A, Rokhlin V. Fast Fourier transforms for nonequispaced data. SIAM
// Journal on Scientific computing. 1993 Nov;14(6):1368-93.
enum class TransformType {
  TYPE_1, // non-uniform to uniform
  TYPE_2, // uniform to non-uniform
  TYPE_3  // non-uniform to non-uniform (not implemented)
};

template<typename FloatType>
struct SpreadOptions {  // see spreadinterp:setup_spreader for defaults.
  // This is the main documentation for these options...
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
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
  bool spread_interp_only;   // 0: NUFFT, 1: spread or interpolation only
  // ES kernel specific consts used in fast eval, depend on precision FLT...
  FloatType ES_beta;
  FloatType ES_halfwidth;
  FloatType ES_c;
  FloatType ES_scale;           // used for spread/interp only
};

template<typename Device, typename FloatType>
class Plan {

 public:
  
  // // The type of the transform. See enum above.
  // TransformType type;

  // // The rank of the transform (number of dimensions). Must be 1, 2 or 3.
  // unsigned int rank;

  // // How many transforms to compute.
  // unsigned int num_transforms;

  // // Number of non-uniform points.
  // unsigned int num_points;

  // // Relative tolerance.
  // FloatType tolerance;

  int type;        // transform type (Rokhlin naming): 1,2 or 3
  int dim;         // overall dimension: 1,2 or 3
  int ntrans;      // how many transforms to do at once (vector or "many" mode)
  int nj;          // number of NU pts in type 1,2 (for type 3, num input x pts)
  int nk;          // number of NU freq pts (type 3 only)
  FloatType tol;         // relative user tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  int nbatch;      // how many batches done to cover all ntrans vectors
  
  int64_t ms;       // number of modes in x (1) dir (historical CMCL name) = N1
  int64_t mt;       // number of modes in y (2) direction = N2
  int64_t mu;       // number of modes in z (3) direction = N3
  int64_t N;        // total # modes (prod of above three)

  int64_t nf1;      // size of internal fine grid in x (1) direction
  int64_t nf2;      // " y
  int64_t nf3;      // " z
  int64_t nf;       // total # fine grid points (product of the above three)
  
  int fftSign;     // sign in exponential for NUFFT defn, guaranteed to be +-1

  FloatType* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FloatType* phiHat2;    // " y-axis.
  FloatType* phiHat3;    // " z-axis.
  
  typename FftwComplexType<FloatType>::Type* fwBatch;    // (batches of) fine grid(s) for FFTW to plan & act on.
                        // Usually the largest working array
  
  int64_t *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FloatType *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FloatType *S, *T, *U;  // pointers to user's target NU pts arrays (no new allocs)
  // CPX* prephase;   // pre-phase, for all input NU pts
  // CPX* deconv;     // reciprocal of kernel FT, phase, all output NU pts
  // CPX* CpBatch;    // working array of prephased strengths
  FloatType *Sp, *Tp, *Up;  // internal primed targs (s'_k, etc), allocated
  // TYPE3PARAMS t3P; // groups together type 3 shift, scale, phase, parameters
  
  // The FFTW plan for FFTs.
  typename FftwPlanType<FloatType>::Type fft_plan;
  SpreadOptions<FloatType> spopts;
  Options options;
};

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H
