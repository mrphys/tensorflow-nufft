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

#include <fftw3.h>

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/finufft_plan.h"


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

// Transform type naming by:
// Dutt A, Rokhlin V. Fast Fourier transforms for nonequispaced data. SIAM
// Journal on Scientific computing. 1993 Nov;14(6):1368-93.
enum class TransformType {
  TYPE_1, // non-uniform to uniform
  TYPE_2, // uniform to non-uniform
  TYPE_3  // non-uniform to non-uniform (not implemented)
};

typedef struct {
  FLT X1,C1,D1,h1,gam1;  // x dim: X=halfwid C=center D=freqcen h,gam=rescale
  FLT X2,C2,D2,h2,gam2;  // y
  FLT X3,C3,D3,h3,gam3;  // z
} TYPE3PARAMS;

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
  FLT tol;         // relative user tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  int nbatch;      // how many batches done to cover all ntrans vectors
  
  BIGINT ms;       // number of modes in x (1) dir (historical CMCL name) = N1
  BIGINT mt;       // number of modes in y (2) direction = N2
  BIGINT mu;       // number of modes in z (3) direction = N3
  BIGINT N;        // total # modes (prod of above three)
  
  BIGINT nf1;      // size of internal fine grid in x (1) direction
  BIGINT nf2;      // " y
  BIGINT nf3;      // " z
  BIGINT nf;       // total # fine grid points (product of the above three)
  
  int fftSign;     // sign in exponential for NUFFT defn, guaranteed to be +-1

  FLT* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FLT* phiHat2;    // " y-axis.
  FLT* phiHat3;    // " z-axis.
  
  FFTW_CPX* fwBatch;    // (batches of) fine grid(s) for FFTW to plan & act on.
                        // Usually the largest working array
  
  BIGINT *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FLT *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FLT *S, *T, *U;  // pointers to user's target NU pts arrays (no new allocs)
  CPX* prephase;   // pre-phase, for all input NU pts
  CPX* deconv;     // reciprocal of kernel FT, phase, all output NU pts
  CPX* CpBatch;    // working array of prephased strengths
  FLT *Sp, *Tp, *Up;  // internal primed targs (s'_k, etc), allocated
  TYPE3PARAMS t3P; // groups together type 3 shift, scale, phase, parameters
  
  // The FFTW plan for FFTs.
  typename FftwPlanType<FloatType>::Type fft_plan;

  spread_opts spopts;
  Options options;
};

// template<typename Device, typename FloatType>
// using PlanPtr = Plan<Device, FloatType>*;

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H
