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

// Switchable-precision interface template for cufinufft. Used by cufinufft.h
// Internal use only: users should link to cufinufft.h

#if (!defined(__CUFINUFFT_H__) && !defined(SINGLE)) || \
  (!defined(__CUFINUFFTF_H__) && defined(SINGLE))
// (note we entered one level of conditional until the end of this header)
// Make sure we don't include double or single headers more than once...

#include <cstdlib>
#include <cufft.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/precision_independent.h"
#include "cufinufft_errors.h"

#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/dataTypes.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/spreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils_fp.h"
#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


#ifndef SINGLE
#define __CUFINUFFT_H__
#else
#define __CUFINUFFTF_H__
#endif


/* Undefine things so we don't get warnings/errors later */
#undef CUFINUFFT_EXECUTE
#undef CUFINUFFT_INTERP
#undef CUFINUFFT_SPREAD
#undef CUFINUFFT2D1_EXEC
#undef CUFINUFFT2D2_EXEC
#undef CUFINUFFT3D1_EXEC
#undef CUFINUFFT3D2_EXEC
#undef CUFINUFFT2D_INTERP
#undef CUFINUFFT2D_SPREAD
#undef CUFINUFFT3D_INTERP
#undef CUFINUFFT3D_SPREAD
/* spreading 2D */
#undef CUSPREAD2D
#undef CUSPREAD2D_NUPTSDRIVEN
#undef CUSPREAD2D_SUBPROB
#undef CUSPREAD2D_PAUL
/* spreading 3d */
#undef CUSPREAD3D_BLOCKGATHER
/* interp */
#undef CUINTERP2D
#undef CUINTERP2D_NUPTSDRIVEN
#undef CUINTERP2D_SUBPROB
/* deconvolve */
#undef CUDECONVOLVE2D
#undef CUDECONVOLVE3D
/* structs */


#ifdef SINGLE

#define CUFINUFFT_EXECUTE cufinufftf_execute
#define CUFINUFFT_INTERP cufinufftf_interp
#define CUFINUFFT_SPREAD cufinufftf_spread
#define CUFINUFFT2D1_EXEC cufinufftf2d1_exec
#define CUFINUFFT2D2_EXEC cufinufftf2d2_exec
#define CUFINUFFT3D1_EXEC cufinufftf3d1_exec
#define CUFINUFFT3D2_EXEC cufinufftf3d2_exec
#define CUFINUFFT2D_INTERP cufinufftf2d_interp
/* spreading 2D */
#define CUSPREAD2D cuspread2df
#define CUSPREAD2D_NUPTSDRIVEN cuspread2df_nuptsdriven
#define CUSPREAD2D_SUBPROB cuspread2df_subprob
#define CUSPREAD2D_PAUL cuspread2df_paul
#define CUSPREAD3D_BLOCKGATHER cuspread3df_blockgather
/* interp */
#define CUINTERP2D cuinterp2df
#define CUINTERP2D_NUPTSDRIVEN cuinterp2df_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2df_subprob
/* deconvolve */
#define CUDECONVOLVE2D cudeconvolve2df
#define CUDECONVOLVE3D cudeconvolve3df
/* structs */

#else

#define CUFINUFFT_EXECUTE cufinufft_execute
#define CUFINUFFT_INTERP cufinufft_interp
#define CUFINUFFT_SPREAD cufinufft_spread
#define CUFINUFFT2D1_EXEC cufinufft2d1_exec
#define CUFINUFFT2D2_EXEC cufinufft2d2_exec
#define CUFINUFFT3D1_EXEC cufinufft3d1_exec
#define CUFINUFFT3D2_EXEC cufinufft3d2_exec
#define CUFINUFFT2D_INTERP cufinufft2d_interp
#define CUFINUFFT2D_SPREAD cufinufft2d_spread
#define CUFINUFFT3D_INTERP cufinufft3d_interp
#define CUFINUFFT3D_SPREAD cufinufft3d_spread
/* spreading 2D */
#define CUSPREAD2D cuspread2d
#define CUSPREAD2D_NUPTSDRIVEN cuspread2d_nuptsdriven
#define CUSPREAD2D_SUBPROB cuspread2d_subprob
#define CUSPREAD2D_PAUL cuspread2d_paul
#define CUSPREAD3D_BLOCKGATHER cuspread3d_blockgather
#define CUINTERP2D cuinterp2d
#define CUINTERP2D_NUPTSDRIVEN cuinterp2d_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2d_subprob
/* deconvolve */
#define CUDECONVOLVE2D cudeconvolve2d
#define CUDECONVOLVE3D cudeconvolve3d


#endif

#define checkCufftErrors(call)

#ifdef __cplusplus
extern "C" {
#endif
int CUFINUFFT_EXECUTE(CUCPX* h_c, CUCPX* h_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT_INTERP(CUCPX* h_c, CUCPX* h_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT_SPREAD(CUCPX* h_c, CUCPX* h_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
#ifdef __cplusplus
}
#endif


// 2d
int CUFINUFFT2D1_EXEC(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT2D2_EXEC(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

// 3d
int CUFINUFFT3D1_EXEC(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT3D2_EXEC(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

// 2d
int CUFINUFFT2D_INTERP(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT2D_SPREAD(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

// 3d
int CUFINUFFT3D_INTERP(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
int CUFINUFFT3D_SPREAD(CUCPX* d_c, CUCPX* d_fk, tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

#endif
