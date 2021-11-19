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

// Switchable-precision interface template for FINUFFT. Used by finufft.h
// Internal use only: users should link to finufft.h
// Barnett 7/1/20

#if (!defined(FINUFFT_H) && !defined(SINGLE)) || (!defined(FINUFFTF_H) && defined(SINGLE))
// (note we entered one level of conditional until the end of this header)
// Make sure we don't include double and single headers more than once each...
#ifndef SINGLE
#define FINUFFT_H
#else
#define FINUFFTF_H
#endif

// Here just what's needed to describe the headers for what finufft provides
#include "tensorflow_nufft/cc/kernels/finufft/cpu/dataTypes.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/nufft_opts.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/finufft_plan_eitherprec.h"

// clear the macros so we can define w/o warnings...
#undef FINUFFT_DEFAULT_OPTS
#undef FINUFFT_MAKEPLAN
#undef FINUFFT_SETPTS
#undef FINUFFT_EXECUTE
#undef FINUFFT_INTERP
#undef FINUFFT_SPREAD
#undef FINUFFT_DESTROY
// precision-switching macros for interfaces FINUFFT provides to outside world
#ifdef SINGLE
#define FINUFFT_DEFAULT_OPTS finufftf_default_opts
#define FINUFFT_MAKEPLAN finufftf_makeplan
#define FINUFFT_SETPTS finufftf_setpts
#define FINUFFT_EXECUTE finufftf_execute
#define FINUFFT_INTERP finufftf_interp
#define FINUFFT_SPREAD finufftf_spread
#define FINUFFT_DESTROY finufftf_destroy
#else
#define FINUFFT_DEFAULT_OPTS finufft_default_opts
#define FINUFFT_MAKEPLAN finufft_makeplan
#define FINUFFT_SETPTS finufft_setpts
#define FINUFFT_EXECUTE finufft_execute
#define FINUFFT_INTERP finufft_interp
#define FINUFFT_SPREAD finufft_spread
#define FINUFFT_DESTROY finufft_destroy
#endif


// all interfaces are C-style even when used from C++...
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ the guru interface ------------------------------------
// (sources in finufft.cpp)
  
void FINUFFT_DEFAULT_OPTS(nufft_opts *o);
int FINUFFT_MAKEPLAN(int type, int dim, BIGINT* n_modes, int iflag, int n_transf, FLT tol, FINUFFT_PLAN* plan, nufft_opts* o);
int FINUFFT_SETPTS(FINUFFT_PLAN plan , BIGINT M, FLT *xj, FLT *yj, FLT *zj, BIGINT N, FLT *s, FLT *t, FLT *u); 
int FINUFFT_EXECUTE(FINUFFT_PLAN plan, CPX* weights, CPX* result);
int FINUFFT_INTERP(FINUFFT_PLAN plan, CPX* weights, CPX* result);
int FINUFFT_SPREAD(FINUFFT_PLAN plan, CPX* weights, CPX* result);
int FINUFFT_DESTROY(FINUFFT_PLAN plan);

  
#ifdef __cplusplus
}
#endif

#endif   // FINUFFT_H or FINUFFTF_H
