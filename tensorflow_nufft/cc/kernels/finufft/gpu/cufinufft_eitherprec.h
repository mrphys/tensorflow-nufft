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
/* spreading 2D */

/* deconvolve */
/* structs */



#define checkCufftErrors(call)

#endif
