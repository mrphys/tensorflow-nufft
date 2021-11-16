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

// Defines C++/C user interface to the FINUFFT plan structs in both precisions:
// it provides finufft_plan (double-prec) and finufftf_plan (single-prec).
// Barnett 7/5/20

// save whether SINGLE defined or not...
#ifdef SINGLE
#define WAS_SINGLE
#endif

#undef SINGLE
#include "tensorflow_nufft/cc/kernels/finufft/finufft_plan_eitherprec.h"
#define SINGLE
#include "tensorflow_nufft/cc/kernels/finufft/finufft_plan_eitherprec.h"
#undef SINGLE

// ... and reconstruct it. (We still clobber the unlikely WAS_SINGLE symbol)
#ifdef WAS_SINGLE
#define SINGLE
#endif
