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

// Defines the C++/C user interface to FINUFFT library.

// It simply combines single and double precision headers, by flipping a flag
// in the main macros which are in tensorflow_nufft/cc/kernels/cufinufft/cufinufft_eitherprec.h
// No usual #ifndef testing is needed; it's done in tensorflow_nufft/cc/kernels/cufinufft/cufinufft_eitherprec.h
// Internal cufinufft routines that are compiled separately for
// each precision should include tensorflow_nufft/cc/kernels/cufinufft/cufinufft_eitherprec.h directly, and not cufinufft.h.

#undef SINGLE
#include <tensorflow_nufft/cc/kernels/cufinufft/cufinufft_eitherprec.h>
#define SINGLE
#include <tensorflow_nufft/cc/kernels/cufinufft/cufinufft_eitherprec.h>
#undef SINGLE
