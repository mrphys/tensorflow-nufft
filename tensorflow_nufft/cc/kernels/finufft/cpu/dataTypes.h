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

// ------------ FINUFFT data type definitions ----------------------------------

#if (!defined(DATATYPES_H) && !defined(SINGLE)) || (!defined(DATATYPESF_H) && defined(SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef SINGLE
#define DATATYPES_H
#else
#define DATATYPESF_H
#endif

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
typedef int64_t BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#define _USE_MATH_DEFINES
#include <complex>          // C++ type
#define COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h>        // C99 type
#define COMPLEXIFY(X) X complex
#endif

#undef FLT
#undef CPX

// Precision-independent real and complex types for interfacing...
// (note these cannot be typedefs since we want dual-precision library)
#ifdef SINGLE
  #define FLT float
#else
  #define FLT double
#endif

#define CPX COMPLEXIFY(FLT)

#endif  // DATATYPES_H or DATATYPESF_H
