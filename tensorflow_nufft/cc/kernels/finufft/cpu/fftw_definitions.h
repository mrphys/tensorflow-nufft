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

#ifndef TENSORFLOW_NUFFT_KERNELS_FINUFFT_FFTW_DEFINITIONS_H
#define TENSORFLOW_NUFFT_KERNELS_FINUFFT_FFTW_DEFINITIONS_H

#include <fftw3.h>

// Here we define typedefs and macros to switch between single and double
// precision library compilation, which need different FFTW commands.


// prec-indep interfaces to FFTW and other math utilities...
#ifdef SINGLE
  typedef fftwf_complex FFTW_CPX;           //  single-prec has fftwf_*
  #define FFTW_EXECUTE fftwf_execute
  #define FFTW_FORGET_WISDOM fftwf_forget_wisdom
  #define FFTW_CLEANUP fftwf_cleanup
  #define FFTW_CLEANUP_THREADS fftwf_cleanup_threads
#else
  typedef fftw_complex FFTW_CPX;           // double-prec has fftw_*
  #define FFTW_EXECUTE fftw_execute
  #define FFTW_FORGET_WISDOM fftw_forget_wisdom
  #define FFTW_CLEANUP fftw_cleanup
  #define FFTW_CLEANUP_THREADS fftw_cleanup_threads
#endif

#endif // TENSORFLOW_NUFFT_KERNELS_FINUFFT_FFTW_DEFINITIONS_H
