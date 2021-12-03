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

#ifndef TENSORFLOW_NUFFT_KERNELS_OMP_API_H_
#define TENSORFLOW_NUFFT_KERNELS_OMP_API_H_

#ifdef _OPENMP
  // Compiled with OpenMP support.
  #include <omp.h>
  #define OMP_GET_NUM_THREADS() omp_get_num_threads()
  #define OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define OMP_GET_THREAD_NUM() omp_get_thread_num()
  #define OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
  // Compiled without OpenMP support. Create dummy versions of the OpenMP API.
  #define OMP_GET_NUM_THREADS() 1
  #define OMP_GET_MAX_THREADS() 1
  #define OMP_GET_THREAD_NUM() 0
  #define OMP_SET_NUM_THREADS(x)
#endif

#endif // TENSORFLOW_NUFFT_KERNELS_OMP_API_H_
