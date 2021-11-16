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

// Header for utils_precindep.cpp, a little library of array and timer stuff.
// Only the precision-independent routines here (get compiled once)

#ifndef UTILS_PRECINDEP_H
#define UTILS_PRECINDEP_H

#include "tensorflow_nufft/cc/kernels/finufft/dataTypes.h"

BIGINT next235even(BIGINT n);

// jfm's timer class
#include <sys/time.h>
class CNTime {
 public:
  void start();
  double restart();
  double elapsedsec();
 private:
  struct timeval initial;
};

// openmp helpers
int get_num_threads_parallel_block();

// thread-safe rand number generator for Windows platform
#ifdef _WIN32
#include <random>
int rand_r(unsigned int *seedp);
#endif

#endif  // UTILS_PRECINDEP_H
