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

// test-wide definitions and headers for use in ../test/*.cpp and ../perftest

#ifndef TEST_DEFS_H
#define TEST_DEFS_H

// responds to SINGLE, and defines FINUFFT?D? used in test/*.cpp
#include <finufft_eitherprec.h>

// convenient finufft internals
#include <utils.h>
#include <utils_precindep.h>
#include <defs.h>

// std stuff
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

// how big a problem to check direct DFT for in 1D...
#define TEST_BIGPROB 1e8

// for omp rand filling
#define TEST_RANDCHUNK 1000000

#endif   // TEST_DEFS_H
