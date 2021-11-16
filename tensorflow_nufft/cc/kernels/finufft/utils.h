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

// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#ifndef UTILS_H
#define UTILS_H

#include "tensorflow_nufft/cc/kernels/finufft/dataTypes.h"

// ahb's low-level array helpers
FLT relerrtwonorm(BIGINT n, CPX* a, CPX* b);
FLT errtwonorm(BIGINT n, CPX* a, CPX* b);
FLT twonorm(BIGINT n, CPX* a);
FLT infnorm(BIGINT n, CPX* a);
void arrayrange(BIGINT n, FLT* a, FLT *lo, FLT *hi);
void indexedarrayrange(BIGINT n, BIGINT* i, FLT* a, FLT *lo, FLT *hi);
void arraywidcen(BIGINT n, FLT* a, FLT *w, FLT *c);

#endif  // UTILS_H
