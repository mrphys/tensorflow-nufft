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

#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>
#include <cufft.h>

#include "tensorflow_nufft/third_party/cuda_samples/helper_cuda.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h"
#include "cuspreadinterp.h"
#include "cudeconvolve.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;



#ifdef __cplusplus
extern "C" {
#endif



#ifdef __cplusplus
}
#endif
