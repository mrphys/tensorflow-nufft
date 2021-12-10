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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuComplex.h>
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"

using namespace std;

namespace tensorflow {
namespace nufft {



int CUSPREAD2D_PAUL(Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: unimplemented error.
	return 1;
}

int CUSPREAD3D_BLOCKGATHER(
  Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: raise not implemented error
  return 1;
}



} // namespace nufft
} // namespace tensorflow

#endif // GOOGLE_CUDA
