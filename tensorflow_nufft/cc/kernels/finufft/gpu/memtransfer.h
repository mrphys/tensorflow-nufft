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

#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h>
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

int ALLOCGPUMEM1D_NUPTS(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
void FREEGPUMEMORY1D(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

int ALLOCGPUMEM2D_NUPTS(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
void FREEGPUMEMORY2D(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);

int ALLOCGPUMEM3D_NUPTS(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
void FREEGPUMEMORY3D(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_plan);
#endif
