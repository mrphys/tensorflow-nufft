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

#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h>
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

__global__
void Deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, 
	CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2);
__global__
void Amplify_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, 
	CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2);

__global__
void Deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, 
	int fw_width, CUCPX* fw, CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, 
	FLT *fwkerhalf3);
__global__
void Amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, int fw_width, 
	CUCPX* fw, CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3);

int CUDECONVOLVE2D(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_mem, int blksize);
int CUDECONVOLVE3D(tensorflow::nufft::Plan<tensorflow::GPUDevice, FLT>* d_mem, int blksize);
#endif
