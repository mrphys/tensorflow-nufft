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

#include <cuda.h>
#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "cudeconvolve.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;

/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up 
// to N/2-1)
__global__
void Deconvolve2DKernel(int ms, int mt, int nf1, int nf2, CUCPX* fw, CUCPX *fk, 
  FLT *fwkerhalf1, FLT *fwkerhalf2)
{
  for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x) {
    int k1 = i % ms;
    int k2 = i / ms;
    int outidx = k1 + k2*ms;
    int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
    int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
    int inidx = w1 + w2*nf1;

    FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
    fk[outidx].x = fw[inidx].x/kervalue;
    fk[outidx].y = fw[inidx].y/kervalue;
  }
}

__global__
void Deconvolve3DKernel(int ms, int mt, int mu, int nf1, int nf2, int nf3, CUCPX* fw, 
  CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3)
{
  for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt*mu; i+=blockDim.x*
    gridDim.x) {
    int k1 = i % ms;
    int k2 = (i / ms) % mt;
    int k3 = (i / ms / mt);
    int outidx = k1 + k2*ms + k3*ms*mt;
    int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
    int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
    int w3 = k3-mu/2 >= 0 ? k3-mu/2 : nf3+k3-mu/2;
    int inidx = w1 + w2*nf1 + w3*nf1*nf2;

    FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)]*
      fwkerhalf3[abs(k3-mu/2)];
    fk[outidx].x = fw[inidx].x/kervalue;
    fk[outidx].y = fw[inidx].y/kervalue;
    //fk[outidx].x = kervalue;
    //fk[outidx].y = kervalue;
  }
}

/* Kernel for copying fk to fw with same amplication */
__global__
void Amplify2DKernel(int ms, int mt, int nf1, int nf2, CUCPX* fw, CUCPX *fk, 
  FLT *fwkerhalf1, FLT *fwkerhalf2)
{
  for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x) {
    int k1 = i % ms;
    int k2 = i / ms;
    int inidx = k1 + k2*ms;
    int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
    int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
    int outidx = w1 + w2*nf1;

    FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
    fw[outidx].x = fk[inidx].x/kervalue;
    fw[outidx].y = fk[inidx].y/kervalue;
  }
}

__global__
void Amplify3DKernel(int ms, int mt, int mu, int nf1, int nf2, int nf3, CUCPX* fw, 
  CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3)
{
  for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt*mu; 
    i+=blockDim.x*gridDim.x) {
    int k1 = i % ms;
    int k2 = (i / ms) % mt;
    int k3 = (i / ms / mt);
    int inidx = k1 + k2*ms + k3*ms*mt;
    int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
    int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
    int w3 = k3-mu/2 >= 0 ? k3-mu/2 : nf3+k3-mu/2;
    int outidx = w1 + w2*nf1 + w3*nf1*nf2;

    FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)]*
      fwkerhalf3[abs(k3-mu/2)];
    fw[outidx].x = fk[inidx].x/kervalue;
    fw[outidx].y = fk[inidx].y/kervalue;
    //fw[outidx].x = fk[inidx].x;
    //fw[outidx].y = fk[inidx].y;
  }
}


int CUDECONVOLVE2D(Plan<GPUDevice, FLT>* d_plan, int blksize)
/* 
  wrapper for deconvolution & amplication in 2D.

  Melody Shih 07/25/19
*/
{
  int threads_per_block = 256;
  int num_blocks = (d_plan->mode_count_ + threads_per_block - 1) / threads_per_block;

  if (d_plan->spread_params_.spread_direction == SpreadDirection::SPREAD) {
    
    switch (d_plan->rank_) {
      case 2:
        for (int t=0; t<blksize; t++) {
          Deconvolve2DKernel<<<num_blocks, threads_per_block>>>(
            d_plan->num_modes_[0], d_plan->num_modes_[1],
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], 
            d_plan->fine_grid_data_ + t * d_plan->grid_size_,
            d_plan->fk + t * d_plan->mode_count_, d_plan->kernel_fseries_data_[0], 
            d_plan->kernel_fseries_data_[1]);
        }
        break;
      case 3:
        for (int t=0; t<blksize; t++) {
          Deconvolve3DKernel<<<num_blocks, threads_per_block>>>(
            d_plan->num_modes_[0], d_plan->num_modes_[1], d_plan->num_modes_[2],
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], d_plan->grid_dims_[2],
            d_plan->fine_grid_data_ + t * d_plan->grid_size_,
            d_plan->fk + t * d_plan->mode_count_, 
            d_plan->kernel_fseries_data_[0], d_plan->kernel_fseries_data_[1], d_plan->kernel_fseries_data_[2]);
        }
        break;
    }
  } else {
    checkCudaErrors(cudaMemset(d_plan->fine_grid_data_,0,d_plan->options_.max_batch_size*d_plan->grid_size_*
      sizeof(CUCPX)));
    switch (d_plan->rank_) {
      case 2:
        for (int t=0; t<blksize; t++) {
          Amplify2DKernel<<<num_blocks, threads_per_block>>>(d_plan->num_modes_[0], 
            d_plan->num_modes_[1], d_plan->grid_dims_[0], d_plan->grid_dims_[1],
            d_plan->fine_grid_data_ + t * d_plan->grid_size_,
            d_plan->fk + t * d_plan->mode_count_,
            d_plan->kernel_fseries_data_[0], d_plan->kernel_fseries_data_[1]);
        }
        break;
      case 3:
        for (int t=0; t<blksize; t++) {
          Amplify3DKernel<<<num_blocks, threads_per_block>>>(d_plan->num_modes_[0],
            d_plan->num_modes_[1], d_plan->num_modes_[2],
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], d_plan->grid_dims_[2],
            d_plan->fine_grid_data_ + t * d_plan->grid_size_,
            d_plan->fk + t * d_plan->mode_count_, 
            d_plan->kernel_fseries_data_[0], d_plan->kernel_fseries_data_[1],
            d_plan->kernel_fseries_data_[2]);
        }
        break;
    }
  }
  return 0;
}
