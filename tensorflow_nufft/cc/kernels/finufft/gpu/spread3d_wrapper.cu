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

#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuComplex.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/memtransfer.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/precision_independent.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUSPREAD3D(Plan<GPUDevice, FLT>* d_plan, int blksize)
/*
  A wrapper for different spreading methods. 

  Methods available:
  (1) Non-uniform points driven
  (2) Subproblem
  (4) Block gather

  Melody Shih 07/25/19
*/
{

  int ier = 0;
  switch(d_plan->options_.spread_method)
  {
    case SpreadMethod::NUPTS_DRIVEN:
      {
        ier = CUSPREAD2D_NUPTSDRIVEN(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      {
        ier = CUSPREAD2D_SUBPROB(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::BLOCK_GATHER:
      {
        ier = CUSPREAD3D_BLOCKGATHER(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    default:
      cerr<<"error: incorrect method, should be 1,2,4"<<endl;
      return 2;
  }
  return ier;
}
