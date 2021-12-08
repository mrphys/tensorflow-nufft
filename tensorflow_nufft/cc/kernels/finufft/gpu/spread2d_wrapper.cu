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

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUSPREAD2D(Plan<GPUDevice, FLT>* d_plan, int blksize)
/*
  A wrapper for different spreading methods.

  Methods available:
  (1) Non-uniform points driven
  (2) Subproblem
  (3) Paul

  Melody Shih 07/25/19
*/
{
  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int M = d_plan->M;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int ier;
  switch(d_plan->options_.spread_method)
  {
    case SpreadMethod::NUPTS_DRIVEN:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_NUPTSDRIVEN(nf1, nf2, M, d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_SUBPROB(nf1, nf2, M, d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::PAUL:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_PAUL(nf1, nf2, M, d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_paul"<<endl;
          return 1;
        }
      }
      break;
    default:
      cout<<"error: incorrect method, should be 1,2,3"<<endl;
      return 2;
  }
#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"[time  ]"<< " Spread " << milliseconds <<" ms"<<endl;
#endif
  return ier;
}

int CUSPREAD2D_NUPTSDRIVEN_PROP(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan)
{
  if (d_plan->spread_params_.sort_points == SortPoints::YES) {

    int bin_size[2];
    bin_size[0] = d_plan->options_.gpu_bin_size.x;
    bin_size[1] = d_plan->options_.gpu_bin_size.y;
    if (bin_size[0] < 0 || bin_size[1] < 0) {
      cout << "error: invalid binsize (binsizex, binsizey) = (";
      cout << bin_size[0] << "," << bin_size[1] << ")" << endl;
      return 1; 
    }

    int numbins[2];
    numbins[0] = ceil((FLT) nf1 / bin_size[0]);
    numbins[1] = ceil((FLT) nf2 / bin_size[1]);

    FLT*   d_kx = d_plan->kx;
    FLT*   d_ky = d_plan->ky;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_idxnupts = d_plan->idxnupts;

    int pirange = d_plan->spread_params_.pirange;

    // Synchronize device before we start. This is essential! Otherwise the
    // next kernel could read the wrong (kx, ky, kz) values.
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*
      sizeof(int)));
    CalcBinSize_noghost_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2,
      bin_size[0], bin_size[1], numbins[0], numbins[1],
      d_binsize, d_kx, d_ky, d_sortidx, pirange);

    int n=numbins[0]*numbins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

    CalcInvertofGlobalSortIdx_2d<<<(M + 1024 - 1) / 1024, 1024>>>(
      M, bin_size[0], bin_size[1], numbins[0], numbins[1],
      d_binstartpts, d_sortidx, d_kx, d_ky,
      d_idxnupts, pirange, nf1, nf2);

  }else{
    int *d_idxnupts = d_plan->idxnupts;

    TrivialGlobalSortIdx_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, d_idxnupts);
  }
  return 0;
}

int CUSPREAD2D_NUPTSDRIVEN(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
  int blksize)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 threadsPerBlock;
  dim3 blocks;

  int ns=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
  int pirange=d_plan->spread_params_.pirange;
  int *d_idxnupts=d_plan->idxnupts;
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  FLT sigma=d_plan->spread_params_.upsampling_factor;

  FLT* d_kx = d_plan->kx;
  FLT* d_ky = d_plan->ky;
  CUCPX* d_c = d_plan->c;
  CUCPX* d_fw = d_plan->fine_grid_data_;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
  blocks.y = 1;
  cudaEventRecord(start);
  if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
    for (int t=0; t<blksize; t++) {
      Spread_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx,
        d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma,
        d_idxnupts, pirange);
    }
  } else {
    for (int t=0; t<blksize; t++) {
      Spread_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky,
        d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, es_beta,
        d_idxnupts, pirange);
    }
  }

#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel Spread_2d_NUptsdriven (%d)\t%.3g ms\n",
    milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
  return 0;
}
int CUSPREAD2D_SUBPROB_PROP(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan)
/*
  This function determines the properties for spreading that are independent
  of the strength of the nodes,  only relates to the locations of the nodes,
  which only needs to be done once.
*/
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;
  int bin_size_x=d_plan->options_.gpu_bin_size.x;
  int bin_size_y=d_plan->options_.gpu_bin_size.y;
  if (bin_size_x < 0 || bin_size_y < 0) {
    cout<<"error: invalid binsize (binsizex, binsizey) = (";
    cout<<bin_size_x<<","<<bin_size_y<<")"<<endl;
    return 1; 
  }
  int numbins[2];
  numbins[0] = ceil((FLT) nf1/bin_size_x);
  numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef DEBUG
  cout<<"[debug  ] Dividing the uniform grids to bin size["
    <<d_plan->options_.gpu_bin_size.x<<"x"<<d_plan->options_.gpu_bin_size.y<<"]"<<endl;
  cout<<"[debug  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

  FLT*   d_kx = d_plan->kx;
  FLT*   d_ky = d_plan->ky;

#ifdef DEBUG
  FLT *h_kx;
  FLT *h_ky;
  h_kx = (FLT*)malloc(M*sizeof(FLT));
  h_ky = (FLT*)malloc(M*sizeof(FLT));

  checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),cudaMemcpyDeviceToHost));
  for (int i=0; i<M; i++) {
    cout<<"[debug ]";
    cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<")"<<endl;
  }
#endif
  int *d_binsize = d_plan->binsize;
  int *d_binstartpts = d_plan->binstartpts;
  int *d_sortidx = d_plan->sortidx;
  int *d_numsubprob = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  int pirange=d_plan->spread_params_.pirange;

  // Synchronize device before we start. This is essential! Otherwise the
  // next kernel could read the wrong (kx, ky, kz) values.
  checkCudaErrors(cudaDeviceSynchronize());

  cudaEventRecord(start);
  checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
  CalcBinSize_noghost_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,
    bin_size_y,numbins[0],numbins[1],d_binsize,d_kx,d_ky,d_sortidx,pirange);
#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n",
    milliseconds);
#endif
#ifdef DEBUG
  int *h_binsize;// For debug
  h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
    sizeof(int),cudaMemcpyDeviceToHost));
  cout<<"[debug ] bin size:"<<endl;
  for (int j=0; j<numbins[1]; j++) {
    cout<<"[debug ] ";
    for (int i=0; i<numbins[0]; i++) {
      if (i!=0) cout<<" ";
      cout <<" bin["<<setw(3)<<i<<","<<setw(3)<<j<<"]="<<
        h_binsize[i+j*numbins[0]];
    }
    cout<<endl;
  }
  free(h_binsize);
  cout<<"[debug ] ----------------------------------------------------"<<endl;
#endif
#ifdef DEBUG
  int *h_sortidx;
  h_sortidx = (int*)malloc(M*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
    cudaMemcpyDeviceToHost));
  cout<<"[debug ]";
  for (int i=0; i<M; i++) {
    cout <<"[debug] point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
  }

#endif

  cudaEventRecord(start);
  int n=numbins[0]*numbins[1];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
  int *h_binstartpts;
  h_binstartpts = (int*)malloc((numbins[0]*numbins[1])*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,
        (numbins[0]*numbins[1])*sizeof(int),
        cudaMemcpyDeviceToHost));
  cout<<"[debug ] Result of scan bin_size array:"<<endl;
  for (int j=0; j<numbins[1]; j++) {
    cout<<"[debug ] ";
    for (int i=0; i<numbins[0]; i++) {
      if (i!=0) cout<<" ";
      cout <<"bin["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)
        <<h_binstartpts[i+j*numbins[0]];
    }
    cout<<endl;
  }
  free(h_binstartpts);
  cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
  cudaEventRecord(start);
  CalcInvertofGlobalSortIdx_2d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
    bin_size_y,numbins[0],numbins[1],d_binstartpts,d_sortidx,d_kx,d_ky,
    d_idxnupts,pirange,nf1,nf2);
#ifdef DEBUG
  int *h_idxnupts;
  h_idxnupts = (int*)malloc(M*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
        cudaMemcpyDeviceToHost));
  for (int i=0; i<M; i++) {
    cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
  }
  free(h_idxnupts);
#endif
  cudaEventRecord(start);
  CalcSubProb_2d<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,
    maxsubprobsize,numbins[0]*numbins[1]);
#ifdef SPREADTIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel CalcSubProb_2d\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
  int* h_numsubprob;
  h_numsubprob = (int*) malloc(n*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins[0]*numbins[1]*
        sizeof(int),cudaMemcpyDeviceToHost));
  for (int j=0; j<numbins[1]; j++) {
    cout<<"[debug ] ";
    for (int i=0; i<numbins[0]; i++) {
      if (i!=0) cout<<" ";
      cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<
        h_numsubprob[i+j*numbins[0]];
    }
    cout<<endl;
  }
  free(h_numsubprob);
#endif
  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts+1);
  thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
  checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));
#ifdef SPREADTIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel Scan Subprob array\t\t%.3g ms\n", milliseconds);
#endif

#ifdef DEBUG
  printf("[debug ] Subproblem start points\n");
  int* h_subprobstartpts;
  h_subprobstartpts = (int*) malloc((n+1)*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_subprobstartpts,d_subprobstartpts,
        (n+1)*sizeof(int),cudaMemcpyDeviceToHost));
  for (int j=0; j<numbins[1]; j++) {
    cout<<"[debug ] ";
    for (int i=0; i<numbins[0]; i++) {
      if (i!=0) cout<<" ";
      cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<
        h_subprobstartpts[i+j*numbins[0]];
    }
    cout<<endl;
  }
  printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
  free(h_subprobstartpts);
#endif
  cudaEventRecord(start);
  int totalnumsubprob;
  checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
    sizeof(int),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
  MapBintoSubProb_2d<<<(numbins[0]*numbins[1]+1024-1)/1024, 1024>>>(
      d_subprob_to_bin,d_subprobstartpts,d_numsubprob,numbins[0]*numbins[1]);
  assert(d_subprob_to_bin != NULL);
        if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin = d_subprob_to_bin;
  assert(d_plan->subprob_to_bin != NULL);
  d_plan->totalnumsubprob = totalnumsubprob;
#ifdef DEBUG
  printf("[debug ] Map Subproblem to Bins\n");
  int* h_subprob_to_bin;
  h_subprob_to_bin = (int*) malloc((totalnumsubprob)*sizeof(int));
  checkCudaErrors(cudaMemcpy(h_subprob_to_bin,d_subprob_to_bin,
        (totalnumsubprob)*sizeof(int),cudaMemcpyDeviceToHost));
  for (int j=0; j<totalnumsubprob; j++) {
    cout<<"[debug ] ";
    cout <<"nsub["<<j<<"] = "<<setw(2)<<h_subprob_to_bin[j];
    cout<<endl;
  }
  free(h_subprob_to_bin);
#endif
#ifdef SPREADTIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
  return 0;
}

int CUSPREAD2D_SUBPROB(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
  int blksize)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int ns=d_plan->spread_params_.nspread;// psi's support in terms of number of cells
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

  // assume that bin_size_x > ns/2;
  int bin_size_x=d_plan->options_.gpu_bin_size.x;
  int bin_size_y=d_plan->options_.gpu_bin_size.y;
  int numbins[2];
  numbins[0] = ceil((FLT) nf1/bin_size_x);
  numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
  cout<<"[info  ] Dividing the uniform grids to bin size["
    <<d_plan->options_.gpu_bin_size.x<<"x"<<d_plan->options_.gpu_bin_size.y<<"]"<<endl;
  cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

  FLT* d_kx = d_plan->kx;
  FLT* d_ky = d_plan->ky;
  CUCPX* d_c = d_plan->c;
  CUCPX* d_fw = d_plan->fine_grid_data_;

  int *d_binsize = d_plan->binsize;
  int *d_binstartpts = d_plan->binstartpts;
  int *d_numsubprob = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts = d_plan->idxnupts;

  int totalnumsubprob=d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  int pirange=d_plan->spread_params_.pirange;

  FLT sigma=d_plan->options_.upsampling_factor;
  cudaEventRecord(start);

  size_t sharedplanorysize = (bin_size_x+2*(int)ceil(ns/2.0))*
                 (bin_size_y+2*(int)ceil(ns/2.0))*
                 sizeof(CUCPX);
  if (sharedplanorysize > 49152) {
    cout<<"error: not enough shared memory"<<endl;
    return 1;
  }

  if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
    for (int t=0; t<blksize; t++) {
      Spread_2d_Subprob_Horner<<<totalnumsubprob, 256,
        sharedplanorysize>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M,
        ns, nf1, nf2, sigma, d_binstartpts, d_binsize, bin_size_x,
        bin_size_y, d_subprob_to_bin, d_subprobstartpts,
        d_numsubprob, maxsubprobsize,numbins[0],numbins[1],
        d_idxnupts, pirange);
    }
  }else{
    for (int t=0; t<blksize; t++) {
      Spread_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
        d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2,
        es_c, es_beta, sigma,d_binstartpts, d_binsize, bin_size_x,
        bin_size_y, d_subprob_to_bin, d_subprobstartpts,
        d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
        d_idxnupts, pirange);
    }
  }
#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel Spread_2d_Subprob (%d)\t\t%.3g ms\n",
    milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
  return 0;
}
