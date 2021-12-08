/* Copyright 2021 University College London. All Rights Reserved.

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

#include "tensorflow_nufft/cc/kernels/nufft_spread.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace nufft {

// __global__ void CalcBinSizeNoGhost2D(
//     int M, int nf1, int this->grid_dims_[1], int  bin_size_x, int bin_size_y,
//     int nbinx, int nbiny, int* bin_size, FLT *x, FLT *y, 
//     int* sortidx, int pirange)
// {
//   int binidx, binx, biny;
//   int oldidx;
//   FLT x_rescaled,y_rescaled;
//   for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
//     x_rescaled=RESCALE(x[i], nf1, pirange);
//     y_rescaled=RESCALE(y[i], this->grid_dims_[1], pirange);
//     binx = floor(x_rescaled/bin_size_x);
//     binx = binx >= nbinx ? binx-1 : binx;
//     binx = binx < 0 ? 0 : binx;
//     biny = floor(y_rescaled/bin_size_y);
//     biny = biny >= nbiny ? biny-1 : biny;
//     biny = biny < 0 ? 0 : biny;
//     binidx = binx+biny*nbinx;
//     oldidx = atomicAdd(&bin_size[binidx], 1);
//     sortidx[i] = oldidx;
//     if (binx >= nbinx || biny >= nbiny) {
//       sortidx[i] = -biny;
//     }
//   }
// }

// __global__ void CalcInvertofGlobalSortIdx2D(
//     int M, int bin_size_x, int bin_size_y, 
//     int nbinx,int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *y, 
//     int* index, int pirange, int nf1, int this->grid_dims_[1])
// {
//   int binx, biny;
//   int binidx;
//   FLT x_rescaled, y_rescaled;
//   for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
//     x_rescaled=RESCALE(x[i], nf1, pirange);
//     y_rescaled=RESCALE(y[i], this->grid_dims_[1], pirange);
//     binx = floor(x_rescaled/bin_size_x);
//     binx = binx >= nbinx ? binx-1 : binx;
//     binx = binx < 0 ? 0 : binx;
//     biny = floor(y_rescaled/bin_size_y);
//     biny = biny >= nbiny ? biny-1 : biny;
//     biny = biny < 0 ? 0 : biny;
//     binidx = binx+biny*nbinx;

//     index[bin_startpts[binidx]+sortidx[i]] = i;
//   }
// }

// template<typename FloatType>
// Status Spreader<GPUDevice, FloatType>::initialize(
//     int rank, int* grid_dims, int num_points,
//     FloatType* points_x, FloatType* points_y, FloatType* points_z,
//     const SpreadParameters<FloatType> params) {

//   if (rank < 2 || rank > 3) {
//     return errors::InvalidArgument("rank must be 2 or 3, got ", rank);
//   }

//   this->rank_ = rank;
//   for (int i = 0; i < 3; i++) {
//     this->grid_dims_[i] = i < rank ? grid_dims[i] : 1;
//   }
//   this->num_points_ = num_points;
//   this->points_[0] = points_x;
//   this->points_[1] = points_y;
//   this->points_[2] = points_z;
//   this->params_ = params;

//   // Allocate some arrays.
//   if (this->indices_points_) {
//     this->device_.deallocate(this->indices_points_);
//   }
//   if (this->sort_indices_) {
//     this->device_.deallocate(this->sort_indices_);
//   }
  
//   size_t num_bytes = sizeof(int) * this->num_points_;
//   switch (this->params_.spread_method) {
//     case SpreadMethod::NUPTS_DRIVEN:
//       this->indices_points_ = reinterpret_cast<int*>(
//           this->device_.allocate(num_bytes));
//       if (this->params_.sort_points == SortPoints::YES) {
//         this->sort_indices_ = reinterpret_cast<int*>(
//             this->device_.allocate(num_bytes));
//       } else {
//         this->sort_indices_ = nullptr;
//       }
//       break;
//     case SpreadMethod::SUBPROBLEM:
//     case SpreadMethod::PAUL:
//       this->indices_points_ = reinterpret_cast<int*>(
//           this->device_.allocate(num_bytes));
//       this->sort_indices_ = reinterpret_cast<int*>(
//           this->device_.allocate(num_bytes));
//       break;
//     case SpreadMethod::BLOCK_GATHER:
//       this->indices_points_ = nullptr;
//       this->sort_indices_ = reinterpret_cast<int*>(
//           this->device_.allocate(num_bytes));
//       break;
//   }

//   switch (this->params_.spread_method) {
//     case SpreadMethod::NUPTS_DRIVEN:

//       if (this->params_.sort_points == SortPoints::YES) {

//         int bin_size[2];
//         bin_size[0] = this->params_.gpu_bin_size.x;
//         bin_size[1] = this->params_.gpu_bin_size.y;
//         if (bin_size[0] < 0 || bin_size[1] < 0) {
//           return errors::Internal(
//               "gpu_bin_size must be >= 0, got ", bin_size[0], ",", bin_size[1]);
//         }

//         int num_bins[2];
//         num_bins[0] = ceil(static_cast<FloatType>(
//             this->grid_dims_[0] / bin_size[0]));
//         num_bins[1] = ceil(static_cast<FloatType>(
//             this->grid_dims_[1] / bin_size[1]));

//         int *d_binsize = d_plan->binsize;
//         int *d_binstartpts = d_plan->binstartpts;
//         int *d_sortidx = d_plan->sortidx;
//         int *d_idxnupts = d_plan->idxnupts;

//         int pirange = this->params_.pirange;

//         // Synchronize device before we start. Otherwise the next kernel could
//         // read the wrong (kx, ky, kz) values.
//         // TODO: is this really necessary?
//         this->device_.synchronize();

//         this->device_.memset(d_binsize, 0,
//                              num_bins[0] * num_bins[1] * sizeof(int));
//         CalcBinSizeNoGhost2D<<<(this->num_points_ + 1024 - 1) / 1024, 1024>>>(
//             this->num_points_, this->grid_dims_[0], this->grid_dims_[1],
//             bin_size[0], bin_size[1], num_bins[0], num_bins[1],
//             d_binsize, this->points_[0], this->points_[1], d_sortidx, pirange);

//         int total_bins = num_bins[0] * num_bins[1];
//         thrust::device_ptr<int> d_ptr(d_binsize);
//         thrust::device_ptr<int> d_result(d_binstartpts);
//         thrust::exclusive_scan(d_ptr, d_ptr + total_bins, d_result);

//         CalcInvertofGlobalSortIdx2D<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size[0],
//           bin_size[1], num_bins[0], num_bins[1], d_binstartpts, d_sortidx, this->points_[0],this->points_[1],
//           d_idxnupts, pirange, nf1, this->grid_dims_[1]);

//       }else{
//         int *d_idxnupts = d_plan->idxnupts;

//         TrivialGlobalSortIdx_2d<<<(M+1024-1)/1024, 1024>>>(M,d_idxnupts);
//       }
  // switch(d_plan->rank_)
  // {
  //   case 2:
  //   {
  //     if (d_plan->options_.spread_method == SpreadMethod::NUPTS_DRIVEN) {
  //       ier = CUSPREAD2D_NUPTSDRIVEN_PROP(
  //           this->grid_dims_[0], this->grid_dims_[1], this->num_points_, d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread2d_nupts_prop, method(%d)\n",
  //             d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return 1;
  //       }
  //     }
  //     if (d_plan->options_.spread_method == SpreadMethod::SUBPROBLEM) {
  //       ier = CUSPREAD2D_SUBPROB_PROP(this->grid_dims_[0],this->grid_dims_[1],this->num_points_,d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread2d_subprob_prop, method(%d)\n",
  //                d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return 1;
  //       }
  //     }
  //     if (d_plan->options_.spread_method == SpreadMethod::PAUL) {
  //       int ier = CUSPREAD2D_PAUL_PROP(this->grid_dims_[0],this->grid_dims_[1],this->num_points_,d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread2d_paul_prop, method(%d)\n",
  //           d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return 1;
  //       }
  //     }
  //   }
  //   break;
  //   case 3:
  //   {
  //     if (d_plan->options_.spread_method == SpreadMethod::BLOCK_GATHER) {
  //       int ier = CUSPREAD3D_BLOCKGATHER_PROP(this->grid_dims_[0],this->grid_dims_[1],this->grid_dims_[2],this->num_points_,d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread3d_blockgather_prop, method(%d)\n",
  //           d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return ier;
  //       }
  //     }
  //     if (d_plan->options_.spread_method == SpreadMethod::NUPTS_DRIVEN) {
  //       ier = CUSPREAD3D_NUPTSDRIVEN_PROP(this->grid_dims_[0],this->grid_dims_[1],this->grid_dims_[2],this->num_points_,d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n",
  //           d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return ier;
  //       }
  //     }
  //     if (d_plan->options_.spread_method == SpreadMethod::SUBPROBLEM) {
  //       int ier = CUSPREAD3D_SUBPROB_PROP(this->grid_dims_[0],this->grid_dims_[1],this->grid_dims_[2],this->num_points_,d_plan);
  //       if (ier != 0 ) {
  //         printf("error: cuspread3d_subprob_prop, method(%d)\n",
  //           d_plan->options_.spread_method);

  //                                       // Multi-GPU support: reset the device ID
  //                                       cudaSetDevice(orig_gpu_device_id);

  //         return ier;
  //       }
  //     }
  //   }
  //   break;
  // }

//   this->is_initialized_ = true;
//   return Status::OK();
// }

template class Spreader<GPUDevice, float>;
template class Spreader<GPUDevice, double>;

} // namespace nufft
} // namespace tensorflow

#endif // GOOGLE_CUDA
