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

template<typename FloatType>
Status Spreader<GPUDevice, FloatType>::initialize(
    int rank, int* grid_dims, int num_points,
    FloatType* points_x, FloatType* points_y, FloatType* points_z,
    const SpreadParameters<FloatType> params) {

  if (rank < 2 || rank > 3) {
    return errors::InvalidArgument("rank must be 2 or 3, got ", rank);
  }

  this->rank_ = rank;
  for (int i = 0; i < 3; i++) {
    this->grid_dims_[i] = i < rank ? grid_dims[i] : 1;
  }
  this->num_points_ = num_points;
  this->points_[0] = points_x;
  this->points_[1] = points_y;
  this->points_[2] = points_z;
  this->params_ = params;

  // Allocate some arrays.
  if (this->indices_points_) {
    this->device_.deallocate(this->indices_points_);
  }
  if (this->sort_indices_) {
    this->device_.deallocate(this->sort_indices_);
  }
  
  size_t num_bytes = sizeof(int) * this->num_points_;
  switch (this->params_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      this->indices_points_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      if (this->params_.sort_points == SortPoints::YES) {
        this->sort_indices_ = reinterpret_cast<int*>(
            this->device_.allocate(num_bytes));
      } else {
        this->sort_indices_ = nullptr;
      }
      break;
    case SpreadMethod::SUBPROBLEM:
    case SpreadMethod::PAUL:
      this->indices_points_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      this->sort_indices_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      break;
    case SpreadMethod::BLOCK_GATHER:
      this->indices_points_ = nullptr;
      this->sort_indices_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      break;
  }

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

  this->is_initialized_ = true;
  return Status::OK();
}

template class Spreader<GPUDevice, float>;
template class Spreader<GPUDevice, double>;

} // namespace nufft
} // namespace tensorflow

#endif // GOOGLE_CUDA
