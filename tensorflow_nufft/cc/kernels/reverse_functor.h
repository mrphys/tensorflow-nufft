/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_NUFFT_KERNELS_REVERSE_FUNCTOR_H_
#define TENSORFLOW_NUFFT_KERNELS_REVERSE_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template<typename Device, typename T>
Status DoReverse(const Device& device, const Tensor& input,
                 const gtl::ArraySlice<int32> axes, Tensor* output);

namespace functor {

// Functor used by ReverseOp to do the computations.
template <typename Device, typename T, int Dims>
struct Reverse {
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  const Eigen::array<bool, Dims>& reverse_dims,
                  typename TTypes<T, Dims>::Tensor output) {
    output.device(d) = input.reverse(reverse_dims);
  }
};

template <typename Device, typename T>
struct Reverse<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::ConstTensor input,
                  const Eigen::array<bool, 0>& reverse_dims,
                  typename TTypes<T, 0>::Tensor output) {
    // Reversing a scalar is copying it.
    output.device(d) = input;
  }
};

}  // namespace functor

namespace internal {

template <typename Device, typename T, int NDIMS>
void HandleReverseCase(const Device& d,
                       const Tensor& input,
                       const gtl::ArraySlice<bool> axes,
                       Tensor* output) {

  typename Eigen::array<bool, NDIMS> axes_di;
  for (int i = 0; i < NDIMS; i++) {
    axes_di[i] = axes[i];
  }

  functor::Reverse<Device, T, NDIMS>()(d,
                                       input.tensor<T, NDIMS>(),
                                       axes_di,
                                       output->tensor<T, NDIMS>());
}

template <typename Device, typename T>
Status DoReverseImpl(const Device& device, const Tensor& input,
                     const gtl::ArraySlice<int32> axes, Tensor* output) {
  if (TensorShapeUtils::IsScalar(input.shape()) || input.NumElements() == 0) {
    *output = input;
  } else {
    const int input_dims = input.dims();
    gtl::InlinedVector<bool, 8> axes_dense(input_dims, false);
    
    for (int d = 0; d < axes.size(); d++) {
      axes_dense[axes[d]] = true;
    }

#define HANDLE_REVERSE(NDIMS)                                               \
  case NDIMS:                                                               \
    HandleReverseCase<Device, T, NDIMS>(device, input, axes_dense, output); \
    return Status::OK();

    switch (input_dims) {
      HANDLE_REVERSE(0);
      HANDLE_REVERSE(1);
      HANDLE_REVERSE(2);
      HANDLE_REVERSE(3);
      HANDLE_REVERSE(4);
      HANDLE_REVERSE(5);
      HANDLE_REVERSE(6);
      HANDLE_REVERSE(7);
      HANDLE_REVERSE(8);
    }
#undef HANDLE_REVERSE
  }

  return Status::OK();
}

}  // namespace internal

}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_KERNELS_REVERSE_FUNCTOR_H_
