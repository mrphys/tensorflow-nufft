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

#define EIGEN_USE_THREADS

#include "reverse_functor.h"

#include "tensorflow/core/framework/register_types.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template<typename Device, typename T>
Status DoReverse(const Device& device, const Tensor& input,
                 const gtl::ArraySlice<int32> axes, Tensor* output) {
  return internal::DoReverseImpl<Device, T>(device, input, axes, output);
}

#define INSTANTIATE_CPU(TYPE)                                               \
  template Status DoReverse<CPUDevice, TYPE>(                               \
      const CPUDevice& device, const Tensor& in,                            \
      const gtl::ArraySlice<int32> perm, Tensor* out);              

TF_CALL_float(INSTANTIATE_CPU);
TF_CALL_double(INSTANTIATE_CPU);
TF_CALL_complex64(INSTANTIATE_CPU);
TF_CALL_complex128(INSTANTIATE_CPU);

}  // namespace tensorflow
