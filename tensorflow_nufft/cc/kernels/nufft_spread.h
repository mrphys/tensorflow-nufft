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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_

namespace tensorflow {
namespace nufft {

// template<typename Device, typename FloatType>
// class SpreaderBase {

//  protected:

//   SpreaderParameters<FloatType> params_;
// };

// template<typename Device, typename FloatType>
// class Spreader;

// #if GOOGLE_CUDA
// template<typename FloatType>
// class Spreader<GPUDevice, FloatType> : public SpreaderBase<GPUDevice, FloatType> {

//   Status initialize(const SpreaderParameters<FloatType>& params);

//   Status spread();

//   Status interp();

// };
// #endif // GOOGLE_CUDA

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_SPREAD_H_
