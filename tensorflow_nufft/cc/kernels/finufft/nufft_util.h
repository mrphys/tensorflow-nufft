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

#ifndef TENSORFLOW_NUFFT_FINUFFT_NUFFT_UTIL_H
#define TENSORFLOW_NUFFT_FINUFFT_NUFFT_UTIL_H

#include "tensorflow_nufft/cc/kernels/finufft/nufft_options.h"

namespace tensorflow {
namespace nufft {

// Finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). If b is specified, the returned number must also be a
// multiple of b (b is a number whose prime factors no larger than 5).
template<typename IntType>
IntType next_smooth_int(IntType n, IntType b = 1);

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_FINUFFT_NUFFT_UTIL_H
