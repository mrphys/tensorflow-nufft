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

#include "tensorflow_nufft/cc/kernels/finufft/nufft_util.h"

#include <cstdint>


namespace tensorflow {
namespace nufft {

template<typename IntType>
IntType next_smooth_int(IntType n, IntType b) {
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;   // even
  IntType nplus = n - 2;    // to cancel out the +=2 at start of loop
  IntType numdiv = 2;       // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0)) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

template int next_smooth_int<int>(int, int);
template int64_t next_smooth_int<int64_t>(int64_t, int64_t);

} // namespace nufft
} // namespace tensorflow
