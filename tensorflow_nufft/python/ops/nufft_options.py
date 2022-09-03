# Copyright 2022 The TensorFlow NUFFT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import typing
import pydantic

from tensorflow_nufft.proto import options_pb2


class Options(pydantic.BaseModel):
  """Represents options for `nufft`.

  Attrs:
    max_batch_size: An optional `int`. The maximum batch size to use during
      the vectorized NUFFT computation. If set, limits the internal
      vectorization batch size to this value. Smaller values may reduce memory
      usage, but may also reduce performance. If not set, the internal batch
      size is chosen automatically.
  """
  max_batch_size: typing.Optional[int] = None

  def _to_proto(self):
    pb = options_pb2.Options()
    if self.max_batch_size is not None:
      pb.max_batch_size = self.max_batch_size
    return pb

  def _from_proto(self, pb):
    if pb.max_batch_size is not None:
      self.max_batch_size = pb.max_batch_size

  class Config:
    validate_assignment = True
