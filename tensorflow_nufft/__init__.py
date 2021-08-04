# Copyright 2021 University College London. All Rights Reserved.
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
"TensorFlow NUFFT."

import importlib
import types

from tensorflow_nufft.__about__ import *

_nufft_ops = importlib.import_module('tensorflow_nufft.python.ops.nufft_ops')
nufft = _nufft_ops.nufft
util = types.SimpleNamespace(estimate_density=_nufft_ops.estimate_density)
del _nufft_ops