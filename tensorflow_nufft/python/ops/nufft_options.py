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
"""Defines options for NUFFT operator."""

import enum
import typing

import pydantic

from tensorflow_nufft.proto import nufft_options_pb2


class FftwPlanningRigor(enum.IntEnum):
  """Represents the planning rigor for the FFTW library.

  Controls the rigor (and time) of the FFTW planning process. More rigorous
  planning takes longer the first time `nufft` is called, but may result in
  faster execution during subsequent calls of similar transforms.

  - **AUTO**: Selects the planning rigor automatically. Currently defaults to
    `MEASURE`.

  - **ESTIMATE**: specifies that, instead of actual measurements of different
    algorithms, a simple heuristic is used to pick a (probably sub-optimal)
    plan quickly.

  - **MEASURE**: tells FFTW to find an optimized plan by actually computing
    several FFTs and measuring their execution time. Depending on your machine,
    this can take some time (often a few seconds).

  - **PATIENT**: like `MEASURE`, but considers a wider range of algorithms and
    often produces a “more optimal” plan (especially for large transforms), but
    at the expense of several times longer planning time (especially for large
    transforms).

  - **EXHAUSTIVE**: like `PATIENT`, but considers an even wider range of
    algorithms, including many that we think are unlikely to be fast, to
    produce the most optimal plan but with a substantially increased planning
    time.
  """
  AUTO = 0
  ESTIMATE = 1
  MEASURE = 2
  PATIENT = 3
  EXHAUSTIVE = 4

  def to_proto(self):  # pylint: disable=missing-function-docstring
    if self == FftwPlanningRigor.AUTO:
      return nufft_options_pb2.FftwPlanningRigor.AUTO
    if self == FftwPlanningRigor.ESTIMATE:
      return nufft_options_pb2.FftwPlanningRigor.ESTIMATE
    if self == FftwPlanningRigor.MEASURE:
      return nufft_options_pb2.FftwPlanningRigor.MEASURE
    if self == FftwPlanningRigor.PATIENT:
      return nufft_options_pb2.FftwPlanningRigor.PATIENT
    if self == FftwPlanningRigor.EXHAUSTIVE:
      return nufft_options_pb2.FftwPlanningRigor.EXHAUSTIVE
    raise ValueError(
        f"Invalid value of `FftwPlanningRigor`. Supported values include "
        f"`AUTO`, `ESTIMATE`, `MEASURE`, `PATIENT` and `EXHAUSTIVE`. "
        f"Got {self.name}."
    )

  @classmethod
  def from_proto(cls, pb):  # pylint: disable=missing-function-docstring
    if pb == nufft_options_pb2.FftwPlanningRigor.AUTO:
      return cls.AUTO
    if pb == nufft_options_pb2.FftwPlanningRigor.ESTIMATE:
      return cls.ESTIMATE
    if pb == nufft_options_pb2.FftwPlanningRigor.MEASURE:
      return cls.MEASURE
    if pb == nufft_options_pb2.FftwPlanningRigor.PATIENT:
      return cls.PATIENT
    if pb == nufft_options_pb2.FftwPlanningRigor.EXHAUSTIVE:
      return cls.EXHAUSTIVE
    raise ValueError(
        f"Invalid value of `FftwPlanningRigor` in protocol buffer. Supported "
        f"values include `AUTO`, `ESTIMATE`, `MEASURE`, `PATIENT` and "
        f"`EXHAUSTIVE`. Got {pb.name}."
    )


class PointBounds(enum.IntEnum):
  """Represents the supported bounds for the nonuniform points.

  Specifies the supported bounds for the nonuniform points. More restrictive
  bounds may result in faster execution.

  ```{note}
  The discrete Fourier transform (DFT) is periodic with respect to the points
  $x$, i.e., $f(k, x + 2\pi) = f(k, x)$. Therefore, it can always be computed
  by shifting the points by a multiple of $2\pi$ to the interval $[-\pi, \pi)$.
  This option affects whether the algorithm used by `nufft` is guaranteed to
  support this.
  ```

  - **STRICT**: points must lie in the range $[-\pi, \pi)$. This is the fastest
    option.

  - **EXTENDED**: points must lie in the range $[-3 \pi, 3 \pi)$. This option
    offers a compromise between flexibility and performance. This is the
    default option.

  - **INFINITE**: accepts points in the range $(-\infty, +\infty)$. This option
    offers the most flexibility, but may have slightly reduced performance.

  ```{attention}
  For options `STRICT` and `EXTENDED`, passing points outside the supported
  bounds is undefined behaviour.
  ```
  """
  STRICT = 0
  EXTENDED = 1
  INFINITE = 2

  def to_proto(self):  # pylint: disable=missing-function-docstring
    if self == PointBounds.STRICT:
      return nufft_options_pb2.PointBounds.STRICT
    if self == PointBounds.EXTENDED:
      return nufft_options_pb2.PointBounds.EXTENDED
    if self == PointBounds.INFINITE:
      return nufft_options_pb2.PointBounds.INFINITE
    raise ValueError(
        f"Invalid value of `PointBounds`. Supported values include "
        f"`STRICT`, `EXTENDED` and `INFINITE`. Got {self.name}."
    )

  @classmethod
  def from_proto(cls, pb):  # pylint: disable=missing-function-docstring
    if pb == nufft_options_pb2.PointBounds.STRICT:
      return cls.STRICT
    if pb == nufft_options_pb2.PointBounds.EXTENDED:
      return cls.EXTENDED
    if pb == nufft_options_pb2.PointBounds.INFINITE:
      return cls.INFINITE
    raise ValueError(
        f"Invalid value of `PointBounds` in protocol buffer. Supported "
        f"values include `STRICT`, `EXTENDED` and `INFINITE`. Got {pb.name}."
    )


class DebuggingOptions(pydantic.BaseModel):
  """Represents options for debugging.

  Example:
    >>> options = tfft.Options()
    >>> # Assert that input points `x` lie within the supported bounds.
    >>> options.debugging.check_bounds = True
    >>> tfft.nufft(k, x, options=options)

  Attributes:
    check_bounds: If `True`, `nufft` will assert that the nonuniform point
      coordinates lie within the supported bounds (as determined by
      `options.point_bounds`). This improves the safety of the operation,
      but may negatively impact performance. Defaults to `False`.
  """
  check_bounds: bool = False

  def to_proto(self):  # pylint: disable=missing-function-docstring
    pb = nufft_options_pb2.DebuggingOptions()
    pb.check_bounds = self.check_bounds
    return pb

  @classmethod
  def from_proto(cls, pb):  # pylint: disable=missing-function-docstring
    obj = cls()
    obj.check_bounds = pb.check_bounds
    return obj


class FftwOptions(pydantic.BaseModel):
  """Represents options for the FFTW library.

  These are only relevant when using the CPU kernels of NUFFT.

  ```{tip}
  You can set the FFTW options of the `nufft` through the `fftw` property of
  `tfft.Options`.
  ```

  Example:
    >>> options = tfft.Options()
    >>> options.fftw.planning_rigor = tfft.FftwPlanningRigor.PATIENT
    >>> tfft.nufft(k, x, options=options)

  Attributes:
    planning_rigor: Controls the rigor (and time) of the planning process.
      See `tfft.FftwPlanningRigor` for more information.
  """
  planning_rigor: FftwPlanningRigor = FftwPlanningRigor.AUTO

  def to_proto(self):
    pb = nufft_options_pb2.FftwOptions()
    pb.planning_rigor = self.planning_rigor.to_proto()
    return pb

  @classmethod
  def from_proto(cls, pb):
    obj = cls()
    obj.planning_rigor = FftwPlanningRigor.from_proto(pb.planning_rigor)
    return obj


class Options(pydantic.BaseModel):
  """Represents options for the `nufft` operator.

  This object can be used to control the behavior of the `nufft` operator.
  These are advanced options which may be useful for performance tuning, but
  are not required for most use cases.

  Example:
    >>> options = tfft.Options()
    >>> options.max_batch_size = 4
    >>> tfft.nufft(x, k, options=options)

  Attributes:
    debugging: Options for debugging. See `tfft.DebuggingOptions` for more
      information.
    fftw: Options for the FFTW library. See `tfft.FftwOptions` for more
      information.
    max_batch_size: An optional `int`. The maximum batch size to use during
      the vectorized NUFFT computation. If set, limits the internal
      vectorization batch size to this value. Smaller values may reduce memory
      usage, but may also reduce performance. If not set, the internal batch
      size is chosen automatically.
    point_bounds: An optional `tfft.PointBounds`. Specifies the supported
      bounds for the nonuniform points. See `tfft.PointBounds` for more
      information. Defaults to `tfft.PointBounds.EXTENDED`.
  """
  debugging: DebuggingOptions = DebuggingOptions()
  fftw: FftwOptions = FftwOptions()
  max_batch_size: typing.Optional[int] = None
  point_bounds: PointBounds = PointBounds.EXTENDED

  def to_proto(self):
    pb = nufft_options_pb2.Options()
    pb.debugging.CopyFrom(self.debugging.to_proto())
    pb.fftw.CopyFrom(self.fftw.to_proto())
    if self.max_batch_size is not None:
      pb.max_batch_size = self.max_batch_size
    pb.point_bounds = self.point_bounds.to_proto()
    return pb

  @classmethod
  def from_proto(cls, pb):
    obj = cls()
    obj.debugging = DebuggingOptions.from_proto(pb.debugging)
    obj.fftw = FftwOptions.from_proto(pb.fftw)
    if pb.max_batch_size is not None:
      obj.max_batch_size = pb.max_batch_size
    obj.point_bounds = PointBounds.from_proto(pb.point_bounds)
    return obj

  class Config:
    validate_assignment = True
