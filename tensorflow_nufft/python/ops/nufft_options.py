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
  r"""Represents the planning rigor for the FFTW library.

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


class PointsRange(enum.IntEnum):
  r"""Represents the supported range for the nonuniform points.

  Specifies the supported range for the nonuniform points. More restrictive
  options may result in faster execution.

  ```{note}
  The discrete Fourier transform (DFT) is periodic with respect to the points
  $k$, i.e., $f(x, k + 2\pi) = f(x, k)$. Therefore, the DFT is defined for
  $k \in (-\infty, +\infty)$ and can always be computed by shifting $k$
  by a multiple of $2\pi$ to the interval $[-\pi, +\pi]$. However, if
  you can promise that the input points lie within a narrower range, the
  algorithm might be able to perform some optimizations.
  ```

  The following options are available:

  - **STRICT**: the algorithm is only guaranteed to support values in the range
    $[-\pi, +\pi]$. This option offers the most opportunities for performance
    optimization.

  - **EXTENDED**: the algorithm is guaranteed to support values in the range
    $[-3 \pi, +3 \pi]$. This option offers a compromise between flexibility and
    performance. Even if your points are in $[-\pi, +\pi]$, this option might
    offer robustness against rounding error. This is the default option.

  - **INFINITE**: the algorithm is guaranteed to support values in the range
    $(-\infty, +\infty)$. This option offers the most flexibility, but cannot
    optimize performance.

  ```{attention}
  For options `STRICT` and `EXTENDED`, passing points outside the supported
  range is undefined behaviour.
  ```
  """
  STRICT = 0
  EXTENDED = 1
  INFINITE = 2

  def to_proto(self):  # pylint: disable=missing-function-docstring
    if self == PointsRange.STRICT:
      return nufft_options_pb2.PointsRange.STRICT
    if self == PointsRange.EXTENDED:
      return nufft_options_pb2.PointsRange.EXTENDED
    if self == PointsRange.INFINITE:
      return nufft_options_pb2.PointsRange.INFINITE
    raise ValueError(
        f"Invalid value of `PointsRange`. Supported values include "
        f"`STRICT`, `EXTENDED` and `INFINITE`. Got {self.name}."
    )

  @classmethod
  def from_proto(cls, pb):  # pylint: disable=missing-function-docstring
    if pb == nufft_options_pb2.PointsRange.STRICT:
      return cls.STRICT
    if pb == nufft_options_pb2.PointsRange.EXTENDED:
      return cls.EXTENDED
    if pb == nufft_options_pb2.PointsRange.INFINITE:
      return cls.INFINITE
    raise ValueError(
        f"Invalid value of `PointsRange` in protocol buffer. Supported "
        f"values include `STRICT`, `EXTENDED` and `INFINITE`. Got {pb.name}."
    )


class DebuggingOptions(pydantic.BaseModel):
  r"""Represents options for debugging.

  Example:
    >>> options = tfft.Options()
    >>> # Assert that input points `x` lie within the supported range.
    >>> options.debugging.check_points_range = True
    >>> tfft.nufft(x, k, options=options)

  Attributes:
    check_points_range: If `True`, `nufft` will assert that the nonuniform
      point coordinates lie within the supported range (as determined by
      `options.points_range`). This improves the safety of the operation,
      but may have a small impact on performance. Defaults to `False`.
  """
  check_points_range: bool = False

  def to_proto(self):  # pylint: disable=missing-function-docstring
    pb = nufft_options_pb2.DebuggingOptions()
    pb.check_points_range = self.check_points_range
    return pb

  @classmethod
  def from_proto(cls, pb):  # pylint: disable=missing-function-docstring
    obj = cls()
    obj.check_points_range = pb.check_points_range
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
    >>> tfft.nufft(x, k, options=options)

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
    points_range: An optional `tfft.PointsRange`. Specifies the supported
      bounds for the nonuniform points. See `tfft.PointsRange` for more
      information. Defaults to `tfft.PointsRange.EXTENDED`.
  """
  debugging: DebuggingOptions = DebuggingOptions()
  fftw: FftwOptions = FftwOptions()
  max_batch_size: typing.Optional[int] = None
  points_range: PointsRange = PointsRange.EXTENDED

  def to_proto(self):
    pb = nufft_options_pb2.Options()
    pb.debugging.CopyFrom(self.debugging.to_proto())
    pb.fftw.CopyFrom(self.fftw.to_proto())
    if self.max_batch_size is not None:
      pb.max_batch_size = self.max_batch_size
    pb.points_range = self.points_range.to_proto()
    return pb

  @classmethod
  def from_proto(cls, pb):
    obj = cls()
    obj.debugging = DebuggingOptions.from_proto(pb.debugging)
    obj.fftw = FftwOptions.from_proto(pb.fftw)
    if pb.max_batch_size is not None:
      obj.max_batch_size = pb.max_batch_size
    obj.points_range = PointsRange.from_proto(pb.points_range)
    return obj

  class Config:
    validate_assignment = True
