/*Copyright 2021 University College London. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status NUFFTBaseShapeFn(InferenceContext* c, int transform_type) {
  // Input shapes.
  ShapeHandle source_shape = c->input(0);
  ShapeHandle points_shape = c->input(1);

  // Validate rank.
  DimensionHandle unused;
  DimensionHandle rank_handle = c->Dim(points_shape, -1);
  Status rank_is_1 = c->WithValue(rank_handle, 1, &unused);
  Status rank_is_2 = c->WithValue(rank_handle, 2, &unused);
  Status rank_is_3 = c->WithValue(rank_handle, 3, &unused);
  if (!(rank_is_1.ok() || rank_is_2.ok() || rank_is_3.ok())) {
    return errors::InvalidArgument(
      "Dimension must be 1, 2 or 3, but is ", c->DebugString(rank_handle));
  }
  if (!c->ValueKnown(rank_handle)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  int64_t rank = c->Value(rank_handle);

  // Get `grid_shape` input.
  ShapeHandle grid_shape;
  if (transform_type == 1) {
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &grid_shape));
    TF_RETURN_IF_ERROR(c->WithRank(grid_shape, rank, &grid_shape));
  }

  // Get number of nonuniform points.
  DimensionHandle num_points = c->Dim(points_shape, -2);

  // For type-1 transforms, verify number of input points in `source`.
  if (transform_type == 1) {
    TF_RETURN_IF_ERROR(c->Merge(num_points, c->Dim(source_shape, -1),
                                &num_points));
  }

  // The `source` input is potentially an N-D batch of elements. Each element
  // in the batch is 1D for type-1 transforms and N-D for type-2 transforms,
  // where N is the rank of the op.
  int64_t source_first_elem_axis;
  switch (transform_type) {
    case 1:  // nonuniform to uniform
      source_first_elem_axis = -1;
      break;
    case 2:  // uniform to nonuniform
      source_first_elem_axis = -rank;
      break;
  }

  // Extract batch shapes and compute output batch shape by broadcasting the
  // shapes of `source` and `points`.
  ShapeHandle source_batch_shape;
  ShapeHandle points_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(source_shape, 0, source_first_elem_axis,
                                 &source_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(points_shape, 0, -2,
                                 &points_batch_shape));
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, source_batch_shape, points_batch_shape, true, &output_batch_shape));

  ShapeHandle output_shape;
  switch (transform_type) {
    case 1:  // nonuniform to uniform
      TF_RETURN_IF_ERROR(c->Concatenate(
          output_batch_shape, grid_shape, &output_shape));
      break;
    case 2:  // uniform to nonuniform
      TF_RETURN_IF_ERROR(c->Concatenate(
          output_batch_shape, c->Vector(num_points), &output_shape));
      break;
  }
  c->set_output(0, output_shape);

  return Status::OK();
}


Status NUFFTShapeFn(InferenceContext* c) {
  // Validate `transform_type` attribute.
  string transform_type_str;
  TF_RETURN_IF_ERROR(c->GetAttr("transform_type", &transform_type_str));

  int transform_type;
  if (transform_type_str == "type_1") {
    transform_type = 1;
  } else if (transform_type_str == "type_2") {
    transform_type = 2;
  } else {
    return errors::InvalidArgument(
        "transform_type attr must be 'type_1' or 'type_2', but is ",
        transform_type_str);
  }

  return NUFFTBaseShapeFn(c, transform_type);
}


Status InterpShapeFn(InferenceContext* c) {
  return NUFFTBaseShapeFn(c, 2);
}


Status SpreadShapeFn(InferenceContext* c) {
  return NUFFTBaseShapeFn(c, 1);
}


REGISTER_OP("Interp")
  .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
  .Attr("Treal: {float32, float64} = DT_FLOAT")
  .Input("source: Tcomplex")
  .Input("points: Treal")
  .Output("target: Tcomplex")
  .Attr("tol: float = 1e-6")
  .SetShapeFn(InterpShapeFn)
  .Doc(R"doc(
Interpolate a regular grid at an arbitrary set of points.

This function can be used to perform the interpolation step of the NUFFT,
without the FFT or the deconvolution.

See also `tfft.nufft`, `tfft.spread`.

source: The source grid. Must have shape `[...] + grid_shape`, where
  `grid_shape` is the shape of the grid and `...` is any number of batch
  dimensions. `grid_shape` must have rank 1, 2 or 3.
points: The target non-uniform point coordinates. Must have shape `[..., M, N]`,
  where `M` is the number of non-uniform points, `N` is the rank of the grid and
  `...` is any number of batch dimensions, which must be broadcastable with the
  batch dimensions of `source`. `N` must be 1, 2 or 3 and must be equal to the
  rank of `grid_shape`. The non-uniform coordinates must be in units of
  radians/pixel, i.e., in the range `[-pi, pi]`.
tol: The desired relative precision. Should be in the range `[1e-06, 1e-01]`
  for `complex64` types and `[1e-14, 1e-01]` for `complex128` types. The
  computation may take longer for smaller values of `tol`.
target: The target point set. Has shape `[..., M]`, where the batch shape `...`
  is the result of broadcasting the batch shapes of `source` and `points`.
)doc");


REGISTER_OP("Spread")
  .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
  .Attr("Treal: {float32, float64} = DT_FLOAT")
  .Attr("Tshape: {int32, int64} = DT_INT32")
  .Input("source: Tcomplex")
  .Input("points: Treal")
  .Input("grid_shape: Tshape")
  .Output("target: Tcomplex")
  .Attr("tol: float = 1e-6")
  .SetShapeFn(SpreadShapeFn)
  .Doc(R"doc(
Spread an arbitrary set of points into a regular grid.

This function can be used to perform the spreading step of the NUFFT, without
the FFT or the deconvolution.

See also `tfft.nufft`, `tfft.interp`.

source: The source point set. Must have shape `[..., M]`, where `M` is the
  number of non-uniform points and `...` is any number of batch dimensions.
points: The source non-uniform point coordinates. Must have shape `[..., M, N]`,
  where `M` is the number of non-uniform points, `N` is the rank of the grid and
  `...` is any number of batch dimensions, which must be broadcastable with the
  batch dimensions of `source`. `N` must be 1, 2 or 3 and must be equal to the
  rank of `grid_shape`. The non-uniform coordinates must be in units of
  radians/pixel, i.e., in the range `[-pi, pi]`.
grid_shape: The shape of the output grid.
tol: The desired relative precision. Should be in the range `[1e-06, 1e-01]`
  for `complex64` types and `[1e-14, 1e-01]` for `complex128` types. The
  computation may take longer for smaller values of `tol`.
target: The target grid. Has shape `[...] + grid_shape`, where the batch shape
  `...` is the result of broadcasting the batch shapes of `source` and `points`.
)doc");


REGISTER_OP("NUFFT")
  .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
  .Attr("Treal: {float32, float64} = DT_FLOAT")
  .Attr("Tshape: {int32, int64} = DT_INT32")
  .Input("source: Tcomplex")
  .Input("points: Treal")
  .Input("grid_shape: Tshape")
  .Output("target: Tcomplex")
  .Attr("transform_type: {'type_1', 'type_2'} = 'type_2'")
  .Attr("fft_direction: {'forward', 'backward'} = 'forward'")
  .Attr("tol: float = 1e-6")
  .SetShapeFn(NUFFTShapeFn)
  .Doc(R"doc(
Compute the non-uniform discrete Fourier transform via NUFFT.

This op supports 1D, 2D and 3D type-1 and type-2 transforms.

.. note::
  1D transforms are only supported on the CPU.

.. [1] Barnett, A.H., Magland, J. and Klinteberg, L. af (2019), A parallel
  nonuniform fast Fourier transform library based on an “exponential of
  semicircle" kernel. SIAM J. Sci. Comput., 41(5): C479–C504.
  https://doi.org/10.1137/18M120885X
.. [2] Shih Y., Wright G., Anden J., Blaschke J. and Barnett A.H. (2021),
  cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
  2021 IEEE International Parallel and Distributed Processing Symposium
  Workshops (IPDPSW), 688–697 https://doi.org/10.1109/IPDPSW52791.2021.00105

source: The source grid, for type-2 transforms, or the source point set, for
  type-1 transforms. If `transform_type` is `"type_2"`, `source` must have shape
  `[...] + grid_shape`, where `grid_shape` is the shape of the grid and `...` is
  any number of batch dimensions. `grid_shape` must have rank 1, 2 or 3. If
  `transform_type` is `"type_1"`, `source` must have shape `[..., M]`, where `M`
  is the number of non-uniform points and `...` is any number of batch
  dimensions.
points: The target non-uniform point coordinates, for type-2 transforms, or the
  source non-uniform point coordinates, for type-1 transforms. Must have shape
  `[..., M, N]`, where `M` is the number of non-uniform points, `N` is the rank
  of the grid and `...` is any number of batch dimensions, which must be
  broadcastable with the batch dimensions of `source`. `N` must be 1, 2 or 3 and
  must be equal to the rank of `grid_shape`. The non-uniform coordinates must be
  in units of radians/pixel, i.e., in the range `[-pi, pi]`.
grid_shape: The shape of the output grid. This argument is required for type-1
  transforms and ignored for type-2 transforms.
transform_type: The type of the transform. A type-2 transform evaluates the DFT
  on a set of arbitrary points given points on a grid (uniform to non-uniform).
  A type-1 transform evaluates the DFT on grid points given a set of arbitrary
  points (non-uniform to uniform).
fft_direction: Defines the sign of the exponent in the formula of the Fourier
  transform. A `"forward"` transform has negative sign and a `"backward"`
  transform has positive sign.
tol: The desired relative precision. Should be in the range `[1e-06, 1e-01]`
  for `complex64` types and `[1e-14, 1e-01]` for `complex128` types. The
  computation may take longer for smaller values of `tol`.
target: The target point set, for type-2 transforms, or the target grid, for
  type-1 transforms. If `transform_type` is `"type_2"`, the output has shape
  `[..., M]`, where the batch shape `...` is the result of broadcasting the
  batch shapes of `source` and `points`. If `transform_type` is `"type_1"`, the
  output has shape `[...] + grid_shape`, where the batch shape `...` is the
  result of broadcasting the batch shapes of `source` and `points`.
)doc");

}  // namespace tensorflow
