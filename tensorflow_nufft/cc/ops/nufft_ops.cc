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

Status NUFFTShapeFn(InferenceContext* c) {

  // Input shapes.
  ShapeHandle source_shape = c->input(0);
  ShapeHandle points_shape = c->input(1);

  // Validate `transform_type` attribute.
  string transform_type_str;
  TF_RETURN_IF_ERROR(c->GetAttr("transform_type", &transform_type_str));
  int transform_type;
  if (transform_type_str == "type_1") {
    transform_type = 1;
  } else if (transform_type_str == "type_2") {
    transform_type = 2;
  }
  else {
    return errors::InvalidArgument(
      "transform_type attr must be 'type_1' or 'type_2', but is ",
      transform_type_str);
  }

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
  int64 rank = c->Value(rank_handle);

  // Validate `grid_shape` attribute.
  ShapeHandle grid_shape;
  if (transform_type == 1) {
    PartialTensorShape grid_tensor_shape;
    TF_RETURN_IF_ERROR(c->GetAttr("grid_shape", &grid_tensor_shape));
    TF_RETURN_IF_ERROR(
      c->MakeShapeFromPartialTensorShape(grid_tensor_shape, &grid_shape));
    TF_RETURN_IF_ERROR(c->WithRank(grid_shape, rank, &grid_shape));
    // if (!c->RankKnown(grid_shape)) {
    //   return errors::InvalidArgument("grid_shape attr must have known rank")
    // } 
    if (!c->FullyDefined(grid_shape)) {
      return errors::InvalidArgument("grid_shape attr must be fully defined");
    }
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
  int64 source_first_elem_axis;
  switch (transform_type) {
    case 1: // nonuniform to uniform
      source_first_elem_axis = -1;
      break;
    case 2: // uniform to nonuniform
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
    case 1: // nonuniform to uniform
      TF_RETURN_IF_ERROR(c->Concatenate(
          output_batch_shape, grid_shape, &output_shape));
      break;
    case 2: // uniform to nonuniform
      TF_RETURN_IF_ERROR(c->Concatenate(
          output_batch_shape, c->Vector(num_points), &output_shape));
      break;
  }
  c->set_output(0, output_shape);

  return Status::OK();
}


REGISTER_OP("NUFFT")
  .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
  .Attr("Treal: {float32, float64} = DT_FLOAT")
  .Input("source: Tcomplex")
  .Input("points: Treal")
  .Output("target: Tcomplex")
  .Attr("transform_type: {'type_1', 'type_2'} = 'type_2'")
  .Attr("j_sign: {'positive', 'negative'} = 'negative'")
  .Attr("epsilon: float = 1e-6")
  .Attr("grid_shape: shape = { unknown_rank: true }")
  .SetShapeFn(NUFFTShapeFn)
  .Doc(R"doc(
Compute the non-uniform fast Fourier transform.
)doc");

} // namespace tensorflow
