/*==============================================================================
Copyright 2021 University College London. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


Status NUFFTShapeFn(shape_inference::InferenceContext* c) {
  
    // const PartialTensorShape output_shape({-1, 3});

    // shape_inference::ShapeHandle shape_handle;
    // TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(output_shape, &shape_handle));
    // c->set_output(0, shape_handle);

    // shape_inference::ShapeAndType shape_and_type(shape_handle, DT_FLOAT);
    // std::vector<shape_inference::ShapeAndType> shapes_and_types({shape_and_type});
    // c->set_output_handle_shapes_and_types(0, shapes_and_types);

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
