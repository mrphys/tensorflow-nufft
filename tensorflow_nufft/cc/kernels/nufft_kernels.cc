/* Copyright 2021 The TensorFlow NUFFT Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/util/bcast.h"

#include "tensorflow_nufft/cc/kernels/nufft_plan.h"
#include "tensorflow_nufft/cc/kernels/reverse_functor.h"
#include "tensorflow_nufft/cc/kernels/transpose_functor.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

enum class OpType { NUFFT, INTERP, SPREAD };

template<typename Device, typename FloatType>
using Complex = typename ComplexType<Device, FloatType>::Type;

template<typename FloatType>
const DataType kRealDType = DataTypeToEnum<FloatType>::value;

template<typename FloatType>
const DataType kComplexDType = DataTypeToEnum<std::complex<FloatType>>::value;


template<typename Device, typename FloatType>
class NUFFTBaseOp : public OpKernel {
 public:
  explicit NUFFTBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& source = ctx->input(0);
    const Tensor& points = ctx->input(1);

    OP_REQUIRES(ctx, source.dtype() == kComplexDType<FloatType>,
                errors::InvalidArgument(
                    "Input `source` must have type ",
                    DataTypeString(kComplexDType<FloatType>), " but got: ",
                    DataTypeString(source.dtype())));
    OP_REQUIRES(ctx, points.dtype() == kRealDType<FloatType>,
                errors::InvalidArgument(
                    "Input `points` must have type ",
                    DataTypeString(kRealDType<FloatType>), " but got: ",
                    DataTypeString(points.dtype())));
    OP_REQUIRES(ctx, points.dims() >= 2,
                errors::InvalidArgument(
                    "Input `points` must have rank of at least 2, but got "
                    "shape: ", points.shape().DebugString()));

    int64_t rank = points.dim_size(points.dims() - 1);
    int64_t num_points = points.dim_size(points.dims() - 2);

    TensorShape grid_shape;
    switch (transform_type_) {
      case TransformType::TYPE_1: {   // nonuniform to uniform
        // Get shape of grid from the `grid_shape` input.
        const Tensor& grid_shape_tensor = ctx->input(2);
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(grid_shape_tensor.shape()),
                    errors::InvalidArgument(
                        "grid_shape must be 1D, but got shape: ",
                        grid_shape_tensor.shape().DebugString()));

        if (grid_shape_tensor.dtype() == DT_INT32) {
          OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
              grid_shape_tensor.vec<int32>(), &grid_shape));

        } else if (grid_shape_tensor.dtype() == DT_INT64) {
          OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
              grid_shape_tensor.vec<int64_t>(), &grid_shape));

        } else {
          LOG(FATAL) << "shape must have type int32 or int64";
        }

        break;
      }
      case TransformType::TYPE_2: {   // uniform to nonuniform
        // Get shape of grid from `source` input.
        OP_REQUIRES(ctx, source.dims() >= rank,
              errors::InvalidArgument(
                "Input `source` must have rank of at least ",
                rank, " but received shape: ",
                source.shape().DebugString()));
        for (int i = source.dims() - rank; i < source.dims(); i++) {
          grid_shape.AddDim(source.dim_size(i));
        }

        break;
      }
    }

    // Get the ranks of the source/point elements.
    int points_elem_rank = 2;  // A points element is always 2D.
    int source_elem_rank;
    switch (transform_type_) {
      case TransformType::TYPE_1:  // nonuniform to uniform
        // Source element is 1D for type-1 transforms.
        source_elem_rank = 1;
        break;
      case TransformType::TYPE_2:  // uniform to nonuniform
        // Source element is N-D for N-D type-2 transforms.
        source_elem_rank = rank;
        break;
    }

    // Split source and points shapes into batch and element shapes.
    TensorShape source_elem_shape;   // Shape of each source element.
    TensorShape points_elem_shape;   // Shape of each points element.
    TensorShape source_batch_shape;  // Shape of source batch.
    TensorShape points_batch_shape;  // Shape of points batch.

    for (int i = 0; i < source.dims(); i++) {
      if (i < source.dims() - source_elem_rank) {
        source_batch_shape.AddDim(source.dim_size(i));
      } else {
        source_elem_shape.AddDim(source.dim_size(i));
      }
    }

    for (int i = 0; i < points.dims(); i++) {
      if (i < points.dims() - points_elem_rank) {
        points_batch_shape.AddDim(points.dim_size(i));
      } else {
        points_elem_shape.AddDim(points.dim_size(i));
      }
    }

    // Reshape input tensors if either has less batch dimensions than the
    // other, by adding leading ones as necessary so that both have the same
    // number of batch dimensions.
    if (source_batch_shape.dims() < points_batch_shape.dims()) {
      // Source has less batch dims than points. Insert ones at the beginning
      // until the two sizes match.
      int diff = points_batch_shape.dims() - source_batch_shape.dims();
      for (int i = 0; i < diff; i++) {
        source_batch_shape.InsertDim(0, 1);
      }
    }

    if (points_batch_shape.dims() < source_batch_shape.dims()) {
      // Points has less batch dims than source. Insert ones at the beginning
      // until the two sizes match.
      int diff = source_batch_shape.dims() - points_batch_shape.dims();
      for (int i = 0; i < diff; i++) {
        points_batch_shape.InsertDim(0, 1);
      }
    }

    Tensor reshaped_source;
    TensorShape source_shape(source_batch_shape);
    source_shape.AppendShape(source_elem_shape);
    OP_REQUIRES(ctx, reshaped_source.CopyFrom(source, source_shape),
                errors::Internal(
                    "Failed to reshape source tensor."));

    Tensor reshaped_points;
    TensorShape points_shape(points_batch_shape);
    points_shape.AppendShape(points_elem_shape);
    OP_REQUIRES(ctx, reshaped_points.CopyFrom(points, points_shape),
                errors::Internal(
                    "Failed to reshape points tensor."));

    // Check that broadcast of source and points batch shapes is valid.
    BCast bcast(BCast::FromShape(source_batch_shape),
                BCast::FromShape(points_batch_shape));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", source.shape().DebugString(),
                    " vs. ", points.shape().DebugString()));

    // Allocate output tensor.
    Tensor* target = nullptr;
    TensorShape target_shape(bcast.output_shape());
    switch (transform_type_) {
      case TransformType::TYPE_1: // nonuniform to uniform
        target_shape.AppendShape(grid_shape);
        break;
      case TransformType::TYPE_2: // uniform to nonuniform
        target_shape.AppendShape({num_points});
        break;
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, target_shape, &target));

    int64_t num_batch_dims = source_batch_shape.dims();
    int64_t num_transforms = 1;

    gtl::InlinedVector<int32, 8> outer_dims;
    gtl::InlinedVector<int32, 8> inner_dims;
    outer_dims.reserve(num_batch_dims);
    inner_dims.reserve(num_batch_dims);

    for (int i = 0; i < num_batch_dims; i++) {
      if (points_batch_shape.dim_size(i) == 1) {
        inner_dims.push_back(i);
        num_transforms *= source_batch_shape.dim_size(i);
      } else {
        outer_dims.push_back(i);
      }
    }

    gtl::InlinedVector<int32, 8> source_perm(reshaped_source.dims());
    gtl::InlinedVector<int32, 8> points_perm(reshaped_points.dims());
    gtl::InlinedVector<int32, 8> target_perm(target->dims());
    std::iota(source_perm.begin(), source_perm.end(), 0);
    std::iota(points_perm.begin(), points_perm.end(), 0);
    std::iota(target_perm.begin(), target_perm.end(), 0);
    std::swap(points_perm[points_perm.size() - 2],
              points_perm[points_perm.size() - 1]);

    for (int i = 0; i < outer_dims.size(); i++) {
      source_perm[i] = outer_dims[i];
      points_perm[i] = outer_dims[i];
      target_perm[i] = outer_dims[i];
    }
    for (int i = 0; i < inner_dims.size(); i++) {
      source_perm[outer_dims.size() + i] = inner_dims[i];
      points_perm[outer_dims.size() + i] = inner_dims[i];
      target_perm[outer_dims.size() + i] = inner_dims[i];
    }

    gtl::InlinedVector<int32, 8> target_iperm(target_perm.size());
    std::fill_n(target_iperm.begin(), target_perm.size(), -1);
    for (int i = 0; i < target_perm.size(); i++) {
      int d = target_perm[i];
      target_iperm[d] = i;
    }

    bool transpose_source = false;
    for (int i = 0; i < source_perm.size(); i++) {
      if (source_perm[i] != i) {
        transpose_source = true;
      }
    }
    bool transpose_target = transpose_source;

    // Reverse points.
    Tensor rpoints;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(kRealDType<FloatType>,
                                           reshaped_points.shape(),
                                           &rpoints));

    OP_REQUIRES_OK(ctx, ::tensorflow::DoReverse<Device, FloatType>(
        ctx->eigen_device<Device>(),
        reshaped_points,
        {reshaped_points.dims() - 1},
        &rpoints));

    /// Transpose points to obtain single-dimension arrays.
    Tensor tpoints;
    TensorShape tpoints_shape = reshaped_points.shape();
    for (int i = 0; i < reshaped_points.dims(); i++) {
      tpoints_shape.set_dim(i, reshaped_points.dim_size(points_perm[i]));
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(kRealDType<FloatType>,
                                           tpoints_shape,
                                           &tpoints));

    OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
        ctx->eigen_device<Device>(),
        rpoints,
        points_perm,
        &tpoints));

    Tensor tsource;
    const Tensor* psource;
    if (transpose_source) {
      TensorShape tsource_shape = reshaped_source.shape();
      for (int i = 0; i < reshaped_source.dims(); i++) {
        tsource_shape.set_dim(i, reshaped_source.dim_size(source_perm[i]));
      }

      OP_REQUIRES_OK(ctx, ctx->allocate_temp(kComplexDType<FloatType>,
                                             tsource_shape,
                                             &tsource));

      OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
          ctx->eigen_device<Device>(),
          reshaped_source,
          source_perm,
          &tsource));

      psource = &tsource;

    } else {
      psource = &reshaped_source;
    }

    Tensor ttarget;
    Tensor* ptarget;
    if (transpose_target) {
      TensorShape ttarget_shape = target->shape();
      for (int i = 0; i < target->dims(); i++) {
        ttarget_shape.set_dim(i, target->dim_size(target_perm[i]));
      }

      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(kComplexDType<FloatType>,
                                        ttarget_shape,
                                        &ttarget));

      ptarget = &ttarget;
    } else {
      ptarget = target;
    }

    // The shape of the grid needs to be reversed for FINUFFT.
    auto grid_shape_vec = grid_shape.dim_sizes();
    if (rank == 2)
      std::swap(grid_shape_vec[0], grid_shape_vec[1]);
    else if (rank == 3)
      std::swap(grid_shape_vec[0], grid_shape_vec[2]);

    // Perform operation.
    OP_REQUIRES_OK(ctx, this->Execute(
        ctx,
        transform_type_,
        static_cast<int>(rank),
        fft_direction_,
        num_transforms,
        static_cast<FloatType>(tol_),
        op_type_,
        outer_dims.size(),
        (int64_t*) psource->shape().dim_sizes().data(),
        (int64_t*) tpoints.shape().dim_sizes().data(),
        grid_shape_vec.begin(),
        num_points,
        (FloatType*) tpoints.data(),
        reinterpret_cast<Complex<Device, FloatType>*>(psource->data()),
        reinterpret_cast<Complex<Device, FloatType>*>(ptarget->data())));

    if (transpose_target) {
      OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
          ctx->eigen_device<Device>(),
          ttarget,
          target_iperm,
          target));
    }
  }

  Status Execute(OpKernelContext* ctx,
                 TransformType type,
                 int rank,
                 FftDirection fft_direction,
                 int num_transforms,
                 FloatType tol,
                 OpType op_type,
                 int64_t batch_rank,
                 int64_t* source_batch_dims,
                 int64_t* points_batch_dims,
                 int64_t* num_modes,
                 int64_t num_points,
                 FloatType* points,
                 Complex<Device, FloatType>* source,
                 Complex<Device, FloatType>* target) {
    // Number of coefficients.
    int num_coeffs = 1;
    for (int d = 0; d < rank; d++) {
      num_coeffs *= num_modes[d];
    }

    // Number of calls to FINUFFT execute.
    int num_calls = 1;
    for (int d = 0; d < batch_rank; d++) {
      num_calls *= points_batch_dims[d];
    }

    // Factors to transform linear indices to subindices and viceversa.
    gtl::InlinedVector<int, 8> source_batch_factors(batch_rank);
    for (int d = 0; d < batch_rank; d++) {
      source_batch_factors[d] = 1;
      for (int j = d + 1; j < batch_rank; j++) {
        source_batch_factors[d] *= source_batch_dims[j];
      }
    }

    gtl::InlinedVector<int, 8> points_batch_factors(batch_rank);
    for (int d = 0; d < batch_rank; d++) {
      points_batch_factors[d] = 1;
      for (int j = d + 1; j < batch_rank; j++) {
        points_batch_factors[d] *= points_batch_dims[j];
      }
    }

    // Obtain pointers to non-uniform data c and Fourier mode
    // coefficients f.
    Complex<Device, FloatType>* c = nullptr;
    Complex<Device, FloatType>* f = nullptr;
    int source_index;
    int target_index;
    int* c_index;
    int* f_index;
    switch (type) {
      case TransformType::TYPE_1:  // nonuniform to uniform
        c = source;
        f = target;
        c_index = &source_index;
        f_index = &target_index;
        break;
      case TransformType::TYPE_2:  // uniform to nonuniform
        c = target;
        f = source;
        c_index = &target_index;
        f_index = &source_index;
        break;
    }

    // NUFFT options.
    Options options;

    if (op_type != OpType::NUFFT) {
      options.spread_only = true;
      options.upsampling_factor = 2.0;
    }

    // Intra-op threading.
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    options.num_threads = worker_threads.num_threads;

    // Make inlined vector from pointer to number of modes. TODO: use inlined
    // vector for all of num_modes.
    int num_modes_int[3] = {1, 1, 1};
    for (int i = 0; i < rank; ++i) {
      num_modes_int[i] = static_cast<int>(num_modes[i]);
    }

    // Make the NUFFT plan.
    auto plan = std::make_unique<Plan<Device, FloatType>>(ctx);
    TF_RETURN_IF_ERROR(plan->initialize(
        type, rank, num_modes_int, fft_direction,
        num_transforms, tol, options));

    // Pointers to a certain batch.
    Complex<Device, FloatType>* c_batch = nullptr;
    Complex<Device, FloatType>* f_batch = nullptr;
    FloatType* points_batch = nullptr;

    FloatType* points_x = nullptr;
    FloatType* points_y = nullptr;
    FloatType* points_z = nullptr;

    gtl::InlinedVector<int, 8> source_batch_indices(batch_rank);

    for (int call_index = 0; call_index < num_calls; call_index++) {
      points_batch = points + call_index * num_points * rank;
      switch (rank) {
        case 1:
          points_x = points_batch;
          break;
        case 2:
          points_x = points_batch;
          points_y = points_batch + num_points;
          break;
        case 3:
          points_x = points_batch;
          points_y = points_batch + num_points;
          points_z = points_batch + num_points * 2;
          break;
      }

      // Set the point coordinates.
      TF_RETURN_IF_ERROR(plan->set_points(
          num_points, points_x, points_y, points_z));

      // Compute indices.
      source_index = 0;
      target_index = call_index;
      int temp_index = call_index;
      for (int d = 0; d < batch_rank; d++) {
        source_batch_indices[d] = temp_index / points_batch_factors[d];
        temp_index %= points_batch_factors[d];
        if (source_batch_dims[d] == 1) {
          source_batch_indices[d] = 0;
        }
        source_index += source_batch_indices[d] * source_batch_factors[d];
      }

      c_batch = c + *c_index * num_transforms * num_points;
      f_batch = f + *f_index * num_transforms * num_coeffs;

      // Execute the NUFFT.
      switch (op_type) {
        case OpType::NUFFT:
          TF_RETURN_IF_ERROR(plan->execute(c_batch, f_batch));
          break;
        case OpType::INTERP:
          TF_RETURN_IF_ERROR(plan->interp(c_batch, f_batch));
          break;
        case OpType::SPREAD:
          TF_RETURN_IF_ERROR(plan->spread(c_batch, f_batch));
          break;
      }
    }
    return Status::OK();
  }

 protected:

  TransformType transform_type_;
  FftDirection fft_direction_;
  float tol_;
  OpType op_type_;
};


template <typename Device, typename FloatType>
class NUFFT : public NUFFTBaseOp<Device, FloatType> {

  public:

  explicit NUFFT(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, FloatType>(ctx) {

    string transform_type_str;
    string fft_direction_str;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("transform_type", &transform_type_str));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fft_direction", &fft_direction_str));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    if (transform_type_str == "type_1") {
      this->transform_type_ = TransformType::TYPE_1;
    } else if (transform_type_str == "type_2") {
      this->transform_type_ = TransformType::TYPE_2;
    }

    if (fft_direction_str == "backward") {
      this->fft_direction_ = FftDirection::BACKWARD;
    } else if (fft_direction_str == "forward") {
      this->fft_direction_ = FftDirection::FORWARD;
    }

    this->op_type_ = OpType::NUFFT;
  }
};


template <typename Device, typename FloatType>
class Interp : public NUFFTBaseOp<Device, FloatType> {

  public:

  explicit Interp(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, FloatType>(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    this->transform_type_ = TransformType::TYPE_2;
    this->fft_direction_ = FftDirection::BACKWARD; // irrelevant

    this->op_type_ = OpType::INTERP;
  }
};


template <typename Device, typename FloatType>
class Spread : public NUFFTBaseOp<Device, FloatType> {

  public:

  explicit Spread(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, FloatType>(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    this->transform_type_ = TransformType::TYPE_1;
    this->fft_direction_ = FftDirection::BACKWARD; // irrelevant

    this->op_type_ = OpType::SPREAD;
  }
};


// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal")
                            .HostMemory("grid_shape"),
                        NUFFT<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal")
                            .HostMemory("grid_shape"),
                        NUFFT<CPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("Interp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal"),
                        Interp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Interp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal"),
                        Interp<CPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("Spread")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal")
                            .HostMemory("grid_shape"),
                        Spread<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Spread")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal")
                            .HostMemory("grid_shape"),
                        Spread<CPUDevice, double>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal")
                            .HostMemory("grid_shape"),
                        NUFFT<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal")
                            .HostMemory("grid_shape"),
                        NUFFT<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("Interp")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal"),
                        Interp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Interp")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal"),
                        Interp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(Name("Spread")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex64>("Tcomplex")
                            .TypeConstraint<float>("Treal")
                            .HostMemory("grid_shape"),
                        Spread<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Spread")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<complex128>("Tcomplex")
                            .TypeConstraint<double>("Treal")
                            .HostMemory("grid_shape"),
                        Spread<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace nufft
}  // namespace tensorflow
