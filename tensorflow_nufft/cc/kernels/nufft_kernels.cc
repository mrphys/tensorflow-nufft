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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_nufft/cc/kernels/nufft.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/util/bcast.h"

#include "tensorflow_nufft/cc/kernels/finufft/cpu/finufft.h"

#include "transpose_functor.h"
#include "reverse_functor.h"

struct nufft_opts { };


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

template<>
struct plan_type<CPUDevice, float> {
  typedef finufftf_plan type;
};

template<>
struct plan_type<CPUDevice, double> {
  typedef finufft_plan type;
};

template<>
struct opts_type<CPUDevice, float> {
  typedef nufft_opts type;
};

template<>
struct opts_type<CPUDevice, double> {
  typedef nufft_opts type;
};

template<>
void default_opts<CPUDevice, float>(
    int type, int dim, typename opts_type<CPUDevice, float>::type* opts) {
  // finufftf_default_opts(opts);
};

template<>
void default_opts<CPUDevice, double>(
    int type, int dim, typename opts_type<CPUDevice, double>::type* opts) {
  // finufft_default_opts(opts);
};

template<>
int makeplan<CPUDevice, float>(
    int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
    typename plan_type<CPUDevice, float>::type* plan,
    typename opts_type<CPUDevice, float>::type* opts,
    const Options& options) {
  return finufftf_makeplan(
    type, dim, nmodes, iflag, ntr, eps, plan, options);
};

template<>
int makeplan<CPUDevice, double>(
    int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
    typename plan_type<CPUDevice, double>::type* plan,
    typename opts_type<CPUDevice, double>::type* opts,
    const Options& options) {
  return finufft_makeplan(
    type, dim, nmodes, iflag, ntr, eps, plan, options);
};

template<>
int setpts<CPUDevice, float>(
    typename plan_type<CPUDevice, float>::type plan,
    int64_t M, float* x, float* y, float* z,
    int64_t N, float* s, float* t, float* u) {
  return finufftf_setpts(plan, M, x, y, z, N, s, t, u);
};

template<>
int setpts<CPUDevice, double>(
    typename plan_type<CPUDevice, double>::type plan,
    int64_t M, double* x, double* y, double* z,
    int64_t N, double* s, double* t, double* u) {
  return finufft_setpts(plan, M, x, y, z, N, s, t, u);
};

template<>
int execute<CPUDevice, float>(
    typename plan_type<CPUDevice, float>::type plan,
    std::complex<float>* c, std::complex<float>* f) {
  return finufftf_execute(plan, c, f);
};

template<>
int execute<CPUDevice, double>(
    typename plan_type<CPUDevice, double>::type plan,
    std::complex<double>* c, std::complex<double>* f) {
  return finufft_execute(plan, c, f);
};

template<>
int interp<CPUDevice, float>(
    typename plan_type<CPUDevice, float>::type plan,
    std::complex<float>* c, std::complex<float>* f) {
  return finufftf_interp(plan, c, f);
};

template<>
int interp<CPUDevice, double>(
    typename plan_type<CPUDevice, double>::type plan,
    std::complex<double>* c, std::complex<double>* f) {
  return finufft_interp(plan, c, f);
};

template<>
int spread<CPUDevice, float>(
    typename plan_type<CPUDevice, float>::type plan,
    std::complex<float>* c, std::complex<float>* f) {
  return finufftf_spread(plan, c, f);
};

template<>
int spread<CPUDevice, double>(
    typename plan_type<CPUDevice, double>::type plan,
    std::complex<double>* c, std::complex<double>* f) {
  return finufft_spread(plan, c, f);
};

template<>
int destroy<CPUDevice, float>(
    typename plan_type<CPUDevice, float>::type plan) {
  return finufftf_destroy(plan);
};

template<>
int destroy<CPUDevice, double>(
    typename plan_type<CPUDevice, double>::type plan) {
  return finufft_destroy(plan);
};

}   // namespace nufft

template<typename T>
struct DoNUFFT<CPUDevice, T> : DoNUFFTBase<CPUDevice, T> {
  Status operator()(OpKernelContext* ctx,
                    int type,
                    int rank,
                    int iflag,
                    int ntrans,
                    T tol,
                    OpType optype,
                    int64_t nbdims,
                    int64_t* source_bdims,
                    int64_t* points_bdims,
                    int64_t* nmodes,
                    int64_t npts,
                    T* points,
                    std::complex<T>* source,
                    std::complex<T>* target) {
    return this->compute(
      ctx, type, rank, iflag, ntrans, tol, optype,
      nbdims, source_bdims, points_bdims,
      nmodes, npts, points, source, target);
  }
};


template <typename Device, typename T>
class NUFFTBaseOp : public OpKernel {

 public:

  explicit NUFFTBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {

    const DataType real_dtype = DataTypeToEnum<T>::value;
    const DataType complex_dtype = DataTypeToEnum<std::complex<T>>::value;

    const Tensor& source = ctx->input(0);
    const Tensor& points = ctx->input(1);

    OP_REQUIRES(ctx, source.dtype() == complex_dtype,
                errors::InvalidArgument(
                  "Input `source` must have type ",
                  DataTypeString(complex_dtype), " but got: ",
                  DataTypeString(source.dtype())));
    OP_REQUIRES(ctx, points.dtype() == real_dtype,
                errors::InvalidArgument(
                  "Input `points` must have type ",
                  DataTypeString(real_dtype), " but got: ",
                  DataTypeString(points.dtype())));
    OP_REQUIRES(ctx, points.dims() >= 2,
                errors::InvalidArgument(
                  "Input `points` must have rank of at least 2, but got "
                  "shape: ", points.shape().DebugString()));

    int64_t rank = points.dim_size(points.dims() - 1);
    int64_t num_points = points.dim_size(points.dims() - 2);

    TensorShape grid_shape;
    switch (transform_type_) {
      case 1: {   // nonuniform to uniform
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
          LOG(FATAL) << "shape must have type int32 or int64_t.";
        }

        break;
      }
      case 2: {   // uniform to nonuniform
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

    // Handle broadcasting.
    gtl::InlinedVector<int64_t, 4> source_batch_shape;
    gtl::InlinedVector<int64_t, 4> points_batch_shape;

    switch (transform_type_) {
      case 1: // nonuniform to uniform
        source_batch_shape.resize(source.dims() - 1);
        break;
      case 2: // uniform to nonuniform
        source_batch_shape.resize(source.dims() - rank);
        break;
    }
    points_batch_shape.resize(points.shape().dims() - 2);

    for (int i = 0; i < source_batch_shape.size(); i++) {
      source_batch_shape[i] = source.dim_size(i);
    }
    for (int i = 0; i < points_batch_shape.size(); i++) {
      points_batch_shape[i] = points.dim_size(i);
    }

    // Reshape input tensors if either has less batch dimensions than the
    // other, by adding leading ones as necessary so that both have the same
    // shape.
    Tensor spoints;
    if (points_batch_shape.size() < source_batch_shape.size()) {
      // Points has less batch dims than source. Insert ones at the beginning
      // until the two sizes match.
      int diff = source_batch_shape.size() - points_batch_shape.size();
      for (int i = 0; i < diff; i++) {
        points_batch_shape.insert(points_batch_shape.begin(), 1);
      }

      TensorShape points_shape(points_batch_shape);
      points_shape.AppendShape(points.shape());

      OP_REQUIRES(ctx, spoints.CopyFrom(points, points_shape),
                  errors::Internal(
                    "Failed to reshape scalar points tensor."));
    } else {

      OP_REQUIRES(ctx, spoints.CopyFrom(points, points.shape()),
                  errors::Internal(
                    "Failed to copy non-scalar points tensor."));
    }

    Tensor ssource;
    if (source_batch_shape.size() < points_batch_shape.size()) {
      // Source has less batch dims than points. Insert ones at the beginning
      // until the two sizes match.
      int diff = points_batch_shape.size() - source_batch_shape.size();
      for (int i = 0; i < diff; i++) {
        source_batch_shape.insert(source_batch_shape.begin(), 1);
      }

      TensorShape source_shape(source_batch_shape);
      source_shape.AppendShape(source.shape());

      OP_REQUIRES(ctx, ssource.CopyFrom(source, source_shape),
                  errors::Internal(
                    "Failed to reshape scalar source tensor."));
    } else {

      OP_REQUIRES(ctx, ssource.CopyFrom(source, source.shape()),
                  errors::Internal(
                    "Failed to copy non-scalar source tensor."));
    }

    BCast bcast(source_batch_shape, points_batch_shape);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                  "Incompatible shapes: ", source.shape().DebugString(),
                  " vs. ", points.shape().DebugString()));

    // Allocate output tensor.
    Tensor* target = nullptr;
    TensorShape target_shape(bcast.output_shape());
    switch (transform_type_) {
      case 1: // nonuniform to uniform
        target_shape.AppendShape(grid_shape);
        break;
      case 2: // uniform to nonuniform
        target_shape.AppendShape({num_points});
        break;
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, target_shape, &target));

    int64_t num_batch_dims = source_batch_shape.size();
    int64_t num_transforms = 1;

    gtl::InlinedVector<int32, 8> outer_dims;
    gtl::InlinedVector<int32, 8> inner_dims;
    outer_dims.reserve(num_batch_dims);
    inner_dims.reserve(num_batch_dims);

    for (int i = 0; i < num_batch_dims; i++) {
      int32 points_dim_size = points_batch_shape[i];
      if (points_batch_shape[i] == 1) {
        inner_dims.push_back(i);
        num_transforms *= source_batch_shape[i];
      } else {
        outer_dims.push_back(i);
      }
    }

    gtl::InlinedVector<int32, 8> source_perm(ssource.dims());
    gtl::InlinedVector<int32, 8> points_perm(spoints.dims());
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
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           spoints.shape(),
                                           &rpoints));

    OP_REQUIRES_OK(ctx, ::tensorflow::DoReverse<Device, T>(
      ctx->eigen_device<Device>(),
      spoints,
      {spoints.dims() - 1},
      &rpoints));

    /// Transpose points to obtain single-dimension arrays.
    Tensor tpoints;
    TensorShape tpoints_shape = spoints.shape();
    for (int i = 0; i < spoints.dims(); i++) {
      tpoints_shape.set_dim(i, spoints.dim_size(points_perm[i]));
    }
    
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
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
      TensorShape tsource_shape = ssource.shape();
      for (int i = 0; i < ssource.dims(); i++) {
        tsource_shape.set_dim(i, ssource.dim_size(source_perm[i]));
      }
      
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<std::complex<T>>::value,
        tsource_shape,
        &tsource));
      
      OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
        ctx->eigen_device<Device>(),
        ssource,
        source_perm,
        &tsource));
      
      psource = &tsource;
      
    } else {
      psource = &ssource;
    }

    Tensor ttarget;
    Tensor* ptarget;
    if (transpose_target) {
      TensorShape ttarget_shape = target->shape();
      for (int i = 0; i < target->dims(); i++) {
        ttarget_shape.set_dim(i, target->dim_size(target_perm[i]));
      }
      
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<std::complex<T>>::value,
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
    OP_REQUIRES_OK(ctx, functor_(
      ctx,
      transform_type_,
      static_cast<int>(rank),
      j_sign_,
      num_transforms,
      static_cast<T>(tol_),
      op_type_,
      outer_dims.size(),
      (int64_t*) psource->shape().dim_sizes().data(),
      (int64_t*) tpoints.shape().dim_sizes().data(),
      grid_shape_vec.begin(),
      num_points,
      (T*) tpoints.data(),
      (std::complex<T>*) psource->data(),
      (std::complex<T>*) ptarget->data()));

    if (transpose_target) {
      OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
        ctx->eigen_device<Device>(),
        ttarget,
        target_iperm,
        target));
    }
  }

 protected:

  int transform_type_;
  int j_sign_;
  float tol_;
  OpType op_type_;

  DoNUFFT<Device, T> functor_;
};


template <typename Device, typename T>
class NUFFT : public NUFFTBaseOp<Device, T> {

  public:

  explicit NUFFT(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, T>(ctx) {

    string transform_type_str;
    string fft_direction_str;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("transform_type", &transform_type_str));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fft_direction", &fft_direction_str));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    if (transform_type_str == "type_1") {
      this->transform_type_ = 1;
    } else if (transform_type_str == "type_2") {
      this->transform_type_ = 2;
    }

    if (fft_direction_str == "backward") {
      this->j_sign_ = 1;
    } else if (fft_direction_str == "forward") {
      this->j_sign_ = -1;
    }

    this->op_type_ = OpType::NUFFT;
  }
};


template <typename Device, typename T>
class Interp : public NUFFTBaseOp<Device, T> {

  public:

  explicit Interp(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, T>(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    this->transform_type_ = 2;
    this->j_sign_ = 1;

    this->op_type_ = OpType::INTERP;
  }
};


template <typename Device, typename T>
class Spread : public NUFFTBaseOp<Device, T> {

  public:

  explicit Spread(OpKernelConstruction* ctx) : NUFFTBaseOp<Device, T>(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &this->tol_));

    this->transform_type_ = 1;
    this->j_sign_ = 1;

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

}  // namespace tensorflow
