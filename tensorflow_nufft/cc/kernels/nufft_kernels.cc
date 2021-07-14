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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "nufft.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"

#include "third_party/finufft/include/finufft.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace finufft {

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
    finufftf_default_opts(opts);
};

template<>
void default_opts<CPUDevice, double>(
        int type, int dim, typename opts_type<CPUDevice, double>::type* opts) {
    finufft_default_opts(opts);
};

template<>
int makeplan<CPUDevice, float>(
        int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
        typename plan_type<CPUDevice, float>::type* plan,
        typename opts_type<CPUDevice, float>::type* opts) {
    return finufftf_makeplan(
        type, dim, nmodes, iflag, ntr, eps, plan, opts);
};

template<>
int makeplan<CPUDevice, double>(
        int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
        typename plan_type<CPUDevice, double>::type* plan,
        typename opts_type<CPUDevice, double>::type* opts) {
    return finufft_makeplan(
        type, dim, nmodes, iflag, ntr, eps, plan, opts);
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
int destroy<CPUDevice, float>(
        typename plan_type<CPUDevice, float>::type plan) {
    return finufftf_destroy(plan);
};

template<>
int destroy<CPUDevice, double>(
        typename plan_type<CPUDevice, double>::type plan) {
    return finufft_destroy(plan);
};

}   // namespace finufft

template<typename T>
struct DoNUFFT<CPUDevice, T> : DoNUFFTBase<CPUDevice, T> {
    Status operator()(OpKernelContext* ctx,
                      int type,
                      int rank,
                      int iflag,
                      int ntr,
                      T epsilon,
                      int64_t* nmodes,
                      int64_t npts,
                      T* points,
                      std::complex<T>* source,
                      std::complex<T>* target) {
        return this->compute(
            ctx, type, rank, iflag, ntr, epsilon,
            nmodes, npts, points, source, target);
    }
};

template <typename Device, typename T>
class NUFFT : public OpKernel {

  public:

    explicit NUFFT(OpKernelConstruction* ctx) : OpKernel(ctx) {

        string transform_type_str;
        string j_sign_str;

        OP_REQUIRES_OK(ctx,
                       ctx->GetAttr("transform_type", &transform_type_str));
        OP_REQUIRES_OK(ctx,
                       ctx->GetAttr("j_sign", &j_sign_str));
        OP_REQUIRES_OK(ctx,
                       ctx->GetAttr("epsilon", &epsilon_));
        OP_REQUIRES_OK(ctx,
                       ctx->GetAttr("grid_shape", &grid_shape_));

        if (transform_type_str == "type_1") {
            transform_type_ = 1;
        } else if (transform_type_str == "type_2") {
            transform_type_ = 2;
        }

        if (j_sign_str == "positive") {
            j_sign_ = 1;
        } else if (j_sign_str == "negative") {
            j_sign_ = -1;
        }

    }

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
        
        int64 nufft_rank = points.dim_size(points.dims() - 1);
        int64 num_points = points.dim_size(points.dims() - 2);

        switch (transform_type_) {
            case 1: // nonuniform to uniform
                
                // TODO: check `grid_shape` input.
                OP_REQUIRES(ctx, grid_shape_.dims() == nufft_rank,
                            errors::InvalidArgument(
                                "A valid `grid_shape` input must be provided "
                                "for NUFFT type-1, but received: ",
                                grid_shape_.DebugString()));

                break;

            case 2: // uniform to nonuniform

                // Get shape from `source` input.
                OP_REQUIRES(ctx, source.dims() >= nufft_rank,
                            errors::InvalidArgument(
                                "Input `source` must have rank of at least ",
                                nufft_rank, " but received shape: ",
                                source.shape().DebugString()));
                for (int i = source.dims() - nufft_rank; i < source.dims(); i++) {
                    grid_shape_.AddDim(source.dim_size(i));
                }

                break;
        }

        // Allocate output tensor.
        Tensor* target = nullptr;
        TensorShape target_shape;

        switch (transform_type_) {
            case 1: // nonuniform to uniform
                target_shape = grid_shape_;
                break;
            case 2: // uniform to nonuniform
                target_shape = TensorShape({num_points});
                break;
        }

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, target_shape, &target));

        // Transpose points to obtain single-dimension arrays.
        Tensor tpoints;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(
                           DataTypeToEnum<T>::value,
                           TensorShape({nufft_rank, num_points}),
                           &tpoints));
        std::vector<int32> perm = {1, 0};
        OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<Device>(
            ctx->eigen_device<Device>(),
            points,
            perm,
            &tpoints));

        // Perform operation.
        OP_REQUIRES_OK(ctx, DoNUFFT<Device, T>()(
            ctx,
            transform_type_,
            static_cast<int>(nufft_rank),
            j_sign_,
            1,
            static_cast<T>(epsilon_),
            (int64_t*) grid_shape_.dim_sizes().data(),
            num_points,
            (T*) tpoints.data(),
            (std::complex<T>*) source.data(),
            (std::complex<T>*) target->data()));
    }

  private:

    int transform_type_;
    int j_sign_;
    float epsilon_;
    TensorShape grid_shape_;

};

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Treal"),
                            NUFFT<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Treal"),
                            NUFFT<CPUDevice, double>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("Treal"),
                            NUFFT<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("Treal"),
                            NUFFT<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
