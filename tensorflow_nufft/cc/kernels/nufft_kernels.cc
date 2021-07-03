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

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/transpose_functor.h"

#include "finufft/finufft.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace finufft {
    
    // Template the float/double interfaces of fiNUFFT.

    template<typename T>
    struct plan_type { };

    template<>
    struct plan_type<float> {
        typedef finufftf_plan type;
    };

    template<>
    struct plan_type<double> {
        typedef finufft_plan type;
    };

    template<typename T>
    void default_opts(nufft_opts* opts) { };

    template<>
    void default_opts<float>(nufft_opts* opts) {
        finufftf_default_opts(opts);
    };

    template<>
    void default_opts<double>(nufft_opts* opts) {
        finufft_default_opts(opts);
    };

    template<typename T>
    int makeplan(
            int type, int dim, int64_t* nmodes, int iflag, int ntr, T eps,
            typename plan_type<T>::type* plan, nufft_opts* opts) { 
        return -1;
    };

    template<>
    int makeplan<float>(
            int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
            typename plan_type<float>::type* plan, nufft_opts* opts) {
        return finufftf_makeplan(
            type, dim, nmodes, iflag, ntr, eps, plan, opts);
    };

    template<>
    int makeplan<double>(
            int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
            plan_type<double>::type* plan, nufft_opts* opts) {
        return finufft_makeplan(
            type, dim, nmodes, iflag, ntr, eps, plan, opts);
    };
    
    template<typename T>
    int setpts(
            typename plan_type<T>::type plan,
            int64_t M, T* x, T* y, T* z,
            int64_t N, T* s, T* t, T* u) {
        return -1;
    };

    template<>
    int setpts<float>(
            typename plan_type<float>::type plan,
            int64_t M, float* x, float* y, float* z,
            int64_t N, float* s, float* t, float* u) {
        return finufftf_setpts(plan, M, x, y, z, N, s, t, u);
    };

    template<>
    int setpts<double>(
            typename plan_type<double>::type plan,
            int64_t M, double* x, double* y, double* z,
            int64_t N, double* s, double* t, double* u) {
        return finufft_setpts(plan, M, x, y, z, N, s, t, u);
    };

    template<typename T>
    int execute(
            typename plan_type<T>::type plan,
            std::complex<T>* c, std::complex<T>* f) {
        return -1;
    };

    template<>
    int execute<float>(
            typename plan_type<float>::type plan,
            std::complex<float>* c, std::complex<float>* f) {
        return finufftf_execute(plan, c, f);
    };

    template<>
    int execute<double>(
            typename plan_type<double>::type plan,
            std::complex<double>* c, std::complex<double>* f) {
        return finufft_execute(plan, c, f);
    };

    template<typename T>
    int destroy(typename plan_type<T>::type plan) {
        return -1;
    };

    template<>
    int destroy<float>(typename plan_type<float>::type plan) {
        return finufftf_destroy(plan);
    };

    template<>
    int destroy<double>(typename plan_type<double>::type plan) {
        return finufft_destroy(plan);
    };
}


template<typename T>
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
        Tensor transposed_points;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(
                           real_dtype,
                           TensorShape({nufft_rank, num_points}),
                           &transposed_points));
        std::vector<int32> perm = {1, 0};
        OP_REQUIRES_OK(ctx, ::tensorflow::DoTranspose<CPUDevice>(
            ctx->eigen_device<CPUDevice>(),
            points,
            perm,
            &transposed_points));

        T* points_x = NULL;
        T* points_y = NULL;
        T* points_z = NULL;
        
        switch (nufft_rank) {
            case 1:
                points_x = (T*) transposed_points.SubSlice(0).data();
                break;
            case 2:
                points_x = (T*) transposed_points.SubSlice(0).data();
                points_y = (T*) transposed_points.SubSlice(1).data();
                break;
            case 3:
                points_x = (T*) transposed_points.SubSlice(0).data();
                points_y = (T*) transposed_points.SubSlice(1).data();
                points_z = (T*) transposed_points.SubSlice(2).data();
                break;
        }

        // Obtain pointers to non-uniform strengths and Fourier mode
        // coefficients.
        std::complex<T>* strengths = nullptr;
        std::complex<T>* coeffs = nullptr;
        switch (transform_type_) {
            case 1: // nonuniform to uniform
                strengths = (std::complex<T>*) source.data();
                coeffs = (std::complex<T>*) target->data();
                break;
            case 2: // uniform to nonuniform
                strengths = (std::complex<T>*) target->data();
                coeffs = (std::complex<T>*) source.data();
                break;
        }

        // NUFFT options.
        nufft_opts opts;
        finufft::default_opts<T>(&opts);

        // Make the NUFFT plan.
        int err;
        typename finufft::plan_type<T>::type plan;
        err = finufft::makeplan<T>(transform_type_,
                                   static_cast<int>(nufft_rank),
                                   (int64_t*) grid_shape_.dim_sizes().data(),
                                   j_sign_,
                                   1,
                                   epsilon_,
                                   &plan,
                                   &opts);

        OP_REQUIRES(ctx, err <= 1,
                    errors::Internal(
                        "Failed during `finufft::makeplan`: ", err));
        
        // Set the point coordinates.
        err = finufft::setpts<T>(plan,
                                 num_points,
                                 points_x,
                                 points_y,
                                 points_z,
                                 0,
                                 NULL,
                                 NULL,
                                 NULL);
            
        OP_REQUIRES(ctx, err <= 1,
                    errors::Internal(
                        "Failed during `finufft::setpts`: ", err));

        // Execute the NUFFT.
        err = finufft::execute<T>(plan, strengths, coeffs);

        OP_REQUIRES(ctx, err <= 1,
                    errors::Internal(
                        "Failed during `finufft::execute`: ", err));

        // Clean up the plan.
        err = finufft::destroy<T>(plan);

        OP_REQUIRES(ctx, err <= 1,
                    errors::Internal(
                        "Failed during `finufft::destroy`: ", err));
    }

  private:

    int transform_type_;
    int j_sign_;
    float epsilon_;
    TensorShape grid_shape_;

};

// namespace functor {

//     template <typename T>
//     struct Nufft<CPUDevice, T> {

//         void operator()(const CPUDevice& d, int size, const T* in, T* out) {

//             for (int i = 0; i < size; ++i) {
//             out[i] = 2 * in[i];
//             }
//     }
// };

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Treal"),
                            NUFFT<float>);

REGISTER_KERNEL_BUILDER(Name("NUFFT")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Treal"),
                            NUFFT<double>);
