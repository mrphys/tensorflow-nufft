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

#include "nufft.h"

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/cufinufft/include/cufinufft.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace finufft {

template<>
struct plan_type<GPUDevice, float> {
    typedef cufinufftf_plan type;
};

template<>
struct plan_type<GPUDevice, double> {
    typedef cufinufft_plan type;
};

template<>
struct opts_type<GPUDevice, float> {
    typedef cufinufft_opts type;
};

template<>
struct opts_type<GPUDevice, double> {
    typedef cufinufft_opts type;
};

template<>
void default_opts<GPUDevice, float>(
        int type, int dim, typename opts_type<GPUDevice, float>::type* opts) {
    cufinufftf_default_opts(type, dim, opts);
};

template<>
void default_opts<GPUDevice, double>(
        int type, int dim, typename opts_type<GPUDevice, double>::type* opts) {
    cufinufft_default_opts(type, dim, opts);
};

template<>
int makeplan<GPUDevice, float>(
        int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
        typename plan_type<GPUDevice, float>::type* plan,
        typename opts_type<GPUDevice, float>::type* opts) {

    int* nmodes_int = new int[dim];
    for (int d = 0; d < dim; d++)
        nmodes_int[d] = static_cast<int>(nmodes[d]);

    int err = cufinufftf_makeplan(
        type, dim, nmodes_int, iflag, ntr, eps, 0, plan, opts);
    
    delete[] nmodes_int;
    return err;
};

template<>
int makeplan<GPUDevice, double>(
        int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
        typename plan_type<GPUDevice, double>::type* plan,
        typename opts_type<GPUDevice, double>::type* opts) {

    int* nmodes_int = new int[dim];
    for (int d = 0; d < dim; d++)
        nmodes_int[d] = static_cast<int>(nmodes[d]);

    int err = cufinufft_makeplan(
        type, dim, nmodes_int, iflag, ntr, eps, 0, plan, opts);
    
    delete[] nmodes_int;
    return err;
};

template<>
int setpts<GPUDevice, float>(
        typename plan_type<GPUDevice, float>::type plan,
        int64_t M, float* x, float* y, float* z,
        int64_t N, float* s, float* t, float* u) {
    return cufinufftf_setpts(M, x, y, z, N, s, t, u, plan);
};

template<>
int setpts<GPUDevice, double>(
        typename plan_type<GPUDevice, double>::type plan,
        int64_t M, double* x, double* y, double* z,
        int64_t N, double* s, double* t, double* u) {
    return cufinufft_setpts(M, x, y, z, N, s, t, u, plan);
};

template<>
int execute<GPUDevice, float>(
        typename plan_type<GPUDevice, float>::type plan,
        std::complex<float>* c, std::complex<float>* f) {
    return cufinufftf_execute(
        reinterpret_cast<cuFloatComplex*>(c),
        reinterpret_cast<cuFloatComplex*>(f),
        plan);
};

template<>
int execute<GPUDevice, double>(
        typename plan_type<GPUDevice, double>::type plan,
        std::complex<double>* c, std::complex<double>* f) {
    return cufinufft_execute(
        reinterpret_cast<cuDoubleComplex*>(c),
        reinterpret_cast<cuDoubleComplex*>(f),
        plan);
};

template<>
int destroy<GPUDevice, float>(
        typename plan_type<GPUDevice, float>::type plan) {
    return cufinufftf_destroy(plan);
};

template<>
int destroy<GPUDevice, double>(
        typename plan_type<GPUDevice, double>::type plan) {
    return cufinufft_destroy(plan);
};

}   // namespace finufft

template<typename T>
struct DoNUFFT<GPUDevice, T> : DoNUFFTBase<GPUDevice, T> {
    Status operator()(OpKernelContext* ctx,
                      int type,
                      int rank,
                      int iflag,
                      int ntrans,
                      T epsilon,
                      int64_t nbdims,
                      int64_t* source_bdims,
                      int64_t* points_bdims,
                      int64_t* nmodes,
                      int64_t npts,
                      T* points,
                      std::complex<T>* source,
                      std::complex<T>* target) {
        return this->compute(
            ctx, type, rank, iflag, ntrans, epsilon,
            nbdims, source_bdims, points_bdims,
            nmodes, npts, points, source, target);
    }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct DoNUFFT<GPUDevice, float>;
template struct DoNUFFT<GPUDevice, double>;

}   // namespace tensorflow

#endif  // GOOGLE_CUDA
