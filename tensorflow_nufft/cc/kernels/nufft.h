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

#ifndef NUFFT_H_
#define NUFFT_H_

#include <complex>
#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {

namespace finufft {

template<typename Device, typename T>
struct plan_type;

template<typename Device, typename T>
struct opts_type;

template<typename Device, typename T>
void default_opts(int type, int dim, typename opts_type<Device, T>::type* opts);

template<typename Device, typename T>
int makeplan(
        int type, int dim, int64_t* nmodes, int iflag, int ntr, T eps,
        typename plan_type<Device, T>::type* plan,
        typename opts_type<Device, T>::type* opts);

template<typename Device, typename T>
int setpts(
        typename plan_type<Device, T>::type plan,
        int64_t M, T* x, T* y, T* z,
        int64_t N, T* s, T* t, T* u);

template<typename Device, typename T>
int execute(
        typename plan_type<Device, T>::type plan,
        std::complex<T>* c, std::complex<T>* f);

template<typename Device, typename T>
int destroy(typename plan_type<Device, T>::type plan);

}   // namespace finufft

template<typename Device, typename T>
struct DoNUFFTBase {
    
    Status compute(OpKernelContext* ctx,
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
        
        T* points_x = NULL;
        T* points_y = NULL;
        T* points_z = NULL;
        
        switch (rank) {
            case 1:
                points_x = points;
                break;
            case 2:
                points_x = points;
                points_y = points + npts;
                break;
            case 3:
                points_x = points;
                points_y = points + npts;
                points_z = points + npts * 2;
                break;
        }

        // Obtain pointers to non-uniform strengths and Fourier mode
        // coefficients.
        std::complex<T>* strengths = nullptr;
        std::complex<T>* coeffs = nullptr;
        switch (type) {
            case 1: // nonuniform to uniform
                strengths = source;
                coeffs = target;
                break;
            case 2: // uniform to nonuniform
                strengths = target;
                coeffs = source;
                break;
        }

        // NUFFT options.
        typename finufft::opts_type<Device, T>::type opts;
        finufft::default_opts<Device, T>(type, rank, &opts);

        // Make the NUFFT plan.
        int err;
        typename finufft::plan_type<Device, T>::type plan;
        err = finufft::makeplan<Device, T>(
            type,
            rank,
            nmodes,
            iflag,
            ntr,
            epsilon,
            &plan,
            &opts);

        if (err > 1) {
            return errors::Internal("Failed during `finufft::makeplan`: ", err);
        }
        
        // Set the point coordinates.
        err = finufft::setpts<Device, T>(
            plan,
            npts, points_x, points_y, points_z,
            0, NULL, NULL, NULL);
            
        if (err > 1) {
            return errors::Internal("Failed during `finufft::setpts`: ", err);
        }

        // Execute the NUFFT.
        err = finufft::execute<Device, T>(plan, strengths, coeffs);

        if (err > 1) {
            return errors::Internal("Failed during `finufft::execute`: ", err);
        }

        // Clean up the plan.
        err = finufft::destroy<Device, T>(plan);

        if (err > 1) {
            return errors::Internal("Failed during `finufft::destroy`: ", err);
        }

        return Status::OK();
    }
};

template<typename Device, typename T>
struct DoNUFFT : DoNUFFTBase<Device, T> {

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
                      std::complex<T>* target);
};

}   // namespace tensorflow

#endif  // NUFFT_H_
