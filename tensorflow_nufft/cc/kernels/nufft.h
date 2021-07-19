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

#ifndef NUFFT_H_
#define NUFFT_H_

#include <complex>
#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"
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

        // Number of coefficients.
        int ncoeffs = 1;
        for (int d = 0; d < rank; d++) {
            ncoeffs *= nmodes[d];
        }

        // Number of calls to FINUFFT execute.
        int ncalls = 1;
        for (int d = 0; d < nbdims; d++) {
            ncalls *= points_bdims[d];
        }

        std::cout << "ncoeffs: " << ncoeffs << std::endl;
        std::cout << "nbdims: " << nbdims << std::endl;
        std::cout << "ncalls: " << ncalls << std::endl;
        std::cout << "ntrans: " << ntrans << std::endl;
        std::cout << "npts: " << npts << std::endl;

        // Factors to transform linear indices to subindices and viceversa.
        gtl::InlinedVector<int, 8> source_bfactors(nbdims);
        for (int d = 0; d < nbdims; d++) {
            source_bfactors[d] = 1;
            for (int j = d + 1; j < nbdims; j++) {
                source_bfactors[d] *= source_bdims[j];
            }
            std::cout << "source_bfactors[d]: " << source_bfactors[d] << std::endl;
        }

        gtl::InlinedVector<int, 8> points_bfactors(nbdims);
        for (int d = 0; d < nbdims; d++) {
            points_bfactors[d] = 1;
            for (int j = d + 1; j < nbdims; j++) {
                points_bfactors[d] *= points_bdims[j];
            }
            std::cout << "points_bfactors[d]: " << points_bfactors[d] << std::endl;
        }
        
        // Obtain pointers to non-uniform strengths and Fourier mode
        // coefficients.
        std::complex<T>* strengths = nullptr;
        std::complex<T>* coeffs = nullptr;
        int csrc;
        int ctgt;
        int* pcs;
        int* pcc;
        switch (type) {
            case 1: // nonuniform to uniform
                strengths = source;
                coeffs = target;
                pcs = &csrc;
                pcc = &ctgt;
                break;
            case 2: // uniform to nonuniform
                strengths = target;
                coeffs = source;
                pcs = &ctgt;
                pcc = &csrc;
                break;
        }

        std::cout << "opts" << std::endl;

        // NUFFT options.
        typename finufft::opts_type<Device, T>::type opts;
        finufft::default_opts<Device, T>(type, rank, &opts);

        std::cout << "makeplan" << std::endl;        
        // Make the NUFFT plan.
        int err;
        typename finufft::plan_type<Device, T>::type plan;
        err = finufft::makeplan<Device, T>(
            type,
            rank,
            nmodes,
            iflag,
            ntrans,
            epsilon,
            &plan,
            &opts);

        if (err > 1) {
            return errors::Internal(
                "Failed during `finufft::makeplan`: ", err);
        }

        // Pointers to a certain batch.
        std::complex<T>* bstrengths = NULL;
        std::complex<T>* bcoeffs = NULL;
        T* bpoints = NULL;

        T* points_x = NULL;
        T* points_y = NULL;
        T* points_z = NULL;

        gtl::InlinedVector<int, 8> source_binds(nbdims);

        for (int c = 0; c < ncalls; c++) {
            
            bpoints = points + c * npts * rank;

            switch (rank) {
                case 1:
                    points_x = bpoints;
                    break;
                case 2:
                    points_x = bpoints;
                    points_y = bpoints + npts;
                    break;
                case 3:
                    points_x = bpoints;
                    points_y = bpoints + npts;
                    points_z = bpoints + npts * 2;
                    break;
            }
            
            std::cout << "setpts" << std::endl;    

            // Set the point coordinates.
            err = finufft::setpts<Device, T>(
                plan,
                npts, points_x, points_y, points_z,
                0, NULL, NULL, NULL);
                
            if (err > 1) {
                return errors::Internal(
                    "Failed during `finufft::setpts`: ", err);
            }

            // Compute indices.
            csrc = 0;
            ctgt = c;
            int ctmp = c;
            for (int d = 0; d < nbdims; d++) {
                source_binds[d] = ctmp / points_bfactors[d];
                ctmp %= points_bfactors[d];
                if (source_bdims[d] == 1) {
                    source_binds[d] = 0;
                }
                csrc += source_binds[d] * source_bfactors[d];
            }

            bstrengths = strengths + *pcs * ntrans * npts;
            bcoeffs = coeffs + *pcc * ntrans * ncoeffs;
            
            std::cout << "c = " << c << std::endl;
            std::cout << "*pcs = " << *pcs << std::endl;
            std::cout << "*pcc = " << *pcc << std::endl;   

            std::cout << "execute" << std::endl;

            // Execute the NUFFT.
            err = finufft::execute<Device, T>(plan, bstrengths, bcoeffs);

            if (err > 1) {
                return errors::Internal(
                    "Failed during `finufft::execute`: ", err);
            }
        }

        // Clean up the plan.
        std::cout << "destroy" << std::endl;

        err = finufft::destroy<Device, T>(plan);

        if (err > 1) {
            return errors::Internal(
                "Failed during `finufft::destroy`: ", err);
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
                      int ntrans,
                      T epsilon,
                      int64_t nbdims,
                      int64_t* source_bdims,
                      int64_t* points_bdims,
                      int64_t* nmodes,
                      int64_t npts,
                      T* points,
                      std::complex<T>* source,
                      std::complex<T>* target);
};

}   // namespace tensorflow

#endif  // NUFFT_H_
