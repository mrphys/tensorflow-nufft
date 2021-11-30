/* Copyright 2017-2021 The Simons Foundation. All Rights Reserved.

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

#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>
#include <cufft.h>

#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/cuda_samples/helper_cuda.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h"
#include "cuspreadinterp.h"
#include "cudeconvolve.h"
#include "memtransfer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;

void SETUP_BINSIZE(int type, int dim, Options& options)
{
	switch(dim)
	{
		case 2:
		{
			options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 32:
				options.gpu_bin_size.x;
      options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 32:
				options.gpu_bin_size.y;
			options.gpu_bin_size.z = 1;
		}
		break;
		case 3:
		{
			switch(options.gpu_spread_method)
			{
				case GpuSpreadMethod::NUPTS_DRIVEN:
				case GpuSpreadMethod::SUBPROBLEM:
				{
					options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 16:
						options.gpu_bin_size.x;
					options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 16:
						options.gpu_bin_size.y;
					options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 2:
						options.gpu_bin_size.z;
				}
				break;
				case GpuSpreadMethod::BLOCK_GATHER:
				{
					options.gpu_obin_size.x = (options.gpu_obin_size.x == 0) ? 8:
						options.gpu_obin_size.x;
					options.gpu_obin_size.y = (options.gpu_obin_size.y == 0) ? 8:
						options.gpu_obin_size.y;
					options.gpu_obin_size.z = (options.gpu_obin_size.z == 0) ? 8:
            options.gpu_obin_size.z;
          options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 4:
            options.gpu_bin_size.x;
          options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 4:
            options.gpu_bin_size.y;
          options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 4:
            options.gpu_bin_size.z;
				}
				break;
			}
		}
		break;
	}
}


#ifdef __cplusplus
extern "C" {
#endif
int CUFINUFFT_MAKEPLAN(int type, int dim, int *nmodes, int iflag,
		       int ntransf, FLT tol, int maxbatchsize,
		       Plan<GPUDevice, FLT>* *d_plan_ptr,
			   const Options& options)
/*
	"plan" stage (in single or double precision).
        See ../docs/cppdoc.md for main user-facing documentation.
        Note that *d_plan_ptr in the args list was called simply *plan there.
        This is the remaining dev-facing doc:

This performs:
                (0) creating a new plan struct (d_plan), a pointer to which is passed
                    back by writing that pointer into *d_plan_ptr.
              	(1) set up the spread option, d_plan.spopts.
		(2) calculate the correction factor on cpu, copy the value from cpu to
		    gpu
		(3) allocate gpu arrays with size determined by number of fourier modes
		    and method related options that had been set in d_plan.opts
		(4) call cufftPlanMany and save the cufft plan inside cufinufft plan
        Variables and arrays inside the plan struct are set and allocated.

	Melody Shih 07/25/19. Use-facing moved to markdown, Barnett 2/16/21.
*/
{
	// TODO: check options.
	//  - If mode_order == FFT, raise unimplemented error.
	//  - If check_bounds == true, raise unimplemented error.

        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        // if (opts == NULL) {
        //     // options might not be supplied to this function => assume device
        //     // 0 by default
        //     cudaSetDevice(0);
        // } else {
        cudaSetDevice(options.gpu_device_id);
        // }

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int ier;

	/* allocate the plan structure, assign address to user pointer. */
	Plan<GPUDevice, FLT>* d_plan = new Plan<GPUDevice, FLT>;
	*d_plan_ptr = d_plan;
        // Zero out your struct, (sets all pointers to NULL)
	memset(d_plan, 0, sizeof(*d_plan));

	// Copy options.
	d_plan->options = options;

  // Select kernel evaluation method.
  if (d_plan->options.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
	  d_plan->options.kernel_evaluation_method = KernelEvaluationMethod::DIRECT;
	}

	// Select upsampling factor. Currently always defaults to 2.
	if (d_plan->options.upsampling_factor == 0.0) {
	  d_plan->options.upsampling_factor = 2.0;
	}

  // Select spreading method.
  if (d_plan->options.gpu_spread_method == GpuSpreadMethod::AUTO) {
    if (dim == 2 && type == 1)
      d_plan->options.gpu_spread_method = GpuSpreadMethod::SUBPROBLEM;
    else if (dim == 2 && type == 2)
      d_plan->options.gpu_spread_method = GpuSpreadMethod::NUPTS_DRIVEN;
    else if (dim == 3 && type == 1)
      d_plan->options.gpu_spread_method = GpuSpreadMethod::SUBPROBLEM;
    else if (dim == 3 && type == 2)
      d_plan->options.gpu_spread_method = GpuSpreadMethod::NUPTS_DRIVEN;
  }

	// this must be set before calling "setup_spreader_for_nufft"
	d_plan->spopts.spread_interp_only = d_plan->options.spread_interp_only;

	/* Setup Spreader */
	ier = setup_spreader_for_nufft(d_plan->spopts, tol,
								   d_plan->options, dim);
	if (ier>1)                           // proceed if success or warning
	  return ier;

	d_plan->dim = dim;
	d_plan->ms = nmodes[0];
	d_plan->mt = nmodes[1];
	d_plan->mu = nmodes[2];

	SETUP_BINSIZE(type, dim, d_plan->options);
	BIGINT nf1=1, nf2=1, nf3=1;
	ier = SET_NF_TYPE12(d_plan->ms, d_plan->spopts, d_plan->options, &nf1,
				  		d_plan->options.gpu_obin_size.x);
	if (ier > 0) return ier;
	if (dim > 1) {
		ier = SET_NF_TYPE12(d_plan->mt, d_plan->spopts, d_plan->options, &nf2,
                      d_plan->options.gpu_obin_size.y);
		if (ier > 0) return ier;
	}
	if (dim > 2) {
		ier = SET_NF_TYPE12(d_plan->mu, d_plan->spopts, d_plan->options, &nf3,
                      d_plan->options.gpu_obin_size.z);
		if (ier > 0) return ier;
	}
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	if (maxbatchsize==0)                    // implies: use a heuristic.
	   maxbatchsize = min(ntransf, 8);      // heuristic from test codes
	d_plan->maxbatchsize = maxbatchsize;
	d_plan->type = type;

	if (d_plan->type == 1)
		d_plan->spopts.spread_direction = SpreadDirection::SPREAD;
	if (d_plan->type == 2)
		d_plan->spopts.spread_direction = SpreadDirection::INTERP;
	// this may move to gpu
	cufinufft::CNTime timer; timer.start();
	FLT *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;
	if (!d_plan->options.spread_interp_only) { // no need to do this if spread/interp only
		
		fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
		onedim_fseries_kernel(nf1, fwkerhalf1, d_plan->spopts);
		if (dim > 1) {
			fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
			onedim_fseries_kernel(nf2, fwkerhalf2, d_plan->spopts);
		}
		if (dim > 2) {
			fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));
			onedim_fseries_kernel(nf3, fwkerhalf3, d_plan->spopts);
		}
	}
#ifdef TIME
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", d_plan->spopts.nspread,
		timer.elapsedsec());
#endif

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = ALLOCGPUMEM1D_PLAN(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_PLAN(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_PLAN(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory plan %.3g s\n", milliseconds/1000);
#endif
	cudaEventRecord(start);
	if (!d_plan->options.spread_interp_only)
	{
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
		if (dim > 1)
			checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
				sizeof(FLT),cudaMemcpyHostToDevice));
		if (dim > 2)
			checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
				sizeof(FLT),cudaMemcpyHostToDevice));
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2 HtoD\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	if (!d_plan->options.spread_interp_only) {
		cufftHandle fftplan;
		switch(d_plan->dim)
		{
			case 1:
			{
				cerr<<"Not implemented yet"<<endl;
			}
			break;
			case 2:
			{
				int n[] = {nf2, nf1};
				int inembed[] = {nf2, nf1};

				//cufftCreate(&fftplan);
				//cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
				cufftResult result = cufftPlanMany(
					&fftplan,dim,n,inembed,1,inembed[0]*inembed[1],
					inembed,1,inembed[0]*inembed[1],CUFFT_TYPE,maxbatchsize);
				if (result != CUFFT_SUCCESS) {
					fprintf(stderr,"[%s] cufftPlanMany failed with error code: %d\n",__func__,result);
					return ERR_CUFFT;
				}
			}
			break;
			case 3:
			{
				int dim = 3;
				int n[] = {nf3, nf2, nf1};
				int inembed[] = {nf3, nf2, nf1};
				int istride = 1;
				cufftResult result = cufftPlanMany(
					&fftplan,dim,n,inembed,istride,inembed[0]*inembed[1]*
					inembed[2],inembed,istride,inembed[0]*inembed[1]*inembed[2],
					CUFFT_TYPE,maxbatchsize);
				if (result != CUFFT_SUCCESS) {
					fprintf(stderr,"[%s] cufftPlanMany failed with error code: %d\n",__func__,result);
    				return ERR_CUFFT;
				}
			}
			break;
		}
		d_plan->fftplan = fftplan;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	if (!d_plan->options.spread_interp_only) {
		free(fwkerhalf1);
		if (dim > 1)
			free(fwkerhalf2);
		if (dim > 2)
			free(fwkerhalf3);
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_SETPTS(int M, FLT* d_kx, FLT* d_ky, FLT* d_kz, int N, FLT *d_s,
	FLT *d_t, FLT *d_u, Plan<GPUDevice, FLT>* d_plan)
/*
	"setNUpts" stage (in single or double precision).

	In this stage, we
		(1) set the number and locations of nonuniform points
		(2) allocate gpu arrays with size determined by number of nupts
		(3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
		    coordinates are stored)
		(4) determine the spread/interp properties that only relates to the
		    locations of nupts (see 2d/spread2d_wrapper.cu,
		    3d/spread3d_wrapper.cu for what have been done in
		    function spread<dim>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:
        
	Input:
	M                 number of nonuniform points
	d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
	                  FLT array) in [-pi, pi). set h_kz to "NULL" if dimension
	                  is less than 3. same for h_ky for dimension 1.
	N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

	Input/Output:
	d_plan            pointer to a Plan<GPUDevice, FLT>. Variables and arrays inside
	                  the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type FLT means either single or double, matching the
	precision of the library version called.

	Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options.gpu_device_id);


	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int dim = d_plan->dim;

	d_plan->M = M;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2, nf3)=(%d,%d,%d) nj=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, nf3, d_plan->M,
		d_plan->ntransf);
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = ALLOCGPUMEM1D_NUPTS(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_NUPTS(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_NUPTS(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory NUpts%.3g s\n", milliseconds/1000);
#endif

	d_plan->kx = d_kx;
	if (dim > 1)
		d_plan->ky = d_ky;
	if (dim > 2)
		d_plan->kz = d_kz;

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			cerr<<"Not implemented yet"<<endl;
		}
		break;
		case 2:
		{
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::NUPTS_DRIVEN) {
				ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_nupts_prop, method(%d)\n",
						  d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::SUBPROBLEM) {
				ier = CUSPREAD2D_SUBPROB_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_subprob_prop, method(%d)\n",
					       d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::PAUL) {
				int ier = CUSPREAD2D_PAUL_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_paul_prop, method(%d)\n",
						d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
		}
		break;
		case 3:
		{
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::BLOCK_GATHER) {
				int ier = CUSPREAD3D_BLOCKGATHER_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_blockgather_prop, method(%d)\n",
						d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::NUPTS_DRIVEN) {
				ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n",
						d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if (d_plan->options.gpu_spread_method == GpuSpreadMethod::SUBPROBLEM) {
				int ier = CUSPREAD3D_SUBPROB_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_subprob_prop, method(%d)\n",
						d_plan->options.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
		}
		break;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tSetup Subprob properties %.3g s\n",
		milliseconds/1000);
#endif

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

int CUFINUFFT_EXECUTE(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
/*
	"exec" stage (single and double precision versions).

	The actual transformation is done here. Type and dimension of the
	transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

	Input/Output:
	d_c   a size d_plan->M CPX array on gpu (input for Type 1; output for Type
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for
	      Type 2; output for Type 1)

	Notes:
        i) Here CPX is a defined type meaning either complex<float> or complex<double>
	    to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

	Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options.gpu_device_id);

	int ier;
	int type=d_plan->type;
	switch(d_plan->dim)
	{
		case 1:
		{
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
		}
		break;
		case 2:
		{
			if (type == 1)
				ier = CUFINUFFT2D1_EXEC(d_c,  d_fk, d_plan);
			if (type == 2)
				ier = CUFINUFFT2D2_EXEC(d_c,  d_fk, d_plan);
			if (type == 3) {
				cerr<<"Not Implemented yet"<<endl;
				ier = ERR_NOTIMPLEMENTED;
			}
		}
		break;
		case 3:
		{
			if (type == 1)
				ier = CUFINUFFT3D1_EXEC(d_c,  d_fk, d_plan);
			if (type == 2)
				ier = CUFINUFFT3D2_EXEC(d_c,  d_fk, d_plan);
			if (type == 3) {
				cerr<<"Not Implemented yet"<<endl;
				ier = ERR_NOTIMPLEMENTED;
			}
		}
		break;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_INTERP(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	// Mult-GPU support: set the CUDA Device ID:
	int orig_gpu_device_id;
	cudaGetDevice(& orig_gpu_device_id);
	cudaSetDevice(d_plan->options.gpu_device_id);

	int ier;

	switch(d_plan->dim)
	{
		case 1:
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
			break;
		case 2:
			ier = CUFINUFFT2D_INTERP(d_c,  d_fk, d_plan);
			break;
		case 3:
			ier = CUFINUFFT3D_INTERP(d_c,  d_fk, d_plan);
			break;
	}

	// Multi-GPU support: reset the device ID
	cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_SPREAD(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	// Mult-GPU support: set the CUDA Device ID:
	int orig_gpu_device_id;
	cudaGetDevice(& orig_gpu_device_id);
	cudaSetDevice(d_plan->options.gpu_device_id);

	int ier;

	switch(d_plan->dim)
	{
		case 1:
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
			break;
		case 2:
			ier = CUFINUFFT2D_SPREAD(d_c,  d_fk, d_plan);
			break;
		case 3:
			ier = CUFINUFFT3D_SPREAD(d_c,  d_fk, d_plan);
			break;
	}

	// Multi-GPU support: reset the device ID
	cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_DESTROY(Plan<GPUDevice, FLT>* d_plan)
/*
	"destroy" stage (single and double precision versions).

	In this stage, we
		(1) free all the memories that have been allocated on gpu
		(2) delete the cuFFT plan

        Also see ../docs/cppdoc.md for main user-facing documentation.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options.gpu_device_id);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Can't destroy a Null pointer.
	if (!d_plan) {
                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);
		return 1;
        }

	if (d_plan->fftplan)
		cufftDestroy(d_plan->fftplan);

	switch(d_plan->dim)
	{
		case 1:
		{
			FREEGPUMEMORY1D(d_plan);
		}
		break;
		case 2:
		{
			FREEGPUMEMORY2D(d_plan);
		}
		break;
		case 3:
		{
			FREEGPUMEMORY3D(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif

	/* free/destruct the plan */
	delete d_plan;
	/* set pointer to NULL now that we've hopefully free'd the memory. */
	d_plan = NULL;

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
	return 0;
}

#ifdef __cplusplus
}
#endif
