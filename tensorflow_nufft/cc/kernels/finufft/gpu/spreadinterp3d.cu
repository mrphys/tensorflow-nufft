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
#include <math.h>
#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <cuda.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils_fp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/precision_independent.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/utils.h"

using namespace std;
using namespace cufinufft;

namespace cufinufft {

static __forceinline__ __device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta, int ns)
	/* ES ("exp sqrt") kernel evaluation at single real argument:
	   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
	   related to an asymptotic approximation to the Kaiser--Bessel, itself an
	   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
	   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
	return abs(x) < ns/2.0 ? exp(es_beta * (sqrt(1.0 - es_c*x*x))) : 0.0;
}

}	// namespace cufinufft

static __inline__ __device__
void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w,
	const double upsampling_factor)
	/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
	   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
	   This is the current evaluation method, since it's faster (except i7 w=16).
	   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
	FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
	// insert the auto-generated code which expects z, w args, writes to ker...
	if (upsampling_factor == 2.0) {     // floating point equality is fine here
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/ker_horner_allw_loop.c"
	}
}

// static __inline__ __device__
// void eval_kernel_vec(FLT *ker, const FLT x, const double w, const double es_c,
//                      const double es_beta)
// {
//     for (int i=0; i<w; i++) {
//         ker[i] = evaluate_kernel(abs(x+i), es_c, es_beta, w);
//     }
// }


/* ---------------------- 3d Spreading Kernels -------------------------------*/
/* Kernels for bin sort NUpts */

/* Kernels for NUptsdriven method */

/* Kernels for Block BlockGather Method */
__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x, int bin_size_y,
	int bin_size_z, int nobinx, int nobiny, int nobinz, int binsperobinx,
	int binsperobiny, int binsperobinz, int* bin_size, FLT *x, FLT *y, FLT *z,
	int* sortidx, int pirange, int nf1, int nf2, int nf3)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nobinx, int nobiny, int nobinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
	int nf2, int nf3)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}


__global__
void Spread_3d_BlockGather(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx % nobinx)*obin_size_x;
	int yoffset=(obidx / nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx / (nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	for (int i=threadIdx.x; i<N; i+=blockDim.x) {
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for (int i=threadIdx.x; i<nupts; i+=blockDim.x) {
		int idx = ptstart+i;
		int b = idxnupts[idx]/M;
		int box[3];
		for (int d=0;d<3;d++) {
			box[d] = b%3;
			if (box[d] == 1)
				box[d] = -1;
			if (box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = idxnupts[idx]%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xstart = xstart < 0 ? 0 : xstart;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		ystart = ystart < 0 ? 0 : ystart;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		zstart = zstart < 0 ? 0 : zstart;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		xend   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		yend   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;
		zend   = zend >= obin_size_z ? obin_size_z-1 : zend;

		for (int zz=zstart; zz<=zend; zz++) {
			FLT disz=abs(z_rescaled-(zz+zoffset));
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta, ns);
			for (int yy=ystart; yy<=yend; yy++) {
				FLT disy=abs(y_rescaled-(yy+yoffset));
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta, ns);
				for (int xx=xstart; xx<=xend; xx++) {
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT disx=abs(x_rescaled-(xx+xoffset));
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for (int n=threadIdx.x; n<N; n+=blockDim.x) {
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}

__global__
void Spread_3d_BlockGather_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int xstartnew,ystartnew,zstartnew,xendnew,yendnew,zendnew;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx%nobinx)*obin_size_x;
	int yoffset=(obidx/nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx/(nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	for (int i=threadIdx.x; i<N; i+=blockDim.x) {
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for (int i=threadIdx.x; i<nupts; i+=blockDim.x) {
		int nidx = idxnupts[ptstart+i];
		int b = nidx/M;
		int box[3];
		for (int d=0;d<3;d++) {
			box[d] = b%3;
			if (box[d] == 1)
				box[d] = -1;
			if (box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = nidx%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);

		xstartnew = xstart < 0 ? 0 : xstart;
		ystartnew = ystart < 0 ? 0 : ystart;
		zstartnew = zstart < 0 ? 0 : zstart;
		xendnew   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yendnew   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zendnew   = zend >= obin_size_z ? obin_size_z-1 : zend;

		for (int zz=zstartnew; zz<=zendnew; zz++) {
			FLT kervalue3 = ker3[zz-zstart];
			for (int yy=ystartnew; yy<=yendnew; yy++) {
				FLT kervalue2 = ker2[yy-ystart];
				for (int xx=xstartnew; xx<=xendnew; xx++) {
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT kervalue1 = ker1[xx-xstart];
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*kervalue3);
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for (int n=threadIdx.x; n<N; n+=blockDim.x) {
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}


