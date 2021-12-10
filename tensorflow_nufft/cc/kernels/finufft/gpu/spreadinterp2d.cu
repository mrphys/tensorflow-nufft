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
#include <thrust/extrema.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/contrib/utils_fp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/utils.h"

using namespace std;
using namespace cufinufft;

#define MAXBINSIZE 1024

namespace cufinufft {


}	// namespace cufinufft


/* ------------------------ 2d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */


// /* Kernels for Paul's Method */
// __global__
// void LocateFineGridPos_Paul(int M, int nf1, int nf2, int  bin_size_x, 
// 	int bin_size_y, int nbinx, int nbiny, int* bin_size, int ns, FLT *x, FLT *y, 
// 	int* sortidx, int* finegridsize, int pirange)
// {
// 	int binidx, binx, biny;
// 	int oldidx;
// 	int xidx, yidx, finegrididx;
// 	FLT x_rescaled,y_rescaled;
// 	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
// 		if (ns%2 == 0) {
// 			x_rescaled=RESCALE(x[i], nf1, pirange);
// 			y_rescaled=RESCALE(y[i], nf2, pirange);
// 			binx = floor(floor(x_rescaled)/bin_size_x);
// 			biny = floor(floor(y_rescaled)/bin_size_y);
// 			binidx = binx+biny*nbinx;
// 			xidx = floor(x_rescaled) - binx*bin_size_x;
// 			yidx = floor(y_rescaled) - biny*bin_size_y;
// 			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
// 		}else{
// 			x_rescaled=RESCALE(x[i], nf1, pirange);
// 			y_rescaled=RESCALE(y[i], nf2, pirange);
// 			xidx = ceil(x_rescaled - 0.5);
// 			yidx = ceil(y_rescaled - 0.5);
			
// 			//xidx = (xidx == nf1) ? (xidx-nf1) : xidx;
// 			//yidx = (yidx == nf2) ? (yidx-nf2) : yidx;

// 			binx = floor(xidx/(float) bin_size_x);
// 			biny = floor(yidx/(float) bin_size_y);
// 			binidx = binx+biny*nbinx;

// 			xidx = xidx - binx*bin_size_x;
// 			yidx = yidx - biny*bin_size_y;
// 			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
// 		}
// 		oldidx = atomicAdd(&finegridsize[finegrididx], 1);
// 		sortidx[i] = oldidx;
// 	}
// }

// __global__
// void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, 
// 		int bin_size_y, int nbinx,int nbiny,int ns, FLT *x, FLT *y, 
// 		int* finegridstartpts, int* sortidx, int* index, int pirange)
// {
// 	FLT x_rescaled, y_rescaled;
// 	int binx, biny, binidx, xidx, yidx, finegrididx;
// 	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
// 		if (ns%2 == 0) {
// 			x_rescaled=RESCALE(x[i], nf1, pirange);
// 			y_rescaled=RESCALE(y[i], nf2, pirange);
// 			binx = floor(floor(x_rescaled)/bin_size_x);
// 			biny = floor(floor(y_rescaled)/bin_size_y);
// 			binidx = binx+biny*nbinx;
// 			xidx = floor(x_rescaled) - binx*bin_size_x;
// 			yidx = floor(y_rescaled) - biny*bin_size_y;
// 			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
// 		}else{
// 			x_rescaled=RESCALE(x[i], nf1, pirange);
// 			y_rescaled=RESCALE(y[i], nf2, pirange);
// 			xidx = ceil(x_rescaled - 0.5);
// 			yidx = ceil(y_rescaled - 0.5);
			
// 			xidx = (xidx == nf1) ? xidx - nf1 : xidx;
// 			yidx = (yidx == nf2) ? yidx - nf2 : yidx;

// 			binx = floor(xidx/(float) bin_size_x);
// 			biny = floor(yidx/(float) bin_size_y);
// 			binidx = binx+biny*nbinx;

// 			xidx = xidx - binx*bin_size_x;
// 			yidx = yidx - biny*bin_size_y;
// 			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
// 		}
// 		index[finegridstartpts[finegrididx]+sortidx[i]] = i;
// 	}
// }


// __global__
// void Spread_2d_Subprob_Paul(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
// 	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, 
// 	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y, 
// 	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
// 	int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, int* fgstartpts,
// 	int* finegridsize, int pirange)
// {
// 	extern __shared__ CUCPX fwshared[];

// 	int xstart,ystart,xend,yend;
// 	int subpidx=blockIdx.x;
// 	int bidx=subprob_to_bin[subpidx];
// 	int binsubp_idx=subpidx-subprobstartpts[bidx];

// 	int ix,iy,outidx;

// 	int xoffset=(bidx % nbinx)*bin_size_x;
// 	int yoffset=(bidx / nbinx)*bin_size_y;

// 	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
// #if 0
// 	FLT ker1[MAX_NSPREAD*10];
//     FLT ker2[MAX_NSPREAD*10];
// #endif
// 	for (int i=threadIdx.x; i<N; i+=blockDim.x) {
// 		fwshared[i].x = 0.0;
// 		fwshared[i].y = 0.0;
// 	}
// 	__syncthreads();

// 	FLT x_rescaled, y_rescaled;
// 	for (int i=threadIdx.x; i<bin_size_x*bin_size_y; i+=blockDim.x) {
// 		int fineidx = bidx*bin_size_x*bin_size_y+i;
// 		int idxstart = fgstartpts[fineidx]+binsubp_idx*maxsubprobsize;
// 		int nupts = min(maxsubprobsize,finegridsize[fineidx]-binsubp_idx*
// 			maxsubprobsize);
// 		if (nupts > 0) {
// 			x_rescaled = x[idxnupts[idxstart]];
// 			y_rescaled = y[idxnupts[idxstart]];

// 			xstart = ceil(x_rescaled - ns/2.0)-xoffset;
// 			ystart = ceil(y_rescaled - ns/2.0)-yoffset;
// 			xend   = floor(x_rescaled + ns/2.0)-xoffset;
// 			yend   = floor(y_rescaled + ns/2.0)-yoffset;
// #if 0
// 			for (int m=0; m<nupts; m++) {
// 				int idx = idxstart+m;
// 				x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
// 				y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);

// 				eval_kernel_vec_Horner(ker1+m*MAX_NSPREAD,xstart+xoffset-
// 					x_rescaled,ns,sigma);
// 				eval_kernel_vec_Horner(ker2+m*MAX_NSPREAD,ystart+yoffset-
// 					y_rescaled,ns,sigma);
// 			}
// #endif
// 			for (int yy=ystart; yy<=yend; yy++) {
// 				FLT kervalue2[10];
// 				for (int m=0; m<nupts; m++) {
// 					int idx = idxstart+m;
// #if 1 
// 					y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
// 					FLT disy = abs(y_rescaled-(yy+yoffset));
// 					kervalue2[m] = evaluate_kernel(disy, es_c, es_beta, ns);
// #else
// 					kervalue2[m] = ker2[m*MAX_NSPREAD+yy-ystart];
// #endif
// 				}
// 				for (int xx=xstart; xx<=xend; xx++) {
// 					ix = xx+ceil(ns/2.0);
// 					iy = yy+ceil(ns/2.0);
// 					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
// 					CUCPX updatevalue;
// 					updatevalue.x = 0.0;
// 					updatevalue.y = 0.0;
// 					for (int m=0; m<nupts; m++) {
// 						int idx = idxstart+m;
// #if 1
// 						x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
// 						FLT disx = abs(x_rescaled-(xx+xoffset));
// 						FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);

// 						updatevalue.x += kervalue2[m]*kervalue1*
// 										 c[idxnupts[idx]].x;
// 						updatevalue.y += kervalue2[m]*kervalue1*
// 										 c[idxnupts[idx]].y;
// #else
// 						FLT kervalue1 = ker1[m*MAX_NSPREAD+xx-xstart];
// 						updatevalue.x += kervalue1*kervalue2[m]*
// 							c[idxnupts[idx]].x;
// 						updatevalue.y += kervalue1*kervalue2[m]*
// 							c[idxnupts[idx]].y;
// #endif
// 					}
// 					atomicAdd(&fwshared[outidx].x, updatevalue.x);
// 					atomicAdd(&fwshared[outidx].y, updatevalue.y);
// 				}
// 			}
// 		}
// 	}
// 	__syncthreads();

// 	/* write to global memory */
// 	for (int k=threadIdx.x; k<N; k+=blockDim.x) {
// 		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
// 		int j = k /( bin_size_x+2*ceil(ns/2.0) );
// 		ix = xoffset-ceil(ns/2.0)+i;
// 		iy = yoffset-ceil(ns/2.0)+j;
// 		if (ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))) {
// 			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
// 			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
// 			outidx = ix+iy*nf1;
// 			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
// 			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
// 			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
// 		}
// 	}
// }
/* --------------------- 2d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */



