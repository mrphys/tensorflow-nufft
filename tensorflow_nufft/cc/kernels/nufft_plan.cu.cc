/* Copyright 2021 University College London. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE - 2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Copyright 2017 - 2021 The Simons Foundation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE - 2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "tensorflow_nufft/cc/kernels/nufft_util.h"
#include "tensorflow_nufft/cc/kernels/omp_api.h"

// NU coord handling macro: if p is true, rescales from [-pi, pi] to [0, N],
// then folds *only* one period below and above, ie [-N, 2N], into the domain
// [0, N]...
#define RESCALE(x, N, p) (p ? \
         ((x * kOneOverTwoPi<FloatType> + (x < -kPi<FloatType> ? 1.5 : \
         (x >= kPi<FloatType> ? -0.5 : 0.5))) * N) : \
         (x < 0 ? x + N : (x >= N ? x - N : x)))

namespace tensorflow {
namespace nufft {

namespace {

template<typename FloatType>
using GpuComplex = typename ComplexType<GPUDevice, FloatType>::Type;

template<typename FloatType>
constexpr cufftType kCufftType = CUFFT_C2C;
template<>
constexpr cufftType kCufftType<float> = CUFFT_C2C;
template<>
constexpr cufftType kCufftType<double> = CUFFT_Z2Z;

template<typename FloatType>
cufftResult cufftExec(
    cufftHandle plan, GpuComplex<FloatType> *idata,
    GpuComplex<FloatType> *odata, int direction);

template<>
cufftResult cufftExec<float>(
    cufftHandle plan, GpuComplex<float> *idata,
    GpuComplex<float> *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}

template<>
cufftResult cufftExec<double>(
    cufftHandle plan, GpuComplex<double> *idata,
    GpuComplex<double> *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

template<typename FloatType>
Status setup_spreader(int rank, FloatType eps, double upsampling_factor,
                      KernelEvaluationMethod kernel_evaluation_method,
                      SpreadParameters<FloatType>& spread_params);

template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadParameters<FloatType> &spread_params);

void set_bin_sizes(TransformType type, int rank, Options& options);

template<typename FloatType>
Status set_grid_size(int ms,
                     int bin_size,
                     const Options& options,
                     const SpreadParameters<FloatType>& spread_params,
                     int* grid_size);

__device__ int CalcGlobalIdxV2(int xidx, int yidx, int zidx, int nbinx, int nbiny, int nbinz) {
  return xidx + yidx * nbinx + zidx * nbinx * nbiny;
}

template<typename FloatType>
__global__ void CalcBinSizeNoGhost2DKernel(int M, int nf1, int nf2, int  bin_size_x, 
    int bin_size_y, int nbinx, int nbiny, int* bin_sizes, FloatType *x, FloatType *y, 
    int* sortidx, int pirange) {
  int binidx, binx, biny;
  int oldidx;
  FloatType x_rescaled, y_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<M; i += gridDim.x * blockDim.x) {
    x_rescaled = RESCALE(x[i], nf1, pirange);
    y_rescaled = RESCALE(y[i], nf2, pirange);
    binx = floor(x_rescaled / bin_size_x);
    binx = binx >= nbinx ? binx - 1 : binx;
    binx = binx < 0 ? 0 : binx;
    biny = floor(y_rescaled / bin_size_y);
    biny = biny >= nbiny ? biny - 1 : biny;
    biny = biny < 0 ? 0 : biny;
    binidx = binx + biny * nbinx;
    oldidx = GpuAtomicAdd(&bin_sizes[binidx], 1);
    sortidx[i] = oldidx;
    if (binx >= nbinx || biny >= nbiny) {
      sortidx[i] = -biny;
    }
  }
}

template<typename FloatType>
__global__ void CalcBinSizeNoGhost3DKernel(int M, int nf1, int nf2, int nf3,
    int bin_size_x, int bin_size_y, int bin_size_z,
    int nbinx, int nbiny, int nbinz, int* bin_sizes, FloatType *x, FloatType *y, FloatType *z,
    int* sortidx, int pirange) {
  int binidx, binx, biny, binz;
  int oldidx;
  FloatType x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<M; i += gridDim.x * blockDim.x) {
    x_rescaled = RESCALE(x[i], nf1, pirange);
    y_rescaled = RESCALE(y[i], nf2, pirange);
    z_rescaled = RESCALE(z[i], nf3, pirange);
    binx = floor(x_rescaled / bin_size_x);
    binx = binx >= nbinx ? binx - 1 : binx;
    binx = binx < 0 ? 0 : binx;

    biny = floor(y_rescaled / bin_size_y);
    biny = biny >= nbiny ? biny - 1 : biny;
    biny = biny < 0 ? 0 : biny;

    binz = floor(z_rescaled / bin_size_z);
    binz = binz >= nbinz ? binz - 1 : binz;
    binz = binz < 0 ? 0 : binz;
    binidx = binx + biny * nbinx + binz * nbinx * nbiny;
    oldidx = GpuAtomicAdd(&bin_sizes[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename FloatType>
__global__ void CalcInvertofGlobalSortIdx2DKernel(int M, int bin_size_x, int bin_size_y, 
    int nbinx, int nbiny, int* bin_startpts, int* sortidx, FloatType *x, FloatType *y, 
    int* index, int pirange, int nf1, int nf2) {
  int binx, biny;
  int binidx;
  FloatType x_rescaled, y_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<M; i += gridDim.x * blockDim.x) {
    x_rescaled = RESCALE(x[i], nf1, pirange);
    y_rescaled = RESCALE(y[i], nf2, pirange);
    binx = floor(x_rescaled / bin_size_x);
    binx = binx >= nbinx ? binx - 1 : binx;
    binx = binx < 0 ? 0 : binx;
    biny = floor(y_rescaled / bin_size_y);
    biny = biny >= nbiny ? biny - 1 : biny;
    biny = biny < 0 ? 0 : biny;
    binidx = binx + biny * nbinx;

    index[bin_startpts[binidx]+sortidx[i]] = i;
  }
}

template<typename FloatType>
__global__ void CalcInvertofGlobalSortIdx3DKernel(int M, int bin_size_x, int bin_size_y,
    int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
    int* sortidx, FloatType *x, FloatType *y, FloatType *z, int* index, int pirange, int nf1,
    int nf2, int nf3) {
  int binx, biny, binz;
  int binidx;
  FloatType x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<M; i += gridDim.x * blockDim.x) {
    x_rescaled = RESCALE(x[i], nf1, pirange);
    y_rescaled = RESCALE(y[i], nf2, pirange);
    z_rescaled = RESCALE(z[i], nf3, pirange);
    binx = floor(x_rescaled / bin_size_x);
    binx = binx >= nbinx ? binx - 1 : binx;
    binx = binx < 0 ? 0 : binx;
    biny = floor(y_rescaled / bin_size_y);
    biny = biny >= nbiny ? biny - 1 : biny;
    biny = biny < 0 ? 0 : biny;
    binz = floor(z_rescaled / bin_size_z);
    binz = binz >= nbinz ? binz - 1 : binz;
    binz = binz < 0 ? 0 : binz;
    binidx = CalcGlobalIdxV2(binx, biny, binz, nbinx, nbiny, nbinz);

    index[bin_startpts[binidx]+sortidx[i]] = i;
  }
}

__global__ void TrivialGlobalSortIdxKernel(int M, int* index) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<M; i += gridDim.x * blockDim.x) {
    index[i] = i;
  }
}

__global__ void CalcSubproblemKernel(int* bin_sizes, int* num_subprob, int max_subprob_size,
    int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i<numbins;
    i += gridDim.x * blockDim.x) {
    num_subprob[i]=ceil(bin_sizes[i]/(float) max_subprob_size);
  }
}

__global__ void MapBinToSubproblemKernel(
    int* subprob_bins, int* subprob_start_pts, int* num_subprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < num_subprob[i]; j++) {
      subprob_bins[subprob_start_pts[i] + j] = i;
    }
  }
}

/* Kernel for copying fw to fk with amplication by prefac / ker */
// Note: assume modeord = 0: CMCL - compatible mode ordering in fk (from -N / 2 up 
// to N / 2 - 1)
template<typename FloatType>
__global__ void Deconvolve2DKernel(
    int ms, int mt, int nf1, int nf2, GpuComplex<FloatType>* fw, GpuComplex<FloatType> *fk, 
    FloatType *fwkerhalf1, FloatType *fwkerhalf2)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<ms * mt; i += blockDim.x * gridDim.x) {
    int k1 = i % ms;
    int k2 = i / ms;
    int outidx = k1 + k2 * ms;
    int w1 = k1 - ms / 2 >= 0 ? k1 - ms / 2 : nf1 + k1 - ms / 2;
    int w2 = k2 - mt / 2 >= 0 ? k2 - mt / 2 : nf2 + k2 - mt / 2;
    int inidx = w1 + w2 * nf1;

    FloatType kervalue = fwkerhalf1[abs(k1 - ms / 2)]*fwkerhalf2[abs(k2 - mt / 2)];
    fk[outidx].x = fw[inidx].x / kervalue;
    fk[outidx].y = fw[inidx].y / kervalue;
  }
}

template<typename FloatType>
__global__ void Deconvolve3DKernel(
    int ms, int mt, int mu, int nf1, int nf2, int nf3, GpuComplex<FloatType>* fw, 
    GpuComplex<FloatType> *fk, FloatType *fwkerhalf1, FloatType *fwkerhalf2, FloatType *fwkerhalf3)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<ms * mt * mu; i += blockDim.x*
    gridDim.x) {
    int k1 = i % ms;
    int k2 = (i / ms) % mt;
    int k3 = (i / ms / mt);
    int outidx = k1 + k2 * ms + k3 * ms * mt;
    int w1 = k1 - ms / 2 >= 0 ? k1 - ms / 2 : nf1 + k1 - ms / 2;
    int w2 = k2 - mt / 2 >= 0 ? k2 - mt / 2 : nf2 + k2 - mt / 2;
    int w3 = k3 - mu / 2 >= 0 ? k3 - mu / 2 : nf3 + k3 - mu / 2;
    int inidx = w1 + w2 * nf1 + w3 * nf1 * nf2;

    FloatType kervalue = fwkerhalf1[abs(k1 - ms / 2)]*fwkerhalf2[abs(k2 - mt / 2)]*
      fwkerhalf3[abs(k3 - mu / 2)];
    fk[outidx].x = fw[inidx].x / kervalue;
    fk[outidx].y = fw[inidx].y / kervalue;
    //fk[outidx].x = kervalue;
    //fk[outidx].y = kervalue;
  }
}

/* Kernel for copying fk to fw with same amplication */
template<typename FloatType>
__global__ void Amplify2DKernel(
    int ms, int mt, int nf1, int nf2, GpuComplex<FloatType>* fw, GpuComplex<FloatType> *fk, 
    FloatType *fwkerhalf1, FloatType *fwkerhalf2)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt; i += blockDim.x * gridDim.x) {
    int k1 = i % ms;
    int k2 = i / ms;
    int inidx = k1 + k2 * ms;
    int w1 = k1 - ms / 2 >= 0 ? k1 - ms / 2 : nf1 + k1 - ms / 2;
    int w2 = k2 - mt / 2 >= 0 ? k2 - mt / 2 : nf2 + k2 - mt / 2;
    int outidx = w1 + w2 * nf1;

    FloatType kervalue = fwkerhalf1[abs(k1 - ms / 2)]*fwkerhalf2[abs(k2 - mt / 2)];
    fw[outidx].x = fk[inidx].x / kervalue;
    fw[outidx].y = fk[inidx].y / kervalue;
  }
}

template<typename FloatType>
__global__ void Amplify3DKernel(
    int ms, int mt, int mu, int nf1, int nf2, int nf3, GpuComplex<FloatType>* fw, 
    GpuComplex<FloatType> *fk, FloatType *fwkerhalf1, FloatType *fwkerhalf2, FloatType *fwkerhalf3)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt * mu; 
    i += blockDim.x * gridDim.x) {
    int k1 = i % ms;
    int k2 = (i / ms) % mt;
    int k3 = (i / ms / mt);
    int inidx = k1 + k2 * ms + k3 * ms * mt;
    int w1 = k1 - ms / 2 >= 0 ? k1 - ms / 2 : nf1 + k1 - ms / 2;
    int w2 = k2 - mt / 2 >= 0 ? k2 - mt / 2 : nf2 + k2 - mt / 2;
    int w3 = k3 - mu / 2 >= 0 ? k3 - mu / 2 : nf3 + k3 - mu / 2;
    int outidx = w1 + w2 * nf1 + w3 * nf1 * nf2;

    FloatType kervalue = fwkerhalf1[abs(k1 - ms / 2)]*fwkerhalf2[abs(k2 - mt / 2)]*
      fwkerhalf3[abs(k3 - mu / 2)];
    fw[outidx].x = fk[inidx].x / kervalue;
    fw[outidx].y = fk[inidx].y / kervalue;
  }
}

/* ES ("exp sqrt") kernel evaluation at single real argument:
    phi(x) = exp(beta.sqrt(1 - (2x / n_s)^2)),    for |x| < kernel_width / 2
    related to an asymptotic approximation to the Kaiser--Bessel, itself an
    approximation to prolate spheroidal wavefunction (PSWF) of order 0.
    This is the "reference implementation", used by eg common / onedim_* 
    2 / 17 / 17 */
template<typename FloatType>
static __forceinline__ __device__ FloatType EvaluateKernel(
    FloatType x, FloatType es_c, FloatType es_beta, int ns) {
  return abs(x) < ns / 2.0 ? exp(es_beta * (sqrt(1.0 - es_c * x * x))) : 0.0;
}

// Fill ker[] with Horner piecewise poly approx to [-w / 2, w / 2] ES kernel eval at
// x_j = x + j,  for j = 0,..,w - 1.  Thus x in [-w / 2,-w / 2 + 1].   w is aka ns.
// This is the current evaluation method, since it's faster (except i7 w = 16).
// Two upsampfacs implemented. Params must match ref formula. Barnett 4 / 24 / 18
template<typename FloatType>
static __inline__ __device__ void EvaluateKernelVectorHorner(
    FloatType *ker, const FloatType x, const int w, 
    const double upsampling_factor) {
  FloatType z = 2 * x + w - 1.0;         // scale so local grid offset z in [-1, 1]
  // insert the auto - generated code which expects z, w args, writes to ker...
  if (upsampling_factor == 2.0) {     // floating point equality is fine here
    #include "tensorflow_nufft/cc/kernels/kernel_horner_sigma2.inc"
  }
}

template<typename FloatType>
static __inline__ __device__ void EvaluateKernelVector(
    FloatType *ker, const FloatType x, const double w, const double es_c, 
    const double es_beta) {
  for (int i = 0; i < w; i++) {
    ker[i] = EvaluateKernel<FloatType>(abs(x + i), es_c, es_beta, w);		
  }
}

template<typename FloatType>
__global__ void SpreadNuptsDriven2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType es_c, FloatType es_beta, int *idxnupts, int pirange) {
  int xstart, ystart, xend, yend;
  int xx, yy, ix, iy;
  int outidx;
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];

  FloatType x_rescaled, y_rescaled;
  FloatType kervalue1, kervalue2;
  GpuComplex<FloatType> cnow;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    cnow = c[idxnupts[i]];

    xstart = ceil(x_rescaled - ns / 2.0);
    ystart = ceil(y_rescaled - ns / 2.0);
    xend = floor(x_rescaled + ns / 2.0);
    yend = floor(y_rescaled + ns / 2.0);

    FloatType x1 = (FloatType)xstart - x_rescaled;
    FloatType y1 = (FloatType)ystart - y_rescaled;
    EvaluateKernelVector(ker1, x1, ns, es_c, es_beta);
    EvaluateKernelVector(ker2, y1, ns, es_c, es_beta);
    for (yy = ystart; yy<=yend; yy++) {
      for (xx = xstart; xx<=xend; xx++) {
        ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        outidx = ix + iy * nf1;
        kervalue1 = ker1[xx - xstart];
        kervalue2 = ker2[yy - ystart];
        GpuAtomicAdd(&fw[outidx].x, cnow.x * kervalue1 * kervalue2);
        GpuAtomicAdd(&fw[outidx].y, cnow.y * kervalue1 * kervalue2);
      }
    }
  }
}

template<typename FloatType>
__global__ void SpreadNuptsDrivenHorner2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType sigma, int* idxnupts, int pirange) {
  int xx, yy, ix, iy;
  int outidx;
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];
  FloatType ker1val, ker2val;

  FloatType x_rescaled, y_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<M; i += blockDim.x * gridDim.x) {
    x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    cnow = c[idxnupts[i]];
    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);

    FloatType x1 = (FloatType)xstart - x_rescaled;
    FloatType y1 = (FloatType)ystart - y_rescaled;
    EvaluateKernelVectorHorner(ker1, x1, ns, sigma);
    EvaluateKernelVectorHorner(ker2, y1, ns, sigma);
    for (yy = ystart; yy<=yend; yy++) {
      for (xx = xstart; xx<=xend; xx++) {
        ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        outidx = ix + iy * nf1;
        ker1val = ker1[xx - xstart];
        ker2val = ker2[yy - ystart];
        FloatType kervalue = ker1val * ker2val;
        GpuAtomicAdd(&fw[outidx].x, cnow.x * kervalue);
        GpuAtomicAdd(&fw[outidx].y, cnow.y * kervalue);
      }
    }
  }
}

template<typename FloatType>
__global__ void SpreadSubproblem2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType es_c, FloatType es_beta, FloatType sigma, int* binstartpts,
    int* bin_sizes, int bin_size_x, int bin_size_y, int* subprob_bins,
    int* subprob_start_pts, int* num_subprob, int max_subprob_size, int nbinx, 
    int nbiny, int* idxnupts, int pirange) {
  // Shared memory pointers cannot be declared with a type template because
  // it results in a "declaration is incompatible with previous declaration"
  // error. To get around this issue, we declare the shared memory pointer as
  // `unsigned char` and then cast it to the appropriate type. See also
  // https://stackoverflow.com / a/27570775 / 9406746
  // Note: `nvcc` emits a warning warning for this code: "#1886 - D: specified
  // alignment (16) is different from alignment (8) specified on a previous
  // declaration". This can be safely ignored and is disabled in the Makefile.
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy;
  int outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = (bidx / nbinx) * bin_size_y;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];
  
  for (int i = threadIdx.x; i<N; i += blockDim.x) {
    fwshared[i].x = 0.0;
    fwshared[i].y = 0.0;
  }
  __syncthreads();

  FloatType x_rescaled, y_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    cnow = c[idxnupts[idx]];

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;

    FloatType x1 = (FloatType)xstart + xoffset - x_rescaled;
    FloatType y1 = (FloatType)ystart + yoffset - y_rescaled;
    EvaluateKernelVector(ker1, x1, ns, es_c, es_beta);
    EvaluateKernelVector(ker2, y1, ns, es_c, es_beta);

    for (int yy = ystart; yy<=yend; yy++) {
      iy = yy + ceil(ns / 2.0);
      if (iy >= (bin_size_y + (int) ceil(ns / 2.0) * 2) || iy<0) break;
      for (int xx = xstart; xx<=xend; xx++) {
        ix = xx + ceil(ns / 2.0);
        if (ix >= (bin_size_x + (int) ceil(ns / 2.0) * 2) || ix<0) break;
        outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
        FloatType kervalue1 = ker1[xx - xstart];
        FloatType kervalue2 = ker2[yy - ystart];
        GpuAtomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2);
        GpuAtomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2);
      }
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int k = threadIdx.x; k<N; k += blockDim.x) {
    int i = k % (int) (bin_size_x + 2 * ceil(ns / 2.0) );
    int j = k /( bin_size_x + 2 * ceil(ns / 2.0) );
    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      outidx = ix + iy * nf1;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
      GpuAtomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
      GpuAtomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
    }
  }
}

template<typename FloatType>
__global__ void SpreadSubproblemHorner2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType sigma, int* binstartpts, int* bin_sizes, int bin_size_x,
    int bin_size_y, int* subprob_bins, int* subprob_start_pts, int* num_subprob,
    int max_subprob_size, int nbinx, int nbiny, int* idxnupts, int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy, outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = (bidx / nbinx) * bin_size_y;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));
  
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];


  for (int i = threadIdx.x; i<N; i += blockDim.x) {
    fwshared[i].x = 0.0;
    fwshared[i].y = 0.0;
  }
  __syncthreads();

  FloatType x_rescaled, y_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    cnow = c[idxnupts[idx]];

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;

    EvaluateKernelVectorHorner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart + yoffset - y_rescaled, ns, sigma);

    for (int yy = ystart; yy<=yend; yy++) {
      iy = yy + ceil(ns / 2.0);
      if (iy >= (bin_size_y + (int) ceil(ns / 2.0) * 2) || iy<0) break;
      FloatType kervalue2 = ker2[yy - ystart];
      for (int xx = xstart; xx<=xend; xx++) {
        ix = xx + ceil(ns / 2.0);
        if (ix >= (bin_size_x + (int) ceil(ns / 2.0) * 2) || ix<0) break;
        outidx = ix + iy * (bin_size_x+ (int) ceil(ns / 2.0) * 2);
        FloatType kervalue1 = ker1[xx - xstart];
        GpuAtomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2);
        GpuAtomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2);
      }
    }
  }
  __syncthreads();

  /* write to global memory */
  for (int k = threadIdx.x; k<N; k += blockDim.x) {
    int i = k % (int) (bin_size_x + 2 * ceil(ns / 2.0) );
    int j = k /( bin_size_x + 2 * ceil(ns / 2.0) );
    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      outidx = ix + iy * nf1;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
      GpuAtomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
      GpuAtomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
    }
  }
}

template<typename FloatType>
__global__ void InterpNuptsDriven2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType es_c, FloatType es_beta, int* idxnupts, int pirange) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<M; i += blockDim.x * gridDim.x) {

    FloatType x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    FloatType y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        
    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);
    GpuComplex<FloatType> cnow;
    cnow.x = 0.0;
    cnow.y = 0.0;
    for (int yy = ystart; yy<=yend; yy++) {
      FloatType disy = abs(y_rescaled - yy);
      FloatType kervalue2 = EvaluateKernel(disy, es_c, es_beta, ns);
      for (int xx = xstart; xx<=xend; xx++) {
        int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        int inidx = ix + iy * nf1;
        FloatType disx = abs(x_rescaled - xx);
        FloatType kervalue1 = EvaluateKernel(disx, es_c, es_beta, ns);
        cnow.x += fw[inidx].x * kervalue1 * kervalue2;
        cnow.y += fw[inidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[i]].x = cnow.x;
    c[idxnupts[i]].y = cnow.y;
  }

}

template<typename FloatType>
__global__ void InterpNuptsDrivenHorner2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType sigma, int* idxnupts, int pirange) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    FloatType x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    FloatType y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);

    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);

    GpuComplex<FloatType> cnow;
    cnow.x = 0.0;
    cnow.y = 0.0;
    FloatType ker1[kMaxKernelWidth];
    FloatType ker2[kMaxKernelWidth];

    EvaluateKernelVectorHorner(ker1, xstart - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart - y_rescaled, ns, sigma);

    for (int yy = ystart; yy <= yend; yy++) {
      FloatType disy = abs(y_rescaled - yy);
      FloatType kervalue2 = ker2[yy - ystart];
      for (int xx = xstart; xx<=xend; xx++) {
        int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        int inidx = ix + iy * nf1;
        FloatType disx = abs(x_rescaled - xx);
        FloatType kervalue1 = ker1[xx - xstart];
        cnow.x += fw[inidx].x * kervalue1 * kervalue2;
        cnow.y += fw[inidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[i]].x = cnow.x;
    c[idxnupts[i]].y = cnow.y;
  }

}

template<typename FloatType>
__global__ void InterpSubproblem2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2,
    FloatType es_c, FloatType es_beta, FloatType sigma, int* binstartpts,
    int* bin_sizes, int bin_size_x, int bin_size_y, int* subprob_bins,
    int* subprob_start_pts, int* num_subprob, int max_subprob_size, int nbinx, 
    int nbiny, int* idxnupts, int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy;
  int outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = (bidx / nbinx) * bin_size_y;
  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));

  for (int k = threadIdx.x;k<N; k += blockDim.x) {
    int i = k % (int) (bin_size_x + 2 * ceil(ns / 2.0) );
    int j = k /( bin_size_x + 2 * ceil(ns / 2.0) );
    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      outidx = ix + iy * nf1;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
      fwshared[sharedidx].x = fw[outidx].x;
      fwshared[sharedidx].y = fw[outidx].y;
    }
  }
  __syncthreads();

  FloatType x_rescaled, y_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    cnow.x = 0.0;
    cnow.y = 0.0;

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;

    for (int yy = ystart; yy<=yend; yy++) {
      FloatType disy = abs(y_rescaled - (yy + yoffset));
      FloatType kervalue2 = EvaluateKernel(disy, es_c, es_beta, ns);
      for (int xx = xstart; xx<=xend; xx++) {
        ix = xx + ceil(ns / 2.0);
        iy = yy + ceil(ns / 2.0);
        outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
        FloatType disx = abs(x_rescaled - (xx + xoffset));
        FloatType kervalue1 = EvaluateKernel(disx, es_c, es_beta, ns);
        cnow.x += fwshared[outidx].x * kervalue1 * kervalue2;
        cnow.y += fwshared[outidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

template<typename FloatType>
__global__ void InterpSubproblemHorner2DKernel(
    FloatType *x, FloatType *y, GpuComplex<FloatType> *c, GpuComplex<FloatType> *fw, int M, 
    const int ns, int nf1, int nf2, FloatType sigma, int* binstartpts, int* bin_sizes, 
    int bin_size_x, int bin_size_y, int* subprob_bins, int* subprob_start_pts, 
    int* num_subprob, int max_subprob_size, int nbinx, int nbiny, int* idxnupts, 
    int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy;
  int outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = (bidx / nbinx) * bin_size_y;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));


  for (int k = threadIdx.x;k<N; k += blockDim.x) {
    int i = k % (int) (bin_size_x + 2 * ceil(ns / 2.0) );
    int j = k /( bin_size_x + 2 * ceil(ns / 2.0) );
    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      outidx = ix + iy * nf1;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
      fwshared[sharedidx].x = fw[outidx].x;
      fwshared[sharedidx].y = fw[outidx].y;
    }
  }
  __syncthreads();

  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];

  FloatType x_rescaled, y_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    cnow.x = 0.0;
    cnow.y = 0.0;

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;

    EvaluateKernelVectorHorner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
    
    for (int yy = ystart; yy<=yend; yy++) {
      FloatType disy = abs(y_rescaled - (yy + yoffset));
      FloatType kervalue2 = ker2[yy - ystart];
      for (int xx = xstart; xx<=xend; xx++) {
        ix = xx + ceil(ns / 2.0);
        iy = yy + ceil(ns / 2.0);
        outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
    
        FloatType kervalue1 = ker1[xx - xstart];
        cnow.x += fwshared[outidx].x * kervalue1 * kervalue2;
        cnow.y += fwshared[outidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

template<typename FloatType>
__global__ void SpreadNuptsDrivenHorner3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType sigma, int* idxnupts, int pirange) {
  int xx, yy, zz, ix, iy, iz;
  int outidx;
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];
  FloatType ker3[kMaxKernelWidth];

  FloatType ker1val, ker2val, ker3val;

  FloatType x_rescaled, y_rescaled, z_rescaled;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<M; i += blockDim.x * gridDim.x) {
    x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int zstart = ceil(z_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);
    int zend = floor(z_rescaled + ns / 2.0);

    FloatType x1 = (FloatType)xstart - x_rescaled;
    FloatType y1 = (FloatType)ystart - y_rescaled;
    FloatType z1 = (FloatType)zstart - z_rescaled;

    EvaluateKernelVectorHorner(ker1, x1, ns, sigma);
    EvaluateKernelVectorHorner(ker2, y1, ns, sigma);
    EvaluateKernelVectorHorner(ker3, z1, ns, sigma);
    for (zz = zstart; zz<=zend; zz++) {
      ker3val = ker3[zz - zstart];
      for (yy = ystart; yy<=yend; yy++) {
        ker2val = ker2[yy - ystart];
        for (xx = xstart; xx<=xend; xx++) {
          ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
          iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
          outidx = ix + iy * nf1 + iz * nf1 * nf2;
          ker1val = ker1[xx - xstart];
          FloatType kervalue = ker1val * ker2val * ker3val;
          GpuAtomicAdd(&fw[outidx].x, c[idxnupts[i]].x * kervalue);
          GpuAtomicAdd(&fw[outidx].y, c[idxnupts[i]].y * kervalue);
        }
      }
    }
  }
}

template<typename FloatType>
__global__ void SpreadNuptsDriven3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType es_c, FloatType es_beta, int* idxnupts, int pirange) {
  int xx, yy, zz, ix, iy, iz;
  int outidx;
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];
  FloatType ker3[kMaxKernelWidth];

  FloatType x_rescaled, y_rescaled, z_rescaled;
  FloatType ker1val, ker2val, ker3val;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i<M; i += blockDim.x * gridDim.x) {
    x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int zstart = ceil(z_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);
    int zend = floor(z_rescaled + ns / 2.0);

    FloatType x1 = (FloatType)xstart - x_rescaled;
    FloatType y1 = (FloatType)ystart - y_rescaled;
    FloatType z1 = (FloatType)zstart - z_rescaled;

    EvaluateKernelVector(ker1, x1, ns, es_c, es_beta);
    EvaluateKernelVector(ker2, y1, ns, es_c, es_beta);
    EvaluateKernelVector(ker3, z1, ns, es_c, es_beta);
    for (zz = zstart; zz<=zend; zz++) {
      ker3val = ker3[zz - zstart];
      for (yy = ystart; yy<=yend; yy++) {
        ker2val = ker2[yy - ystart];
        for (xx = xstart; xx<=xend; xx++) {
          ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
          iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
          outidx = ix + iy * nf1 + iz * nf1 * nf2;

          ker1val = ker1[xx - xstart];
          FloatType kervalue = ker1val * ker2val * ker3val;

          GpuAtomicAdd(&fw[outidx].x, c[idxnupts[i]].x * kervalue);
          GpuAtomicAdd(&fw[outidx].y, c[idxnupts[i]].y * kervalue);
        }
      }
    }
  }
}

template<typename FloatType>
__global__ void SpreadSubproblemHorner3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType sigma, int* binstartpts, int* bin_sizes, int bin_size_x,
    int bin_size_y, int bin_size_z,
    int* subprob_bins, int* subprob_start_pts, int* num_subprob,
    int max_subprob_size, int nbinx, int nbiny, int nbinz, int* idxnupts,
    int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend, zstart, zend;
  int bidx = subprob_bins[blockIdx.x];
  int binsubp_idx = blockIdx.x - subprob_start_pts[bidx];
  int ix, iy, iz, outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = ((bidx / nbinx)%nbiny) * bin_size_y;
  int zoffset = (bidx/ (nbinx * nbiny)) * bin_size_z;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0))*
    (bin_size_z + 2 * ceil(ns / 2.0));


  for (int i = threadIdx.x; i<N; i += blockDim.x) {
    fwshared[i].x = 0.0;
    fwshared[i].y = 0.0;
  }
  __syncthreads();
  FloatType x_rescaled, y_rescaled, z_rescaled;
  GpuComplex<FloatType> cnow;

  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    FloatType ker1[kMaxKernelWidth];
    FloatType ker2[kMaxKernelWidth];
    FloatType ker3[kMaxKernelWidth];

    int nuptsidx = idxnupts[ptstart + i];
    x_rescaled = RESCALE(x[nuptsidx],nf1, pirange);
    y_rescaled = RESCALE(y[nuptsidx],nf2, pirange);
    z_rescaled = RESCALE(z[nuptsidx],nf3, pirange);
    cnow = c[nuptsidx];

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;
    zend   = floor(z_rescaled + ns / 2.0) - zoffset;

    EvaluateKernelVectorHorner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker3, zstart + zoffset - z_rescaled, ns, sigma);

      for (int zz = zstart; zz<=zend; zz++) {
      FloatType kervalue3 = ker3[zz - zstart];
      iz = zz + ceil(ns / 2.0);
      if (iz >= (bin_size_z + (int) ceil(ns / 2.0) * 2) || iz<0) break;
      for (int yy = ystart; yy<=yend; yy++) {
        FloatType kervalue2 = ker2[yy - ystart];
        iy = yy + ceil(ns / 2.0);
        if (iy >= (bin_size_y + (int) ceil(ns / 2.0) * 2) || iy<0) break;
        for (int xx = xstart; xx<=xend; xx++) {
          ix = xx + ceil(ns / 2.0);
          if (ix >= (bin_size_x + (int) ceil(ns / 2.0) * 2) || ix<0) break;
          outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2)+
            iz * (bin_size_x + ceil(ns / 2.0) * 2)*
               (bin_size_y + ceil(ns / 2.0) * 2);
          FloatType kervalue1 = ker1[xx - xstart];
          GpuAtomicAdd(&fwshared[outidx].x,
                       cnow.x * kervalue1 * kervalue2 * kervalue3);
          GpuAtomicAdd(&fwshared[outidx].y,
                       cnow.y * kervalue1 * kervalue2 * kervalue3);
        }
      }
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i = n % static_cast<int>(bin_size_x + 2 * ceil(ns / 2.0));
    int j = static_cast<int>(n /(bin_size_x + 2 * ceil(ns / 2.0))) %
            static_cast<int>(bin_size_y + 2 * ceil(ns / 2.0));
    int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)));

    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    iz = zoffset - ceil(ns / 2.0) + k;

    if (ix < (nf1 + ceil(ns / 2.0)) &&
        iy < (nf2 + ceil(ns / 2.0)) &&
        iz < (nf3 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      outidx = ix + iy * nf1 + iz * nf1 * nf2;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                      k * (bin_size_x + ceil(ns / 2.0) * 2) *
                          (bin_size_y + ceil(ns / 2.0) * 2);
      GpuAtomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
      GpuAtomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
    }
  }
}

template<typename FloatType>
__global__ void SpreadSubproblem3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType es_c, FloatType es_beta, int* binstartpts, int* bin_sizes,
    int bin_size_x, int bin_size_y, int bin_size_z, int* subprob_bins,
    int* subprob_start_pts, int* num_subprob, int max_subprob_size,
    int nbinx, int nbiny, int nbinz, int* idxnupts, int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>)) unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared = reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend, zstart, zend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy, iz, outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = ((bidx / nbinx)%nbiny) * bin_size_y;
  int zoffset = (bidx/ (nbinx * nbiny)) * bin_size_z;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0))*
    (bin_size_z + 2 * ceil(ns / 2.0));

  for (int i = threadIdx.x; i<N; i += blockDim.x) {
    fwshared[i].x = 0.0;
    fwshared[i].y = 0.0;
  }
  __syncthreads();
  FloatType x_rescaled, y_rescaled, z_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i<nupts; i += blockDim.x) {
    FloatType ker1[kMaxKernelWidth];
    FloatType ker2[kMaxKernelWidth];
    FloatType ker3[kMaxKernelWidth];
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    z_rescaled = RESCALE(z[idxnupts[idx]], nf3, pirange);
    cnow = c[idxnupts[idx]];

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;
    zend   = floor(z_rescaled + ns / 2.0) - zoffset;

    FloatType x1 = (FloatType)xstart + xoffset - x_rescaled;
    FloatType y1 = (FloatType)ystart + yoffset - y_rescaled;
    FloatType z1 = (FloatType)zstart + zoffset - z_rescaled;

    EvaluateKernelVector(ker1, x1, ns, es_c, es_beta);
    EvaluateKernelVector(ker2, y1, ns, es_c, es_beta);
    EvaluateKernelVector(ker3, z1, ns, es_c, es_beta);
    for (int zz = zstart; zz<=zend; zz++) {
      FloatType kervalue3 = ker3[zz - zstart];
      iz = zz + ceil(ns / 2.0);
      if (iz >= (bin_size_z + (int) ceil(ns / 2.0) * 2) || iz < 0) break;
      for (int yy = ystart; yy<=yend; yy++) {
        FloatType kervalue2 = ker2[yy - ystart];
        iy = yy + ceil(ns / 2.0);
        if (iy >= (bin_size_y + (int) ceil(ns / 2.0) * 2) || iy < 0) break;
        for (int xx = xstart; xx<=xend; xx++) {
          FloatType kervalue1 = ker1[xx - xstart];
          ix = xx + ceil(ns / 2.0);
          if (ix >= (bin_size_x + (int) ceil(ns / 2.0) * 2) || ix < 0) break;
          outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2) +
                   iz * (bin_size_x + ceil(ns / 2.0) * 2) *
                        (bin_size_y + ceil(ns / 2.0) * 2);
          GpuAtomicAdd(&fwshared[outidx].x,
                    cnow.x * kervalue1 * kervalue2 * kervalue3);
          GpuAtomicAdd(&fwshared[outidx].y,
                    cnow.y * kervalue1 * kervalue2 * kervalue3);
        }
      }
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i = n % static_cast<int>(bin_size_x + 2 * ceil(ns / 2.0));
    int j = static_cast<int>(n /(bin_size_x + 2 * ceil(ns / 2.0))) %
            static_cast<int>(bin_size_y + 2 * ceil(ns / 2.0));
    int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) *
            (bin_size_y + 2 * ceil(ns / 2.0)));

    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    iz = zoffset - ceil(ns / 2.0) + k;
    if (ix < (nf1 + ceil(ns / 2.0)) &&
        iy < (nf2 + ceil(ns / 2.0)) &&
        iz < (nf3 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      outidx = ix + iy * nf1 + iz * nf1 * nf2;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                      k * (bin_size_x + ceil(ns / 2.0) * 2) *
                      (bin_size_y + ceil(ns / 2.0) * 2);
      GpuAtomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
      GpuAtomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
    }
  }
}

template<typename FloatType>
__global__ void InterpNuptsDriven3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType es_c, FloatType es_beta, int *idxnupts, int pirange) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    FloatType x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    FloatType y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    FloatType z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);
    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int zstart = ceil(z_rescaled - ns / 2.0);
    int xend = floor(x_rescaled + ns / 2.0);
    int yend = floor(y_rescaled + ns / 2.0);
    int zend = floor(z_rescaled + ns / 2.0);
    GpuComplex<FloatType> cnow;
    cnow.x = 0.0;
    cnow.y = 0.0;
    for (int zz = zstart; zz <= zend; zz++) {
      FloatType disz = abs(z_rescaled - zz);
      FloatType kervalue3 = EvaluateKernel(disz, es_c, es_beta, ns);
      for (int yy = ystart; yy <= yend; yy++) {
        FloatType disy = abs(y_rescaled - yy);
        FloatType kervalue2 = EvaluateKernel(disy, es_c, es_beta, ns);
        for (int xx = xstart; xx <= xend; xx++) {
          int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
          int iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);

          int inidx = ix + iy * nf1 + iz * nf2 * nf1;

          FloatType disx = abs(x_rescaled - xx);
          FloatType kervalue1 = EvaluateKernel(disx, es_c, es_beta, ns);
          cnow.x += fw[inidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fw[inidx].y * kervalue1 * kervalue2 * kervalue3;
        }
      }
    }
    c[idxnupts[i]].x = cnow.x;
    c[idxnupts[i]].y = cnow.y;
  }
}

template<typename FloatType>
__global__ void InterpNuptsDrivenHorner3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType sigma, int *idxnupts, int pirange) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    FloatType x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
    FloatType y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
    FloatType z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

    int xstart = ceil(x_rescaled - ns / 2.0);
    int ystart = ceil(y_rescaled - ns / 2.0);
    int zstart = ceil(z_rescaled - ns / 2.0);

    int xend   = floor(x_rescaled + ns / 2.0);
    int yend   = floor(y_rescaled + ns / 2.0);
    int zend   = floor(z_rescaled + ns / 2.0);

    GpuComplex<FloatType> cnow;
    cnow.x = 0.0;
    cnow.y = 0.0;

    FloatType ker1[kMaxKernelWidth];
    FloatType ker2[kMaxKernelWidth];
    FloatType ker3[kMaxKernelWidth];

    EvaluateKernelVectorHorner(ker1, xstart - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart - y_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker3, zstart - z_rescaled, ns, sigma);

    for (int zz = zstart; zz <= zend; zz++) {
      FloatType kervalue3 = ker3[zz - zstart];
      int iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
      for (int yy = ystart; yy <= yend; yy++) {
        FloatType kervalue2 = ker2[yy - ystart];
        int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        for (int xx = xstart; xx <= xend; xx++) {
          int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          int inidx = ix + iy * nf1 + iz * nf2 * nf1;
          FloatType kervalue1 = ker1[xx - xstart];
          cnow.x += fw[inidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fw[inidx].y * kervalue1 * kervalue2 * kervalue3;
        }
      }
    }
    c[idxnupts[i]].x = cnow.x;
    c[idxnupts[i]].y = cnow.y;
  }
}

template<typename FloatType>
__global__ void InterpSubproblem3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType es_c, FloatType es_beta, int* binstartpts, int* bin_sizes,
    int bin_size_x, int bin_size_y, int bin_size_z, int* subprob_bins,
    int* subprob_start_pts, int* num_subprob, int max_subprob_size, int nbinx,
    int nbiny, int nbinz, int* idxnupts, int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>))
  unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared =
      reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend, zstart, zend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy, iz;
  int outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size,
                  bin_sizes[bidx] - binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = ((bidx / nbinx)%nbiny) * bin_size_y;
  int zoffset = (bidx/ (nbinx * nbiny)) * bin_size_z;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0))*
      (bin_size_z + 2 * ceil(ns / 2.0));

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i = n % static_cast<int>(bin_size_x + 2 * ceil(ns / 2.0));
    int j = static_cast<int>(n /(bin_size_x + 2 * ceil(ns / 2.0))) %
            static_cast<int>(bin_size_y + 2 * ceil(ns / 2.0));
    int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) *
                 (bin_size_y + 2 * ceil(ns / 2.0)));

    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    iz = zoffset - ceil(ns / 2.0) + k;
    if (ix < (nf1 + ceil(ns / 2.0)) &&
        iy < (nf2 + ceil(ns / 2.0)) &&
        iz < (nf3 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      outidx = ix + iy * nf1 + iz * nf1 * nf2;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                      k * (bin_size_x + ceil(ns / 2.0) * 2) *
                          (bin_size_y + ceil(ns / 2.0) * 2);
      fwshared[sharedidx].x = fw[outidx].x;
      fwshared[sharedidx].y = fw[outidx].y;
    }
  }

  __syncthreads();

  FloatType x_rescaled, y_rescaled, z_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    z_rescaled = RESCALE(z[idxnupts[idx]], nf3, pirange);
    cnow.x = 0.0;
    cnow.y = 0.0;

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    zstart = ceil(z_rescaled - ns / 2.0) - zoffset;
    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;
    zend   = floor(z_rescaled + ns / 2.0) - zoffset;

    for (int zz = zstart; zz <= zend; zz++) {
      FloatType disz = abs(z_rescaled - zz);
      FloatType kervalue3 = EvaluateKernel(disz, es_c, es_beta, ns);
      iz = zz + ceil(ns / 2.0);
      for (int yy = ystart; yy <= yend; yy++) {
        FloatType disy = abs(y_rescaled - yy);
        FloatType kervalue2 = EvaluateKernel(disy, es_c, es_beta, ns);
        iy = yy + ceil(ns / 2.0);
        for (int xx = xstart; xx <= xend; xx++) {
          ix = xx + ceil(ns / 2.0);
          outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2) +
                    iz * (bin_size_x + ceil(ns / 2.0) * 2) *
                    (bin_size_y + ceil(ns / 2.0) * 2);

          FloatType disx = abs(x_rescaled - xx);
          FloatType kervalue1 = EvaluateKernel(disx, es_c, es_beta, ns);
          cnow.x += fwshared[outidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fwshared[outidx].y * kervalue1 * kervalue2 * kervalue3;
        }
      }
    }
    c[idxnupts[idx]].x = cnow.x;
    c[idxnupts[idx]].y = cnow.y;
  }
}

template<typename FloatType>
__global__ void InterpSubproblemHorner3DKernel(
    FloatType *x, FloatType *y, FloatType *z, GpuComplex<FloatType> *c,
    GpuComplex<FloatType> *fw, int M, const int ns, int nf1, int nf2, int nf3,
    FloatType sigma, int* binstartpts, int* bin_sizes, int bin_size_x,
    int bin_size_y, int bin_size_z, int* subprob_bins, int* subprob_start_pts,
    int* num_subprob, int max_subprob_size, int nbinx, int nbiny, int nbinz,
    int* idxnupts, int pirange) {
  extern __shared__ __align__(sizeof(GpuComplex<FloatType>))
  unsigned char fwshared_[];
  GpuComplex<FloatType> *fwshared =
      reinterpret_cast<GpuComplex<FloatType>*>(fwshared_);

  int xstart, ystart, xend, yend, zstart, zend;
  int subpidx = blockIdx.x;
  int bidx = subprob_bins[subpidx];
  int binsubp_idx = subpidx - subprob_start_pts[bidx];
  int ix, iy, iz;
  int outidx;
  int ptstart = binstartpts[bidx]+binsubp_idx * max_subprob_size;
  int nupts = min(max_subprob_size, bin_sizes[bidx]-binsubp_idx * max_subprob_size);

  int xoffset = (bidx % nbinx) * bin_size_x;
  int yoffset = ((bidx / nbinx)%nbiny) * bin_size_y;
  int zoffset = (bidx/ (nbinx * nbiny)) * bin_size_z;

  int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0))*
      (bin_size_z + 2 * ceil(ns / 2.0));

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i = n % static_cast<int>(bin_size_x + 2 * ceil(ns / 2.0));
    int j = static_cast<int>(n /(bin_size_x + 2 * ceil(ns / 2.0))) %
            static_cast<int>(bin_size_y + 2 * ceil(ns / 2.0));
    int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) *
                 (bin_size_y + 2 * ceil(ns / 2.0)));

    ix = xoffset - ceil(ns / 2.0) + i;
    iy = yoffset - ceil(ns / 2.0) + j;
    iz = zoffset - ceil(ns / 2.0) + k;
    if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0)) &&
        iz < (nf3 + ceil(ns / 2.0))) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      outidx = ix + iy * nf1 + iz * nf1 * nf2;
      int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                      k * (bin_size_x + ceil(ns / 2.0) * 2) *
                      (bin_size_y + ceil(ns / 2.0) * 2);
      fwshared[sharedidx].x = fw[outidx].x;
      fwshared[sharedidx].y = fw[outidx].y;
    }
  }
  __syncthreads();
  FloatType ker1[kMaxKernelWidth];
  FloatType ker2[kMaxKernelWidth];
  FloatType ker3[kMaxKernelWidth];
  FloatType x_rescaled, y_rescaled, z_rescaled;
  GpuComplex<FloatType> cnow;
  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    int idx = ptstart + i;
    x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
    z_rescaled = RESCALE(z[idxnupts[idx]], nf3, pirange);
    cnow.x = 0.0;
    cnow.y = 0.0;

    xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
    ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
    zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

    xend   = floor(x_rescaled + ns / 2.0) - xoffset;
    yend   = floor(y_rescaled + ns / 2.0) - yoffset;
    zend   = floor(z_rescaled + ns / 2.0) - zoffset;

    EvaluateKernelVectorHorner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
    EvaluateKernelVectorHorner(ker3, zstart + zoffset - z_rescaled, ns, sigma);
      for (int zz = zstart; zz <= zend; zz++) {
      FloatType kervalue3 = ker3[zz - zstart];
      iz = zz + ceil(ns / 2.0);
      for (int yy = ystart; yy <= yend; yy++) {
        FloatType kervalue2 = ker2[yy - ystart];
        iy = yy + ceil(ns / 2.0);
        for (int xx = xstart; xx <= xend; xx++) {
          ix = xx + ceil(ns / 2.0);
          outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2)+
               iz * (bin_size_x + ceil(ns / 2.0) * 2)*
                  (bin_size_y + ceil(ns / 2.0) * 2);
          FloatType kervalue1 = ker1[xx - xstart];
          cnow.x += fwshared[outidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fwshared[outidx].y * kervalue1 * kervalue2 * kervalue3;
            }
          }
    }
    c[idxnupts[idx]].x = cnow.x;
    c[idxnupts[idx]].y = cnow.y;
  }
}

}  // namespace

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::initialize(
    TransformType type,
    int rank,
    int* num_modes,
    FftDirection fft_direction,
    int num_transforms,
    FloatType tol,
    const Options& options) {

  this->idx_nupts_ = nullptr;
  this->sort_idx_ = nullptr;
  this->num_subprob_ = nullptr;
  this->bin_sizes_ = nullptr;
  this->bin_start_pts_ = nullptr;
  this->subprob_bins_ = nullptr;
  this->subprob_start_pts_ = nullptr;
  this->subprob_count_ = 0;
  this->c_ = nullptr;
  this->f_ = nullptr;

  if (type == TransformType::TYPE_3) {
    return errors::Unimplemented("type - 3 transforms are not implemented");
  }
  if (rank < 2 || rank > 3) {
    return errors::Unimplemented("rank ", rank, " is not implemented");
  }
  if (num_transforms < 1) {
    return errors::InvalidArgument("num_transforms must be >= 1");
  }

  // TODO(jmontalt): check options.
  //  - If mode_order == FFT, raise unimplemented error.
  //  - If check_bounds == true, raise unimplemented error.

  // Copy options.
  this->options_ = options;

  // Select kernel evaluation method.
  if (this->options_.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
    this->options_.kernel_evaluation_method = KernelEvaluationMethod::DIRECT;
  }

  // Select upsampling factor. Currently always defaults to 2.
  if (this->options_.upsampling_factor == 0.0) {
    this->options_.upsampling_factor = 2.0;
  }

  // Configure threading (irrelevant for GPU computation, but is used for some
  // CPU computations).
  if (this->options_.num_threads == 0) {
    this->options_.num_threads = OMP_GET_MAX_THREADS();
  }

  // Select whether or not to sort points.
  if (this->options_.sort_points == SortPoints::AUTO) {
    this->options_.sort_points = SortPoints::YES;
  }

  // Select spreading method.
  if (this->options_.spread_method == SpreadMethod::AUTO) {
    if (rank == 2 && type == TransformType::TYPE_1)
      this->options_.spread_method = SpreadMethod::SUBPROBLEM;
    else if (rank == 2 && type == TransformType::TYPE_2)
      this->options_.spread_method = SpreadMethod::NUPTS_DRIVEN;
    else if (rank == 3 && type == TransformType::TYPE_1)
      this->options_.spread_method = SpreadMethod::SUBPROBLEM;
    else if (rank == 3 && type == TransformType::TYPE_2)
      this->options_.spread_method = SpreadMethod::NUPTS_DRIVEN;
  }

  // This must be set before calling setup_spreader_for_nufft.
  this->spread_params_.spread_only = this->options_.spread_only;

  // Setup spreading options.
  TF_RETURN_IF_ERROR(setup_spreader_for_nufft(
      rank, tol, this->options_, this->spread_params_));

  this->rank_ = rank;
  this->num_modes_[0] = num_modes[0];
  this->num_modes_[1] = (this->rank_ > 1) ? num_modes[1] : 1;
  this->num_modes_[2] = (this->rank_ > 2) ? num_modes[2] : 1;
  this->mode_count_ = this->num_modes_[0] * this->num_modes_[1] *
                      this->num_modes_[2];

  // Set the bin sizes.
  set_bin_sizes(type, rank, this->options_);

  // Set the grid sizes.
  TF_RETURN_IF_ERROR(set_grid_size(
      this->num_modes_[0], this->options_.gpu_obin_size.x,
      this->options_, this->spread_params_, &this->grid_dims_[0]));
  if (rank > 1) {
    TF_RETURN_IF_ERROR(set_grid_size(
        this->num_modes_[1], this->options_.gpu_obin_size.y,
        this->options_, this->spread_params_, &this->grid_dims_[1]));
  } else {
    this->grid_dims_[1] = 1;
  }
  if (rank > 2) {
    TF_RETURN_IF_ERROR(set_grid_size(
        this->num_modes_[2], this->options_.gpu_obin_size.z,
        this->options_, this->spread_params_, &this->grid_dims_[2]));
  } else {
    this->grid_dims_[2] = 1;
  }

  this->grid_size_ = this->grid_dims_[0] * this->grid_dims_[1] *
                     this->grid_dims_[2];
  this->fft_direction_ = fft_direction;
  this->num_transforms_ = num_transforms;
  this->type_ = type;

  // Select maximum batch size.
  if (this->options_.max_batch_size == 0)
    // Heuristic from test codes.
    this->options_.max_batch_size = min(num_transforms, 8);

  if (this->type_ == TransformType::TYPE_1)
    this->spread_params_.spread_direction = SpreadDirection::SPREAD;
  if (this->type_ == TransformType::TYPE_2)
    this->spread_params_.spread_direction = SpreadDirection::INTERP;

  // Compute bin dimension sizes.
  this->bin_dims_[0] = this->options_.gpu_bin_size.x;
  this->bin_dims_[1] = this->rank_ > 1 ? this->options_.gpu_bin_size.y : 1;
  this->bin_dims_[2] = this->rank_ > 2 ? this->options_.gpu_bin_size.z : 1;

  // Compute number of bins.
  this->num_bins_[0] = 1;
  this->num_bins_[1] = 1;
  this->num_bins_[2] = 1;
  this->bin_count_ = 1;
  for (int i = 0; i < this->rank_; i++) {
    this->num_bins_[i] =
        (this->grid_dims_[i] + this->bin_dims_[i] - 1) / this->bin_dims_[i];
    this->bin_count_ *= this->num_bins_[i];
  }

  // Allocate memory for the bins.
  size_t bin_bytes = sizeof(int) * this->bin_count_;
  switch (this->options_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      if (this->spread_params_.sort_points == SortPoints::YES) {
        this->bin_sizes_ = reinterpret_cast<int*>(
            this->device_.allocate(bin_bytes));
        this->bin_start_pts_ = reinterpret_cast<int*>(
            this->device_.allocate(bin_bytes));
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      this->bin_sizes_ = reinterpret_cast<int*>(
          this->device_.allocate(bin_bytes));
      this->bin_start_pts_ = reinterpret_cast<int*>(
          this->device_.allocate(bin_bytes));
      this->num_subprob_ = reinterpret_cast<int*>(
          this->device_.allocate(bin_bytes));
      this->subprob_start_pts_ = reinterpret_cast<int*>(
          this->device_.allocate(sizeof(int) * (this->bin_count_ + 1)));
      break;
    case SpreadMethod::PAUL:
    case SpreadMethod::BLOCK_GATHER:
      return errors::Unimplemented("Invalid spread method: ",
          static_cast<int>(this->options_.spread_method));
  }

  // Perform some actions not needed in spread / interp only mode.
  if (!this->options_.spread_only) {
    // Allocate fine grid and set convenience pointer.
    TF_RETURN_IF_ERROR(this->context_->allocate_temp(
        DataTypeToEnum<std::complex<FloatType>>::value,
        TensorShape({this->grid_size_ * this->options_.max_batch_size}),
        &this->grid_tensor_));
    this->grid_data_ = reinterpret_cast<DType*>(
        this->grid_tensor_.flat<std::complex<FloatType>>().data());

    // For each dimension, calculate Fourier coefficients of the kernel for
    // deconvolution. The computation is performed on the CPU before
    // transferring the results the GPU.
    Tensor kernel_fseries_host[3];
    FloatType* kernel_fseries_host_data[3];
    for (int i = 0; i < this->rank_; i++) {
      // Number of Fourier coefficients.
      int num_coeffs = this->grid_dims_[i] / 2 + 1;

      // Allocate host memory and calculate the Fourier series on the CPU.
      AllocatorAttributes attr;
      attr.set_on_host(true);
      TF_RETURN_IF_ERROR(this->context_->allocate_temp(
          DataTypeToEnum<FloatType>::value, TensorShape({num_coeffs}),
          &kernel_fseries_host[i], attr));
      kernel_fseries_host_data[i] = reinterpret_cast<FloatType*>(
          kernel_fseries_host[i].flat<FloatType>().data());
      kernel_fseries_1d(this->grid_dims_[i], this->spread_params_,
                        kernel_fseries_host_data[i]);

      // Allocate device memory and save convenience accessors.
      TF_RETURN_IF_ERROR(this->context_->allocate_temp(
          DataTypeToEnum<FloatType>::value, TensorShape({num_coeffs}),
          &this->fseries_tensor_[i]));
      this->fseries_data_[i] = reinterpret_cast<FloatType*>(
          this->fseries_tensor_[i].flat<FloatType>().data());

      // Now copy coefficients to device.
      size_t num_bytes = sizeof(FloatType) * num_coeffs;
      this->device_.memcpyHostToDevice(
          reinterpret_cast<void*>(this->fseries_data_[i]),
          reinterpret_cast<void*>(kernel_fseries_host_data[i]),
          num_bytes);
    }

    // Make the cuFFT plan.
    int fft_dims[3];
    int *input_embed = fft_dims;
    int *output_embed = fft_dims;
    int input_distance = 0;
    int output_distance = 0;
    int input_stride = 1;
    int output_stride = 1;
    int batch_size = this->options_.max_batch_size;
    switch (this->rank_) {
      case 2: {
        fft_dims[0] = this->grid_dims_[1];
        fft_dims[1] = this->grid_dims_[0];
        input_distance = input_embed[0] * input_embed[1];
        output_distance = input_distance;
        break;
      }
      case 3: {
        fft_dims[0] = this->grid_dims_[2];
        fft_dims[1] = this->grid_dims_[1];
        fft_dims[2] = this->grid_dims_[0];
        input_distance = input_embed[0] * input_embed[1] * input_embed[2];
        output_distance = input_distance;
        break;
      }
      default:
        return errors::Unimplemented("Invalid rank: ", this->rank_);
    }

    cufftResult result = cufftPlanMany(
        &this->fft_plan_, this->rank_, fft_dims,
        input_embed, input_stride, input_distance,
        output_embed, output_stride, output_distance,
        kCufftType<FloatType>, batch_size);

    if (result != CUFFT_SUCCESS) {
      return errors::Internal("cufftPlanMany failed with code: ", result);
    }
  }
  return Status::OK();
}

template<typename FloatType>
Plan<GPUDevice, FloatType>::~Plan() {
  if (this->fft_plan_)
    cufftDestroy(this->fft_plan_);

  // Free memory allocated on the device. Some of these pointers are not
  // guaranteed to be allocated, but that's ok because `deallocate` will
  // perform no operation if passed a null pointer.
  this->device_.deallocate(this->bin_sizes_);
  this->bin_sizes_ = nullptr;
  this->device_.deallocate(this->bin_start_pts_);
  this->bin_start_pts_ = nullptr;
  this->device_.deallocate(this->subprob_bins_);
  this->subprob_bins_ = nullptr;
  this->device_.deallocate(this->num_subprob_);
  this->num_subprob_ = nullptr;
  this->device_.deallocate(this->subprob_start_pts_);
  this->subprob_start_pts_ = nullptr;
  this->device_.deallocate(this->idx_nupts_);
  this->idx_nupts_ = nullptr;
  this->device_.deallocate(this->sort_idx_);
  this->sort_idx_ = nullptr;
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::set_points(
    int num_points,
    FloatType* points_x,
    FloatType* points_y,
    FloatType* points_z) {

  this->num_points_ = num_points;
  this->points_[0] = points_x;
  this->points_[1] = this->rank_ > 1 ? points_y : nullptr;
  this->points_[2] = this->rank_ > 2 ? points_z : nullptr;

  if (this->idx_nupts_ != nullptr)
    this->device_.deallocate(this->idx_nupts_);
  if (this->sort_idx_ != nullptr)
    this->device_.deallocate(this->sort_idx_);

  size_t num_bytes = sizeof(int) * this->num_points_;
  switch (this->options_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      this->idx_nupts_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      if (this->spread_params_.sort_points == SortPoints::YES) {
        this->sort_idx_ = reinterpret_cast<int*>(
            this->device_.allocate(num_bytes));
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      this->idx_nupts_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      this->sort_idx_ = reinterpret_cast<int*>(
          this->device_.allocate(num_bytes));
      break;
    case SpreadMethod::PAUL:
    case SpreadMethod::BLOCK_GATHER:
      return errors::Unimplemented("Invalid spread method: ",
          static_cast<int>(this->options_.spread_method));
  }

  TF_RETURN_IF_ERROR(this->init_spreader());

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::execute(DType* d_c, DType* d_fk) {
  switch (this->type_) {
    case TransformType::TYPE_1:
      TF_RETURN_IF_ERROR(this->execute_type_1(d_c, d_fk));
      break;
    case TransformType::TYPE_2:
      TF_RETURN_IF_ERROR(this->execute_type_2(d_c, d_fk));
      break;
    case TransformType::TYPE_3:
      return errors::Unimplemented("type 3 transform is not implemented");
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::interp(DType* d_c, DType* d_fk) {
  int batch_size;
  DType* d_fkstart;
  DType* d_cstart;

  for (int i = 0; i * this->options_.max_batch_size < this->num_transforms_;
       i++) {
    batch_size = min(this->num_transforms_ - i * this->options_.max_batch_size,
                     this->options_.max_batch_size);
    d_cstart  = d_c  + i * this->options_.max_batch_size * this->num_points_;
    d_fkstart = d_fk + i * this->options_.max_batch_size * this->mode_count_;

    this->c_ = d_cstart;
    this->grid_data_ = d_fkstart;

    TF_RETURN_IF_ERROR(this->interp_batch(batch_size));
  }

  thrust::device_ptr<FloatType> dev_ptr(reinterpret_cast<FloatType*>(d_c));
  thrust::transform(thrust::cuda::par.on(this->device_.stream()), dev_ptr,
                    dev_ptr + 2 * this->num_transforms_ * this->num_points_,
                    dev_ptr, thrust::placeholders::_1 * static_cast<FloatType>(
                        this->spread_params_.kernel_scale));

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::spread(DType* d_c, DType* d_fk) {
  int batch_size;
  DType* d_fkstart;
  DType* d_cstart;

  for (int i = 0;
       i * this->options_.max_batch_size < this->num_transforms_;
       i++) {
    batch_size = min(this->num_transforms_ - i * this->options_.max_batch_size,
      this->options_.max_batch_size);
    d_cstart   = d_c + i * this->options_.max_batch_size * this->num_points_;
    d_fkstart  = d_fk + i * this->options_.max_batch_size * this->mode_count_;

    this->c_  = d_cstart;
    this->grid_data_ = d_fkstart;

    TF_RETURN_IF_ERROR(this->spread_batch(batch_size));
  }

  thrust::device_ptr<FloatType> dev_ptr(reinterpret_cast<FloatType*>(d_fk));
  thrust::transform(thrust::cuda::par.on(this->device_.stream()), dev_ptr,
                    dev_ptr + 2 * this->num_transforms_ * this->mode_count_,
                    dev_ptr, thrust::placeholders::_1 * static_cast<FloatType>(
                        this->spread_params_.kernel_scale));

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::execute_type_1(DType* d_c, DType* d_fk) {
  int batch_size;
  DType* d_fkstart;
  DType* d_cstart;
  for (int i = 0;
       i * this->options_.max_batch_size < this->num_transforms_;
       i++) {
    batch_size = min(this->num_transforms_ - i * this->options_.max_batch_size,
                     this->options_.max_batch_size);
    d_cstart = d_c + i * this->options_.max_batch_size * this->num_points_;
    d_fkstart = d_fk + i * this->options_.max_batch_size * this->mode_count_;
    this->c_ = d_cstart;
    this->f_ = d_fkstart;

    // Set fine grid to zero.
    size_t grid_bytes = sizeof(DType) * this->grid_size_ *
                        this->options_.max_batch_size;
    this->device_.memset(this->grid_data_, 0, grid_bytes);

    // Step 1: Spread
    TF_RETURN_IF_ERROR(this->spread_batch(batch_size));

    // Step 2: FFT
    auto result = cufftSetStream(this->fft_plan_, this->device_.stream());
    if (result != CUFFT_SUCCESS) {
      return errors::Internal(
          "Failed to associate cuFFT plan with CUDA stream: ", result);
    }
    result = cufftExec<FloatType>(
      this->fft_plan_, this->grid_data_, this->grid_data_,
      static_cast<int>(this->fft_direction_));
    if (result != CUFFT_SUCCESS) {
      return errors::Internal("cuFFT execute failed with code: ", result);
    }

    // Step 3: deconvolve and shuffle
    TF_RETURN_IF_ERROR(this->deconvolve_batch(batch_size));
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::execute_type_2(DType* d_c, DType* d_fk) {
  int batch_size;
  DType* d_fkstart;
  DType* d_cstart;
  for (int i = 0;
       i * this->options_.max_batch_size < this->num_transforms_;
       i++) {
    batch_size = min(this->num_transforms_ - i * this->options_.max_batch_size,
                     this->options_.max_batch_size);
    d_cstart  = d_c  + i * this->options_.max_batch_size * this->num_points_;
    d_fkstart = d_fk + i * this->options_.max_batch_size * this->mode_count_;

    this->c_ = d_cstart;
    this->f_ = d_fkstart;

    // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
    TF_RETURN_IF_ERROR(this->deconvolve_batch(batch_size));

    // Step 2: FFT
    this->device_.synchronize();  // Is this necessary?
    auto result = cufftSetStream(this->fft_plan_, this->device_.stream());
    if (result != CUFFT_SUCCESS) {
      return errors::Internal(
          "Failed to associate cuFFT plan with CUDA stream: ", result);
    }
    result = cufftExec<FloatType>(
      this->fft_plan_, this->grid_data_, this->grid_data_,
      static_cast<int>(this->fft_direction_));
    if (result != CUFFT_SUCCESS) {
      return errors::Internal("cuFFT execute failed with code: ", result);
    }

    // Step 3: interpolate
    TF_RETURN_IF_ERROR(this->interp_batch(batch_size));
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::spread_batch(int batch_size) {
  switch (this->options_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      TF_RETURN_IF_ERROR(this->spread_batch_nupts_driven(batch_size));
      break;
    case SpreadMethod::SUBPROBLEM:
      TF_RETURN_IF_ERROR(this->spread_batch_subproblem(batch_size));
      break;
    case SpreadMethod::PAUL:
    case SpreadMethod::BLOCK_GATHER:
      return errors::Unimplemented("spread method not implemented");
  }

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::interp_batch(int batch_size) {
  switch (this->options_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      TF_RETURN_IF_ERROR(this->interp_batch_nupts_driven(batch_size));
      break;
    case SpreadMethod::SUBPROBLEM:
      TF_RETURN_IF_ERROR(this->interp_batch_subproblem(batch_size));
      break;
    case SpreadMethod::PAUL:
    case SpreadMethod::BLOCK_GATHER:
      return errors::Unimplemented("interp method not implemented");
  }

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::spread_batch_nupts_driven(int batch_size) {
  dim3 threads_per_block;
  dim3 num_blocks;

  int kernel_width = this->spread_params_.kernel_width;
  int pirange = this->spread_params_.pirange;
  FloatType es_c = this->spread_params_.kernel_c;
  FloatType es_beta = this->spread_params_.kernel_beta;
  FloatType sigma = this->spread_params_.upsampling_factor;

  GpuComplex<FloatType>* d_c = this->c_;
  GpuComplex<FloatType>* d_fw = this->grid_data_;

  threads_per_block.x = 16;
  threads_per_block.y = 1;
  num_blocks.x = (this->num_points_ + threads_per_block.x - 1) /
                 threads_per_block.x;
  num_blocks.y = 1;

  switch (this->rank_) {
    case 2:
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadNuptsDriven2DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], d_c + t * this->grid_size_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], es_c, es_beta,
                this->idx_nupts_, pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadNuptsDrivenHorner2DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], sigma,
                this->idx_nupts_, pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    case 3:
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadNuptsDriven3DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], this->points_[2], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], this->grid_dims_[2],
                es_c, es_beta, this->idx_nupts_, pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadNuptsDrivenHorner3DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], this->points_[2], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], this->grid_dims_[2],
                sigma, this->idx_nupts_, pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    default:
      return errors::Unimplemented("Invalid rank: ", this->rank_);
  }

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::spread_batch_subproblem(int batch_size) {
  int kernel_width = this->spread_params_.kernel_width;
  FloatType es_c = this->spread_params_.kernel_c;
  FloatType es_beta = this->spread_params_.kernel_beta;
  int max_subprob_size = this->options_.gpu_max_subproblem_size;

  GpuComplex<FloatType>* d_c = this->c_;
  GpuComplex<FloatType>* d_fw = this->grid_data_;
  int pirange = this->spread_params_.pirange;

  FloatType sigma = this->options_.upsampling_factor;

  // GPU kernel configuration.
  int num_blocks = this->subprob_count_;
  int threads_per_block = 256;
  size_t shared_memory_size = sizeof(GpuComplex<FloatType>);
  for (int i = 0; i < this->rank_; i++) {
    shared_memory_size *= (this->bin_dims_[i] + 2 * ((kernel_width + 1) / 2));
  }
  if (shared_memory_size > this->device_.sharedMemPerBlock()) {
    return errors::ResourceExhausted(
        "Insuficient shared memory for GPU kernel. Need ", shared_memory_size,
        " bytes, but only ", this->device_.sharedMemPerBlock(), " bytes are "
        "available on the device.");
  }

  switch (this->rank_) {
    case 2:
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadSubproblem2DKernel<FloatType>, num_blocks,
                threads_per_block, shared_memory_size, this->device_.stream(),
                this->points_[0], this->points_[1], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], es_c, es_beta, sigma,
                this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
                this->bin_dims_[1], this->subprob_bins_,
                this->subprob_start_pts_, this->num_subprob_, max_subprob_size,
                this->num_bins_[0], this->num_bins_[1], this->idx_nupts_,
                pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadSubproblemHorner2DKernel<FloatType>, num_blocks,
                threads_per_block, shared_memory_size, this->device_.stream(),
                this->points_[0], this->points_[1], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], sigma,
                this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
                this->bin_dims_[1], this->subprob_bins_,
                this->subprob_start_pts_, this->num_subprob_, max_subprob_size,
                this->num_bins_[0], this->num_bins_[1], this->idx_nupts_,
                pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    case 3:
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadSubproblem3DKernel<FloatType>, num_blocks,
                threads_per_block, shared_memory_size, this->device_.stream(),
                this->points_[0], this->points_[1], this->points_[2],
                d_c + t * this->num_points_, d_fw + t * this->grid_size_,
                this->num_points_, kernel_width, this->grid_dims_[0],
                this->grid_dims_[1], this->grid_dims_[2], es_c, es_beta,
                this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
                this->bin_dims_[1], this->bin_dims_[2], this->subprob_bins_,
                this->subprob_start_pts_, this->num_subprob_, max_subprob_size,
                this->num_bins_[0], this->num_bins_[1], this->num_bins_[2],
                this->idx_nupts_, pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                SpreadSubproblemHorner3DKernel<FloatType>, num_blocks,
                threads_per_block, shared_memory_size, this->device_.stream(),
                this->points_[0], this->points_[1], this->points_[2],
                d_c + t * this->num_points_, d_fw + t * this->grid_size_,
                this->num_points_, kernel_width, this->grid_dims_[0],
                this->grid_dims_[1], this->grid_dims_[2], sigma,
                this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
                this->bin_dims_[1], this->bin_dims_[2], this->subprob_bins_,
                this->subprob_start_pts_, this->num_subprob_, max_subprob_size,
                this->num_bins_[0], this->num_bins_[1], this->num_bins_[2],
                this->idx_nupts_, pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    default:
      return errors::Unimplemented("Invalid rank: ", this->rank_);
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::interp_batch_nupts_driven(int batch_size) {
  dim3 threads_per_block;
  dim3 num_blocks;

  int kernel_width = this->spread_params_.kernel_width;
  FloatType es_c = this->spread_params_.kernel_c;
  FloatType es_beta = this->spread_params_.kernel_beta;
  FloatType sigma = this->options_.upsampling_factor;
  int pirange = this->spread_params_.pirange;

  GpuComplex<FloatType>* d_c = this->c_;
  GpuComplex<FloatType>* d_fw = this->grid_data_;

  switch (this->rank_) {
    case 2:
      threads_per_block.x = 32;
      threads_per_block.y = 1;
      num_blocks.x = (this->num_points_ + threads_per_block.x - 1) /
                     threads_per_block.x;
      num_blocks.y = 1;
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                InterpNuptsDriven2DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], es_c, es_beta,
                this->idx_nupts_, pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                InterpNuptsDrivenHorner2DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], sigma,
                this->idx_nupts_, pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    case 3:
      threads_per_block.x = 16;
      threads_per_block.y = 1;
      num_blocks.x = (this->num_points_ + threads_per_block.x - 1) /
                     threads_per_block.x;
      num_blocks.y = 1;
      switch (this->options_.kernel_evaluation_method) {
        case KernelEvaluationMethod::DIRECT:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                InterpNuptsDriven3DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], this->points_[2], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], this->grid_dims_[2],
                es_c, es_beta, this->idx_nupts_, pirange));
          }
          break;
        case KernelEvaluationMethod::HORNER:
          for (int t = 0; t < batch_size; t++) {
            TF_CHECK_OK(GpuLaunchKernel(
                InterpNuptsDrivenHorner3DKernel<FloatType>, num_blocks,
                threads_per_block, 0, this->device_.stream(), this->points_[0],
                this->points_[1], this->points_[2], d_c + t * this->num_points_,
                d_fw + t * this->grid_size_, this->num_points_, kernel_width,
                this->grid_dims_[0], this->grid_dims_[1], this->grid_dims_[2],
                sigma, this->idx_nupts_, pirange));
          }
          break;
        default:
          return errors::Internal(
              "Invalid kernel evaluation method: ", static_cast<int>(
                  this->options_.kernel_evaluation_method));
      }
      break;
    default:
      return errors::Unimplemented("Invalid rank: ", this->rank_);
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::interp_batch_subproblem(int batch_size) {
    int kernel_width = this->spread_params_.kernel_width;
    FloatType es_c = this->spread_params_.kernel_c;
    FloatType es_beta = this->spread_params_.kernel_beta;
    int max_subprob_size = this->options_.gpu_max_subproblem_size;

  GpuComplex<FloatType>* d_c = this->c_;
  GpuComplex<FloatType>* d_fw = this->grid_data_;

  int subprob_count = this->subprob_count_;
  int pirange = this->spread_params_.pirange;

  FloatType sigma = this->options_.upsampling_factor;

  // GPU kernel configuration.
  int num_blocks = subprob_count;
  int threads_per_block = 256;
  size_t shared_memory_size = sizeof(GpuComplex<FloatType>);
  for (int i = 0; i < this->rank_; i++) {
    shared_memory_size *= (this->bin_dims_[i] + 2 * ((kernel_width + 1) / 2));
  }
  if (shared_memory_size > this->device_.sharedMemPerBlock()) {
    return errors::ResourceExhausted(
        "Insuficient shared memory for GPU kernel. Need ", shared_memory_size,
        " bytes, but only ", this->device_.sharedMemPerBlock(), " bytes are "
        "available on the device.");
  }

  switch (this->rank_) {
    case 2:
      if (this->options_.kernel_evaluation_method ==
          KernelEvaluationMethod::HORNER) {
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              InterpSubproblemHorner2DKernel<FloatType>, num_blocks,
              threads_per_block, shared_memory_size, this->device_.stream(),
              this->points_[0], this->points_[1], d_c + t * this->num_points_,
              d_fw + t * this->grid_size_, this->num_points_, kernel_width,
              this->grid_dims_[0], this->grid_dims_[1], sigma,
              this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
              this->bin_dims_[1], this->subprob_bins_, this->subprob_start_pts_,
              this->num_subprob_, max_subprob_size, this->num_bins_[0],
              this->num_bins_[1], this->idx_nupts_, pirange));
        }
      } else {
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              InterpSubproblem2DKernel<FloatType>, num_blocks,
              threads_per_block, shared_memory_size, this->device_.stream(),
              this->points_[0], this->points_[1], d_c + t * this->num_points_,
              d_fw + t * this->grid_size_, this->num_points_, kernel_width,
              this->grid_dims_[0], this->grid_dims_[1], es_c, es_beta, sigma,
              this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
              this->bin_dims_[1], this->subprob_bins_, this->subprob_start_pts_,
              this->num_subprob_, max_subprob_size, this->num_bins_[0],
              this->num_bins_[1], this->idx_nupts_, pirange));
        }
      }
      break;
    case 3:
      for (int t = 0; t < batch_size; t++) {
        if (this->options_.kernel_evaluation_method ==
            KernelEvaluationMethod::HORNER) {
          TF_CHECK_OK(GpuLaunchKernel(
              InterpSubproblemHorner3DKernel<FloatType>, num_blocks,
              threads_per_block, shared_memory_size, this->device_.stream(),
              this->points_[0], this->points_[1], this->points_[2],
              d_c + t * this->num_points_, d_fw + t * this->grid_size_,
              this->num_points_, kernel_width, this->grid_dims_[0],
              this->grid_dims_[1], this->grid_dims_[2], sigma,
              this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
              this->bin_dims_[1], this->bin_dims_[2], this->subprob_bins_,
              this->subprob_start_pts_, this->num_subprob_,
              max_subprob_size, this->num_bins_[0], this->num_bins_[1],
              this->num_bins_[2], this->idx_nupts_, pirange));
        } else {
          TF_CHECK_OK(GpuLaunchKernel(
              InterpSubproblem3DKernel<FloatType>, num_blocks,
              threads_per_block, shared_memory_size, this->device_.stream(),
              this->points_[0], this->points_[1], this->points_[2],
              d_c + t * this->num_points_, d_fw + t * this->grid_size_,
              this->num_points_, kernel_width, this->grid_dims_[0],
              this->grid_dims_[1], this->grid_dims_[2], es_c, es_beta,
              this->bin_start_pts_, this->bin_sizes_, this->bin_dims_[0],
              this->bin_dims_[1], this->bin_dims_[2], this->subprob_bins_,
              this->subprob_start_pts_, this->num_subprob_, max_subprob_size,
              this->num_bins_[0], this->num_bins_[1], this->num_bins_[2],
              this->idx_nupts_, pirange));
        }
      }
      break;
  }

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::deconvolve_batch(int batch_size) {
  int threads_per_block = 256;
  int num_blocks = (this->mode_count_ + threads_per_block - 1) /
                   threads_per_block;

  if (this->spread_params_.spread_direction == SpreadDirection::SPREAD) {
    switch (this->rank_) {
      case 2:
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              Deconvolve2DKernel<FloatType>, num_blocks, threads_per_block, 0,
              this->device_.stream(), this->num_modes_[0], this->num_modes_[1],
              this->grid_dims_[0], this->grid_dims_[1],
              this->grid_data_ + t * this->grid_size_,
              this->f_ + t * this->mode_count_, this->fseries_data_[0],
              this->fseries_data_[1]));
        }
        break;
      case 3:
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              Deconvolve3DKernel<FloatType>, num_blocks, threads_per_block, 0,
              this->device_.stream(), this->num_modes_[0], this->num_modes_[1],
              this->num_modes_[2], this->grid_dims_[0], this->grid_dims_[1],
              this->grid_dims_[2], this->grid_data_ + t * this->grid_size_,
              this->f_ + t * this->mode_count_, this->fseries_data_[0],
              this->fseries_data_[1], this->fseries_data_[2]));
        }
        break;
    }
  } else {
    // Set fine grid to zero.
    size_t grid_bytes = sizeof(GpuComplex<FloatType>) * this->grid_size_ *
                        this->options_.max_batch_size;
    this->device_.memset(this->grid_data_, 0, grid_bytes);
    switch (this->rank_) {
      case 2:
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              Amplify2DKernel<FloatType>, num_blocks, threads_per_block, 0,
              this->device_.stream(), this->num_modes_[0], this->num_modes_[1],
              this->grid_dims_[0], this->grid_dims_[1],
              this->grid_data_ + t * this->grid_size_,
              this->f_ + t * this->mode_count_, this->fseries_data_[0],
              this->fseries_data_[1]));
        }
        break;
      case 3:
        for (int t = 0; t < batch_size; t++) {
          TF_CHECK_OK(GpuLaunchKernel(
              Amplify3DKernel<FloatType>, num_blocks, threads_per_block, 0,
              this->device_.stream(), this->num_modes_[0], this->num_modes_[1],
              this->num_modes_[2], this->grid_dims_[0], this->grid_dims_[1],
              this->grid_dims_[2], this->grid_data_ + t * this->grid_size_,
              this->f_ + t * this->mode_count_, this->fseries_data_[0],
              this->fseries_data_[1], this->fseries_data_[2]));
        }
        break;
    }
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::init_spreader() {
  switch (this->options_.spread_method) {
    case SpreadMethod::NUPTS_DRIVEN:
      TF_RETURN_IF_ERROR(this->init_spreader_nupts_driven());
      break;
    case SpreadMethod::SUBPROBLEM:
      TF_RETURN_IF_ERROR(this->init_spreader_subproblem());
      break;
    case SpreadMethod::PAUL:
    case SpreadMethod::BLOCK_GATHER:
      return errors::Unimplemented("Invalid spread method");
  }
  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::init_spreader_nupts_driven() {
  int num_blocks = (this->num_points_ + 1024 - 1) / 1024;
  int threads_per_block = 1024;

  if (this->spread_params_.sort_points == SortPoints::YES) {
    // This may not be necessary.
    this->device_.synchronize();

    // Calculate bin sizes.
    this->device_.memset(this->bin_sizes_, 0, this->bin_count_ * sizeof(int));
    switch (this->rank_) {
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcBinSizeNoGhost2DKernel<FloatType>,
            num_blocks, threads_per_block, 0, this->device_.stream(),
            this->num_points_, this->grid_dims_[0], this->grid_dims_[1],
            this->bin_dims_[0], this->bin_dims_[1], this->num_bins_[0],
            this->num_bins_[1], this->bin_sizes_, this->points_[0],
            this->points_[1], this->sort_idx_, this->spread_params_.pirange));
        break;
      case 3:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcBinSizeNoGhost3DKernel<FloatType>,
            num_blocks, threads_per_block, 0, this->device_.stream(),
            this->num_points_, this->grid_dims_[0], this->grid_dims_[1],
            this->grid_dims_[2], this->bin_dims_[0], this->bin_dims_[1],
            this->bin_dims_[2], this->num_bins_[0], this->num_bins_[1],
            this->num_bins_[2], this->bin_sizes_, this->points_[0],
            this->points_[1], this->points_[2], this->sort_idx_,
            this->spread_params_.pirange));
        break;
      default:
        return errors::Unimplemented("Invalid rank: ", this->rank_);
    }

    thrust::device_ptr<int> d_bin_sizes(this->bin_sizes_);
    thrust::device_ptr<int> d_bin_start_points(this->bin_start_pts_);
    thrust::exclusive_scan(thrust::cuda::par.on(this->device_.stream()),
                           d_bin_sizes, d_bin_sizes + this->bin_count_,
                           d_bin_start_points);

    switch (this->rank_) {
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcInvertofGlobalSortIdx2DKernel<FloatType>,
            num_blocks, threads_per_block, 0, this->device_.stream(),
            this->num_points_, this->bin_dims_[0], this->bin_dims_[1],
            this->num_bins_[0], this->num_bins_[1], this->bin_start_pts_,
            this->sort_idx_, this->points_[0], this->points_[1],
            this->idx_nupts_, this->spread_params_.pirange,
            this->grid_dims_[0], this->grid_dims_[1]));
        break;
      case 3:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcInvertofGlobalSortIdx3DKernel<FloatType>, num_blocks,
            threads_per_block, 0, this->device_.stream(), this->num_points_,
            this->bin_dims_[0], this->bin_dims_[1], this->bin_dims_[2],
            this->num_bins_[0], this->num_bins_[1], this->num_bins_[2],
            this->bin_start_pts_, this->sort_idx_, this->points_[0],
            this->points_[1], this->points_[2], this->idx_nupts_,
            this->spread_params_.pirange, this->grid_dims_[0],
            this->grid_dims_[1], this->grid_dims_[2]));
        break;
      default:
        return errors::Unimplemented("Invalid rank: ", this->rank_);
    }
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        TrivialGlobalSortIdxKernel,
        num_blocks, threads_per_block, 0, this->device_.stream(),
        this->num_points_, this->idx_nupts_));
  }

  return Status::OK();
}

template<typename FloatType>
Status Plan<GPUDevice, FloatType>::init_spreader_subproblem() {
  int num_blocks = (this->num_points_ + 1024 - 1) / 1024;
  int threads_per_block = 1024;

  int max_subprob_size = this->options_.gpu_max_subproblem_size;

  int pirange = this->spread_params_.pirange;

  // This may not be necessary.
  this->device_.synchronize();

  // Calculate bin sizes.
  this->device_.memset(this->bin_sizes_, 0, this->bin_count_ * sizeof(int));
  switch (this->rank_) {
    case 2:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcBinSizeNoGhost2DKernel<FloatType>,
          num_blocks, threads_per_block, 0, this->device_.stream(),
          this->num_points_, this->grid_dims_[0],
          this->grid_dims_[1], this->bin_dims_[0], this->bin_dims_[1],
          this->num_bins_[0], this->num_bins_[1], this->bin_sizes_,
          this->points_[0], this->points_[1], this->sort_idx_, pirange));
      break;
    case 3:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcBinSizeNoGhost3DKernel<FloatType>,
          num_blocks, threads_per_block, 0, this->device_.stream(),
          this->num_points_, this->grid_dims_[0], this->grid_dims_[1],
          this->grid_dims_[2], this->bin_dims_[0],
          this->bin_dims_[1], this->bin_dims_[2], this->num_bins_[0],
          this->num_bins_[1], this->num_bins_[2], this->bin_sizes_,
          this->points_[0], this->points_[1], this->points_[2],
          this->sort_idx_, pirange));
      break;
    default:
      return errors::Unimplemented("Invalid rank: ", this->rank_);
  }

  thrust::device_ptr<int> d_bin_sizes(this->bin_sizes_);
  thrust::device_ptr<int> d_bin_start_pts(this->bin_start_pts_);
  thrust::exclusive_scan(thrust::cuda::par.on(this->device_.stream()),
                         d_bin_sizes, d_bin_sizes + this->bin_count_,
                         d_bin_start_pts);

  switch (this->rank_) {
    case 2:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcInvertofGlobalSortIdx2DKernel<FloatType>, num_blocks,
          threads_per_block, 0, this->device_.stream(), this->num_points_,
          this->bin_dims_[0], this->bin_dims_[1], this->num_bins_[0],
          this->num_bins_[1], this->bin_start_pts_, this->sort_idx_,
          this->points_[0], this->points_[1], this->idx_nupts_, pirange,
          this->grid_dims_[0], this->grid_dims_[1]));
      break;
    case 3:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcInvertofGlobalSortIdx3DKernel<FloatType>, num_blocks,
          threads_per_block, 0, this->device_.stream(), this->num_points_,
          this->bin_dims_[0], this->bin_dims_[1], this->bin_dims_[2],
          this->num_bins_[0], this->num_bins_[1], this->num_bins_[2],
          this->bin_start_pts_, this->sort_idx_, this->points_[0],
          this->points_[1], this->points_[2], this->idx_nupts_, pirange,
          this->grid_dims_[0], this->grid_dims_[1], this->grid_dims_[2]));
      break;
    default:
      return errors::Unimplemented("Invalid rank: ", this->rank_);
  }

  TF_CHECK_OK(GpuLaunchKernel(
      CalcSubproblemKernel, num_blocks, threads_per_block, 0,
      this->device_.stream(), this->bin_sizes_, this->num_subprob_,
      max_subprob_size, this->bin_count_));

  thrust::device_ptr<int> d_subprob_sizes(this->num_subprob_);
  thrust::device_ptr<int> d_subprob_start_pts(this->subprob_start_pts_ + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(this->device_.stream()),
                         d_subprob_sizes, d_subprob_sizes + this->bin_count_,
                         d_subprob_start_pts);
  this->device_.memset(this->subprob_start_pts_, 0, sizeof(int));

  this->device_.memcpyDeviceToHost(&this->subprob_count_,
                                   &this->subprob_start_pts_[this->bin_count_],
                                   sizeof(int));

  // Maybe deallocate before allocating, as this function could be called more
  // than once during the lifetime of the plan.
  if (this->subprob_bins_ != nullptr)
    this->device_.deallocate(this->subprob_bins_);
  size_t subprob_bytes = this->subprob_count_ * sizeof(int);
  this->subprob_bins_ = reinterpret_cast<int*>(
      this->device_.allocate(subprob_bytes));

  num_blocks = (this->bin_count_ + 1024 - 1) / 1024;
  threads_per_block = 1024;

  TF_CHECK_OK(GpuLaunchKernel(
      MapBinToSubproblemKernel, num_blocks, threads_per_block, 0,
      this->device_.stream(), this->subprob_bins_, this->subprob_start_pts_,
      this->num_subprob_, this->bin_count_));

  return Status::OK();
}

namespace {
// Initializes spreader kernel parameters given desired NUFFT tol eps,
// upsampling factor ( = sigma in paper, or R in Dutt - Rokhlin), and ker eval
// meth
// Also sets all default options in SpreadParameters<FloatType>.
// Must call before any kernel evals done.
template<typename FloatType>
Status setup_spreader(int rank, FloatType eps, double upsampling_factor,
                      KernelEvaluationMethod kernel_evaluation_method,
                      SpreadParameters<FloatType>& spread_params) {
  if (upsampling_factor != 2.0) {
    if (kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
      return errors::Internal(
          "Horner kernel evaluation only supports the standard "
          "upsampling factor of 2.0, but got ", upsampling_factor);
    }
    if (upsampling_factor <= 1.0) {
      return errors::Internal(
          "upsampling_factor must be > 1.0, but is ", upsampling_factor);
    }
  }

  // defaults... (user can change after this function called)
  spread_params.spread_direction = SpreadDirection::SPREAD;
  spread_params.pirange = 1;             // user also should always set this
  spread_params.upsampling_factor = upsampling_factor;

  // as in FINUFFT v2.0, allow too - small - eps by truncating to eps_mach...
  if (eps < kEpsilon<FloatType>) {
    eps = kEpsilon<FloatType>;
  }

  // Set kernel width w (aka ns) and ES kernel beta parameter, in spread_params.
  int ns = std::ceil(-log10(eps / (FloatType)10.0));  // 1 digit per power of 10
  if (upsampling_factor != 2.0)           // override ns for custom sigma
    ns = std::ceil(-log(eps) / (kPi<FloatType> * sqrt(
        1.0 - 1.0 / upsampling_factor)));  // formula, gamma = 1
  ns = max(2, ns);               // we don't have ns = 1 version yet
  if (ns > kMaxKernelWidth) {         // clip to match allocated arrays
    ns = kMaxKernelWidth;
  }
  spread_params.kernel_width = ns;

  // Values to simplify kernel evaluation.
  spread_params.kernel_half_width = static_cast<FloatType>(ns) / 2;
  spread_params.kernel_c = 4.0 / static_cast<FloatType>(ns * ns);

  // Set the kernel beta parameter. The following results in reasonable beta
  // values for upsampling factor of 2.0, with some tweaks for small width
  // kernels.
  FloatType beta_over_ns = 2.30;
  if (ns == 2) beta_over_ns = 2.20;
  if (ns == 3) beta_over_ns = 2.26;
  if (ns == 4) beta_over_ns = 2.38;
  // Override beta for non - default oversampling factors.
  if (upsampling_factor != 2.0) {
    FloatType gamma = 0.97;  // This value must match the one in generated code.
    beta_over_ns = gamma * kPi<FloatType> * (1 - 1 / (2 * upsampling_factor));
  }
  spread_params.kernel_beta = beta_over_ns * static_cast<FloatType>(ns);

  if (spread_params.spread_only)
    spread_params.kernel_scale = calculate_scale_factor(rank, spread_params);

  return Status::OK();
}

// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader.  Barnett 10 / 30 / 17
template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadParameters<FloatType>& spread_params) {
  TF_RETURN_IF_ERROR(setup_spreader(
      rank, eps, options.upsampling_factor,
      options.kernel_evaluation_method, spread_params));

  spread_params.sort_points = options.sort_points;
  spread_params.spread_method = options.spread_method;
  spread_params.gpu_bin_size = options.gpu_bin_size;
  spread_params.gpu_obin_size = options.gpu_obin_size;
  spread_params.pirange = 1;
  spread_params.num_threads = options.num_threads;

  return Status::OK();
}

void set_bin_sizes(TransformType type, int rank, Options& options) {
  switch (rank) {
    case 2:
      options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 32 :
          options.gpu_bin_size.x;
      options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 32 :
          options.gpu_bin_size.y;
      options.gpu_bin_size.z = 1;
      break;
    case 3:
      switch (options.spread_method) {
        case SpreadMethod::NUPTS_DRIVEN:
        case SpreadMethod::SUBPROBLEM:
          options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 16 :
              options.gpu_bin_size.x;
          options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 16 :
              options.gpu_bin_size.y;
          options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 2 :
              options.gpu_bin_size.z;
          break;
        case SpreadMethod::BLOCK_GATHER:
          options.gpu_obin_size.x = (options.gpu_obin_size.x == 0) ? 8 :
              options.gpu_obin_size.x;
          options.gpu_obin_size.y = (options.gpu_obin_size.y == 0) ? 8 :
              options.gpu_obin_size.y;
          options.gpu_obin_size.z = (options.gpu_obin_size.z == 0) ? 8 :
              options.gpu_obin_size.z;
          options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 4 :
              options.gpu_bin_size.x;
          options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 4 :
              options.gpu_bin_size.y;
          options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 4 :
              options.gpu_bin_size.z;
          break;
      }
      break;
  }
}

template<typename FloatType>
Status set_grid_size(int ms,
                     int bin_size,
                     const Options& options,
                     const SpreadParameters<FloatType>& spread_params,
                     int* grid_size) {
  // For spread / interp only, we do not apply oversampling.
  if (options.spread_only) {
    *grid_size = ms;
  } else {
    *grid_size = static_cast<int>(options.upsampling_factor * ms);
  }

  // This is required to avoid errors.
  if (*grid_size < 2 * spread_params.kernel_width)
    *grid_size = 2 * spread_params.kernel_width;

  // Check if array size is too big.
  if (*grid_size > kMaxArraySize) {
    return errors::Internal(
        "Upsampled dim size too big: ", *grid_size, " > ", kMaxArraySize);
  }

  // Find the next smooth integer.
  if (options.spread_method == SpreadMethod::BLOCK_GATHER)
    *grid_size = next_smooth_int(*grid_size, bin_size);
  else
    *grid_size = next_smooth_int(*grid_size);

  // For spread / interp only mode, make sure that the grid size is valid.
  if (options.spread_only && *grid_size != ms) {
    return errors::Internal(
        "Invalid grid size: ", ms, ". Value should be even, "
        "larger than the kernel (", 2 * spread_params.kernel_width,
        ") and have no prime factors larger than 5.");
  }

  return Status::OK();
}

}  // namespace

template class Plan<GPUDevice, float>;
template class Plan<GPUDevice, double>;

}  // namespace nufft
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
