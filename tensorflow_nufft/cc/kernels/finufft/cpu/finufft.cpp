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

#define EIGEN_USE_THREADS

#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>

#include "tensorflow_nufft/cc/kernels/finufft/cpu/finufft_eitherprec.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/finufft_definitions.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/dataTypes.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/utils.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/utils_precindep.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/spreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/cpu/fftw_definitions.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"
#include "tensorflow_nufft/cc/kernels/nufft_util.h"

extern "C" {
  #include "tensorflow_nufft/cc/kernels/finufft/cpu/contrib/legendre_rule_fast.h"
}


using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


/* Computational core for FINUFFT.

   Based on Barnett 2017-2018 finufft?d.cpp containing nine drivers, plus
   2d1/2d2 many-vector drivers by Melody Shih, summer 2018.
   Original guru interface written by Andrea Malleo, summer 2019, mentored
   by Alex Barnett. Many rewrites in early 2020 by Alex Barnett & Libin Lu.

   As of v1.2 these replace the old hand-coded separate 9 finufft?d?() functions
   and the two finufft2d?many() functions. The (now 18) simple C++ interfaces
   are in simpleinterfaces.cpp.

Algorithm summaries taken from old finufft?d?() documentation, Feb-Jun 2017:

   TYPE 1:
     The type 1 NUFFT proceeds in three main steps:
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the kernel
        Fourier series coeffs (not merely FFT of kernel), shuffle to output.
     The kernel coeffs are precomputed in what is called step 0 in the code.
   Written with FFTW style complex arrays. Step 3a internally uses CPX,
   and Step 3b internally uses real arithmetic and FFTW style complex.

   TYPE 2:
     The type 2 algorithm proceeds in three main steps:
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.
   Written with FFTW style complex arrays. Step 0 internally uses CPX,
   and Step 1 internally uses real arithmetic and FFTW style complex.

   TYPE 3:
     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1.
     Beyond this, the new twists are:
     i) nf1, number of upsampled points for the type-1, depends on the product
       of interval widths containing input and output points (X*S).
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf1.
   No references to FFTW are needed here. CPX arithmetic is used.

   MULTIPLE STRENGTH VECTORS FOR THE SAME NONUNIFORM POINTS (n_transf>1):
     maxBatchSize (set to max_num_omp_threads) times the RAM is needed, so
     this is good only for small problems.


Design notes for guru interface implementation:

* Since finufft_plan is C-compatible, we need to use malloc/free for its
  allocatable arrays, keeping it quite low-level. We can't use std::vector
  since that would only survive in the scope of each function.

* Thread-safety: FINUFFT plans are passed as pointers, so it has no global
  state apart from that associated with FFTW (and the is_fftw_initialized).
*/



// ---------- local math routines (were in common.cpp; no need now): --------

// We macro because it has no FLT args but gets compiled for both prec's...
#ifdef SINGLE
#define SET_NF_TYPE12 set_nf_type12f
#else
#define SET_NF_TYPE12 set_nf_type12
#endif
int SET_NF_TYPE12(BIGINT ms, const Options& options,
                  SpreadOptions<FLT> spopts, BIGINT *nf)
// Type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms. Returns 0 if success, else an
// error code if nf was unreasonably big (& tell the world).
{
  // for spread/interp only, we do not apply oversampling (Montalt 6/8/2021).
  if (options.spread_interp_only) {
    *nf = ms;
  } else {
    *nf = (BIGINT)(options.upsampling_factor * ms);       // manner of rounding not crucial
  }
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread; // otherwise spread fails
  if (*nf<MAX_NF) {
    *nf = next_smooth_int(*nf);                       // expensive at huge nf
  } else {
    fprintf(stderr,"[%s] nf=%.3g exceeds MAX_NF of %.3g, so exit without attempting even a malloc\n",__func__,(double)*nf,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  // for spread/interp only, make sure that the grid shape is valid
  // (Montalt 6/8/2021).
  if (options.spread_interp_only && *nf != ms) {
    fprintf(stderr,"[%s] ms=%d is not a valid grid size. It should be even, larger than the kernel (%d) and have no prime factors larger than 5.\n",__func__,ms,2*spopts.nspread);
    return ERR_GRIDSIZE_NOTVALID;
  }
  return 0;
}

int setup_spreader_for_nufft(const Options& options,
                             SpreadOptions<FLT> &spopts, FLT eps, int rank)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Return status of setup_spreader. Uses pass-by-ref. Barnett 10/30/17
{
  // this must be set before calling setup_spreader
  spopts.spread_interp_only = options.spread_interp_only;
  // this calls spreadinterp.cpp...
  int ier = setup_spreader(spopts, eps, options.upsampling_factor,
                           static_cast<int>(options.kernel_evaluation_method) - 1, // We subtract 1 temporarily, as spreader expects values of 0 or 1 instead of 1 and 2.
                           options.verbosity, options.show_warnings, rank);
  // override various spread opts from their defaults...
  spopts.verbosity = options.verbosity;
  spopts.sort = static_cast<int>(options.sort_points); // could make rank or CPU choices here?
  spopts.pad_kernel = options.pad_kernel; // (only applies to kerevalmeth=0)
  spopts.check_bounds = options.check_bounds;
  spopts.num_threads = options.num_threads;
  if (options.num_threads_for_atomic_spread >= 0) // overrides
    spopts.atomic_threshold = options.num_threads_for_atomic_spread;
  if (options.max_spread_subproblem_size > 0)        // overrides
    spopts.max_subproblem_size = options.max_spread_subproblem_size;
  return ier;
} 

void set_nhg_type3(FLT S, FLT X,
                   const Options& options, SpreadOptions<FLT> spopts,
		               BIGINT *nf, FLT *h, FLT *gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss = spopts.nspread + 1;      // since ns may be odd
  FLT Xsafe=X, Ssafe=S;              // may be tweaked locally
  if (X==0.0)                        // logic ensures XS>=1, handle X=0 a/o S=0
    if (S==0.0) {
      Xsafe=1.0;
      Ssafe=1.0;
    } else Xsafe = max(Xsafe, 1/S);
  else
    Ssafe = max(Ssafe, 1/X);
  // use the safe X and S...
  FLT nfd = 2.0*options.upsampling_factor*Ssafe*Xsafe/PI + nss;
  if (!isfinite(nfd)) nfd=0.0;                // use FLT to catch inf
  *nf = (BIGINT)nfd;
  //printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread;
  if (*nf<MAX_NF)                             // otherwise will fail anyway
    *nf = next_smooth_int(*nf);                   // expensive at huge nf
  *h = 2*PI / *nf;                            // upsampled grid spacing
  *gam = (FLT)*nf / (2.0*options.upsampling_factor*Ssafe);  // x scale fac to x'
}

void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, SpreadOptions<FLT> opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 FLTs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18.
  Fixed num_threads 7/20/20
 */
{
  FLT J2 = opts.nspread/2.0;            // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 3.0*J2);  // not sure why so large? cannot exceed MAX_NQUAD
  FLT f[MAX_NQUAD];
  double z[2*MAX_NQUAD], w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  std::complex<FLT> a[MAX_NQUAD];
  for (int n=0;n<q;++n) {               // set up nodes z_n and vals f_n
    z[n] *= J2;                         // rescale nodes
    f[n] = J2*(FLT)w[n] * evaluate_kernel((FLT)z[n], opts); // vals & quadr wei
    a[n] = exp(2*PI*IMA*(FLT)(nf/2-z[n])/(FLT)nf);  // phase winding rates
  }
  BIGINT nout=nf/2+1;                   // how many values we're writing to
  int nt = min(nout,(BIGINT)opts.num_threads);         // how many chunks
  std::vector<BIGINT> brk(nt+1);        // start indices for each thread
  for (int t=0; t<=nt; ++t)             // split nout mode indices btw threads
    brk[t] = (BIGINT)(0.5 + nout*t/(double)nt);
#pragma omp parallel num_threads(nt)
  {                                     // each thread gets own chunk to do
    int t = FINUFFT_GET_THREAD_NUM();
    std::complex<FLT> aj[MAX_NQUAD];    // phase rotator for this thread
    for (int n=0;n<q;++n)
      aj[n] = pow(a[n],(FLT)brk[t]);    // init phase factors for chunk
    for (BIGINT j=brk[t];j<brk[t+1];++j) {          // loop along output array
      FLT x = 0.0;                      // accumulator for answer at this j
      for (int n=0;n<q;++n) {
        x += f[n] * 2*real(aj[n]);      // include the negative freq
        aj[n] *= a[n];                  // wind the phases
      }
      fwkerhalf[j] = x;
    }
  }
}

void onedim_nuft_kernel(BIGINT nk, FLT *k, FLT *phihat, SpreadOptions<FLT> opts)
/*
  Approximates exact 1D Fourier transform of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi,pi],
  for a kernel with x measured in grid-spacings. (See previous routine for
  FT definition).

  Inputs:
  nk - number of freqs
  k - frequencies, dual to the kernel's natural argument, ie exp(i.k.z)
       Note, z is in grid-point units, and k values must be in [-pi,pi] for
       accuracy.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  phihat - real Fourier transform evaluated at freqs (alloc for nk FLTs)

  Barnett 2/8/17. openmp since cos slow 2/9/17
 */
{
  FLT J2 = opts.nspread/2.0;        // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 2.0*J2);     // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (opts.verbosity) printf("q (# ker FT quadr pts) = %d\n",q);
  FLT f[MAX_NQUAD]; double z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  for (int n=0;n<q;++n) {
    z[n] *= J2;                                    // quadr nodes for [0,J/2]
    f[n] = J2*(FLT)w[n] * evaluate_kernel((FLT)z[n], opts);  // w/ quadr weights
    //    printf("f[%d] = %.3g\n",n,f[n]);
  }
#pragma omp parallel for num_threads(opts.num_threads)
  for (BIGINT j=0;j<nk;++j) {          // loop along output array
    FLT x = 0.0;                    // register
    for (int n=0;n<q;++n) x += f[n] * 2*cos(k[j]*z[n]);  // pos & neg freq pair
    phihat[j] = x;
  }
}  

void deconvolveshuffle1d(SpreadDirection dir, FLT prefac,FLT* ker, BIGINT ms,
			 FLT *fk, BIGINT nf1, FFTW_CPX* fw, ModeOrder mode_order)
/*
  if dir == SpreadDirection::SPREAD: copies fw to fk with amplification by prefac/ker
  if dir == SpreadDirection::INTERP: copies fk to fw (and zero pads rest of it), same amplification.

  mode_order=0: use CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
          1: use FFT-style (from 0 to N/2-1, then -N/2 up to -1).

  fk is size-ms FLT complex array (2*ms FLTs alternating re,im parts)
  fw is a FFTW style complex array, ie FLT [nf1][2], essentially FLTs
       alternating re,im parts.
  ker is real-valued FLT array of length nf1/2+1.

  Single thread only, but shouldn't matter since mostly data movement.

  It has been tested that the repeated floating division in this inner loop
  only contributes at the <3% level in 3D relative to the fftw cost (8 threads).
  This could be removed by passing in an inverse kernel and doing mults.

  todo: rewrite w/ C++-complex I/O, check complex divide not slower than
        real divide, or is there a way to force a real divide?

  Barnett 1/25/17. Fixed ms=0 case 3/14/17. mode_order flag & clean 10/25/17
*/
{
  BIGINT kmin = -ms/2, kmax = (ms-1)/2;    // inclusive range of k indices
  if (ms==0) kmax=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*kmin, pn = 0;       // CMCL mode-ordering case (2* since cmplx)
  if (mode_order==ModeOrder::FFT) { pp = 0; pn = 2*(kmax+1); }   // or, instead, FFT ordering
  if (dir == SpreadDirection::SPREAD) {    // read fw, write out to fk...
    for (BIGINT k=0;k<=kmax;++k) {                    // non-neg freqs k
      fk[pp++] = prefac * fw[k][0] / ker[k];          // re
      fk[pp++] = prefac * fw[k][1] / ker[k];          // im
    }
    for (BIGINT k=kmin;k<0;++k) {                     // neg freqs k
      fk[pn++] = prefac * fw[nf1+k][0] / ker[-k];     // re
      fk[pn++] = prefac * fw[nf1+k][1] / ker[-k];     // im
    }
  } else {    // read fk, write out to fw w/ zero padding...
    for (BIGINT k=kmax+1; k<nf1+kmin; ++k) {  // zero pad precisely where needed
      fw[k][0] = fw[k][1] = 0.0; }
    for (BIGINT k=0;k<=kmax;++k) {                    // non-neg freqs k
      fw[k][0] = prefac * fk[pp++] / ker[k];          // re
      fw[k][1] = prefac * fk[pp++] / ker[k];          // im
    }
    for (BIGINT k=kmin;k<0;++k) {                     // neg freqs k
      fw[nf1+k][0] = prefac * fk[pn++] / ker[-k];     // re
      fw[nf1+k][1] = prefac * fk[pn++] / ker[-k];     // im
    }
  }
}

void deconvolveshuffle2d(SpreadDirection dir,FLT prefac,FLT *ker1, FLT *ker2,
			 BIGINT ms, BIGINT mt,
			 FLT *fk, BIGINT nf1, BIGINT nf2, FFTW_CPX* fw,
			 ModeOrder mode_order)
/*
  2D version of deconvolveshuffle1d, calls it on each x-line using 1/ker2 fac.

  if dir == SpreadDirection::SPREAD: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir == SpreadDirection::INTERP: copies fk to fw (and zero pads rest of it), same amplification.

  mode_order=0: use CMCL-compatible mode ordering in fk (each rank increasing)
          1: use FFT-style (pos then negative, on each rank)

  fk is complex array stored as 2*ms*mt FLTs alternating re,im parts, with
    ms looped over fast and mt slow.
  fw is a FFTW style complex array, ie FLT [nf1*nf2][2], essentially FLTs
       alternating re,im parts; again nf1 is fast and nf2 slow.
  ker1, ker2 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1
       respectively.

  Barnett 2/1/17, Fixed mt=0 case 3/14/17. mode_order 10/25/17
*/
{
  BIGINT k2min = -mt/2, k2max = (mt-1)/2;    // inclusive range of k2 indices
  if (mt==0) k2max=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*k2min*ms, pn = 0;   // CMCL mode-ordering case (2* since cmplx)
  if (mode_order == ModeOrder::FFT) { pp = 0; pn = 2*(k2max+1)*ms; }  // or, instead, FFT ordering
  if (dir == SpreadDirection::INTERP)               // zero pad needed x-lines (contiguous in memory)
    for (BIGINT j=nf1*(k2max+1); j<nf1*(nf2+k2min); ++j)  // sweeps all dims
      fw[j][0] = fw[j][1] = 0.0;
  for (BIGINT k2=0;k2<=k2max;++k2, pp+=2*ms)          // non-neg y-freqs
    // point fk and fw to the start of this y value's row (2* is for complex):
    deconvolveshuffle1d(dir,prefac/ker2[k2],ker1,ms,fk + pp,nf1,&fw[nf1*k2],mode_order);
  for (BIGINT k2=k2min;k2<0;++k2, pn+=2*ms)           // neg y-freqs
    deconvolveshuffle1d(dir,prefac/ker2[-k2],ker1,ms,fk + pn,nf1,&fw[nf1*(nf2+k2)],mode_order);
}

void deconvolveshuffle3d(SpreadDirection dir,FLT prefac,FLT *ker1, FLT *ker2,
			 FLT *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 FLT *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 FFTW_CPX* fw, ModeOrder mode_order)
/*
  3D version of deconvolveshuffle2d, calls it on each xy-plane using 1/ker3 fac.

  if dir == SpreadDirection::SPREAD: copies fw to fk with ampl by prefac/(ker1(k1)*ker2(k2)*ker3(k3)).
  if dir == SpreadDirection::INTERP: copies fk to fw (and zero pads rest of it), same amplification.

  mode_order=0: use CMCL-compatible mode ordering in fk (each rank increasing)
          1: use FFT-style (pos then negative, on each rank)

  fk is complex array stored as 2*ms*mt*mu FLTs alternating re,im parts, with
    ms looped over fastest and mu slowest.
  fw is a FFTW style complex array, ie FLT [nf1*nf2*nf3][2], effectively
       FLTs alternating re,im parts; again nf1 is fastest and nf3 slowest.
  ker1, ker2, ker3 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1,
       and nf3/2+1 respectively.

  Barnett 2/1/17, Fixed mu=0 case 3/14/17. mode_order 10/25/17
*/
{
  BIGINT k3min = -mu/2, k3max = (mu-1)/2;    // inclusive range of k3 indices
  if (mu==0) k3max=-1;           // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2*k3min*ms*mt, pn = 0; // CMCL mode-ordering (2* since cmplx)
  if (mode_order == ModeOrder::FFT) { pp = 0; pn = 2*(k3max+1)*ms*mt; }  // or FFT ordering
  BIGINT np = nf1*nf2;  // # pts in an upsampled Fourier xy-plane
  if (dir == SpreadDirection::INTERP)           // zero pad needed xy-planes (contiguous in memory)
    for (BIGINT j=np*(k3max+1);j<np*(nf3+k3min);++j)  // sweeps all dims
      fw[j][0] = fw[j][1] = 0.0;
  for (BIGINT k3=0;k3<=k3max;++k3, pp+=2*ms*mt)      // non-neg z-freqs
    // point fk and fw to the start of this z value's plane (2* is for complex):
    deconvolveshuffle2d(dir,prefac/ker3[k3],ker1,ker2,ms,mt,
			fk + pp,nf1,nf2,&fw[np*k3],mode_order);
  for (BIGINT k3=k3min;k3<0;++k3, pn+=2*ms*mt)       // neg z-freqs
    deconvolveshuffle2d(dir,prefac/ker3[-k3],ker1,ker2,ms,mt,
			fk + pn,nf1,nf2,&fw[np*(nf3+k3)],mode_order);
}


// --------- batch helper functions for t1,2 exec: ---------------------------

int spreadinterpSortedBatch(int batchSize, Plan<CPUDevice, FLT>* p, CPX* cBatch, CPX* fBatch=NULL)
/*
  Spreads (or interpolates) a batch of batchSize strength vectors in cBatch
  to (or from) the batch of fine working grids p->fwBatch, using the same set of
  (index-sorted) NU points p->X,Y,Z for each vector in the batch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  Returns 0 (no error reporting for now).
  Notes:
  1) cBatch is already assumed to have the correct offset, ie here we
     read from the start of cBatch (unlike Malleo). fwBatch also has zero offset
  2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  Barnett 5/19/20, based on Malleo 2019.
  3) the 4th parameter is used when doing interp/spread only. When received,
     input/output data is read/written from/to this pointer instead of from/to
     the internal array p->fWBatch. Montalt 5/8/2021
*/
{
  // opts.spreader_threading: 1 sequential multithread, 2 parallel single-thread.
  // omp_sets_nested deprecated, so don't use; assume not nested for 2 to work.
  // But when nthr_outer=1 here, omp par inside the loop sees all threads...
  int nthr_outer = p->options.spreader_threading == SpreaderThreading::SEQUENTIAL_MULTI_THREADED ? 1 : batchSize;

  if (fBatch == NULL) {
    fBatch = (CPX*) p->fwBatch;
  }

  #pragma omp parallel for num_threads(nthr_outer)
  for (int i=0; i<batchSize; i++) {
    CPX *fwi = fBatch + i*p->nf;  // start of i'th fw array in wkspace
    CPX *ci = cBatch + i*p->nj;            // start of i'th c array in cBatch
    spreadinterpSorted(p->sortIndices, p->nf1, p->nf2, p->nf3, (FLT*)fwi, p->nj,
                       p->X, p->Y, p->Z, (FLT*)ci, p->spopts, p->didSort);
  }
  return 0;
}

int deconvolveBatch(int batchSize, Plan<CPUDevice, FLT>* p, CPX* fkBatch)
/*
  Type 1: deconvolves (amplifies) from each interior fw array in p->fwBatch
  into each output array fk in fkBatch.
  Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  again looping over fk in fkBatch and fw in p->fwBatch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  This is mostly a loop calling deconvolveshuffle?d for the needed rank batchSize
  times.
  Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
*/
{
  // since deconvolveshuffle?d are single-thread, omp par seems to help here...
#pragma omp parallel for num_threads(batchSize)
  for (int i=0; i<batchSize; i++) {
    FFTW_CPX *fwi = p->fwBatch + i*p->nf;  // start of i'th fw array in wkspace
    CPX *fki = fkBatch + i*p->N;           // start of i'th fk array in fkBatch
    
    // Call routine from common.cpp for the rank; prefactors hardcoded to 1.0...
    if (p->rank_ == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, 1.0, p->phiHat1,
                          p->ms, (FLT *)fki,
                          p->nf1, fwi, p->options.mode_order);
    else if (p->rank_ == 2)
      deconvolveshuffle2d(p->spopts.spread_direction,1.0, p->phiHat1,
                          p->phiHat2, p->ms, p->mt, (FLT *)fki,
                          p->nf1, p->nf2, fwi, p->options.mode_order);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, 1.0, p->phiHat1,
                          p->phiHat2, p->phiHat3, p->ms, p->mt, p->mu,
                          (FLT *)fki, p->nf1, p->nf2, p->nf3,
                          fwi, p->options.mode_order);
  }
  return 0;
}


// since this func is local only, we macro its name here...
#ifdef SINGLE
#define GRIDSIZE_FOR_FFTW gridsize_for_fftwf
#else
#define GRIDSIZE_FOR_FFTW gridsize_for_fftw
#endif

int* GRIDSIZE_FOR_FFTW(Plan<CPUDevice, FLT>* p){
// local helper func returns a new int array of length rank, extracted from
// the finufft plan, that fftw_plan_many_dft needs as its 2nd argument.
  int* nf;
  if(p->rank_ == 1){ 
    nf = new int[1];
    nf[0] = (int)p->nf1;
  }
  else if (p->rank_ == 2){ 
    nf = new int[2];
    nf[0] = (int)p->nf2;
    nf[1] = (int)p->nf1; 
  }   // fftw enforced row major ordering, ie dims are backwards ordered
  else{ 
    nf = new int[3];
    nf[0] = (int)p->nf3;
    nf[1] = (int)p->nf2;
    nf[2] = (int)p->nf1;
  }
  return nf;
}




// --------------- rest is the 5 user guru (plan) interface drivers: -----------


// PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
// Populates the fields of finufft_plan which is pointed to by "p".
// opts is ptr to a nufft_opts to set options, or NULL to use defaults.
// For some of the fields, if "auto" selected, choose the actual setting.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and instantiates the fft_plan
int FINUFFT_MAKEPLAN(TransformType type, int rank, BIGINT* n_modes, int iflag,
                     int ntrans, FLT tol, Plan<CPUDevice, FLT> **plan,
                     const Options& options)
{
  cout << scientific << setprecision(15);  // for commented-out low-lev debug

  // Allocate fresh plan struct.
  Plan<CPUDevice, FLT>* p = new Plan<CPUDevice, FLT>;
  *plan = p;

  // Keep a deep copy of the options. Changing the input structure after this
  // has no effect.
  p->options = options;

  if ((rank != 1) && (rank != 2) && (rank != 3)) {
    fprintf(stderr, "[%s] Invalid rank (%d), should be 1, 2 or 3.\n",__func__,rank);
    return ERR_DIM_NOTVALID;
  }
  if (ntrans < 1) {
    fprintf(stderr,"[%s] ntrans (%d) should be at least 1.\n",__func__,ntrans);
    return ERR_NTRANS_NOTVALID;
  }
  
  // get stuff from args...
  p->type_ = type;
  p->rank_ = rank;
  p->ntrans = ntrans;
  p->tol = tol;
  p->fftSign = (iflag>=0) ? 1 : -1;         // clean up flag input

  // Choose kernel evaluation method.
  if (p->options.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
    p->options.kernel_evaluation_method = KernelEvaluationMethod::HORNER;
  }

  // Choose overall number of threads.
  int num_threads = FINUFFT_GET_MAX_THREADS(); // default value
  if (p->options.num_threads > 0)
    num_threads = p->options.num_threads; // user override
  p->options.num_threads = num_threads;   // update options with actual number

  // Select batch size.
  if (p->options.max_batch_size == 0) {
    p->nbatch = 1 + (ntrans - 1) / num_threads;
    p->batchSize = 1 + (ntrans - 1) / p->nbatch;
  } else {
    p->batchSize = min(p->options.max_batch_size, ntrans);
    p->nbatch = 1+(ntrans-1)/p->batchSize;
  }

  // Choose default spreader threading configuration.
  if (p->options.spreader_threading == SpreaderThreading::AUTO)
    p->options.spreader_threading = SpreaderThreading::PARALLEL_SINGLE_THREADED;
  if (p->options.spreader_threading != SpreaderThreading::SEQUENTIAL_MULTI_THREADED &&
      p->options.spreader_threading != SpreaderThreading::PARALLEL_SINGLE_THREADED) {
    fprintf(stderr,"[%s] illegal options.spreader_threading!\n",__func__);
    return ERR_SPREAD_THREAD_NOTVALID;
  }

  if (type != TransformType::TYPE_3) {    // read in user Fourier mode array sizes...
    p->ms = n_modes[0];
    p->mt = (rank>1) ? n_modes[1] : 1;       // leave as 1 for unused dims
    p->mu = (rank>2) ? n_modes[2] : 1;
    p->N = p->ms*p->mt*p->mu;               // N = total # modes
  }
  
  // Heuristic to choose default upsampling factor.
  if (p->options.upsampling_factor == 0.0) {  // indicates auto-choose
    p->options.upsampling_factor = 2.0;       // default, and need for tol small
    if (tol >= (FLT)1E-9) {                   // the tol sigma=5/4 can reach
      if (type == TransformType::TYPE_3)
        p->options.upsampling_factor = 1.25;  // faster b/c smaller RAM & FFT
      else if ((rank==1 && p->N>10000000) || (rank==2 && p->N>300000) || (rank==3 && p->N>3000000))  // type 1,2 heuristic cutoffs, double, typ tol, 12-core xeon
        p->options.upsampling_factor = 1.25;
    }
    if (p->options.verbosity > 1)
      printf("[%s] set auto upsampling_factor=%.2f\n",__func__,p->options.upsampling_factor);
  }

  // use opts to choose and write into plan's spread options...
  int ier = setup_spreader_for_nufft(p->options, p->spopts, tol, rank);
  if (ier>1)                                 // proceed if success or warning
    return ier;

  // set others as defaults (or unallocated for arrays)...
  p->X = NULL; p->Y = NULL; p->Z = NULL;
  p->phiHat1 = NULL; p->phiHat2 = NULL; p->phiHat3 = NULL;
  p->nf1 = 1; p->nf2 = 1; p->nf3 = 1;  // crucial to leave as 1 for unused dims
  p->sortIndices = NULL;               // used in all three types
  
  //  ------------------------ types 1,2: planning needed ---------------------
  if (type == TransformType::TYPE_1 || type == TransformType::TYPE_2) {

    // Give FFTW all threads (or use o.spreader_threading?).
    int fftw_threads = num_threads;

    // Note: batchSize not used since might be only 1.

    // Now place FFTW initialization in a lock, courtesy of OMP. Makes FINUFFT
    // thread-safe (can be called inside OMP) if -DFFTW_PLAN_SAFE used...
    #pragma omp critical
    {
      static bool is_fftw_initialized = 0; // the only global state of FINUFFT

      if (!is_fftw_initialized) {
        FFTW_INIT(); // setup FFTW global state; should only do once
        FFTW_PLAN_TH(fftw_threads); // ditto
        FFTW_PLAN_SF(); // if -DFFTW_PLAN_SAFE, make FFTW thread-safe
        is_fftw_initialized = 1;      // insure other FINUFFT threads don't clash
      }
    }

    if (type == TransformType::TYPE_1)
      p->spopts.spread_direction = SpreadDirection::SPREAD;
    else // if (type == TransformType::TYPE_2)
      p->spopts.spread_direction = SpreadDirection::INTERP;

    if (p->options.show_warnings) {  // user warn round-off error...
      if (EPSILON*p->ms>1.0)
        fprintf(stderr,"%s warning: rounding err predicted eps_mach*N1 = %.3g > 1 !\n",__func__,(double)(EPSILON*p->ms));
      if (EPSILON*p->mt>1.0)
        fprintf(stderr,"%s warning: rounding err predicted eps_mach*N2 = %.3g > 1 !\n",__func__,(double)(EPSILON*p->mt));
      if (EPSILON*p->mu>1.0)
        fprintf(stderr,"%s warning: rounding err predicted eps_mach*N3 = %.3g > 1 !\n",__func__,(double)(EPSILON*p->mu));
    }
    
    // determine fine grid sizes, sanity check..
    int nfier = SET_NF_TYPE12(p->ms, p->options, p->spopts, &(p->nf1));
    if (nfier) return nfier;    // nf too big; we're done
    p->phiHat1 = (FLT*)malloc(sizeof(FLT)*(p->nf1/2 + 1));
    if (rank > 1) {
      nfier = SET_NF_TYPE12(p->mt, p->options, p->spopts, &(p->nf2));
      if (nfier) return nfier;
      p->phiHat2 = (FLT*)malloc(sizeof(FLT)*(p->nf2/2 + 1));
    }
    if (rank > 2) {
      nfier = SET_NF_TYPE12(p->mu, p->options, p->spopts, &(p->nf3)); 
      if (nfier) return nfier;
      p->phiHat3 = (FLT*)malloc(sizeof(FLT)*(p->nf3/2 + 1));
    }

    if (p->options.verbosity) { // "long long" here is to avoid warnings with printf...
      printf("[%s] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld)\n               ntrans=%d num_threads=%d batchSize=%d ", __func__,
             rank, type, (long long)p->ms,(long long)p->mt,
             (long long) p->mu, (long long)p->nf1,(long long)p->nf2,
             (long long)p->nf3, ntrans, num_threads, p->batchSize);
      if (p->batchSize==1)          // spreader_threading has no effect in this case
        printf("\n");
      else
        printf(" spreader_threading=%d\n", p->options.spreader_threading);
    }

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid rank
    CNTime timer; timer.start();
    onedim_fseries_kernel(p->nf1, p->phiHat1, p->spopts);
    if (rank>1) onedim_fseries_kernel(p->nf2, p->phiHat2, p->spopts);
    if (rank>2) onedim_fseries_kernel(p->nf3, p->phiHat3, p->spopts);
    if (p->options.verbosity) printf("[%s] kernel fser (ns=%d):\t\t%.3g s\n",__func__,p->spopts.nspread, timer.elapsedsec());

    timer.restart();
    p->nf = p->nf1*p->nf2*p->nf3;      // fine grid total number of points
    if (p->nf * p->batchSize > MAX_NF) {
      fprintf(stderr, "[%s] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",__func__);
      return ERR_MAXNALLOC;
    }
    p->fwBatch = FFTW_ALLOC_CPX(p->nf * p->batchSize);    // the big workspace
    if (p->options.verbosity) printf("[%s] fwBatch %.2fGB alloc:   \t%.3g s\n", __func__,(double)1E-09*sizeof(CPX)*p->nf*p->batchSize, timer.elapsedsec());
    if(!p->fwBatch) {      // we don't catch all such mallocs, just this big one
      fprintf(stderr, "[%s] FFTW malloc failed for fwBatch (working fine grids)!\n",__func__);
      free(p->phiHat1); free(p->phiHat2); free(p->phiHat3);
      return ERR_ALLOC;
    }

    timer.restart();            // plan the FFTW

    int *ns = GRIDSIZE_FOR_FFTW(p);

    #pragma omp critical
    {
      p->fft_plan = FFTW_PLAN_MANY_DFT(
          /* int rank */ rank, /* const int *n */ ns, /* int howmany */ p->batchSize,
          /* fftw_complex *in */ p->fwBatch, /* const int *inembed */ NULL,
          /* int istride */ 1, /* int idist */ p->nf,
          /* fftw_complex *out */ p->fwBatch, /* const int *onembed */ NULL,
          /* int ostride */ 1, /* int odist */ p->nf,
          /* int sign */ p->fftSign, /* unsigned flags */ p->options.fftw_flags);
    }

    if (p->options.verbosity) printf("[%s] FFTW plan (mode %d, num_threads=%d):\t%.3g s\n", __func__,p->options.fftw_flags, fftw_threads, timer.elapsedsec());
    delete []ns;

  } else {  // -------------------------- type 3 (no planning) ------------

    // TODO: add error
  }

  return ier;         // report setup_spreader status (could be warning)
}


// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int FINUFFT_SETPTS(Plan<CPUDevice, FLT>* p, BIGINT nj, FLT* xj, FLT* yj, FLT* zj,
                   BIGINT nk, FLT* s, FLT* t, FLT* u)
/* For type 1,2: just checks and (possibly) sorts the NU xyz points, in prep for
   spreading. (The last 4 arguments are ignored.)
   For type 3: allocates internal working arrays, scales/centers the NU points
   and NU target freqs (stu), evaluates spreading kernel FT at all target freqs.
*/
{
  int d = p->rank_;     // abbrev for spatial rank
  CNTime timer; timer.start();
  p->nj = nj;    // the user only now chooses how many NU (x,y,z) pts

  if (p->type_ != TransformType::TYPE_3) {  // ------------------ TYPE 1,2 SETPTS -------------------
                     // (all we can do is check and maybe bin-sort the NU pts)
    p->X = xj;       // plan must keep pointers to user's fixed NU pts
    p->Y = yj;
    p->Z = zj;
    int ier = spreadcheck(p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->options.verbosity>1) printf("[%s] spreadcheck (%d):\t%.3g s\n", __func__, p->spopts.check_bounds, timer.elapsedsec());
    if (ier)         // no warnings allowed here
      return ier;    
    timer.restart();
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->nj);
    if (!p->sortIndices) {
      fprintf(stderr,"[%s] failed to allocate sortIndices!\n",__func__);
      return ERR_SPREAD_ALLOC;
    }
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->options.verbosity) printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__,p->didSort, timer.elapsedsec());

    
  } else {   // ------------------------- TYPE 3 SETPTS -----------------------
    // TODO: add error

  }
  return 0;
}
// ............ end setpts ..................................................


int FINUFFT_SPREADINTERP(Plan<CPUDevice, FLT>* p, CPX* cj, CPX* fk) {

  double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulated timing

  for (int b=0; b*p->batchSize < p->ntrans; b++) { // .....loop b over batches

    // current batch is either batchSize, or possibly truncated if last one
    int thisBatchSize = min(p->ntrans - b*p->batchSize, p->batchSize);
    int bB = b*p->batchSize;         // index of vector, since batchsizes same
    CPX* cjb = cj + bB*p->nj;        // point to batch of weights
    CPX* fkb = fk + bB*p->N;         // point to batch of mode coeffs
    if (p->options.verbosity>1) printf("[%s] start batch %d (size %d):\n",__func__, b,thisBatchSize);
    
    spreadinterpSortedBatch(thisBatchSize, p, cjb, fkb);
  }                                                   // ........end b loop
  
  return 0;
}


int FINUFFT_INTERP(Plan<CPUDevice, FLT>* p, CPX* cj, CPX* fk) {
    
  return FINUFFT_SPREADINTERP(p, cj, fk);
}


int FINUFFT_SPREAD(Plan<CPUDevice, FLT>* p, CPX* cj, CPX* fk) {

  return FINUFFT_SPREADINTERP(p, cj, fk);
}


// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int FINUFFT_EXECUTE(Plan<CPUDevice, FLT>* p, CPX* cj, CPX* fk){
/* See ../docs/cguru.doc for current documentation.

   For given (stack of) weights cj or coefficients fk, performs NUFFTs with
   existing (sorted) NU pts and existing plan.
   For type 1 and 3: cj is input, fk is output.
   For type 2: fk is input, cj is output.
   Performs spread/interp, pre/post deconvolve, and fftw_execute as appropriate
   for each of the 3 types.
   For cases of ntrans>1, performs work in blocks of size up to batchSize.
   Return value 0 (no error diagnosis yet).
   Barnett 5/20/20, based on Malleo 2019.
*/
  CNTime timer; timer.start();
  
  if (p->type_ != TransformType::TYPE_3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulated timing
    if (p->options.verbosity)
      printf("[%s] start ntrans=%d (%d batches, bsize=%d)...\n", __func__, p->ntrans, p->nbatch, p->batchSize);
    
    for (int b=0; b*p->batchSize < p->ntrans; b++) { // .....loop b over batches

      // current batch is either batchSize, or possibly truncated if last one
      int thisBatchSize = min(p->ntrans - b*p->batchSize, p->batchSize);
      int bB = b*p->batchSize;         // index of vector, since batchsizes same
      CPX* cjb = cj + bB*p->nj;        // point to batch of weights
      CPX* fkb = fk + bB*p->N;         // point to batch of mode coeffs
      if (p->options.verbosity>1) printf("[%s] start batch %d (size %d):\n",__func__, b,thisBatchSize);
      
      // STEP 1: (varies by type)
      timer.restart();
      if (p->type_ == TransformType::TYPE_1) {  // type 1: spread NU pts p->X, weights cj, to fw grid
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec();
      } else {          //  type 2: amplify Fourier coeffs fk into 0-padded fw
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      }
             
      // STEP 2: call the pre-planned FFT on this batch
      timer.restart();
      FFTW_EXECUTE(p->fft_plan);   // if thisBatchSize<batchSize it wastes some flops
      t_fft += timer.elapsedsec();
      if (p->options.verbosity>1)
        printf("\tFFTW exec:\t\t%.3g s\n", timer.elapsedsec());
      
      // STEP 3: (varies by type)
      timer.restart();        
      if (p->type_ == TransformType::TYPE_1) {   // type 1: deconvolve (amplify) fw and shuffle to fk
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      } else {          // type 2: interpolate unif fw grid to NU target pts
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec(); 
      }
    }                                                   // ........end b loop
    
    if (p->options.verbosity) {  // report total times in their natural order...
      if(p->type_ == TransformType::TYPE_1) {
        printf("[%s] done. tot spread:\t\t%.3g s\n",__func__,t_sprint);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot deconvolve:\t\t\t%.3g s\n", t_deconv);
      } else {
        printf("[%s] done. tot deconvolve:\t\t%.3g s\n",__func__,t_deconv);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot interp:\t\t\t%.3g s\n",t_sprint);
      }
    }
  }

  else {  // ----------------------------- TYPE 3 EXEC ---------------------

    // TODO: raise error

  }
  
  return 0; 
}


// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
int FINUFFT_DESTROY(Plan<CPUDevice, FLT>* p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be NULL or correctly
// allocated.
{
  if (!p)                // NULL ptr, so not a ptr to a plan, report error
    return 1;
  FFTW_FREE(p->fwBatch);   // free the big FFTW (or t3 spread) working array
  free(p->sortIndices);
  if (p->type_ == TransformType::TYPE_1 || p->type_ == TransformType::TYPE_2) {
    FFTW_DESTROY_PLAN(p->fft_plan);
    free(p->phiHat1);
    free(p->phiHat2);
    free(p->phiHat3);
  } else {               // free the stuff alloc for type 3 only
    // FINUFFT_DESTROY(p->innerT2plan);   // if NULL, ignore its error code
    // free(p->CpBatch);
    // free(p->Sp); free(p->Tp); free(p->Up);
    // free(p->X); free(p->Y); free(p->Z);
    // free(p->prephase);
    // free(p->deconv);
  }
  free(p);
  return 0;              // success
}
