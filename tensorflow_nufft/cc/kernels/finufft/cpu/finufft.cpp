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




// We macro because it has no FLT args but gets compiled for both prec's...

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

int spreadinterpSortedBatch(int batch_size, Plan<CPUDevice, FLT>* p, CPX* cBatch, CPX* fBatch=nullptr)
/*
  Spreads (or interpolates) a batch of batch_size strength vectors in cBatch
  to (or from) the batch of fine working grids p->grid_data_, using the same set of
  (index-sorted) NU points p->points_[0],Y,Z for each vector in the batch.
  The direction (spread vs interpolate) is set by p->spread_params_.spread_direction.
  Returns 0 (no error reporting for now).
  Notes:
  1) cBatch is already assumed to have the correct offset, ie here we
     read from the start of cBatch (unlike Malleo). grid_data_ also has zero offset
  2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  Barnett 5/19/20, based on Malleo 2019.
  3) the 4th parameter is used when doing interp/spread only. When received,
     input/output data is read/written from/to this pointer instead of from/to
     the internal array p->fWBatch. Montalt 5/8/2021
*/
{
  // opts.spread_threading: 1 sequential multithread, 2 parallel single-thread.
  // omp_sets_nested deprecated, so don't use; assume not nested for 2 to work.
  // But when nthr_outer=1 here, omp par inside the loop sees all threads...
  int nthr_outer = p->options_.spread_threading == SpreadThreading::SEQUENTIAL_MULTI_THREADED ? 1 : batch_size;

  if (fBatch == nullptr) {
    fBatch = (CPX*) p->grid_data_;
  }

  BIGINT grid_size_0 = p->grid_dims_[0];
  BIGINT grid_size_1 = 1;
  BIGINT grid_size_2 = 1;
  if (p->rank_ > 1) grid_size_1 = p->grid_dims_[1];
  if (p->rank_ > 2) grid_size_2 = p->grid_dims_[2];

  #pragma omp parallel for num_threads(nthr_outer)
  for (int i=0; i<batch_size; i++) {
    CPX *fwi = fBatch + i*p->grid_size_;  // start of i'th fw array in wkspace
    CPX *ci = cBatch + i*p->num_points_;            // start of i'th c array in cBatch
    spreadinterpSorted(p->sortIndices, grid_size_0, grid_size_1, grid_size_2,
                       (FLT*)fwi, p->num_points_, p->points_[0], p->points_[1], p->points_[2],
                       (FLT*)ci, p->spread_params_, p->didSort);
  }
  return 0;
}

int deconvolveBatch(int batch_size, Plan<CPUDevice, FLT>* p, CPX* fkBatch)
/*
  Type 1: deconvolves (amplifies) from each interior fw array in p->grid_data_
  into each output array fk in fkBatch.
  Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  again looping over fk in fkBatch and fw in p->grid_data_.
  The direction (spread vs interpolate) is set by p->spread_params_.spread_direction.
  This is mostly a loop calling deconvolveshuffle?d for the needed rank batch_size
  times.
  Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
*/
{
  // since deconvolveshuffle?d are single-thread, omp par seems to help here...
#pragma omp parallel for num_threads(batch_size)
  for (int i=0; i<batch_size; i++) {
    FFTW_CPX *fwi = p->grid_data_ + i*p->grid_size_;  // start of i'th fw array in wkspace
    CPX *fki = fkBatch + i*p->mode_count_;           // start of i'th fk array in fkBatch
    
    if (p->rank_ == 1)
      deconvolveshuffle1d(p->spread_params_.spread_direction, 1.0, p->phiHat1,
                          p->num_modes_[0], (FLT *)fki,
                          p->grid_dims_[0], fwi, p->options_.mode_order);
    else if (p->rank_ == 2)
      deconvolveshuffle2d(p->spread_params_.spread_direction,1.0, p->phiHat1,
                          p->phiHat2, p->num_modes_[0], p->num_modes_[1], (FLT *)fki,
                          p->grid_dims_[0], p->grid_dims_[1], fwi, p->options_.mode_order);
    else
      deconvolveshuffle3d(p->spread_params_.spread_direction, 1.0, p->phiHat1,
                          p->phiHat2, p->phiHat3, p->num_modes_[0], p->num_modes_[1], p->num_modes_[2],
                          (FLT *)fki, p->grid_dims_[0], p->grid_dims_[1], p->grid_dims_[2],
                          fwi, p->options_.mode_order);
  }
  return 0;
}







// --------------- rest is the 5 user guru (plan) interface drivers: -----------





// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int FINUFFT_SETPTS(Plan<CPUDevice, FLT>* p, BIGINT nj, FLT* xj, FLT* yj, FLT* zj,
                   BIGINT nk, FLT* s, FLT* t, FLT* u)
/* For type 1,2: just checks and (possibly) sorts the NU xyz points, in prep for
   spreading. (The last 4 arguments are ignored.)
   For type 3: allocates internal working arrays, scales/centers the NU points
   and NU target freqs (stu), evaluates spreading kernel FT at all target freqs.
*/
{
  p->num_points_ = nj;    // the user only now chooses how many NU (x,y,z) pts

  BIGINT grid_size_0 = p->grid_dims_[0];
  BIGINT grid_size_1 = 1;
  BIGINT grid_size_2 = 1;
  if (p->rank_ > 1) grid_size_1 = p->grid_dims_[1];
  if (p->rank_ > 2) grid_size_2 = p->grid_dims_[2];

  if (p->type_ != TransformType::TYPE_3) {  // ------------------ TYPE 1,2 SETPTS -------------------
                     // (all we can do is check and maybe bin-sort the NU pts)
    p->points_[0] = xj;       // plan must keep pointers to user's fixed NU pts
    p->points_[1] = yj;
    p->points_[2] = zj;
    int ier = spreadcheck(grid_size_0, grid_size_1, grid_size_2, p->num_points_, xj, yj, zj, p->spread_params_);

    if (ier)         // no warnings allowed here
      return ier;    

    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->num_points_);
    if (!p->sortIndices) {
      fprintf(stderr,"[%s] failed to allocate sortIndices!\n",__func__);
      return ERR_SPREAD_ALLOC;
    }
    p->didSort = indexSort(p->sortIndices, grid_size_0, grid_size_1, grid_size_2, p->num_points_, xj, yj, zj, p->spread_params_);

    
  } else {   // ------------------------- TYPE 3 SETPTS -----------------------
    // TODO: add error

  }
  return 0;
}
// ............ end setpts ..................................................


int FINUFFT_SPREADINTERP(Plan<CPUDevice, FLT>* p, CPX* cj, CPX* fk) {

  double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulated timing

  for (int b=0; b*p->batch_size_ < p->num_transforms_; b++) { // .....loop b over batches

    // current batch is either batch_size, or possibly truncated if last one
    int thisBatchSize = min(p->num_transforms_ - b*p->batch_size_, p->batch_size_);
    int bB = b*p->batch_size_;         // index of vector, since batchsizes same
    CPX* cjb = cj + bB*p->num_points_;        // point to batch of weights
    CPX* fkb = fk + bB*p->mode_count_;         // point to batch of mode coeffs
    if (p->options_.verbosity>1) printf("[%s] start batch %d (size %d):\n",__func__, b,thisBatchSize);
    
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
   For cases of num_transforms>1, performs work in blocks of size up to batch_size.
   Return value 0 (no error diagnosis yet).
   Barnett 5/20/20, based on Malleo 2019.
*/
  CNTime timer; timer.start();
  
  if (p->type_ != TransformType::TYPE_3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulated timing
    if (p->options_.verbosity)
      printf("[%s] start num_transforms=%d (%d batches, bsize=%d)...\n", __func__, p->num_transforms_, p->num_batches_, p->batch_size_);
    
    for (int b=0; b*p->batch_size_ < p->num_transforms_; b++) { // .....loop b over batches

      // current batch is either batch_size, or possibly truncated if last one
      int thisBatchSize = min(p->num_transforms_ - b*p->batch_size_, p->batch_size_);
      int bB = b*p->batch_size_;         // index of vector, since batchsizes same
      CPX* cjb = cj + bB*p->num_points_;        // point to batch of weights
      CPX* fkb = fk + bB*p->mode_count_;         // point to batch of mode coeffs
      if (p->options_.verbosity>1) printf("[%s] start batch %d (size %d):\n",__func__, b,thisBatchSize);
      
      // STEP 1: (varies by type)
      timer.restart();
      if (p->type_ == TransformType::TYPE_1) {  // type 1: spread NU pts p->points_[0], weights cj, to fw grid
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec();
      } else {          //  type 2: amplify Fourier coeffs fk into 0-padded fw
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      }
             
      // STEP 2: call the pre-planned FFT on this batch
      timer.restart();
      FFTW_EXECUTE(p->fft_plan_);   // if thisBatchSize<batch_size it wastes some flops
      t_fft += timer.elapsedsec();
      if (p->options_.verbosity>1)
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
    
    if (p->options_.verbosity) {  // report total times in their natural order...
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


