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

#ifndef SPREAD_OPTS_H
#define SPREAD_OPTS_H

#include "tensorflow_nufft/cc/kernels/finufft/dataTypes.h"

// C-compatible options struct for spreader.
// (mostly internal to spreadinterp.cpp, with a little bleed to finufft.cpp)

typedef struct spread_opts {  // see spreadinterp:setup_spreader for defaults.
  // This is the main documentation for these options...
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  int chkbnds;            // 0: don't check NU pts in 3-period range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  int kerpad;             // 0: no pad w to mult of 4, 1: do pad
                          // (this helps SIMD for kerevalmeth=0, eg on i7).
  int nthreads;           // # threads for spreadinterp (0: use max avail)
  int sort_threads;       // # threads for sort (0: auto-choice up to nthreads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int debug;              // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold;   // num threads before switching spreadSorted to using atomic ops
  double upsampfac;       // sigma, upsampling factor
  int spreadinterponly;   // 0: NUFFT, 1: spread or interpolation only
  // ES kernel specific consts used in fast eval, depend on precision FLT...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
  FLT ES_scale;           // used for spread/interp only
} spread_opts;

#endif   // SPREAD_OPTS_H
