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

/* These are functions that do not rely on FLT.
   They are organized by originating file.
*/

#ifndef PRECISION_INDEPENDENT_H
#define PRECISION_INDEPENDENT_H

/* Common Kernels from spreadinterp3d */
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz,
                  int bnx, int bny, int bnz);

/* spreadinterp 2d */

__global__
void CalcSubProb_2d_Paul(int* finegridsize, int* num_subprob,
	int maxsubprobsize, int bin_size_x, int bin_size_y);

/* spreadinterp3d */


__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
                       int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);

__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts,
                           int* d_numsubprob,int numbins);

__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz,
                   int nobinx, int nobiny, int nobinz, int* binsize);

__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz,
          int nobinx, int nobiny, int nobinz, int* binsize);

__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz,
                    int nobinx, int nobiny, int nobinz, int* binsize, int* index,
                    int* binstartpts, int M);

#endif
