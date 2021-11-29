#if (!defined(SPREADINTERP_H) && !defined(SINGLE)) || \
  (!defined(SPREADINTERPF_H) && defined(SINGLE))

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "utils_fp.h"

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


#define MAX_NSPREAD 16     // upper bound on w, ie nspread, even when padded
                           // (see evaluate_kernel_vector); also for common


#ifdef SINGLE
#define SPREADINTERPF_H
#else
#define SPREADINTERP_H
#endif

// struct tensorflow::nufft::SpreadOptions<FLT> {      // see cnufftspread:setup_spreader for defaults.
//   int nspread;            // w, the kernel width in grid pts
//   int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
//   int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
//   FLT upsampling_factor;          // sigma, upsampling factor, default 2.0
//   bool spread_interp_only; // 0: NUFFT, 1: spread or interpolation only
//   // ES kernel specific...
//   FLT ES_beta;
//   FLT ES_halfwidth;
//   FLT ES_c;
//   FLT ES_scale;           // used for spread/interp only
// };

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
namespace cufinufft {
FLT evaluate_kernel(FLT x, const tensorflow::nufft::SpreadOptions<FLT> &opts);
} // namespace cufinufft

int setup_spreader(tensorflow::nufft::SpreadOptions<FLT> &opts, FLT eps, FLT upsampling_factor,
                   tensorflow::nufft::KernelEvaluationMethod kernel_evaluation_method,
                   int dim);

#endif  // SPREADINTERP_H
