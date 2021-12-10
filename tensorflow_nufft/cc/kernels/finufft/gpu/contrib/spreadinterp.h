#if (!defined(SPREADINTERP_H) && !defined(SINGLE)) || \
  (!defined(SPREADINTERPF_H) && defined(SINGLE))

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "utils_fp.h"

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"



#ifdef SINGLE
#define SPREADINTERPF_H
#else
#define SPREADINTERP_H
#endif

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
namespace cufinufft {
FLT evaluate_kernel(FLT x, const tensorflow::nufft::SpreadParameters<FLT> &opts);
} // namespace cufinufft

#endif  // SPREADINTERP_H
