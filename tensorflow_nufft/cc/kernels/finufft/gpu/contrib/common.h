#ifndef COMMON_H
#define COMMON_H

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

#include "dataTypes.h"
#include "utils.h"
#include "utils_fp.h"
#include "spreadinterp.h"

// constants needed within common
#define MAX_NQUAD 100              // max number of positive quadr nodes
// increase this if you need >1TB RAM...
#define MAX_NF    (BIGINT)1e11     // too big to ever succeed (next235 takes 1s)


// common.cpp provides...
int setup_spreader_for_nufft(tensorflow::nufft::SpreadOptions<FLT> &spopts, FLT eps,
                             const tensorflow::nufft::Options& options,
                             int dim);
int SET_NF_TYPE12(BIGINT ms, tensorflow::nufft::SpreadOptions<FLT> spopts,
                  const tensorflow::nufft::Options& options, BIGINT *nf,
                  BIGINT b);
void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, tensorflow::nufft::SpreadOptions<FLT> opts);
#endif  // COMMON_H
