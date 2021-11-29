#include "spreadinterp.h"
#include <stdlib.h>
#include <vector>
#include <math.h>

using namespace tensorflow::nufft;


FLT calculate_scale_factor(SpreadOptions<FLT> &opts, int dim, FLT dummy = 0.0) {
  // Calculates the scaling factor for spread/interp only.
  // Dummy param is used to trigger float/double overloading and avoid
  // redefinition errors.
  // This evaluates the integral of the kernel numerically via the trapezoidal
  // rule.
  BIGINT n = 100;
  FLT h = 2.0 / n;
  FLT x = -1.0;
  FLT sum = 0.0;
  for (BIGINT i = 1; i < n; i++) {
    x += h;
    sum += exp(opts.ES_beta * sqrt(1.0 - x * x));
  }
  sum += 1.0;
  sum *= h;
  // Note that 'c' is not included in the formula above. That seems to result
  // in incorrect results. Instead, applying this correction seems to work. 
  sum *= sqrt(1.0 / opts.ES_c);
  FLT scale = sum;
  if (dim > 1) { scale *= sum; }
  if (dim > 2) { scale *= sum; }
  return 1.0 / scale;
}

int setup_spreader(SpreadOptions<FLT> &opts,FLT eps, FLT upsampling_factor,
                   KernelEvaluationMethod kernel_evaluation_method, int dim)
// Initializes spreader kernel parameters given desired NUFFT tolerance eps,
// upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), and ker eval meth
// (etiher 0:exp(sqrt()), 1: Horner ppval).
// Also sets all default options in SpreadOptions<FLT>. See cnufftspread.h for opts.
// Must call before any kernel evals done.
// Returns: 0 success, 1, warning, >1 failure (see error codes in utils.h)
{
  if (upsampling_factor != 2.0) {   // nonstandard sigma
    if (kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
      fprintf(stderr,"setup_spreader: nonstandard upsampling_factor=%.3g cannot be handled by Horner evaluation\n",(double)upsampling_factor);
      return HORNER_WRONG_BETA;
    }
    if (upsampling_factor <= 1.0) {
      fprintf(stderr,"setup_spreader: error, upsampling_factor=%.3g is <=1.0\n",(double)upsampling_factor);
      return ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampling_factor > 4.0)
      fprintf(stderr,"setup_spreader: warning, upsampling_factor=%.3g is too large to be beneficial!\n",(double)upsampling_factor);
  }
    
  // defaults... (user can change after this function called)
  opts.spread_direction = 1;    // user should always set to 1 or 2 as desired
  opts.pirange = 1;             // user also should always set this
  opts.upsampling_factor = upsampling_factor;

  // as in FINUFFT v2.0, allow too-small-eps by truncating to eps_mach...
  int ier = 0;
  if (eps<EPSILON) {
    fprintf(stderr,"setup_spreader: warning, increasing tol=%.3g to eps_mach=%.3g.\n",(double)eps,(double)EPSILON);
    eps = EPSILON;
    ier = WARN_EPS_TOO_SMALL;
  }

  // Set kernel width w (aka ns) and ES kernel beta parameter, in opts...
  int ns = std::ceil(-log10(eps/(FLT)10.0));   // 1 digit per power of ten
  if (upsampling_factor != 2.0)           // override ns for custom sigma
    ns = std::ceil(-log(eps) / (PI*sqrt(1-1/upsampling_factor)));  // formula, gamma=1
  ns = max(2,ns);               // we don't have ns=1 version yet
  if (ns>MAX_NSPREAD) {         // clip to match allocated arrays
    fprintf(stderr,"%s warning: at upsampling_factor=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d.\n",__func__,
	    upsampling_factor,(double)eps,ns,MAX_NSPREAD);
    ns = MAX_NSPREAD;
    ier = WARN_EPS_TOO_SMALL;
  }
  opts.nspread = ns;
  opts.ES_halfwidth=(FLT)ns/2;   // constants to help ker eval (except Horner)
  opts.ES_c = 4.0/(FLT)(ns*ns);

  FLT betaoverns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns==2) betaoverns = 2.20;  // some small-width tweaks...
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  if (upsampling_factor != 2.0) {          // again, override beta for custom sigma
    FLT gamma=0.97;              // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma*PI*(1-1/(2*upsampling_factor));  // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (FLT)ns;    // set the kernel beta parameter
  if (opts.spread_interp_only)
    opts.ES_scale = calculate_scale_factor(opts, dim);
  //fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d beta=%.6f\n",(double)upsampling_factor,ns,(double)opts.ES_beta); // user hasn't set debug yet
  return ier;
}

namespace cufinufft {

FLT evaluate_kernel(FLT x, const SpreadOptions<FLT> &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}

} // namespace cufinufft
