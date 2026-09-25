#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfastpt/cfastpt.h"
#include "basics.h"
#include "cosmo3D.h"
#include "pt_cfastpt.h"
#include "structs.h"

#include "log.c/src/log.h"

void get_FPT_bias(void) 
{
  static double cache[MAX_SIZE_ARRAYS];

  if (fdiff(cache[1], Ntable.random))
  {
    FPTIA.k_min     = 1.e-5;
    FPTIA.k_max     = 1.e+6;
    FPTbias.N       = 350 + 200 * Ntable.FPTboost;
    if (FPTbias.tab != NULL) {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(7, FPTbias.N);
  }
  if (fdiff(cache[0], cosmology.random) || fdiff(cache[1], Ntable.random))
  {
    const double dlogk = (log(FPTbias.k_max) - log(FPTbias.k_min))/FPTbias.N;

    #pragma omp parallel for
    for (int i=0; i<FPTbias.N; i++) 
    {
      FPTbias.tab[5][i] = exp(log(FPTbias.k_min) + i*dlogk);
      FPTbias.tab[6][i] = p_lin(FPTbias.tab[5][i], 1.0);
    }

    double Pout[5][FPTbias.N];
    Pd1d2(FPTbias.tab[5], FPTbias.tab[6], FPTbias.N, Pout[0]);
    Pd2d2(FPTbias.tab[5], FPTbias.tab[6], FPTbias.N, Pout[1]);
    Pd1s2(FPTbias.tab[5], FPTbias.tab[6], FPTbias.N, Pout[2]);
    Pd2s2(FPTbias.tab[5], FPTbias.tab[6], FPTbias.N, Pout[3]);
    Ps2s2(FPTbias.tab[5], FPTbias.tab[6], FPTbias.N, Pout[4]);

    #pragma omp parallel for
    for (int i=0; i<FPTbias.N; i++) 
    {
      FPTbias.tab[0][i] = Pout[0][i]; // Pd1d2
      FPTbias.tab[1][i] = Pout[1][i]; // Pd2d2
      FPTbias.tab[2][i] = Pout[2][i]; // Pd1s2
      FPTbias.tab[3][i] = Pout[3][i]; // Pd2s2
      FPTbias.tab[4][i] = Pout[3][i]; // Pd2s2
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

// ---------------------------------------------------------------------------
// Optional up-sampling of the TATT table (Ntable.FPTupsample = U > 1)
// ---------------------------------------------------------------------------
// get_FPT_IA runs FAST-PT on Nraw log-spaced nodes between k = 1e-5 and 1e6
// (c/H0 units): Nraw = 270 + 200*FPTboost, i.e. 24.6 nodes per decade in k at
// accuracyboost 1. The Limber integrands in cosmo2D.c read the table with
// interpol1d, a straight line in ln k between two nodes. A straight line misses
// the curvature of the spectrum: its error is about (dlnk^2/8)|d^2P/dlnk^2|,
// so it grows with the TATT amplitudes that multiply these rows. Measured on
// the DES Y3 MagLim 3x2pt vector (release covariance, 462 scale-cut points)
// at A1 = 5, A2 = -5, alpha1 = alpha2 = -5, bias_ta = 2: the straight lines
// shift the vector by chi2 = 7.4 at Omega_m = 0.3, 1e9 A_s = 2.19 and by
// chi2 = 1.4e6 at Omega_m = 0.8, 1e9 A_s = 4.5 (U = 1 against U = 64 below).
// Raising accuracyboost is not a clean cure, because Nraw = 270, 470, 670, ...
// are not nested: every boost moves the nodes, so the zig-zag error of the
// straight lines changes place rather than shrinking steadily.
// With U > 1 FAST-PT still runs on the same Nraw nodes. Once per cosmology,
// FPT_IA_upsample passes a cubic spline in ln k through each row and stores
// it on N = U*Nraw nodes of spacing dlnk/U (FPTIA.N = N, so cosmo2D.c needs no
// change). Every U-th new node is an old node and keeps FAST-PT's value
// exactly. The error left in the lookup is the spline error, ~dlnk^4, plus a
// straight-line error that is U^2 times smaller, and the lookup is still a
// single interpol1d. In the measurement above each doubling of U shrinks the
// change by a factor of 8 to 20; U = 32 is within chi2 = 1.1e-5 (Omega_m = 0.3)
// and 1.0 (Omega_m = 0.8) of U = 64, at no measurable run-time cost. This
// removes the interpolation error only: FAST-PT's own values on the Nraw nodes
// are unchanged. A row that never changes sign is splined in ln|P|, as
// CosmoSIS's tatt module interpolates such rows log-log: spectra are close to
// power laws in k, so ln|P| is smoother than P. A row that changes sign is
// splined in P itself. interpol1d returns the last value beyond the last
// FAST-PT node; the new table repeats that value there, so nothing changes
// in that range.
static void FPT_IA_upsample(
    double** const raw,  // [12][Nraw] FAST-PT output on the FAST-PT nodes
    const int Nraw,
    double** const tab,  // [12][N]    up-sampled table (FPTIA.tab), N = U*Nraw
    const int N,
    const double lnkmin,
    const double lnkmax
  )
{
  const int U = N/Nraw;
  const double dlnk  = (lnkmax - lnkmin)/Nraw; // FAST-PT node spacing
  const double dlnkU = (lnkmax - lnkmin)/N;    // the spacing cosmo2D.c reads

  double* x = (double*) malloc1d(Nraw);
  for (int i=0; i<Nraw; i++) {
    x[i] = lnkmin + i*dlnk;
  }
  #pragma omp parallel for
  for (int r=0; r<10; r++)
  {
    int pos = 1;
    int neg = 1;
    for (int i=0; i<Nraw; i++) {
      pos = pos && (raw[r][i] > 0);
      neg = neg && (raw[r][i] < 0);
    }
    const int uselog = pos || neg;
    const double sgn = neg ? -1.0 : 1.0;

    double* y = (double*) malloc1d(Nraw);
    for (int i=0; i<Nraw; i++) {
      y[i] = uselog ? log(sgn*raw[r][i]) : raw[r][i];
    }
    gsl_interp_accel* acc = gsl_interp_accel_alloc();
    gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, Nraw);
    gsl_spline_init(spline, x, y, Nraw);
    for (int j=0; j<N; j++)
    {
      const int i = j/U;
      if (j == i*U) { // an old node
        tab[r][j] = raw[r][i];
      }
      else if (i >= Nraw - 1) { // past the last FAST-PT node
        tab[r][j] = raw[r][Nraw - 1];
      }
      else {
        const double v = gsl_spline_eval(spline, lnkmin + j*dlnkU, acc);
        tab[r][j] = uselog ? sgn*exp(v) : v;
      }
    }
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free(y);
  }
  free(x);

  #pragma omp parallel for
  for (int j=0; j<N; j++)
  {
    tab[10][j] = exp(lnkmin + j*dlnkU);
    tab[11][j] = p_lin(tab[10][j], 1.0);
  }
}

void get_FPT_IA(void) 
{
  static double cache[MAX_SIZE_ARRAYS];
  // Ntable.FPTupsample = U > 1: FAST-PT writes to raw (Nraw nodes) and
  // FPTIA.tab holds the up-sampled table (FPTIA.N = U*Nraw). U = 1: FAST-PT
  // writes to FPTIA.tab directly (raw stays NULL), exactly as before.
  static double** raw = NULL;
  static int Nraw = 0;

  if (fdiff(cache[1], Ntable.random))
  {
    // k_min is an option (init_FPTIA_kmin; default 1e-5 c/H0 = 3.3e-9 h/Mpc).
    // Upstream CosmoLike (core v4.11.7) starts at 0.05 c/H0 = 1.7e-5 h/Mpc.
    // The Limber lookups of this table never go below k = (l + 1/2)/chi with
    // l = 1 and chi(z = 3) for the DES Y3 sources: k > 1.0 c/H0 at Omega_m 0.3,
    // > 0.66 c/H0 even at Omega_m 0.05, so either value covers them; a lookup
    // below k_min would give zero (cosmo2D.c). The 11 decades from 1e-5 amplify
    // rounding in cFASTPT's FFT-based TATT terms. Measured on DES Y3 MagLim
    // (des_y3 ppe/cfastpt_kmin_test, A1 = 5, A2 = -5, alpha = -5, bias_ta = 2,
    // Omega_m 0.3, nested grids, 1100 fixed nodes, CAMB kmax fixed):
    //   k_min 1e-5: turning FMA contraction off moves the vector by chi2 70-80,
    //               and the accuracyboost steps do not converge;
    //   k_min 0.05: turning FMA contraction off moves it by 1e-5, and the steps
    //               fall 4x in amplitude per doubling, to chi2 7e-5 at 4 -> 8.
    FPTIA.k_min = (Ntable.FPTkmin > 0) ? Ntable.FPTkmin : 1.e-5;
    FPTIA.k_max = 1.e+6;
    // The base node count is an option (init_FPTIA_base_nodes; default 270).
    // Upstream CosmoLike (core v4.11.7) uses 1100: 24.6 nodes per decade in k
    // become 100 per decade, so FAST-PT samples P_lin and its own output more
    // finely before any interpolation.
    // Additive growth (+200 per boost) moves every node at every boost. With
    // the nested rule (init_FPTIA_nested_nodes, m > 0) the count is base*m:
    // the nodes lnk_min + i*(lnk_max - lnk_min)/N exclude k_max, so the nodes
    // of N = base*m contain those of base*m/2, and a boost only adds nodes.
    // Measured on DES Y3 MagLim (des_y3 ppe/desy3_nested_grids): neither option
    // makes the TATT vector converge in accuracyboost at k_min = 1e-5: even with
    // the nodes held fixed, a tiny change of the input P_lin, or compiling with
    // FMA contraction off, moves cFASTPT's output by as much as one accuracyboost
    // step (k_min = 0.05 removes this, see above).
    const int base = (Ntable.FPTbase > 0 ? Ntable.FPTbase : 270);
    Nraw        = (Ntable.FPTnest > 0) ? base * Ntable.FPTnest : base + 200 * Ntable.FPTboost;
    FPTIA.N     = (Ntable.FPTupsample > 1 ? Ntable.FPTupsample : 1) * Nraw;

    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(12, FPTIA.N);
    if (raw != NULL) {
      free(raw);
      raw = NULL;
    }
    if (FPTIA.N > Nraw) {
      raw = (double**) malloc2d(12, Nraw);
    }
  }
  if (fdiff(cache[0], cosmology.random) || fdiff(cache[1], Ntable.random))
  {
    double** const tab = (raw != NULL) ? raw : FPTIA.tab;
    double lim[3];
    lim[0] = log(FPTIA.k_min);
    lim[1] = log(FPTIA.k_max);
    lim[2] = (lim[1] - lim[0])/Nraw;
    
    #pragma omp parallel for
    for (int i=0; i<Nraw; i++) 
    {
      tab[10][i] = exp(lim[0] + i*lim[2]);
      tab[11][i] = p_lin(tab[10][i], 1.0);
    }

    IA_tt(tab[10], tab[11], Nraw, tab[0], tab[1]);
    
    IA_ta(tab[10], tab[11], Nraw, tab[2], tab[3], tab[4], tab[5]);
    
    IA_mix(tab[10], tab[11], Nraw, tab[6], tab[7], tab[8], tab[9]);
    
    #pragma omp parallel for
    for (int i=0; i<Nraw; i++) {
      tab[7][i] *= 4.;
    }
    if (raw != NULL) {
      FPT_IA_upsample(raw, Nraw, FPTIA.tab, FPTIA.N, lim[0], lim[1]);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}
