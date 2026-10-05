"""Scaling-law plan for the redshift foundation model.

Two things this file does and one it deliberately does not.

DOES  compute the FLOP cost of a training run for THIS architecture, which is
      not 6ND end to end -- the encoder is, the FFT correlation is not.
DOES  lay out three runs whose losses identify the three constants of
      L(C) = E + K * C^(-gamma), and say what each costs in wall clock.
DOES NOT  assume the exponents.  Kaplan's alpha_N = 0.076 and Chinchilla's
      alpha = 0.34 / beta = 0.28 are fitted to autoregressive text
      cross-entropy in nats.  Only the FUNCTIONAL FORM transfers; the constants
      are what the three runs are for.  Quoting 0.076 as if it applied here
      would be the same error as quoting 92.6% as a property of a cut.

References for the form:
  Kaplan et al. 2020        L(N) = (Nc/N)^0.076,  C_train ~ 6ND
  Hoffmann et al. 2022      L(N,D) = E + A/N^alpha + B/D^beta,
                            N_opt ∝ C^0.5, D_opt ∝ C^0.5  (~20 tokens/param)
"""
import numpy as np

# ---------------------------------------------------------------- the task
P_IN = 7525            # log-lambda grid points per spectrum (DESIGN.md 2)
CH = 2                 # flux*sqrt(ivar) and sqrt(ivar)
Z_BINS = 13400         # dense log(1+z) posterior at 40 km/s (DESIGN.md 1)
N_SPECTRA = 19_652_625 + 18_659_804 - 2_200   # sv3 + main - seen - held out
N_ARCH = 64            # learned rest-frame archetypes, all classes
K_PEAK = 32            # candidates handed to the reranker

# ---------------------------------------------------------------- hardware
GPU_PEAK = 312e12      # A100 80GB, bf16 dense
N_GPU = 4              # measured budget: 4x A100
MFU = 0.40             # a dilated 1D CNN is bandwidth-bound; 40% is generous


def flops_per_spectrum(n_enc, n_rr=15e6):
    """Forward FLOPs, by component.  Only the encoder obeys 6ND."""
    enc = 2 * n_enc * P_IN                    # every weight applied at every position
    fft = 5 * N_ARCH * P_IN * np.log2(P_IN)   # correlate each archetype over all shifts
    lstsq = 2 * N_ARCH * P_IN                 # closed-form amplitudes
    # reranker: explicit transformer count, not 2*N*K.  An earlier version
    # used 2*N*K and was 3.8x low because it ignored attention entirely.
    d_rr, L_rr, ffn = 768, 8, 4
    par_rr = L_rr * (4 * d_rr * d_rr + 2 * ffn * d_rr * d_rr)
    rr = 2 * par_rr * K_PEAK + 2 * L_rr * 2 * K_PEAK * K_PEAK * d_rr
    return dict(encoder=enc, fft=fft, lstsq=lstsq, reranker=rr,
                total=enc + fft + lstsq + rr)


def cost(n_enc, n_spec, epochs=1.0):
    """Training FLOPs for one run, and wall clock on the real budget."""
    f = flops_per_spectrum(n_enc)
    fwd = f['total'] * n_spec * epochs
    total = 3 * fwd                            # fwd + bwd ~ 3x fwd
    naive6nd = 6 * n_enc * (n_spec * epochs * P_IN)
    sec = total / (GPU_PEAK * N_GPU * MFU)
    return dict(C=total, C_6ND=naive6nd, hours=sec / 3600,
                tok_per_param=(n_spec * epochs * P_IN) / n_enc, frac=f)


print('=' * 74)
print('the 6ND approximation, checked against this architecture')
print('=' * 74)
f = flops_per_spectrum(15e6)
for k in ('encoder', 'fft', 'lstsq', 'reranker'):
    print('  %-9s %10.3e FLOP/spectrum  %5.1f%%'
          % (k, f[k], 100 * f[k] / f['total']))
c = cost(15e6, 1e6)
print('  6ND would say %.3e, the component sum says %.3e  -> ratio %.2f'
      % (c['C_6ND'], c['C'], c['C_6ND'] / c['C']))

print()
print('=' * 74)
print('three runs, spaced a decade apart in compute, on 4x A100')
print('=' * 74)
print('%-6s %10s %12s %10s %10s %10s' %
      ('run', 'N enc', 'spectra', 'tok/param', 'C [FLOP]', 'wall [h]'))
runs = [('A', 1.5e6, 300_000), ('B', 6.0e6, 3_000_000), ('C', 15.0e6, 30_000_000)]
for nm, n, d in runs:
    ep = d / N_SPECTRA
    c = cost(n, min(d, N_SPECTRA), epochs=max(1.0, ep) if d > N_SPECTRA else 1.0)
    c = cost(n, d if d <= N_SPECTRA else N_SPECTRA,
             epochs=1.0 if d <= N_SPECTRA else d / N_SPECTRA)
    print('%-6s %10.2e %12s %10.0f %10.2e %10.1f'
          % (nm, n, f'{d:,}', c['tok_per_param'], c['C'], c['hours']))
print('\n  data ceiling: %s spectra = %.2e input positions'
      % (f'{N_SPECTRA:,}', N_SPECTRA * P_IN))
print('  run C uses %.1f%% of it in one epoch'
      % (100 * 30_000_000 / N_SPECTRA))

print()
print('=' * 74)
print('what three points can and cannot identify')
print('=' * 74)
print('  L(C) = E + K*C^(-gamma) has THREE unknowns and three runs give three')
print('  losses: exactly determined, zero residual, NO error bar and no')
print('  goodness of fit.  A fourth run off the compute-optimal ray is what')
print('  turns this from an interpolation into a measurement.')
print()


def solve_EKg(Cs, Ls):
    """Exact solve for E, K, gamma from three (C, L) points, log-spaced."""
    from scipy.optimize import brentq
    c1, c2, c3 = Cs
    l1, l2, l3 = Ls

    def g(gam):
        return ((l1 - l2) * (c2**-gam - c3**-gam)
                - (l2 - l3) * (c1**-gam - c2**-gam))
    gam = brentq(g, 1e-4, 2.0)
    K = (l1 - l2) / (c1**-gam - c2**-gam)
    E = l1 - K * c1**-gam
    return E, K, gam


print('  worked example -- if the three runs returned 4.10, 3.40, 2.95 nats:')
Cs = [cost(n, d)['C'] for _, n, d in runs]
E, K, g = solve_EKg(Cs, [4.10, 3.40, 2.95])
print('     E = %.3f nats   K = %.3e   gamma = %.3f' % (E, K, g))
for mult, lab in ((1, 'run C'), (10, '10x'), (100, '100x'), (1e6, 'infinite')):
    CC = Cs[-1] * mult
    print('     %-9s C = %.2e  ->  L = %.3f' % (lab, CC, E + K * CC**-g))
print('     sensitivity: move run A by +-0.05 nats ->  E = %.3f to %.3f'
      % (solve_EKg(Cs, [4.15, 3.40, 2.95])[0],
         solve_EKg(Cs, [4.05, 3.40, 2.95])[0]))

print()
print('=' * 74)
print('an independent prior on E, from the blind reading')
print('=' * 74)
uni = np.log(Z_BINS)
print('  a posterior that knows nothing costs log(%d) = %.2f nats' % (Z_BINS, uni))
for f_blind, lab in ((0.113, 'objects with nothing above 6 sigma (93/822)'),
                     (0.019, 'objects with nothing above 4 sigma (15/822)')):
    print('  if only those are uninformative: E >= %.3f nats   [%s]'
          % (f_blind * uni, lab))
print('  this is a LOWER bound and it ignores label noise in the training')
print('  redshifts, which pushes E up -- see DESIGN.md 6c.')
