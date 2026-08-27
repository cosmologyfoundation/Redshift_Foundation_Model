"""Direct measurement of the 4000A break -- a redshift estimator that is not
a chi2 template fit.

Looking at high-SNR LRG spectra, the break is obvious to the eye: the flux
roughly doubles across it.  A chi2 fit with free continuum terms hides that
signal, because the polynomials can reproduce the jump at any redshift.  So
measure the break itself, the way it is defined observationally (Balogh et al.
1999 narrow D4000): the ratio of the mean f_nu in 4000-4100A to 3850-3950A.

For every trial z we slide the two bands and ask "how big is the jump here?".
The answer needs no continuum model at all, so nothing can absorb it.
"""
import numpy as np
import engine as E

BLUE = (3850.0, 3950.0)
RED = (4000.0, 4100.0)
W = E.NATIVE_WAVE


def d4000_scan(flux, ivar, zmin=0.0, zmax=1.40, dz=0.0015):
    """Return z grid, break strength D(z), and its significance.

    Vectorised: the bands are contiguous in wavelength, so every band sum is a
    difference of two cumulative sums.
    """
    f = np.nan_to_num(flux).astype(np.float64)
    w = np.maximum(np.nan_to_num(ivar), 0.0).astype(np.float64)
    fn = f * W ** 2          # the index is defined in f_nu
    wn = w / W ** 4
    cfw = np.concatenate([[0.0], np.cumsum(fn * wn)])
    cw = np.concatenate([[0.0], np.cumsum(wn)])
    zs = np.arange(zmin, zmax, dz)
    one = 1.0 + zs

    def band(lo, hi):
        i0 = np.searchsorted(W, lo * one)
        i1 = np.searchsorted(W, hi * one)
        return cfw[i1] - cfw[i0], cw[i1] - cw[i0]

    nb, sb = band(*BLUE)
    nr, sr = band(*RED)
    inside = (BLUE[0] * one >= W[0]) & (RED[1] * one <= W[-1])
    good = inside & (sb > 0) & (sr > 0)
    D = np.zeros(zs.size); S = np.zeros(zs.size)
    sb_ = np.where(good, sb, 1.0); sr_ = np.where(good, sr, 1.0)
    fb = nb / sb_; fr = nr / sr_
    ok = good & (fb > 0)
    D[ok] = (fr / np.where(fb == 0, 1.0, fb))[ok]
    S[ok] = ((fr - fb) / np.sqrt(1.0 / sb_ + 1.0 / sr_))[ok]
    return zs, D, S


def break_candidates(flux, ivar, nmax=4, smin=3.0, sep=0.03):
    """Redshifts where a real 4000A break sits, strongest first."""
    zs, D, S = d4000_scan(flux, ivar)
    score = np.where((D > 1.25) & (S > smin), S * np.minimum(D, 3.0), 0.0)
    out = []
    sc = score.copy()
    for _ in range(nmax):
        k = int(np.argmax(sc))
        if sc[k] <= 0:
            break
        out.append((zs[k], float(D[k]), float(S[k])))
        sc[np.abs(zs - zs[k]) < sep] = 0.0
    return out
