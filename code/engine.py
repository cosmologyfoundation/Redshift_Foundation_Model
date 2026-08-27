"""Multi-component matched-filter redshift scan.

A model is a small linear basis {T_1..T_K} on a rest-frame ln(lambda) grid.
At every redshift (= every integer lag) we solve the weighted normal equations

      M(z) c = v(z),      M_ab = sum_i w_i T_a(i-k) T_b(i-k)
                          v_a  = sum_i w_i f_i T_a(i-k)
      dchi2(z) = v . c

Every M_ab and v_a is a cross-correlation, so all redshifts are evaluated in
K(K+1)/2 + K inverse FFTs.  Coarse scan -> candidate peaks -> exact refinement
on the native pixel grid with a non-negativity constraint.
"""
import numpy as np
from numpy.fft import rfft, irfft
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import nnls

C_KMS = 299792.458
DLN = 1.5e-4
OBS_MIN, OBS_MAX = 3600.0, 9824.0
Z_MIN, Z_MAX = -0.012, 6.05
REST_MIN, REST_MAX = OBS_MIN / (1 + Z_MAX), OBS_MAX / (1 + Z_MIN)

ND = int(np.ceil(np.log(OBS_MAX / OBS_MIN) / DLN)) + 1
NT = int(np.ceil(np.log(REST_MAX / REST_MIN) / DLN)) + 1
LNOBS = np.log(OBS_MIN) + DLN * np.arange(ND)
LNREST = np.log(REST_MIN) + DLN * np.arange(NT)
OBSW, RESTW = np.exp(LNOBS), np.exp(LNREST)
NFULL = ND + NT - 1
LFFT = 1
while LFFT < NFULL:
    LFFT *= 2
_K = np.arange(NFULL) - (NT - 1)
ZGRID = np.exp(np.log(OBS_MIN) - np.log(REST_MIN) + _K * DLN) - 1.0
_ok = (ZGRID >= Z_MIN) & (ZGRID <= Z_MAX)
ZSLICE = slice(int(np.argmax(_ok)), int(NFULL - np.argmax(_ok[::-1])))
ZS = ZGRID[ZSLICE]
NZ = ZS.size
NATIVE_WAVE = 3600.0 + 0.8 * np.arange(7781)


def _nwin(kms):
    return max(3, int(round(kms / (C_KMS * DLN))) | 1)


def highpass(y, kms, robust=True):
    if not kms:                       # 0 -> keep the continuum untouched
        return y
    n = _nwin(kms)
    b = median_filter(y, n, mode="nearest") if robust else \
        uniform_filter1d(y, n, mode="nearest")
    return y - b


def prepare(flux, ivar, widths):
    """Native linear grid -> log grid; one high-passed copy per filter width."""
    w = np.maximum(np.nan_to_num(ivar), 0.0).astype(np.float64)
    f = np.nan_to_num(flux).astype(np.float64)
    bad = w <= 0
    if bad.any() and (~bad).sum() > 10:
        f = f.copy()
        f[bad] = np.interp(NATIVE_WAVE[bad], NATIVE_WAVE[~bad], f[~bad])
    fl = np.interp(OBSW, NATIVE_WAVE, f)
    wl = np.interp(OBSW, NATIVE_WAVE, w) * (OBSW * DLN) / 0.8
    # clip extreme single-pixel outliers (cosmics, sky residuals)
    r = (fl - median_filter(fl, 9, mode="nearest")) * np.sqrt(wl)
    wl[np.abs(r) > 12] = 0.0
    return {hw: highpass(fl, hw) for hw in set(widths)}, wl


class Model:
    """One template family: basis vectors + filter width + allowed z range."""

    def __init__(self, name, basis, hp_kms, zlim, prior=0.0, cov=None):
        B = np.atleast_2d(np.asarray(basis, dtype=np.float64))
        B = np.array([highpass(b, hp_kms, robust=False) for b in B])
        B /= np.sqrt(np.mean(B ** 2, axis=1, keepdims=True)) + 1e-30
        self.name, self.B, self.hp, self.zlim, self.prior = name, B, hp_kms, zlim, prior
        self.K = B.shape[0]
        self.Bf = [rfft(b[::-1], LFFT) for b in B]
        self.Pf = {}
        for a in range(self.K):
            for b in range(a, self.K):
                self.Pf[(a, b)] = rfft((B[a] * B[b])[::-1], LFFT)
        self.zmask = (ZS >= zlim[0]) & (ZS <= zlim[1])
        # Where the template actually carries information.  Outside this the
        # model must be treated as UNDEFINED (zero weight), never as a
        # prediction of zero flux -- otherwise every redshift that shifts real
        # data into the uncovered region is punished with a spurious chi2,
        # which biases the answer systematically towards low z.
        self.cov = np.ones(NT, bool) if cov is None else np.asarray(cov, bool)


def scan_model(m, G, W, ridge=1e-3):
    """dchi2(z) for one model. G=rfft(w*f_highpassed), W=rfft(w)."""
    K, nz = m.K, NZ
    v = np.empty((nz, K))
    for a in range(K):
        v[:, a] = irfft(G * m.Bf[a], LFFT)[:NFULL][ZSLICE]
    M = np.empty((nz, K, K))
    for a in range(K):
        for b in range(a, K):
            x = irfft(W * m.Pf[(a, b)], LFFT)[:NFULL][ZSLICE]
            M[:, a, b] = x
            M[:, b, a] = x
    tr = np.maximum(np.trace(M, axis1=1, axis2=2), 1e-12)
    M[:, np.arange(K), np.arange(K)] += ridge * tr[:, None] / K
    try:
        c = np.linalg.solve(M, v[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(M.reshape(-1, K), v.reshape(-1), rcond=None)[0]
    d = np.einsum("ij,ij->i", v, c)
    d[~np.isfinite(d)] = 0.0
    d[~m.zmask] = 0.0
    return np.maximum(d, 0.0), c


class Scanner:
    def __init__(self, models):
        self.models = models
        self.widths = sorted({m.hp for m in models})

    def coarse(self, flux, ivar):
        fh, w = prepare(flux, ivar, self.widths)
        W = rfft(w, LFFT)
        Gs = {hw: rfft(w * fh[hw], LFFT) for hw in self.widths}
        D = np.empty((len(self.models), NZ))
        for i, m in enumerate(self.models):
            D[i], _ = scan_model(m, Gs[m.hp], W)
        return D, fh, w


def peaks(d, nmax=6, sep=40):
    """Top local maxima of a dchi2 curve."""
    out = []
    dd = d.copy()
    for _ in range(nmax):
        k = int(np.argmax(dd))
        if dd[k] <= 0:
            break
        out.append(k)
        dd[max(0, k - sep):k + sep] = 0.0
    return out


def refine(m, fh, w, z0, halfwidth_kms=400.0, nstep=41):
    """Exact NNLS fit on the log grid over a fine z range around z0."""
    y = fh[m.hp]
    best = (-1.0, z0, None)
    zs = z0 + (1 + z0) * np.linspace(-halfwidth_kms, halfwidth_kms, nstep) / C_KMS
    sw = np.sqrt(w)
    ysw = y * sw
    for z in zs:
        lam = LNOBS - np.log(1 + z)
        A = np.empty((ND, m.K))
        for a in range(m.K):
            A[:, a] = np.interp(lam, LNREST, m.B[a], left=0.0, right=0.0)
        A *= sw[:, None]
        try:
            c, _ = nnls(A, ysw)
        except Exception:
            continue
        r = A @ c
        d = float(2 * r @ ysw - r @ r)
        if d > best[0]:
            best = (d, z, c)
    return best
