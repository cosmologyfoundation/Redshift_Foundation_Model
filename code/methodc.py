"""Method C: two-stage template redshift measurement.

  stage 1  high-passed matched filter, every redshift at once via FFT
           -> a shortlist of candidates (recall matters, precision does not)
  stage 2  refit every candidate against ONE common, unfiltered data vector,
           each model getting the same Legendre nuisance terms
           -> the decision (precision matters)

Two rules learned the hard way and enforced here:
  * chi2 from differently-filtered data is not comparable between models, so
    the decision stage uses a single shared data vector;
  * where a learned basis has no training coverage it must contribute ZERO
    WEIGHT, never a prediction of zero flux, or every redshift that shifts real
    data into the uncovered region is punished and the answer is dragged low.
"""
import numpy as np
import engine as E, templates as T, d4000

NPOLY, NCAND, NPK = 8, 20, 8


def make_poly(n):
    x = np.linspace(-1, 1, E.ND)
    return np.vstack([np.polynomial.legendre.legval(x, [0] * i + [1])
                      for i in range(n)]) if n else np.zeros((0, E.ND))


POLY = make_poly(NPOLY)


def gather(Z, D, nmax=NCAND, sep_kms=600.0):
    items = [(D[i, j], i, Z[i, j]) for i in range(Z.shape[0])
             for j in range(Z.shape[1]) if D[i, j] > 0]
    items.sort(key=lambda t: -t[0])
    keep = []
    for _, mi, z in items:
        if any(mj == mi and abs(z - zz) / (1 + z) * E.C_KMS < sep_kms
               for mj, zz in keep):
            continue
        keep.append((mi, z))
        if len(keep) >= nmax:
            break
    return keep


def score(flux, ivar, models, cands, npoly=NPOLY, poly=None, fine=True):
    fh, w = E.prepare(flux, ivar, [0.0])
    y = fh[0.0]
    s = np.median(np.abs(y)) + 1e-9
    y = y / s; w = w * s * s
    sw = np.sqrt(w)
    P = POLY if poly is None else poly

    def fit(m, z):
        lam = E.LNOBS - np.log(1 + z)
        cov = np.interp(lam, E.LNREST, m.cov.astype(float),
                        left=0., right=0.) > 0.5
        if cov.sum() < 200:
            return -np.inf
        swc = sw * cov
        ysc = y * swc
        A = np.empty((E.ND, m.K + npoly))
        for a in range(m.K):
            A[:, a] = np.interp(lam, E.LNREST, m.B[a], left=0., right=0.)
        if npoly:
            A[:, m.K:] = P.T
        A = A * swc[:, None]
        c, *_ = np.linalg.lstsq(A, ysc, rcond=1e-8)
        r = A @ c
        d = float(2 * r @ ysc - r @ r)
        if npoly:
            Ap = P.T * swc[:, None]
            cp, *_ = np.linalg.lstsq(Ap, ysc, rcond=1e-8)
            rp = Ap @ cp
            d -= float(2 * rp @ ysc - rp @ rp)
        return d

    out = []
    for mi, m in enumerate(models):
        for z in cands:
            if m.zlim[0] <= z <= m.zlim[1]:
                out.append({'m': mi, 'z': z, 'd': fit(m, z), 'K': m.K})
    out.sort(key=lambda o: -o['d'])
    if fine and out:
        b = out[0]; m = models[b['m']]
        bz, bs = b['z'], -np.inf
        for z in b['z'] + (1 + b['z']) * np.linspace(-250, 250, 21) / E.C_KMS:
            if m.zlim[0] <= z <= m.zlim[1]:
                dd = fit(m, z)
                if dd > bs:
                    bs, bz = dd, z
        if np.isfinite(bs):
            b['z'], b['d'] = bz, bs
    return out


class MethodC:
    def __init__(self):
        self.s1 = T.stage1_models()
        self.sc = E.Scanner(self.s1)
        self.s2 = T.stage2_models()

    def __call__(self, flux, ivar):
        D, _, _ = self.sc.coarse(flux, ivar)
        n = len(self.s1)
        Z = np.zeros((n, NPK)); Dv = np.zeros((n, NPK))
        for i in range(n):
            for j, k in enumerate(E.peaks(D[i], nmax=NPK, sep=50)):
                Z[i, j] = E.ZS[k]; Dv[i, j] = D[i][k]
        zs = {round(z, 5) for _, z in gather(Z, Dv)}
        for bz, _, _ in d4000.break_candidates(flux, ivar, nmax=3):
            zs.add(round(bz, 5))
        r = score(flux, ivar, self.s2, sorted(zs))
        if not r:
            return 0.0, -1, 0.0
        b = r[0]
        alt = next((o for o in r[1:]
                    if abs(o['z'] - b['z']) / (1 + b['z']) > 0.005), None)
        return b['z'], b['m'], b['d'] - (alt['d'] if alt else 0.0)
