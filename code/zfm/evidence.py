"""The reader's two instruments, computed inside the model for every candidate.

What blind reading measured (DESIGN.md 3, 5; memory lens-lrg-visual-signature):

                 4000 A break at my z     best emission line     line / break
    ELG          median  1.2 sigma        median 29.5 sigma      27
    LRG          median 15.2 sigma        median  2.3 sigma      0.15

Each class is invisible to the other class's instrument, so the pair of them is
the type call, and "what else is true at this z" is the reranker's job (sec 5).
The trained reranker could see neither: its input was the class, z, and nine
numbers of S around the peak.

Everything here is a fixed function of the input spectrum -- no parameters, no
labels, no Redrock -- so it costs no backward pass and can be checked against
speclib.py, the code the reader's numbers were measured with.  ev_validate.py
does that check.

Definitions follow speclib exactly, moved from the native 0.8 A grid to the
40 km/s log grid (0.8 A = 40 km/s at 6000 A, so pixel counts carry over):
  continuum   ivar-weighted running mean, 301 px            speclib.continuum
  line sigma  boxcar matched filter on the residual, / noise speclib.significance
              narrow 3 px (70 km/s), broad 75 px (3000 km/s)
  break       (mean_red - mean_blue) / sqrt(1/w_b + 1/w_r),  speclib.break_amplitude
              blue 3750-3950, red 4050-4250 rest, >= 20 good px each

One correction speclib does not need: data.py resamples with np.interp and does
not rescale ivar, so a log pixel narrower than 0.8 A (blue of 6000 A) carries
the full native ivar and every significance there would be inflated by up to
sqrt(0.8/0.48) = 1.29.  ivar is scaled by (pixel width / 0.8 A) here so a
window's summed ivar equals the native information in it.
"""
import numpy as np
import torch
import torch.nn.functional as F

from grids import LAM_OBS, DLOG, OBS_LO, N_OBS, Z_GRID, N_Z

# vacuum wavelengths (DESI is vacuum; an air/vacuum mix-up was a real bug here
# once -- memory lens-line-prior-findings).  sign: +1 emission, -1 absorption.
LINES = [
    # name        lambda    sign  broad (read on the 3000 km/s map)
    ('Lya',      1215.67,  +1,  True),
    ('NV',       1240.81,  +1,  True),
    ('SiIV',     1396.76,  +1,  True),
    ('CIV',      1549.48,  +1,  True),
    ('CIII]',    1908.73,  +1,  True),
    ('MgII',     2799.49,  +1,  True),
    ('[OII]',    3728.48,  +1,  False),
    ('[NeIII]',  3870.16,  +1,  False),
    ('CaK',      3934.78,  -1,  False),
    ('CaH',      3969.59,  -1,  False),
    ('Hd',       4102.89,  +1,  False),
    ('Gband',    4305.61,  -1,  False),
    ('Hg',       4341.68,  +1,  False),
    ('Hb',       4862.68,  +1,  False),
    ('[OIII]a',  4960.30,  +1,  False),
    ('[OIII]b',  5008.24,  +1,  False),
    ('Mgb',      5176.70,  -1,  False),
    ('NaD',      5894.60,  -1,  False),
    ('Ha',       6564.61,  +1,  False),
    ('[NII]',    6585.27,  +1,  False),
    ('[SII]a',   6718.29,  +1,  False),
    ('[SII]b',   6732.67,  +1,  False),
]
M = len(LINES)
LAM = np.array([l[1] for l in LINES])
SIGN = np.array([l[2] for l in LINES], np.float32)
BROAD = np.array([l[3] for l in LINES])
EMIS = SIGN > 0

CONT_PX, NARROW_PX, BROAD_PX = 301, 3, 75
TOL_PX = 3             # a line "is at" its predicted pixel within +-120 km/s
N_PEAKS = 16           # strongest narrow peaks checked for being unexplained
BLUE, RED = (3750., 3950.), (4050., 4250.)
MIN_PX = 20

# feature layout, so a reader of a trained model can name every input
# Continuum SHAPE, per candidate z (DESIGN.md 5, measured 2026-09-22).
# diag_alias.py: for an alias error the truth is among the candidates 93.6% of
# the time (ELG 99.2%) and 78.6% of them show the SAME number of >=6 sigma lines
# at both redshifts -- one line read as another line is symmetric in the lines.
# What is NOT symmetric is the continuum: the same observed shape is a different
# rest-frame shape at each candidate z.  These are rest-frame band fluxes, i.e.
# a colour measured from the spectrum itself -- no photometry, no selection
# quantity (sec 2).  A faint ELG has SNR ~1 per pixel but a band holds several
# hundred pixels, so the shape is measurable where single features are not.
NB = 12
BAND_EDGES = np.geomspace(700.0, 10000.0, NB + 1)
BAND_MID = np.sqrt(BAND_EDGES[:-1] * BAND_EDGES[1:])
BAND_X = np.log10(BAND_MID)

NAMES = ([f'narrow:{l[0]}' for l in LINES] + [f'broad:{l[0]}' for l in LINES]
         + [f'inwin:{l[0]}' for l in LINES]
         + ['break', 'break_valid', 'best_emis', 'line_minus_break',
            'n_emis>=3', 'n_emis>=6', 'n_emis>=8', 'n_abs<=-3', 'n_inwin',
            'unexpl_max', 'n_unexpl>=8',
            'floor_narrow', 'floor_broad', 'log_median_flux'])
N_FEAT = len(NAMES)
SHAPE_NAMES = ([f'shape:b{i:02d}' for i in range(NB)] + [f'bandsnr:b{i:02d}' for i in range(NB)]
               + [f'bandok:b{i:02d}' for i in range(NB)] + ['shape_slope', 'shape_blue_red'])
N_SHAPE = len(SHAPE_NAMES)

# Line PARTNERS, per candidate z (DESIGN.md 5, 2026-09-28).
# diag_alias.py: 78.6% of alias errors show the SAME number of >=6 sigma lines at
# both redshifts, and the alias loss trained on that moved nothing.  A line COUNT
# is symmetric; a line's PARTNER is not.  Read one strong line as [O III] 5008 and
# 4960 must sit at 1/2.98 of it; read it as [O II] and it must split into
# 3727/3730; read it as H beta and H alpha must be >= 2.86x.  Atomic physics
# (fixed ratios, Case B lower bounds), fitted to no label.
# Amplitude on the 3 px boxcar: A = sum(res ivar)/sum(ivar), variance 1/sum(ivar).
# A line of fixed velocity width spans the same number of log pixels at any
# lambda, so its peak flux density goes as F/lambda: A_x = r (lam_y/lam_x) A_y.
PAIRS = [  # name, lambda_x, lambda_y, r = F_x/F_y, 'eq' exact | 'ge' lower bound
    ('oiii',  4960.295, 5008.240, 1 / 2.98, 'eq'),
    ('nii',   6549.86,  6585.27,  1 / 3.0,  'eq'),
    ('ha_hb', 6564.61,  4862.68,  2.86,     'ge'),   # Case B; dust only raises it
    ('hb_hg', 4862.68,  4341.68,  1 / 0.468, 'ge'),
]
DOUBLETS = [  # name, blue, red (vacuum), min F_b/F_r, min F_r/F_b
    ('oii',  3727.092, 3729.875, 1 / 1.5, 0.35),   # density-limited 3729/3727 in [0.35, 1.5]
    ('mgii', 2796.352, 2803.531, 1.0,     0.5),    # optically thin 2:1 .. thick 1:1
]
PAIR_NAMES = ([f'pair:{p[0]}' for p in PAIRS] + [f'pairok:{p[0]}' for p in PAIRS]
              + [f'dbl_b:{d[0]}' for d in DOUBLETS] + [f'dbl_r:{d[0]}' for d in DOUBLETS]
              + [f'dbl_short:{d[0]}' for d in DOUBLETS] + [f'dblok:{d[0]}' for d in DOUBLETS])
N_PAIR = len(PAIR_NAMES)


def _pix(lam_obs):
    """Nearest log-grid pixel, -1 outside the observed window."""
    j = np.rint(np.log(lam_obs / OBS_LO) / DLOG).astype(np.int64)
    return np.where((j >= 0) & (j < N_OBS), j, -1)


# (M, N_Z): pixel of every line at every z bin
LINE_PIX = torch.from_numpy(_pix(LAM[:, None] * (1 + Z_GRID[None, :])))


def _win(lo, hi):
    """(N_Z,) first and one-past-last pixel of a rest window at each z."""
    a = np.ceil(np.log(lo * (1 + Z_GRID) / OBS_LO) / DLOG).astype(np.int64)
    b = np.floor(np.log(hi * (1 + Z_GRID) / OBS_LO) / DLOG).astype(np.int64) + 1
    return torch.from_numpy(np.clip(a, 0, N_OBS)), torch.from_numpy(np.clip(b, 0, N_OBS))


BWIN, RWIN = _win(*BLUE), _win(*RED)
BANDW = [_win(BAND_EDGES[i], BAND_EDGES[i + 1]) for i in range(NB)]
BAND_A = torch.stack([w[0] for w in BANDW])          # (NB, N_Z)
BAND_B = torch.stack([w[1] for w in BANDW])
_pz = lambda lam: torch.from_numpy(_pix(lam * (1 + Z_GRID)))
_pf = lambda lam: torch.from_numpy((np.log(lam * (1 + Z_GRID) / OBS_LO) / DLOG).astype(np.float32))
PF_X = torch.stack([_pf(p[1]) for p in PAIRS])                   # (P, N_Z) fractional pixel
PF_Y = torch.stack([_pf(p[2]) for p in PAIRS])
PAIR_IN = torch.stack([(_pz(p[1]) >= 0) & (_pz(p[2]) >= 0) for p in PAIRS])
PAIR_R = torch.tensor([p[3] * p[2] / p[1] for p in PAIRS], dtype=torch.float32)
PAIR_GE = torch.tensor([p[4] == 'ge' for p in PAIRS])
DF_B = torch.stack([_pf(d[1]) for d in DOUBLETS])
DF_R = torch.stack([_pf(d[2]) for d in DOUBLETS])
DBL_IN = torch.stack([(_pz(d[1]) >= 0) & (_pz(d[2]) >= 0) for d in DOUBLETS])
# flux bounds F_b >= qb F_r and F_r >= qr F_b, moved to amplitudes (A ~ F/lambda)
DBL_QB = torch.tensor([d[3] * d[2] / d[1] for d in DOUBLETS], dtype=torch.float32)
DBL_QR = torch.tensor([d[4] * d[1] / d[2] for d in DOUBLETS], dtype=torch.float32)
PIXW = torch.from_numpy((LAM_OBS * DLOG / 0.8).astype(np.float32))   # log px / native px


def _cs(v):
    """Cumulative sum with a leading zero, so sum(v[a:b]) = cs[b] - cs[a]."""
    return F.pad(v, (1, 0)).cumsum(-1)


def _box(v, n):
    """Centred boxcar SUM of width n (odd), zero outside the spectrum."""
    h = n // 2
    c = _cs(F.pad(v, (h, h)))
    return c[..., n:] - c[..., :-n]


SKY_WIN = 121        # px: local ivar reference, wide against a sky line (few px)


def spectrum_maps(x, sky=None):
    """x (B, 2, N_OBS) = [flux*sqrt(ivar), sqrt(ivar)] as data.py builds it,
    UNCLIPPED (the clip lives in ZFM.forward, in front of the encoder only).

    Returns per-spectrum maps (float32, no grad):
      narrow, broad  (B, N_OBS) line significance
      brk            (B, N_Z)   4000 A break significance at every z bin
      brk_ok         (B, N_Z)   >= 20 good pixels in both windows
      mflux          (B,)       median flux over good pixels
    """
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        x = x.float()
        s = x[:, 1]
        # ivar < 1e-6 is no measurement (the MultimodalUniverse convention);
        # without this, pixels whose interpolated ivar is ~1e-20 were kept and
        # two stars came out at 1e13 sigma
        good = s > 1e-3
        # sky: DESIGN.md 6e.  A sky-line residual is larger than the pipeline ivar
        # admits, so residual/sigma is inflated exactly where ivar dips.  Blind
        # reading of 100 teacher flags found 28 of the 41 unreadable ones owed
        # their ">= 10 sigma" to such a spike.  With sky = r, a pixel whose ivar is
        # below r x its local mean is treated as no measurement.  None = the old
        # behaviour, which every checkpoint trained before 2026-09-23 expects.
        if sky is not None:
            iv0 = torch.where(good, s * s, torch.zeros_like(s))
            loc = _box(iv0, SKY_WIN) / _box(good.float(), SKY_WIN).clamp_min(1)
            good = good & (iv0 >= sky * loc)
            # and a pixel within +-3 of a bad one is not trusted either: blind224
            # OBJ002 had seven 15-67 sigma "lines", every one beside ivar = 0
            bad = (~good).float().unsqueeze(1)
            good = good & (F.max_pool1d(bad, 7, 1, 3).squeeze(1) == 0)
        flux = torch.where(good, x[:, 0] / s.clamp_min(1e-12), torch.zeros_like(s))
        ivar = torch.where(good, s * s, torch.zeros_like(s)) * PIXW.to(x.device)
        fi = flux * ivar
        cont = _box(fi, CONT_PX) / _box(ivar, CONT_PX).clamp_min(1e-30)
        res = torch.where(good, flux - cont, torch.zeros_like(flux))
        ri = res * ivar

        def sig(n):
            w = _box(ivar, n)
            return torch.where(w > 1e-6 * n, _box(ri, n) / w.clamp_min(1e-30).sqrt(),
                               torch.zeros_like(w))

        narrow, broad = sig(NARROW_PX), sig(BROAD_PX)
        r3, w3 = _box(ri, NARROW_PX), _box(ivar, NARROW_PX)

        c1, c0, cg = _cs(fi), _cs(ivar), _cs(good.float())
        dev = x.device
        (ba, bb), (ra, rb) = [(w[0].to(dev), w[1].to(dev)) for w in (BWIN, RWIN)]

        def wsum(c, a, b):
            return c[:, b] - c[:, a]

        wb, wr = wsum(c0, ba, bb), wsum(c0, ra, rb)
        mb = wsum(c1, ba, bb) / wb.clamp_min(1e-30)
        mr = wsum(c1, ra, rb) / wr.clamp_min(1e-30)
        ok = (wsum(cg, ba, bb) >= MIN_PX) & (wsum(cg, ra, rb) >= MIN_PX)
        brk = torch.where(ok, (mr - mb) / (1 / wb.clamp_min(1e-30) + 1 / wr.clamp_min(1e-30)).sqrt(),
                          torch.zeros_like(mr))

        fm = torch.where(good, flux, torch.full_like(flux, float('nan')))
        mflux = torch.nanmedian(fm, 1).values.nan_to_num(0.0)
        return dict(narrow=narrow, broad=broad, brk=brk, brk_ok=ok, mflux=mflux,
                    c1=c1, c0=c0, cg=cg, r3=r3, w3=w3)


def _slog(v):
    return torch.sign(v) * torch.log1p(v.abs())


def shape_features(maps, zb):
    """(B, K, N_SHAPE): the rest-frame continuum shape at each candidate z.

    Scale is divided out (shape only) because brightness is already one feature
    and is a selection quantity on its own (BGS r<20 vs ELG g<23.6).
    """
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        B, K = zb.shape
        dev = zb.device
        a = BAND_A.to(dev)[:, zb].permute(1, 2, 0).reshape(B, K * NB)      # (B, K*NB)
        b = BAND_B.to(dev)[:, zb].permute(1, 2, 0).reshape(B, K * NB)
        g = lambda c: (c.gather(1, b) - c.gather(1, a)).view(B, K, NB)
        S1, S0, NG = g(maps['c1']), g(maps['c0']), g(maps['cg'])
        ok = (NG >= MIN_PX) & (S0 > 0)
        m = torch.where(ok, S1 / S0.clamp_min(1e-30), torch.zeros_like(S1))
        snr = m * S0.clamp_min(0).sqrt()
        nok = ok.sum(-1, keepdim=True).clamp_min(1)
        rms = ((m * m * ok).sum(-1, keepdim=True) / nok).sqrt().clamp_min(1e-12)
        shape = torch.where(ok, m / rms, torch.zeros_like(m))
        x = torch.from_numpy(BAND_X.astype(np.float32)).to(dev).view(1, 1, NB)
        w = ok.float()
        xb = (w * x).sum(-1, keepdim=True) / nok
        yb = (w * shape).sum(-1, keepdim=True) / nok
        num = (w * (x - xb) * (shape - yb)).sum(-1)
        den = (w * (x - xb) ** 2).sum(-1).clamp_min(1e-6)
        slope = num / den
        half = NB // 2
        blue = (shape[..., :half] * w[..., :half]).sum(-1) / w[..., :half].sum(-1).clamp_min(1)
        red = (shape[..., half:] * w[..., half:]).sum(-1) / w[..., half:].sum(-1).clamp_min(1)
        return torch.cat([shape, _slog(snr), ok.float(),
                          slope.unsqueeze(-1), (blue - red).unsqueeze(-1)], -1)


def _at(m, idx):
    """m (B, N) gathered at idx (B, ...) -> idx.shape"""
    return m.gather(1, idx.reshape(idx.shape[0], -1)).view(idx.shape)


def _ati(m, f):
    """m (B, N) linearly interpolated at fractional pixels f (B, ...)"""
    N = m.shape[1]
    f = f.clamp(0, N - 1.001)
    i0 = f.floor().long()
    t = f - i0
    return (1 - t) * _at(m, i0) + t * _at(m, i0 + 1)


def pair_features(maps, zb, slog=True):
    """(B, K, N_PAIR): partner-line consistency at each candidate z.

    Both members are read at FRACTIONAL pixels a fixed ln(lam_x/lam_y)/DLOG apart,
    after one common shift (half-pixel steps within +-TOL_PX) that puts the
    stronger member on its peak.  Reading each at its own rounded pixel cost up
    to a pixel of relative phase, and a real [O III] doublet came out at -2.7
    sigma (test_pairs.py); a common shift leaves the ratio intact.

    pair:x      chi of (A_x - r A_y); 'ge' pairs keep only the shortfall (<= 0).
                A single line read as [O III] 5008: -13 sigma; the doublet: ~0.
    dbl_b/r     each doublet member's significance at the best common shift.
    dbl_short   the weaker member's shortfall below its physical lower bound
                (<= 0).  A mean-minus-midpoint contrast was tried first and came
                out POSITIVE for one line read as [O II]: the shift puts one
                member on the line and the midpoint off it.
    """
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        narrow, r3, w3 = maps['narrow'], maps['r3'], maps['w3']
        dev = zb.device
        good = (w3 > 0).float()
        amp = torch.where(w3 > 0, r3 / w3.clamp_min(1e-30), torch.zeros_like(r3))
        var = torch.where(w3 > 0, 1 / w3.clamp_min(1e-30), torch.zeros_like(r3))
        zero = torch.zeros(1, device=dev)
        offs = torch.arange(-2 * TOL_PX, 2 * TOL_PX + 1, device=dev) * 0.5
        tab = lambda T: T.to(dev)[:, zb].permute(1, 2, 0)                  # (B,K,Q)
        best = lambda score: offs[score.argmax(-1)]                        # (B,K,Q)
        okat = lambda f: _ati(good, f) > 0.999                              # both neighbours good

        fx, fy = tab(PF_X), tab(PF_Y)
        sh = best(_ati(narrow, fy.unsqueeze(-1) + offs))
        fx, fy = fx + sh, fy + sh
        ok = tab(PAIR_IN) & okat(fx) & okat(fy)
        r = PAIR_R.to(dev)
        chi = (_ati(amp, fx) - r * _ati(amp, fy)) / (_ati(var, fx) + r * r * _ati(var, fy)).clamp_min(1e-30).sqrt()
        chi = torch.where(PAIR_GE.to(dev), chi.clamp_max(0), chi)
        chi = torch.where(ok, chi, zero)

        fb, fr = tab(DF_B), tab(DF_R)
        sh = best(_ati(narrow, fb.unsqueeze(-1) + offs) + _ati(narrow, fr.unsqueeze(-1) + offs))
        fb, fr = fb + sh, fr + sh
        okd = tab(DBL_IN) & okat(fb) & okat(fr)
        Ab, Ar, Vb, Vr = _ati(amp, fb), _ati(amp, fr), _ati(var, fb), _ati(var, fr)
        sgn = torch.where(Ab + Ar >= 0, 1.0, -1.0)          # absorption doublets too
        Ab, Ar = Ab * sgn, Ar * sgn
        qb, qr = DBL_QB.to(dev), DBL_QR.to(dev)             # amplitude-space bounds
        c1 = (Ab - qb * Ar) / (Vb + qb * qb * Vr).clamp_min(1e-30).sqrt()
        c2 = (Ar - qr * Ab) / (Vr + qr * qr * Vb).clamp_min(1e-30).sqrt()
        short = torch.minimum(c1, c2).clamp_max(0)
        sb = torch.where(okd, _ati(narrow, fb), zero)
        sr = torch.where(okd, _ati(narrow, fr), zero)
        short = torch.where(okd, short, zero)
        f = [chi, sb, sr, short]
        if slog:
            f = [_slog(v) for v in f]
        return torch.cat([f[0], ok.float(), f[1], f[2], f[3], okd.float()], -1)


def candidate_features(maps, zb, slog=True, shape=False, pairs=False):
    """zb (B, K) candidate z bins -> (B, K, N_FEAT).  slog=False returns raw
    sigma values, which is what ev_validate.py compares against speclib."""
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        narrow, broad = maps['narrow'], maps['broad']
        B, K = zb.shape
        dev = zb.device
        # signed extreme within +-TOL_PX: a line sits within 120 km/s of its
        # predicted pixel (bin quantisation plus real velocity offsets)
        pos = F.max_pool1d(narrow.unsqueeze(1), 2 * TOL_PX + 1, 1, TOL_PX).squeeze(1)
        neg = -F.max_pool1d(-narrow.unsqueeze(1), 2 * TOL_PX + 1, 1, TOL_PX).squeeze(1)
        ext = torch.where(pos.abs() >= neg.abs(), pos, neg)

        pix = LINE_PIX.to(dev)[:, zb].permute(1, 2, 0)            # (B,K,M)
        inwin = pix >= 0
        p = pix.clamp_min(0).reshape(B, -1)
        nar = torch.where(inwin, ext.gather(1, p).view(B, K, M), torch.zeros(1, device=dev))
        bro = torch.where(inwin, broad.gather(1, p).view(B, K, M), torch.zeros(1, device=dev))

        sign = torch.from_numpy(SIGN).to(dev)
        emis = torch.from_numpy(EMIS).to(dev)
        # the value that counts for a line: narrow for galaxy lines, broad for
        # the quasar lines, signed so that "present" is positive
        lb = torch.from_numpy(BROAD).to(dev)
        val = torch.where(lb, bro, nar) * sign
        ev_e = torch.where(emis & inwin, val, torch.zeros(1, device=dev))
        best = ev_e.max(-1).values
        brk = maps['brk'].gather(1, zb)
        brk_ok = maps['brk_ok'].gather(1, zb).float()

        # strongest narrow peaks anywhere, and whether this z explains them
        loc = (narrow == F.max_pool1d(narrow.unsqueeze(1), 7, 1, 3).squeeze(1)) & (narrow > 3)
        pv, pp = torch.where(loc, narrow, torch.zeros_like(narrow)).topk(N_PEAKS, 1)
        d = (pp.view(B, 1, N_PEAKS, 1) - pix.view(B, K, 1, M)).abs()
        expl = ((d <= TOL_PX) & inwin.view(B, K, 1, M)).any(-1)              # (B,K,P)
        un = torch.where(expl | (pv.view(B, 1, -1) <= 0), torch.zeros(1, device=dev),
                         pv.view(B, 1, -1).expand(B, K, -1))

        g = lambda v: v.view(B, 1, 1).expand(B, K, 1) if v.dim() == 1 else v.unsqueeze(-1)
        raw = torch.cat([
            nar, bro, inwin.float(),
            g(brk), g(brk_ok), g(best),
            g(torch.zeros_like(best)),                       # filled below
            g((ev_e >= 3).sum(-1).float()), g((ev_e >= 6).sum(-1).float()),
            g((ev_e >= 8).sum(-1).float()),
            g(((val <= -3) & ~emis & inwin).sum(-1).float()),
            g(inwin.sum(-1).float()),
            g(un.max(-1).values), g((un >= 8).sum(-1).float()),
            g(narrow.abs().amax(1)), g(broad.abs().amax(1)),
            g(torch.log10(maps['mflux'].clamp_min(1e-3))),
        ], -1)
        i_lmb = NAMES.index('line_minus_break')
        raw[..., i_lmb] = _slog(best) - _slog(brk)
        if shape:
            raw = torch.cat([raw, shape_features(maps, zb)], -1)
        if pairs:
            # appended after the slog block's indices, already in final form
            raw = torch.cat([raw, pair_features(maps, zb, slog=slog)], -1)
        if not slog:
            return raw
        out = raw.clone()
        sl = [NAMES.index(n) for n in NAMES
              if n.startswith(('narrow', 'broad')) or n in ('break', 'best_emis', 'unexpl_max',
                                                             'floor_narrow', 'floor_broad')]
        out[..., sl] = _slog(raw[..., sl])
        cnt = [NAMES.index(n) for n in NAMES if n.startswith('n_')]
        out[..., cnt] = raw[..., cnt] / 4.0
        return out
