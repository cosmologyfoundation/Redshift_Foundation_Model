"""Cross-class alias negatives (DESIGN.md 4 item 2, 7b-1).

For an object at true z, every emission line i that is really PRESENT in its
spectrum (>= 5 sigma at its true position, measured by evidence.py) would also
be explained by a different emission line j at

        z' = (1 + z) * lambda_i / lambda_j - 1

-- the redshift at which the model reads line i as line j.  Pairs span classes:
the one type error of cohort XV was [O II] 3727 against C IV 1549 on the same
pixel, which a within-class alias set would never contain.

Measured before this was written (alias_cover2.py, 2026-09-22, 50,048 spectra,
two seeds): these present-line aliases hold 20% of all catastrophics after the
evidence reranker (null 3%) and 27% of ELG catastrophics (null 5.6%).  That is
the ceiling of what this loss can move; the other ~73% of ELG errors are not
line confusions of this kind.

Only the label's LOCATION is used (option A, sec 6c); which lines are present is
read from the spectrum.
"""
import numpy as np
import torch

import evidence as E
from grids import Z_LO, Z_HI, REST_LO, OBS_LO, DLOG, S_LO, N_Z

PI, PJ = zip(*[(i, j) for i in range(E.M) for j in range(E.M)
               if i != j and E.EMIS[i] and E.EMIS[j]])
PI, PJ = np.array(PI), np.array(PJ)
RATIO = torch.from_numpy((E.LAM[PI] / E.LAM[PJ]).astype(np.float64))
THR = 5.0            # a line counts as present at >= 5 sigma
MIN_SEP = 64         # an alias must be farther from the truth than topk's own sep


def alias_bins(maps, zb, zt, A=32):
    """maps: evidence.spectrum_maps(x); zb (B,) true bin; zt (B,) true z.
    Returns bins (B, A) and mask (B, A): up to A alias bins per object, the
    strongest present lines first."""
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        dev = zb.device
        f = E.candidate_features(maps, zb.view(-1, 1), slog=False)[:, 0]
        nar, bro = f[:, :E.M], f[:, E.M:2 * E.M]
        val = torch.where(torch.from_numpy(E.BROAD).to(dev), bro, nar) * torch.from_numpy(E.SIGN).to(dev)
        vi = val[:, torch.from_numpy(PI).to(dev)]                         # (B,P)
        zp = (1 + zt.double().view(-1, 1)) * RATIO.to(dev).view(1, -1) - 1
        s = torch.log((1 + zp) * REST_LO / OBS_LO) / DLOG
        bins = (torch.round(s).long() - S_LO).clamp(0, N_Z - 1)
        ok = ((vi >= THR) & (zp >= Z_LO) & (zp <= Z_HI)
              & ((bins - zb.view(-1, 1)).abs() > MIN_SEP))
        score = torch.where(ok, vi, torch.full_like(vi, -1.0))
        k = min(A, score.shape[1])
        top, idx = score.topk(k, 1)
        return bins.gather(1, idx), top > 0


def surface_margin(logp_z, zb, bins, mask, margin=1.0):
    """softplus(margin + log P(alias) - log P(truth)), mean over real aliases."""
    la = logp_z.gather(1, bins)
    lt = logp_z.gather(1, zb.view(-1, 1))
    l = torch.nn.functional.softplus(margin + la - lt) * mask
    return l.sum() / mask.sum().clamp_min(1)
