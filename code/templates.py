"""Template families for Method C.

Starts as pure atomic physics; learned per-class bases in $PSCRATCH replace
them once the three-way consensus has produced a clean training set.
"""
import numpy as np, os
from scipy.ndimage import uniform_filter1d
import engine as E, lines as LN

R, LR = E.RESTW, E.LNREST
BASE = os.environ.get('CLM_SCRATCH',
                      '/pscratch/sd/s/shunth/claude_learning_map')
BASIS_DIR = os.path.join(BASE, 'bases')

CLASSES = ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']
ZLIM = {'STAR': (-0.006, 0.006), 'BGS': (-0.010, 0.80),
        'LRG': (-0.010, 1.50), 'ELG': (-0.010, 1.90), 'QSO': (-0.010, 6.00)}
HP = {'STAR': 12000., 'BGS': 12000., 'LRG': 16000.,
      'ELG': float(os.environ.get('CLM_HP_ELG', 2500.)), 'QSO': 30000.}

BALMER_EM = [(6564.61, 1.00), (4862.68, 0.35), (4341.68, 0.16),
             (4102.89, 0.09), (3971.20, 0.05), (3890.15, 0.04)]
FORBID_EM = [(3727.092, 0.60), (3729.875, 0.80), (3869.86, 0.25),
             (3968.59, 0.08), (4364.44, 0.04), (4960.295, 0.40),
             (5008.240, 1.20), (6549.86, 0.10), (6585.27, 0.30),
             (6718.29, 0.13), (6732.67, 0.10), (6302.05, 0.03),
             (7137.77, 0.03), (9071.1, 0.03), (9533.2, 0.07)]


def _step(edge, depth, kms):
    return uniform_filter1d(np.where(R < edge, -depth, 0.0), E._nwin(kms),
                            mode='nearest')


def analytic():
    em = [LN.gauss_comb(LR, BALMER_EM, 70.), LN.gauss_comb(LR, FORBID_EM, 70.)]
    old = [_step(4000., .60, 1500.), LN.gauss_comb(LR, list(LN.OLD_ABS), 200.)]
    star = [LN.gauss_comb(LR, LN.BALMER_ABS, 900.) + _step(3647., .35, 2000.),
            LN.gauss_comb(LR, list(LN.OLD_ABS), 120.)]
    qt = LN.gauss_comb(LR, LN.QSO_BROAD, 3000.)
    trans = np.ones_like(R)
    trans[R < 1216] = 0.60; trans[R < 1026] = 0.35; trans[R < 912] = 0.05
    trans = uniform_filter1d(trans, E._nwin(3000.), mode='nearest')
    qso = [trans * (R / 3000.) ** a for a in (-1.5, -0.5, 0.5)]
    qso += [(1. - trans) * (R / 3000.) ** -0.5, qt]
    return {'STAR': star, 'BGS': old + em, 'LRG': old, 'ELG': em, 'QSO': qso}


def _path(c):
    return os.path.join(BASIS_DIR, f'basis_{c}.npz')


def _load(c, npc=None):
    p = _path(c)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    B, cov = d['B'], d['cov']
    if npc:
        B = B[:npc + 1]
    if c == 'STAR':                     # learned in the observed frame
        B = np.array([np.interp(LR, E.LNOBS, b, left=0., right=0.) for b in B])
        cov = np.interp(LR, E.LNOBS, cov.astype(float), left=0., right=0.) > .5
    return B, cov


# A sixth family, found by looking: narrow-line AGN / extreme emission-line
# objects.  Very high equivalent width narrow lines on a weak blue continuum
# with no 4000A break -- a shape neither BGS (break + stellar absorption) nor
# ELG (learned at z>0.7 where [O II] dominates) nor QSO (all lines broad) has.
# name, redshift range, stage-1 high-pass.  Several entries may serve one
# target class -- a quasar at z=0.5 and one at z=3 are different objects to a
# matched filter, and forcing one template to cover both helps neither.
# Splitting QSO into two regimes was tested and cost 0.4 points: each
# specialist trains on less data and the two compete, and both costs exceed
# what specialisation buys at this sample size.  Reverted.
EXTRA = [('NLAGN', (-0.010, 0.550), 2500.)]
NPC_S2 = 8
NPC_S1 = 5


def stage2_models():
    """Continuum-carrying models that decide between candidates."""
    A = analytic()
    ms = []
    for c in CLASSES:
        # only skip a per-class model if a replacement is genuinely active in
        # EXTRA -- keying this off a file on disk left QSO with no template
        # at all after the regime split was reverted
        if any(nm.startswith(c) for nm, _, _ in EXTRA):
            continue
        got = _load(c, npc=NPC_S2)
        if got is None:
            ms.append(E.Model(c + 'a', A[c], 0.0, ZLIM[c]))
        else:
            ms.append(E.Model(c, got[0], 0.0, ZLIM[c], cov=got[1]))
    for nm, zl, _ in EXTRA:
        g = _load(nm, npc=NPC_S2)
        if g is not None:
            ms.append(E.Model(nm, g[0], 0.0, zl, cov=g[1]))
    return ms


def stage1_models():
    """Line-sensitive candidate generators (analytic always kept)."""
    A = analytic()
    ms = []
    for c in CLASSES:
        ms.append(E.Model(c + 'a', A[c], HP[c], ZLIM[c]))
        got = _load(c, npc=5)
        if got is not None:
            ms.append(E.Model(c + 'h', got[0], HP[c], ZLIM[c], cov=got[1]))
    for nm, zl, hp in EXTRA:
        g = _load(nm, npc=NPC_S1)
        if g is not None:
            ms.append(E.Model(nm + 'h', g[0], hp, zl, cov=g[1]))
    return ms
