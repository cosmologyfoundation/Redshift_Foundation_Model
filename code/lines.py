"""Vacuum rest-frame wavelengths (Angstrom) and template definitions.

Sources: SDSS spectral line table (vacuum), NIST, standard HII-region /
Case-B recombination ratios, Vanden Berk et al. (2001) QSO composite.
No Redrock / AION material used.
"""
import numpy as np

# ---------------------------------------------------------------- line lists
# (wavelength_vac, relative_flux)  -- emission positive
SF_EMISSION = [                       # star-forming galaxy / HII region
    (3727.092, 0.60), (3729.875, 0.80),      # [O II] doublet
    (3869.86,  0.25), (3968.59,  0.08),      # [Ne III]
    (3890.15,  0.06),                        # H8 / He I blend
    (3971.20,  0.05),                        # H-eps
    (4102.89,  0.09),                        # H-delta
    (4341.68,  0.16),                        # H-gamma
    (4364.44,  0.04),                        # [O III] 4363
    (4862.68,  0.35),                        # H-beta
    (4960.295, 0.40),                        # [O III]
    (5008.240, 1.20),                        # [O III]
    (5877.25,  0.04),                        # He I
    (6302.05,  0.03),                        # [O I]
    (6549.86,  0.10),                        # [N II]
    (6564.61,  1.00),                        # H-alpha
    (6585.27,  0.30),                        # [N II]
    (6718.29,  0.13), (6732.67,  0.10),      # [S II]
    (7137.77,  0.03),                        # [Ar III]
    (9071.1,   0.03), (9533.2,  0.07),       # [S III]
]

# Old stellar population (LRG) *and* G/K/M stars: same absorption physics.
# negative = absorption.  width in km/s given per line (broader = molecular band)
OLD_ABS = [
    (3934.777, -1.00,  6.0),   # Ca II K
    (3969.588, -0.85,  6.0),   # Ca II H (+ H-eps)
    (4102.89,  -0.22,  6.0),   # H-delta
    (4305.61,  -0.45, 10.0),   # G band (CH)
    (4341.68,  -0.22,  6.0),   # H-gamma
    (4383.55,  -0.25,  6.0),   # Fe I
    (4457.0,   -0.18, 10.0),   # Ca I / Fe blend
    (4531.0,   -0.15, 10.0),
    (4863.0,   -0.18,  6.0),   # H-beta
    (5176.7,   -0.55,  8.0),   # Mg b
    (5271.0,   -0.25,  8.0),   # Fe I / Ca I
    (5335.0,   -0.18,  8.0),
    (5895.6,   -0.40,  6.0),   # Na D
    (6564.61,  -0.15,  6.0),   # H-alpha
    (8500.36,  -0.45,  6.0),   # Ca II triplet
    (8544.44,  -0.60,  6.0),
    (8664.52,  -0.50,  6.0),
]
# 4000 A break: modelled separately as a step (see build_template)

# A/F star: Balmer series in absorption dominates
BALMER_ABS = [
    (6564.61, -0.80), (4862.68, -1.00), (4341.68, -0.95), (4102.89, -0.90),
    (3971.20, -0.80), (3890.15, -0.70), (3836.47, -0.60), (3798.98, -0.50),
    (3771.70, -0.42), (3751.22, -0.35), (3735.43, -0.30), (3722.99, -0.26),
    (3713.03, -0.22), (3704.91, -0.19),
    (3934.777, -0.35), (3969.588, -0.30),   # Ca II H&K (weak in A, grows to F)
]

# QSO broad lines (Vanden Berk-like relative fluxes), width in km/s
QSO_BROAD = [
    (1025.72, 0.10, 3000.), (1033.82, 0.08, 3000.),   # Ly-beta + O VI
    (1215.67, 1.00, 3000.),                            # Ly-alpha
    (1240.81, 0.12, 3000.),                            # N V
    (1305.53, 0.05, 3000.), (1335.31, 0.04, 3000.),    # O I, C II
    (1399.80, 0.10, 3000.),                            # Si IV + O IV]
    (1549.48, 0.42, 3500.),                            # C IV
    (1640.40, 0.05, 3500.), (1663.5, 0.03, 3500.),
    (1857.40, 0.04, 3500.),                            # Al III
    (1908.73, 0.20, 3500.),                            # C III]
    (2326.0,  0.04, 3500.),
    (2799.12, 0.25, 4000.),                            # Mg II
    (3426.85, 0.03, 1500.),                            # [Ne V]
    (3728.48, 0.05, 1000.),                            # [O II]
    (4102.89, 0.05, 3000.), (4341.68, 0.09, 3000.),
    (4862.68, 0.25, 4000.),                            # H-beta
    (4960.30, 0.09, 1200.), (5008.24, 0.27, 1200.),    # [O III]
    (6564.61, 0.75, 4000.),                            # H-alpha
    (6585.27, 0.10, 1200.),
]


def gauss_comb(lnw, lines, sigma_kms_default, c=299792.458):
    """Sum of gaussians on a ln(lambda) grid. `lines` = (lam, amp[, sig_kms])."""
    out = np.zeros_like(lnw)
    for L in lines:
        lam, amp = L[0], L[1]
        sig = L[2] if len(L) > 2 else sigma_kms_default
        s = sig / c                       # sigma in ln-lambda units
        d = (lnw - np.log(lam)) / s
        m = np.abs(d) < 6
        out[m] += amp * np.exp(-0.5 * d[m] ** 2)
    return out
