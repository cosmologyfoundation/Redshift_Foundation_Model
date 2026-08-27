"""Reusable DESI spectrum tools: BRZ camera coaddition, rendering, SNR.

COADDITION follows desispec.coaddition.coadd_cameras exactly.  The three arms
sit on a common 0.8 A grid --

    B  3600.0 - 5800.0   2751 px
    R  5760.0 - 7620.0   2326 px
    Z  7520.0 - 9824.0   2881 px

so their wavelength values coincide exactly in the two overlap regions
(5760-5800 and 7520-7620) and the merged grid is simply their union, 7781
points.  desispec assigns non-overlapping pixels directly and accumulates
inverse-variance weighted sums on overlapping ones; for a pixel covered by a
single arm that reduces to the arm's own values, so one formula covers both:

    flux(lambda) = sum_i ivar_i(lambda) * flux_i(lambda) / sum_i ivar_i(lambda)
    ivar(lambda) = sum_i ivar_i(lambda)
    mask(lambda) = OR_i mask_i(lambda),  cleared where ivar > 0

Masked pixels (mask != 0) contribute zero weight, which is what makes the sum
safe to write unconditionally.

RENDERING uses the two views that were established to work, for reasons
recorded in the report:
  * an inverse-variance weighted continuum, which is the only view in which a
    4000 A break or a percent-deep absorption line is visible at all
  * a significance spectrum, residual / sigma, with the filter width MATCHED to
    the feature being looked for -- a filter narrower than its feature reports
    the feature several times too faint
"""
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

BANDS = ('B', 'R', 'Z')
DR1 = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron'
HPIX = os.path.join(DR1, 'healpix')
CATALOG = os.path.join(DR1, 'zcatalog/v1/zall-pix-iron.fits')

# the union grid, fixed so every spectrum this library produces is comparable
WAVE = np.round(np.arange(3600.0, 9824.0 + 0.4, 0.8), 4)
NW = len(WAVE)

# widths, in km/s, of the things worth looking for -- used to match filters
WIDTH_KMS = {'nebular': 70., 'stellar_balmer': 900., 'qso_broad': 3000.,
             'break': 6000.}


def coadd_path(survey, program, healpix):
    return os.path.join(HPIX, survey, program, str(healpix // 100),
                        str(healpix),
                        f'coadd-{survey}-{program}-{healpix}.fits')


def coadd_cameras(path, rows=None):
    """Combine the B, R and Z arms onto the common grid.

    Returns wave, flux, ivar, mask, and the FIBERMAP TARGETIDs, with flux and
    ivar shaped (nrow, NW).  `rows` selects spectra by index within the file.
    """
    with fits.open(path, memmap=True) as h:
        tid = h['FIBERMAP'].data['TARGETID']
        n_all = len(tid)
        idx = np.arange(n_all) if rows is None else np.atleast_1d(rows)
        num = np.zeros((len(idx), NW))
        den = np.zeros((len(idx), NW))
        msk = np.zeros((len(idx), NW), dtype=np.int32)
        for b in BANDS:
            w = h[f'{b}_WAVELENGTH'].data
            f = np.asarray(h[f'{b}_FLUX'].data[idx], float)
            iv = np.asarray(h[f'{b}_IVAR'].data[idx], float)
            m = np.asarray(h[f'{b}_MASK'].data[idx], np.int32)
            # map this arm onto the union grid; the grids align exactly, so a
            # nearest-index lookup is exact rather than an interpolation
            j = np.round((w - WAVE[0]) / 0.8).astype(int)
            ok = (j >= 0) & (j < NW)
            j = j[ok]
            iv_eff = np.where(m[:, ok] != 0, 0.0, iv[:, ok])
            num[:, j] += iv_eff * f[:, ok]
            den[:, j] += iv_eff
            msk[:, j] |= m[:, ok]
        flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        ivar = den
        msk[ivar > 0] = 0
        return WAVE, flux, ivar, msk, tid[idx]


def continuum(flux, ivar, width_px=121):
    """Inverse-variance weighted running mean: the view that shows a break."""
    good = ivar > 0
    num = uniform_filter1d(np.where(good, flux * ivar, 0.), width_px, axis=-1)
    den = uniform_filter1d(np.where(good, ivar, 0.), width_px, axis=-1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def px_for(width_kms, lam=6000.0):
    """Matched-filter width in pixels for a feature of the given velocity width."""
    return max(3, int(round(width_kms / 299792.458 * lam / 0.8)) | 1)


def significance(flux, ivar, width_kms=70.0, cont_px=301):
    """Residual in units of sigma, on a flat noise floor.

    Normalising by the NOISE rather than by the continuum matters: dividing by
    the continuum amplifies noise wherever the continuum is faint, which is
    exactly where Ca II H and K sit for a low-redshift galaxy.
    """
    c = np.nan_to_num(continuum(flux, ivar, cont_px))
    res = np.where(ivar > 0, flux - c, 0.)
    n = px_for(width_kms)
    a = uniform_filter1d(res * np.where(ivar > 0, ivar, 0.), n, axis=-1) * n
    b = np.sqrt(np.maximum(
        uniform_filter1d(np.where(ivar > 0, ivar, 0.), n, axis=-1) * n, 1e-30))
    return a / b


def snr_stats(flux, ivar):
    """Several ways of saying 'how good is this spectrum', per object.

    They do not measure the same thing, which is the point:
      median_snr   typical flux over its own error -- continuum detectability
      cont_snr     the same on a smoothed continuum -- what a break can use
      npix_good    how much of the spectrum is usable at all
    """
    good = ivar > 0
    err = np.divide(1.0, np.sqrt(np.maximum(ivar, 1e-30)),
                    out=np.full_like(ivar, np.nan), where=good)
    with np.errstate(invalid='ignore'):
        s = np.where(good, flux / err, np.nan)
        c = continuum(flux, ivar, 121)
        cs = np.where(good, c * np.sqrt(np.maximum(ivar, 0)), np.nan)
    return dict(median_snr=np.nanmedian(s, axis=-1),
                cont_snr=np.nanmedian(cs, axis=-1),
                npix_good=good.sum(axis=-1),
                median_flux=np.nanmedian(np.where(good, flux, np.nan), axis=-1))


def render(wave, flux, ivar, out_png, titles=None, widths=None, xlim=None):
    """One row per spectrum: continuum on the left, significance on the right.

    `widths` gives the velocity width to match the filter to, per spectrum.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    flux = np.atleast_2d(flux); ivar = np.atleast_2d(ivar)
    n = len(flux)
    if widths is None:
        widths = [70.0] * n
    if titles is None:
        titles = [''] * n
    lo, hi = xlim or (wave[0], wave[-1])
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.5 * n), squeeze=False,
                             gridspec_kw={'width_ratios': [1.9, 1.1]})
    for i in range(n):
        c = continuum(flux[i], ivar[i], 121)
        s = significance(flux[i], ivar[i], widths[i])
        axL, axR = axes[i]
        axL.plot(wave, c, lw=1.1, color='C0')
        f = np.isfinite(c)
        if f.any():
            a, b = np.nanpercentile(c[f], [1, 99]); pad = .25 * (b - a) + 1e-9
            axL.set_ylim(a - pad, b + pad)
        axL.set_ylabel('continuum')
        axL.set_title(titles[i], fontsize=9, loc='left')
        axR.plot(wave, s, lw=.5, color='0.2')
        axR.axhline(0, color='0.6', lw=.5)
        for y in (3, -3):
            axR.axhline(y, color='C0', lw=.7, ls=':')
        m = np.nanmax(np.abs(s)) if np.isfinite(s).any() else 1
        axR.set_ylim(-min(m * 1.1, 30), min(m * 1.1, 30))
        axR.set_ylabel('sigma')
        axR.set_title(f'matched at {widths[i]:.0f} km/s '
                      f'({px_for(widths[i])} px)', fontsize=8, loc='left')
        for ax in (axL, axR):
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            ax.grid(which='major', alpha=.35)
            ax.grid(which='minor', alpha=.12)
    for ax in axes[-1]:
        ax.set_xlabel('observed wavelength (A)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=95)
    plt.close()
    return out_png

def local_step_significance(flux, ivar, inner_A=100.0, gap_A=25.0, outer_A=350.0):
    """Significance of a LOCALISED continuum step at each wavelength.

    NOT a 4000 A break detector, despite what it was built for.  Measured on
    100 targets: it locks onto Ha 30% of the time, [O III] 5007 12.5%, Hb
    12.5%, Na D 10%, and the 4000 A break only **1.2%**.  It finds real
    features (76.2% match a known line against a 25.6% null, +10.8 sigma) --
    just not the one it was named after.  See strongest_step_redshift.

    Comparing the mean flux either side of a point is not a break detector: on
    a smoothly rising continuum that difference is large everywhere, and the
    first version of this function returned >5 sigma for 100% of stars and
    quasars. It measured slope.

    A second difference removes any linear trend exactly --

        S(lam) ~ [ <f>_near_red - <f>_near_blue ]
                 - (b_in / b_out) * [ <f>_far_red - <f>_far_blue ]

    -- so it responds only to structure localised at lam, which is what a
    4000 A break is.
    """
    def wmean(nA):
        n = max(3, int(round(nA / 0.8)))
        w = np.where(ivar > 0, ivar, 0.)
        fw = np.where(ivar > 0, flux * ivar, 0.)
        num = uniform_filter1d(fw, n, axis=-1) * n
        den = uniform_filter1d(w, n, axis=-1) * n
        m = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        v = np.divide(1.0, den, out=np.full_like(den, np.inf), where=den > 0)
        return m, v, n

    m_in, v_in, n_in = wmean(inner_A)
    m_out, v_out, n_out = wmean(outer_A)
    g = int(round(gap_A / 0.8))
    s_in = n_in // 2 + g
    s_out = n_in + g + n_out // 2
    b_in, b_out = 2 * s_in, 2 * s_out
    k = b_in / b_out

    def sh(a, d):
        return np.roll(a, d, axis=-1)

    near = sh(m_in, -s_in) - sh(m_in, s_in)
    far = sh(m_out, -s_out) - sh(m_out, s_out)
    vnear = sh(v_in, -s_in) + sh(v_in, s_in)
    vfar = sh(v_out, -s_out) + sh(v_out, s_out)
    s = (near - k * far) / np.sqrt(np.maximum(vnear + k * k * vfar, 1e-30))
    edge = s_out + n_out
    s[..., :edge] = 0.0
    s[..., -edge:] = 0.0
    return s

def strongest_step_redshift(wave, flux, ivar, zmin=-0.005, zmax=1.6, rest=4000.0):
    """Redshift implied by the strongest local step.  DO NOT USE AS A REDSHIFT.

    Kept because report section 27 documents its failure and must reproduce.
    Scored against the catalogue on ten LRG targets it gets **1 of 10** within
    2%, where reading the same rendered figures by eye gets **8 of 10**; across
    a full batch it is 0-10% per class.

    The reason is structural, not a tuning problem: the 4000 A break is a
    SHAPE spanning several hundred angstroms, not a feature that can be pointed
    at, so any statistic hunting localised structure at the 100-350 A scale
    finds the genuinely localised things -- emission lines, metal absorption --
    and steps straight over a change in continuum level.  For a passive galaxy,
    use a template.  To measure a break whose redshift is already known, use
    break_amplitude below.
    """
    s = np.atleast_2d(local_step_significance(flux, ivar))
    lo = rest * (1 + zmin)
    hi = rest * (1 + zmax)
    ok = (wave > max(lo, wave[0] + 120)) & (wave < min(hi, wave[-1] - 120))
    ss = np.where(ok, s, -np.inf)
    j = np.argmax(ss, axis=-1)
    return wave[j] / rest - 1.0, ss[np.arange(len(ss)), j]

def break_amplitude(wave, flux, ivar, z, blue=(3750., 3950.), red=(4050., 4250.)):
    """Significance of the 4000 A break at a KNOWN redshift.

    A measurement, not a detector -- it is told where to look.  Windows follow
    the Balogh D4000 definition widened to use the full width of the break:

        S = (mean_red - mean_blue) / sqrt(var_red + var_blue)

    This is what makes the faint end work.  A dark-programme LRG at z = 1.12
    with a per-pixel SNR of 0.62 and no line above 3.8 sigma still gives an
    8.4 sigma break here, because it integrates ~1050 pixels.  The naive
    ceiling is SNR/px * sqrt(N) ~ 32; the realised 8.4 is the break's
    fractional contrast.

    It also separates the classes cleanly, which the line filter cannot:
    LRG 8.3-8.8 sigma against ELG 1.4, while the line filter gives LRG 3.9-4.7
    against ELG 11.6-13.4.  Each class is invisible to the other's instrument.
    """
    flux = np.atleast_2d(flux); ivar = np.atleast_2d(ivar)
    z = np.atleast_1d(z)
    out = np.full(len(flux), np.nan)
    npx = np.zeros(len(flux), int)
    for i in range(len(flux)):
        lam = wave / (1 + z[i])
        b = (lam > blue[0]) & (lam < blue[1]) & (ivar[i] > 0)
        r = (lam > red[0]) & (lam < red[1]) & (ivar[i] > 0)
        if b.sum() < 20 or r.sum() < 20:
            continue
        wb, wr = ivar[i][b].sum(), ivar[i][r].sum()
        mb = (flux[i][b] * ivar[i][b]).sum() / wb
        mr = (flux[i][r] * ivar[i][r]).sum() / wr
        out[i] = (mr - mb) / np.sqrt(1 / wb + 1 / wr)
        npx[i] = int(b.sum() + r.sum())
    return out, npx
