"""The three grids, and the one fact that makes the architecture work.

On a log-lambda grid with a constant velocity step, a redshift is a pure
TRANSLATION:  lambda_obs = lambda_rest * (1+z), so
              log lambda_obs = log lambda_rest + log(1+z).

Everything else follows from that.  The observed grid and the rest grid use the
SAME step, so correlating one against the other over all integer shifts is a
scan over all redshifts on the grid, and an FFT does the whole scan at once.
"""
import numpy as np

C_KMS = 299792.458
VBIN = 40.0                      # DESIGN.md 1
DLOG = np.log1p(VBIN / C_KMS)    # 1.33425e-4

OBS_LO, OBS_HI = 3600.0, 9824.0          # the DESI BRZ union grid
Z_LO, Z_HI = -0.01, 4.00                 # QSO reach 4; stars sit at ~0

N_OBS = int(np.ceil(np.log(OBS_HI / OBS_LO) / DLOG)) + 1
LAM_OBS = OBS_LO * np.exp(np.arange(N_OBS) * DLOG)

REST_LO = OBS_LO / (1.0 + Z_HI)
REST_HI = OBS_HI / (1.0 + Z_LO)
N_REST = int(np.ceil(np.log(REST_HI / REST_LO) / DLOG)) + 1
LAM_REST = REST_LO * np.exp(np.arange(N_REST) * DLOG)

# shift s such that  LAM_OBS[i] = LAM_REST[i - s] * (1+z)
S_LO = int(np.floor(np.log((1 + Z_LO) * REST_LO / OBS_LO) / DLOG))
S_HI = int(np.ceil(np.log((1 + Z_HI) * REST_LO / OBS_LO) / DLOG))
N_Z = S_HI - S_LO + 1
SHIFTS = np.arange(S_LO, S_HI + 1)
Z_GRID = np.exp(SHIFTS * DLOG) * OBS_LO / REST_LO - 1.0

# the native DESI grid the h5 files are stored on (linear, 0.8 A)
LAM_NATIVE = np.arange(7781) * 0.8 + 3600.0

CLASSES = ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']
BITS = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS': 60, 'STAR': 61}


def z_to_bin(z):
    """Nearest z-grid bin.  Values off the grid are clipped, and the caller is
    expected to drop those rather than train on a clipped target."""
    s = np.log((1.0 + np.asarray(z)) * REST_LO / OBS_LO) / DLOG
    return np.clip(np.rint(s).astype(np.int64) - S_LO, 0, N_Z - 1)


def bin_to_z(b):
    return np.exp((np.asarray(b) + S_LO) * DLOG) * OBS_LO / REST_LO - 1.0


def z_in_grid(z):
    return (np.asarray(z) >= Z_LO) & (np.asarray(z) <= Z_HI)


if __name__ == '__main__':
    print('velocity bin      %.1f km/s   dlog = %.6e' % (VBIN, DLOG))
    print('observed grid     %5d points  %.1f - %.1f A' % (N_OBS, LAM_OBS[0], LAM_OBS[-1]))
    print('rest grid         %5d points  %.1f - %.1f A' % (N_REST, LAM_REST[0], LAM_REST[-1]))
    print('redshift grid     %5d bins    z = %.4f - %.4f' % (N_Z, Z_GRID[0], Z_GRID[-1]))
    print('uniform posterior costs log(%d) = %.3f nats' % (N_Z, np.log(N_Z)))
    for z in (0.0, 0.42, 1.0, 2.5, 3.9):
        b = z_to_bin(z)
        print('  z = %.4f -> bin %6d -> %.6f   (round trip err %.2e)'
              % (z, b, bin_to_z(b), abs(bin_to_z(b) - z)))
