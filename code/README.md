# Code

The measurement path, and the tooling that keeps a blind test blind.

These files were developed and run on NERSC Perlmutter against DESI DR1 at
`/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron`. They are included so the
method is inspectable and reproducible; paths are NERSC-specific and would need
changing elsewhere. Requires `numpy`, `scipy`, `astropy`, `matplotlib`.

## Order of use

```
make_cohort.py --write-sealed-key   # select, extract, render, seal the answers
check.py                            # assert the cohort invariants
                                    # -> read the images, record the type calls
measure3.py                         # fit the family each call selected
final3.py                           # open the sealed key ONCE; score; draw
```

`zoom.py` produces the four-width diagnostic for spectra too faint to read from
flux alone.

## Measurement path

| file | role |
|---|---|
| `speclib.py` | BRZ camera coaddition on the shared 0.8 Å grid; continuum, line significance and break amplitude; rendering |
| `engine.py` | the FFT scan. Redshift is a pure translation in ln λ, so one transform evaluates every trial redshift |
| `templates.py` | per-class families and redshift ranges |
| `methodc.py` | two-stage fit — candidates from high-passed matched filters, then a decision against one common unfiltered data vector |
| `d4000.py` | 4000 Å break candidates |
| `lines.py` | vacuum line list (air wavelengths cost ~80 km/s; this was verified empirically) |

## Two names that mean what they say

`speclib.local_step_significance` and `speclib.strongest_step_redshift` were
originally called `break_significance` and `break_redshift`. They are **not**
break detectors: measured on 100 objects, the peak lands on Hα 24% of the time,
Mg b 12%, and on the 4000 Å break **1%**. As a redshift estimator the second
scores 1/10 where reading the same figures by eye scores 8/10.

They are kept unchanged because a report section documents that failure, but
the names now describe the behaviour. To measure a break whose redshift is
already known, use `speclib.break_amplitude`.
