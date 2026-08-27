# Measuring redshift from DESI spectra, without Redrock

Determining spectroscopic redshift and object type from **flux and inverse
variance alone** — and, just as much, learning when the answer cannot be
determined and should be refused.

The type is read from an unlabelled image of the spectrum. That reading then
selects which template family is fitted, and the fit gives the redshift.

---

## Results

Two sealed blind tests on fresh DESI DR1 spectra. In both, the reference
redshifts and types were written to a file at mode 600 during the build and
opened **once**, after every call was already on disk.

| | Part IV | Part V |
|---|---|---|
| spectra | 100 (20 per class) | 50 (10 per class) |
| redshift within ±1000 km/s | **95 / 100** | **47 / 50** |
| type, 5-way vs target class | 89 / 100 | **46 / 50** |
| **type, 3-way vs Redrock `SPECTYPE`** | **98 / 100** | **50 / 50** |
| the imaging target class, same comparison | 99 / 100 | 50 / 50 |

<p align="center">
  <img src="figures/c3_redshift.png" width="47%">
  <img src="figures/c3_type.png" width="47%">
</p>

**Part V is exact on the three-way classification** — every one of the fifty
gets the same STAR / GALAXY / QSO type Redrock assigns.

**BGS went from 13/20 to 10/10** between the two tests, from one change:
BGS-vs-LRG is a redshift cut (z ≈ 0.42, worth 92.6% over 20,000 objects), not
something visible in a spectrum, so it is assigned *after* the redshift is
measured instead of guessed from the break position. The BGS↔LRG cell that
dominated Part IV's errors is now empty.

Of Part V's three redshift misses: one is a true information limit (every line
past the 9824 Å edge), one is a 1226 km/s quasar near-miss against a 1000 km/s
mark — not a misidentification — and one is a **23σ line I mistook for a sky
residual**.

---

## What's here

| path | what it is |
|---|---|
| **[METHOD.md](METHOD.md)** | how the type is read and the redshift is measured — start here |
| **[report/redshift_report.html](report/redshift_report.html)** | the full report, 36 sections, 24 figures. Open in a browser |
| `report/redshift_report.ipynb` | the same, executed, with all code |
| `figures/` | the headline figures from both blind tests |
| `code/` | the measurement path and the blind-test tooling |

### Code

| file | role |
|---|---|
| `speclib.py` | official BRZ camera coaddition; continuum, significance and break statistics; rendering |
| `engine.py` | the FFT redshift scan — redshift is a pure translation in ln λ, so one transform covers every z |
| `templates.py` | per-class template families and their redshift ranges |
| `methodc.py` | two-stage fit: candidate generation, then decision on a common data vector |
| `d4000.py` | 4000 Å break candidate generation |
| `make_cohort.py` | builds a blind cohort and writes the sealed key |
| `measure3.py` | fits the family the visual call selected; assigns BGS/LRG from the fitted z |
| `final3.py` | opens the sealed key once, scores, draws the figures |
| `zoom.py` | four-width diagnostic for spectra too faint to read from flux |
| `check.py` | asserts the cohort invariants; exits non-zero if broken |

---

## The blind protocol

No Redrock information reaches selection, inputs or figures. That excludes `Z`,
`ZWARN`, `SPECTYPE`, `DELTACHI2`, and the `ZCAT_PRIMARY` / `MAIN_PRIMARY` dedup
flags — those rank by `ZWARN` and would smuggle the answer in through the
selection. Those values appear only when scoring work that is already finished.

- **selection columns**: `TARGETID`, `SURVEY`, `PROGRAM`, `HEALPIX`,
  `COADD_FIBERSTATUS`, `DESI_TARGET`, `TSNR2_*`
- **filters**: `SURVEY=="main"`, `TARGETID>0`, `COADD_FIBERSTATUS==0`
- **class**: `DESI_TARGET` bits, priority QSO(2) > LRG(0) > ELG(1) >
  BGS_ANY(60) > MWS_ANY(61)=STAR. These are *photometric target classes*, not
  spectroscopic truth — what the spectrum turns out to be is a finding
- **repeated TARGETIDs**: keep the highest `TSNR2_<class>` row
- **SNR**: measured from the spectrum, `median(|flux|·√ivar)` — never `TSNR2`,
  which describes the observation and not the object
- **renders**: an opaque id and nothing else, shuffled across classes

---

## Things worth knowing before reading the report

**"SNR" names three different quantities.** `TSNR2` is a property of the
*observation*; SNR per pixel is how *bright the object* is; feature significance
decides whether the redshift is *measurable*. `TSNR2` correlates with how bright
a star is at **−0.002** — exactly zero, because exposure depth cannot change
that. The survey equalises *evidence*, not photons: in dark time an LRG's
per-pixel SNR halves while its break significance is unchanged, and the extra
exposure buys distance (median z 0.41 → 0.81) rather than precision.

**Different classes carry the redshift in structurally different places.** Half
of an ELG's redshift information lives in **7 pixels** of 7781; an LRG needs
**1425**. That is a difference in kind, not difficulty, and it is why one
instrument cannot serve both — a line filter on an LRG returns 3.9σ, a break
statistic on an ELG returns 1.4σ.

**The 4000 Å break is a shape, not a feature.** A local-step detector finds it
1% of the time and locks onto Hα or Na D instead; reading the whole continuum
shape by eye beat it 8:1.

**Every remaining error is discrete.** Correct answers sit 1.9–2.9 σ_z from the
truth; wrong ones are 3,470–13,362 σ_z away, and **not one error in 12,000 falls
within 5σ_z**. The two populations do not overlap, so the remaining problem is
*which peak*, not *how precisely*.

---

## Status and honest limits

- Both blind tests are small — n = 100 and n = 50. Several results in this
  project looked strong at n ≈ 40 and moved substantially at n ≈ 20,000; the
  report flags each case where that happened.
- Scores are agreement with Redrock, which is itself a measurement. Where
  `ZWARN != 0`, the two disagree without either being demonstrated wrong.
- The target class is photometric intent. Some disagreements with it are the
  imaging label being wrong, and the report separates those from genuine errors.
- The report records what failed as carefully as what worked, including
  conclusions of mine that later evidence overturned.
