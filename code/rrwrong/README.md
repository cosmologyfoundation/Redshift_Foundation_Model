# rrwrong — which Redrock redshifts are wrong

`rrwrong_filter.py IN.fits OUT.parquet [--thr 0.01]` scores every row of a DR1 zcatalog file with
P_WRONG = P(Redrock Z is catastrophically wrong) and sets KEEP = COADD_FIBERSTATUS==0 & P_WRONG<=thr.
Model: `/pscratch/sd/s/shunth/rrwrong/rrwrong_model_v1.joblib`.

## Truth used for learning
DESI EDR visual inspection (Lan+22 arXiv:2208.08516; Alexander+22 arXiv:2208.08517), 18,653 targets
with VI_QUALITY>=2.5. Redrock outputs for them at three depths: sv1 per-night (105,760 rows, ~main depth),
sv1 tile-cumulative and sv1 healpix. Wrong = |dz|/(1+z_VI) > 0.0033 (0.01 if VI says QSO);
VI star with |Z|<0.005 counts as right. `build_vi.py`, `label_vi.py`, `train_final.py`.

## Features (`feats.py`)
Redrock: Z, log DELTACHI2, ZERR/(1+z), CHI2/NPIXELS, NPIXELS, ZWARN, SPECTYPE, normalised COEFF_0-4, max|COEFF|.
Depth: TSNR2_{ELG,LRG,QSO,BGS,LYA}, COADD_EXPTIME. Imaging: grzW1W2 mags and colours, fibre mags,
flux S/N, EBV, MORPHTYPE, SHAPE_R, SERSIC, Gaia G, BP-RP, parallax, proper motion, MASKBITS. Target bits.

## Validation
| test | ZWARN0 & DCHI2>25 | model |
|---|---|---|
| VI, leave-one-field-out, per-night depth, equal keep 81% | wrong 1.58% | 0.35% |
| 1,008,385 main targets on 2 tiles, equal keep 88%: pair disagreement | 0.73% | 0.16% |
| same, P<=0.01 (keep 83.7%) | | 0.05% |
| Gaia MWS_WD with Z>0.1 (2,022), fraction removed | 30.8% | 100% (P>0.01) |
| blind-read Redrock errors (110), fraction removed | 13.6% | 78% (P>0.01) / 66% (P>0.05) |
| controls removed (Gaia WD at rest / blind Redrock-right) | 1.0% / 0% | 5.6% / 12% (P>0.01) |

Full main zpix (20.1M, TARGETID>0), keep fraction at P<=0.01 vs ZWARN0&DCHI2>25:
BGS 97.7/97.6, ELG 71.6/70.9, LRG 94.7/96.8, MWS 98.1/98.2, QSO 76.0/80.8, OTHER 31.8/47.3.

## Limits
- Repeat pairs only see errors that are not reproduced on the second tile; repeated systematic
  errors are measured by the VI and known-wrong rows instead.
- OTHER (secondary targets only) had no VI training rows: its scores are an extrapolation.
- P_WRONG is calibrated to the VI mix (11.7% wrong at per-night depth), not to main; use it as a ranking.
