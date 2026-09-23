# DESIGN — a foundation model that reads a spectrum and returns type and redshift

Every number in this file is measured on DESI DR1 in `METHOD.md`, on **1,100
spectra read blind across eighteen cohorts** plus the larger offline studies. The
design is almost entirely determined by four of them; the architecture is the
easy part.

This is meant to be built from. Where a section states a constraint, the
measurement that forces it is on the same line, so a decision can be reopened
by taking a new measurement rather than by argument.

**Revision history is the point.** Each version changed because of a
measurement, not a preference:

| version | what changed | the measurement that forced it |
|---|---|---|
| v1 → v2 | the class stopped gating the redshift search (§3) | type right → 1.6% redshift failures, type wrong → 70.0%, over 449 claims |
| v2 → v3 | no width-based class prior (§3); alias negatives span classes (§4); refusal becomes a calibrated probability, not a gate (§5) | a narrow 18.4σ C IV read as [O II]; a Lyα forest edge outscoring broad lines 7 times in 11; refusal measured at 1 saved / 1 lost |
| v3 → v4 | the reader's own feature positions become an input to the search, not just a check (§3); refusal is decided on **corroboration**, not on the strongest single peak (§5) | a correctly-read 4000 Å break discarded by the fit (Δz = 0.75); refusal now measured at **2 saved, 3 lost**, and all three losses were several weak lines agreeing |
| v4 → v5 | **abstention removed.** Always answer; mark a low-confidence answer as a guess and report guesses separately (§5) | replaying cohort XVI with its three refusals turned into marked guesses gives **49/50 redshift and 50/50 type**, against 48/50 and 47/47 with abstention. All three guesses were right |
| v5 → v6 | the guess flag becomes the place where every weak discriminant is spent, because marking is free while refusing is not (§5) | cohort XVII, first run under the new policy: **unmarked answers 46/46 redshift and 45/46 type**, marked answers 3/4 and 2/4. The mark carried three of the four errors |
| v6 → v7 | an **evidence floor** enters the confidence head as a *global* property of the spectrum, and the posterior's **width** becomes a scored output, not only its peak (§4, §5, §6) | OBJ045 of cohort XVIII contains nothing above 3.9σ anywhere; its answer and the reference are reachable through 2.1σ and 2.5σ. Over **770 claims across 16 cohorts**, <4σ is 9/14 right and ≥6σ is 665/682 (97.5%) |
| v7 → v8 | the class **stops selecting the template set** at inference too, not only in the loss (§3); and the reported class follows the evidence only when the evidence is **decisive by a factor of 3** (§5) | scoring every family on one shared candidate list over 506 objects: **overall 480→481/506, but 7/22→13/22 on the objects whose class was called wrong**. Letting the reported class follow the fit at ×3 is **+4/−0 on the 5-way against the DESI target class**; at ×1 it is +11/−11 |
| v8 → v9 | the class output is **five-way at the point of reading**, not a three-way call plus a redshift cut (§1, §3) | the eye emits BGS/LRG directly; rescoring the same cohort-19 calls under the five-way vocabulary gives **45/50 against 44/50**, but the one object that moved is a star mislabelled LRG by the target bits, so on the 19 real galaxies the eye and the z < 0.42 cut agree everywhere at 18/19. n = 20: a procedure change, not evidence |
| v9 → v10 | evaluation reports **two** class scores, because the DESI target label is not always a single class (§6); and the BGS/LRG ceiling is raised by giving the class head **brightness as well as redshift** (§1, §3) | **13.6% of the 822 objects scored so far carry more than one class bit**, and 17 of my 82 type "errors" name a class the object actually has: 740/822 against the drawn bit, **757/822 against any bit it carries**. Separately, the z<0.42 cut is within 0.5 pts of the best possible z-only rule (96.4 vs 96.97%), but z + median flux reaches **97.71%** |
| v10 → v11 | **nothing is learned from Redrock output** (§0, §6b, new §6c). `SPECTYPE` and `SUBTYPE` stop being labels, `DELTACHI2`/`CHI2` stop being loss inputs, and the one remaining dependence — Redrock's `Z` as the redshift label — is stated as an open problem rather than assumed | three enforced rules had come to rest on `SPECTYPE`: the below-floor guess statistic, the `TYPEFAC` justification, and the class-head label. All three survive re-derivation from the imaging target bits, and the floor statistic gets **stronger**: below 6σ, 81/93 carry a galaxy bit (87%) and **none carries a star bit**, against a 71% base rate |
| v11 → v12 | a **scaling-law plan** with three costed runs, and the finding that this problem is **data-limited, not compute-limited** (new §9); an ELG call is now checked against the **continuum at its own fitted redshift** (§3, §5) | `C = 6ND` verified for this architecture (encoder 99.6% of FLOPs, component sum / 6ND = 1.00). Run C is **11.4 h on 4× A100 and already 78% of all DR1 spectra in one epoch**. The ELG-break rule measured on 184 ELG calls: at ≥6σ it fires 6 times, **5 saved / 1 broken**, and cohort XX went 42 → 43/50 |
| v12 → v13 | the encoder's residual block becomes **ConvNeXt-style** — depthwise k=7 → LayerNorm → 4× inverted bottleneck — with no downsampling and no patching (§3) | measured, 27 runs, 3 arms × 3 sizes × 3 seeds on §7's label-free shift task. At matched parameters ConvNeXt beats the dilated stack **5.9σ at 100k and 5.5σ at 300k**; the 30k point has 2sd = 5.99 nats and cannot resolve anything. My own objection — that `LayerNorm` destroys the amplitude — was **refuted**: swapping it for BatchNorm is *worse* at both informative sizes |
| v13 → v14 | **measured, not planned.** The data ladder is run and fitted (§9.6–9.8); the correlation surface is **normalised** and the class logit loses a constant offset (§4); the reranker gets a **final LayerNorm** and training aborts on a diverging rank loss (§3, §4); the grid's NMAD floor is corrected (§9.4c); the E ≥ 1.07 prior is withdrawn (§9.5); the ranking premise is re-measured on the trained surface (§3) | four points at fixed size give val z loss **1.3914 / 1.0358 / 0.8385 / 0.7376** at 30k / 100k / 300k / 1M. The floor form's 1M prediction, registered before the run, missed by **0.007**, a pure power law by **0.093**; E ≈ **0.60**, 18M ≈ **0.63**. Twice before that the ladder appeared to bend because a loss term nobody was fitting had diverged (rank loss **46** at 300k, **136** at 1M). On the trained surface **68.8%** of catastrophics are ranking failures, not 85–89% |
| v14 → v15 | the training labels are **audited against physics** and the audit is **verified by Gaia astrometry**, independent of both the spectrum and Redrock (§6b, §6c, new §6d); the full target bitmask is rejected as a training label and kept as a label-reliability signal (§1, §6d) | over all 19,649,702 rows, **342,851 single-class labels (1.94%) sit outside the redshift range their own target class can occupy**. For the 2,022 `MWS_WD` targets labelled z > 0.1, Gaia gives **99.9% star-like** against **0.6%** on a known-quasar control and 99.7% on a known-star control — a star cannot be at z = 1.5, so those redshift labels are wrong, proven without a spectrum. Sub-bits flag only the stellar case: `MWS_WD` is enriched **26.8×** among labels at 1.45–1.70, while no sub-bit exceeds 1.5× for the suspicious ELG, QSO, LRG or BGS groups |
| v15 → v16 | the **goal is stated — surpass Redrock** — and evaluation splits into **agreement** on a corroborated set, **detection** on a proven-wrong set, and a **net count on blind-read disagreements** (new §6e); proven-wrong labels leave the *z* loss and become the detection test set; the model gains a detection output, `P_model(z_Redrock)` with a reason (§4, §6, §6e) | the 1M disagreement rate with Redrock is **1.09% ± 0.05%**; Redrock's own error rate on `ZWARN = 0` references is **2/200 = 1.0% (Wilson 0.27–3.57%)**. They are indistinguishable, so the headline number can no longer tell a better model from one that copies Redrock's mistakes |
| v16 → v17 | the model is built in **three stages defined by what labels each may touch**: label-free pre-training (known shift, masked reconstruction, evidence-map prediction; may include `ZWARN ≠ 0`), post-training (supervised → self-distillation from depth → calibration and detection), fine-tuning (heads only) (§7); the reranker reads **evidence at each candidate z** (§3, §5); an implementation-status table and the ±50 input-clip defect (§2) | §9: the problem is data-limited, and pre-training is the only part of the design free of the Redrock-label problem. Evidence A/B at 300k, two seeds, the same 50,048 spectra: catastrophic **1.169/1.181% → 1.089/1.117%**, **229 saved / 157 broken, McNemar p = 2.9 × 10⁻⁴** — the pre-registered rule says HELPS. By class: QSO −13% (38/14), LRG −10% (23/12), ELG only −4% (107/87) — the reranker features do not fix ELG, which is why alias negatives are the next post-training step |

---

## Implementation status — designed is not built

Added 2026-09-22 after the owner found that the reader's lessons were in this
file but not in the code: the trained reranker saw the class, z and nine numbers
of `S`, and the loss was plain NLL + BCE + rank. Nothing tracked design against
code, so this file behaved like prose, and prose corrections are the kind that
recur (3× and 11×, against 0× for enforced ones). `zfm/design_status.py` holds
this table and `train.py` prints every unbuilt item at startup.

| design item | § | code | status |
|---|---|---|---|
| class is a marginal of one joint surface, never a gate | 3 | `model.Surface` | built |
| ConvNeXt encoder, no downsampling | 3 | `model.Encoder` | built |
| normalised correlation surface; class logit without the ln N_Z offset | 4.6–7 | `--norm_corr` | built |
| reranker final LayerNorm; abort on diverging rank loss | 3, 4 | `--rr_final_norm` | built |
| reranker reads **what else is true at each candidate z**: the 4000 Å break, every line's narrow and broad σ, corroboration counts, the strongest unexplained peak, the evidence floor, brightness | 3, 5 | `evidence.py`, `--evidence` | **built 2026-09-22**; validated against `speclib.py` (break r = 1.0000, line r = 0.9946). **A/B at 300k, two seeds: HELPS** (pre-registered, `results/prereg_ev.txt`) — catastrophic 1.169/1.181% → 1.089/1.117%, **229 saved / 157 broken, McNemar p = 2.9 × 10⁻⁴**; QSO 38/14, LRG 23/12, ELG 107/87 |
| dense z loss over the whole grid | 4.1 | `train.py` | built |
| multi-label BCE on the target bits | 4.5 | `train.py` | built |
| cross-class alias negatives | 4.2 | `alias.py`, `--alias` | **built 2026-09-22**; 30k check: trains (margin 1.26 → 0.009) but does **not** move the targeted alias share — see §5 |
| surface recall loss: the truth must beat the K-th peak | 3 | `--recall` | **built 2026-09-22**; 300k A/B: **no effect** (1.127% → 1.132%, 143 saved / 147 broken, p = 0.86) and the targeted share did not move (truth proposed 74.0% → 72.9%). The term is a **mean over objects and 98% already satisfy it**, so it was 0.002 with a negligible gradient while the 2% tail carries no weight. It did improve the surface — z loss 0.8513 → 0.8253, rank 0.550 → 0.466 — just not where the catastrophics are. Fixed the same day (`--recall_hard`, mean over violators only): the term then bit exactly the right tail — violator fraction 0.1–0.3% in training, matching the 0.3% of objects whose truth is not proposed — and **still moved nothing**: 1.127% → 1.160%, 137 saved / 169 broken, p = 0.076, truth proposed 74.0% → 74.3%. **Recall is limited by information, not by the loss**: on non-alias catastrophics the truth carries **2.43σ** at its own redshift, so there is no signal at the truth to push above the threshold. Both versions improved the fit without touching the tail (z loss 0.8513 → 0.8253 / 0.8405, rank 0.550 → 0.466 / 0.505) |
| penalty for a predicted line that is absent, or a ≥ 8σ feature left unexplained | 4.3 | — | **not built** as a loss; the reranker receives the information through `--evidence` |
| per-candidate continuum shape (rest-frame band fluxes) | 5 | — | **not built** — the measured next step for alias errors |
| bounded (ε-mixture) z loss | 4.4, 6e | `train.mix_nll`, `--eps` | **built and A/B'd at 150k: no detectable effect; default ε = 0** |
| proven-wrong labels leave the z loss | 6e | — | **not built** |
| reader-asserted feature positions enter `S` as a prior | 3 | — | **not built** |
| confidence output: evidence floor, posterior width, corroboration count | 5 | — | **not built** as outputs; available only as reranker inputs |
| class head sees brightness | v10 | — | **not built** for `logit_c`; the reranker sees `log_median_flux` |
| detection output `P_model(z_Redrock)` with a reason | 6e | `teacher.py` (deep-tail teacher, ≥ 10σ) | **prototype measured 2026-09-22**: blind reading of 100 flags, teacher right 36/45 decidable disputes (80%), Redrock wrong on ≥ 36/100 (≥ 36× the base rate); 28 of 41 undecidable flags were sky residuals — see §6e |
| sky/artifact mask in the evidence maps | 3, 6e | `evidence.spectrum_maps(sky=0.3)` | **built 2026-09-23**; used for the detection flag (224 → 170; out of sample Redrock wrong on ≥ 46/99, teacher right 78% of decidable disputes). **Not for the reranker**: retrained with masked features, 154 saved / 190 broken, p = 0.059 |
| eval refuses a headline without the stratification table | 6 | — | **not built** |
| encoder input without the ±50 clip | 2 | `ZFM(softclip=True)`, `--softclip` | **built and A/B'd, no effect; clamp kept** |
| label-free pre-training: known shift, masked reconstruction, evidence-map prediction | 7a | — | **not built** |
| self-distillation from depth (SV3 degraded to main depth) | 7b-2 | `deep_tail.py`, `teacher.py`, `--deep_frac --degrade --teacher` | **built 2026-09-22; neither degradation (182/222) nor teacher labels (130/150) help at 150k** — §7b-2 |
| confidence calibration on a held-out set | 7b-3 | — | **not built** |
| fine-tuning with a frozen encoder | 7c | — | **not built** |

---

## 0. The four measurements that determine the design

| # | measurement | consequence |
|---|---|---|
| 1 | **0.0% of errors fall within 5σ_z**; 85–89% are the templates preferring the wrong peak | not a regression problem. Identification among discrete hypotheses |
| 2 | a same-shape MLP probe scores **0.17** where the trained head scores **0.006**; the old loss supervised **0.7%** of positions | the representation is not the bottleneck. The head and the loss are |
| 3 | additive per-line scoring lets a one-line explanation beat the whole comb (**+43.6 pts** over null), and it **does not scale** (rays α≈−0.25 against −0.73 globally) | the scoring function must be non-additive. More data will not fix it |
| 4 | **type right → redshift wrong 1.6%; type wrong → 70.0%** (449 claims, 43×) | the type must never gate the redshift search |

---

## 1. Output parameterisation

**Five classes: STAR / BGS / LRG / ELG / QSO, emitted by the class head as a
set.** BGS, LRG and ELG are *selection* categories rather than spectral ones,
and the `z < 0.42` cut still scores 92.6% against 75% for reading BGS/LRG off
the break. But a cut applied after z makes the class a deterministic function
of the redshift and cannot put mass on both classes where they overlap, so it
survives only as a **baseline to beat**, never as a component (§5, *The class
output is five-way*; v8 → v9). The head is **multi-label**, because 13.6% of
objects carry more than one class bit (v9 → v10).

**The class labels are the DESI target bits and nothing else** — never
Redrock's `SPECTYPE` (§6c; v10 → v11). As built, STAR is `MWS_ANY` (bit 61),
BGS bit 60, LRG 0, ELG 1, QSO 2. The bits are imaging selections with measured
defects — **~4% of QSO targets are stars**, and in one 50-object cohort it was
5 of 10 — and those defects are properties of the label to be modelled and
reported (§6 scores against the drawn bit and against the full mask), not a
reason to avoid the bits.

*Superseded here:* earlier versions of this section said there were three
classes only, that BGS/LRG/ELG should be derived by rule after z, and that the
target bits must never be trained on. The versions cited above replaced all
three; this paragraph had not been updated to match.

**Redshift as a dense posterior over a log(1+z) grid**, 40 km/s per bin
(the DESI pixel at 6000 Å). **As built: 12,140 bins over z ∈ [−0.01, 4.0]**
(`zfm/grids.py`). The lower edge is negative on purpose: stellar labels have a
median of −8 km/s and 56% are blueshifted, so they must not be clipped at
z = 0. The reported z is the **centre** of the chosen bin, so even a perfect
choice carries a rounding error of NMAD 4.945 × 10⁻⁵ (§9.4c). `argmax` is the
answer; the whole curve is the diagnostic — aliases, double peaks and the
single-line ridges are visible in it directly.

**Abstention is a first-class output**, not a threshold on softmax. See §5.

---

## 2. Input

Two channels, ≈7525 points: **`flux × √ivar`** and **`√ivar`**.

`ivar` is not optional. A passive galaxy's best single feature is **3.9σ**; its
redshift exists only in the coherent sum, so the normalisation must be by
*noise*, not by signal. A flux-only model cannot read an LRG.

The only preprocessing is a coordinate change: official BRZ coaddition, then
resample to a log-λ grid so that **redshift becomes a translation**.

Explicitly **not** in the input: target bits, photometry, magnitude, TSNR2,
fibre, survey/programme. All of these encode the *selection*, and a model that
learns the selection fails when the selection changes — measured: `--zmax 1.6`
turned the QSO target class from 20/20 real quasars into 9/20 and dropped the
score six points for reasons that had nothing to do with the reader. Resolution
matrix / LSF is also out: measurement 1 says precision work has nothing to gain.

**Measured defect, not yet fixed: the ±50 clip destroys bright stars.** `data.py`
clipped both channels to ±50, which bounds `flux × √ivar` at 50σ per pixel. Over
95,972 main spectra (`zfm/clip_audit.py`), **24.6% of STAR targets have more than
1% of their pixels clipped and 11.0% more than half**; for a bright star the
encoder sees a flat line at the clip. Galaxies are untouched (BGS 0.53%, LRG, ELG
and QSO ≤ 0.21%), which is why no redshift metric showed it: stars are the
easiest class to put at z ≈ 0 even with half the spectrum gone. Since 2026-09-22
the clip lives in `ZFM.forward`, in front of the encoder only — bit-identical
encoder input (`ev_identity.py`: max difference 0.000) — so `evidence.py` reads
the whole spectrum. Replacing the clip (e.g. a per-spectrum scale) changes the
encoder's input and needs its own A/B.

**A/B, 2026-09-23: the clip damages the input but not the answer.** `--softclip`
gives the encoder `50·asinh(x/50)`, which is within 1% of the identity below
|x| = 12, monotone, and keeps a bright star's shape (2460 → 229 instead of 50).
The transform is a checkpoint buffer, so every loader applies it without being
told, and old checkpoints load clamped, bit-identically. Pre-registered in
`results/prereg_clip150k.txt`: 150k, `--evidence`, d_rr 320, seeds 42 and 43,
one GPU per arm, 99,424 validation spectra.

- **Redshift: no detectable effect.** Catastrophic 1.240 → 1.257% (seed 42) and
  1.278 → 1.283% (seed 43); pooled **368 saved / 390 broken, p = 0.45**. The
  validation z loss moved −0.009 and +0.008, inside the 0.02 guard.
- **Class: not established.** Seed 42's clamp arm had a **collapsed class head**
  (`class_any_bit` 19.7% against about 84% for the other three), which alone
  produces a spurious 4912 saved / 14 broken. The pre-registered rule is met on
  paper, but only through that broken baseline, so it is not taken. The clean
  seed, 43: on the 5,197 spectra with more than 1% of pixels clipped, top class
  95.29 → 95.52%, 16 / 4, p = 0.012, short of the 0.01 threshold; clipped STAR
  targets 99.96 → 99.88% (0 / 4); unclipped spectra 83.88 → 80.94% (954 / 3672),
  which is class-head instability between runs, since their inputs barely change.
- **Why the defect costs nothing.** A spectrum pinned at +50 is itself the
  signature of a very bright object, and very bright objects are stars: the
  clamped model calls **99.96% of clipped stars STAR, with 0% catastrophic z**.
- **Flag: no difference, and my §6e attribution was wrong.** On the same 150k
  main rows the clamp model flags 279, softclip 258 (219 shared), and flagged
  rows with more than 1% clipped pixels are 12 vs 15. OBJ085, OBJ135 and OBJ136 —
  the three bright stars `EV_300000` misplaced — are put at z = 0 and **not
  flagged by either arm**, clamp included. Their false flags were an error of that
  one checkpoint, not of the clip.

Decision: **clamp kept** (pre-registered: no detectable effect → do not adopt).
The softclip code stays available. The finding that matters more is the class
head: **one seed in four collapsed**, so any class-metric comparison must first
check `class_any_bit` in every arm.

---

## 3. Architecture — and the one thing v2 changes

```
spectrum (2 × 7525, log-λ)
      │
      ├─► 1D ConvNeXt-block encoder (14.24 M as built, shift-equivariant, NO downsampling)
      │
      ├─► learned rest-frame archetype bank, per class
      │        └─ FFT cross-correlation over all shifts
      │           ÷ ‖e‖·‖w‖, × learned temperature (§4 item 6)
      │
      ▼
   S(class, z)   ← a 5 × 12,140 score surface, ALL classes, ALL redshifts
      │
      ├─► marginals:  P(class) = f(S)      P(z) = f(S)
      │
      └─► top-K peaks of S over the WHOLE surface (K = 256, classes mixed)
                 └─► reranker over candidates (d = 320, L = 8, 9.87 M, pre-LN + final LayerNorm)
                        └─► class · z · significance · n-corroborating-features
```

**v1 had the class select the family and then fit z inside it. That is wrong,
and it is the most expensive thing in the whole design.** Measurement 4: the
gate multiplies the redshift failure rate by 43×, and two thirds of all
redshift failures are downstream of a type error. So:

> **`S` is computed for every class at every redshift, and the class is a
> marginal of the same surface that produces `z` — never a switch in front of
> it.**

Ungating is necessary and **not sufficient**. Run on the five failures of the
most recent cohort it recovered only 2 of 5: for two the correct redshift was
never among the candidates, and for one χ² preferred the wrong family even
though the right one reached the right answer. With a shared candidate set the
failure simply moves from the gate to the ranker — which is where 85–89% of
errors lived on the previous architecture. On this one, measured on the trained
surface, it is **68.8%**, and 31.2% of catastrophics never reach the ranker at
all (see *Re-measured on the trained surface* below). Both halves have to be right, which is what §4 is for.

### The reader's measurements must enter the search, not just audit it

`measure3.py` requires a per-object record of the wavelengths worth measuring
and refuses to run without it — but it only ever *checks* that record. The
redshift search generates its own candidates and the recorded positions are
discarded.

Cohort XVI made the cost visible. OBJ028 has nothing above 4σ anywhere; its
redshift is carried by the 4000 Å break alone. The image pass recorded
"break-shaped rise 7300–8500", and at the reference redshift z = 0.9369 the
break sits at **7748 Å — inside that window**. The type call was right as well.
The fit still answered z = 0.1828, Δz = 0.75, because nothing in the pipeline
can use a position the reader measured.

So in the model, the reader's role has an architectural counterpart:
**positions asserted by the front end must enter the score surface as a prior
over z, not as a post-hoc audit.** Concretely, a measured feature position
contributes a term to `S(class, z)` for every `(line, z)` pair that would place
a line there — which is a soft constraint, not a gate, so a wrong reading costs
margin instead of the answer (the same principle as §3's headline). This is the
one place where the human-facing procedure and the model design have been out
of step: the procedure produces evidence the pipeline is not allowed to consume.

### Line width is not a class feature

Cohort XV supplied the cleanest possible case. OBJ009 has exactly one
measurable feature, **18.4σ at 3881 Å**, and it is **narrow**: 18.4σ on the
70 km/s filter against 7.9σ on the 3000 km/s one. It is C IV 1549 in a faint
quasar. Reading it as nebular put the object at z = 0.04 instead of z = 1.50 —
[O II] 3727 and C IV 1549 differ by a factor of 2.406, so the same pixel is
either redshift depending only on which family is allowed.

So the model must not carry **narrow ⇒ galaxy** or **broad ⇒ quasar** as a
prior anywhere: not in the architecture, not in the loss, not in a feature.
Width is evidence about the *line*, never about the *class*. The same holds at
the other end of the redshift range: at z > 2 the **Lyman-α forest edge is a
break**, and the coded comparison "broad must exceed the break" would have
rejected **7 of 11 correct quasar calls** in cohort XV. A step is a step; only
what else sits at the same redshift says what produced it.

Both are the same statement, and it is the reason for the joint surface above:
**class is decided by coherence across the whole spectrum at one redshift, not
by any local property of a feature.**

### Why a CNN and not a pixel-level transformer

- Measurement 2 says the encoder is not the bottleneck; spending capacity there
  is the move that has lost every time in this project's record.
- Redshift is a translation in log-λ. A CNN gets equivariance from its
  structure; attention has to learn it from data, and measurement 3 says this
  failure mode does not improve with data.
- Feature widths span 2–3 pixels (narrow line), ~75 pixels (broad line) and
  several hundred (the 4000 Å break, which is a *shape*: a local-step filter
  finds it 1.2% of the time, while matching the whole continuum by eye beat it
  8:1). A fixed patch size is wrong for all three at once.

### Which residual block — measured, not argued

The block inside that stack is **ConvNeXt-style**: depthwise k=7 → `LayerNorm` →
4× inverted bottleneck, with **no downsampling stages and no stem patching**.

27 runs — 3 arms × 3 data sizes × 3 seeds — on §7's label-free shift task, at
matched parameters (1.57 M vs 1.62 M, so matched FLOPs too):

| validation loss [nats] | dilated k=5 | **ConvNeXt** | ConvNeXt + BatchNorm |
|---|---|---|---|
| 30 k | 9.474 ± 3.354 | 11.350 ± 2.580 | 2.088 ± 0.703 |
| 100 k | 1.600 ± 0.315 | **0.273 ± 0.022** | 0.631 ± 0.078 |
| 300 k | 0.558 ± 0.051 | **0.295 ± 0.045** | 0.348 ± 0.056 |

ConvNeXt wins by **5.9σ at 100k and 5.5σ at 300k**. The 30k row is not evidence
against: its pooled 2sd is **5.99 nats**, so it cannot resolve a difference
smaller than six nats — 374 optimiser steps is not a converged model for any arm.

**Two of my own arguments against ConvNeXt were wrong and one was not tested.**

* *Downsampling breaks translation equivariance* — correct, and the reason the
  four ConvNeXt stages are **not** used. Only the block is adopted.
* *`LayerNorm(dim)` normalises across channels at each position, destroying the
  per-position amplitude, and amplitude is measured evidence (median flux
  separates ELG from BGS with zero overlap over 120 objects, worth 0.74 points
  on BGS vs LRG)* — **refuted.** The third arm exists to test exactly this, and
  BatchNorm is *worse* than LayerNorm at both informative sizes (0.631 vs 0.273,
  0.348 vs 0.295). It only wins at 30k, where nothing is resolvable.
* *The encoder is not the bottleneck anyway* — **not tested, and still true.**
  See the limits below.

**What this measurement does not establish.** All three arms reach 0.997–0.998
exact-shift accuracy and the no-encoder null already reaches **0.763**, so the
task is saturated and the loss difference is a difference in the *calibration*
of the correlation peak, not in the ability to read a spectrum. Measurement 2
says 85–89% of redshift errors are ranking errors and that recall headroom is
~1.5 points, so this improves **the component with the smallest measured
headroom on the one axis that has been measured** — which is not the same as an
unimportant component, since the encoder also shapes the surface the reranker
ranks on. **ConvNeXt is adopted
because at equal cost it is measurably better on the one thing that could be
measured — not because it is shown to improve a redshift.**

Recorded honestly: the pre-registered rule (`ab/PREREG.md`) demanded the same
direction at all three sizes and therefore returned NO CHANGE. That rule was
badly written — it treats a zero-power measurement as a contradiction rather
than as an absence of evidence — and the decision to adopt anyway was taken
knowing that, not by rewriting the rule.

### Where the attention goes, and why

On the **candidate peaks**, not the pixels. 85–89% of errors are "the right
answer is among the candidates and is ranked wrong"; improving stage-1 recall
caps at ~1.5 points. It is a ranking problem.

**Read that claim narrowly.** It licenses "recall has little headroom". It does
*not* license "the encoder does not matter": the encoder also produces the
`S(class, z)` surface the reranker ranks, so a better-separated surface makes
ranking easier and the encoder therefore acts on both stages. Both numbers were
also measured on the **previous** architecture, not on this design, and are
carried over as an assumption rather than a result. One token per candidate
`(class, z)` carrying its residual, its explained and **unexplained** lines and
its per-line velocity offsets, so candidates can be compared against each other
— which is the only place the non-additivity of measurement 3 can live in the
architecture rather than only in the loss.

### Re-measured on the trained surface

Measured 2026-09-09 on the D = 100k checkpoint (before surface normalisation),
30,000 unseen validation spectra, a candidate counted correct within 8 bins
(≈1.1 × 10⁻³ in Δz/(1+z)):

| | recall@256 |
|---|---|
| all objects | 98.95% |
| **catastrophic objects** | **68.8%** |
| non-catastrophic | 99.37% |

Re-measured on the D = 1M checkpoint with the normalised surface and the
reranker's final LayerNorm, the catastrophic figure is **80.7%** at the same
sep = 64, and **86.2%** at sep = 16 (below). The 68.8% belongs to the older,
un-normalised model and is kept as the historical value, not the current one.

So **68.8% of catastrophics are ranking failures and 31.2% were never proposed**.
A perfect reranker would take the catastrophic rate from 1.37% to **0.43%**; the
remaining 0.43% is unreachable from the reranker axis, which bounds what the
N-ladder can buy. Where the correct candidate sat when it was present: rank 2 in
25.2% of cases, top 10 in 55.3%, top 25 in 68.4%, top 100 in 88.3%. **K does not
shrink**: cutting to K = 25 would discard 31.6% of the recoverable errors.

Two cautions before reading the 31.2% as a surface failure:

* **The peak picker WAS erasing answers — measured.** `topk_peaks` blanks ±64
  bins around each peak, per class: 8.5 × 10⁻³ in Δz/(1+z), eight times the
  correctness tolerance and 13–32× a line width. Swept on the D = 1M checkpoint
  over 100,096 validation spectra (1,127 catastrophics), every value scored on
  the **same** objects with the catastrophic set fixed by the model as it runs:

  | sep | Δz/(1+z) | recall, all | recall, catastrophic | perfect-reranker floor | McNemar vs 64 |
  |---|---|---|---|---|---|
  | 4 | 5.3 × 10⁻⁴ | 99.750% | 80.7% | 0.22% | 82 gained / 82 lost, p = 0.94 |
  | 8 | 1.1 × 10⁻³ | 99.811% | 85.3% | 0.17% | 101 / 50, p = 4.7 × 10⁻⁵ |
  | **16** | 2.1 × 10⁻³ | 99.815% | **86.2%** | **0.16%** | **105 / 44, p = 8.9 × 10⁻⁷** |
  | 32 | 4.3 × 10⁻³ | 99.610% | 83.1% | 0.19% | 67 / 41, p = 0.016 |
  | **64 (in the code)** | 8.5 × 10⁻³ | 99.452% | 80.7% | 0.22% | — |
  | 128 | 1.7 × 10⁻² | 99.340% | 75.9% | 0.27% | 23 / 78, p = 7.7 × 10⁻⁸ |

  **sep = 16 recovers 5.5 points of catastrophic recall over sep = 64** and
  survives Bonferroni for five comparisons (needs p < 0.01) by five orders of
  magnitude; on all objects it is 414 gained against 50 lost, p ≈ 10⁻⁶³. sep = 4
  is a wash (82/82), so the parameter's purpose is real and only its value was
  wrong — a shoulder is a line width wide, not thirty. A paired test is required
  here: the binomial standard error on 293 catastrophics called the same effect
  p = 0.04, because it treats arms measured on identical objects as independent
  samples.

  **Retrained at sep = 16, and it collected none of it.** `sep` also selects the
  candidates the reranker is *trained* on, so the sweep above is a ceiling, not
  a gain. Two 300k runs, seed 42, back to back, everything else identical:

  | | loss_z | loss_rank | catastrophic | class |
  |---|---|---|---|---|
  | sep = 16 | 0.8403 | 0.5825 | 1.236% | 0.7983 |
  | sep = 64 | 0.8385 | 0.5697 | 1.172% | 0.8357 |

  Every metric is equal or worse; `loss_z` sits inside the 0.011 seed noise and
  catastrophic is +1.3σ. Both checkpoints record their own `cfg` sep, and the
  sep = 64 arm was a genuine rerun that reproduced the earlier 300k point to
  every printed digit. **`--sep` stays at 64.**

  **What the retrain actually changed was the ceiling, not the result.** Sweeping
  each of those two checkpoints over 100,096 spectra:

  | model | best sep | recall, catastrophic | perfect-reranker floor | realised catastrophic |
  |---|---|---|---|---|
  | trained at sep = 16 | 16 | **82.1%** | **0.23%** | 1.26% |
  | trained at sep = 64 | 16 | 79.0% | 0.26% | 1.24% |
  | trained at sep = 64 | 64 | 76.3% | 0.29% | 1.24% |

  Training at 16 lifted its own ceiling above anything the sep = 64 model
  reaches, and its realised error still did not move. **The gap between the
  floor and the realised rate is ~1.0 points in both** (1.03 and 0.95) — the
  reranker was handed more correct candidates and converted a smaller share of
  them. So the binding constraint is not recall; it is what §3 already calls
  ranking, and the 5.5 points is what the peak picker withholds, never an
  improvement available by changing `sep`.

  *(The two sweeps are not paired with each other: each model's own errors define
  its catastrophic set, 1,265 against 1,245. Only the within-table McNemar
  columns are paired comparisons.)*
* **Most catastrophics are not the classic line aliases.** Of 410: the model answers
  z < 0.05 in 19.5%, the label is z < 0.05 in 7.1%. Against a permutation null,
  six discrete ratios (1+z_model)/(1+z_rr) stand out — strongest 1.334 at 12.5σ,
  then 1.758 at 7.4σ — and hold **17.6%**. The other **55.8%** have no ratio
  structure beyond the null. Matching against a list of line pairs is not a test:
  240 pairs at 1% tolerance cover the ratio axis about five times, and the null
  already matches 89.7%.

### No tokenizer, no patching, no VQ

A spectrum is a continuous measurement with a known noise model, not a discrete
string. Quantisation preferentially discards low-amplitude structure — which is
exactly the evidence (the 3.9σ feature) — and a codebook has no way to carry
`ivar`. Patching breaks translation equivariance at sub-patch scale and
manufactures aliases, and aliases are already most of the error budget.

The only "vocabulary" is the archetype bank: continuous, learned, and
**capacity-limited on purpose** — too many archetypes and every redshift has a
template, and the aliases return. This is the main hyperparameter to sweep, and
it must be swept on a *stratified* metric (§6).

---

## 4. Loss

1. **Dense supervision over the whole z grid.** Measurement 2: the old loss
   supervised 0.7% of positions.
2. **Explicit alias negatives, including across classes.** For each object
   compute the redshifts at which a *single* line explains the data — the ridge
   positions — and supervise against them directly. **The alias set must span
   classes, not just line lists within one class.** The one type error of
   cohort XV was [O II] 3727 against C IV 1549 on the same pixel: a
   within-galaxy alias set would never have contained the competing
   explanation. Every (line_i, line_j) pair across the full rest-frame list
   generates an alias at z' = (1+z)·λ_i/λ_j − 1, and those are the negatives
   that matter.
3. **An unexplained-line penalty.** If the chosen `(class, z)` predicts a strong
   line the data does not have, or leaves a ≥8σ feature in the data
   unexplained, penalise it. This is measurement 3 attacked at its root:
   additivity is what lets one line outvote the comb.
4. **Bounded loss on the labels.** The reference is wrong about **1%** of the
   time (independently measured 2 in 200, and the ELG catastrophic rate falls
   1.4% → 0.45% under `DELTACHI2 ≥ 25`). Do not fit past that — mechanically,
   with an ε-mixture likelihood, and with labels proven wrong removed from the
   *z* loss (§6e). A model that fits Redrock past its error rate is learning to
   copy the errors it exists to catch.
5. **Fit the target bits as a set, and not past their measured error.** They
   are the class labels (§1, §6c): multi-label BCE against the bitmask, never a
   softmax over one arbitrarily drawn bit, because 13.6% of objects carry more
   than one class bit and all of them are valid. Like the redshift label in
   item 4, they are wrong at a measured rate — ~4% of QSO targets are stars,
   ~6–12% of targets disagree with what the object turns out to be — so a class
   head that reproduces the bits beyond that rate is learning the selection,
   not the spectrum. *Superseded:* this item used to read "Never fit the
   target bits", the same rule §1 dropped at v10 → v11.

### Measured in training: three ways the loss hid a fault (2026-09-09 to 09-15)

6. **Normalise the correlation before the softmax.** The surface was a raw dot
   product between encoder output and archetypes, so its sharpness followed the
   object's flux, not the quality of the match. The step-0 z loss came out
   **18.71** against the pre-registered ln 12,140 = **9.404** — exactly
   ln N + s²/2 with s = 4.15 measured on real spectra. `norm_corr` divides by
   ‖e‖·‖w‖ and applies a gain calibrated to start at 9.53 (`NORM_GAIN = 536.2`).
   At 300k it lowered val z loss from 1.0806 to 0.8552. The step-0 assert is
   bounded at [8.3, 10.7] from 10 seeds × 4 batches (mean 9.50, sd 0.20); an
   earlier [9.3, 9.9] built around one seed refused a healthy seed 43.
7. **The class logit is a mean over z, not a sum.** `logsumexp` over 12,140 bins
   carries a constant +ln N_Z = 9.40, so every class logit started strongly
   positive and the class loss began at 26.9 instead of ln 2.
8. **Watch every loss term, because the global clip couples them.** The reranker
   is a pre-LN transformer and was built without a final LayerNorm, so its
   residual stream was never normalised: 2.34 × 10⁴ at 300k and 3.90 × 10⁵ at 1M,
   against √d = 17.9. At 1M the rank loss rose from step ~16k to **136**, its
   gradient on the encoder was **578×** the z gradient, and clipping the total
   norm to 1.0 cut z's update to **4 × 10⁻⁴** of its size — z stopped learning
   halfway while its own loss looked ordinary. With a final LayerNorm the same run
   holds the rank loss at 0.31–0.72 through that region and ends at 0.42.
   `train.py` now aborts when the 200-step rank mean exceeds 2 ln K after step
   2000. **Open:** even the healthy 300k run clips every step by ~16× (factor
   0.061), so `max_norm = 1.0` is acting as a hidden learning-rate cut.

---

## 5. Confidence and refusal

Train the confidence head to predict the **matched-filter significance at the
model's own answer**. It is the only confidence signal that has survived
testing: monotonic at every depth (>10σ → ~99%, <3σ → ~17%), while a χ² margin
is 96.0% even at 60% coverage — a margin does not know whether it is right.

Two refusals are mandatory, both measured:

- **A line-class call with no line.** An ELG call whose strongest 70 km/s peak
  is below **8σ**: 5/14 right, against **76/76** above it, over 400 objects.
  Below the line the output must be a refusal, not a different class — "take
  the largest discriminant" scores **0 of 3** on those cases.
- **One line is not two.** A single line is right 6/7; two or more, 29/30. At
  cohort scale the same law: grouping quasar targets by how many broad UV lines
  fall in the window gives 4 lines **9/9**, 3 lines 13/14, 2 lines 3/5, 1 line
  **0/4**. The model must output the corroborating-feature count and abstain at 1.

**Refusal is not free, and the earlier version of this section said it was.**
It changes no *call*, so it cannot corrupt a right answer into a wrong one —
but it can withhold a right answer. Both cases have now happened, one per
cohort:

| cohort | why it was refused | truth | the fit would have been |
|---|---|---|---|
| XIV | bright, smooth, nothing over 4.2σ on four filters | STAR, z = 0 | **wrong** (z = 0.50) |
| XV | faint, strongest feature 5.4σ, in the OH forest | QSO, z = 2.222 | **right** (z = 2.210) |

Three more refusals in cohort XVI bring the tally to **2 saved, 3 lost**, and
the three losses share one mechanism:

| refusal | independent features agreeing at the fitted z | the fit | refusing was |
|---|---|---|---|
| XIV / OBJ031 | **0** | wrong | right |
| XVI / OBJ049 | **3** | wrong | right |
| XV / OBJ004 | **3** | right | a lost answer |
| XVI / OBJ011 | **4** | right | a lost answer |
| XVI / OBJ039 | **4** | right | a lost answer |

Every costly refusal was an object with **several individually weak lines that
agree on one redshift**. The 8σ test looks at the *strongest single peak*, so
corroboration is invisible to it — the single-line law (one line 6/7, two or
more 29/30) running in the direction nobody checked.

**Superseded, and the replacement is simpler: do not refuse at all.** Replaying
cohort XVI with its three refusals turned into *marked guesses* gives **49/50
redshift and 50/50 type**, against 48/50 and 47/47 with abstention, and all
three guesses were correct. The corroboration count above remains the right
*confidence* signal; it is no longer a gate.

The decisive case is OBJ049. Refusing routed it to a fit over **every** family,
where χ² chose z = 0.176 — wrong. Guessing QSO restricted the family and the
same machinery returned z = 1.761 against a reference of 1.759. **An
unconstrained fit was worse than a guessed constraint.** That is not in tension
with removing the type gate (§3); it is the other half of the same measurement:
a *wrong* type is catastrophic (70% redshift failure), a *right* type is worth a
great deal, and the way to have the second without risking the first is a soft
prior over a joint surface rather than a switch.

So the refusal head becomes a **guess flag**: the model always emits its best
(class, z), and separately emits how much corroboration that answer has. The
catalogue user chooses the operating point; the model never withholds.

### Marking is free, so every weak discriminant belongs in the flag

This changes what counts as a useful rule. A discriminant that separates 50%
reliability from 95% is worthless as a *gate* — it would block as many right
answers as wrong ones — but it is worth having as a *flag*, because the mark
changes the label and not the answer.

The worked case is the quasar branch. Over **101 QSO calls across twelve
cohorts**:

| | calls | right | reliability |
|---|---|---|---|
| broad ≥ narrow | 93 | 88 | **95%** |
| narrow > broad | 8 | 4 | **50%** |

As a refusal that rule blocks 4 right calls to stop 4 wrong ones — useless. As
a flag it fires on 8% of quasar calls and halves their reliability, at no cost.

Cohort XVII produced the error it is for, twice: a galaxy at z = 1.03 and one at
z = 1.34, each with a broad-looking **Mg II 2799** landing at 5700 and 6550 Å,
read as quasars. In both the narrow filter already outscored the broad one
(6.3 against 5.1, and 9.6 against 7.8) — the discriminant was right and the eye
overrode it. Both redshifts came out correct regardless, because Mg II sits at
the same redshift either way; only the type was lost.

**The model consequence:** the confidence head should not be trained on a single
scalar. It should receive every discriminant the front end computed — narrow
against broad, corroborating-feature count, break amplitude, median flux — and
learn which combinations are unreliable. Under a no-refusal policy there is no
reason to threshold any of them individually.

In the model this is not a threshold but the shape of the refusal head: it must
be fed the **corroboration structure of the chosen candidate** — how many
independent features the answer explains — and not a single significance. That
is the same quantity §3's reranker already builds per candidate, so the refusal
head reads it rather than recomputing it.

n = 5. The rule changes no answer that is currently given, so like the 8σ ELG
rule it cannot break what already works. Report refusals and claims separately
and never let a refusal score as a correct answer on a fit the model declined
to stand behind.

### The evidence floor is a property of the spectrum, not of the candidate

Every discriminant above asks *which* answer to prefer. OBJ045 of cohort XVIII
asks a prior question: is there anything here at all. Its strongest feature —
across the 70 km/s filter, the 3000 km/s filter and the local step statistic,
over the whole 3600–9824 Å window — is **3.9σ, and it sits at neither the given
redshift nor the reference one**. The answer that was written (z = 1.4301) rests
on 2.1σ; the reference (z = 0.2750) is reachable only through 2.5σ of narrow
emission and 3.4σ of Mg b absorption, both *weaker than a noise peak belonging to
nothing*. The type call was right and defensible — median flux 0.067 puts it in
the ELG band — but the redshift was a draw from the noise.

Measured over **770 claims across 16 cohorts**, binned on that global maximum:

| strongest feature anywhere | n | z right | rate |
|---|---|---|---|
| < 4σ | 14 | 9 | **64%** |
| 4 – 6σ | 74 | 63 | 85% |
| 6 – 8σ | 64 | 63 | 98% |
| 8 – 12σ | 100 | 99 | 99% |
| 12 – 20σ | 146 | 141 | 97% |
| > 20σ | 372 | 362 | 97% |

The floor is **global**: it does not depend on class, and it sits below the
per-class rules (the 8σ ELG line rule, the QSO width comparison), which all
assume a feature exists to argue about.

**How the threshold was chosen.** Under a no-refusal policy the marker's cost is
not lost answers — it is dilution of the mark. So pick the threshold on what the
*unmarked* group buys:

| flag at | marks | of the 33 wrong | unmarked accuracy |
|---|---|---|---|
| < 4σ | 14 | 5 | 96.3% |
| < 5σ | 50 | 13 | 97.2% |
| **< 6σ** | **88** | **16** | **97.5%** |
| < 8σ | 152 | 17 | 97.4% |

6σ is the smallest threshold at which the unmarked group stops getting purer;
8σ marks twice as many objects and buys nothing. The price is real and should be
stated: on cohort XVIII this marks 10 objects rather than 4, and 9 of the 10 are
right.

**Two model consequences.**

1. The confidence head's inputs cannot all be per-candidate. §3's reranker
   supplies corroboration *for a candidate*; the floor is a statistic of the
   **whole spectrum before any candidate exists** — the max of the filter bank
   over all wavelengths. Feed it in alongside, as a global token.

2. This failure is not a misread feature. It is a **point estimate demanded where
   the posterior is flat**, which is a fault in the output format. §1 already
   emits a dense posterior over log(1+z); the fix is to stop scoring only its
   argmax. Add a coverage term: the width of the smallest credible interval
   containing the truth must match its stated probability, and §6 must report
   median posterior width alongside accuracy. A model that answers "somewhere
   between 0.2 and 1.6" on OBJ045 is *correct*; one that answers "1.4301" is
   wrong even when the number happens to land.

n = 770 for the floor, n = 1 for the object. The rule changes no answer, so it
cannot break what already works.

### The class must not select the template set at inference either

§3 already says the class is a marginal and never a switch, and §4 supervises
it that way. The reader's own step 2 was still doing the opposite: the eye
called a family and the fit was then run **only on that family's templates**.
Cohort XIX removed it — every family is scored on one shared candidate list
built from every family's coarse correlation, so no region goes unexamined
because the called type had no reason to look there.

Measured on 506 objects across 15 cohorts:

| | all 506 | class call **right** (484) | class call **wrong** (22) |
|---|---|---|---|
| family-gated | 480 (94.9%) | 473 (97.7%) | 7 (**31.8%**) |
| every family scored | 481 (95.1%) | 468 (96.7%) | 13 (**59.1%**) |

**Overall the two are indistinguishable.** That is the point: the gate is
invisible in aggregate and decisive exactly where it bites. It is the
*mechanism* of the 43× lever — gating is what converts a class error into a
redshift error — and removing it rescues 6 of the 15 fatal ones.

The price is 5 objects whose class was right, three of them a star that the QSO
family explains at z ≈ 2.998. A veto on overrides landing near a family's
redshift-limit edge was written and tested to remove those: it does not, because
2.998 is mid-grid for the QSO family (limits −0.01 to 6.0). The preference is a
real property of the template set, not an optimiser artefact. **Rejected and
recorded** — the repair belongs in the template bank, not in a veto.

**The model consequence** is that §3's joint `S(class, z)` surface must be
evaluated over the **full** grid at inference, with the class prior added in
log space rather than used to mask rows. A masked row cannot be recovered by a
reranker; a down-weighted one can.

### When may the reported class follow the fit?

Ungating z raises a second question the reader had never had to answer: if the
fit prefers a different family, does the *label* change too? Measured by the
factor the fit must beat the call by:

| fit beats call by | 5-way vs DESI target class |
|---|---|
| never (call only) | 441/496 |
| ×4 | 444 (+3/−0) |
| **×3** | **445 (+4/−0)** |
| ×2 | 445 (+5/−1) |
| ×1 (always) | 441 (+11/−11) |

×3 is the largest factor that still gains and the only one that loses nothing.
Below ×2 the fit starts overruling correct calls — which is the
same asymmetry as everywhere else in this file: **the evidence should be allowed
to overturn the reading only when it is decisive, and to inform the redshift
always.**

**The model consequence:** the class marginal and the class *decision* are not
the same output. The marginal is read off `S(class, z)` at every step; the
decision commits only past a margin, and below that margin the reader's prior
stands. §5's guess flag is where the difference is reported.

### The two scores are different objects, and only one of them scores the reader

`DESI_TARGET` bits are imaging colours recorded **before** the spectrum exists;
Redrock's `SPECTYPE` is which template family fits **after** it. Five classes
against three, and SPECTYPE collapses BGS, LRG and ELG into GALAXY — so it
cannot score the distinction the 5-way is about, and the 5-way is only possible
against the target bits. Their agreement with each other (43–44 of 50 in recent
cohorts) measures DESI's target selection, not this reader, and does not belong
in a report of it.

For the model this is a **labelling** constraint. The standing rule for this
project is that **nothing is learned from Redrock output**, so `SPECTYPE` is not
a training label and not a source for selecting rules. The class labels are the
DESI target bits and nothing else — imaging colours recorded before the
spectrum exists — carried with their known defects: ~6–12% of bits disagree
with what the object turns out to be, and 13.6% name more than one class. Train
on the bitmask as a set, report against it, and never average a target-bit score
with anything else.

### The two failure modes are different problems (measured 2026-09-22)

`zfm/diag_alias.py`, 99,424 validation spectra, on the 300k evidence model
(1.127% catastrophic). Errors split by whether the chosen z reads a line that is
really present as a *different* line — `alias.py`'s own rule:

| | share of errors | truth among the K candidates | median rank of the truth | best line at the truth | at the chosen z |
|---|---|---|---|---|---|
| **alias** | 19.6% (ELG 26.7%) | **93.6%** (ELG **99.2%**) | 3 | 9.60σ | 9.81σ |
| **non-alias** | 80.4% | 69.1% (ELG 63.4%) | 14 | 2.43σ | 3.53σ |

1. **An alias error is not a recall failure and not a ranking-loss failure. The
   evidence is symmetric.** In 78.6% of them the truth and the chosen z show the
   **same number** of ≥ 6σ lines, and in 93.6% the truth shows **at most one**.
   One line explained as a different line looks the same from the lines alone —
   and the wrong one is marginally *stronger* (9.81 vs 9.60σ). Training can push
   the alias down because it knows the answer; at inference nothing distinguishes
   them. This is why the alias-negative loss trained (margin 1.26 → 0.009) and
   moved the targeted quantity **not at all** at 30k (ELG alias share 24.8% →
   25.9%; catastrophic 1.529% → 1.475%, 137 saved / 110 broken, p = 0.098).
   **What separates them is the continuum shape**, which is the other half of the
   reader's ELG signature (§5) and is not yet a per-candidate feature. Next:
   rest-frame band fluxes at each candidate z — a colour measured from the
   spectrum itself, needing no photometry and no selection quantity.

   **Built and tested, 2026-09-22 — it does not pay.** `evidence.shape_features`
   (`--shape`): 12 log-spaced rest-frame bands from 700 to 10,000 Å, their
   weighted mean flux, SNR and validity, plus a slope and a blue−red difference,
   38 features per candidate. A/B at 150k, seed 42, 99,424 validation spectra:
   catastrophic **1.227% → 1.261%**, **178 saved / 212 broken, p = 0.095** — no
   gain, if anything a loss, and QSO is the worst row (+10%, 17/36). The targeted
   quantity did not move either (ELG alias share 24.3% → 25.8%). The z loss is
   unchanged (0.9584 → 0.9588), so this is not collateral damage; the features
   simply do not earn their place.
   **Why it was expected to work, and what that taught instead.** On the 2,259
   real catastrophics of the two 300k checkpoints, a logistic regression on
   (features at the truth − features at the chosen z) separates them at **0.811**
   from evidence alone and **0.833** with shape, and on ELG alias errors shape
   alone (0.807) matches all existing evidence (0.807). So **the information is
   present and linearly separable, and the reranker still does not use it** —
   including the evidence it already had. Random alias pairs are not this test:
   there the lines alone score 0.984, because a random alias is an easy one.
   **Corrected the same day — that inference was wrong.** `diag_learn.py`, 80,128
   validation and 20,096 training-distribution objects: a linear probe trained on
   **training** data (truth against the runner-up, no selection on failures) and
   allowed to re-pick between the reranker's own top two is **worse** — 1.357%
   against 1.132%, 98 saved / 278 broken, p = 4.5 × 10⁻²¹. It disagrees with the
   reranker on 19.9% of objects and is right on 96.5% of them against the
   reranker's 97.7%. So the 0.83 was **post-hoc separability on a population
   selected by the model's own errors**, not an exploitable rule; separable after
   the fact is not the same as decidable in advance. Two further numbers from the
   same run: the model is **not overfitting** (1.319% on fresh training-distribution
   objects against 1.132% on validation — though these are not the objects it
   trained on, so this measures the distribution, not the fit), and its candidate
   scores are **mostly surface-driven** — the top-two logit difference is fitted
   at R² = 0.415 by the surface peak and Δz alone, 0.522 with every evidence
   feature added.
2. **A non-alias error is mostly an evidence-floor object.** The truth carries
   **2.43σ** at its own redshift — below §5's floor — and 31% of the time the
   surface never proposes it. These are not ranking failures at all; they are
   objects whose redshift is in the coherent continuum, and they bound what the
   reranker can do from above.

### An emission-line call must be checked against the continuum at its own z

Three of the five real class errors of cohort XX were the same shape, and it is
the third time this shape has appeared: the lines gave the right redshift, the
object was called ELG on those lines, and **the 4000 Å break at that very
redshift was never looked at.** The first two corrections were prose and did not
hold.

Measured on all 184 ELG calls across nineteen cohorts, 166 with a usable break
amplitude at their own fitted redshift:

| | median break at the fitted z |
|---|---|
| ELG calls that are ELG targets (141) | **1.25σ** |
| ELG calls that are not (25) | **2.64σ** |

Reassigning an ELG call to the break classes when that break exceeds a threshold:

| threshold | fires | not-ELG among them | saved | broken |
|---|---|---|---|---|
| ≥ 3σ | 23 | 10 | 10 | 13 |
| ≥ 4σ | 15 | 8 | 8 | 7 |
| ≥ 5σ | 9 | 6 | 6 | 3 |
| **≥ 6σ** | **6** | **5** | **5** | **1** |
| ≥ 8σ | 1 | 0 | 0 | 1 |

At 6σ all five reassignments land on the right class once the `z < 0.42` split
is applied; the one break is a genuine ELG target with an 11.8σ break at
z = 1.02. Net **+4** on the 5-way, and cohort XX went 42 → 43/50.

This is **not** a refusal — it swaps one positive answer for another — so the
"marking is free, refusing is not" asymmetry that governs §5 does not apply, and
the saved/broken count is the whole test.

**The model consequence:** the reranker of §3 must score each candidate on the
continuum *and* the lines at that candidate's redshift, not on whichever
produced the candidate. A peak found by a line detector and a peak found by a
break detector arrive at the reranker as the same object, and the reranker's
job is to ask what else is true at that z. This is the same requirement as §3's
"the reader's measurements must enter the search" — evaluated at the candidate,
not at the origin of the candidate.

### The class output is five-way, and BGS/LRG is not a post-hoc cut

Earlier versions had the reader emit STAR / GALAXY / ELG / QSO and assign
BGS versus LRG afterwards from the fitted redshift, on the measurement that the
`z < 0.42` cut scores 92.6% over 20,000 targets while the best spectral
threshold scores 73%. That measurement is unchanged and still in `METHOD.md`.
What changed is where the class is produced: **the class head emits five
classes, and BGS/LRG is one of its outputs, not a rule applied to another
output.**

The reason is structural rather than empirical. A post-hoc cut on `z` makes the
class a *deterministic function* of the redshift, which:

1. destroys the class marginal exactly where §3 needs it — an object at
   z ≈ 0.42 has a genuinely split class posterior, and a cut reports certainty;
2. cannot express the case the cut gets wrong. 7.4% of BGS+LRG targets sit on
   the wrong side of any cut, because BGS reaches 0.419 and LRG starts at 0.350.
   A head with a joint `S(class, z)` surface can put mass on *both* at the same
   redshift; a threshold cannot;
3. makes the class error and the redshift error perfectly correlated by
   construction, which is the opposite of what §3's decoupling is for.

The empirical result is a wash and should be reported as one: on cohort XIX the
eye scores 19/20 and the cut 18/20, and the single disagreement is an object
that is neither — a star at z = 0 that the LRG target bit selected. On the 19
that really are galaxies the two agree everywhere.

**The model consequence:** the class head has five outputs, trained on the DESI
target bits alone (§6b) — Redrock's `SPECTYPE` is not a label this project
learns from — and the `z < 0.42` cut survives only as a *baseline to beat* in
§6, never as a component.

### Where the BGS/LRG error actually comes from — three parts, measured

Asked whether the BGS/LRG accuracy drop is "mainly the overlap", the answer is
no: the overlap is the second-largest of three parts, and the largest is a
labelling artefact.

**1. The label is not always a single class.** `DESI_TARGET` is a bitmask, and
**550,088 of 8,828,736 BGS-or-LRG targets (6.2%) carry both bits**. The cohort
builder drew such an object for one class and scored against that one. Over the
822 objects scored so far, 112 (13.6%) carry more than one class bit, and **17
of the 82 type errors name a class the object actually has** — 13 BGS/LRG and 4
ELG/QSO:

| scored against | |
|---|---|
| the bit the object was drawn for | 740/822 = **90.0%** |
| any class bit the object carries | 757/822 = **92.1%** |

This also explains a discrepancy in `METHOD.md`: the `z < 0.42` cut scores
92.6% when both-bit objects are counted on both sides and **96.4%** on
BGS-only against LRG-only. Same cut, same data, different label convention.

**2. Genuine overlap in redshift.** On BGS-only against LRG-only with equal
priors, the cut scores 96.44% and **the best possible function of z alone is
96.97%** — so cut *placement* accounts for only 0.5 points and the remaining
≈3 points are irreducible from the redshift. All of it sits in one place:

| z | BGS | LRG | minority share |
|---|---|---|---|
| 0.00 – 0.30 | 4513 | 40 | 0.9% |
| 0.30 – 0.42 | 1204 | 105 | 8.0% |
| **0.42 – 0.55** | **247** | **554** | **30.8%** |
| 0.55 – 1.60 | 23 | 5284 | 0.4% |

**3. But "overlap" is a statement about z-space, and the spectrum has more
axes.** BGS is magnitude-selected and LRG is not: median flux **2.095 against
0.384, a factor of 5.5**. Adding it lifts the ceiling from 96.97% to
**97.71%** — about a quarter of the "irreducible" error was only irreducible
*given z*. (Measured on calibrated median flux, not `SNR_MEDIAN`: BGS is
observed in bright time and LRG in dark, so any signal-to-noise statistic
partly measures the programme, which §2 excludes on purpose.)

**Model consequences.**

* §6 reports **both** class scores — against the drawn bit and against the full
  mask — and never only the flattering one. A single number over a multi-label
  target is a category error.
* The BGS/LRG decision needs the **continuum normalisation**, not just the
  redshift. This is an argument for keeping `flux × √ivar` rather than a
  per-object standardised input: standardising away the amplitude throws out
  the axis worth 0.74 points here.
* The training loss should treat a multi-bit target as a **set**, not a class:
  cross-entropy against one arbitrarily chosen bit teaches the head that the
  other, equally valid, answer is wrong.

---

## 6c. The label problem, stated rather than assumed

This project does not learn from Redrock output. That rule is easy to state and
was not easy to keep: three enforced rules had quietly come to rest on
`SPECTYPE` before it was noticed. All three were re-derived from the imaging
target bits and all three survived — the below-floor statistic got stronger.

What remains is one dependence that cannot be re-derived, and it should be
written down rather than assumed away.

**The class labels are solved.** The DESI target bits are imaging colours
recorded before the spectrum exists. They are not Redrock output. They are
defective in known, measured ways — ~6–12% disagree with what the object turns
out to be, 13.6% name more than one class — and those defects are properties of
the label to be modelled, not removed.

**The redshift label is not solved.** A model that outputs a redshift posterior
needs redshifts to train on, and in DR1 at scale the only ones that exist are
Redrock's. Visual inspection is SV1-only with zero overlap with the samples of
interest. So there are four options and no free one:

| | what it costs |
|---|---|
| train on Redrock `Z` | inherits Redrock's failure modes, which is the thing the project exists to escape; every measured leak so far is a *confident* line misidentification, and those would be taught |
| self-supervised only (§7: predict the relative shift between two views of the same object) | needs no redshift label at all, but produces a *relative* shift, not an absolute z — it is a pretraining objective, not the task |
| train on the reader's own blind redshifts | 1150 objects. Correct at 96–98%, but three to four orders of magnitude too few |
| physical-model supervision — fit a line comb and a continuum family directly, scoring how well one z explains the whole spectrum | no external label at all, but it *is* a template fit, and an incomplete forward model measured worse than none |

The honest position is that **§7's pretraining is the only part of the design
that is fully free of the problem**, and the supervised head has to make a
choice that is not mine to make silently. It is written here so it is made
explicitly.

**Decided 2026-09-08: option A.** The project owner chose Redrock `Z` as the
redshift label for the supervised model. Everything in §9 is trained on it, so
every number there is a score **against Redrock**, and the model inherits its
failure modes. One measured instance: **11,996 targets carrying only the STAR
bit have `Z > 0.1`** (0.30% of 3,944,521), median z 1.53, 90th percentile 1.68 —
the z ≈ 1.45–1.70 zone where a `ZWARN = 0` reference already placed a white
dwarf whose Balmer series sat at rest. **Settled: those labels are wrong.** Gaia astrometry, joined on `TARGETID`,
gives 99.9% star-like (parallax or proper motion above 5σ) for all 2,022, against
99.7% on a known-star control and **0.6%** on a control of known quasars at the
same redshift. A Milky Way star cannot be at z = 1.5. The spectra alone could not
settle it — the Balmer-at-rest test fires on 51% of stars overall but only 5.4%
at SNR/px 2–5, so its 0/10 on the validation sample had P ≈ 16% even if all ten
were stars — and neither could the alternative that they are quasars: broad lines
at the label redshift appear in 0.3% of them at SNR 5–15, where real quasar
targets give 93.2%.
**Decided 2026-09-22: training scope z < 1.6.** The owner restricted training to
label z < 1.6 (`train.py --zmax 1.6`, filtered in `data.py`) after being shown its
measured cost on the 300k evidence model, 99,424 validation spectra: it removes
4.4% of objects but only **7.9% of ELG catastrophics** (ELG's rate is 2.48% below
1.6), and **33.2% of QSOs** — the easier third (2.0% catastrophic above 1.6
against 4.3% below). At inference the model still meets those objects. So the
validation stream is **never** filtered (the constructor refuses it), and every
evaluation reports z < 1.6 and all objects side by side. The better-targeted
alternative, a per-class physical-range filter on the labels (§6b item 4), was
proposed and remains open.

**Decided 2026-09-17: AION as a comparison baseline.** The standing rule is that no
Redrock or AION resource is used. The project owner lifted it for AION in one
scope: run the public `polymathic-ai/aion-base` checkpoint, spectrum only, over the
992,821-row SV3 `ZWARN = 0` file and compare its redshift with Redrock's
(`zfm`'s sibling `aion/aion_z.py`, `aion/aion_compare.py`). AION's outputs are a
**baseline to be scored**, like Redrock's; they are never a training label, an
input, or a source of rules for this model, and AION's method is not read. The
comparison inherits §6e's caveat: agreement with Redrock is not accuracy, for
AION as for this model.

Separately, of the 61 validation catastrophics where the model answers z ≈ 0
and the label is higher, only 5 are STAR-only targets — that failure mode is
mostly not mislabelled stars.

---

## 6d. How a label is verified

A label cannot be proved right at 19.65 M rows: there is no independent truth at
that scale. What can be had is a **per-population error rate with a confidence
interval**, and three routes give it. Every one of them needs a **positive and a
negative control measured on the same axis**, because a null result from a test
of unknown power says nothing — the 0/10 above is exactly that mistake.

| route | settles | cost | limit |
|---|---|---|---|
| **Gaia parallax + proper motion** (`zfm/gaia_check.py`) | is it a Milky Way star? | minutes | Gaia is magnitude-limited: 100% of the WD group has a solution but only 25.7% of the quasar control, and a *missing* measurement is not evidence |
| **template-free physics on the spectrum** | which redshift explains the lines | minutes | power depends on SNR — Balmer-at-rest fires on 5.4% of real stars at SNR/px 2–5 and 76% above 15 |
| **blind reading by eye** | anything, including class | hours | ~100 objects for ±10%, ~400 for ±5%, and the sample must be mixed with controls and unsealed only after scoring |

The Gaia route is the strongest available here because it touches neither the
spectrum nor Redrock. Recipe: join on `TARGETID` against
`/global/cfs/cdirs/desi/public/dr1/target/catalogs/dr9/1.1.1/targets/main/resolve/{bright,dark}`,
locating the file by `HEALPIX // 64` (the spectro grid is nside 64 nested, the
target files nside 8). Guard the join on the bits this project actually reads —
the five class bits and `MWS_WD` — which agree **100.00%**; demanding all 64 bits
refuses a correct join over `SCND_ANY`, which differs on 27 of 2,022 and is never
read.

**What the bitmask is for.** The full `DESI_TARGET` / `BGS_TARGET` /
`MWS_TARGET` / `SCND_TARGET` masks are *not* a better training label: most bits
describe the imaging region (`*_NORTH`, `*_SOUTH`), the fibre's purpose (`SKY`,
`STD_*`), the observing programme (streams, dwarf galaxies, `BACKUP_*`, all of
`SCND_*`) or a magnitude cut already carried by `flux × √ivar` — and predicting
them teaches the selection, which §2 rejects for a measured reason. They are also
not portable: six `desi_mask` bits and 37 `mws_mask` bits mean different things in
main and SV3, while the five class bits are identical. Their measured value is as
a **label-reliability signal**: `MWS_WD` is enriched 26.8× among STAR-only labels
at 1.45–1.70, `MWS_NEARBY` 19.6×, and Gaia then proves those labels wrong. For the
suspicious ELG, QSO, LRG and BGS groups no sub-bit exceeds 1.5×, so the signal is
specific to the stellar case. Use them to weight the loss (§4 item 5) and to
stratify §6's report — never to filter the validation set, which would remove the
hard objects and flatter every number.

---

## 6e. The goal: agree with Redrock where it is right, catch it where it is wrong

Stated by the project owner on 2026-09-17: **the model exists to surpass
Redrock.** That has two halves, and under option A (§6c) they pull against each
other:

1. **Agreement.** Where Redrock is right, the model's answer must not be far
   from Redrock's.
2. **Detection.** The model must be able to say of a given object that Redrock
   is probably wrong — and be right when it says it.

### Why the headline number can measure neither

Every catastrophic figure in §9 is a **disagreement rate with Redrock**
(|Δz|/(1+z) > 0.1 against the label). It mixes two things with opposite
meanings: the model's own errors, and Redrock errors the model declined to
copy. At 1M it is **1.09% ± 0.05%** (≈545 of 50,016). Redrock's own error rate
on `ZWARN = 0` references, measured independently of it, is **2 in 200 = 1.0%
(Wilson 95%: 0.27–3.57%)**. The two cannot be told apart.

So from here on:

- a model that copied Redrock perfectly would score **0.00%** and would have
  surpassed nothing;
- a model that was right on every object would score **≈1%**, not 0;
- a falling disagreement rate can mean the model is improving **or** that it is
  learning to reproduce Redrock's mistakes, and the number alone cannot say
  which.

**A disagreement rate with Redrock is not a measure of beating it, and must
never be reported as one.** It stays in §9 as a training diagnostic, labelled as
disagreement.

### Surpassing is a net count on the disagreements

Where the model and Redrock agree, replacing one by the other changes nothing.
Every consequence of using the model instead of Redrock lives in the
disagreement set, so the test is:

> **The model beats Redrock iff, among objects where they disagree, the model
> is right more often than Redrock.**

Among disagreements there are three outcomes — model right, Redrock right, both
wrong — and the net gain per object observed is
`P(disagree) × [P(model right | disagree) − P(Redrock right | disagree)]`.
This is the same fix/break count that caught this project's own regressions: an
evidence-rescoring change fixed 3/3 failures and broke 10/47 successes, a net
loss that no aggregate agreement figure would have shown.

It is also cheap. The disagreement set is ~1% of the data, so blind reading
(§6d route 3) concentrates entirely on the objects that matter: **100
disagreements resolve the model-vs-Redrock split to ±10%, 400 to ±5%.** Read
them blind as always — the reader returns an answer before either the model's
or Redrock's is unsealed — and add a matched draw of **agreements**, because
objects where both are wrong are invisible in the disagreement set and bound
how many Redrock errors the model copies.

### Three numbers, each on its own set

| requirement | measured on | metric |
|---|---|---|
| **agreement** | objects whose Redrock z is corroborated by a route in §6d that does not use Redrock | catastrophic and NMAD **against Redrock**, stratified as in §6 |
| **detection** | objects whose Redrock z is **proven wrong** independently — the 2,022 Gaia-star `MWS_WD` targets at z > 0.1 are the first such set; blind-read disagreements add more | recall of the flag at a fixed flag rate, and its precision against the ~1% base rate |
| **net** | a blind-read draw of disagreements plus a matched draw of agreements | model right − Redrock right, per object, with an interval |

Pass criteria are registered before the first measurement, not after. The
defaults proposed here:

- **agreement** — on the corroborated set the model's catastrophic rate must be
  **below half of Redrock's own error rate (< 0.5%)**. The arithmetic: if the
  model fixes a fraction f of Redrock's ~1% errors, it may break at most
  ~f × 1% of the ~99% it should leave alone, or the swap loses. At f ≈ ½ that is
  0.5%. NMAD against Redrock ≤ 2× the rounding floor (≈ 1 × 10⁻⁴), still 5×
  under the 5 × 10⁻⁴ requirement (§9.4c).
- **detection** — the flag's precision must exceed the base rate by a margin
  whose interval excludes it; recall is reported, not thresholded, until the
  proven-wrong set is larger.
- **net** — the interval on (model right − Redrock right) among disagreements
  must exclude zero. Until it does, the claim is "agrees with Redrock", not
  "beats it".

### Measured: the first detection result (2026-09-22)

**Set.** The deep-tail teacher (§7b-2) flagged 224 SV3 objects where its answer
carries ≥ 10σ evidence and differs from Redrock by > 0.1. 100 were drawn and
mixed with 50 controls (teacher and Redrock agree), shuffled into OBJ001–150,
rendered with no identity, and read blind (`blind224/`). Every answer was frozen
(`answers_FROZEN_before_key.txt`, read-only) before the key was opened once.

**Controls — the reading is sound.** 50/50 agree with both at the catastrophic
level; 48 within 0.01(1+z), the other 2 passive galaxies off by 1.0–1.5%.

**Candidates.** My blind z sides with:

| | teacher | Redrock | neither |
|---|---|---|---|
| all 100 | **40** | 14 | 46 |
| excluding the 41 I marked `??` (no readable feature) | **36** | 9 | 14 |

- Where exactly one side matches the reading, **the teacher is right 36/45 = 80%
  (95% CI 65–90%)**.
- **Redrock is demonstrably wrong on at least 36 of the 100 flagged objects**,
  against a ~1% base rate: the flag is **≥ 36× enriched** for Redrock errors.
  3 more are **blends** (two galaxies in one fibre) where each side found a real
  system (OBJ008, OBJ082, OBJ095).
- Redrock's failures found: **quasars put at low z with the broad lines ignored**
  (OBJ062 −0.001 → 2.22, OBJ144 0.17 → 2.65, OBJ057 0.92 → 2.64, OBJ117, OBJ118,
  OBJ148); **stars at z ≈ 1–1.6** (OBJ007 STAR target called QSO at 1.56, OBJ145
  at 1.04 — the white-dwarf pattern of §6c); **one line misidentified** in ~20
  galaxies.
- Caveat: 4 teacher wins are single-line quasars read as Lyα rather than C IV on
  the strength of a blueward flux decrement; if all four were C IV the share
  falls to 32/45 = 71%.

**The defect it exposed.** Of the 41 undecidable candidates, **28 owe their
"≥ 10σ" evidence to a sky residual or a data artifact** (teacher's best line
median 11.9σ). `evidence.py` does not mask ivar troughs, so a sky line counts as
strong evidence — which corrupts the flag and every reranker feature built on
the same maps. Masking pixels whose ivar falls well below its local median is
the next fix, and the detection precision should be re-measured after it.

**Fixed 2026-09-23 (`evidence.spectrum_maps(x, sky=r)`).** A pixel whose ivar is
below r × its 121-px local mean is treated as no measurement, and so is any pixel
within ±3 of a bad one (OBJ002's seven 15–67σ "lines" all sat beside ivar = 0).
`sky=None` reproduces the old maps bit for bit, which every checkpoint trained
before this date expects. **r = 0.3**, chosen on the blind-read objects by
separating two groups, not by maximising flag precision: of the 28 sky-driven
flags only **39%** still pass the ≥ 10σ test, while **81/81** confidently read,
teacher-correct objects keep it; 3.7% of pixels are masked. Flag precision on
the same 100 (in-sample, a direction only):

| rule | flags | teacher right | precision | teacher wins kept |
|---|---|---|---|---|
| as used (no mask) | 100 | 40 | 40% | 100% |
| **sky mask 0.3** | **71** | **37** | **52%** | **92%** |
| mask + ≥ 2 lines ≥ 5σ or break ≥ 10σ | 47 | 28 | 60% | 70% |

The remaining false flags are not sky: single real lines (the alias problem —
one line does not fix z) and continuum-level data defects (negative flux, arm
ramps, zeroed stretches) that a pixel mask cannot see. Corroboration is kept as
a **confidence tier, not a filter**, because the cases it drops are mostly
single-line Lyα quasars the teacher had right (§5: marking is free, refusing is
not). On all 100,000 deep-tail rows: **224 → 170 flags** (54 dropped, none
added), **116 tier A** (corroborated) and **54 tier B**
(`/pscratch/sd/s/shunth/zfm_flags_v2.npz`).

**Out-of-sample check, 2026-09-23 (`blind170/`).** The 99 v2 flags not read in
blind224, plus 50 fresh controls, read blind and frozen before the key opened.
Controls **50/50** agree with both sides. Candidates: my z sides with the
teacher **46**, Redrock **14**, neither 39; on the decidable ones **35 vs 10 —
teacher right 78% (95% CI 63–89%)**, the same as the in-sample 80%. **Redrock is
wrong on at least 46 of 99 flags (≥ 46× the ~1% base rate)**, up from 36/100
before the mask. The corroboration tiers came out **backwards**: tier A
(corroborated) 31 teacher / 14 Redrock, tier B (single strong line) **15 / 0**.
Corroborated flags include blends and a real line combined with a spurious one,
so the tier is **not** a confidence signal and is withdrawn as one (n = 26 in
tier B: a direction).

**The mask does not help the redshift reranker.** Retrained with masked evidence
features (`--sky 0.3`), 300k, seed 42, against the identical-config unmasked
baseline on 99,424 validation spectra: catastrophic **1.127% → 1.164%, 154 saved
/ 190 broken, p = 0.059** — no gain, if anything a loss, mostly ELG (+6%). So the
mask is kept **for the detection flag, where it is measured to help, and not for
the reranker's features**. The likely reason is that the reranker already learns
to discount pixels near low ivar from the √ivar channel, while the fixed 10σ
test of the flag cannot.

**Generalised to main spectra, 2026-09-23 (`blindmain/`).** The same flag
(`EV_300000_d320_s42_nc_fn_ev`, sky-masked 10σ strength test) run on 150,000
main-survey spectra fires on **254 (0.169%)**: STAR 0.059%, BGS 0.200%,
LRG 0.136%, ELG 0.177%, QSO 0.312% of each target class
(`/pscratch/sd/s/shunth/zfm_mainflag_merged.npz`). Two independent checks:

- **Gaia, no spectra read.** Of the 34 flags where the model says star, 8 have
  Gaia astrometry and **all 8 are star-like** (95% CI 63–100%); controls on the
  same test: stars the model and Redrock both put at rest 98.2% star-like, z > 0.5
  galaxies 0%.
- **Blind reading.** 100 flags plus 50 controls, read blind and frozen before the
  key opened. Controls **45/50** agree with both sides; the 5 misses are all
  objects I marked `??` (no readable feature), and 39/39 of the rest agree.
  Candidates: my z sides with the model **51**, Redrock **14**, neither 35; on the
  decidable ones **35 vs 10 — model right 78% (95% CI 63–89%)**, identical to
  SV3 deep-tail in-sample 80% and out-of-sample 78%. **Redrock is wrong on at
  least 51 of 100 main flags**, against a ~1% base rate. The flag does not need
  SV3 depth to work.

Two recurring Redrock errors among the model-right flags: stars given
`GALAXY` at z ≈ 1.64–1.69 (7 objects, the model says z = 0), and BGS targets —
bright galaxies — given z > 1.

**Three of the ten decidable false flags looked like the ±50 clip — withdrawn.**
OBJ085, OBJ135 and OBJ136 are bright stars (median flux 2460, 1134, 478) with
93–99.9% of pixels clipped, which `EV_300000` put at z = 0.57, 0.57 and 2.61. I
attributed that to the clip and predicted 83% precision once it was fixed. The
clip A/B (§2) refutes it: a **clamped** 150k model puts all three at z = 0 and does
not flag them, and softclip changes the number of clipped flagged rows from 12 to
15, not towards zero. Their false flags belong to that one checkpoint.

Recall is not measured. As an order of magnitude: ~1% of 150,000 is ~1,500
Redrock errors (Wilson range 400–5,400), and the flag catches ≥ 51% of 254
≈ 130, so it finds roughly a tenth of them. The flag is precise, not complete.

### What this changes in the model

1. **Proven-wrong labels leave the redshift loss and become the detection test
   set.** A label shown wrong by a route independent of Redrock (Gaia for the
   2,022 `MWS_WD`; blind reading for the rest) is removed from the *z* loss and
   kept in the class loss if its class bit is right — the `MWS_WD` targets are
   stars, only their z is wrong. Training on them would teach exactly the error
   the model must detect. They are never used to filter validation (§6d).
2. **The redshift loss must stop fitting past Redrock's error rate** (§4 item 4).
   A mixture likelihood `−log[(1−ε)·p(z_label) + ε/N_Z]` with ε ≈ 0.01 does this
   mechanically: once the model puts less than ε/N_Z on a label, its gradient
   toward that label vanishes, so a confident, different answer is no longer
   pushed back to Redrock's. It was not in `zfm/train.py` (plain cross-entropy).
   **Built and tested, 2026-09-23** (`results/prereg_eps150k.txt`):
   `train.mix_nll` applies it to both the z loss (N = N_Z) and the rank loss
   (N = candidates); ε = 0 is the old loss exactly, and at a logit gap of 30
   the gradient toward the label falls from 1 to 10⁻⁷. Arms: ε = 0.01 and 0.05,
   seeds 42 and 43, 150k, against the clip A/B's clamp arms (same config, same
   draw). New test set `zfm/known_wrong.py`: labels proven wrong without
   Redrock — 2,022 Gaia-confirmed white dwarfs labelled z > 0.1 — and 110
   objects my frozen blind reads put against Redrock, with 2,000 at-rest white
   dwarfs and 157 blind reads that agree with Redrock as controls.

   | | seed 42 base → ε 0.01 → ε 0.05 | seed 43 base → ε 0.01 → ε 0.05 |
   |---|---|---|
   | wrong-label WDs answered z = 0 | 85.1 → 83.2 → 81.4% | 80.0 → 76.1 → 82.9% |
   | wrong-label WDs copying Redrock's z | 5.6 → 4.6 → 3.8% | 7.8 → 7.5 → 5.9% |
   | at-rest WDs right | 98.5 → 97.7 → 97.8% | 98.0 → 97.5 → 98.0% |
   | blind: Redrock wrong, model right | 90.9 → 90.9 → 91.8% | 90.0 → 88.2 → 93.6% |
   | 100k validation catastrophic vs Redrock | 1.240 → 1.233% | 1.278 → 1.237% |

   **The pre-registered McNemar test is invalid on this set, and I only saw it
   from the baseline.** The two *baselines* — the same loss, different seeds —
   differ on the 2,022 white dwarfs at **112 vs 214, p = 1.7 × 10⁻⁸**. Those
   objects are one homogeneous population, so their errors move together, and
   a second training run moves them all at once. McNemar counts every one as
   independent evidence. By the letter of the rule, ε = 0.01 "HURTS" on the
   target (226 / 344, p = 9 × 10⁻⁷) and on the at-rest control (23 / 51,
   p = 0.0015), and ε = 0.05 splits (104 / 179 and 191 / 133). Measured against
   the run-to-run spread instead, every change (−1.9, −3.9, −3.7 and +2.9
   points) is inside the 5.1-point gap between the two baselines. The 100k guard
   did not move (349 / 301, p = 0.065, if anything toward agreement).

   **Verdict: no detectable effect at 150k; not adopted, default ε = 0.** The one
   consistent sign is the mechanism's own: all four arms copy Redrock's
   white-dwarf error less (5.6/7.8% → 4.6/7.5/3.8/5.9%), but what leaves
   Redrock's answer does not reach z = 0. That is 4 of 4 in one direction
   (sign test p = 0.125), so it stays a direction. At 150k the baselines copy
   only 6–8% of these errors in the first place, so there is little
   memorisation for the loss to remove. The test is worth repeating at a scale
   where the baseline memorises more, with ≥ 3 seeds per arm, judged against
   the seed spread.

   **Class-head instability is now the larger defect.** `class_any_bit` across
   the six 150k runs: 0.846, 0.767, 0.581, 0.447, 0.437 and 0.197. Four of six
   are far below the healthy ~0.85, whatever the input transform or z loss. It
   does not show in z, but the class output is not trustworthy until it is
   fixed.
3. **A detection output.** For an object with a Redrock z, the model reports
   `P_model(z_Redrock)` — its own posterior mass within tolerance of Redrock's
   answer — alongside its confidence (§5: evidence floor, feature significance
   at its own answer). The flag is **confident and P_model(z_Redrock) low**,
   and it carries its reason: the ≥ 8σ feature Redrock's z leaves unexplained,
   the doublet ratio that contradicts it, the Balmer series at rest. A flag
   with a reason can be checked in seconds; a bare score cannot. Redrock's z is
   the object *being audited* here, not an input the model learns from, and
   nothing else from Redrock enters: `DELTACHI2`, `ZWARN`, `SPECTYPE` stay out
   under the hard rule. Selection already fixes `ZWARN = 0`, so every Redrock
   error in this set is one Redrock was confident about — the flag has to work
   where Redrock's own warning is, by construction, silent.
4. **A memorisation test, on the set that exists today.** Run the 1M checkpoint
   on the 2,022 Gaia-proven stars and split them by whether they were in its
   training draw. Answering z ≈ 0 on held-out ones is detection; answering
   z ≈ 1.5 on training ones but not on held-out ones is memorisation of
   Redrock's error. Controls on the same axis: the Gaia known-star and
   known-quasar samples of §6c.

---

## 6. Evaluation protocol

This is not an afterthought; it is where most of this project's mistakes were
found, and one of them was mistaking a selection change for a model regression.

- **Stratify by how many diagnostic features fall in the observed window**, not
  only by class. Report the window composition of the evaluation set before the
  headline number. This is no longer an inference: it was tested by changing
  that one variable and nothing else. Six cohorts drawn with no redshift cut
  average **49.3/50**; three drawn with `--zmax 1.6` average **44.0/50**;
  removing the cut again returned **48/50**. The reader was identical.
- **Report refusals separately from claims**, and never let a refusal count as
  a correct answer.
- Report `|Δz| < 0.1` (identification) separately from 1000 km/s (precision).
- Always report the **target bits as a baseline** on the same objects — they
  scored 42–43/50 on the two most recent cohorts, worse than the read.
- Coverage–accuracy curve for the abstention head.
- Blind protocol: no Redrock quantity in selection except `ZWARN = 0`.
- **Never report agreement with Redrock as accuracy.** Report the three
  numbers of §6e — agreement on a corroborated set, detection on a
  proven-wrong set, and the net count on blind-read disagreements.

**Enforce these, do not document them.** Counted over nine cohorts, corrections
that were given a consumer that fails loudly recurred **0 times**; corrections
left as prose recurred **3 times** and **11 times**. The eval harness should
refuse to print a headline number without the stratification table, exactly as
`measure3.py` now refuses to run when a >8σ filter peak is unmentioned or when
a line-class is called with no line.

---

## 6b. The training set that exists, and what is wrong with it

Two files are built and validated on the exact 7781-point BRZ grid
(3600-9824 A, 0.8 A, wavelength vector carried as a SHA256 in the attributes):

| file | rows | GALAXY / QSO / STAR |
|---|---|---|
| `dr1_main_primary_tgt_zwarn0_brz.h5` | 18,659,804 | 69.9% / 8.3% / 21.7% |
| `dr1_sv3_primary_tgt_zwarn0_brz.h5` | 992,821 | 66.4% / 4.4% / 29.3% |

`TSNR2` also settles the SV3-versus-main depth question with a number instead of
a recollection: the median TSNR2 ratio is 12-15 across all five templates, and
TSNR2 is an SNR *squared*, so SV3 is **3.5-3.8x deeper in SNR**. Any score
measured on SV3 has to be reported against SV3, never against main.

**The fields this design needs are present, but not all of them may be used.**
`BRZ_FLUX`, `BRZ_IVAR` and `BRZ_MASK` for the two input channels; the DESI
target bits for the class labels. `SPECTYPE`, `SUBTYPE`, `DELTACHI2` and `CHI2`
are Redrock output and are **excluded as sources to learn from** — the earlier
version of this file listed them as label and loss inputs, which was wrong.
`Z` is discussed in §6c;
`TARGETID`, `SURVEY`, `PROGRAM` and `HEALPIX`, which are exactly the four-tuple
the sealed keys match on; and the target bits for the section 6 baseline. Spot
checks find no non-finite flux, no negative ivar and no all-masked rows, with
7728-7781 usable pixels per spectrum.

**Sampling from them without repeats is safe, and checked in three senses.**
`TARGETID` is unique inside each file (0 duplicates in 18,659,804 and 992,821
rows) and the two files share **0 TARGETIDs**, even though 87% of the sv3
healpix are also in main. That is `ZCAT_PRIMARY` doing it: it marks one row per
`TARGETID` across the whole catalogue, so an object observed in both surveys
lands in exactly one file. The same flag that disqualifies these files for blind
cohorts is what makes pooling them safe. `code/h5pick.py` draws from either or
both, excludes anything in the shared seen list, asserts all three properties,
writes the seen list before the payload, and **refuses `--blind`**.

Three things are nevertheless wrong with it as a *training* set, in order of
severity:

1. **It contains the evaluation set.** 978 of the 1000 TARGETIDs read in the
   blind tests are in these files - 750 in main, 228 in sv3. Training on them
   as delivered makes every blind score in `METHOD.md` meaningless.
   `code/heldout_targetids.json` carries the list; the trainer must **assert**
   the intersection is empty before the first step, not filter quietly.
2. **The sample is doubly selected on Redrock having succeeded.** The build
   filter is `ZCAT_PRIMARY == True and OBJTYPE == "TGT" and ZWARN == 0`. Both
   conditions select for objects the reference pipeline found easy, and
   `ZWARN = 0` removes hard objects by construction. A model trained here is
   calibrated on the easy half, and its refusal head (section 5) will be
   calibrated on a population that contains almost none of the cases refusal
   exists for. Any headline number must be reported with this stated, and a
   `ZWARN != 0` holdout should be built before the refusal head is trusted.
3. **`ZCAT_PRIMARY` dedup is baked in, so these files cannot draw blind
   cohorts.** The blind protocol forbids `*_PRIMARY` (those flags rank by
   ZWARN). `TSNR2` was added to both files on 2026-09-05 - all 32 columns, 100%
   finite and positive, verified against 692 objects whose TSNR2 this project
   had recorded independently from the zcatalog (**691 agree to 0.1%**). That
   makes depth *reportable* but does **not** make the files re-dedupable: each
   holds exactly one row per TARGETID, so the alternative observations are not
   there to choose between. Evaluation cohorts must keep coming from the
   zcatalog path.

   The single disagreement in that check is the reason why. cohort16/OBJ049 is
   one object with two observations: the cohort builder took the `main` row
   (TSNR2 178.0) and `ZCAT_PRIMARY` took the `sv3` row (172.1). **The two dedup
   rules select different spectra of the same target.** Excluding by TARGETID
   still removes it from both, but the paths cannot be merged.

4. **Some labels are outside the physical range of their own target class.**
   `zfm/label_ranges.py` over all 19,649,702 rows, with ranges fixed from
   the selection design rather than fitted: **342,851 single-class labels (1.94%)**
   are out of range — QSO-only at stellar velocity 94,384 (8.34% of QSO-only),
   BGS-only at stellar velocity 89,905, LRG-only below z = 0.1 37,581, ELG-only
   above z = 1.635 (where [O II] leaves the red edge) 68,760, STAR-only above
   z = 0.005 15,645. A further **552,573 rows (2.81%) carry no class bit at all**
   and 7.08% carry several. A "low" flag is usually a wrong *class* bit with a
   right redshift; a "high" flag is a candidate wrong *redshift*. §6d verifies
   them.

   **Verified by Gaia** (`zfm/gaia_flags.py`, 253,966 objects, 244,459 matched,
   zero disagreements on the bits used). "star-like" = parallax or proper motion
   above 5σ, computed only on objects that HAVE a Gaia solution:

   | flagged population | n | with Gaia | star-like | in-range control |
   |---|---|---|---|---|
   | QSO-only at stellar velocity | 91,249 | 24.0% | **46.9%** | **0.06%** (16,200 measured) |
   | BGS-only z < 0.002 | 87,799 | 35.2% | **83.3%** | **3.23%** (990 measured) |
   | LRG-only z < 0.002 | 33,758 | 14.1% | 76.9% | **0.67%** (597 measured) |
   | ELG-only z < 0.002 | 16,726 | 5.8% | 66.1% | **15.4%** (319 measured) |
   | STAR-only z > 0.1, not `MWS_WD` | 9,434 | 9.5% | 81.7% | 98.2% (stars at rest) |

   The coverage column is itself the result: an in-range LRG or ELG has a Gaia
   source 0.01–0.02% of the time, a flagged one 6–14% of the time. These objects
   are point sources with measurable astrometry, which is what a star is.
   **For the galaxy and quasar classes the redshift label is right and the class
   bit is wrong** — the opposite of the `MWS_WD` case in §6c, where the class bit
   is right and the redshift is wrong.

   The LRG and ELG controls needed every in-range object to produce any measured
   ones at all (2,285,993 and 3,363,644 draws for 597 and 319 Gaia solutions).
   LRG comes out at 0.67% [0.26, 1.71] against 76.9% flagged — a factor of 115.
   **ELG is the weak one**: its own control is 15.4% [11.8, 19.7], so an in-range
   ELG that happens to carry a Gaia source is already likely to be a contaminant,
   and the flagged excess is 4.3× rather than 100×. The second limit stands for
   every row: only 6–35% of each population has any astrometry, so nothing is
   established about the remainder.

None of the four is a missing feature. The first three are properties of the
selection, which is the thing this project has twice mistaken for a property of
the reader; the fourth is label noise, to be measured and bounded (§4 items 4–5)
rather than removed.

---

## 7. Pre-train, post-train, fine-tune — what makes it a *foundation* model

Coverage, not parameter count. ~30M is the right size given measurement 2.

Three stages. They differ in **what labels they may touch**, because the label
is this project's hardest constraint (§6c) and each stage is defined by how it
stands relative to it. *Status 2026-09-22: designed, none of it built* (see
the implementation table); the supervised half of 7b is what `zfm` trains today.

| stage | learns | labels | trains |
|---|---|---|---|
| **7a pre-train** | the structure of spectra: lines, continuum, breaks, redshift as translation | **none** | the encoder |
| **7b post-train** | the redshift and class model, then aligned to §6e — agree with Redrock where it is right, catch it where it is wrong | Redrock `Z` (option A), target bits, independently verified objects | surface, reranker, heads; encoder at a small learning rate |
| **7c fine-tune** | a specific data condition or downstream task | few, task-specific | heads only, encoder frozen |

### 7a. Pre-train — no label of any kind

**Why it is worth doing.** §9 measured a **data-limited** problem: the loss is
still falling from 30k to 1M. Pre-training is the only way to show the model more
spectra without more labels, and it is the **one part of the design entirely free
of the Redrock-label problem**.

Three objectives, none needing a label:

1. **Known shift.** Translate the spectrum in log-λ by a known Δ, crop back to
   the observed window, predict Δ. The relative-shift task of the original §7; it
   trains "redshift is a translation" directly, and it is the task on which
   ConvNeXt beat the dilated stack (27 runs, §3).
2. **Masked reconstruction**, ivar-weighted: hide a stretch of spectrum, predict
   it. The model must learn how lines and continuum co-occur — [O III] 5008
   implies 4960.
3. **Predict the evidence maps.** From a masked input, predict `evidence.py`'s
   line and break maps. They have no parameters and no labels, so this teaches
   the encoder the two instruments the reader used, which is what §0 measurement
   3 says the scoring lacked.

**Data.** Because no label is needed, pre-training **can include the
`ZWARN ≠ 0` objects** — the 1.22 M `ZWARN = 4` spectra the selection removes
(§6b), which the model has never seen and which are exactly the objects Redrock
was unsure of. Dropping a filter is not learning from Redrock. It needs a new h5
build. Train across surveys (DESI + SDSS), SNR regimes and arms, so that what
transfers is instrument-independent.

**Excluded.** No AION weights or methods (hard rule; §6c records the one scoped
exception, which is a scored baseline only). No free-latent autoencoder: a free
latent hides the answer where it cannot be checked, and every real gain in this
project came from rendering something and looking at it. The validation set and
the Gaia-proven set are excluded **by `TARGETID`** from pre-training as well.

### 7b. Post-train — from a foundation to beating Redrock

Three steps, in this order:

**7b-1. Supervised.** What `zfm` trains today, plus the design items not yet
built (§4, implementation table):
- the ε-mixture z loss, so ~1% wrong labels are not memorised (§4 item 4);
- cross-class alias negatives — the ELG fix. The evidence A/B of 2026-09-22
  (two seeds) showed the reranker features barely move ELG (107 saved / 87
  broken) while QSO moves clearly (38 / 14); ELG's error is one line read as another, and the
  reranker seeing the evidence is not the same as the loss teaching the
  distinction;
- the evidence reranker (`--evidence`);
- proven-wrong labels out of the z loss (§6e).

**7b-2. Self-distillation from depth — a label source that is not Redrock.**
*Corrected 2026-09-22 (`zfm/depth_ratio.py`, TSNR2 by class, ~100 blocks per file):*
SV3 is **not** uniformly 2–4× deeper. The median SV3/main TSNR2 ratio is **1.29×
for ELG and LRG**, 1.77× for BGS and 2.23× for QSO; only the deep tail reaches
2.6–4× (90th percentiles: ELG 420 vs 164, LRG 286 vs 100). A typical SV3 ELG is
barely deeper than a main one, so a teacher on it knows no more than the student.
**Only the deep tail may teach** — e.g. TSNR2 ≥ 2× the main median for its class.
SV3 is deeper than main in that tail.

**Built and tested, 2026-09-22 — neither version pays.** Deep tail defined per
**program** (TSNR2_ELG ≥ 2× the same-program main median: dark 253, bright 61,
backup 16), 256,115 rows at a median 2.67× depth; 100,000 drawn
(`zfm/deep_tail.py`). The first version compared against the all-program median
(43.5), which let 533,410 rows through at an apparent 3.57× — normal dark-time
spectra judged against bright-time ones. Three arms, 150k each, seed 42,
`--zmax 1.6`, one training example in five drawn from the deep tail, identical
draws in every arm (`deep_check.py` passed: non-deep examples identical, A vs 0
only spectra, B vs A only labels). Scored on 95,089 validation spectra with label
z < 1.6:

| | catastrophic | saved / broken | p |
|---|---|---|---|
| 0 → A (degraded to main depth) | 1.023% → 1.065% | 182 / 222 | ≈ 0.05, worse |
| A → B (teacher labels where ≥ 10σ) | 1.065% → 1.086% | 130 / 150 | ≈ 0.23 |
| 0 → B | 1.023% → 1.086% | 171 / 231 | ≈ 0.003, worse |

Degraded copies did not transfer to real shallow spectra; the likely reason is
that the degradation scales ivar and adds Gaussian noise but does not reproduce
what makes a real shallow spectrum shallow (sky residuals, fewer exposures'
systematics). The teacher changed catastrophic-level labels on only **182 of
100,000** rows (strong evidence, disagreeing with Redrock by > 0.1), i.e. ~54
training examples — too few to measure. Its useful product is those
disagreements: **224 deep-tail objects where the teacher has ≥ 10σ evidence and
Redrock differs by more than 0.1** — the first model-generated candidate list
for §6e's detection set (`/pscratch/sd/s/shunth/zfm_teacher.npz`), to be
verified by blind reading. Degrade an SV3 spectrum to main depth by adding noise
drawn from its own ivar, and the same object exists at two depths. Where the
model's answer on the **deep** version carries strong evidence (≥10σ at its own
answer, which measured 99%+ correct, §5), that answer labels the **shallow**
version. It aims at the low-SNR regime where both the model and Redrock are
weakest, with a label that comes from the model reading better data. Risk:
self-reinforcement, a wrong deep answer teaching itself. Only strong-evidence
objects are used, and a blind-read sample audits them.

**7b-3. Calibration and detection.**
- confidence calibrated on a held-out set, so that 90% means 90%;
- the `P_model(z_Redrock)` detection output (§6e);
- verified objects — blind-read disagreements, the Gaia-proven set — are used
  for **evaluation, or k-fold only**. They are the only source of §6e's
  detection set, so they cannot all be trained on.

### 7c. Fine-tune — adapt to a condition

- **Depth**: SV3 vs main, later data releases — freeze the encoder, adapt the
  surface and reranker.
- **Other instruments**: SDSS, later 4MOST or PFS. SDSS redshifts come from its
  own pipeline, not Redrock, so they are usable — to be confirmed per survey,
  because the rule is "do not learn from Redrock", not "do not learn from DESI".
- **Downstream heads**: e.g. QSO subtypes, the BGS/LRG boundary. QSO has the
  highest catastrophic rate of any class (3.69% at 300k, 2026-09-22) and may
  justify its own head.

### How each stage has to prove itself

**Pre-training's value can be measured honestly only one way: at a fixed number
of labels, with and without it.** The from-scratch curve exists — 30k 1.391,
100k 1.036, 300k 0.839 (§9.6). If pre-training brings 30k labels to the
from-scratch 100k level, its value is "3× effective data", which is the unit a
data-limited problem is priced in. That fits inside the 600,000-spectrum cap and
is the most informative experiment per GPU hour available. Every stage and every
step inside it is an A/B with a pre-registered rule and saved/broken counts, as
the evidence reranker was.

**Open decisions (the owner's, 2026-09-22):**
1. whether these stage definitions match the owner's — post-train as supervised
   + alignment, fine-tune as downstream adaptation;
2. whether pre-training includes `ZWARN ≠ 0` objects (a new h5 build);
3. the size of the first experiment — proposed: pre-train vs from scratch at a
   fixed 30k labels, with the unlabeled pre-training size set by the owner
   (cap 600,000; sizes are always asked before a run).

---

## 8. What I would not build, with the numbers

| not building | why |
|---|---|
| continuous z regression | 0.0% of errors within 5σ_z |
| finer sampling, LSF work, better noise model | same |
| scaling data to fix the alias ridges | ELG rays α≈−0.25; on-ray share flat |
| added physics priors / hand-built line lists **as a score** | an *incomplete* forward model is worse than none; the hand list had an air/vacuum bug and no UV lines. *Scope, 2026-09-22:* `evidence.py` does use a line list (vacuum, with the UV lines) — but only to **read** significances at a candidate's z as reranker inputs, never as an additive score on `S`; additive per-line scoring is what formed the ELG rays |
| a separate BGS/LRG/ELG classifier | selection categories, not spectral ones |
| any prior of the form narrow ⇒ galaxy or broad ⇒ quasar | a narrow 18.4σ C IV in a faint quasar, and a Lyman-α forest edge that outscores broad lines on a step filter, both in one cohort |
| a hand statistic to **detect** the 4000 Å break | scanning it over z makes a **star** detector: stars median 39.2σ vs galaxies 9.4σ, 90% of stars fake a ≥10σ break. *Scope, 2026-09-22:* `evidence.py` computes the break at every z but the reranker reads it only **at each candidate's z** — a measurement at a given z, as `speclib.break_amplitude` is, not a detector choosing z |

---

## 9. Scaling law — three runs, and what they can tell you

The exponents in the literature are fitted to autoregressive text cross-entropy
in nats: Kaplan's α_N = 0.076, Chinchilla's α = 0.34 / β = 0.28, E = 1.69. Only
the **functional form** transfers. Quoting those constants for a redshift
posterior over 12,140 bins would be the same category error as quoting 92.6% as
a property of the BGS/LRG cut. The three runs exist to fit our own.

### 9.1 `C ≈ 6ND` holds here, and it was checked rather than assumed

The rule is derived for transformers where every parameter touches every token.
This architecture is not that — it has an FFT correlation whose cost is
independent of N. Per spectrum, forward:

| component | FLOP | share |
|---|---|---|
| dilated CNN encoder (15 M, all 7525 positions) | 2.26 × 10¹¹ | **99.6%** |
| FFT correlation, 64 archetypes over all shifts | 3.10 × 10⁷ | 0.0% |
| closed-form amplitudes | 9.63 × 10⁵ | 0.0% |
| reranker transformer (d = 768, L = 8, 57 M, K = 32 tokens) | 3.65 × 10⁹ | 1.6% |

**Component sum ÷ 6ND = 0.98.** An earlier version of this table used `2 N K`
for the reranker and was **3.8× low**, because that ignores attention
entirely; the number above is an explicit transformer count. It does not
change the conclusion — at K = 32 the attention term is 0.7% of the
reranker's own cost, and the reranker is 1.6% of the total. `C = 6ND` is correct here, but for a reason
worth writing down: the encoder applies every weight at every one of the 7525
positions, which is exactly the transformer accounting. Change the encoder to
something that downsamples and this stops being true.

*As built (§3) the reranker is d = 320 over K = 256 tokens, not the d = 768 /
K = 32 counted above, and its FLOP share has not been recounted.*

### 9.2 The three runs

Spaced a decade apart in compute, on the measured budget of 4 × A100 at 40% MFU:

| run | N (encoder) | spectra | tokens/param | C [FLOP] | wall clock |
|---|---|---|---|---|---|
| A | 1.5 M | 300 k | 1,505 | 2.1 × 10¹⁶ | 1.2 min |
| B | 6.0 M | 3.0 M | 3,762 | 8.2 × 10¹⁷ | 27 min |
| C | 15 M | 30 M | 15,050 | 2.0 × 10¹⁹ | **11.4 h** |

### 9.3 The finding: this is a data-limited problem

Run C is eleven hours, and it already consumes **78% of every DR1 spectrum in a
single epoch**. The compute ladder is cheap and the data ceiling is hard:
38.3 M spectra, 2.88 × 10¹¹ input positions, and there is no more DR1.
*The training files hold **19,652,625** of them after `ZCAT_PRIMARY`,
`OBJTYPE = TGT` and `ZWARN = 0` (§6b); that, not 38.3 M, is the 18M of §9.8.*

That is the reverse of the regime the scaling laws were fitted in, and it
changes what the ladder is for. Two cautions on the arithmetic:

* Counted in **positions**, Chinchilla-optimal at this D would want N ≈ 10¹⁰ —
  a thousand times larger than 15 M. Do not believe it: a spectrum's 7525
  positions are a smooth continuum plus a handful of lines, nothing like 7525
  independent tokens, so the token count is inflated and the 20-tokens-per-
  parameter ratio does not transfer.
* Counted in **independent samples** — which is the honest unit, since the
  positions within one spectrum are strongly correlated — D is 38 M against
  N = 15 M, and the classical conclusion is the opposite: more data first.

The two framings disagree, which is why §9.4 measures rather than assumes.

### 9.4 What three points can and cannot identify

Fit `L(C) = E + K·C^(−γ)` along the compute-optimal ray. Three unknowns, three
runs: **exactly determined, zero residual, no error bar and no goodness of
fit.** With the illustrative losses 4.10 / 3.40 / 2.95 nats it solves to
E = 1.618, K = 74.8, γ = 0.091, projecting L → 2.70 at 10× run C, 2.50 at 100×,
and E = 2.00 at infinite compute.

**And the fit is fragile exactly where it matters**: moving run A by ±0.05 nats
moves the floor E from **1.90 to 1.14**. Half a percent of loss on the cheapest
run swings the answer to "where does this stop" by 40%. So:

> **Run a fourth point off the compute-optimal ray** — same C as run B at
> double the parameters and half the data. That is what turns three
> interpolated constants into a measurement with a residual.

### 9.4b Why the encoder is not a transformer, in FLOPs

*Written for K = 32; as built K = 256, so the reranker's attention term is no
longer negligible and has not been recounted (§9.1).* The reranker attends over
K = 32 candidates, so its attention is negligible
(0.7% of its own cost). A **pixel-level** transformer encoder is a different
object entirely — P = 7525 tokens means 5.7 × 10⁷ pairs per layer per head:

| encoder | params | FLOP/spectrum | vs. the dilated CNN | attention share |
|---|---|---|---|---|
| d = 256, L = 6 | 4.7 M | 4.19 × 10¹¹ | **2×** | 83% |
| d = 512, L = 12 | 37.7 M | 1.96 × 10¹² | **9×** | 71% |
| d = 768, L = 12 | 84.9 M | 3.37 × 10¹² | **15×** | 62% |
| d = 1024, L = 24 | 302 M | 1.01 × 10¹³ | **45×** | 55% |

Most of that is spent on attention between pixels that are hundreds of
ångström apart and physically unrelated. Patching to 16 would cut it 256×, and
§3 rejects patching for a measured reason: it breaks translation equivariance
below the patch scale and manufactures aliases, which are already most of the
error budget.

### 9.4c NMAD is already saturated, and it is not the metric

Measured on **870 blind redshift claims**:

| | |
|---|---|
| NMAD, all objects | 1.115 × 10⁻⁴ |
| NMAD, catastrophics removed | 1.044 × 10⁻⁴ |
| NMAD of the core (\|Δz\|/(1+z) < 0.003, 94.6% of objects) | 1.032 × 10⁻⁴ |
| catastrophic rate \|Δz\|/(1+z) > 0.1 | **3.91%** |
| floor from rounding to the 40 km/s grid (bin centre reported) | **4.945 × 10⁻⁵** |
| DESI requirement (ELG) | 5.0 × 10⁻⁴ |

**Corrected 2026-09-15.** v12 and v13 quoted the floor as 5.71 × 10⁻⁵, which is
1.4826 × w/√12 — the Gaussian NMAD factor applied to the *standard deviation*
of a uniform error. The NMAD of a uniform rounding error of width w is
1.4826 × w/4 = **4.945 × 10⁻⁵**, confirmed by simulation (4.948) and on 399,880
real labels rounded to their nearest bin (4.945).

Three things follow.

1. **Removing 3.91% of catastrophic failures moves NMAD by 6%.** NMAD is
   nearly blind to the failure mode that dominates everything else in this file.
2. **NMAD is within 2.3× of the grid floor and already 4.5× better than the
   survey requirement.** There is less than a factor of 2.3 available, ever,
   and no reason to want it — which is measurement 1 restated: 0.0% of errors
   are within 5σ_z, so precision work has nothing to gain.
3. **The error distribution is bimodal**: 81.8% of claims land within 100 km/s,
   3.91% are catastrophic, and 14.3% fall in between. NMAD measures the left
   mode; the loss floor E measures the right one. Optimising one tells you
   almost nothing about the other.

So the scaling ladder should be read on the **loss** and on the **catastrophic
rate**, and NMAD reported only to show it has not regressed. §6's stratified
metric is the one that moves.

### 9.5 An independent prior on E

E is not a free constant to be discovered only by fitting. A posterior that
knows nothing costs log(13 400) = **9.50 nats**. From the blind reading,
**11.3% of spectra have nothing above 6σ anywhere** (93 of 822, §5's evidence
floor). If those and only those were uninformative, **E ≥ 1.07 nats**.

That is a lower bound and it is loose in both directions: it ignores the label
noise in the training redshifts (§6c), which pushes E up, and it treats
below-floor objects as carrying no information at all, when measurably 82% of
them still get the right answer. But it means the fitted E can be **checked**
rather than only reported — and an E far below 1.07 would indicate the model is
memorising the training redshifts rather than reading the spectra.

The calculation is in `code/scaling.py`.

**Withdrawn 2026-09-15.** The clean ladder fits E = 0.60 ± 0.05, far below 1.07,
and the model is not memorising: the validation blocks are held out and the
validation loss sits *below* the training loss at every point. The bound was
wrong on three counts: the grid has 12,140 bins (ln = 9.404, not 9.50); the 11.3%
comes from class-balanced blind cohorts, not the survey mix; and treating a
spectrum with nothing above 6σ as carrying no information contradicts §2's own
point that a passive galaxy's redshift lives in the coherent sum of features
each below that threshold. An earlier fit that **matched** this bound to four
decimals (1.0623 against 1.0627) was an artifact of a diverging loss (§9.6).

### 9.6 Measured: the data ladder

Run 2026-09-09 to 09-15. Fixed model — encoder 14.24 M, reranker d = 320, 30.4 M
total — global batch 32, seed 42, one pass, OneCycle over each point's own
steps, and the same 50,016 held-out validation spectra at every point.
Throughput: 72.5 spectra/s per A100 at batch 32 (28.6 GB; GPU-bound, the data
pipeline delivers 10,423/s), ×2.59 with DDP on four cards. A global batch of 128
was worse at every learning rate tried at fixed D (best 1.688 against 1.420 at
30k), so 32 is fixed. A same-seed rerun reproduced every printed digit.

| D | steps | val z loss | rank loss | NMAD | catastrophic |
|---|---|---|---|---|---|
| 30 k | 937 | 1.3914 | 1.03 | 8.044 × 10⁻⁵ | 2.27% |
| 100 k | 3,125 | 1.0358 | 1.10 | 6.900 × 10⁻⁵ | 1.39% |
| 300 k | 9,375 | 0.8385 | 0.57 | 6.345 × 10⁻⁵ | 1.17% |
| 1 M | 31,250 | 0.7376 | 0.42 | 6.107 × 10⁻⁵ | 1.09% |

30k and 100k were run before the reranker's final LayerNorm, 300k and 1M after.
Adding it moved the 300k point by 0.0167, under the 0.02 set in advance for
reusing the two smaller points.

**The form was tested before it was fitted.** From 30k / 100k / 300k, before the
1M run: the floor form predicted 0.7445 and a pure power law 0.6442. Measured
0.7376 — **0.007 from the floor form, 0.093 from the power law.** On all four:

| form | parameters | residual rms |
|---|---|---|
| E + A·D^−β | E = 0.6005 ± 0.034, β = 0.508 ± 0.047 | **0.0052** |
| A·D^−β | β = 0.195 | 0.0388, systematic (+ − − +) |

The local exponent falls 0.245 → 0.192 → 0.107.

**Twice before, the ladder looked like this and was wrong.** The unnormalised
surface gave 1.4204 / 1.1379 / 1.0806 and a fitted E = 1.0623 — agreeing with
§9.5's prior to four decimals — while the rank loss on the same runs read
1.20 / 1.37 / **46.1**. The next 1M point, at 0.7852, had a rank loss of 136 and a
z loss that had stopped learning (§4 item 8). Neither fault showed in the z loss
being fitted. **A ladder point is accepted only when every loss term at that
point is healthy.**

### 9.7 Measured: noise and capacity at 300k

| run | val z | rank | catastrophic | class | params |
|---|---|---|---|---|---|
| d = 320, seed 42 | 0.8385 | 0.570 | 1.172% | 0.836 | 30.4 M |
| d = 320, seed 43 | 0.8273 | 0.550 | 1.176% | 0.819 | 30.4 M |
| d = 960, seed 42 | 0.8837 | 0.747 | 1.296% | 0.791 | 109.1 M |

* **Seed noise, from one pair: 0.011** in z loss. Changing the GPU count at a fixed
  global batch moved a point by up to 0.017, confounded with the LayerNorm change.
* **The larger reranker is worse, and re-tuning the learning rate does not save
  it.** The first d = 960 run was +0.045 on val and +0.052 on *train*, which
  looked like under-training rather than overfitting, so the learning rate was
  swept at 300k with everything else fixed:

  | d_rr | lr | train_z | val_z | vs baseline | loss_rank | catastrophic |
  |---|---|---|---|---|---|---|
  | **320** | 3e-4 | **0.8988** | **0.8385** | — | 0.570 | 1.172% |
  | 960 | 1.5e-4 | 1.0454 | 0.9571 | +0.119 | 0.744 | 1.294% |
  | 960 | 3e-4 | 0.9654 | 0.8837 | +0.045 | 0.747 | 1.296% |
  | 960 | 6e-4 | 1.0421 | 0.9804 | +0.142 | 0.890 | 1.575% |
  | 960 | 1.2e-3 | 0.9538 | 0.8934 | +0.055 | 0.799 | 1.386% |

  Every arm is outside the 0.011 seed noise and outside the 0.02 threshold set
  in advance, and **the training loss never reaches the baseline at any learning
  rate** — so this is not overfitting, and it is not a learning-rate artifact
  either: both a lower and a higher rate are worse than the original. **A 3.6×
  larger reranker does not pay at 300k spectra.** The one lever left untested is
  *steps*: 9,375 optimisation steps at a global batch of 32 may simply be too few
  for 109 M parameters, and step count is confounded with D in this ladder, so it
  cannot be varied here without breaking the comparison. Without the final
  LayerNorm the same run diverged outright (rank loss 26,479).

  Read together with §3's sep result — where a *higher* candidate recall also
  failed to lower the realised error — the reranker axis has now twice been given
  more (more candidates, more parameters) and twice converted none of it.

### 9.8 What the ladder says about 18M

| | value | 68% | 95% |
|---|---|---|---|
| E | 0.60 | ± 0.034 | 0.54 – 0.67 |
| 3 M | 0.677 | 0.653 – 0.691 | 0.597 – 0.708 |
| **18 M** | **0.631** | 0.605 – 0.656 | 0.567 – 0.682 |

Those intervals come from four points and one degree of freedom, and the
per-point noise (≈ 0.008 from the seed pair) is larger than the fit's residual
rms (0.0052), so they are **about 1.5× too narrow**: read 18M as **0.63 ± 0.04**
and E as **0.60 ± 0.05**. Leave-one-out predictions miss by +0.112, −0.020,
+0.014 and −0.031.

1. **Most of the data axis is already spent.** 1M → 18M buys ≈ 0.11 nats, closing
   about 77% of what remains to E.
2. **NMAD approaches the rounding floor, not zero.** 1.235× the 4.945 × 10⁻⁵ floor
   at 1M, ≈ 5.35 × 10⁻⁵ (8% above it) extrapolated to 18M. Beyond that, precision
   comes only from reporting z finer than a bin — interpolating the posterior
   between bins, or a finer grid — never from data. It is already 8× better than
   the 5 × 10⁻⁴ requirement.
3. **The catastrophic rate cannot be extrapolated yet.** 2.27 → 1.39 → 1.17 →
   1.09%: steps of 10.4σ, 3.1σ, then 1.2σ on 50,016 spectra.
   It is also a disagreement rate with Redrock, and at 1.09% it is already
   indistinguishable from Redrock's own ~1% error rate, so below here it cannot
   separate improvement from copying (§6e).
4. **E belongs to this grid.** A different bin width changes ln N_Z and the scale of
   the loss, so E is not comparable across grids.

**Open:** the 30k point's class score is 0.039 against 0.80–0.90 elsewhere and is
unexplained; the final LayerNorm lowered the 300k class score from 0.905 to
0.836; the `sep` sweep (§3); the clip at 1.0 (§4 item 8); and the class breakdown
of the 56 non-star catastrophics where the model answers z ≈ 0 (§6c).

