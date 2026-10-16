# FM_DESIGN — a foundation model for DESI spectra (v1, 2026-10-08; §9 added 2026-10-12)

This replaces the framing of DESIGN.md. DESIGN.md designed a **redshift model**:
templates, 12,140 z bins, top-K, evidence, reranker. It called that model a
foundation model, and it was never one (2026-10-08 audit, §0). This document
starts from the definition instead of from the task.

Every claim is marked **[measured]**, **[fact]** (data or code as it exists)
or **[not measured]**.

---

## 0. Definition and audit

A foundation model is trained on **broad data, mostly with self-supervision at
scale**, and can then be **adapted to a wide range of downstream tasks**
(Bommasani et al. 2021). The model is not designed for any one task. Tasks are
attached afterwards.

zfm (C15) against that definition:

| requirement | zfm | |
|---|---|---|
| broad data | 1.5M clean-2 ZWARN=0 spectra | partly [fact] |
| self-supervised | trained on Redrock `Z` and target bits; masked pre-training tried once and not used | no [fact] |
| many downstream tasks | z and class only. 16.2 of 30.6M parameters (template 6.3M + reranker 9.9M) can only answer z | no [fact] |
| task-agnostic backbone | the encoder sees 79 px (±37 Å) by design, has no whole-spectrum vector, and was trained on z labels | no [fact] |

Conclusion: no part of zfm is a foundation-model backbone. The backbone is new.

---

## 1. Data

> **Owner rule, 2026-10-09 evening, supersedes the clean-2 rule below:** no cleaning.
> - Every split uses the original ZWARN=0 h5 files (`--clean 0`, explicit).
> - The test set is not fixed. `zfm_test60k_raw.h5` was a one-off convenience for comparing the 2026-10-09 models.
> - Everything in §8 was trained on clean-2.
>
> **Owner rules added 2026-10-09 / 10-10 (apply to §9):**
> - Class mix: every draw is 20% each of QSO / LRG / ELG / BGS / STAR by priority class (QSO > LRG > ELG > BGS > STAR), rows with no class bit dropped (`data.Spectra(mix=True)`). Approved exceptions, training only: §9.8.
> - Held-out (early stopping) and test are each ≥ 20% of the training size. For 1M training: test = 200k from validation blocks 0–2200, held-out = 200k from blocks 2200–4400; training excludes blocks 0–4400.
> - Fill each A100 to 32–36 GB; one seed (42).

| | value | |
|---|---|---|
| corpus | clean-2 only (owner 2026-10-08): main 17,432,075 + sv3 947,646 = **18,379,721** spectra, all with unique `TARGETID`, none shared between the two files | [fact] |
| main programs | dark 7,964,453 / bright 8,264,479 / backup 1,203,143 | [fact] |
| input | log-λ grid 7,526 px at 40 km/s, x = [flux·√ivar, √ivar], from `data.py` | [fact], reused |
| splits | the fixed seed-777 validation-block permutation (`data.Spectra`). The test set is the first 60k of the clean-2 validation blocks. Overlap with every training pool is 0 | [measured] `test_fm.py` |
| pre-training size | **owner's decision**; the cap is 600k unless an exception is granted | — |

Redrock enters in two places only:
- the clean-2 row filter (owner rule 2026-09-27)
- the label of the redshift downstream task (owner 2026-10-08: 「z 先用 redrock」)

It enters neither the pre-training objective nor the backbone.

---

## 2. Backbone

Two requirements that zfm's encoder does not meet:
- **global context**: a representation can relate [O II] to Hβ, 1,991 px apart
- **a whole-spectrum output**: a token sequence *and* one global vector

| candidate | shape | why | risk |
|---|---|---|---|
| **B1 hierarchical ConvNeXt + Transformer** | ConvNeXt stages with stride 2 ×3 (7,526 → ~940 tokens), then a 6-layer Transformer over the tokens, plus a [CLS] vector | keeps convolutional locality at fine scale, and gives global context on top | downsampling breaks shift-equivariance below 8 px (320 km/s) [not measured] |
| **B2 patch Transformer** | 16-px patches (640 km/s) → 470 tokens, 12 layers, d = 512, [CLS] | the standard FM backbone (AION, AstroCLIP-style) | no built-in translation; sub-patch precision must come from the head [not measured] |

- Size: 30–60M parameters for the first comparison, the same budget for both.
- Which one wins is unknown. It is decided by Step 1 (§6), not argued here.

---

## 3. Pre-training objectives (no labels)

| | objective | status |
|---|---|---|
| **P1** | masked spectrum modelling: hide spans and reconstruct flux·√ivar in noise units (Huber), only where measured. **Mask ratio and span length are parameters to measure.** The only measured setting is 15% in 32-px spans (`pretrain.py`). Compare 15 / 30 / 50% and choose by the downstream probes, not by reconstruction loss. (An earlier draft said 30–50%; that was borrowed from image MAE practice with no measurement on spectra.) | first objective. The old 14M encoder learned it: held-out recon 0.463 against 4.962 for predict-zero, near the ≈0.425 floor if the ivar is right (the floor is computed, not measured) [measured 2026-09-23] |
| P2 | two-view contrastive | **not on coadds**. Two augmented views of one coadd share one noise realisation, and matching that noise solves the task (shift task: 92% at random init) [measured]. Needs two independent exposures, which are not in the h5 files [fact]; needs a new extraction |
| P3 | spectrum ↔ imaging alignment (AstroCLIP-style) | the DR1 `lsdr9-photometry` VAC is on disk [fact]; join coverage [not measured]. Later stage |

---

## 3b. Reconstruction as a capability

P1 *is* reconstruction (R1 below). Making reconstruction a delivered capability
means keeping the decoder as an output and scoring it. Only R2 and R3 are new.

| | input → output | truth | status |
|---|---|---|---|
| **R1 inpainting** | spectrum with masked or bad pixels → the missing pixels | the hidden observed pixels, which are noisy | this is P1 |
| **R2 from the vector** | the global vector alone → all 7,526 px | the observed spectrum | new: a vector-to-spectrum decoder; the residuals feed T6 anomaly search |
| **R3 denoising** | a main-depth spectrum → a deeper one | SV3 deep-tail spectra degraded to main depth (`deep_tail.py` degrade) | new |

- B1 needs an upsampling decoder (~940 tokens → 7,526 px). B2 uses a standard MAE decoder.
- **Scoring:** χ² per pixel relative to the noise floor, split by line, continuum and sky regions; line fluxes from the reconstruction vs `fastspecfit`.
- A reconstruction gain that lowers the probe scores is not progress.

## 4. Downstream tasks (the evaluation of the foundation model)

- **Protocol:** the backbone is **frozen**; each task gets a linear probe and a small MLP head.
- **Label budgets:** 1k / 10k / 100k.
- **Secondary:** full fine-tuning.

All label sources are on disk at NERSC [fact]. The coverage of each one in clean-2 is [not measured] and is Step 0.

| task | label source | metric |
|---|---|---|
| T1 redshift | Redrock `Z` (clean-2) | catastrophic \|Δz\|/(1+z) > 0.1, NMAD; heads: (a) a z-bin classifier on the tokens, (b) zfm's template + reranker on top of the backbone |
| T2 class | DESI target bits (5-way, bitmask-normalised) | accuracy, any-bit |
| T3 stellar parameters | DR1 `mws` VAC: Teff, log g, [Fe/H], RV | RMSE / scatter |
| T4 galaxy physics | `fastspecfit` line fluxes and EWs; `cigale` / `stellar-mass-emline` stellar mass | scatter vs the catalogue |
| T5 absorbers | `mgii-absorber`, `dla-cnn-gp` flags | AUC |
| T6 rare objects | `strong-lensing` candidates, by nearest-neighbour retrieval in embedding space | recall@k |
| T7 Redrock failure | Gaia-proven and repeat-disagreement sets from rrwrong | AUC vs P_WRONG |

Caveat: several catalogues (fastspecfit, cigale, mws) were fitted using Redrock's
`Z`. They are downstream **labels** only, never inputs [fact].

---

## 5. Baselines and success criteria (pre-registered before any run)

**Baselines:**
- the same backbone at random init, frozen (the floor)
- the same backbone trained from scratch on each task
- for T1: zfm C15 and Redrock agreement
- optionally AION (the owner's 2026-09-17 exception allows it as a scored baseline)

**It is a foundation model if all of these hold:**
1. Frozen pre-trained probes beat frozen random-init probes on **every** task (T1–T7).
2. Label efficiency: pre-trained at 10k ≥ scratch at 100k on at least 4 of 7 tasks.
3. At full labels, T1 is not worse than scratch: the catastrophic difference on the same 60k clean-2 test objects is within counting noise, or lower (bootstrap over test spectra).
4. Downstream improves with pre-training data size (Step 2).

**Not evidence:** a lower reconstruction or val-z loss on its own. This failed
four times in DESIGN.md as a predictor of catastrophic.

---

## 6. Ladder

| step | what | output | owner decision |
|---|---|---|---|
| 0 | join the VACs to clean-2 rows; build the probe harness (frozen features → linear / MLP, fixed splits, the 1k/10k/100k budgets) | coverage table per task | — |
| 1 | B1 vs B2, P1 pre-training on the same pool, same budget, one seed (42; owner 2026-10-08: seeds differ by ~3 of 60k on clean-2); full probe suite | the backbone | pre-training size |
| 2 | pre-training scale with the winner (e.g. 300k → 1M → 3M → all clean-2) | downstream vs data | each size |
| 3 | add P2 (needs the exposure extraction) and P3 (photometry) | objective A/Bs | — |
| 4 | T1 head study: z-bin head vs template + reranker head, frozen vs fine-tuned; compare to C15 | the redshift product | — |

---

## 7. What carries over from zfm

| kept | role |
|---|---|
| `data.py`, the log grid, the clean-2 h5 files, the validation blocks | input pipeline |
| `pretrain.py` masking and Huber-in-noise-units loss | starting point for P1 |
| template + reranker | **one** possible T1 head |
| `evidence.py` | diagnostics and a probe target, not a backbone input |
| rrwrong, the Gaia and VI sets | T7 labels and evaluation |

**Not kept as the backbone:** the 79-px encoder.

---

## 8. T1 head design: zfm on top of the native FM (2026-10-09, owner: 「tokenizer 要用這個 foundation model」)

The FM (`specfm_native.py`, p = 8, 974 tokens × 768) replaces zfm's 79-px
ConvNeXt encoder. Everything after the encoder is zfm's, unchanged, because it
is the part that measured well: 0.048–0.060% catastrophic on clean-2 [measured].

### 8.1 Why not a pooled head
The frozen-token + mean-pooled z head gave 9.28% catastrophic [measured,
2026-10-09, old FM]. Mean pooling throws away *where* a feature is, and z is
position. zfm's surface keeps position at every pixel. The FM tokens therefore
feed the surface, not a pooled vector.

### 8.2 Pipeline

```
BRZ_FLUX, BRZ_IVAR (7,781 px native, raw)
  -> FM (NativePrep + 12-layer Transformer, no hiding)        974 tokens x 768
  -> un-patchify: Linear(768 -> 8 x 64) per token              (B, 64, 7,781) native
  -> resample native -> log-lambda (fixed 2-tap linear map)     (B, 64, 7,526) 40 km/s
  -> 2 ConvNeXt1D blocks on the 40 km/s grid, 1x1 -> 8 ch       (B, 8, 7,526)
  -> zfm Surface: 40 archetypes, FFT over 12,140 z bins         S(class, z)
  -> top-K 256 (sep 64) -> Reranker (8 layers) + evidence       pick
  -> class head on FM tokens (mean + max) + surface summary
```

- c = 8 into the surface, as in zfm (`c_corr`).
- **40 km/s is built in (owner 2026-10-09: 「40km/s 也要先融入」).** The adapter
  convolutions run on the log grid, so they are redshift-equivariant: shifting
  the input 40 bins shifts the output 40 bins, error 0 [measured, selftest].
  The surface FFT, the z bins and the reranker are on the same grid.
- Code: `zfm_native.py` (train / test / selftest). Selftest [measured]: GPU
  resample = `np.interp` to 2.7e-7; evidence input = the old data.py path to
  2.9e-6; labelled native batches correct. GPU smoke (random FM, 30 steps):
  init z 8.86 / class 1.609 / rank 5.51, as zfm's pre-registered values;
  5.1 GB at 8 per GPU. Head = 17.05M (adapter 0.46M, surface 6.29M,
  reranker 9.90M) + FM 85.83M.
- **The resample is of learned features, not of data.** The FM sees raw
  pixels and raw ivar, so the 0.665 noise-variance problem does not arise.
  The 40 km/s grid survives only where its property is needed: a redshift is
  a shift, so the FFT scan works.
- **Evidence features** (`evidence.py`) are hand-made and need flux·√ivar on
  the log grid. They are computed from the native arrays by the same
  `np.interp` the old pipeline used, as an input to the reranker only. This
  is the one place interpolated data remains; arm E0 drops it to measure
  what it is worth.
- Sub-token precision comes from the un-patchify layer (8 sub-positions per
  token) and the parabola read-out on S. Whether p = 8 is fine enough is
  [not measured]; NMAD vs C4 answers it.

### 8.2b Head U (un-patchify) vs head T (token-reading) — owner 2026-10-09: 「一個展開一個不展開 分開設計 要做比對」

Both read the same FM, both build in 40 km/s, both use the same evidence
features, reranker size (d 320, 8 layers), class-head inputs from the FM
tokens, data, steps and batch. They differ only in how the z candidates and
their scores are made.

| | **U: un-patchify** (§8.2) | **T: token-reading** |
|---|---|---|
| tokens → | back to 7,781 px × 64, resample, 2 ConvNeXt on the 40 km/s grid | never expanded |
| dense z scan | FFT correlation with 40 archetypes, all 12,140 bins, 5 classes | `RestAttn`: 16 learned queries cross-attend to the 973 tokens at each of 759 coarse z bins (16 fine bins = 640 km/s each) |
| 40 km/s | adapter convolutions + FFT shift on the log grid | RoPE on the log-λ axis: keys rotated by the token's position, queries by the trial z shift, so each score depends only on the token's **rest-frame** wavelength |
| candidates | top-256 peaks, sep 64 fine bins | top-256 coarse bins, sep 2 coarse (32 fine) |
| fine z | parabola on log P(z) around the picked bin | second `RestAttn` (32 queries) per candidate → offset ±16 fine bins (smooth L1, weight 0.1) |
| pick | reranker over candidates | reranker over candidates (input: the candidate vector, its coarse log P, evidence at centre+offset) |
| loss | z NLL (12,140) + 0.1 class + 1.0 rank | coarse z NLL (759) + 0.1 class + 1.0 rank + 0.1 offset |
| redshift-equivariance | adapter shift error 0 [measured, selftest] | RestAttn: tokens and trial shift both +137 bins → output change 1.0e-15 [measured, selftest] |

What the comparison can and cannot say:
- It answers "un-patchify + templates vs read tokens directly" as two whole
  designs. It does not isolate one component: T also has no archetype bank.
- T's precision comes from the regressed offset; if T loses only on NMAD and
  not on catastrophic, the token resolution is the limit, not the reading.

CPU check [measured, tiny FM d 64, 4 spectra, 4 AdamW steps]: both heads
train (U z 9.45→5.38; T offset 11.5→4.7, coarse z 6.63 = ln 759 at init,
rank 5.55 = ln 256); every head parameter receives a gradient (only the FM's
unused reconstruction layer does not). GPU speed and memory: not measured yet
(the node is busy with the FM pre-training).

### 8.3 Arms (one seed, 42; same data, steps and head for all)

| arm | FM weights | FM trained? | question |
|---|---|---|---|
| **N-ft** | SPECFMN p8 ep5 | fine-tuned, FM lr = 0.1 × head lr | the redshift product — run with **both heads U and T** |
| **N-fr** | SPECFMN p8 ep5 | frozen | is the FM usable as-is (FM_DESIGN §4 protocol) |
| **N-rand** | random init | fine-tuned, same lr | does pre-training matter, or only the architecture |
| E0 | as N-ft | fine-tuned | no evidence features: is the FM replacing hand-made features |
| ref | zfm C4 (300k) | — | existing 60k clean-2 test result, 0.055–0.060% |

**Scratch-baseline correction (2026-10-09, before any rand result).** The
N-rand arm trains the random FM at 0.1 × lr, the fine-tuning rate. That
handicaps a scratch model, so "ft beats rand" could be the learning rate and
not the pre-training. Added arm **U-rand1**: head U, random FM at the full lr
(`--fm_lr_mult 1.0`). Pre-training counts as helping only if U-ft beats both.

### 8.4 Data and training
- clean-2 only, `--clean 2` explicit (owner rule). z label = Redrock `Z`; class = target bits.
- Test: the same 60k clean-2 objects as C4 (first 60k of the clean-2
  validation blocks). The FM pre-training excluded these blocks, so the FM
  never saw a test spectrum.
- Train size: **owner's decision** (C4 used 300k).
- Optimiser: AdamW, zfm's lr 3e-4 for the head, 3e-5 for the FM, global batch
  32, loss = z NLL + 0.1 class + 1.0 rank (zfm's). Muon is not used on the FM
  (Muon on an AdamW-pretrained model is [not measured]).
- Code needed: labelled native batches in `data.py` (now
  `assert native implies labelfree`); `zfm_native.py` holding the head; the
  FM checkpoint loaded by path.

### 8.5 Success criteria (fixed before running)
On the 60k clean-2 test, counting noise (Poisson on catastrophic counts,
bootstrap over spectra):
1. N-ft catastrophic ≤ C4 within counting noise.
2. N-ft beats N-rand by more than counting noise; otherwise pre-training
   added nothing to z at this label size.
3. NMAD (sub-bin read-out) within 1.5 × C4's 2.5e-5; otherwise p = 8 is too coarse.
4. N-fr is reported, not required to pass.

### 8.6 Results, 60k clean-2 test (same objects and order as C4; one seed)

| arm | catastrophic | n | NMAD (sub-bin) | <0.003 | class |
|---|---|---|---|---|---|
| C4 s42 (zfm) | 0.060% | 36 | 1.97e-5 | 0.9974 | — |
| **U-ft** | **0.052%** | **31** | 2.46e-5 | 0.9983 | 0.957 |
| T-ft | 0.087% | 52 | 8.48e-5 | 0.9974 | 0.960 |

Paired bootstrap (2,000 resamples of test objects):
- C4 − U-ft: +5 [−9, +19]
- C4 − T-ft: −16 [−32, 0]
- U-ft − T-ft: −21 [−37, −6]

Criteria (§8.5) for U-ft:
1. Pass: within counting noise of C4.
2. Pending: the rand arms.
3. Pass: NMAD is 1.25 × C4, inside the 1.5 × limit.

T-ft fails criterion 3 (4.3 ×) and loses to U by more than noise. **The
un-patchify head is the better design as built.** T is a first version
(coarse width, offset regression untuned), so this does not show that
reading tokens cannot work.

Of the 36 + 31 catastrophics of C4 and U-ft, only 8 are shared.

**U-rand** (random FM at 0.1 × lr, same head and recipe):
- 0.065% (39), NMAD 2.34e-5, class 0.947.
- U-ft − U-rand: −8 [−19, +3]. Pre-training is **not detectable on z** at 300k labels.
- Class any-bit: U-ft 0.9571 vs U-rand 0.9472, +0.0099 [+0.0089, +0.0109]. 809 objects are right only with pre-training, 213 only without. Pre-training **helps the class**.
**T-rand:**
- 0.068% (41), NMAD 8.49e-5, class 0.9597.
- T-ft − T-rand: +11 [−3, +25]. The pre-trained FM is not better here, and nominally worse.
- Class: +0.0003 [−0.0002, +0.0009], no difference.
- So under head T, pre-training shows no effect on z or class.

**U-rand1 (random FM at the full lr; the fair scratch baseline):**
- 0.057% (34), NMAD 2.87e-5, class 0.9548.
- U-ft − U-rand1: −3 [−14, +9] on catastrophic.
- Class +0.0023 [+0.0014, +0.0031]: 4× smaller than against U-rand, still above noise.
- So most of U-ft's apparent class gain over U-rand was the 0.1× lr handicap, not pre-training.

**Verdict at 300k labels (one seed):**
- Pre-training adds nothing measurable to z under either head.
- It adds 0.2 points of class accuracy under head U.
- NMAD: U-ft 2.46e-5 vs U-rand1 2.87e-5 (bootstrap not run).
- The architecture (FM-shaped encoder + un-patchify + zfm head) matches C4 without pre-training.
- Per FM_DESIGN §5, this FM does not yet earn the name: criterion 1 (beats random init) holds only weakly, and only for class.

---

## 9. ConvNeXt tokenizer → Transformer for z (2026-10-09 – 10-12)

Owner, 2026-10-09: 「tokenizer 只保留 ConvNeXt：先單獨用重建任務訓練 ConvNeXt 編碼器 + 解碼器（自編碼器，中間沒有 Transformer）；之後由一個 Transformer 讀它的輸出去算 z」.
All of §9 uses the original ZWARN=0 files (no cleaning) and the 20% class mix unless §9.8 says otherwise. Code: `code/convtok/`.

### 9.1 Tokenizer v1: ConvNeXt autoencoder (`convae.py`)
- Input: native BRZ flux + ivar → fixed resample to the 40 km/s log grid (7,526 px). [fact]
- Encoder stages 64 → 192 → 384 → 768 channels: F40 (64 × 7,526) and TOK (941 × 768, 320 km/s each). FSQ bottleneck on TOK. [fact]
- Trained by reconstruction only (Huber δ = 1 on (pred − f)·√ivar), 300k × 5 epochs. [fact]
- Noise-floor check: Huber of N(0,1) = 0.424; a reconstruction loss below it means noise is being copied. [fact]

| FSQ | bits / token | recon (others) | |
|---|---|---|---|
| 15.4 bit | 15.4 | 0.401 | below the floor: copies noise; stopped [measured] |
| 5,5,5,5 | 9.3 | 0.438 | [measured] |
| 5,5,5 | 7.0 | 0.437 | [measured] |

### 9.2 Tokenizer v2: dual-scale (`dualtok.py`, owner table 2026-10-09)
- Narrow tokens 1,882 × 384 (4 bins = 160 km/s); broad tokens 118 × 768 (64 bins = 2,560 km/s); token positions; brightness log10 s. [fact]
- The decoders are forced to one scale: broad → Gaussian low-pass (σ = 12 bins = 480 km/s), narrow → high-pass (x − low-pass). [fact]
- FSQ: narrow 5,5 (4.6 bit), broad 7,7,7,5,5,5 (15.4 bit). Final recon (others) 0.346; the 5,5,5 version 0.290. **Both are below the 0.424 floor, so narrow tokens still pass some noise** (owner chose to proceed). [measured]
- Checkpoint `DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt`, 54.6M parameters, **frozen** in everything below (owner: 「不微调」). [fact]

### 9.3 Redshift Transformer on the tokenizer
- `ztrans.py` (v1 tokenizer): [CLS] + 941 tokens (+ own 8 F40 bins), L 12, d 768. Heads: [CLS] → 759 coarse z bins (16 × 40 km/s); top-8 → 2 decoder blocks with rest-frame RoPE → pick + 33 fine offsets + parabola; [CLS] → class. [fact]
- `ztrans2.py` (ZT2, v2 tokenizer): sequence [CLS] + 118 broad + 1,882 narrow = 2,001 tokens; RoPE angles from token centres in 40 km/s bins (periods 8–32,768), so attention sees wavelength ratios; learned absolute positions; brightness added to [CLS]; H 32. [fact]
- Line names are not used in this sequence: before z is known an observed line cannot be named. [design]

300k × 1 epoch, gbs 32, 46,528-spectrum test (blocks 0–400), one seed [measured]:

| model | heads | catastrophic | NMAD |
|---|---|---|---|
| ztrans, 9.3 bit | 16 | 2.93% | — |
| ztrans, 9.3 bit | 32 | 2.35% | 1.7e-4 |
| ztrans, 7.0 bit | 16 | 2.89% | — |
| ztrans, 7.0 bit | 32 | 2.26% | 1.6e-4 |
| ZT2 (dual) | 32 | 2.38% (class 0.953) | 1.8e-4 |

lr 6e-4 at gbs 128 diverged (held-out catastrophic 26 → 15 → 22.7%); 3e-4 is used from then on. [measured]

### 9.4 ZT2, 1M × 6-epoch schedule, stopped after 5 (owner: 「先5輪」)
gbs 160, lr 3e-4, torch.compile, 35 GB per GPU. Held-out 200k catastrophic by epoch: 1.71 → 1.60 → 1.32 → 1.20 → 1.11%; never stalled. [measured]
Test 200,064 (epoch 5): **1.051%** (2,103), NMAD 1.14e-4, class 0.974; STAR 0.13, BGS 0.35, LRG 0.35, ELG 2.15, QSO 2.27%. [measured]
Note: the LR schedule was built for 6 epochs and stopped at 5, so it never annealed to 0. [fact]

### 9.5 Who is wrong on the catastrophics (rrjudge, repeats)
`rrjudge.py` reads both z (label and model) with the fixed instruments of `zfm/evidence.py`:
- ≥ 6σ lines, 4000 Å break, unexplained ≥ 8σ peaks, contradicting lines
- [O III] / [O II] / Hα–Hβ physics vetoes
- DESI's ELG criterion and a QuasarNet-style broad-line count, reported alongside

On 2,378 controls with a decoy z it never accused Redrock (0.0%). [measured]

ZT5 test errors (1,944):
- 60 have evidence that Redrock is wrong.
- 17 have evidence that the model is wrong.
- 1,867 the rule cannot decide. [measured]

**Rule (owner, 2026-10-11): an undecided error is a limit of the method, not of the data. Without evidence against the label, a disagreement counts as a model error.** Do not report "undecided" as label noise or as "unknown".

Redrock vs Redrock on independent coadds (`repeat_z.py`: other SURVEY/PROGRAM coadd in zall-pix-iron, both ZWARN = 0, |Δz|/(1+z) > 0.1):
- Coverage: only 6,582 of 200,064 test objects (3.3%) have such a repeat.
- 43 conflicts = 0.02% of all; 0.65% of those with a repeat.
- Within the repeat subset:
  - 10 of ZT5's 13 catastrophics are conflicts.
  - 33 of 6,569 agreements are conflicts (0.50%).
  - In 6 of the 10, ZT5 equals the *other* Redrock z. [measured]
- Limits: 3.3% coverage, biased bright (ZT5 catastrophic there 0.20% vs 0.97% overall), n = 13, and agreement between two Redrock runs is not correctness. [fact]

### 9.6 ZT5: 40 km/s detail + line sentences (`ztrans4.py`, `ztrans5.py`)
- **40 km/s detail**: every narrow token also receives its own 4 bins × 67 channels, i.e. tokenizer input (3) + first ConvNeXt stage (64), via LayerNorm + Linear (zero-init). [fact]
- **Line sentence (B)**: for each of the 8 candidates, the sentence is [SOS = decoder state] + 22 rest-frame line words + [EOS]. The line list is `evidence.LINES`, vacuum.
  - Each word = name embedding + a 1-D conv over ±32 bins × 67 ch at the line's position for that z + the encoder's narrow and broad tokens there; lines off the spectrum keep only name + flag.
  - A 2-layer Transformer (d 256, 8 heads, 2.9M) reads the sentence. Its SOS score is added to the pick logit, so it can only re-rank the candidates.
  - No line ratio or co-occurrence is given. [fact]
- 1M × full 6 epochs from scratch, same split / lr / gbs as ZT2. Held-out: 1.65 / 1.32 / 1.21 / 1.12 / 1.03 / 1.02%. [measured]
- Test: **0.972%** (1,944), NMAD 1.12e-4, class 0.975; STAR 0.105, BGS 0.357, LRG 0.342, ELG 2.005, QSO 2.049%. [measured]
- Against ZT2: 559 errors fixed, 400 new. [measured]
- Not separable [fact], because three things changed together:
  - 40 km/s detail
  - line sentences
  - one more epoch with a fully annealed LR

### 9.7 Where ELG / QSO errors concentrate (ZT5 test, `elgqso_strata.py`)
All counted against the label (§9.5 rule). [measured]

ELG (802 / 40,009):
- No ≥ 6σ line at the label z: 18% of ELGs, 9.8% error rate, **86.5% of ELG errors**.
- ≥ 2 such lines: 0.02%.
- DELTACHI2 9–16: 26% error rate. z < 0.6: 8.5%.
- Swaps: [O III] ↔ [O II] 14%; Hα ↔ [O III]/[O II] 10%; model z ≈ 0 6%; half match no line ratio.

QSO (820 / 40,019):
- Redrock SPECTYPE = GALAXY: 26% of QSO targets, 6.6% error rate, **83.5% of QSO errors**; SPECTYPE = QSO: 0.41%.
- No ≥ 5σ broad line at the label z: 90% of errors.
- z < 0.5: 8.6%; z > 2: 0.39%.

DELTACHI2 and SPECTYPE are used here only to describe where errors fall, never as model inputs.

### 9.8 Class-mix experiments (owner-approved exceptions to the 20% mix, training only)
Held-out and test stay 20% each; no control arms (owner).

| run | start | train mix QSO / ELG / LRG / BGS / STAR | epochs, peak lr | test catastrophic | ELG | QSO | LRG | BGS | STAR | class |
|---|---|---|---|---|---|---|---|---|---|---|
| ZT5 | scratch | 20 / 20 / 20 / 20 / 20 | 6, 3e-4 | 0.972% | 2.005 | 2.049 | 0.342 | 0.357 | 0.105 | 0.975 |
| ZT5ft35 | ZT5 | 35 / 35 / 10 / 10 / 10 | +2, 1e-4 | 0.905% | 1.832 | 1.892 | 0.337 | 0.362 | 0.102 | 0.975 |
| ZT5ft50 | ZT5ft35 | 50 / 50 / 0 / 0 / 0 | +2, 1e-4 | **0.845%** | 1.690 | 1.752 | 0.327 | 0.352 | 0.102 | **0.589** |
| ZT5m40 | scratch | 40 / 40 / 6.67 / 6.67 / 6.67 | 6, 3e-4 | 0.904% | 1.800 | 1.904 | 0.355 | 0.355 | 0.105 | 0.973 |

[measured]
- **z of the untrained classes did not degrade under ZT5ft50, but the class head collapsed.** It predicts QSO 52% and STAR 0.3%. A class with zero share kills the class head even when z holds. Fix options:
  - restore the ZT5ft35 class head
  - keep a few % of every class
- Fine-tune gains cannot be separated from "more training", because there were no control arms. [fact]
- Each epoch replays the same 1M stream in the same order, so training loss rises within an epoch where the stream gets harder. Compare the same step across epochs. [measured]
- The early-stop baseline is not seeded with the init model in fine-tunes: ztrans2 starts best at 1e9. [fact, to fix]

ZT5m40 held-out by epoch: 1.515 / 1.326 / 1.158 / 1.017 / 0.957 / 0.939% (ZT5 1.649 / 1.323 / 1.207 / 1.116 / 1.028 / 1.018). BGS/LRG trailed early (+35/+56 at epoch 1) and nearly caught up by epoch 6 (+5/+8). Test: same total as ZT5ft35 (1,808 vs 1,811) from scratch in one run; ELG −82, QSO −58, LRG +5 vs ZT5. [measured]

### 9.9 Infrastructure
- `chain_*.sh` moves a run to a fresh 4 h node after an epoch when the remaining time < 1.15 × epoch + 15 min, and resumes from the per-epoch state. `--stop_epoch`, `--init`, `--mix_train`, `--eval_ckpts` were added to ztrans2. [fact]
- `test_ids.py` recovers TARGETID / rows of a test stream (200,064 / 200,064 Redrock Z identical). [measured]
- Each script's header names the owner instruction it implements. [fact]

### 9.10 Tokenizer v3 (`dualtok3.py`, 2026-10-12)
- Same architecture as v2. Changes, all label-free:
  - masked reconstruction (masked positions weighted 4×)
  - denoising (added noise u/ivar)
  - multi-scale significance loss at 1 / 3 / 9 / 27 / 81 / 243 px
  - train mix QSO 45 / ELG 45 / others 3.33
  - 1M × 5 epochs [fact]
- Reconstruction: plain per-pixel 0.424, exactly the noise floor (v2: 0.387, i.e. copies noise). No scale is below the floor; 243 px is still 1.50 (masked). [measured]
- Injection probe (16k spectra, linear probe at the known position), v3 vs v2:
  - [O II] 5σ: TPR @ 1% FPR 65% vs 60%
  - broad line 8σ: 53% vs 44%
  - both far below the ideal detector (99.6% at 5σ) [measured]
- Downstream z, ZT5 architecture, frozen tokenizer, one seed:

| run | v2 | v3 |
|---|---|---|
| 300k × 1, 20% mix, 60k test | **858 (1.43%)**; ELG 354, QSO 329 | 919 (1.53%); ELG 373, QSO 357 |
| 300k × 10 at 45/45 mix, stopped after epoch 5: held-out ELG+QSO | 811 / 790 / 734 / 697 | 928 / 785 / 750 / 721 |

[measured]
- **Verdict:** v3 does not help ELG/QSO and is slightly worse overall. Better linear readability of injected lines did not carry over to z. v2 stays the tokenizer.
- The objective, the mix and the data size changed together, so the cause is not separated.

### 9.11 Muon + SwiGLU (2026-10-13)
- Code:
  - `muonopt.py` puts Muon (match_rms_adamw) on the hidden 2-D matrices and AdamW on the rest. The rest is biases, norms, convolutions, embeddings and the in/out projections; Muon cannot take 1-D params.
  - `--swiglu` (parameter-matched) is available in the tokenizer ConvNeXt blocks and in every Transformer MLP. [fact]
- Tokenizer DUALTOKMS:
  - v2 objective, Muon 2e-3 + SwiGLU, 20% mix, 1M × 10.
  - Reconstruction 0.243 (others 0.231, bright 0.451) vs v2 0.387 / 0.346 / 1.125: it reproduces far more noise (floor 0.424) and fits bright spectra much better. [measured]
- ZT5MS: ZT5 architecture + SwiGLU + Muon 1e-3, on DUALTOKMS, 45/45 mix, 1M × 6. Test 200k [measured]:

| | catastrophic | ELG | QSO | LRG | BGS | STAR | NMAD | class |
|---|---|---|---|---|---|---|---|---|
| ZT5m40 | **0.904%** | **1.800** | **1.904** | **0.355** | **0.355** | **0.105** | **1.124e-4** | **97.32** |
| ZT5MS | 1.004% | 2.000 | 2.014 | 0.442 | 0.427 | 0.135 | 1.180e-4 | 96.86 |

- **Verdict:** ZT5MS is worse on every class. Four changes were made together, so the cause is not separated. ZT5m40 stays the best model.

### 9.12 Error attribution, by eye and for QSO (2026-10-13)
- `look_elgqso.py`, ZT5m40 test. Wrong and right spectra inside the no-line subgroup have the same S/N per pixel, but the wrong ones differ:
  - lower label z (ELG 0.80 vs 1.23)
  - lower DELTACHI2 (12 vs 42)
  - ELG errors are mostly featureless, or ONE line given another identity
  - `evidence.py` under-counts visible [O II] (the 3-px filter splits the doublet) [measured / by eye, 10 per page]
- `qsojudge.py`: broad lines (FWHM 4,000 km/s, ±1,500 km/s shift, S ≥ 4) at both z; a verdict needs ≥ 2 lines on one side and 0 on the other.
  - Controls: 0 / 1,125 false accusations.
  - ZT5m40 QSO errors (762):
    - label wrong 68, all Redrock SPECTYPE GALAXY
    - model wrong 14, mostly bright BAL QSOs, a real model weakness
    - no evidence 680, counted as model errors
  - QSO catastrophic: 1.904% as counted, 1.734% without the 68. [measured]
- Deep / shallow pairs (`count_deep.py`): our rows are DESI primary, the deepest coadd. In the never-trained validation blocks, a shallower independent coadd with ≥ 2× lower TSNR2 exists for QSO 1,458, ELG 1,252, all 23,005. This is a possible test of model vs Redrock on the same shallow data, against deep Redrock. [measured]

