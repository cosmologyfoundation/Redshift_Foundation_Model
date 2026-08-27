# Method — how the type is read and the redshift is measured

Two steps, in this order. The image decides **what kind of object** it is; a
template fit decides **where** its redshift is. That division is not stylistic:
it was measured. On 90 objects where both were possible, reading the redshift by
eye scored **50/90**, while the template chosen by that same visual call scored
**88/90**. The eye is a good classifier and a poor ruler.

---

## Step 0 — Build the spectrum

DESI delivers three arms separately. They share a common 0.8 Å grid
(B 3600–5800, R 5760–7620, Z 7520–9824), so their wavelengths coincide exactly
in the two overlap regions and the merged grid is their union: **7781 points**.
Following `desispec.coaddition.coadd_cameras`, one expression covers overlapping
and single-arm pixels alike:

```
flux(λ) = Σᵢ ivarᵢ(λ)·fluxᵢ(λ) / Σᵢ ivarᵢ(λ)
ivar(λ) = Σᵢ ivarᵢ(λ)
mask(λ) = OR of the arm masks
```

Masked and non-finite pixels get `ivar = 0`, which is what makes the sum safe to
write unconditionally. Implemented in `code/speclib.py::coadd_cameras`.

---

## Step 1 — Read the type from the image

The spectrum is rendered with **no identifying information** — no class, no
SNR, no `TSNR2`, no programme, only an opaque id — and read as one of four
categories. Note that **BGS and LRG are not among them**; see step 3.

| call | what it looks like |
|---|---|
| **STAR** | Balmer series in absorption at *rest* (3835 / 3889 / 3970 / 4102 / 4341 / 4862 / 6565), often Ca II K 3934, Mg b 5177, Na D 5896, Ca II triplet 8498/8542/8662 — all unshifted |
| **QSO** | broad emission lines. At z ≳ 2 the comb Lyα 1216 / Si IV 1400 / C IV 1549 / C III] 1909 / Mg II 2799 fixes z from several lines at once, with a Lyman-α forest depression blueward of Lyα |
| **ELG** | blue, break-less continuum with narrow emission lines |
| **GALAXY** | a 4000 Å break: continuum low in the blue, rising through a few hundred Å, flat or slowly rising to the red |

### Mandatory checks before committing

**Before calling STAR, look at 6500–8000 Å for emission.** A galaxy at z ≈ 0.1
has a continuum that rises and then turns over, and that shape reads as stellar.
Two objects were lost to this; in both, a strong Hα line sat at the correct
redshift in a part of the spectrum never examined, because the blue-end
continuum had already settled the question.

**Never dismiss a red-end spike as sky without measuring it.** A **23σ** [O II]
line at 9514 Å was discarded as a sky residual purely because of where it fell.
The sky list is ±6 Å around known wavelengths; that line was 55 Å from the
nearest entry and was never actually masked, only mentally.

### When the flux plot is not enough

Faint objects get a four-panel diagnostic (`code/zoom.py`), because a single
filter width is what hid two stars and a broad-line quasar in an earlier round:

| panel | width | finds |
|---|---|---|
| continuum | 121 px | the 4000 Å break, overall shape |
| blue zoom | 3600–5600 Å | Ca II H+K, the Balmer series, the break |
| significance | 70 km/s | nebular emission lines |
| significance | 900 / 3000 km/s | stellar Balmer absorption / quasar broad lines |

**A filter narrower than its feature reports the feature as absent.** This
caused four separate failures: a high-pass narrower than the [O II] doublet, a
90 km/s filter on a 900 km/s stellar line, a 3-pixel filter on a 500-pixel
break, and a y-scale set by the noise instead of the signal. *Measuring nothing
is not evidence that nothing is there.*

---

## Step 2 — Measure the redshift with the family the image chose

Only the template family matching the visual call is fitted, over its own
redshift range (`code/measure3.py`). Two stages:

1. **candidates** — high-passed matched filters, all redshifts at once via one
   FFT, since redshift is a pure translation in ln λ. Recall matters here,
   precision does not. Every model gets a guaranteed quota, because Δχ² from
   differently-filtered models is not comparable across models.
2. **decision** — every candidate refitted against one common, unfiltered data
   vector with shared Legendre nuisance terms.

Two rules learned the hard way and enforced in `code/methodc.py`:

- where a learned basis has no training coverage it must contribute **zero
  weight**, never a prediction of zero flux, or every redshift that shifts data
  into the uncovered region is punished and answers are dragged low;
- the decision stage must use a **single shared data vector** across models.

No abstention: every object gets an answer.

---

## Step 3 — Assign BGS vs LRG from the redshift, not from the image

**This is a redshift cut, not a spectral distinction.** BGS is a magnitude cut
and LRG a colour cut; a bright red galaxy can satisfy both, and Redrock labels
them all simply `GALAXY`. Measured over 20,000 targets:

| | n | z (5–95 pct) | median |
|---|---|---|---|
| BGS | 10,000 | 0.056 – 0.419 | 0.220 |
| LRG | 10,000 | 0.350 – 1.046 | 0.730 |

A single cut **z < 0.42 → BGS** scores **92.6%**. Judging it by eye from the
break edge scored **75%**. Deciding the type first meant guessing a quantity the
next step measures exactly, so the order is inverted for this one pair.

**The ceiling is ~93%, not higher.** 7.4% of the two populations sit on the
wrong side of any cut — real survey design, not a fixable method error. On one
40-object cohort the same cut scored 39/40, which would have been a misleading
thing to report.

---

## Step 4 — Confidence, when abstention is allowed

The best confidence signal is the **matched-filter significance of the class's
defining feature, measured at your own answer** — no truth required. It is
monotonic at every depth (>10σ → ~99% correct, <3σ → ~17%) and beats every χ²
margin rule tried.

Two things that do **not** work as confidence, both tested:

- the **Cramér-Rao bound** σ_z. Useless, and the reason is instructive: **0.0%
  of errors fall within 5σ_z** of the truth while correct answers sit at
  1.9–2.9σ_z. Every remaining error is a *wrong peak*, and at the wrong peak the
  fit is exactly as well determined as at the right one.
- the **template's winning margin** over the runner-up: 96.0% even at 60%
  coverage. A χ² margin does not know whether it is right.

---

## What this implies about where effort is worth spending

| finding | consequence |
|---|---|
| no error is a precision failure (0.0% within 5σ_z) | noise models, weighting, LSF work, finer sampling: **nothing to gain** |
| only 11–15% of errors are missing candidates | improving stage-1 recall caps at ~1.5 points |
| 85–89%: the template prefers the wrong redshift | the templates themselves are the limit |
| refusal catches ~3/4 of all errors | knowing when not to answer beat every modelling change |

The remaining problem is **discrete — which peak, not how precisely.** Most
tempting work addresses the continuous problem, which is already ~50× better
than required.
