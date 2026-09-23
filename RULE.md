# RULE — read this before looking at any spectrum

This is the operating checklist, not the report. `METHOD.md` explains *why*
each rule exists and shows the measurements behind it; this file is what gets
followed. **Read it before every judgement.** When a judgement turns out wrong,
the correction goes *here*, in §6, and only then into `METHOD.md`.

---

## 0. Non-negotiables, before anything else

1. **Look at the image first.** Every object, every time. No call is made from
   a number that was not preceded by looking.
2. **The image pass outputs a type and a list of wavelengths worth measuring —
   never a redshift.** Not even informally, not even in prose. The eye scores
   50/90 as a ruler against 88/90 for the template it selects. Any z stated
   before `significance()` has run is not a result.
   *Enforced:* `measure3.py` exits if `candidates.json` is missing an entry, or
   if any entry contains a redshift.
3. **Rank features by significance, never by height on the plot and never by
   where they sit.** The eye ranks by flux; the correct ranking is flux ÷ local
   noise. At the red end these are nearly uncorrelated.
4. **No Redrock information anywhere** — not `Z`, `SPECTYPE`, `DELTACHI2`, nor
   the `*_PRIMARY` dedup flags. The single exception is `ZWARN == 0` as a
   selection filter.

---

## 1. The five types — decision order

Ask these in order. Stop at the first that answers.

| # | question | instrument | answer |
|---|---|---|---|
| 1 | whole Balmer series in absorption at **rest** (3970 / 4102 / 4341 / 4862 / 6565)? | 900 km/s filter | yes → **STAR** |
| 2 | any feature significant at **3000 km/s** and *not* at 70 km/s? | 3000 km/s filter | yes → **QSO** |
| 3 | **median flux** = `nanmedian(flux[ivar>0])`, in 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹ | the y-axis | **< 0.867 → ELG** · > 0.867 → go to 4 |
| 4 | fitted redshift, *after* step 2 of the measurement | the fit | z < 0.42 → **BGS** · z ≥ 0.42 → **LRG** |

Steps 1–3 are read from the picture. Step 4 is a rule applied to a number:
**BGS and LRG are not spectrally separable.** The z-cut scores 92% against 73%
for the best spectral threshold, and every spectral override tried lowered it.

**Question 3 is the one that caused five repeated errors.** Loud narrow lines
pull the eye to the numerator; the answer is the denominator. Median flux over
120 objects: ELG 0.087–0.808, BGS 0.930–1270. No overlap.

### Shape, as a cross-check on steps 1–3

| | continuum | lines |
|---|---|---|
| STAR | any; often molecular bands (TiO 6200/6800/7100/7600) | absorption at **rest** |
| QSO | smooth power law, often a Lyα forest depression below Lyα | **broad**; two lines with a fixed ratio pin z |
| ELG | faint, flat, **no break**, SNR/px ≈ 1 | 1–4 narrow, 15–40σ; [O II] strongest 10/10 |
| BGS / LRG | a 4000 Å break: low blue → rise → red plateau | break 6–23σ, lines ≲ 5σ (LRG); BGS may have both |

---

## 2. When one glance does not settle it

**Never leave a single tentative call standing.** A lone guess cannot be
eliminated, only defended.

1. Write down **two or three candidate types**.
2. Run **every** candidate's own discriminant — all of them, before deciding.
   `zoom.py` renders the four widths at once so this is cheap.
3. If a candidate's own instrument shows nothing, that candidate is
   **eliminated**. Record which, and why.
4. If everything is eliminated but one, take it and mark it low confidence.
5. If nothing survives, say so. At <3σ the expected accuracy is ~17%.

Measured: this settled 8 of 50 on two separate cohorts, all correctly, including
one object where both candidates were eliminated by measurement.

---

## 3. The redshift

**I do not judge the redshift. The code does.** The type call from §1 selects
the template family; `measure3.py` fits only that family.

Two things outrank the fit, because no redshift can change them:

- **[O III] doublet** — 4960.3/5008.2 = 1.009657, resolved everywhere in range.
  A pair carrying that ratio pins z with no template.
  *Guards:* not for QSO calls, z > 0.005, Hβ ≥ 3σ. Fires 4 of 30 eligible,
  4/4 correct, 0 broken.
- **any fixed line ratio or spectral edge** — a Balmer series at rest, the
  C III]/Mg II ratio 1.4665, a Lyman limit. Both reference errors found so far
  were settled this way, never by a template.

**A redshift within |Δz| < 0.1 of the reference counts as correct.** Standing
rule. Differences inside it are not flagged, figured or discussed.

---

## 4. The SPECTYPE comparison

`SPECTYPE` is Redrock's own 3-way label, and it is **sealed until scoring** — it
is never an input. What is compared against it is my 5-way call collapsed:

| my call | → SPECTYPE |
|---|---|
| STAR | STAR |
| BGS · LRG · ELG | GALAXY |
| QSO | QSO |

So BGS↔LRG↔ELG confusions vanish in this comparison, and it isolates the three
decisions that are genuinely spectral. Current: **197/200**, against **193/200**
for the DESI photometric target bits on the same objects.

**The reference is wrong about 1% of the time** — 2 in 200, both `ZWARN = 0`.
Report Redrock's number as the headline and put the measurement beside it; do
not re-score in my own favour.

---

## 5. Confidence and abstention

- Feature significance **at my own answer** is the confidence: >10σ → ~99%
  correct, <3σ → ~17%.
- An answer resting on a single line is right 6/7; on two or more, 29/30.
- Say "low confidence" when the discriminants come back empty. Over 250 objects
  I said it 3 times and was right to all 3 — the flag is accurate and badly
  under-used.

---

## 6. Corrections log — every rule here was bought with an error

Each row is a rule that exists because it was broken first. **New errors are
appended here before anything else is written.**

| # | the error | the correction now in force |
|---|---|---|
| 1 | dismissed a 23σ [O II] at 9514 Å as sky because of where it sat; repeated on a 26σ line at 7014 Å | rank by significance, never by height or position (§0.3) |
| 2 | stated eyeball redshifts while narrating figures; one was wrong by 0.38 | the image pass emits no redshift, enforced by `measure3.py` (§0.2) |
| 3 | wrote the rule above, then did not record candidate wavelengths on the next cohort | `measure3.py` refuses to run without a per-object entry (§0.2) |
| 4 | template fit explained one 36σ line as [O II]; it was half of an obvious doublet — 102,876 km/s | the [O III] doublet veto (§3) |
| 5 | called five bright break galaxies ELG because the lines were loud | median flux is question 3, before shape (§1) |
| 6 | left single tentative calls standing on ambiguous objects | enumerate candidates and eliminate (§2) |
| 7 | proposed rescoring by total line evidence: fixed 3/3 failures, broke 10/47 successes | every fix is run against the answers already right, before adoption |
| 8 | reported a 5-way score falling 47→45→43 without checking why | a metric moving one way three times gets checked, not explained |

### How to add a row

State the error in one line with its cost in km/s or in objects. State the
correction as something *checkable*. If the correction cannot be given a
consumer that fails loudly, say so explicitly rather than writing it down and
hoping — **more prose is not a repair**.
