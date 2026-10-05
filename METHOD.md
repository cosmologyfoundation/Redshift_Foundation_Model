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
SNR, no `TSNR2`, no programme, only an opaque id.

### The decision order, start to finish

The five classes are not read in one step. Three questions are asked in this
order, each answered by a different part of the plot, and each cheap enough to
answer before the next:

| # | question | what to look at | outcome |
|---|---|---|---|
| 1 | is the whole Balmer series in absorption at **rest**? | 900 km/s filter at 3970 / 4102 / 4341 / 4862 / 6565 | yes → **STAR**, done |
| 2 | is any emission feature **broad** (thousands of km/s)? | 3000 km/s filter | yes → **QSO**, done |
| 3 | how **bright** is the continuum? | the y-axis: median flux | > 0.867 → BGS or LRG · < 0.265 → **ELG**, *but only if a 70 km/s peak reaches 8σ* (step 4) |
| 4 | for the bright ones only: what is the fitted redshift? | step 2, after the type call | z < 0.42 → **BGS** · z ≥ 0.42 → **LRG** |

Only the first three are read from the picture; the fourth is a rule applied to
a number, because BGS and LRG are **not spectrally separable** — see step 3.

**Question 3 has a precondition.** Median flux ranks galaxies against each
other; it is not a test for being a galaxy, and applying it without a detected
line produced three confident ELG calls on a star, a quasar and a z = 1.01
galaxy. See *The one mandatory refusal* in step 4.

**Every filter also reports where its maximum sits, and that wavelength must be
visited.** A discriminant peak is a candidate: either it is the feature being
used, or the candidate record must name it and say why it was rejected. The
cost of not doing this was measured on one object — the 3000 km/s filter peaked
at **225σ at 8643 Å**, which is Hα at z = 0.316, and the red half of the
spectrum had been written off as sky residual without measuring it. The type
came out QSO instead of GALAXY and the redshift was wrong by Δz = 0.911, on the
brightest object in the cohort (SNR 27.7). Sky lines are narrow; the 3000 km/s
filter does not survive them, which is exactly why its maximum there was real.

`measure3.py` refuses to run when the strongest of the 70 and 3000 km/s filters
exceeds 8σ at a wavelength the candidate record never mentions. On the cohort
that produced the error it fires on 10 of 50 objects and catches both type
failures; the other eight are stellar band heads and Balmer wings, dismissed in
one clause each. The step statistic is deliberately excluded — its maximum is a
continuum location rather than a candidate line, and including it made the
guard fire on half the cohort, which is how a guard gets routed around.

**Definitions, so the thresholds above are usable rather than decorative.**

- **median flux** is `numpy.nanmedian(flux[ivar > 0])` over the whole coadded
  spectrum — the median of every pixel the pipeline did not mask, with no
  continuum fitting and no smoothing. The masked pixels must be excluded or the
  zeros drag the median down.
- **the units are DESI's**, 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹, exactly as they come out of
  `desispec` — so **0.867 is 0.867 × 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹**. A threshold in
  flux is only meaningful in a stated calibration; re-derive it on your own
  data rather than porting the number.
- **the 900 / 3000 km/s filters** are `speclib.significance(flux, ivar,
  width_kms=…)`: subtract a running-median continuum, then convolve with a
  Gaussian of that velocity width and divide by the propagated noise, so the
  output is in units of σ at every pixel. 70 km/s matches nebular emission,
  900 km/s stellar Balmer absorption, 3000 km/s quasar broad lines.
- **"broad" in question 2** means a feature that is significant on the 3000
  km/s filter and *not* on the 70 km/s one; a narrow line is the reverse. That
  contrast, not the apparent width on the plot, is the test.

**Question 3 is the one that took five repeated errors to learn.** The
continuum's height, not its lines, is what separates ELG from the bright
galaxies: median flux over 120 objects is **0.087–0.808** for ELG against
**0.930–1270** for BGS, with no overlap at all, and the rule scored 30/30 on a
fresh 15 + 15 draw. The shape table below describes what each class *looks*
like; the height decides between the two that look alike.

| call | what it looks like |
|---|---|
| **STAR** | Balmer series in absorption at *rest* (3835 / 3889 / 3970 / 4102 / 4341 / 4862 / 6565), often Ca II K 3934, Mg b 5177, Na D 5896, Ca II triplet 8498/8542/8662 — all unshifted |
| **QSO** | broad emission lines. At z ≳ 2 the comb Lyα 1216 / Si IV 1400 / C IV 1549 / C III] 1909 / Mg II 2799 fixes z from several lines at once, with a Lyman-α forest depression blueward of Lyα |
| **ELG** | **median flux below ~0.87** — a faint, flat, break-less continuum at SNR/px ≈ 1 carrying almost no information, with one to a few narrow emission spikes standing **15–40σ** above it. The *ratio* is the signature, not either half. [O II] 3727/3729 is the strongest line in 10 of 10 — DESI selects ELGs as [O II] emitters |
| **GALAXY** | **median flux above ~0.87** — a 4000 Å break: continuum low in the blue, rising through a few hundred Å, flat or slowly rising to the red, and **no emission lines at all**. Measured on ten LRGs: break **6.0–22.9σ**, best line median **2.3σ** — the exact inverse of the ELG row above |

**The ratio of the two statistics is the type call.** Measured on two fresh
cohorts of ten, same code, at each object's own fitted redshift:

| | 4000 Å break | best emission line | line / break |
|---|---|---|---|
| ELG | median 1.2σ (max 5.3) | median 29.5σ | **27** |
| LRG | median 15.2σ (min 6.0) | median 2.3σ | **0.15** |

Zero overlap in either column. Each class is invisible to the other class's
instrument, which is also why the type call has to come first: it decides which
statistic carries the redshift.

**For an ELG, the count of visible lines brackets the redshift before any
fitting.** Verified 10/10 on a dedicated cohort:

| what is visible | implied z | why |
|---|---|---|
| [O II] + Hβ + [O III] | z < 0.96 | [O III] 5008 leaves the red end at 0.96 |
| [O II] + Hγ, no Hβ | 1.02 < z < 1.26 | Hβ leaves at 1.02, Hγ at 1.26 |
| [O II] alone (+ [Ne III]) | z > 1.23 | everything else has left |

The lines run out one at a time, from the red end, in a fixed order. This is
also the cheapest way to feel the **redshift desert**: the 4000 Å break leaves
the window at **z = 1.46**, Ca H/K at 1.49 and [O II] at 1.63, while Lyα does
not enter until 1.96. Between 1.63 and 1.96 a galaxy without an AGN has nothing
in the window at all — measured on the corpus, that interval holds 8027 quasars
against 1282 galaxies, a factor of 95. Above z = 1.5 the difficulty is a
property of the class, not of the redshift.

**The commonest reading error is using only the line half of that ratio.**
Four times across three blind tests a break galaxy was called ELG because
strong narrow lines took the eye while the continuum was bright and had a
break: measured at the fitted redshift, those four have breaks of 8.8, 5.3, 4.0
and 2.9σ at SNR/px 2.5–5.4, against 28 correct ELG calls with a median break of
**0.6σ** at a median SNR/px of **1.1**. Check the denominator before calling
ELG: a bright continuum with a measurable break is not one, however loud the
lines.

*Pre-registered, then tested out of sample:* "break > 5σ at the fitted z ⇒ not
an ELG" catches 2 of those 4 and costs 0 of the 28. On the next cohort, drawn
after the threshold was fixed, it wrongly rejected **0 of 10** correct ELG calls
(largest break among them 3.1σ). Cost is now confirmed on **48 correct calls with 0 wrong
rejections**. The benefit test arrived on a later cohort: a fifth break galaxy
was called ELG and its break measures **exactly 5.0σ**, which the strict
inequality misses. Running score: **cost 0 of 48, benefit 2 of 5.** The
threshold is *not* being lowered to catch it — that is tuning on the failure,
which is what turned the rescoring rule into a regression. It remains a flag,
not an override.

**Checking the continuum before calling ELG worked.** In that cohort six objects
had strong narrow lines on a bright or break-shaped continuum; all six were
called GALAXY and all six were right. The confusion went 1, 1, 2 → **0**, and
the 5-way type score 43/50 → **49/50**.

### When the picture does not settle the type, enumerate and eliminate

Reading the image is the first step and it is not optional. But a single
glance settles the type only when one signature is unmistakable — a Balmer
series at rest, a broad-line comb with a Lyman forest, a faint flat continuum
under a 30σ spike. When it does not:

1. **Write down every type the picture could support.** Two or three, not one.
   A single tentative call is the failure mode: it becomes the answer by
   default and nothing tests it.
2. **Run every candidate's own discriminant, all of them, before deciding.**
   Each class is measured with the instrument that class is visible to — and
   each is nearly invisible to the others':

   | candidate | what to measure | what confirms it |
   |---|---|---|
   | STAR | 900 km/s filter at the *rest* Balmer wavelengths | absorption at rest; also check 6500–8000 Å for emission before committing |
   | QSO | 3000 km/s filter | a broad feature, and a second line at the same z; below z ≈ 2.5 look for Mg II + C III] with a fixed ratio, above it for Lyα with a forest and a Lyman break |
   | ELG | 70 km/s filter, and the 4000 Å break at the candidate z | lines 15–40σ **and** a break under ~5σ on a faint flat continuum — the ratio, not either half |
   | GALAXY | the 4000 Å break at the candidate z | a break of 6–23σ with lines ≲ 5σ |

3. **Only then commit to a type, and only then fit the redshift.** The type
   call selects the template family, so a type decided casually is a redshift
   decided casually.

On the SV3 test this settled **8 of 50** that one glance did not. Three were
the STAR-versus-low-z-galaxy pair, decided by a ±15 to ±40σ absorption forest on
the 900 km/s panel. One had *both* candidates eliminated — nothing above 3σ on
either the 70 km/s or the 3000 km/s filter, so neither ELG nor QSO — which left
the break-shaped continuum as the only surviving reading. Enumerating makes
elimination possible; a single tentative call cannot be eliminated, only
defended.

The four-width diagnostic (`code/zoom.py`) exists to make step 2 cheap: it puts
the continuum, the 70 km/s, the 900 km/s and the 3000 km/s views side by side,
so all four discriminants are read at once rather than one being guessed.

### ELG versus BGS: read the y-axis, not the lines

These two are the pair that caused every repeated type error, and they separate
on **continuum brightness with no overlap at all**. Median flux over 120
studied objects: BGS **0.930 – 1270**, ELG **0.087 – 0.808**. Pre-registered
threshold **median flux > 0.867 ⇒ BGS**, then tested on 15 + 15 drawn
afterwards: **30/30**. **That threshold separates ELG from BGS and nothing
else** — measured afterwards on LRG as well, 38 of 60 LRG fall below it and 36
below the faintest BGS, so the three-class rule is three bands: < 0.265 is ELG
(55 ELG, 1 LRG), 0.265–0.867 is ambiguous and 23% of galaxies land there, and
above 0.867 is never an ELG (0 of 60). A threshold is only valid over the
classes it was measured on, with the eye reading the y-axis alone also scoring
**30/30** and agreeing with the rule on every object. On that fresh draw the
gap was ELG max 0.262 against BGS min 1.220.

Line-based alternatives are much weaker, measured on the same 120: SNR/px 98.3%
but with a real overlap (BGS min 2.23 against ELG max 5.82), red/blue continuum
ratio 82.5%, count of >5σ lines 59.2%. H-alpha is *in the window* for 60/60 BGS
and only 6/60 ELG — the redshift ranges genuinely do not overlap, BGS being
z = 0.05–0.4 and ELG z = 0.6–1.6 — but only **37/60** BGS show it above 3σ,
because many BGS are passive, so "H-alpha visible ⇒ BGS" is 86% precision, not
a glance.

**The decisive case:** an ELG at z = 0.112, right inside the BGS redshift range,
where every line- or redshift-based argument says BGS. Its median flux is 0.105,
four times below the faintest BGS in the sample, and brightness called it
correctly. BGS is *defined* by r < 20 and ELG by g < 23.6 regardless of
redshift, so this reads the selection rather than the physics — a legitimate
visual cue, but it will fail wherever the two populations overlap in brightness.

### How that discriminant was found, which matters more than the discriminant

**Correct physics is not a usable rule.** Reasoning from the target definitions
gives a clean prediction — the redshift ranges do not overlap, so H-alpha is
inside the window for every BGS and outside it for every ELG — and the
prediction is exactly right: 60/60 against 6/60. It is still useless as a
discriminant, because a rule needs the feature to be *seen*, and physics only
guarantees it *exists*. Only 37/60 BGS show H-alpha above 3σ; the rest are
passive. "H-alpha visible ⇒ BGS" is 86% precision, which is not a glance.

**So enumerate and score candidate quantities rather than reasoning to a single
answer.** Four candidates were measured on the same 120 objects — median flux
100%, SNR/px 98.3% but with a real overlap, red/blue continuum ratio 82.5%,
count of >5σ lines 59.2%. The winner was not the one the physics pointed at,
and it would not have been found by thinking harder about the physics. Scoring
several is cheap; being right about which one to derive is not.

**The pre-registration cycle, run end to end for the first time.** Threshold
fixed on 120 earlier objects, written down, then tested on 30 drawn afterwards:
30/30, with the eye agreeing on every one. Compare the other pre-registered
rule in this file, "break > 5σ ⇒ not an ELG", still standing at cost 0 of 48
and benefit 2 of 5 — registering a threshold does not make it a good one, it
only makes the answer honest.

**A note on where the answer was.** The discriminant is the height of the
y-axis, present on every render from the first cohort onward. It was missed
through five repeated errors because attention went to the lines, which is
where the interesting information is. The least informative part of the plot
carried the cleanest separation.

### The image pass produces a type and candidate positions — never a redshift

This is a hard constraint, not a preference. The eye scored **50/90** as a
ruler against **88/90** for the template the same eye selected. So the output
of this step is:

1. one of the four calls above, and
2. a list of **wavelengths worth measuring** — not what they are, not what z
   they imply.

**Any redshift stated before `significance()` has been run is not a result and
must not be written down**, including informally while narrating a figure.

This is enforced, not requested: `measure3.py` **refuses to run** unless
`data/candidates.json` holds a per-object entry for every object in
`calls.json`. The rule existed as prose for one cohort and was simply not
followed — the wavelengths were never recorded, so when the fit returned a
wrong redshift there was nothing on disk to contradict it. See *Why rules are
written as mechanisms* at the end of this file. The
failure mode this prevents is recorded below, and it recurred *after* the rule
existed in prose, because a prohibition on a conclusion cannot be obeyed at a
moment when the measurement needed to obey it has not been taken yet.

### Mandatory checks before committing

**Before calling STAR, look at 6500–8000 Å for emission.** A galaxy at z ≈ 0.1
has a continuum that rises and then turns over, and that shape reads as stellar.
Two objects were lost to this; in both, a strong Hα line sat at the correct
redshift in a part of the spectrum never examined, because the blue-end
continuum had already settled the question.

**Rank candidate lines by significance, never by apparent height, and never by
position.** The error runs in both directions and both have now happened:

- a **23σ** [O II] line at 9514 Å discarded as a sky residual purely because of
  where it fell (the sky list is ±6 Å around known wavelengths; that line was
  55 Å from the nearest entry and was never actually masked, only mentally);
- and, on a later cohort, a **26σ** [O II] line at 7014 Å dismissed the same way
  while a **2.1σ** bump near 5580 Å was promoted in its place — a 0.38 error in
  z, from a visual pass.

The mechanism is that **the eye ranks by height in flux; the correct ranking is
height in units of the local noise.** Where sky emission is strong the pipeline
`ivar` is small, so a tall flux excursion scores low, and a modest one in a
quiet window scores high. At the red end of a faint spectrum the two orderings
are close to uncorrelated.

There is also no reason to apply a positional sky prior on top of the
statistic. Measured across ten faint ELG panels, the bright airglow wavelengths
(5577, 6300, 6364, Na 5890) **never once exceeded 4σ** — `significance()`
already suppresses sky through the inverse variance. Adding a positional prior
on top of it can only subtract information.

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

### The [O III] doublet overrides the fit

`4960.3 / 5008.2 = 1.009657`, whatever the redshift, and DESI resolves the pair
everywhere it is in range (77 Å apart at z = 0.6, ~96 pixels). A resolved pair
carrying that ratio therefore pins z with **no template, no continuum model and
no line-strength assumption**. It is a structural constraint, not a score, so it
outranks the fit (`speclib.oiii_doublet_redshift`, applied in `measure3.py`).

It exists because on a 50-object blind test the fit explained a single 36σ line
at 8054 Å as [O II] at z = 1.160, when that line was the red half of a 19σ
doublet at z = 0.608 — a 102,876 km/s error against a double peak visible in
the raw flux plot.

Three guards, each added because the unguarded version failed on it: skip
objects called QSO (narrow peaks on broad lines hit the ratio by chance and
overturned two correct quasar answers), require z > 0.005 (those chance pairs
implied z = −0.14 and −0.16), and require Hβ above 3σ at the implied z.
Measured: fires on 4 of 30 eligible objects, agrees with the reference 4/4,
rescues 1 failure, breaks 0 correct answers — **47/50 → 48/50**.

**Tested and rejected: making the fit agree with the recorded candidate
wavelengths.** `candidates.json` had been write-only, which is the shape of
every dead rule here, so the obvious repair is to require the fitted redshift to
explain at least one wavelength the image pass wrote down. Over 117 objects with
a usable candidate it catches **1 of 6** failures and raises **4 false alarms
among 111 correct answers** — one catch per four false alarms, because eyeball
break positions are not accurate to 300 Å. Not adopted.

**Every proposed fix is run against the answers that were already right, not
only against the failures.** A rescoring rule tried in the same session fixed
3 of 3 failures and overturned 10 of the 47 successes (47 → 40); it was
discarded. The successes are where a fix does its damage, because that is where
the mass is.

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

A single cut **z < 0.42 → BGS** scores **92.6%**, and re-tested on 100 BGS/LRG
targets from five blind tests it scores **92%** against the best spectral
threshold's **73%** — and that 73% is optimistic, because the threshold was
tuned on the same objects it was scored on. Adding a spectral override on top
of the cut only lowers it (91% at break > 20). In the overlap region
0.20 < z < 0.45 the BGS breaks run 1.6–22.2σ and the LRG breaks 5.1–36.9σ:
different medians, overlapping ranges. **BGS and LRG are selection criteria,
not spectral classes** — BGS is magnitude-limited, LRG is colour-selected, and
many BGS targets are themselves passive red galaxies. The break-versus-line
ratio separates passive from emission-line galaxies with zero overlap, and
separates nothing within the passive population. Judging it by eye from the
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

### The one mandatory refusal: an ELG call with no line

Confidence is usually advisory. There is one case where it is not, and it was
found by taking three faint objects and calling them ELG on brightness alone.

The rule that produced them is question 3, `median flux < 0.265 → ELG`, which
is 98% pure — *on galaxies*. It was measured on 180 galaxies and it ranks
galaxies against each other. It says nothing about whether the object in front
of it is a galaxy at all. Applied without that precondition it converted three
objects with no measurable feature into confident ELG calls; they are a **STAR
at z = 0.0002**, a **QSO at z = 0.6988**, and a **galaxy at z = 1.0103**.

The precondition, measured over 400 objects across eight cohorts:

| ELG calls | n | redshift right |
|---|---|---|
| strongest 70 km/s peak **≥ 8σ** | 76 | **76 (100%)** |
| strongest 70 km/s peak **< 8σ** | 14 | **5 (36%)** |

So: **an ELG call requires a 70 km/s emission peak at ≥ 8σ.** The 14 that fail
it are 3.5% of the sample; the remaining 386 are right 97.9% of the time.

Two things make this rule trustworthy rather than a threshold fitted to three
failures:

- **It changes no call.** It only converts a silent wrong answer into a
  declared refusal, so it cannot break anything that already works — the
  failure mode that killed evidence rescoring (3/3 fixed, 10/47 broken).
- **The scope is the finding, not the number.** Across *all* objects a
  strongest peak below 5σ still gets the redshift right 89% of the time,
  because a break and a stellar Balmer series do not need emission lines. A
  missing line is fatal only inside an ELG call, where the line *is* the
  evidence.

And there is no better answer than refusal. On the three cases the standard
fallback "take the largest discriminant" (§2.5 of `RULE.md`) scores **0 of 3** —
it points at ELG, GALAXY and QSO respectively, and the truths are STAR, QSO and
GALAXY. When nothing was measured, nothing is the answer.

---

## A selection cut is a cut on the evidence

Four SV3 blind tests scored 50, 49, 43 and 44 out of 50. The first reading of
that was that the reader had got worse. It is measurable, and it is wrong.

Tests X and XI were built with no redshift limit; XII and XIII with
`--zmax 1.6`. That single parameter changes what the QSO **target class**
contains:

| | X + XI (no cut) | XII + XIII (`z ≤ 1.6`) |
|---|---|---|
| QSO targets that are quasars by `SPECTYPE` | **20 / 20** | **9 / 20** |
| their redshift range | 1.61 – 2.50 | −0.00 – 1.46 |
| errors among QSO targets | 0 / 20 | 11 / 20 |

The mechanism is geometry. Over 3600–9824 Å a quasar's broad UV lines enter and
leave the window at fixed redshifts: Mg II 2799 at z = 0.286, C III] 1909 at
0.886, C IV 1549 at 1.324, Lyα 1216 at 1.962. So **below z = 0.886 a quasar has
exactly one broad line in range**, and the `z ≤ 1.6` cut raises the share of the
sampled redshift range where QSO has at most one feature from **25% to 55%**.

Grouping all 40 QSO targets by how many of those four lines fell in the window:

| broad UV lines in range | n | read correctly |
|---|---|---|
| 4 | 9 | **9 / 9** |
| 3 | 14 | 13 / 14 |
| 2 | 5 | 3 / 5 |
| **1** | 4 | **0 / 4** |
| 0 (not a quasar at all) | 8 | 4 / 8 |

This is the single-line law of step 4 — one line 6/7, two or more 29/30 —
appearing at cohort scale. Brightness moves with it and cannot be cleanly
separated at n = 40 (the one-line group is also the faintest, median flux
0.26–0.34), but the line count is the quantity the cut changed *by
construction*, and it is the one with a mechanism: a single line cannot be
checked against a second.

Two consequences, both now enforced:

- `make_cohort3.py` prints what the requested redshift range leaves in the
  window for each class **before it draws anything**, and flags any class the
  cut makes thinner than the unrestricted range. It is pure geometry — rest
  wavelengths and the spectrograph range — so it leaks nothing. The flag
  compares against the full 0–3.5 range on purpose: an earlier version warned
  whenever a class was thin *anywhere*, which fired on all four classes every
  time and would have been ignored within a session.
- **A score is not comparable across cohorts until the selection is.** Of the
  seven errors in test XIII, five are QSO targets and two are not; the two that
  are not — a bright galaxy whose Hα was discarded with the rest of the red end,
  and an ELG called with no line — are reading errors that the sample does not
  excuse.

---

## The controlled test of the selection effect

The claim that the score drop across cohorts XII-XIV was the *selection* and not
the reader was tested by changing one variable. Cohort XV was drawn exactly like
XIV -- same survey, same classes, same `ZWARN = 0`, same procedure, `RULE.md`
unread -- with `--zmax 1.6` removed.

| cohorts | selection | redshift \|dz\| < 0.1 |
|---|---|---|
| VI-XI (six) | no redshift cut | 50, 49, 50, 48, 50, 49 -- mean **49.3** |
| XII-XIV (three) | `--zmax 1.6` | 43, 44, 45 -- mean **44.0** |
| **XV** | **cut removed** | **48 / 50** |

The score returned. In cohort XV every quasar target is a quasar and every one
sits at z = 1.86-2.90, where three to five broad UV lines are in the window at
once; all eleven were read correctly. Under the cut, quasar targets were quasars
9 times in 20 and the survivors had one line.

**A score is not comparable across cohorts until the selection is.**

### Two things cohort XV added

- **At z > 2 the Lyman-alpha forest edge is a break.** The coded form of
  question 2 -- broad must exceed both the narrow line and the break -- would
  have rejected **7 of the 11 correct quasar calls**, because the sharp drop
  blueward of Lyman-alpha outscores the broad lines on the step filter. The eye
  does not make this mistake, because it sees the forest and the line series
  together.
- **Refusal is not free.** The single abstention was a faint quasar whose
  strongest feature was 4.2 sigma; refusing was correct by the measured rule,
  and the ungated fit would nevertheless have got it right to \|dz\| = 0.012.
  Over two cohorts abstention has now saved one wrong answer and cost one right
  one. Not enough to move the threshold; enough to keep counting.

---

## The type call is a 43× lever on the redshift

Measured over **449 claims across nine cohorts** (one abstention removed):

| | redshift right | redshift wrong | P(wrong) |
|---|---|---|---|
| type right | 422 | 7 | **1.6%** |
| type wrong | 6 | 14 | **70.0%** |

A wrong type multiplies the redshift failure rate by **43×**, and **67% of all
redshift failures** (14 of 21) sit downstream of a type error. This is a
property of the *architecture*, not of any particular rule: the type call
selects which template family is fitted, so a type error removes the correct
answer from the search before the search begins.

The obvious repair is to remove the gate — fit every family, rank by χ². It was
run on cohort15's five failures and recovered **2 of 5**:

- one was recovered outright (the fit found the right redshift under a family
  the call had excluded);
- one was a low-redshift degeneracy where STAR and GALAXY give the same z;
- for **two** the correct redshift was **never in the candidate set** at all, so
  no amount of re-ranking could reach it;
- for **one** the correct family did reach the reference redshift and χ²
  preferred the wrong family anyway.

So ungating is necessary and not sufficient: with a shared candidate set the
failure moves from the gate to the ranker, which is where 85–89% of errors
already live. Both have to be right.

### Why a new hand statistic does not fix it

Three of cohort15's five failures had a **35–52σ 4000 Å break at the reference
redshift** that no step of the procedure looked for. The natural fix — scan
`break_amplitude` over redshift and take the maximum as a GALAXY detector — was
tested on all 450 objects:

| SPECTYPE | n | median max-break | p90 |
|---|---|---|---|
| GALAXY | 271 | 9.4σ | 22.5σ |
| QSO | 76 | 34.3σ | 101.1σ |
| **STAR** | 103 | **39.2σ** | 163.8σ |

**90% of stars produce a ≥10σ "break" somewhere in 0 < z < 1.6.** The statistic
is a detector for steep continua, not for breaks — which is exactly what
correction 13 recorded and what `break_amplitude`'s own docstring says. It was
re-derived anyway, which is one more data point for the prose-does-not-hold
finding.

What actually separates a star from a galaxy with a break is **coherence**: a
single redshift that explains the break *and* Ca H/K *and* the G band *and*
Mg b at once. No single statistic carries that; a joint fit does. Which is the
argument for removing the gate rather than adding a discriminant.

---

## Two repairs that were tested and rejected

Both were proposed to fix specific failures, both were run against the 400
objects already read, and both were dropped. They are recorded because the
argument for each still looks good in prose.

### Hα / [N II] as a structural veto — rejected

The [O III] doublet veto works because 4960.3/5008.2 = 1.009657 is fixed by
atomic physics. **Hα / [N II] 6564.6/6585.3 = 1.003147** and **[S II]
6718.3/6732.7 = 1.002144** are equally fixed, and as a *reading aid* the ratio
is excellent: it identified the line and pinned the redshift on six objects of
one 50-object cohort by eye.

As an override of the template fit it fails. Over 400 objects it fires on 54
non-STAR/QSO calls, agrees with the fit on 51, **fixes 0 and breaks 1** — the
break being a passive galaxy at z = 0.726 pushed to z = −0.02 by a pair of
noise peaks at rest. The separation is only 21 Å, so at DESI's resolution the
ratio is far less selective than [O III]'s 48 Å, and a 5σ threshold admits
sky-residual pairs freely: on one bright object the search returned six pairs
between 40σ and 47σ, of which one was real.

### Relaxing the break veto on the QSO branch — rejected

Two quasars were lost in one cohort because the local-step statistic fired on
the blue edge of the very broad line that should have identified them: on
OBJ042 the step reads 12.0σ at 4679 Å and the 3000 km/s filter reads 7.3σ at
4711 Å — 32 Å apart, one feature — and question 2 requires broad to exceed the
break.

The repair "if the two peaks are within 150 Å, they are one feature, so the
break may not veto" was tested at gaps of 60, 100, 150 and 250 Å:

| gap | calls flipped to QSO | quasars recovered | correct calls broken |
|---|---|---|---|
| 60 Å | 25 | 1 | 24 |
| 100 Å | 38 | 2 | 35 |
| 150 Å | 39 | 2 | **36** |
| 250 Å | 61 | 2 | 57 |

The reason is structural: **a 4000 Å break *is* a broad feature**, so the step
and the 3000 km/s filter peak in the same place on almost every galaxy and cool
star — 231 of 400 objects. Co-location carries no information.

What the test did establish is the size of the real problem. Only **4 quasars
in 400** are missed this way, and every one has a broad significance below 9σ.
That is the faint-quasar limit — a single weak Mg II bump on a faint continuum —
and it is not fixed by reordering the questions.

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

---

## The reference has an error rate of its own

`ZWARN = 0` means Redrock did not flag itself. It does not mean Redrock is
right, and the two are not the same claim.

In test VII, OBJ032 carries the whole Balmer series in absorption **at rest** —
Hδ −128σ, Hγ −144σ, Hβ −126σ, on a spectrum with SNR/px 61.6 — and its DESI
target class is STAR. Redrock places it at **z = 1.6536 with ZWARN = 0**, an
offset of **186,803 km/s**, scored as a failure of the reading. At that
redshift every diagnostic line is off the red end, the break window falls
outside the spectrograph entirely, and the C IV the solution implies lands on
Hδ absorption.

**A second one, and the rate.** OBJ004 in test IX: Redrock QSO z = 3.9703
against z = 1.1755, offset 168,575 km/s, `ZWARN = 0`. Three arguments, none
needing a template. The two broad lines sit at 4152.4 and 6089.4 Å, a ratio of
**1.4665**, and Mg II 2799.1 / C III] 1908.7 is **1.4665** — at z = 3.97 the
4152 Å line would be 835 Å rest, where a quasar has no strong line, leaving a
10.4σ feature unexplained. At z = 3.97 the Lyman limit lands at 4533 Å and the
object should be black below it; the measured flux there is **0.347** against
**0.260** redward of the supposed Lyα, so the blue side is *brighter*. And the
Lyα forest region shows no depression at all.

That is **2 references wrong in 200 objects, about 1%** — a floor on any blind
score against this reference, invisible unless each failure is examined rather
than counted. Note what settled both: a series at rest, a fixed line ratio, a
spectral edge. **Quantities no redshift can change are the only things that
outrank a fit** — the same family as the [O III] doublet veto.

**A redshift within |Δz| < 0.1 of the reference counts as correct.** This is a
standing rule, not a per-report choice. It is the identification criterion: at
z = 0.5 it is about 20,000 km/s, so it asks whether the right object at roughly
the right distance came out, and a difference inside it is not flagged, not
figured and not discussed.

**Report both redshift criteria.** ±1000 km/s (|Δz|/(1+z) < 0.003) is the
standard catastrophic-failure threshold and measures precision. |Δz| < 0.1 —
about 20,000 km/s at z = 0.5 — measures identification instead: whether the
right object at roughly the right distance came out. Over the 200 blind objects
the two give **191/200** and **197/200**. Quote whichever is asked for, but say
which one it is: the looser number is not a redshift accuracy, and it forgives
at least one true line misidentification (|Δz| = 0.098, 22,480 km/s) that lands
just inside it.

So a blind score is a score against a particular reference, never against
truth. **Report the reference's number as the headline and put the measurement
beside it**; do not quietly re-score in your own favour.

---

## Why rules here are written as mechanisms

Twice in one session a rule in this file was broken while its text was in front
of me. The account "I forgot" is wrong and picks the wrong repair — the rule
had already been written down twice. What actually separates the rules that
hold from the rules that do not:

| | cost of complying | cost of skipping | outcome |
|---|---|---|---|
| state no redshift from the image | one fewer sentence | — | kept |
| record the wavelengths worth measuring | 50 objects, one by one | none | dropped |
| never dismiss a red-end spike as sky | a measurement not yet taken | none | dropped twice |

Both halves of the first rule sit in the same paragraph. The free half survived
and the expensive one did not, and the result still looked compliant because
nothing downstream read the file the rule was supposed to produce.

Two repairs, matching two failure shapes:

1. **Skipping is free** → give the rule a consumer that fails loudly.
   `measure3.py` now exits rather than run without `candidates.json`;
   `code/docsync.py` exits non-zero when a finding is missing from these docs.
2. **The rule cannot be obeyed when it is needed** → a prohibition on a
   *conclusion* is unusable at a moment when the measurement needed to obey it
   has not been taken. Rewrite it as a constraint on the *output* of that step.

What does not work: writing it down again, in more places, more emphatically.
**More prose is not a repair.**

**The generalisation, which took three failures to reach.** Each repair above
was built for the rule that had just broken, and each held -- the rules that
got a mechanism did not fail again. What kept failing were the rules whose turn
had not yet come, because every new rule was born as prose. The repair rate was
one rule per failure, and every rule written since was queued behind it.

So: **a rule gets a consumer at the moment it is written, before it fails.**
Two rules that had not yet broken are now enforced by `code/slideguard.py` (all
three identities on a per-object figure; the reader is named Opus 5) and by the
redshift check in `measure3.py`'s candidates guard. If a rule cannot be given a
consumer, say so explicitly rather than writing it down and hoping.
