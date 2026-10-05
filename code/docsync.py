"""Assert that every finding is reflected in METHOD.md / README.md.  Exit
non-zero when one is not.

This exists because the rule "write what you learned into the docs" was
followed for half a session's findings and silently dropped for the other half.
Complying costs a rewrite; skipping cost nothing, and nothing checked -- so
skipping won.  A rule with no consumer is not a rule.

Run it after every blind test, alongside check.py.  When it fails, the repair
is to write the finding into the docs, never to delete the row.

Adding a finding: append a row with a marker string that MUST appear in
METHOD.md or README.md.  Choose a marker that cannot survive a rewrite that
drops the substance -- a measured number is ideal, a common word is useless.
"""
import os, sys, io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (finding, marker that must appear in the docs)
FINDINGS = [
    ('blind protocol: no Redrock in selection',      'ZCAT_PRIMARY'),
    ('ZWARN==0 is a required selection filter',      'ZWARN==0'),
    ('ZWARN==0 is not the same as correct',          '186,803'),
    ('the eye is a classifier, not a ruler',         '50/90'),
    ('image pass emits no redshift',                 'never a redshift'),
    ('rank lines by significance, not height',       'height in units'),
    ('candidates.json is enforced, not requested',   'refuses to run'),
    ('never dismiss a red-end spike as sky',         '9514'),
    ('ELG signature: faint continuum, 15-40 sigma',  '15–40σ'),
    ('LRG signature: break 6.0-22.9 sigma',          '6.0–22.9'),
    ('break vs line ratio separates the classes',    'line / break'),
    ('line count brackets an ELG redshift',          'leaves the red end at 0.96'),
    ('the redshift desert above z = 1.5',            'desert'),
    ('[O III] doublet overrides the fit',            '1.009657'),
    ('every fix is tested against the successes',    'already right'),
    ('BGS vs LRG is a redshift cut',                 '92.6%'),
    ('no spectral rule beats the BGS/LRG cut',       'selection criteria,\nnot spectral classes'),
    ('feature significance is the confidence',       '10σ'),
    ('four blind tests, 200 objects',                '191/200'),
    ('the looser |dz|<0.1 criterion is reported too', '197/200'),
    ('|dz| < 0.1 counts as correct, as a standing rule', 'standing rule, not a per-report'),
    ('enumerate candidate types when the picture is unclear', 'enumerate and eliminate'),
    ('SV3 is deeper, so its score is not comparable',  '2–4× the per-pixel SNR'),
    ('the target bits score worse than the image',   '193/200'),
    ('the reference has a ~1% error rate',           '1.4665'),
    ('candidate-agreement check was rejected',       'one catch per four'),
    ('rules are written as mechanisms',              'cost of skipping'),
    ('a new rule gets a consumer when written',      'before it fails'),
    ('ELG called on the line half alone',            'Check the denominator'),
    ('ELG vs BGS separates on brightness',           'read the y-axis, not the lines'),
    ('the flux threshold is three bands, not one',   'only valid over the\nclasses it was measured on'),
    ('take-the-largest fallback was measured wrong', 'MEASURED WRONG'),
    ('an ELG call requires a line',                  'ELG call requires a 70 km/s emission peak'),
    ('refusal is mandatory below 8 sigma',           '386 are right 97.9%'),
    ('visit the wavelength each filter points at',   '8643'),
    ('the Ha/[N II] ratio veto was rejected',        'fixes 0 and breaks 1'),
    ('the break/broad co-location repair was rejected', '231 of 400'),
    ('the type call is a 43x lever on the redshift', '**43×**'),
    ('the selection effect was tested by controlling it', 'The score returned'),
    ('the Lyman-alpha forest edge is a break',        '7 of the 11 correct quasar calls'),
    ('scanning the break over z is a star detector', '**39.2σ**'),
    ('the design doc exists and is driven by the measurements', 'DESIGN'),
    ('the design removes the type gate',              'never a switch in front of'),
    ('the design forbids a width-based class prior', 'Width is evidence about the *line*'),
    ('alias negatives must span classes',            'must span\n   classes'),
    ('refusal is not free, and the design says so',  '2 saved, 3 lost'),
    ('abstention was removed in favour of marked guesses', 'do not refuse at all'),
    ('an unconstrained fit is worse than a guessed constraint', 'worse than a guessed constraint'),
    ('marking is free so weak discriminants become flags', 'Marking is free'),
    ('the evidence floor is global, and 6 sigma is where the unmarked group plateaus',
     'evidence floor is a property of the spectrum'),
    ('the class must not gate the template set at inference',
     'class must not select the template set at inference'),
    ('the target bits and SPECTYPE are different objects',
     'two scores are different objects'),
    ('the class output is five-way, not a three-way call plus a z cut',
     'class output is five-way'),
    ('the DESI target label is a bitmask and 13.6% name more than one class',
     'label is not always a single class'),
    ('nothing is learned from Redrock output, and the redshift label is open',
     'The label problem, stated rather than assumed'),
    ('the scaling plan is three costed runs and the problem is data-limited',
     'this is a data-limited problem'),
    ('NMAD is saturated and measures the other mode of a bimodal error',
     'NMAD is already saturated'),
    ('the residual block is ConvNeXt, measured at matched parameters',
     'Which residual block'),
    ('an ELG call is checked against the continuum at its own fitted redshift',
     'median break at the fitted z'),
    ('the QSO narrow-beats-broad flag is 50% vs 95%',  '101 QSO calls across twelve'),
    ('the training files contain the eval set',      '978 of the 1000 TARGETIDs'),
    ('the training sample is doubly selected',       'doubly selected on Redrock having succeeded'),
    ('the two h5 files can be pooled without repeats', 'share **0 TARGETIDs**'),
    ('TSNR2 is present and verified against the record', '691 agree to 0.1%'),
    ('the two dedup rules pick different spectra',    'select different spectra of the same target'),
    ('the reader output is audited but not consumed',  'positions asserted by the front end'),
    ('refusal is decided on corroboration',            '**2 saved, 3 lost**'),
    ('a selection cut is a cut on the evidence',      'from **25% to 55%**'),
    ('a checklist propagates a wrong rule',           'propagates a wrong rule faster'),
    ('QSO targets below z=1.6 are mostly not quasars', '**9 / 20**'),
    ('correct physics is not a usable rule',         'not a usable rule'),
    ('the five-class decision order, in order',      'The decision order, start to finish'),
    ('RULE.md is a record, not a pre-read checklist', 'not read before judging a spectrum'),
    ('RULE.md carries the corrections log',          'Corrections log'),
    ('the thresholds carry units and a recipe',      'erg s'),
    ('median flux is defined operationally',         'nanmedian(flux[ivar > 0])'),
    ('enumerate and score candidate quantities',     'enumerate and score'),
    ('the pre-registration cycle completed',         'pre-registration cycle'),
    ('the pre-registered threshold is not retuned',  'cost 0 of 48, benefit 2 of 5'),
]

# The notebook is a deliverable too, and leaving it out of this scan is how a
# checker gives false assurance: docsync reported 21/21 green while the report
# was three blind tests and 150 objects out of date.  A mechanism that checks
# the wrong artefact is worse than none, because it says "ok".
docs = {}
for name in ('METHOD.md', 'README.md', 'RULE.md', 'DESIGN.md',
             'report/redshift_report.ipynb'):
    p = os.path.join(ROOT, name)
    docs[name] = io.open(p, encoding='utf-8').read() if os.path.exists(p) else ''
blob = '\n'.join(docs.values())

# findings that must appear in the notebook specifically, not only in the docs
NOTEBOOK_ONLY = {'three blind tests, 150 objects',
                 'the target bits score worse than the image',
                 'ZWARN==0 is not the same as correct',
                 '[O III] doublet overrides the fit'}

nb = docs.get('report/redshift_report.ipynb', '')
# A marker replace once turned "GALAXY" into "GALA1.009657Y" in METHOD.md and
# this checker did not notice, because it only asks whether markers are present.
# Also assert the class names are intact.
CLASSES = ['**STAR**', '**QSO**', '**ELG**', '**GALAXY**']
broken = [c for c in CLASSES if c not in docs.get('METHOD.md', '')]
if broken:
    print('CORRUPTED CLASS NAME(S) IN METHOD.md: %s' % ', '.join(broken))

missing = [(f, m) for f, m in FINDINGS if m not in blob]
missing += [(f + '  [class name corrupted]', f) for f in broken]
missing += [(f + '  [not in the notebook]', m) for f, m in FINDINGS
            if m in blob and f in NOTEBOOK_ONLY and m not in nb]
print('doc coverage: %d findings, %d documented, %d missing\n'
      % (len(FINDINGS), len(FINDINGS) - len(missing), len(missing)))
for f, m in FINDINGS:
    where = ' '.join(n.split('.')[0] for n, t in docs.items() if m in t) or '--'
    print('  %-6s %-46s %s' % ('ok' if m in blob else 'MISS', f, where))
if missing:
    print('\n%d FINDING(S) NOT IN THE DOCS:' % len(missing))
    for f, m in missing:
        print('   %-46s (marker %r)' % (f, m))
    sys.exit(1)
print('\nall findings are documented')
