"""Step 2: redshift from the family the picture selected, then BGS/LRG from z.

The image separates STAR / QSO / ELG / GALAXY.  For GALAXY it does NOT try to
call BGS versus LRG, because that is not a spectral distinction: measured over
20,000 targets it is a redshift cut at z~0.42 worth 92.6%, while judging it by
eye from the break edge scored 75%.  So the galaxy families are fitted together
and the label is assigned from the fitted redshift afterwards.

No abstention.  The sealed key is not read here.
"""
import os, json
import numpy as np
from multiprocessing import Pool

R = os.environ.get('C3', '/pscratch/sd/s/shunth/claude_learning_map/cohort3')
ZCUT = 0.42
FAMILY = {'STAR': 'STAR', 'BGS': 'BGS', 'LRG': 'LRG', 'ELG': 'ELG',
          'QSO': 'QSO', 'NLAGN': 'ELG'}
# A call may carry a trailing '?', meaning "this is a guess".  The family is
# the same either way -- the mark changes what is reported, not what is fitted.
# There is no abstention: refusal was measured at 2 saved against 3 right
# answers thrown away, and every costly refusal was several weak lines agreeing
# on one redshift, which the strongest-peak test cannot see.
# Standing instruction, 2026-09-06: "looking with your eyes means telling the
# five types apart -- STAR QSO BGS LRG ELG."  So the image pass emits the full
# five-way call and BGS/LRG is no longer assigned from the fitted redshift.
# 'GALAXY' stays accepted because fifteen earlier cohorts were recorded with it
# and their calls.json files are the record of what was actually said.
WANTED = {'STAR': {'STAR'}, 'QSO': {'QSO'}, 'ELG': {'ELG'},
          'BGS': {'BGS'}, 'LRG': {'LRG'}, 'GALAXY': {'BGS', 'LRG'}}
_G = {}


def _init():
    import methodc as MC, engine as E, d4000
    mc = MC.MethodC()
    _G.update(mc=mc, E=E, MC=MC, d4=d4000,
              fam=[FAMILY[m.name.rstrip('a')] for m in mc.s2])


# Standing instruction, 2026-09-06: "having guessed the type, when you compute
# the redshift you must run EVERY formula and look at EVERY region that needs
# looking at, and only then decide the most likely z -- not only the regions
# that type needs."
#
# So the family no longer selects the template set.  Every family is scored on
# a candidate list built from every family's own coarse correlation, and the
# per-family best is recorded.  The call then enters as a PRIOR: the called
# family wins ties, and another family takes the answer only if it beats the
# called one by MARGIN.  MARGIN = inf reproduces the old family-gated fit;
# MARGIN = 0 is a pure winner-take-all across families.  Measured on 506
# objects across 15 cohorts (replay14.py, split.py):
#
#                       total          type RIGHT      type WRONG
#   gated (old)     480/506 94.9%    473/484 97.7%     7/22 31.8%
#   ungated (0)     481/506 95.1%    468/484 96.7%    13/22 59.1%
#
# Overall the two are indistinguishable.  The whole effect of the gate is on
# the 22 objects whose type was called wrong, where it is the mechanism of the
# 43x lever: gating turns a type error into a redshift error, and removing it
# rescues 6 more of them.  The price is 5 objects whose type was right, three
# of them a star that the QSO family explains at z ~ 2.998.  A veto on
# overrides landing near a family's redshift-limit edge was tested to remove
# those and does not: 2.998 is mid-grid for the QSO family (limits -0.01 to
# 6.0), so the preference is real and not an optimiser artefact.  Rejected.
MARGIN = float(os.environ.get('MARGIN', '0'))
TYPEFAC = float(os.environ.get('TYPEFAC', '3'))
ELG_BREAK = float(os.environ.get('ELG_BREAK', '6'))


def _all_families(a):
    """Return {family: (z, d)} for every family, plus the candidate count."""
    fl, iv, call = a
    mc, E, MC, d4, fam = _G['mc'], _G['E'], _G['MC'], _G['d4'], _G['fam']
    D, _, _ = mc.sc.coarse(fl, iv)
    cand = set()
    # every family's own coarse correlation contributes candidates, so no
    # region is skipped because the called type had no reason to look there
    for i in range(len(mc.s1)):
        for k in E.peaks(D[i], nmax=4, sep=50):
            cand.add(round(E.ZS[k], 5))
    for bz, _, _ in d4.break_candidates(fl, iv, nmax=3):
        cand.add(round(bz, 5))
    cands = sorted(cand)
    best = {}
    for f in ('STAR', 'BGS', 'LRG', 'ELG', 'QSO'):
        sel = [m for m, ff in zip(mc.s2, fam) if ff == f]
        if not sel:
            continue
        r = MC.score(fl, iv, sel, cands, fine=True)
        if r:
            best[f] = (float(r[0]['z']), float(r[0]['d']))
    return best, len(cands)


def _one(a):
    call = a[2]
    best, ncand = _all_families(a)
    if not best:
        return 0.0, {}, 0
    keep = WANTED[call]
    mine = max(((f, v) for f, v in best.items() if f in keep),
               key=lambda t: t[1][1], default=None)
    top = max(best.items(), key=lambda t: t[1][1])
    if mine is None:
        pick = top
    elif top[0] in keep or top[1][1] - mine[1][1] <= MARGIN:
        pick = mine
    else:
        pick = top
    # Correction 24.  The reported TYPE follows the fit only when the fit is
    # decisive.  On cohort19/OBJ029 the STAR family beat the called GALAXY by
    # 546 to 230 and the object is a star; the call was reported anyway.
    # Measured over 496 objects against the DESI target class, taking the
    # fit's family when it beats the called one by a factor:
    #
    #   never   441/496          x3   445/496  (+4 / -0)   <- adopted
    #   x4      444/496 (+3/-0)  x2   445/496  (+5 / -1)
    #                            x1   441/496 (+11 / -11)
    #
    # x3 is the largest factor that gains and the only one that loses nothing.
    # Below it the fit starts overruling correct calls.  An earlier version of
    # this comment also quoted a 3-way column scored against Redrock SPECTYPE;
    # that is Redrock output and is not a source this project learns from, so
    # it has been removed.  The choice is unchanged: x3 is selected by the
    # target-bit column on its own.
    newfam = None
    if (mine is not None and top[0] not in keep and mine[1][1] > 0
            and top[1][1] > TYPEFAC * mine[1][1]):
        newfam = top[0]
    return pick[1][0], best, newfam


if __name__ == '__main__':
    d = np.load(os.path.join(R, 'data', 'cohort.npz'), allow_pickle=True)
    calls = json.load(open(os.path.join(R, 'data', 'calls.json')))

    # The image pass is required to emit, per object, the wavelengths worth
    # measuring -- METHOD.md step 1.  On cohort6 that was written as prose and
    # simply not done: only a list of "needed zoom" ids was recorded, so when
    # the template fit returned z=1.16 for an object whose [O III] doublet was
    # plainly visible, there was nothing on disk to contradict it.  Refusing to
    # run is what makes the rule bind.
    cp = os.path.join(R, 'data', 'candidates.json')
    if not os.path.exists(cp):
        raise SystemExit('candidates.json is missing: the image pass must '
                         'record, per object, the wavelengths worth measuring '
                         'before any redshift is fitted.')
    cand = json.load(open(cp))

    # The image pass may record WHERE to look.  It may not record a redshift,
    # not even informally: the eye scores 50/90 as a ruler against 88/90 for
    # the template it selects, and an eyeball z that reaches the page is a
    # published wrong answer.  This guard is here BEFORE the rule has failed
    # in this form -- the pattern of the previous three failures was that a
    # rule got a mechanism only after it broke.
    import re as _re
    _Z = _re.compile(r'\bz\s*[=~\u2248]\s*-?\d', _re.I)
    stated = sorted(k for k, v in cand.items()
                    if isinstance(v, str) and _Z.search(v))
    if stated:
        raise SystemExit(
            'candidates.json states a redshift for %d object(s): %s. The image '
            'pass records wavelengths worth measuring, never a redshift -- '
            'remove them and re-run.' % (len(stated), ', '.join(stated[:5])))

    missing = [k for k in calls if not isinstance(cand.get(k), str)]
    if missing:
        raise SystemExit(
            'candidates.json is missing per-object entries for %d object(s), '
            'e.g. %s. The image pass must record what it saw for every object, '
            'not just the hard ones -- otherwise a wrong fit has nothing to be '
            'checked against.' % (len(missing), ', '.join(sorted(missing)[:5])))
    _SL_LAM = d['lam']
    ids = [str(x) for x in d['id']]
    GUESS = {k: str(v).endswith('?') for k, v in calls.items()}
    calls = {k: str(v).rstrip('?') for k, v in calls.items()}
    bad_call = sorted(k for k, v in calls.items() if v not in WANTED)
    if bad_call:
        raise SystemExit(
            'calls.json has %d call(s) that are not one of %s (optionally with '
            'a trailing "?" to mark a guess): %s. Abstaining was removed on '
            '2026-09-05 -- always answer, and mark a guess.'
            % (len(bad_call), '/'.join(sorted(WANTED)), ', '.join(bad_call[:5])))

    # RULE.md 0.4 / correction 14.  Every filter reports WHERE its maximum
    # sits.  On OBJ037 of cohort14 the 3000 km/s filter peaked at 225 sigma at
    # 8643 A -- H-alpha at z=0.316, the largest number in the whole cohort --
    # and the image pass wrote "broad emission bump at 4400 A" and never
    # mentioned 8643.  The red half had been dismissed as a sky forest.  Cost:
    # type wrong and dz = 0.911 on the brightest object in the set, and the
    # third repeat of "dismissed by position instead of by measurement".
    #
    # So the wavelength of the largest discriminant must appear in that
    # object's candidate string, or be named after the word "rejected".
    # Writing it down does not force it to be interpreted correctly -- but not
    # writing it down is exactly what happened three times.
    import speclib as _SL2
    _THR = 8.0
    _bad = []
    for _i, _o in enumerate(ids):
        _fl, _iv = d['flux'][_i], d['ivar'][_i]
        # only the two LINE-finding filters.  The step statistic's maximum is a
        # continuum location, not a candidate line, and on a stellar continuum
        # it is meaningless (correction 13) -- including it made this guard
        # fire on half the cohort, which is how a guard gets routed around.
        _best = (0.0, 0.0, None)
        for _w in (70.0, 3000.0):
            _sg = _SL2.significance(_fl, _iv, _w)
            _j = int(np.nanargmax(_sg))
            if float(_sg[_j]) > _best[0]:
                _best = (float(_sg[_j]), float(_SL_LAM[_j]), _w)
        if _best[0] < _THR:
            continue
        _nums = [float(x) for x in
                 _re.findall(r'\b(\d{4})(?:\.\d+)?\b', cand[_o])]
        if not any(abs(x - _best[1]) < 60 for x in _nums):
            _bad.append((_o, _best[1], _best[0], _best[2]))
    if _bad:
        raise SystemExit(
            'candidates.json never mentions the wavelength where the strongest '
            'line filter peaks, for %d object(s):\n%s\n'
            'Go to that wavelength and look. If it is not the feature you are '
            'using, name it in the candidate string and say why -- "7026, TiO '
            'band head, rejected". Dismissing a whole region as sky is not a '
            'rejection: sky lines are narrow, and the 3000 km/s filter does '
            'not survive them.'
            % (len(_bad), '\n'.join(
                '   %-8s %8.1f A  %6.1f sigma  (%g km/s filter)'
                % (o, l, t, w) for o, l, t, w in _bad[:12])))

    # RULE.md 5 / correction 15.  An ELG call needs a line.
    #
    # This guard exists because correction 12 -- the same error, written as
    # prose after cohort13 -- did not stop it happening three more times in
    # cohort14.  Across 400 objects, ELG calls with a 70 km/s peak at or above
    # 8 sigma are 76/76 on redshift; below it, 5/14.  Below the line the
    # reader must mark the call as a guess with a trailing '?', so the guess is
    # on the record rather than hidden inside a confident-looking answer.
    _thin = []
    for _i, _o in enumerate(ids):
        if calls[_o] != 'ELG':
            continue
        _pk = float(np.nanmax(_SL2.significance(d['flux'][_i], d['ivar'][_i],
                                                70.0)))
        if _pk < 8.0 and not GUESS[_o]:
            _thin.append((_o, _pk))
    if _thin:
        raise SystemExit(
            'ELG called on %d object(s) whose strongest 70 km/s peak is under '
            '8 sigma, without marking the call as a guess:\n%s\n'
            'Median flux ranks galaxies against each other; it is not a test '
            'for being a galaxy. Above 8 sigma an ELG call is right 76/76; '
            'below it, 5/14. Answer anyway -- write "ELG?" so the guess is on '
            'the record.'
            % (len(_thin), '\n'.join('   %-8s strongest line %4.1f sigma'
                                      % (o, p) for o, p in _thin)))

    _qso = []
    for _i, _o in enumerate(ids):
        if calls[_o] != 'QSO' or GUESS[_o]:
            continue
        _br = float(np.nanmax(_SL2.significance(d['flux'][_i], d['ivar'][_i], 3000.0)))
        _na = float(np.nanmax(_SL2.significance(d['flux'][_i], d['ivar'][_i], 70.0)))
        if _na > _br:
            _qso.append((_o, _br, _na))
    if _qso:
        raise SystemExit(
            'QSO called on %d object(s) whose narrow filter outscores the broad '
            'one, without marking the call as a guess:\n%s\n'
            'Over 101 QSO calls, broad >= narrow is 88/93 right and narrow > '
            'broad is 4/8. Answer anyway -- write "QSO?" so the guess is on the '
            'record.'
            % (len(_qso), '\n'.join('   %-8s broad %5.1f  narrow %5.1f'
                                     % t for t in _qso[:12])))

    # RULE.md 5.2 / correction 21.  The evidence floor.
    #
    # OBJ045 of cohort18 contains nothing above 3.9 sigma anywhere: not a line,
    # not a break, not a slope.  The redshift it was given (1.4301 against a
    # reference 0.2750) rested on a 2.1 sigma bump, and the reference itself is
    # only reachable through 2.5 sigma.  Measured over 770 claims across 16
    # cohorts, binned by the strongest of the three filters ANYWHERE in the
    # spectrum: < 4 sigma is 9/14 right, 4-6 is 63/74, and >= 6 is 665/682
    # (97.5%).  A < 6 sigma flag marks 11% of objects and carries 16 of the 33
    # wrong redshifts.
    #
    # As a refusal it would be worthless -- it discards 72 correct answers to
    # catch 16 wrong ones -- which is the third discriminant in a row to come
    # out weak as a gate and strong as a flag.  So it demands the mark, never
    # the silence.
    _floor = []
    for _i, _o in enumerate(ids):
        if GUESS[_o]:
            continue
        _fl, _iv = d['flux'][_i], d['ivar'][_i]
        _b = max(float(np.nanmax(_SL2.significance(_fl, _iv, 70.0))),
                 float(np.nanmax(_SL2.significance(_fl, _iv, 3000.0))),
                 float(np.nanmax(np.abs(_SL2.local_step_significance(_fl, _iv)))))
        if _b < 6.0:
            _floor.append((_o, _b))
    if _floor:
        raise SystemExit(
            '%d object(s) have nothing above 6 sigma anywhere -- no line, no '
            'break, no slope -- and the call is not marked as a guess:\n%s\n'
            'Below 6 sigma the redshift is right 72/88; above it, 665/682. '
            'Answer anyway, and add the trailing "?": the spectrum does not '
            'contain the answer, so the number is a draw from the noise no '
            'matter how the fit reports it. If the type is also a guess, guess '
            'a galaxy class: of 93 below-6-sigma objects across 16 cohorts, 81 '
            'carry a galaxy TARGET BIT (87%) and NONE carries a star bit, '
            'against a 71% galaxy base rate -- and in the 4-6 sigma band it is '
            '72 of 78 (92%) with 0 stars. Measured on the imaging target bits, '
            'not on Redrock output.'
            % (len(_floor), '\n'.join(
                '   %-8s strongest feature anywhere %4.1f sigma' % t
                for t in _floor[:12])))

    call = np.array([calls[i] for i in ids])
    FL, IV = d['flux'], d['ivar']
    with Pool(int(os.environ.get('SLURM_CPUS_ON_NODE', 64)),
              initializer=_init) as p:
        res = p.map(_one, [(FL[i], IV[i], call[i])
                           for i in range(len(ids))], 2)
    z = np.array([r[0] for r in res])
    perfam = [r[1] for r in res]
    newfam = [r[2] for r in res]
    json.dump({i: {f: list(v) for f, v in b.items()}
               for i, b in zip(ids, perfam)},
              open(os.path.join(R, 'data', 'perfamily.json'), 'w'), indent=1)
    _ov = [(i, c, max(b, key=lambda f: b[f][1]))
           for i, c, b in zip(ids, call, perfam)
           if b and max(b, key=lambda f: b[f][1]) not in WANTED[c]]
    print('  every family scored; the best-fitting family differs from the '
          'call on %d/%d object(s)%s'
          % (len(_ov), len(ids),
             (': ' + ', '.join('%s %s->%s' % t for t in _ov[:8])) if _ov else ''))
    # --- structural veto: an [O III] doublet outranks the template fit ------
    # The doublet ratio is fixed by atomic physics, so where a resolved pair
    # carries it the redshift is not a matter of template preference.  Applied
    # only to non-QSO calls, and only when H-beta corroborates; see
    # speclib.oiii_doublet_redshift for why each guard exists.  Conflicts are
    # printed rather than applied silently.
    import speclib as _SL
    C_KMS = 299792.458
    n_veto = 0
    for i in range(len(ids)):
        if call[i] in ('STAR', 'QSO'):
            continue
        dbl = _SL.oiii_doublet_redshift(_SL_LAM, FL[i], IV[i])
        if dbl is None:
            continue
        zd = dbl[0]
        if abs(zd - z[i]) / (1 + zd) * C_KMS < 1000:
            continue
        print(f'  [O III] doublet veto  {ids[i]}: template z={z[i]:.4f} -> '
              f'doublet z={zd:.4f}  (pair {dbl[1]:.1f}/{dbl[2]:.1f} A, '
              f'{dbl[3]:.0f} sigma)')
        z[i] = zd
        n_veto += 1
    print(f'  doublet veto applied to {n_veto} object(s)')

    # correction 24: the reported family follows the fit where the fit is
    # decisive (TYPEFAC), otherwise it is the call
    fam_call = call.copy()
    _flip = []
    for i in range(len(ids)):
        if newfam[i] is not None:
            _flip.append((ids[i], call[i], newfam[i]))
            fam_call[i] = ('GALAXY' if newfam[i] in ('BGS', 'LRG')
                           else newfam[i])
    if _flip:
        print('  the fit beat the call by more than %gx on %d object(s), so the '
              'reported type follows it: %s'
              % (TYPEFAC, len(_flip),
                 ', '.join('%s %s->%s' % t for t in _flip)))

    # Correction 28.  An ELG call must be checked against the CONTINUUM at
    # its own fitted redshift, not only against the line that produced it.
    #
    # This is lens-elg-called-on-lines-alone recurring for the third time, and
    # the first two versions were prose.  Cohort20 lost OBJ032 (an LRG at
    # z = 0.272 whose H-alpha/[N II] pair I read correctly and then called ELG)
    # and OBJ044 (an LRG at z = 0.938) exactly this way: the lines gave the
    # right redshift and the 4000 A break at that redshift was never looked at.
    #
    # Measured on all 184 ELG calls across 19 cohorts, 166 with a usable break
    # amplitude at the fitted z.  Median break: 1.25 sigma when the call is an
    # ELG target, 2.64 sigma when it is not.  As a reassignment:
    #
    #   flag at   fires   not-ELG among them   saved   broken
    #     >= 3      23            10             10       13
    #     >= 4      15             8              8        7
    #     >= 5       9             6              6        3
    #     >= 6       6             5              5        1     <- adopted
    #     >= 8       1             0              0        1
    #
    # At 6 sigma all five reassignments land on the right class once the
    # z < 0.42 cut is applied, and the one break is cohort14/OBJ019, a real
    # ELG target with an 11.8 sigma break at z = 1.02.  Net +4 on the 5-way.
    # This is not a refusal -- it swaps one positive answer for another -- so
    # the "marking is free, refusing is not" asymmetry does not apply and the
    # saved/broken count is the whole test.
    import d4000 as _d4
    _swap = []
    for i in range(len(ids)):
        if fam_call[i] != 'ELG':
            continue
        _zs, _D, _S = _d4.d4000_scan(FL[i], IV[i])
        _j = int(np.argmin(np.abs(_zs - z[i])))
        if abs(_zs[_j] - z[i]) > 0.01 or not np.isfinite(_S[_j]):
            continue
        if float(_S[_j]) >= ELG_BREAK:
            # assign the split here rather than routing through 'GALAXY':
            # fam_call is a '<U4' array and 'GALAXY' truncates to 'GALA',
            # which the later `fam_call == 'GALAXY'` test silently misses --
            # the run then REPORTS the reassignment and writes a broken label.
            fam_call[i] = 'BGS' if z[i] < ZCUT else 'LRG'
            _swap.append((ids[i], float(_S[_j]), float(z[i]), fam_call[i]))
    if _swap:
        print('  %d ELG call(s) have a >= %g sigma 4000 A break at their own '
              'fitted redshift, so they are break galaxies: %s'
              % (len(_swap), ELG_BREAK,
                 ', '.join('%s -> %s (%.1f sigma, z %.3f)'
                           % (t[0], t[3], t[1], t[2]) for t in _swap)))

    # A five-way call stands as it is.  The z < 0.42 cut is applied only to a
    # legacy 'GALAXY' call, which is what the older cohorts recorded.
    assert set(np.unique(fam_call)) <= {'STAR', 'BGS', 'LRG', 'ELG', 'QSO',
                                        'GALAXY'}, np.unique(fam_call)
    final = fam_call.copy()
    g = fam_call == 'GALAXY'
    if g.any():
        final[g] = np.where(z[g] < ZCUT, 'BGS', 'LRG')
        print('  %d legacy GALAXY call(s) split by the z < %g cut; five-way '
              'calls were left alone' % (int(g.sum()), ZCUT))
    guess = np.array([GUESS[i] for i in ids])
    np.savez(os.path.join(R, 'data', 'pred.npz'), id=np.array(ids),
             call=call, mytype=final, z=z, tid=d['tid'], cls=d['cls'],
             famfit=np.array([x or '' for x in newfam]),
             snr=d['snr'], guess=guess)
    print(f'wrote pred.npz for {len(ids)} objects')
    if guess.any():
        print(f'  {int(guess.sum())} call(s) marked as a guess: '
              f'{", ".join(np.array(ids)[guess])}')
    for c in ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']:
        m = final == c
        if m.any():
            print(f'  {c:5s} n={int(m.sum()):3d}  z {z[m].min():7.4f} '
                  f'- {z[m].max():7.4f}')
    print(f'  of the {int(g.sum())} read as GALAXY: '
          f'{int((final[g]=="BGS").sum())} BGS, {int((final[g]=="LRG").sum())} LRG')
