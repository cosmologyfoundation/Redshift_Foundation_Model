"""Assert the invariants the learning loop depends on.  Exit non-zero if broken.

Run after every pick.py.  The defect this exists to catch stayed hidden for two
sessions precisely because every individual run *looked* fine: the batch file
appeared, the spectra fetched, the analysis produced numbers.  Nothing compared
the pieces against each other.
"""
import os, sys, json, glob
import numpy as np

BASE = os.environ.get('CLM_SCRATCH',
                      '/pscratch/sd/s/shunth/claude_learning_map')
SEEN = os.path.join(BASE, 'seen_targets.json')
fails = []


def bad(msg):
    fails.append(msg)
    print(f'  FAIL  {msg}')


def ok(msg):
    print(f'  ok    {msg}')


if not os.path.exists(SEEN):
    print('no seen_targets.json'); sys.exit(1)
with open(SEEN) as f:
    book = json.load(f)
ids = book['targetids']
sessions = book['sessions']
# only batch_NNN.json belongs to the pick.py loop.  batch_hi20.json and any
# other named file is a one-off study and has no session record, so globbing
# 'batch_*.json' both crashed the id parse and inflated the count.
batches = sorted(glob.glob(os.path.join(BASE, 'batch_[0-9][0-9][0-9].json')))
other = sorted(set(glob.glob(os.path.join(BASE, 'batch_*.json'))) - set(batches))

# cohorts and one-off studies append a session record but write no
# batch_NNN.json -- they are a different mechanism.  A numbered pick.py record
# carries 'chunks' (current sampler) or 'rows' (the original contiguous slab);
# everything else is a named study.
pick_sessions = [q for q in sessions if 'chunks' in q or 'rows' in q]
cohort_sessions = [q for q in sessions if not ('chunks' in q or 'rows' in q)]

print(f'learning state: {len(sessions)} sessions '
      f'({len(pick_sessions)} pick, {len(cohort_sessions)} named), '
      f'{len(ids)} targets, {len(batches)} batch files'
      + (f', {len(other)} one-off batch file(s) ignored' if other else '')
      + '\n')

# 1. the seen list itself
if len(ids) != len(set(ids)):
    d = len(ids) - len(set(ids))
    bad(f'seen list contains {d} duplicate TARGETIDs')
else:
    ok('seen list has no duplicates')

# 2. sessions and batch files agree
if len(pick_sessions) != len(batches):
    bad(f'{len(pick_sessions)} pick session records but {len(batches)} '
        f'batch files -- a run wrote a batch without recording it')
else:
    ok(f'pick session count matches batch file count ({len(batches)})')

sids = [s.get('id') for s in sessions]
if len(sids) != len(set(sids)):
    bad('duplicate session ids')

# 3. every batch is recorded, and no target is in two batches
seen = set(ids)
owner = {}
for p in batches:
    sid = int(os.path.basename(p)[6:9])
    with open(p) as f:
        picked = json.load(f)
    b = set(q['targetid'] for q in picked)
    missing = b - seen
    if missing:
        bad(f'batch_{sid:03d}: {len(missing)} of {len(b)} targets are NOT in '
            f'the seen list -- they can be redrawn')
    else:
        ok(f'batch_{sid:03d}: all {len(b)} targets recorded')
    for t in b:
        if t in owner and owner[t] != sid:
            bad(f'TARGETID {t} appears in batch {owner[t]} and batch {sid}')
        owner[t] = sid

# 4. fetched spectra match their batch
for p in batches:
    sid = int(os.path.basename(p)[6:9])
    sp = os.path.join(BASE, f'spec_{sid:03d}.npz')
    if not os.path.exists(sp):
        print(f'  --    batch_{sid:03d}: not fetched yet')
        continue
    with open(p) as f:
        want = [q['targetid'] for q in json.load(f)]
    got = np.load(sp, allow_pickle=True)['targetid']
    if list(got) != want:
        bad(f'spec_{sid:03d}.npz TARGETIDs do not match batch_{sid:03d}.json')
    else:
        ok(f'spec_{sid:03d}.npz matches its batch')

# 5. anything on disk the seen list has never heard of
orphans = set(owner) - seen
if orphans:
    bad(f'{len(orphans)} targets exist in batch files but not the seen list')

# 6. every cohort obeys the ZWARN==0 selection rule (added 2026-09-02)
#
# This reads ONLY the zwarn field of each sealed key -- never z, never
# spectype -- and prints only counts.  ZWARN==0 is an allowed selection
# quantity under the amended blind protocol, so checking it does not break the
# seal; printing a redshift or a spectype would.
for kp in sorted(glob.glob(os.path.join(BASE, 'cohort*', 'data',
                                        'key_SEALED.json'))):
    name = os.path.basename(os.path.dirname(os.path.dirname(kp)))
    with open(kp) as f:
        key = json.load(f)
    flagged = sum(1 for v in key.values() if int(v.get('zwarn', 0)) != 0)
    # The rule dates from 2026-09-02.  A cohort built under it declares so in
    # data/selection_rules.json; one built before it is reported, not failed --
    # a rule cannot be violated retroactively.
    rp = os.path.join(os.path.dirname(kp), 'selection_rules.json')
    claims = False
    if os.path.exists(rp):
        with open(rp) as f:
            claims = bool(json.load(f).get('zwarn0'))
    if flagged and claims:
        bad(f'{name}: {flagged} of {len(key)} objects have ZWARN != 0 but the '
            f'cohort declares the ZWARN==0 rule -- the filter did not hold')
    elif flagged:
        print(f'  note  {name}: {flagged} of {len(key)} objects have '
              f'ZWARN != 0; built before the 2026-09-02 rule, not a failure')
    else:
        ok(f'{name}: all {len(key)} objects ZWARN==0'
           + ('' if claims else ' (predates the rule, but clean)'))

    ip = os.path.join(os.path.dirname(kp), 'items.json')
    if os.path.exists(ip):
        with open(ip) as f:
            want = set(int(q['tid']) for q in json.load(f))
        have = set(int(k) for k in key)
        if want != have:
            bad(f'{name}: sealed key covers {len(have)} targets but the '
                f'cohort has {len(want)} ({len(want - have)} unkeyed)')
        else:
            ok(f'{name}: sealed key covers exactly the cohort')

print()
if fails:
    print(f'{len(fails)} INVARIANT(S) BROKEN')
    sys.exit(1)
print('all invariants hold')
