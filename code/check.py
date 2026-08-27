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
batches = sorted(glob.glob(os.path.join(BASE, 'batch_*.json')))

print(f'learning state: {len(sessions)} sessions, {len(ids)} targets, '
      f'{len(batches)} batch files\n')

# 1. the seen list itself
if len(ids) != len(set(ids)):
    d = len(ids) - len(set(ids))
    bad(f'seen list contains {d} duplicate TARGETIDs')
else:
    ok('seen list has no duplicates')

# 2. sessions and batch files agree
if len(sessions) != len(batches):
    bad(f'{len(sessions)} session records but {len(batches)} batch files -- '
        f'a run wrote a batch without recording it')
else:
    ok('session count matches batch file count')

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

print()
if fails:
    print(f'{len(fails)} INVARIANT(S) BROKEN')
    sys.exit(1)
print('all invariants hold')
