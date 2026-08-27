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
WANTED = {'STAR': {'STAR'}, 'QSO': {'QSO'}, 'ELG': {'ELG'},
          'GALAXY': {'BGS', 'LRG'}}
_G = {}


def _init():
    import methodc as MC, engine as E, d4000
    mc = MC.MethodC()
    _G.update(mc=mc, E=E, MC=MC, d4=d4000,
              fam=[FAMILY[m.name.rstrip('a')] for m in mc.s2])


def _one(a):
    fl, iv, call = a
    mc, E, MC, d4, fam = _G['mc'], _G['E'], _G['MC'], _G['d4'], _G['fam']
    D, _, _ = mc.sc.coarse(fl, iv)
    cand = set()
    for i in range(len(mc.s1)):
        for k in E.peaks(D[i], nmax=4, sep=50):
            cand.add(round(E.ZS[k], 5))
    for bz, _, _ in d4.break_candidates(fl, iv, nmax=3):
        cand.add(round(bz, 5))
    keep = WANTED[call]
    sel = [m for m, f in zip(mc.s2, fam) if f in keep] or mc.s2
    r = MC.score(fl, iv, sel, sorted(cand), fine=True)
    return r[0]['z'] if r else 0.0


if __name__ == '__main__':
    d = np.load(os.path.join(R, 'data', 'cohort.npz'), allow_pickle=True)
    calls = json.load(open(os.path.join(R, 'data', 'calls.json')))
    ids = [str(x) for x in d['id']]
    call = np.array([calls[i] for i in ids])
    FL, IV = d['flux'], d['ivar']
    with Pool(int(os.environ.get('SLURM_CPUS_ON_NODE', 64)),
              initializer=_init) as p:
        z = np.array(p.map(_one, [(FL[i], IV[i], call[i])
                                  for i in range(len(ids))], 2))
    # the galaxy label is decided by the measured redshift, not by eye
    final = call.copy()
    g = call == 'GALAXY'
    final[g] = np.where(z[g] < ZCUT, 'BGS', 'LRG')
    np.savez(os.path.join(R, 'data', 'pred.npz'), id=np.array(ids),
             call=call, mytype=final, z=z, tid=d['tid'], cls=d['cls'],
             snr=d['snr'])
    print(f'wrote pred.npz for {len(ids)} objects  (z cut {ZCUT} for BGS/LRG)')
    for c in ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']:
        m = final == c
        if m.any():
            print(f'  {c:5s} n={int(m.sum()):3d}  z {z[m].min():7.4f} '
                  f'- {z[m].max():7.4f}')
    print(f'  of the {int(g.sum())} read as GALAXY: '
          f'{int((final[g]=="BGS").sum())} BGS, {int((final[g]=="LRG").sum())} LRG')
