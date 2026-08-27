"""Open the sealed key ONCE, score, and make the two figures.

Nothing upstream has read data/key_SEALED.json: the cohort was selected on
Redrock-free columns, the renders carried only an opaque id, the type calls
were written down before anything was compared, and the redshifts were fitted
with the family each visual call selected.

Comparison is against Redrock only, as asked.  Redrock's SPECTYPE has just
STAR / GALAXY / QSO, so the five-way matrix uses the target class on the
reference axis and the three-way agreement against SPECTYPE is reported
alongside.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.environ.get('C3', '/pscratch/sd/s/shunth/claude_learning_map/cohort3')
OUT = os.path.expanduser('~/claude_learning_map/slides')
CN = ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']
STY = {'STAR': ('o', '#e8b923'), 'BGS': ('s', '#4fae8f'),
       'LRG': ('^', '#d4763a'), 'ELG': ('v', '#8f8fc4'),
       'QSO': ('D', '#d94f9a')}
C = 299792.458

p = np.load(os.path.join(R, 'data', 'pred.npz'), allow_pickle=True)
key = json.load(open(os.path.join(R, 'data', 'key_SEALED.json')))
faint = set(json.load(open(os.path.join(R, 'data', 'faint.json'))))
ids, mytype, z, tid, cls = (list(p['id']), p['mytype'], p['z'], p['tid'],
                            p['cls'])
zr = np.array([key[str(int(t))]['z'] for t in tid])
st = np.array([key[str(int(t))]['spectype'] for t in tid])
zw = np.array([key[str(int(t))]['zwarn'] for t in tid])
np.savez(os.path.join(R, 'data', 'scored.npz'), id=np.array(ids),
         mytype=mytype, z=z, zref=zr, spectype=st, zwarn=zw, tclass=cls,
         tid=tid, snr=p['snr'],
         faint=np.array([str(i) in faint for i in ids]))

dv = np.abs(z - zr) / (1 + zr) * C
ok = dv < 1000
G = {'STAR': 'STAR', 'BGS': 'GALAXY', 'LRG': 'GALAXY', 'ELG': 'GALAXY',
     'QSO': 'QSO'}
m3 = np.array([G[c] for c in mytype]); t3 = np.array([G[c] for c in cls])
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7.2))
hi = max(4.2, np.nanmax(zr) * 1.05, np.nanmax(z) * 1.05)
ax.plot([-0.1, hi], [-0.1, hi], color='0.15', lw=1.0)
for c in CN:
    m = cls == c
    mk, col = STY[c]
    ax.scatter(zr[m], z[m], marker=mk, s=48, facecolor=col, edgecolor='0.25',
               linewidth=.6, label=c, zorder=3)
ax.set_xlim(-0.15, hi); ax.set_ylim(-0.15, hi)
ax.set_xlabel('Redrock z', fontsize=12); ax.set_ylabel('my model z', fontsize=12)
ax.set_title('my redshift vs Redrock redshift\n'
             f'n={len(z)}, inside +/-1000 km/s: {int(ok.sum())} '
             f'({100*ok.mean():.0f}%)', fontsize=12)
ax.grid(alpha=.3); ax.legend(loc='lower right', fontsize=10)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'c3_redshift.png'), dpi=130)
plt.close(fig)

M = np.zeros((5, 5), int)
for i, r in enumerate(CN):
    for j, c in enumerate(CN):
        M[i, j] = int(((cls == r) & (mytype == c)).sum())
fig, ax = plt.subplots(figsize=(7.8, 7.2))
ax.imshow(M, cmap='Blues', vmin=0, vmax=M.max())
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(M[i, j]), ha='center', va='center', fontsize=14,
                color='white' if M[i, j] > M.max() * .55 else '#1a3a6b')
ax.set_xticks(range(5)); ax.set_xticklabels(CN, fontsize=11)
ax.set_yticks(range(5)); ax.set_yticklabels(CN, fontsize=11)
ax.set_xlabel('model type  (read from the image alone)', fontsize=12)
ax.set_ylabel('target class', fontsize=12)
ax.set_title(f'type agreement\nn={M.sum()}, diagonal {np.trace(M)} '
             f'({100*np.trace(M)/M.sum():.0f}%)', fontsize=12)
ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
ax.grid(which='minor', color='white', lw=1.6); ax.tick_params(which='minor', length=0)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'c3_type.png'), dpi=130)
plt.close(fig)

fa = np.array([str(i) in faint for i in ids])
print(f'redshift within 1000 km/s : {int(ok.sum())}/{len(z)}')
for c in CN:
    m = cls == c
    print(f'   {c:5s} {int((ok & m).sum()):2d}/{int(m.sum()):2d}')
print(f'\ntype 5-way vs target class     : {int((mytype == cls).sum())}/{len(z)}')
print(f'type 3-way vs Redrock SPECTYPE : {int((m3 == st).sum())}/{len(z)}')
print(f'target class vs SPECTYPE       : {int((t3 == st).sum())}/{len(z)}')
print(f'\nZWARN!=0 in the reference      : {int((zw != 0).sum())}')
print(f'\nsplit by whether I could see a feature:')
for nm, m in (('feature visible', ~fa), ('no feature (flagged)', fa)):
    print(f'   {nm:22s} n={int(m.sum()):3d}  type {int((mytype[m]==cls[m]).sum()):3d}'
          f'  z {int(ok[m].sum()):3d}')
print('\nconfusion (rows target class, cols mine):')
print('        ' + ''.join(f'{c:>7s}' for c in CN))
for i, r in enumerate(CN):
    print(f'  {r:5s} ' + ''.join(f'{M[i, j]:7d}' for j in range(5)))
