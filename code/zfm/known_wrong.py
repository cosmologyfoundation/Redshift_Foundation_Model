"""Does a model copy Redrock's errors?  Sets where the label is known wrong.

DESIGN.md 6e: agreement with Redrock is not accuracy, so a loss meant to stop
fitting wrong labels must be judged where the label is wrong, by a route
independent of Redrock:
  wd_wrong   main MWS_WD STAR-only targets with label z > 0.1 (2,022): Gaia
             parallax/proper motion shows 99.9% are stars -> truth z = 0
  wd_rest    MWS_WD STAR-only with label |z| < 0.005, 2,000 sampled: control
  rr_wrong   blind224/blind170/blindmain objects my frozen blind read placed
             against Redrock and with the model/teacher (decidable only)
  rr_right   the same cohorts where my read agrees with Redrock (decidable
             candidates the teacher lost + every decidable control)
Truth is Gaia (z=0) or my blind z.  Redrock's z is carried only to count how
often a model reproduces it.  Nothing here is trained on.
"""
import argparse, os, re, h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from grids import LAM_OBS, LAM_NATIVE, BITS, CLASSES, bin_to_z
from data import P, FILES
from model import ZFM
ap = argparse.ArgumentParser(); ap.add_argument('--arms', required=True); ap.add_argument('--tag', required=True)
A = ap.parse_args()
rank = int(os.environ.get('SLURM_PROCID', 0)); world = int(os.environ.get('SLURM_NTASKS', 1))
torch.cuda.set_device(rank % torch.cuda.device_count())
OUT = f'/pscratch/sd/s/shunth/zfm_knownwrong_{A.tag}'
B = os.path.expanduser('~/claude_learning_map/')

def build():
    items = []                                     # (set, file, row, truth, z_rr)
    with h5py.File(P + FILES['main'][0], 'r') as f:
        fs = f['COADD_FIBERSTATUS'][:]; z = f['Z'][:]
        dt = f['DESI_TARGET'][:].astype(np.int64); mw = f['MWS_TARGET'][:].astype(np.int64)
    cls = np.stack([((dt >> BITS[c]) & 1).astype(bool) for c in CLASSES], 1)
    wd = cls[:, CLASSES.index('STAR')] & (cls.sum(1) == 1) & (fs == 0) & (((mw >> 1) & 1) == 1)
    hi = np.where(wd & (z > 0.1))[0]
    rest = np.sort(np.random.default_rng(0).choice(np.where(wd & (np.abs(z) < 0.005))[0], 2000, replace=False))
    items += [('wd_wrong', 'main', int(r), 0.0, float(z[r])) for r in hi]
    items += [('wd_rest', 'main', int(r), 0.0, float(z[r])) for r in rest]
    for d, fk in (('blind224', 'sv3'), ('blind170', 'sv3'), ('blindmain', 'main')):
        K = np.load(B + d + '/SEALED_key.npz')
        me = {}
        for l in open(B + d + '/answers_FROZEN_before_key.txt'):
            p = l.split()
            if len(p) >= 4 and re.fullmatch(r'OBJ\d+', p[0]):
                me[int(p[0][3:])] = (float(p[2]), p[3])
        for i in range(len(K['row'])):
            o = int(K['obj'][i]); zme, mark = me[o]
            if mark == '??':
                continue
            zr, zt = float(K['z_rr'][i]), float(K['z_teacher'][i])
            ok = lambda a: abs(a - zme) / (1 + zme) < 0.1
            if not bool(K['candidate'][i]):
                s = 'rr_right' if ok(zr) else None
            else:
                s = 'rr_wrong' if (ok(zt) and not ok(zr)) else 'rr_right' if (ok(zr) and not ok(zt)) else None
            if s:
                items.append((s, fk, int(K['row'][i]), zme, zr))
    return items

if rank == 0 and not os.path.exists(OUT + '_items.npz'):
    it = build()
    np.savez(OUT + '_items.tmp.npz', set=np.array([i[0] for i in it]), src=np.array([i[1] for i in it]),
             row=np.array([i[2] for i in it]), truth=np.array([i[3] for i in it]), z_rr=np.array([i[4] for i in it]))
    os.replace(OUT + '_items.tmp.npz', OUT + '_items.npz')
import time
while not os.path.exists(OUT + '_items.npz'):
    time.sleep(2)
I = dict(np.load(OUT + '_items.npz'))   # arrays in memory: a lazy NpzFile shared by forked workers corrupts its reads
idx = np.arange(len(I['row']))[rank::world]

class Rows(Dataset):
    def __len__(self): return len(idx)
    def __getitem__(self, j):
        if not hasattr(self, 'h'): self.h = {k: h5py.File(P + v[0], 'r') for k, v in FILES.items()}
        k = int(idx[j]); h = self.h[str(I['src'][k])]; r = int(I['row'][k])
        fl = h['BRZ_FLUX'][r].astype(np.float64); iv = np.maximum(h['BRZ_IVAR'][r].astype(np.float64), 0)
        F_, V_ = np.interp(LAM_OBS, LAM_NATIVE, fl), np.interp(LAM_OBS, LAM_NATIVE, iv)
        return torch.from_numpy(np.stack([F_ * np.sqrt(V_), np.sqrt(V_)]).astype(np.float32)), k

nets = {}
for stem in A.arms.split(','):
    ck = torch.load(f'/pscratch/sd/s/shunth/zfm_ckpt/{stem}.pt', map_location='cpu', weights_only=False); c = ck['cfg']
    m = ZFM(c['d_enc'], c['enc_layers'], c['c_corr'], c['n_per'], c['d_rr'], c['rr_layers'], c['K'],
            norm_corr=c['norm_corr'], rr_final_norm=c['rr_final_norm'], sep=c.get('sep', 64),
            evidence=c.get('evidence', False), shape=c.get('shape', False), sky=c.get('sky')).cuda().eval()
    m.load_state_dict(ck['model']); nets[stem] = m
out = {k: [] for k in ['k'] + list(nets)}
with torch.no_grad():
    for x, k in DataLoader(Rows(), batch_size=64, num_workers=24, pin_memory=True):
        x = x.cuda(); out['k'].append(k.numpy())
        for s, m in nets.items():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = m(x)
            b = o['cand_logit'].float().argmax(1)
            out[s].append(bin_to_z(o['cand_z'].gather(1, b.view(-1, 1)).squeeze(1).cpu().numpy()))
np.savez(f'{OUT}_r{rank}.npz', **{k: np.concatenate(v) for k, v in out.items()})
print(f'rank {rank}: {len(idx)} objects', flush=True)
os._exit(0)
