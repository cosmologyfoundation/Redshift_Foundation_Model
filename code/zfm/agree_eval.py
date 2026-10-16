ＦＦ"""Agreement on corroborated labels + sub-bin NMAD (results/prereg_agree.txt).
Four ranks over the same validation stream as ev_ab_eval.py; per-object arrays saved."""
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader
from grids import bin_to_z, z_to_bin, N_Z
from data import Spectra
from model import ZFM
import evidence as E
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=100096); ap.add_argument('--arms', required=True)
ap.add_argument('--tag', default='agree')
ap.add_argument('--clean_val', type=int, nargs='?', const=1, default=0, help='validation blocks minus cleaned-away rows: --clean_val = v1, --clean_val 2 = v2')
ap.add_argument('--val_blocks_main', type=int, default=400, help='validation blocks, as train.py --val_blocks_main')
A = ap.parse_args()
rank = int(os.environ.get('SLURM_PROCID', 0)); world = int(os.environ.get('SLURM_NTASKS', 1))
torch.cuda.set_device(rank % torch.cuda.device_count())
nets = {}
for stem in A.arms.split(','):
    ck = torch.load(f'/pscratch/sd/s/shunth/zfm_ckpt/{stem}.pt', map_location='cpu', weights_only=False); c = ck['cfg']
    m = ZFM(c['d_enc'], c['enc_layers'], c['c_corr'], c['n_per'], c['d_rr'], c['rr_layers'], c['K'],
            norm_corr=c['norm_corr'], rr_final_norm=c['rr_final_norm'], sep=c.get('sep', 64),
            evidence=c.get('evidence', False), shape=c.get('shape', False), sky=c.get('sky')).cuda().eval()
    m.load_state_dict(ck['model']); nets[stem] = m
NM = E.NAMES; ix = lambda k: NM.index(k)
ld = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, clean=A.clean_val, val_blocks_main=A.val_blocks_main), batch_size=32, num_workers=16,
                pin_memory=True, drop_last=True)
cap = A.n // world
out = {k: [] for k in ['zt', 'cls', 'corr', 'route'] + [s + x for s in nets for x in ('', '_i')]}
n = 0
with torch.no_grad():
    for x, zb, cl, z in ld:
        if n >= cap: break
        x = x.cuda(); zbl = zb.cuda().view(-1, 1)
        f = E.candidate_features(E.spectrum_maps(x, sky=0.3), zbl, slog=False)[:, 0].float()
        brk_ok = (f[:, ix('break_valid')] > 0) & (f[:, ix('break')] >= 6)
        a_ = f[:, ix('n_emis>=8')] >= 2
        b_ = brk_ok & ((f[:, ix('n_emis>=3')] >= 1) | (f[:, ix('n_abs<=-3')] >= 1))
        c_ = f[:, ix('n_abs<=-3')] >= 3
        clean = f[:, ix('n_unexpl>=8')] == 0
        out['corr'].append(((a_ | b_ | c_) & clean).cpu().numpy())
        out['route'].append((a_.int() + 2 * b_.int() + 4 * c_.int()).cpu().numpy())
        out['zt'].append(z.numpy()); out['cls'].append(cl.numpy())
        for s, m in nets.items():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = m(x)
            best = o['cand_logit'].float().argmax(1)
            b = o['cand_z'].gather(1, best.view(-1, 1)).squeeze(1)
            lp = o['logp_z'].float()
            bm, bp = (b - 1).clamp(0, N_Z - 1), (b + 1).clamp(0, N_Z - 1)
            l0, lm, lpp = lp.gather(1, b.view(-1, 1))[:, 0], lp.gather(1, bm.view(-1, 1))[:, 0], lp.gather(1, bp.view(-1, 1))[:, 0]
            den = lm - 2 * l0 + lpp
            d = torch.where(den < 0, 0.5 * (lm - lpp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
            out[s].append(bin_to_z(b.cpu().numpy()))
            out[s + '_i'].append(bin_to_z(b.cpu().numpy() + d.cpu().numpy()))
        n += len(z)
np.savez(f'results/{A.tag}_r{rank}.npz', **{k: np.concatenate(v) for k, v in out.items()})
print(f'rank {rank}: {n} objects', flush=True)
os._exit(0)
