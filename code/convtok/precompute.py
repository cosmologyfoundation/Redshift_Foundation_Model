"""Run the frozen dual-scale tokenizer ONCE and store its outputs (owner 2026-10-10: 「把 tokenizer 的輸出先算好存起來」).

Per split and per rank (4 ranks, the same sharding the streaming runs used), memmapped .npy files in OUT/:
  {split}_r{rank}_narrow.npy  (count, 1882, 384) float16     narrow-line tokens
  {split}_r{rank}_broad.npy   (count,  118, 768) float16     broad-line tokens
  {split}_r{rank}_meta.npz    count, zb (z bin), z, cls (5 target bits), log10s (brightness)
Splits (original ZWARN=0, no cleaning, class mix 20% each):
  train  300k from the training blocks (validation blocks 0-1300 excluded)
  es     8,192 from validation blocks 400-1300 (early stopping)
  test   up to 60,000 from validation blocks 0-400 (the pool holds ~46.5k at a 20% class mix)
Positions are fixed by the tokenizer (narrow 4i+1.5, broad 64j+31.5 bins) and are rebuilt in the model, not stored.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from data import Spectra
from dualtok import DualTok, NN, NB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['train', 'es', 'test'], required=True)
    ap.add_argument('--size', type=int, default=300000); ap.add_argument('--n_es', type=int, default=8192)
    ap.add_argument('--n_test', type=int, default=60000); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--drop_last', action='store_true', help='test: workers 8, bs 32, drop_last = the stream ztrans/ztrans2 tested on')
    a = ap.parse_args()
    rank = int(os.environ.get('SLURM_PROCID', 0)); world = int(os.environ.get('SLURM_NTASKS', 1))
    torch.cuda.set_device(rank % torch.cuda.device_count()); dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tck = torch.load(a.tok, map_location='cpu', weights_only=False)
    tok = DualTok(lv_n=tuple(int(x) for x in tck['cfg'].get('lv_n', '5,5').split(','))).to(dev).eval()
    tok.load_state_dict(tck['model'])
    nw = a.workers
    if a.split == 'train':
        ds = Spectra(a.size, seed=a.seed, rank=rank, world=world, clean=0, native=True, val_blocks_main=1300)
        lim = nw * math.ceil(a.size / (world * nw))
    elif a.split == 'es':
        ds = Spectra(1 << 22, seed=0, val=True, rank=rank, world=world, clean=0, native=True, val_blocks_main=1300, val_from_main=400)
        lim = a.n_es // world
    else:
        ds = Spectra(1 << 22, seed=0, val=True, rank=rank, world=world, clean=0, native=True)
        lim = a.n_test // world
    ub = lim + a.bs
    os.makedirs(a.out, exist_ok=True)
    pre = '%s/%s_r%d' % (a.out, a.split, rank)
    NA = np.lib.format.open_memmap(pre + '_narrow.npy', mode='w+', dtype=np.float16, shape=(ub, NN, 384))
    BA = np.lib.format.open_memmap(pre + '_broad.npy', mode='w+', dtype=np.float16, shape=(ub, NB, 768))
    ZB, Z, CL, LS = [], [], [], []
    n = 0; t0 = time.time()
    with torch.no_grad():
        for fl, iv, zb, c, z in DataLoader(ds, batch_size=a.bs, num_workers=nw, pin_memory=True, drop_last=a.drop_last):
            if n >= lim:
                break
            k = min(len(z), lim - n)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hn, hb, cn, cb, f, nn_, s = tok.encode(fl[:k].to(dev), iv[:k].to(dev))
            NA[n:n + k] = hn.float().cpu().numpy().astype(np.float16)
            BA[n:n + k] = hb.float().cpu().numpy().astype(np.float16)
            ZB.append(zb[:k].numpy()); Z.append(z[:k].numpy()); CL.append(c[:k].numpy()); LS.append(torch.log10(s).float().cpu().numpy())
            n += k
    NA.flush(); BA.flush(); del NA, BA
    np.savez(pre + '_meta.npz', count=n, zb=np.concatenate(ZB), z=np.concatenate(Z), cls=np.concatenate(CL), log10s=np.concatenate(LS))
    print('rank %d %s: %d spectra (limit %d) in %.0fs -> %s' % (rank, a.split, n, lim, time.time() - t0, pre), flush=True)


if __name__ == '__main__':
    main()
