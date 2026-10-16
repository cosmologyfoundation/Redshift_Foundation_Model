"""Redshift Transformer trained on the PRE-COMPUTED dual-scale tokenizer outputs (precompute.py), with torch.compile
(owner 2026-10-10: 「先暫停掉 用 torch.compile 把 tokenizer 的輸出先算好存起來 準備好之後重新train」).

Same model and heads as ztrans2.py (all tokenizer outputs: narrow 1,882 x 384, broad 118 x 768, their log-lambda
positions, brightness; [CLS] coarse z + 8 candidates with rest-frame RoPE + 33 fine offsets + class), minus the
tokenizer, which is frozen and already applied.  Training: 300k x up to --epochs, early stop when neither the
early-stop loss nor its NMAD improves for --patience epochs, best epoch tested.  gbs 128 (32 x 4), lr 6e-4.
Each rank reads its own precomputed shard; every rank takes the same number of steps (the smallest shard).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from grids import N_Z, S_LO, CLASSES
from train import ddp_setup
from model import topk_peaks
import ztrans2 as Z2

CKDIR = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'


class TokSet(Dataset):
    def __init__(self, pre):
        self.pre = pre
        m = np.load(pre + '_meta.npz')
        self.n = int(m['count']); self.zb, self.z, self.cls, self.ls = m['zb'], m['z'], m['cls'], m['log10s']
        self.na = self.ba = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.na is None:                                        # open lazily, once per worker
            self.na = np.load(self.pre + '_narrow.npy', mmap_mode='r'); self.ba = np.load(self.pre + '_broad.npy', mmap_mode='r')
        return (torch.from_numpy(np.array(self.na[i])), torch.from_numpy(np.array(self.ba[i])), torch.tensor(float(self.ls[i])),
                torch.tensor(int(self.zb[i])), torch.from_numpy(self.cls[i].astype(np.float32)), torch.tensor(float(self.z[i])))


class ZT3(Z2.ZTrans):
    def __init__(self, **kw):
        super().__init__(**kw)
        del self.tok                                                # tokenizer outputs come from disk

    def forward(self, hn, hb, ls):
        B = hn.shape[0]
        c0 = self.cls.expand(B, -1, -1) + self.s_in(ls.view(B, 1, 1).to(self.cls.dtype))
        x = torch.cat([c0, self.b_in(hb) + self.typ[0], self.n_in(hn) + self.typ[1]], 1) + self.pos
        x = self.encode(x)
        g, mem = x[:, 0], x[:, 1:]
        lc = self.coarse(g).float()
        _, cb_ = topk_peaks(lc.detach().unsqueeze(1), self.Kc, sep=2)
        cen = cb_ * Z2.CW + (Z2.CW - 1) / 2.0
        qa = (cen + S_LO).float().unsqueeze(-1) * self.om
        q = self.qemb(torch.stack([cen.float() / N_Z, torch.log_softmax(lc, -1).gather(1, cb_).detach()], -1))
        for b in self.dec:
            q = b(q, mem, self.kc, self.ks, qa.cos(), qa.sin())
        q = self.dnorm(q)
        return dict(coarse=lc, cand_cb=cb_, cand_cen=cen, pick=self.pick(q).squeeze(-1).float(),
                    off=self.offh(q).float(), logit_c=self.chead(g).float())

    def encode(self, x):                                            # the part torch.compile speeds up
        for b in self.enc:
            x = b(x, self.ec, self.es)
        return self.norm(x)


def run_eval(core, loader, world, dev):
    ZT, ZP, CL, CP, LS, n = [], [], [], [], [], 0
    with torch.no_grad():
        for hn, hb, ls, zb, c, z in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = core(hn.to(dev).float(), hb.to(dev).float(), ls.to(dev))
                L_ = Z2.losses(o, zb.to(dev), c.to(dev))
            ZT.append(z.numpy()); ZP.append(Z2.readz(o)); CL.append(c.numpy()); CP.append(o['logit_c'].argmax(1).cpu().numpy())
            LS.append(float(L_[0] + L_[1] + L_[2]) * len(z)); n += len(z)
    loc = tuple(np.concatenate(v) for v in (ZT, ZP, CL, CP)) + (np.array([sum(LS), n]),)
    if world > 1:
        buf = [None] * world; dist.all_gather_object(buf, loc)
        loc = tuple(np.concatenate([b_[i] for b_ in buf]) for i in range(4)) + (sum(b_[4] for b_ in buf),)
    return loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='precompute.py output directory')
    ap.add_argument('--H', type=int, default=32); ap.add_argument('--L', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--resume', default='')
    ap.add_argument('--tag', default='ZT3'); ap.add_argument('--max_steps', type=int, default=0)
    a = ap.parse_args()
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    core = ZT3(H=a.H, L=a.L).to(dev)
    if a.compile:
        core.encode = torch.compile(core.encode)
    net = DDP(core, device_ids=[local], find_unused_parameters=False) if world > 1 else core
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    tr_set = TokSet('%s/train_r%d' % (a.data, rank))
    nmin = torch.tensor(len(tr_set), device=dev)
    if world > 1:
        dist.all_reduce(nmin, op=dist.ReduceOp.MIN)
    spe = int(nmin) // a.bs
    if a.max_steps:
        spe = min(spe, a.max_steps)
    steps = spe * a.epochs; gbs = a.bs * world
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    es = DataLoader(TokSet('%s/es_r%d' % (a.data, rank)), batch_size=64, num_workers=4, pin_memory=True)
    te = DataLoader(TokSet('%s/test_r%d' % (a.data, rank)), batch_size=64, num_workers=4, pin_memory=True)
    stem = '%s_H%d_L%d_300000_e%d_gbs%d_lr%g_s%d_cl0_mix' % (a.tag, a.H, a.L, a.epochs, gbs, a.lr, a.seed)
    state_fn, best_fn = CKDIR + stem + '_state.pt', CKDIR + stem + '_best.pt'
    ep0, best, stall, hist = 0, dict(loss=1e9, nmad=1e9, epoch=0), 0, []
    if a.resume and os.path.exists(a.resume):
        st = torch.load(a.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        ep0, best, stall, hist = st['epoch'], st['best'], st['stall'], st['hist']
    if is0:
        print('ZT3 %s | precomputed tokenizer outputs %s | transformer %.2fM: L %d, d 768, %d heads | train %d per rank (min), '
              '%d steps/epoch, gbs %d, lr %.1e, up to %d epochs, patience %d | compile %s | start epoch %d'
              % (stem, a.data, sum(p.numel() for p in core.parameters()) / 1e6, a.L, a.H, int(nmin), spe, gbs, a.lr,
                 a.epochs, a.patience, a.compile, ep0 + 1), flush=True)
    t0 = time.time()
    for ep in range(ep0, a.epochs):
        net.train(); run = []; te0 = time.time()
        g = torch.Generator(); g.manual_seed(a.seed * 1000 + ep * 10 + rank)
        tr = DataLoader(tr_set, batch_size=a.bs, sampler=RandomSampler(tr_set, generator=g), num_workers=a.workers,
                        pin_memory=True, drop_last=True, persistent_workers=False, prefetch_factor=4)
        for it, (hn, hb, ls, zb, c, z) in enumerate(tr):
            if it >= spe:
                break
            hn, hb, ls = hn.to(dev, non_blocking=True).float(), hb.to(dev, non_blocking=True).float(), ls.to(dev)
            zb, c = zb.to(dev), c.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(hn, hb, ls)
                L_ = Z2.losses(o, zb, c)
                loss = L_[0] + L_[1] + L_[2] + 0.1 * L_[3]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step()
            run.append([float(v) for v in L_])
            if not all(math.isfinite(v) for v in run[-1]):
                raise RuntimeError('non-finite loss at epoch %d step %d: %s' % (ep + 1, it, run[-1]))
            if is0 and (it + 1) % 500 == 0:
                el = time.time() - te0
                print('  ep %d step %5d/%d  coarse %.4f  pick %.4f  offset %.4f  class %.4f  %.0f spectra/s  mem %.1f GB'
                      % (ep + 1, it + 1, spe, *np.mean(run[-500:], 0), (it + 1) * gbs / el, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        core.eval()
        zt, zp, cl, cp, ls_ = run_eval(core, es, world, dev)
        d = (zp - zt) / (1 + zt)
        e_loss, e_nmad, e_cat = float(ls_[0] / ls_[1]), float(Z2.nmad(d)), float((np.abs(d) > 0.1).mean())
        imp = (['loss'] if e_loss < best['loss'] - 1e-3 else []) + (['NMAD'] if e_nmad < best['nmad'] * 0.99 else [])
        if imp:
            stall = 0; best = dict(loss=min(best['loss'], e_loss), nmad=min(best['nmad'], e_nmad), epoch=ep + 1)
            if is0:
                torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep + 1}, best_fn)
        else:
            stall += 1
        hist.append(dict(epoch=ep + 1, train=np.mean(run, 0).tolist(), es_loss=e_loss, es_nmad=e_nmad, es_catastrophic=e_cat,
                         improved=imp, seconds=time.time() - t0))
        if is0:
            print('  EPOCH %d (%.0fs)  train coarse %.4f pick %.4f offset %.4f | early-stop set (%d): loss %.4f  NMAD %.3e  '
                  'catastrophic %.3f%% | improved: %s | stall %d/%d' % (ep + 1, time.time() - te0, *np.mean(run, 0)[:3], len(zt),
                                                                       e_loss, e_nmad, 100 * e_cat, ','.join(imp) or 'none', stall, a.patience), flush=True)
            torch.save({'model': core.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep + 1,
                        'best': best, 'stall': stall, 'hist': hist}, state_fn)
        if stall >= a.patience:
            if is0:
                print('  EARLY STOP after epoch %d; best epoch %d' % (ep + 1, best['epoch']), flush=True)
            break
    if world > 1:
        dist.barrier()
    core.load_state_dict(torch.load(best_fn, map_location='cpu', weights_only=False)['model']); core.eval()
    zt, zp, cl, cp, _ = run_eval(core, te, world, dev)
    if is0:
        d = (zp - zt) / (1 + zt); bad = np.abs(d) > 0.1
        prio = np.full(len(cl), -1)
        for name in ['STAR', 'BGS', 'ELG', 'LRG', 'QSO']:
            prio[cl[:, CLASSES.index(name)] > 0] = CLASSES.index(name)
        res = dict(stem=stem, cfg=vars(a), best_epoch=best['epoch'], history=hist, n_test=len(zt), catastrophic=float(bad.mean()),
                   n_catastrophic=int(bad.sum()), nmad=float(Z2.nmad(d)), within_0p003=float((np.abs(d) < 0.003).mean()),
                   class_any_bit=float(cl[np.arange(len(cp)), cp].mean()), seconds=time.time() - t0,
                   per_class={c_: dict(n=int((prio == i).sum()), catastrophic=float(bad[prio == i].mean()),
                                       n_catastrophic=int(bad[prio == i].sum()), nmad=float(Z2.nmad(d[prio == i])))
                              for i, c_ in enumerate(CLASSES)})
        print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1), flush=True)
        os.makedirs('results', exist_ok=True); json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        np.savez('results/%s_test.npz' % stem, zt=zt, zp=zp, cls=cl, cls_pred=cp)
        print('RUN DONE', flush=True)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
