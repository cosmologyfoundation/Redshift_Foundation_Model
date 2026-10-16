"""Dual-scale ConvNeXt tokenizer: narrow-line tokens + broad-line tokens (owner 2026-10-09, table:
  窄線 token 1,882 x 384 (4 bins = 160 km/s) | 寬線 token 118 x 768 (64 bins = 2,560 km/s) | token positions | brightness).
ConvNeXt only, no Transformer; trained by reconstruction, no labels.

  BRZ_FLUX, BRZ_IVAR -> native prep [asinh f, log n, measured] -> fixed resample to the 40 km/s log grid (7,526, pad 7,552)
  ENCODER (ConvNeXt on the log grid; each stage halves the length)
    7552x64 -> 3776x128 -> 1888x384  = NARROW features (first 1,882 kept; the last 6 cover padding only)
            -> 944x512 -> 472x768 -> 236x768 -> 118x768 = BROAD features (receptive field ~ +-15,000 km/s)
  BOTTLENECKS (FSQ): narrow 384 -> 3 codes (5,5,5 = 7.0 bits / 160 km/s); broad 768 -> 6 codes (7,7,7,5,5,5 = 15.4 bits
    / 2,560 km/s).  NOT measured: whether 7 bits per 160 km/s lets noise through -- the 0.424 floor check says.
  DECODERS, each forced to one scale:
    broad  : codes -> 118 -> 236 -> 472 (ConvNeXt) -> 1 ch -> linear upsample x16 -> Gaussian LOW-pass (sigma 12 bins)
    narrow : codes -> 1888 -> 3776 -> 7552 (ConvNeXt) -> 1 ch -> HIGH-pass (x - the same Gaussian)
    spectrum = broad + narrow  -> crop 7,526 -> fixed map back to the 7,781 native pixels
    sigma 12 bins = 480 km/s (FWHM ~1,130 km/s): narrow lines (50-300 km/s) can only come from the narrow tokens,
    continuum and broad QSO lines (2,000-10,000 km/s) mostly from the broad tokens.  The split scale is a DEFAULT.
  LOSS: Huber of (pred - f) * n on every measured native pixel (true ivar), per spectrum then over spectra.
"""
import argparse, contextlib, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from grids import N_OBS, LAM_OBS, LAM_NATIVE
from data import Spectra
import muonopt
from train import ddp_setup
from model import ConvNeXt1D
from convtok import Prep, Interp
from convae import FSQ, stage, ustage

NPAD = 7552                                   # 118 x 64
NN, NB = 1882, 118                            # narrow / broad token counts
SIG = 12.0                                    # split scale, bins of 40 km/s


def gauss_kernel(sig):
    r = int(round(4 * sig)); x = torch.arange(-r, r + 1, dtype=torch.float)
    k = torch.exp(-0.5 * (x / sig) ** 2); return (k / k.sum()).view(1, 1, -1)


class DualTok(nn.Module):
    def __init__(self, nb=2, lv_n=(5, 5, 5), lv_b=(7, 7, 7, 5, 5, 5), sig=SIG, swiglu=False):
        super().__init__()
        ch = (64, 128, 384, 512, 768, 768, 768)
        self.prep = Prep()
        self.to_log = Interp(LAM_NATIVE, LAM_OBS)
        self.to_nat = Interp(LAM_OBS, LAM_NATIVE)
        sw = dict(swiglu=swiglu)
        self.enc = nn.ModuleList([stage(3, ch[0], nb, False, **sw)] + [stage(ch[i], ch[i + 1], nb, True, **sw) for i in range(len(ch) - 1)])
        self.qn_in, self.fsq_n, self.qn_out = nn.Linear(ch[2], len(lv_n)), FSQ(lv_n), nn.Linear(len(lv_n), ch[2])
        self.qb_in, self.fsq_b, self.qb_out = nn.Linear(ch[-1], len(lv_b)), FSQ(lv_b), nn.Linear(len(lv_b), ch[-1])
        self.dec_n = nn.Sequential(*[ConvNeXt1D(ch[2], swiglu=swiglu) for _ in range(nb)], ustage(ch[2], ch[1], nb, **sw), ustage(ch[1], ch[0], nb, **sw))
        self.rec_n = nn.Conv1d(ch[0], 1, 1)
        self.dec_b = nn.Sequential(*[ConvNeXt1D(ch[-1], swiglu=swiglu) for _ in range(nb)], ustage(ch[-1], 384, nb, **sw), ustage(384, 128, nb, **sw))
        self.rec_b = nn.Conv1d(128, 1, 1)
        self.register_buffer('gk', gauss_kernel(sig), persistent=False)
        self.register_buffer('pos_n', torch.arange(NN).float() * 4 + 1.5, persistent=False)     # token centres, bins
        self.register_buffer('pos_b', torch.arange(NB).float() * 64 + 31.5, persistent=False)

    def lowpass(self, x):                                                   # (B,1,L) -> (B,1,L)
        r = self.gk.shape[-1] // 2
        return F.conv1d(F.pad(x, (r, r), mode='replicate'), self.gk.to(x.dtype))

    def encode(self, flux, ivar):
        """-> narrow (B,1882,384), broad (B,118,768), codes_n (B,1888,3), codes_b (B,118,6), f, n (native), s"""
        x, f, n, s = self.prep(flux, ivar)
        h = F.pad(self.to_log(x), (0, NPAD - N_OBS))
        for i, st in enumerate(self.enc):
            h = st(h)
            if i == 2:
                hn = h.transpose(1, 2)                                      # (B,1888,384)
        hb = h.transpose(1, 2)                                              # (B,118,768)
        return hn[:, :NN], hb, self.fsq_n(self.qn_in(hn)), self.fsq_b(self.qb_in(hb)), f, n, s

    def decode(self, cn, cb):
        """-> broad component, narrow component (log grid, B x 7552), total on native pixels (B x 7781)"""
        hb = self.dec_b(self.qb_out(cb).transpose(1, 2))                    # (B,128,472)
        hn = self.dec_n(self.qn_out(cn).transpose(1, 2))                    # (B,64,7552)
        with torch.autocast('cuda', enabled=False):                         # fixed filters and the read-out in fp32
            b = self.rec_b(hb.float())
            b = self.lowpass(F.interpolate(b, size=NPAD, mode='linear', align_corners=False))
            nr = self.rec_n(hn.float())
            nr = nr - self.lowpass(nr)
            tot = self.to_nat((b + nr)[:, 0, :N_OBS])
        return b[:, 0], nr[:, 0], tot

    def forward(self, flux, ivar):
        hn, hb, cn, cb, f, n, s = self.encode(flux, ivar)
        b, nr, tot = self.decode(cn, cb)
        return tot, f, n, s

    @torch.no_grad()
    def embed(self, flux, ivar):
        """the tokenizer's outputs: narrow tokens, broad tokens, their positions (40 km/s bins), brightness"""
        hn, hb, cn, cb, f, n, s = self.encode(flux, ivar)
        return dict(narrow=hn, broad=hb, pos_narrow=self.pos_n, pos_broad=self.pos_b, log10_scale=torch.log10(s),
                    codes_narrow=cn[:, :NN], codes_broad=cb)


def loss_fn(net, flux, iv, per_spec=False):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, f, n, s = net(flux, iv)
    pix = n > 0
    hub = F.huber_loss((pred - f) * n, torch.zeros_like(f), delta=1.0, reduction='none')
    hub0 = F.huber_loss(-f * n, torch.zeros_like(f), delta=1.0, reduction='none')
    cnt = pix.sum(1).clamp_min(1)
    ps, ps0 = (hub * pix).sum(1) / cnt, (hub0 * pix).sum(1) / cnt
    ok = pix.any(1)
    if not per_spec:
        return ps[ok].mean(), ps0[ok].mean()
    return ps[ok].mean(), ps0[ok].mean(), ps, (f.abs() * n).amax(1)


def selftest():
    torch.manual_seed(0)
    m = DualTok(nb=1).eval()
    print('params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    with torch.no_grad():
        e = m.embed(fl, iv)
        hn, hb, cn, cb, f, n, s = m.encode(fl, iv)
        b, nr, tot = m.decode(cn, cb)
    print({k: tuple(v.shape) for k, v in e.items()})
    print('broad comp', tuple(b.shape), 'narrow comp', tuple(nr.shape), 'total native', tuple(tot.shape))
    # the split really is a split: the low-pass of the narrow component is ~0, the high-pass of the broad one is small
    x = torch.randn(1, 1, NPAD)
    hp = x - m.lowpass(x)
    print('low-pass of a high-passed signal: rms %.3e (input rms %.3f)' % (m.lowpass(hp).pow(2).mean().sqrt().item(), x.std().item()))
    m.train(); l, l0 = loss_fn(m, fl, iv); l.backward()
    print('loss %.3f zero %.3f; params without grad: %s' % (l, l0, [k for k, p in m.named_parameters() if p.grad is None]))
    st = m.enc[0]; xx = torch.randn(1, 3, NPAD)
    print('stage-0 shift test: %.2e' % (torch.roll(st(xx), 40, -1) - st(torch.roll(xx, 40, -1)))[..., 100:-100].abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--size', type=int, default=0); ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--clean', type=int, default=-1); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_val', type=int, default=8192); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--muon', action='store_true', help='Muon on the ConvNeXt up/down matrices (muonopt.CONVNEXT_HIDDEN), AdamW on the rest')
    ap.add_argument('--muon_lr', type=float, default=None)
    ap.add_argument('--swiglu', action='store_true', help='SwiGLU MLP in every ConvNeXt block (parameter-matched)')
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--tag', default='DUALTOK')
    ap.add_argument('--lv_n', default='5,5', help='narrow FSQ levels; 5,5,5 (7.0 bits/160 km/s) copied noise (0.290 < 0.424 floor)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    lv_n = tuple(int(x) for x in a.lv_n.split(','))
    core = DualTok(lv_n=lv_n, swiglu=a.swiglu).to(dev)
    net = DDP(core, device_ids=[local]) if world > 1 else core
    if a.compile:
        net = torch.compile(net)
    mu, aw, _ = muonopt.split(core.named_parameters(), muonopt.CONVNEXT_HIDDEN) if a.muon else ([], list(core.parameters()), [])
    opt = torch.optim.AdamW(aw, lr=a.lr, weight_decay=0.01, fused=True)
    gbs = a.bs * world * a.accum; spe = max(a.size // gbs, 1); steps = spe * a.epochs
    if a.max_steps:
        steps = min(steps, a.max_steps)
    wu = max(1, int(0.05 * steps))
    lam = lambda s: min((s + 1) / wu, max(0.0, (steps - s) / max(1, steps - wu)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    if mu:
        om = muonopt.muon(mu, a.muon_lr or a.lr)
        opt, sched = muonopt.MultiOpt([opt, om]), muonopt.MultiSched([sched, torch.optim.lr_scheduler.LambdaLR(om, lam)])
        if is0:
            print('Muon on %d ConvNeXt matrices (%.2fM, lr %g), AdamW on the rest (%.2fM)' % (len(mu), sum(p.numel() for p in mu) / 1e6,
                  a.muon_lr or a.lr, sum(p.numel() for p in aw) / 1e6), flush=True)
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, labelfree=True, clean=0, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
    stem = '%s_%d_n%s_cl0_mix_e%d_bs%d' % (a.tag, a.size, ''.join(str(x) for x in lv_n), a.epochs, gbs)
    if is0:
        print('DUALTOK %s | narrow 1882 x 384 (160 km/s, %.1f bits) + broad 118 x 768 (2560 km/s, 15.4 bits) | params %.2fM | '
              'size %d x %d epochs, gbs %d (%d x %d x %d) steps %d | data: original ZWARN=0, no cleaning, class mix 20%% each'
              % (stem, float(np.log2(np.prod(lv_n))), sum(p.numel() for p in core.parameters()) / 1e6, a.size, a.epochs, gbs, a.bs, world, a.accum, steps), flush=True)
    ckdir = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'

    def validate():
        core.eval()
        va = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, labelfree=True, clean=0, native=True),
                        batch_size=64, num_workers=8, pin_memory=True, drop_last=True)
        L_, L0, PS, SN, n = [], [], [], [], 0
        with torch.no_grad():
            for fl, iv in va:
                if n >= a.n_val // world:
                    break
                l, l0, ps, sn = loss_fn(core, fl.to(dev), iv.to(dev), per_spec=True)
                L_.append(float(l)); L0.append(float(l0)); PS.append(ps); SN.append(sn); n += len(fl)
        PS, SN = torch.cat(PS), torch.cat(SN); br = SN > 50
        out = torch.tensor([np.mean(L_), np.mean(L0), PS[br].mean().item() if br.any() else float('nan'),
                            PS[~br].mean().item(), br.float().mean().item()], device=dev)
        if world > 1:
            dist.all_reduce(out); out = out / world
        core.train()
        return [float(v) for v in out]

    def cycle():
        while True:
            nb = 0
            for b_ in tr:
                nb += 1
                yield b_
            print('  [rank %d] loader pass ended after %d batches at step %d' % (rank, nb, it), flush=True)
    it = 0; stream = cycle(); run = []; t0 = tlast = time.time(); itlast = 0; per_epoch = []
    net.train(); opt.zero_grad(set_to_none=True)
    for ep in range(a.epochs):
        acc = [0.0, 0.0]
        for mi in range(spe * a.accum):
            if it >= steps:
                break
            fl, iv = next(stream)
            fl, iv = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True)
            last = (mi + 1) % a.accum == 0
            ctx = net.no_sync() if (world > 1 and not last and hasattr(net, 'no_sync')) else contextlib.nullcontext()
            with ctx:
                l, l0 = loss_fn(net, fl, iv)
                (l / a.accum).backward()
            acc = [acc[0] + float(l.detach()) / a.accum, acc[1] + float(l0) / a.accum]
            if not last:
                continue
            if it == 0 and is0:
                print('  init recon %.3f  predict-zero %.3f  (noise floor 0.424)' % tuple(acc), flush=True)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append(acc); acc = [0.0, 0.0]; it += 1
            if not math.isfinite(run[-1][0]):
                raise RuntimeError('non-finite loss at step %d' % it)
            if is0 and (it % 100 == 0 or it == steps):
                torch.cuda.synchronize(); now = time.time(); r = np.mean(run[-100:], 0)
                print('  ep %d step %5d/%d  recon %.4f (zero %.4f)  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                      % (ep + 1, it, steps, r[0], r[1], (it - itlast) * gbs / (now - tlast), torch.cuda.max_memory_allocated() / 1e9,
                         now - t0, (now - t0) / it * (steps - it)), flush=True)
                tlast = now; itlast = it
        v = validate()
        per_epoch.append(dict(epoch=ep + 1, step=it, val_recon=v[0], val_recon_zero=v[1], val_recon_bright=v[2],
                              val_recon_faint=v[3], frac_bright=v[4], seconds=time.time() - t0))
        if is0:
            print('  EPOCH %d done  step %d  val recon %.4f (zero %.4f; noise floor 0.424) | max|S/N|>50: %.4f (%.1f%%)  others: %.4f'
                  '  elapsed %.0fs' % (ep + 1, it, v[0], v[1], v[2], 100 * v[4], v[3], time.time() - t0), flush=True)
            if not a.max_steps:
                torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep + 1}, ckdir + '%s_ep%d.pt' % (stem, ep + 1))
    if is0:
        res = dict(cfg=vars(a), steps=it, train_seconds=time.time() - t0, per_epoch=per_epoch)
        print(json.dumps(res, indent=1), flush=True)
        if not a.max_steps:
            os.makedirs('results', exist_ok=True); json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        print('RUN DONE', flush=True)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
