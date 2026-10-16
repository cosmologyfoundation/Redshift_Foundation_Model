"""ConvNeXt tokenizer = ConvNeXt autoencoder, NO Transformer (owner 2026-10-09: 「tokenizer 只保留 ConvNeXt：先單獨用重建
任務訓練 ConvNeXt 編碼器 + 解碼器（自編碼器，中間沒有 Transformer）；之後由一個 Transformer 讀它的輸出去算 z」).

  BRZ_FLUX, BRZ_IVAR (7,781 px native)
   -> native prep [asinh f, log n, measured] -> fixed resample to the 40 km/s log grid (7,526 -> padded 7,528)
   ENCODER (ConvNeXt, all on the log grid, so a redshift is a shift)
     stage 0: Conv 3->64 + 2 ConvNeXt(64)                       -> F40 (B, 64, 7,526)   the 40 km/s output
     stage 1: down x2 -> 192, 2 ConvNeXt
     stage 2: down x2 -> 384, 2 ConvNeXt
     stage 3: down x2 -> 768, 2 ConvNeXt                         -> TOK (B, 941, 768)    one token per 8 bins (320 km/s)
   BOTTLENECK: Linear 768 -> 6, FSQ (levels 7,7,7,5,5,5 = 42,875 codes, ~15.4 bits per token)
     Without a bottleneck an autoencoder copies its input, noise included.  NOT measured: whether 15 bits per
     320 km/s is too tight or too loose -- the noise-floor check below says which.
   DECODER (mirror): Linear 6 -> 768, 2 ConvNeXt, up x2 ... -> 64 ch at 40 km/s -> 1x1 -> f on the log grid
   -> fixed 2-tap map back to the 7,781 native pixels -> Huber of (pred - f) * n on every measured native pixel,
      averaged per spectrum then over spectra.  Interpolated noise never enters the loss.

Noise-floor check: a perfect model of the TRUE spectrum scores 0.424 (Gaussian noise, correct ivar).  Well below
0.424 means the bottleneck lets noise through (copying); far above means it is too tight or the encoder is weak.
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
from train import ddp_setup
from model import ConvNeXt1D
from convtok import Prep, Interp

NPAD = 7528                         # 941 x 8


class FSQ(nn.Module):
    def __init__(self, levels=(7, 7, 7, 5, 5, 5)):
        super().__init__()
        self.register_buffer('hlev', (torch.tensor(levels, dtype=torch.float) - 1) / 2, persistent=False)

    def forward(self, z):
        h = self.hlev.to(z.dtype)
        b = torch.tanh(z) * h
        q = b + (torch.round(b) - b).detach()                     # straight-through rounding
        return q / h


def stage(cin, cout, n, down, swiglu=False):
    layers = []
    if down:
        layers += [nn.GroupNorm(1, cin), nn.Conv1d(cin, cout, 2, stride=2)]
    elif cin != cout:
        layers += [nn.Conv1d(cin, cout, 7, padding=3)]
    return nn.Sequential(*layers, *[ConvNeXt1D(cout, swiglu=swiglu) for _ in range(n)])


def ustage(cin, cout, n, swiglu=False):
    return nn.Sequential(nn.GroupNorm(1, cin), nn.ConvTranspose1d(cin, cout, 2, stride=2), *[ConvNeXt1D(cout, swiglu=swiglu) for _ in range(n)])


class ConvAE(nn.Module):
    def __init__(self, ch=(64, 192, 384, 768), nb=2, levels=(7, 7, 7, 5, 5, 5)):
        super().__init__()
        self.ch = ch
        self.prep = Prep()
        self.to_log = Interp(LAM_NATIVE, LAM_OBS)
        self.to_nat = Interp(LAM_OBS, LAM_NATIVE)
        self.enc = nn.ModuleList([stage(3, ch[0], nb, False)] + [stage(ch[i], ch[i + 1], nb, True) for i in range(len(ch) - 1)])
        self.q_in, self.fsq, self.q_out = nn.Linear(ch[-1], len(levels)), FSQ(levels), nn.Linear(len(levels), ch[-1])
        self.dec0 = nn.Sequential(*[ConvNeXt1D(ch[-1]) for _ in range(nb)])
        self.dec = nn.ModuleList([ustage(ch[i + 1], ch[i], nb) for i in reversed(range(len(ch) - 1))])
        self.rec = nn.Conv1d(ch[0], 1, 1)

    def encode(self, flux, ivar):
        """-> F40 (B, 64, 7526), TOK (B, 941, 768), codes (B, 941, 6), f, n (native), s"""
        x, f, n, s = self.prep(flux, ivar)
        h = F.pad(self.to_log(x), (0, NPAD - N_OBS))
        F40 = None
        for i, st in enumerate(self.enc):
            h = st(h)
            if i == 0:
                F40 = h[..., :N_OBS]
        tok = h.transpose(1, 2)                                    # (B, 941, 768)
        return F40, tok, self.fsq(self.q_in(tok)), f, n, s

    def decode(self, codes):
        h = self.dec0(self.q_out(codes).transpose(1, 2))
        for st in self.dec:
            h = st(h)
        with torch.autocast('cuda', enabled=False):
            return self.to_nat(self.rec(h.float())[:, 0, :N_OBS])  # f on native pixels

    def forward(self, flux, ivar):
        F40, tok, codes, f, n, s = self.encode(flux, ivar)
        return self.decode(codes), f, n, s

    @torch.no_grad()
    def embed(self, flux, ivar):
        F40, tok, codes, f, n, s = self.encode(flux, ivar)
        return tok, F40, codes, torch.log10(s)


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
    m = ConvAE(ch=(8, 16, 24, 32), nb=1).eval()
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    with torch.no_grad():
        F40, tok, codes, f, n, s = m.encode(fl, iv)
        pred = m.decode(codes)
    print('F40', tuple(F40.shape), 'TOK', tuple(tok.shape), 'codes', tuple(codes.shape), 'pred', tuple(pred.shape))
    print('code values per dim:', [sorted(set(np.round(codes[..., i].flatten().numpy() * float(m.fsq.hlev[i]), 3).tolist())) for i in range(2)])
    x = torch.randn(1, 3, NPAD); st = m.enc[0]
    print('stage-0 shift test (40 bins): max error interior %.2e' % (torch.roll(st(x), 40, -1) - st(torch.roll(x, 40, -1)))[..., 100:-100].abs().max().item())
    m.train(); l, l0 = loss_fn(m, fl, iv); l.backward()
    print('loss %.3f zero %.3f; params without grad: %s' % (l, l0, [k for k, p in m.named_parameters() if p.grad is None]))
    print('params full size: %.2fM' % (sum(p.numel() for p in ConvAE().parameters()) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--size', type=int, default=0); ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--clean', type=int, default=-1); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_val', type=int, default=8192); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--tag', default='CONVAE')
    ap.add_argument('--levels', default='7,7,7,5,5,5', help='FSQ levels per code dim (the bottleneck)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    levels = tuple(int(x) for x in a.levels.split(','))
    core = ConvAE(levels=levels).to(dev)
    net = DDP(core, device_ids=[local]) if world > 1 else core
    if a.compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    gbs = a.bs * world * a.accum; spe = max(a.size // gbs, 1); steps = spe * a.epochs
    if a.max_steps:
        steps = min(steps, a.max_steps)
    wu = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / wu, max(0.0, (steps - s) / max(1, steps - wu))))
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, labelfree=True, clean=0, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
    stem = '%s_%d_fsq%s_cl0_mix_e%d_bs%d' % (a.tag, a.size, ''.join(str(x) for x in levels), a.epochs, gbs)
    if is0:
        print('CONVAE %s | ConvNeXt autoencoder, no Transformer | params %.2fM | tokens 941 x 768, F40 64 x 7526, codes 941 x %d '
              '(~%.1f bits/token) | size %d x %d epochs, gbs %d (%d x %d x %d) steps %d | data: original ZWARN=0, no cleaning, '
              'class mix 20%% each' % (stem, sum(p.numel() for p in core.parameters()) / 1e6, len(levels), float(np.log2(np.prod(levels))), a.size, a.epochs, gbs,
                                        a.bs, world, a.accum, steps), flush=True)
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

    def cycle():                     # step-based epochs on a cycling loader: all ranks always take the same steps
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
