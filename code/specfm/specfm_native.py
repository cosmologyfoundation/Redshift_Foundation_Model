"""Spectrum foundation model, native input (FM_DESIGN.md; owner 2026-10-09: "直接把 BRZ_FLUX / BRZ_IVAR /
BRZ_MASK 放進模型").  Label-free masked-patch reconstruction on the DESI 7781-px BRZ grid.

Why not specfm.py's input: it interpolated onto a log grid (correlated noise, bad pixels leaking into good
ones), converted to S/N (brightness weakened) and clipped at +-50 (bright spectra flattened; a S/N ~ 100
star was reconstructed as a flat 50).  A Transformer with learned positions does not need redshift to be a
shift, so none of that is required here.

Preprocessing (NativePrep, inside the model; nothing is clipped, nothing is interpolated):
  measured  m = ivar > 0 and (mask == 0 if a mask is given)
  scale     s = median |flux| over measured pixels (per spectrum, > 0)      -> kept, fed as log10 s
  flux      f = flux / s                                                     (reconstruction target)
  noise     n = sqrt(ivar) * s       so (prediction - f) * n = (prediction * s - flux) * sqrt(ivar), in sigma
  input     3 channels per pixel: asinh(f) * m, log(n) * m, m                (asinh only compresses the input)
Tokens    7781 px -> ceil(7781/p) patches + token 0 = learned vector + Linear(log10 s): the brightness lives there.
Loss      Huber(delta 1) of (prediction - f) * n on masked, measured pixels -- noise units, target NOT clipped.
          The reconstruction head runs in fp32 (bf16 is 0.4% relative = 0.4 sigma at S/N 100).

Basis.  MEASURED: mask != 0 implies ivar = 0 on clean-2 (mask_audit.py).  Everything else is a DEFAULT to be
tested: the median scale, asinh/log input transforms, d/L/heads, p = 16 px (12.8 A, 24-67 km/s per px),
15% scattered patches, pre-LN, lr.  Token 0 is still not trained by any loss: use the patch-output mean.
"""
import argparse, contextlib, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data import Spectra
from train import ddp_setup
from model import ConvNeXt1D

N_NATIVE = 7781


class NativePrep(nn.Module):
    def forward(self, flux, ivar, mask=None, hide=None):
        """hide (B, 7781) bool: pixels the model must not see (masked patches); excluded from the scale too,
        otherwise the hidden pixels would leak through the median."""
        flux, ivar = flux.float(), ivar.float()
        m = ivar > 0
        if mask is not None:
            m = m & (mask == 0)
        vis = m if hide is None else m & ~hide
        af = torch.where(vis, flux.abs(), torch.full_like(flux, float('nan')))
        s = torch.nanmedian(af, dim=1).values
        s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s)).clamp_min(1e-6)
        f = torch.where(m, flux / s[:, None], torch.zeros_like(flux))
        n = torch.where(m, ivar.clamp_min(0).sqrt() * s[:, None], torch.zeros_like(flux))
        mf = m.float()
        x = torch.stack([torch.asinh(f) * mf, torch.log(n.clamp_min(1e-12)) * mf, mf], 1)
        return x, f, n, s


class SpecFMNative(nn.Module):
    def __init__(self, d=768, L=12, p=16, heads=12, stem_c=0, stem_n=0):
        super().__init__()
        self.p, self.np_ = p, math.ceil(N_NATIVE / p)
        self.prep = NativePrep()
        # optional ConvNeXt front end at native resolution (FM_DESIGN B1, owner 2026-10-09 "直接走d"):
        # Conv1d 3 -> stem_c (k 7), stem_n ConvNeXt1D blocks (dw k 7 + 4x MLP, layer scale 1e-6), then each
        # 16-px patch of stem_c channels -> Linear -> d.  Receptive field 7 + 6 stem_n px.  DEFAULT, untested.
        self.stem_c = stem_c
        if stem_c:
            self.stem = nn.Conv1d(3, stem_c, 7, padding=3)
            self.blocks = nn.ModuleList([ConvNeXt1D(stem_c) for _ in range(stem_n)])
            self.inp = nn.Linear(stem_c * p, d)
        else:
            self.inp = nn.Linear(3 * p, d)
        self.tok0 = nn.Parameter(torch.zeros(1, 1, d))
        self.scale_in = nn.Linear(1, d)
        self.msk = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.np_ + 1, d) * 0.02)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, L, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.rec = nn.Linear(d, p)

    def patches(self, x):
        B, C, N = x.shape
        x = F.pad(x, (0, self.np_ * self.p - N))
        return x.view(B, C, self.np_, self.p).permute(0, 2, 1, 3).reshape(B, self.np_, C * self.p)

    def encode(self, x, s, m):
        """x (B,3,7781) prepared input, s (B,) scale, m (B,n_patches) bool hidden -> hidden states (B, 1+n, d)"""
        B = x.shape[0]
        # hide masked pixels in the INPUT (all channels, incl. the measured flag): a convolution would otherwise
        # carry a hidden patch's pixels into its visible neighbours' tokens
        hid = m.repeat_interleave(self.p, 1)[:, :x.shape[-1]]
        x = x * (~hid).unsqueeze(1).to(x.dtype)
        if self.stem_c:
            z = self.stem(x)
            for b in self.blocks:
                z = b(z)
            x = z
        t = self.inp(self.patches(x))
        t = torch.where(m.unsqueeze(-1), self.msk.expand(B, self.np_, -1), t)
        t0 = self.tok0.expand(B, -1, -1) + self.scale_in(torch.log10(s).view(B, 1, 1).to(t.dtype))
        return self.tf(torch.cat([t0, t], 1) + self.pos)

    def head(self, h):
        """fp32 reconstruction head: (B, 1+n, d) -> f prediction (B, 7781)"""
        with torch.autocast('cuda', enabled=False):
            return self.rec(h[:, 1:].float()).reshape(h.shape[0], -1)[:, :N_NATIVE]

    def forward(self, flux, ivar, mask=None, m=None):
        if m is None:
            m = torch.zeros(flux.shape[0], self.np_, dtype=torch.bool, device=flux.device)
        x, f, n, s = self.prep(flux, ivar, mask, hide=m.repeat_interleave(self.p, 1)[:, :flux.shape[-1]])
        return self.head(self.encode(x, s, m)), f, n, s

    @torch.no_grad()
    def embed(self, flux, ivar, mask=None):
        """raw BRZ arrays -> per-patch states (B, n, d), their mean (B, d) [the spectrum vector], log10 scale (B,)"""
        x, f, n, s = self.prep(flux, ivar, mask)
        h = self.encode(x, s, torch.zeros(x.shape[0], self.np_, dtype=torch.bool, device=x.device))[:, 1:]
        return h, h.mean(1), torch.log10(s)

    @torch.no_grad()
    def reconstruct(self, flux, ivar, mask=None, m=None):
        """raw BRZ arrays -> reconstructed flux in the input's units (B, 7781) = prediction * scale"""
        pred, f, n, s = self(flux, ivar, mask, m)
        return pred * s[:, None]


def masks(B, n, frac, g, dev):
    k = max(1, int(round(frac * n)))
    idx = torch.rand(B, n, generator=g, device=dev).argsort(1)[:, :k]
    return torch.zeros(B, n, dtype=torch.bool, device=dev).scatter_(1, idx, True)


def loss_fn(net, core, fl, iv, g, frac, per_spec=False):
    m = masks(fl.shape[0], core.np_, frac, g, fl.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, f, n, s = net(fl, iv, None, m)
    pix = m.repeat_interleave(core.p, 1)[:, :N_NATIVE] & (n > 0)
    r = (pred - f) * n
    r0 = -f * n                                            # predict zero = -S/N
    hub = F.huber_loss(r, torch.zeros_like(r), delta=1.0, reduction='none')
    hub0 = F.huber_loss(r0, torch.zeros_like(r0), delta=1.0, reduction='none')
    l = hub[pix].mean(); l0 = hub0[pix].mean()
    if not per_spec:
        return l, l0
    cnt = pix.sum(1).clamp_min(1)
    snr = (f.abs() * n).amax(1)                            # max |S/N| per spectrum, unclipped
    return l, l0, (hub * pix).sum(1) / cnt, snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--d', type=int, default=768); ap.add_argument('--L', type=int, default=12)
    ap.add_argument('--p', type=int, default=16); ap.add_argument('--mask', type=float, default=0.15)
    ap.add_argument('--bs', type=int, default=64, help='per GPU'); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--clean', type=int, default=-1, help='0 = original ZWARN=0 files (owner rule 2026-10-09: no cleaning); pass explicitly')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--n_val', type=int, default=8192)
    ap.add_argument('--tag', default='SPECFMN'); ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--stem_c', type=int, default=0); ap.add_argument('--stem_n', type=int, default=0)
    ap.add_argument('--accum', type=int, default=1, help='micro-batches per optimiser step (global batch = bs x GPUs x accum)')
    a = ap.parse_args()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    core = SpecFMNative(a.d, a.L, a.p, max(1, a.d // 64), a.stem_c, a.stem_n).to(dev)
    net = DDP(core, device_ids=[local]) if world > 1 else core
    if a.compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    gbs = a.bs * world * a.accum; spe = max(a.size // gbs, 1); steps = spe * a.epochs
    if a.max_steps:
        steps = min(steps, a.max_steps)
    wu = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / wu, max(0.0, (steps - s) / max(1, steps - wu))))
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, labelfree=True, clean=a.clean, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
    g = torch.Generator(device=dev); g.manual_seed(a.seed * 100 + rank)
    npar = sum(p.numel() for p in core.parameters())
    if is0:
        print('SPECFM-native stem %dx%d d=%d L=%d p=%d tokens=%d params %.2fM | size %d x %d epochs, gbs %d (%d x %d) steps %d warmup %d mask %.2f compile %s'
              % (a.stem_c, a.stem_n, a.d, a.L, a.p, core.np_ + 1, npar / 1e6, a.size, a.epochs, gbs, a.bs * a.accum, world, steps, wu, a.mask, a.compile), flush=True)
    stem = '%s_%d_s%dx%d_d%d_L%d_p%d_cl%d_e%d_bs%d' % (a.tag, a.size, a.stem_c, a.stem_n, a.d, a.L, a.p, a.clean, a.epochs, gbs)
    ckdir = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'

    def validate():
        core.eval(); gv = torch.Generator(device=dev); gv.manual_seed(999 + rank)
        va = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, labelfree=True, clean=a.clean, native=True),
                        batch_size=64, num_workers=8, pin_memory=True, drop_last=True)
        L_, L0, PS, SN, n = [], [], [], [], 0
        with torch.no_grad():
            for fl, iv in va:
                if n >= a.n_val // world:
                    break
                l, l0, ps, sn = loss_fn(core, core, fl.to(dev), iv.to(dev), gv, a.mask, per_spec=True)
                L_.append(float(l)); L0.append(float(l0)); PS.append(ps); SN.append(sn); n += len(fl)
        PS, SN = torch.cat(PS), torch.cat(SN)
        bright = SN > 50
        out = torch.tensor([np.mean(L_), np.mean(L0), PS[bright].mean().item() if bright.any() else float('nan'),
                            PS[~bright].mean().item(), bright.float().mean().item()], device=dev)
        if world > 1:
            dist.all_reduce(out); out = out / world
        core.train()
        return [float(v) for v in out]

    opt.zero_grad(set_to_none=True)
    net.train(); t0 = time.time(); it = 0; run = []; tlast = time.time(); itlast = 0; per_epoch = []
    for ep in range(a.epochs):
        if it >= steps:
            break
        micro = 0; acc = [0.0, 0.0]
        for fl, iv in tr:
            if it >= steps:
                break
            fl, iv = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True)
            last = (micro + 1) % a.accum == 0
            ctx = net.no_sync() if (world > 1 and not last and hasattr(net, 'no_sync')) else contextlib.nullcontext()
            with ctx:
                l, l0 = loss_fn(net, core, fl, iv, g, a.mask)
                (l / a.accum).backward()
            acc[0] += float(l) / a.accum; acc[1] += float(l0) / a.accum; micro += 1
            if not last:
                continue
            l = torch.tensor(acc[0]); l0 = torch.tensor(acc[1]); acc = [0.0, 0.0]
            if it == 0 and is0:
                print('  init recon %.3f  predict-zero %.3f' % (float(l), float(l0)), flush=True)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append([float(l), float(l0)]); it += 1
            if not math.isfinite(float(l)):
                raise RuntimeError('non-finite loss at step %d' % it)
            if is0 and (it % 100 == 0 or it == steps):
                torch.cuda.synchronize(); now = time.time(); dt = now - tlast; r = np.mean(run[-100:], 0)
                print('  ep %d step %5d/%d  recon %.4f (zero %.4f)  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                      % (ep + 1, it, steps, r[0], r[1], (it - itlast) * gbs / dt, torch.cuda.max_memory_allocated() / 1e9,
                         now - t0, (now - t0) / it * (steps - it)), flush=True)
                tlast = now; itlast = it
        v = validate()
        per_epoch.append(dict(epoch=ep + 1, step=it, seconds=time.time() - t0, val_recon=v[0], val_recon_zero=v[1],
                              val_recon_bright=v[2], val_recon_faint=v[3], frac_bright=v[4]))
        if is0:
            print('  EPOCH %d done  step %d  val recon %.4f (zero %.4f) | max|S/N|>50: %.4f (%.1f%% of spectra)  others: %.4f  elapsed %.0fs'
                  % (ep + 1, it, v[0], v[1], v[2], 100 * v[4], v[3], time.time() - t0), flush=True)
            if not a.max_steps:
                torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep + 1, 'step': it}, ckdir + '%s_ep%d.pt' % (stem, ep + 1))
    if is0:
        res = dict(cfg=vars(a), params=npar, gbs=gbs, steps=it, train_seconds=time.time() - t0,
                   spectra_per_s=it * gbs / (time.time() - t0), per_epoch=per_epoch)
        print(json.dumps(res, indent=1), flush=True)
        if not a.max_steps:
            json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
