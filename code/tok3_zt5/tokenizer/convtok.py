"""ConvNeXt tokenizer foundation model on the 40 km/s log-lambda grid (owner 2026-10-09: 「先來訓練一個 ConvNeXt 的
tokenizer ... 讓資料學到 40km/s 並在輸出表示出來 還有可以重建光譜」; inputs flux + ivar; 1M x 5 passes).
A new model, independent of specfm_native.py.

  BRZ_FLUX, BRZ_IVAR (7,781 px native 0.8 A)
   -> native prep: s = median |flux| (visible px), f = flux/s, n = sqrt(ivar)*s, x = [asinh f, log n, measured]
   -> fixed 2-tap resample to the 40 km/s log grid (7,526 bins)       input only; the loss never lives here
   -> ConvNeXt stem on the log grid (C ch, n_stem blocks)              redshift = shift -> equivariant features
   -> tokens: every tb = 8 bins (320 km/s) -> Linear -> d              941 tokens, equal width in velocity
   -> Transformer, L layers, pre-LN; positions = learned absolute + RoPE on the log-lambda axis (bins), so an
      attention score depends on the tokens' wavelength RATIO, which a redshift does not change
   -> decoder: Linear(d -> tb x C_dec) per token, one ConvNeXt block   (B, C_dec, 7,526) = the 40 km/s output
   -> 1x1 conv -> predicted f on the log grid -> fixed 2-tap map back to the 7,781 native pixels
   -> loss: Huber of (pred - f) * n on hidden, measured NATIVE pixels -- true ivar, no interpolated noise

Masking (the pretext task).  The patch FM's 15% scattered patches were too easy: reconstruction only beat a
33-px running median by ~0.06 above the 0.424 noise floor, and pre-training moved z by nothing measurable.  Here:
contiguous spans of 4-8 tokens (1,280-2,560 km/s); half the spans centred on the spectrum's own highest-S/N tokens
(label-free -- the data itself, no Redrock), half uniform; target fraction --mask (default 0.30, NOT measured).
Leakage guard: native pixels feeding any hidden bin are zeroed in the input (+1 px margin) and excluded from the
scale, hidden log bins are zeroed again before the stem (checked in --selftest: changing hidden pixels changes
nothing).  Loss is averaged per spectrum, then over spectra (bright spectra dominated the pixel mean before).
"""
import argparse, contextlib, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from grids import N_OBS, LAM_OBS, LAM_NATIVE, DLOG, OBS_LO
from data import Spectra
from train import ddp_setup
from model import ConvNeXt1D

N_NAT = 7781


class Prep(nn.Module):
    def forward(self, flux, ivar, hide=None):
        flux, ivar = flux.float(), ivar.float()
        m = ivar > 0
        vis = m if hide is None else m & ~hide
        s = torch.nanmedian(torch.where(vis, flux.abs(), torch.full_like(flux, float('nan'))), 1).values
        s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s)).clamp_min(1e-6)
        f = torch.where(m, flux / s[:, None], torch.zeros_like(flux))
        n = torch.where(m, ivar.clamp_min(0).sqrt() * s[:, None], torch.zeros_like(flux))
        mf = m.float()
        return torch.stack([torch.asinh(f) * mf, torch.log(n.clamp_min(1e-12)) * mf, mf], 1), f, n, s


def _taps(x_from, x_to):
    """2-tap linear interpolation indices/weights from grid x_from onto points x_to (np.interp end clamping)"""
    pos = np.clip(np.interp(x_to, x_from, np.arange(len(x_from))), 0, len(x_from) - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), len(x_from) - 2)
    return torch.from_numpy(i0), torch.from_numpy((pos - i0).astype(np.float32))


class Interp(nn.Module):
    def __init__(self, x_from, x_to):
        super().__init__()
        i0, w = _taps(x_from, x_to)
        self.register_buffer('i0', i0, persistent=False); self.register_buffer('w', w, persistent=False)

    def forward(self, e):
        return e[..., self.i0] * (1 - self.w) + e[..., self.i0 + 1] * self.w


def rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos.to(q.dtype), sin.to(q.dtype)), rope(k, cos.to(k.dtype), sin.to(k.dtype))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class ConvTok(nn.Module):
    def __init__(self, d=768, L=12, heads=12, C=128, n_stem=4, tb=8, C_dec=64):
        super().__init__()
        self.tb, self.nt, self.C_dec = tb, math.ceil(N_OBS / tb), C_dec
        self.prep = Prep()
        self.to_log = Interp(LAM_NATIVE, LAM_OBS)        # native -> 40 km/s, inputs only
        self.to_nat = Interp(LAM_OBS, LAM_NATIVE)        # 40 km/s -> native, predictions only
        self.stem = nn.Conv1d(3, C, 7, padding=3)
        self.blocks = nn.ModuleList([ConvNeXt1D(C) for _ in range(n_stem)])
        self.inp = nn.Linear(C * tb, d)
        self.tok0, self.scale_in = nn.Parameter(torch.zeros(1, 1, d)), nn.Linear(1, d)
        self.msk = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.nt + 1, d) * 0.02)
        self.layers = nn.ModuleList([Block(d, heads) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        hd = d // heads
        u = np.concatenate([[0.0], np.arange(self.nt) * tb + (tb - 1) / 2])       # token centres, 40 km/s bins
        om = 2 * np.pi / np.geomspace(16, 32768, hd // 2)                          # periods 640 km/s .. whole grid
        ang = torch.from_numpy(np.outer(u, om)).float()
        self.register_buffer('cos', ang.cos(), persistent=False); self.register_buffer('sin', ang.sin(), persistent=False)
        self.unpatch = nn.Linear(d, tb * C_dec)
        self.dec = ConvNeXt1D(C_dec)
        self.rec = nn.Conv1d(C_dec, 1, 1)
        # native pixels whose input or prediction touches log bin j: from to_log taps (input side)
        self.register_buffer('nat_i0', self.to_nat.i0.clone(), persistent=False)

    def hidden_native(self, hide_log):
        """native pixels to hide: every pixel that feeds or is fed by a hidden log bin, +1 px margin"""
        h = hide_log[:, self.nat_i0] | hide_log[:, self.nat_i0 + 1]
        return F.max_pool1d(h.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0

    def encode(self, flux, ivar, mtok):
        """-> token states (B, 1+nt, d), f, n (native), s, hide_nat"""
        B = flux.shape[0]
        hide_log = mtok.repeat_interleave(self.tb, 1)[:, :N_OBS]
        hide_nat = self.hidden_native(hide_log)
        x, f, n, s = self.prep(flux, ivar, hide=hide_nat)
        x = x * (~hide_nat).unsqueeze(1).to(x.dtype)
        xl = self.to_log(x) * (~hide_log).unsqueeze(1).to(x.dtype)
        z = self.stem(xl)
        for b in self.blocks:
            z = b(z)
        z = F.pad(z, (0, self.nt * self.tb - N_OBS))
        t = self.inp(z.view(B, -1, self.nt, self.tb).permute(0, 2, 1, 3).reshape(B, self.nt, -1))
        t = torch.where(mtok.unsqueeze(-1), self.msk.expand(B, self.nt, -1).to(t.dtype), t)
        t0 = self.tok0.expand(B, -1, -1) + self.scale_in(torch.log10(s).view(B, 1, 1))
        h = torch.cat([t0.to(t.dtype), t], 1) + self.pos
        for lay in self.layers:
            h = lay(h, self.cos, self.sin)
        return self.norm(h), f, n, s, hide_nat

    def features(self, h):
        """token states -> the 40 km/s output representation (B, C_dec, 7,526)"""
        B = h.shape[0]
        e = self.unpatch(h[:, 1:]).view(B, self.nt, self.tb, self.C_dec).permute(0, 3, 1, 2).reshape(B, self.C_dec, -1)
        return self.dec(e[..., :N_OBS])

    def forward(self, flux, ivar, mtok=None):
        if mtok is None:
            mtok = torch.zeros(flux.shape[0], self.nt, dtype=torch.bool, device=flux.device)
        h, f, n, s, hide_nat = self.encode(flux, ivar, mtok)
        e = self.features(h)
        with torch.autocast('cuda', enabled=False):
            pred = self.to_nat(self.rec(e.float()).squeeze(1))         # f on the native pixels
        return pred, f, n, s, hide_nat

    @torch.no_grad()
    def embed(self, flux, ivar):
        """-> tokens (B, nt, d), 40 km/s features (B, C_dec, 7,526), spectrum vector (B, d), log10 scale (B,)"""
        mt = torch.zeros(flux.shape[0], self.nt, dtype=torch.bool, device=flux.device)
        h, f, n, s, _ = self.encode(flux, ivar, mt)
        return h[:, 1:], self.features(h), h[:, 1:].mean(1), torch.log10(s)


def span_masks(core, flux, ivar, frac, g, peak=0.5):
    """spans of 4-8 tokens; a share `peak` of spans centred on high-S/N tokens (label-free), the rest uniform"""
    B, nt, dev = flux.shape[0], core.nt, flux.device
    with torch.no_grad():
        x, f, n, s = core.prep(flux, ivar)
        snr = core.to_log((f * n).abs().unsqueeze(1)).squeeze(1)
        snr = F.pad(snr, (0, nt * core.tb - N_OBS)).view(B, nt, core.tb).amax(-1)
        K = max(2, int(math.ceil(frac * nt / 6.0)))
        kp = int(round(K * peak))
        c = [torch.randint(0, nt, (B, K - kp), generator=g, device=dev)]
        if kp:
            c.insert(0, torch.multinomial(snr.float().clamp_min(0) ** 2 + 1e-6, kp, replacement=True, generator=g))
        c = torch.cat(c, 1)
        ln = torch.randint(4, 9, (B, K), generator=g, device=dev)
        lo = c - ln // 2
        t = torch.arange(nt, device=dev).view(1, 1, -1)
        return ((t >= lo.unsqueeze(-1)) & (t < (lo + ln).unsqueeze(-1))).any(1)


def loss_fn(net, core, flux, iv, g, frac, per_spec=False):
    mt = span_masks(core, flux, iv, frac, g)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, f, n, s, hide_nat = net(flux, iv, mt)
    pix = hide_nat & (n > 0)
    hub = F.huber_loss((pred - f) * n, torch.zeros_like(f), delta=1.0, reduction='none')
    hub0 = F.huber_loss(-f * n, torch.zeros_like(f), delta=1.0, reduction='none')
    cnt = pix.sum(1).clamp_min(1)
    ps, ps0 = (hub * pix).sum(1) / cnt, (hub0 * pix).sum(1) / cnt
    ok = pix.any(1)
    l, l0 = ps[ok].mean(), ps0[ok].mean()                                  # per spectrum, then over spectra
    if not per_spec:
        return l, l0, mt.float().mean()
    return l, l0, ps, (f.abs() * n).amax(1), hub[pix].mean()


def selftest():
    torch.manual_seed(0)
    core = ConvTok(d=64, L=2, heads=4, C=16, n_stem=1, C_dec=8).eval()
    fl = torch.randn(2, N_NAT) + 1; iv = torch.rand(2, N_NAT) + 0.5
    g = torch.Generator(); g.manual_seed(1)
    mt = span_masks(core, fl, iv, 0.3, g)
    print('tokens %d, mask fraction %.3f' % (core.nt, mt.float().mean().item()))
    with torch.no_grad():
        p0, f, n, s, hn = core(fl, iv, mt)
        fl2 = fl.clone(); fl2[hn] += 100 * torch.randn(int(hn.sum()))
        p1 = core(fl2, iv, mt)[0]
    print('leakage: change only hidden native pixels -> max |pred diff| %.2e (hidden native px %d)' % ((p0 - p1).abs().max().item(), hn.sum().item()))
    # equivariance of the stem on the log grid
    xl = torch.randn(1, 3, N_OBS); y0, y1 = core.stem(xl), core.stem(torch.roll(xl, 40, -1))
    for b in core.blocks:
        y0, y1 = b(y0), b(y1)
    print('stem shift test (40 bins = 1600 km/s): max error interior %.2e' % (torch.roll(y0, 40, -1) - y1)[..., 100:-100].abs().max().item())
    # RoPE: scores depend on position differences only
    q, k = torch.randn(1, 1, 3, 16), torch.randn(1, 1, 3, 16)
    om = 2 * np.pi / np.geomspace(16, 32768, 8)
    def sc(u):
        a = torch.from_numpy(np.outer(u, om)).float()
        return rope(q, a.cos(), a.sin()) @ rope(k, a.cos(), a.sin()).transpose(-1, -2)
    print('RoPE shift test (all positions +1337 bins): max score diff %.2e' % (sc(np.array([5., 100., 3000.])) - sc(np.array([1342., 1437., 4337.]))).abs().max().item())
    print('round trip native -> log -> native on a smooth curve: max rel error %.2e'
          % ((core.to_nat(core.to_log(torch.from_numpy(np.sin(LAM_NATIVE / 300.)).float())) - torch.from_numpy(np.sin(LAM_NATIVE / 300.)).float()).abs().max().item()))
    print('embed shapes:', [tuple(t.shape) for t in core.embed(fl, iv)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--size', type=int, default=0)
    ap.add_argument('--d', type=int, default=768); ap.add_argument('--L', type=int, default=12)
    ap.add_argument('--C', type=int, default=128); ap.add_argument('--n_stem', type=int, default=4)
    ap.add_argument('--C_dec', type=int, default=64); ap.add_argument('--mask', type=float, default=0.30)
    ap.add_argument('--bs', type=int, default=64, help='per GPU'); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--clean', type=int, default=-1, help='must be 0: original ZWARN=0 files (owner rule 2026-10-09)')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--n_val', type=int, default=8192)
    ap.add_argument('--epochs', type=int, default=5); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--tag', default='CONVTOK')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    core = ConvTok(a.d, a.L, max(1, a.d // 64), a.C, a.n_stem, 8, a.C_dec).to(dev)
    net = DDP(core, device_ids=[local], find_unused_parameters=False) if world > 1 else core
    if a.compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    gbs = a.bs * world * a.accum; steps = max(a.size // gbs, 1) * a.epochs
    if a.max_steps:
        steps = min(steps, a.max_steps)
    wu = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / wu, max(0.0, (steps - s) / max(1, steps - wu))))
    # class mix 20% each is data.Spectra's default (owner 2026-10-09); no cleaning (clean=0)
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, labelfree=True, clean=0, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
    g = torch.Generator(device=dev); g.manual_seed(a.seed * 100 + rank)
    npar = sum(p.numel() for p in core.parameters())
    stem = '%s_%d_C%dx%d_d%d_L%d_m%g_cl0_mix_e%d_bs%d' % (a.tag, a.size, a.C, a.n_stem, a.d, a.L, a.mask, a.epochs, gbs)
    if is0:
        print('CONVTOK %s | tokens %d (8 bins = 320 km/s) | params %.2fM (stem %.2fM) | size %d x %d epochs, gbs %d (%d x %d x %d) '
              'steps %d warmup %d lr %.1e mask %.2f | data: original ZWARN=0, no cleaning, class mix 20%% each'
              % (stem, core.nt + 1, npar / 1e6, sum(p.numel() for p in list(core.stem.parameters()) + list(core.blocks.parameters())) / 1e6,
                 a.size, a.epochs, gbs, a.bs, world, a.accum, steps, wu, a.lr, a.mask), flush=True)
    ckdir = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'

    def validate():
        core.eval(); gv = torch.Generator(device=dev); gv.manual_seed(999 + rank)
        va = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, labelfree=True, clean=0, native=True),
                        batch_size=64, num_workers=8, pin_memory=True, drop_last=True)
        L_, L0, PS, SN, LP, n = [], [], [], [], [], 0
        with torch.no_grad():
            for fl, iv in va:
                if n >= a.n_val // world:
                    break
                l, l0, ps, sn, lp = loss_fn(core, core, fl.to(dev), iv.to(dev), gv, a.mask, per_spec=True)
                L_.append(float(l)); L0.append(float(l0)); PS.append(ps); SN.append(sn); LP.append(float(lp)); n += len(fl)
        PS, SN = torch.cat(PS), torch.cat(SN); br = SN > 50
        out = torch.tensor([np.mean(L_), np.mean(L0), PS[br].mean().item() if br.any() else float('nan'),
                            PS[~br].mean().item(), br.float().mean().item(), np.mean(LP)], device=dev)
        if world > 1:
            dist.all_reduce(out); out = out / world
        core.train()
        return [float(v) for v in out]

    # Step-based epochs on a cycling loader (2026-10-09 fix): the first run hung at step ~730 because rank 2's
    # loader ended its pass early and entered validate()'s all_reduce while ranks 0,1,3 were still in gradient
    # all_reduces (NCCL timeout).  Every rank now runs exactly spe x accum micro-batches per epoch, restarting its
    # loader if a pass ends early, and says so in the log.
    def cycle():
        while True:
            nb = 0
            for b_ in tr:
                nb += 1
                yield b_
            print('  [rank %d] loader pass ended after %d batches (%d spectra) at step %d' % (rank, nb, nb * a.bs, it), flush=True)
    stream = cycle()
    spe = max(a.size // gbs, 1)
    net.train(); t0 = time.time(); it = 0; run = []; tlast = time.time(); itlast = 0; per_epoch = []
    opt.zero_grad(set_to_none=True)
    for ep in range(a.epochs):
        if it >= steps:
            break
        micro = 0; acc = [0.0, 0.0, 0.0]
        for _ in range(spe * a.accum):
            if it >= steps:
                break
            fl, iv = next(stream)
            fl, iv = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True)
            last = (micro + 1) % a.accum == 0
            ctx = net.no_sync() if (world > 1 and not last and hasattr(net, 'no_sync')) else contextlib.nullcontext()
            with ctx:
                l, l0, mf = loss_fn(net, core, fl, iv, g, a.mask)
                (l / a.accum).backward()
            acc = [acc[0] + float(l.detach()) / a.accum, acc[1] + float(l0) / a.accum, acc[2] + float(mf) / a.accum]; micro += 1
            if not last:
                continue
            if it == 0 and is0:
                print('  init recon %.3f  predict-zero %.3f  mask fraction %.3f' % tuple(acc), flush=True)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append(acc); acc = [0.0, 0.0, 0.0]; it += 1
            if not math.isfinite(run[-1][0]):
                raise RuntimeError('non-finite loss at step %d' % it)
            if is0 and (it % 100 == 0 or it == steps):
                torch.cuda.synchronize(); now = time.time(); r = np.mean(run[-100:], 0)
                print('  ep %d step %5d/%d  recon %.4f (zero %.4f) mask %.3f  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                      % (ep + 1, it, steps, r[0], r[1], r[2], (it - itlast) * gbs / (now - tlast),
                         torch.cuda.max_memory_allocated() / 1e9, now - t0, (now - t0) / it * (steps - it)), flush=True)
                tlast = now; itlast = it
        v = validate()
        per_epoch.append(dict(epoch=ep + 1, step=it, seconds=time.time() - t0, val_recon=v[0], val_recon_zero=v[1],
                              val_recon_bright=v[2], val_recon_faint=v[3], frac_bright=v[4], val_recon_pixelmean=v[5]))
        if is0:
            print('  EPOCH %d done  step %d  val recon %.4f (zero %.4f) | max|S/N|>50: %.4f (%.1f%%)  others: %.4f | '
                  'pixel-mean %.4f  elapsed %.0fs' % (ep + 1, it, v[0], v[1], v[2], 100 * v[4], v[3], v[5], time.time() - t0), flush=True)
            if not a.max_steps:
                torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep + 1, 'step': it}, ckdir + '%s_ep%d.pt' % (stem, ep + 1))
    if is0:
        res = dict(cfg=vars(a), params=npar, gbs=gbs, steps=it, train_seconds=time.time() - t0,
                   spectra_per_s=it * gbs / (time.time() - t0), per_epoch=per_epoch)
        print(json.dumps(res, indent=1), flush=True)
        if not a.max_steps:
            os.makedirs('results', exist_ok=True); json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        print('RUN DONE', flush=True)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
