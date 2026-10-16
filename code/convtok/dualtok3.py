"""Dual-scale tokenizer v3: same architecture as dualtok.py, new training objective and data mix (owner 2026-10-12:
「如果 tokenizer 用 45% 45% ELG QSO 的比例來練呢？」 -> 「同時改 先練tokenizer」).

Why: v2 reconstructs below the noise floor (0.346 < 0.424) -- its per-pixel loss rewards copying noise exactly as much as
keeping a faint line, and a 4 sigma ELG line is only ~1-1.5 sigma per pixel.  Three changes, no labels anywhere:
  1. MASKED reconstruction: 25-45 random windows of 25-75 native px (20-60 A) are hidden from the input (ivar -> 0);
     the loss is on every measured pixel, but masked positions count 4x (lambda_vis = 0.25).  Noise inside a hidden
     window cannot be predicted from the rest, so capacity goes to signal that other parts of the spectrum imply.
  2. DENOISING: the visible input gets extra Gaussian noise of variance u/ivar, u ~ U(0, 1) per spectrum, and its ivar
     is divided by (1+u); the target is the original spectrum.
  3. MULTI-SCALE loss: the ivar-weighted residual is summed in boxcars of 1, 3, 9, 27, 81, 243 native px
     (~40 .. 9,700 km/s) and divided by sqrt(sum ivar): the residual's significance at that scale (matched filter).
     Huber(delta 1) per scale, scales averaged.  Pure noise gives N(0,1) at every scale -> Huber 0.424 each, so
     every scale has the same noise floor and a value below it means noise is being reproduced at that width.
Data: training class mix QSO 45 / ELG 45 / LRG, BGS, STAR 3.33 each (target bits, not Redrock); original ZWARN=0,
no cleaning, labelfree.  Validation keeps 20% each.  Reported every epoch: the v2 metric (plain per-pixel recon, no mask,
no noise) for comparison, and the masked / visible loss per scale.
"""
import argparse, contextlib, json, math, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from dualtok import DualTok, loss_fn as loss_v2
from data import Spectra
import muonopt
from train import ddp_setup

SCALES = (1, 3, 9, 27, 81, 243)
LAM_VIS = 0.25


def make_mask(B, L, gen, dev):
    """(B, L) bool: union of 25-45 windows of 25-75 px per spectrum (about 15-30% of the pixels)"""
    K = 45
    k = torch.randint(25, 46, (B, 1), generator=gen, device=dev)
    c = torch.randint(0, L, (B, K), generator=gen, device=dev)
    w = torch.randint(25, 76, (B, K), generator=gen, device=dev)
    use = torch.arange(K, device=dev)[None] < k
    lo = torch.where(use, (c - w // 2).clamp(0, L), torch.full_like(c, L))
    hi = torch.where(use, (c + w // 2).clamp(0, L), torch.full_like(c, L))
    d = torch.zeros(B, L + 1, device=dev)
    d.scatter_add_(1, lo, torch.ones_like(lo, dtype=d.dtype)); d.scatter_add_(1, hi, -torch.ones_like(hi, dtype=d.dtype))
    return d.cumsum(1)[:, :L] > 0


def box(x, k):
    return F.avg_pool1d(x.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=True).squeeze(1) * k


def ms_terms(pred, f, n, region):
    """per scale: mean Huber of the residual significance over positions whose centre is in `region` (B, L) bool"""
    w = (n * n).float(); r = (pred.float() - f.float())
    out = []
    for k in SCALES:
        num, den = box(w * r, k), box(w, k)
        cnt = box((w > 0).float(), k)
        S = torch.where(den > 0, num / den.clamp_min(1e-30).sqrt(), torch.zeros_like(num))
        ok = region & (w > 0) & (cnt >= 0.5 * k)
        h = F.huber_loss(S, torch.zeros_like(S), delta=1.0, reduction='none')
        out.append((h * ok).sum() / ok.sum().clamp_min(1))
    return torch.stack(out)


def loss_v3(net, flux, iv, gen):
    flux, iv = flux.float(), iv.float().clamp_min(0)
    B, L = flux.shape; dev = flux.device
    mask = make_mask(B, L, gen, dev)
    u = torch.rand(B, 1, generator=gen, device=dev)
    meas = iv > 0
    noise = torch.randn(B, L, generator=gen, device=dev) * torch.where(meas, (u / iv.clamp_min(1e-30)).sqrt(), torch.zeros_like(iv))
    fin = torch.where(mask, torch.zeros_like(flux), flux + noise)
    ivin = torch.where(mask, torch.zeros_like(iv), iv / (1 + u))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, _, _, s = net(fin, ivin)
    s = s.float()
    f = torch.where(meas, flux / s[:, None], torch.zeros_like(flux))            # target: ORIGINAL spectrum, input's scale
    n = torch.where(meas, iv.sqrt() * s[:, None], torch.zeros_like(flux))       # ORIGINAL ivar
    tm, tv = ms_terms(pred, f, n, mask), ms_terms(pred, f, n, ~mask)
    return tm.mean() + LAM_VIS * tv.mean(), tm.detach(), tv.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=300000); ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--clean', type=int, default=-1); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_val', type=int, default=8192); ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--muon', action='store_true', help='Muon on the ConvNeXt up/down matrices (muonopt.CONVNEXT_HIDDEN), AdamW on the rest')
    ap.add_argument('--muon_lr', type=float, default=None)
    ap.add_argument('--swiglu', action='store_true', help='SwiGLU MLP in every ConvNeXt block (parameter-matched)')
    ap.add_argument('--compile', action='store_true'); ap.add_argument('--tag', default='DUALTOK3')
    ap.add_argument('--lv_n', default='5,5')
    ap.add_argument('--mix_train', default='QSO:0.45,ELG:0.45,LRG:0.0333,BGS:0.0333,STAR:0.0334')
    a = ap.parse_args()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 100 + rank)
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
    mixw = {k: float(v) for k, v in (x.split(':') for x in a.mix_train.split(','))}
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, labelfree=True, clean=0, native=True, mix=mixw),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
    stem = '%s_%d_n%s_cl0_q45e45_e%d_bs%d' % (a.tag, a.size, ''.join(str(x) for x in lv_n), a.epochs, gbs)
    if is0:
        print('DUALTOK3 %s | objective: masked (x4) + denoise + multi-scale %s px | train mix %s | params %.2fM | %d x %d epochs, '
              'gbs %d, steps %d | data: original ZWARN=0, no cleaning, labelfree' % (stem, SCALES, mixw,
              sum(p.numel() for p in core.parameters()) / 1e6, a.size, a.epochs, gbs, steps), flush=True)
    ckdir = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'

    def validate():
        core.eval()
        va = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, labelfree=True, clean=0, native=True),
                        batch_size=64, num_workers=8, pin_memory=True, drop_last=True)
        g = torch.Generator(device=dev); g.manual_seed(12345 + rank)
        V2, TM, TV, n = [], [], [], 0
        with torch.no_grad():
            for fl, iv in va:
                if n >= a.n_val // world:
                    break
                fl, iv = fl.to(dev), iv.to(dev)
                V2.append(float(loss_v2(core, fl, iv)[0]))
                _, tm, tv = loss_v3(core, fl, iv, g)
                TM.append(tm); TV.append(tv); n += len(fl)
        out = torch.cat([torch.tensor([np.mean(V2)], device=dev), torch.stack(TM).mean(0), torch.stack(TV).mean(0)])
        if world > 1:
            dist.all_reduce(out); out = out / world
        core.train()
        return out.tolist()

    def cycle():
        while True:
            for b_ in tr:
                yield b_
    it = 0; stream = cycle(); run = []; t0 = tlast = time.time(); itlast = 0; per_epoch = []
    net.train(); opt.zero_grad(set_to_none=True)
    for ep in range(a.epochs):
        acc = 0.0
        for mi in range(spe * a.accum):
            if it >= steps:
                break
            fl, iv = next(stream)
            fl, iv = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True)
            last = (mi + 1) % a.accum == 0
            ctx = net.no_sync() if (world > 1 and not last and hasattr(net, 'no_sync')) else contextlib.nullcontext()
            with ctx:
                l, tm, tv = loss_v3(net, fl, iv, gen)
                (l / a.accum).backward()
            acc += float(l.detach()) / a.accum
            if not last:
                continue
            if it == 0 and is0:
                print('  init loss %.3f | masked per scale %s | visible %s (noise floor 0.424 each)'
                      % (acc, ' '.join('%.3f' % x for x in tm.tolist()), ' '.join('%.3f' % x for x in tv.tolist())), flush=True)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append(acc); acc = 0.0; it += 1
            if not math.isfinite(run[-1]):
                raise RuntimeError('non-finite loss at step %d' % it)
            if is0 and (it % 100 == 0 or it == steps):
                torch.cuda.synchronize(); now = time.time()
                print('  ep %d step %5d/%d  loss %.4f  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                      % (ep + 1, it, steps, np.mean(run[-100:]), (it - itlast) * gbs / (now - tlast),
                         torch.cuda.max_memory_allocated() / 1e9, now - t0, (now - t0) / it * (steps - it)), flush=True)
                tlast = now; itlast = it
        v = validate()
        k = len(SCALES)
        per_epoch.append(dict(epoch=ep + 1, step=it, val_recon_v2metric=v[0], val_masked=v[1:1 + k], val_visible=v[1 + k:],
                              seconds=time.time() - t0))
        if is0:
            print('  EPOCH %d done  step %d | v2 metric (plain per-pixel, no mask/noise) %.4f (v2 tokenizer 0.346; floor 0.424) | '
                  'masked per scale %s | visible per scale %s | scales %s px | elapsed %.0fs'
                  % (ep + 1, it, v[0], ' '.join('%.3f' % x for x in v[1:1 + k]), ' '.join('%.3f' % x for x in v[1 + k:]),
                     SCALES, time.time() - t0), flush=True)
            if not a.max_steps:
                torch.save({'model': core.state_dict(), 'cfg': dict(vars(a), lv_n=a.lv_n), 'epoch': ep + 1},
                           ckdir + '%s_ep%d.pt' % (stem, ep + 1))
    if is0:
        res = dict(cfg=vars(a), steps=it, train_seconds=time.time() - t0, per_epoch=per_epoch, scales=SCALES)
        print(json.dumps(res, indent=1), flush=True)
        if not a.max_steps:
            os.makedirs('results', exist_ok=True); json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        print('RUN DONE', flush=True)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
