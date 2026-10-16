"""Redshift Transformer on the ConvNeXt tokenizer (convae.py).  Owner 2026-10-09: tokenizer = ConvNeXt only;
「之後由一個 Transformer 讀它的輸出去算 z」, 「直接交給transformer 但是要設計好用16heads 或 32 heads」,
data 300k train / 60k test.  No templates, no FFT scan: the Transformer does everything.

  ConvAE encoder -> TOK (941 x 768, 320 km/s each)  +  F40 (64 x 7,526, 40 km/s)
  token t = TOK_t + Linear(its own 8 bins of F40, 512 -> 768)     keeps the position inside a token
  [CLS] + 941 tokens -> Transformer encoder, L layers, d 768, H heads (16 or 32)
       positions: learned absolute + RoPE on the log-lambda axis (scores see wavelength RATIOS; a redshift does not
       change them).  [CLS] sits at position 0 with no rotation partner meaning; it learns where to look.
  heads
    coarse  [CLS] -> 759 coarse z bins (16 fine bins = 640 km/s each), cross-entropy
    fine    top-8 coarse bins -> 8 candidate queries -> 2 decoder blocks (cross-attention to the tokens with
            REST-FRAME RoPE: keys rotated by observed position, queries by the candidate's shift, so a score is a
            function of the token's rest wavelength; then self-attention among candidates)
            -> pick logit, and 33 offset logits (fine bins -16..+16 around the coarse centre)
    class   [CLS] -> 5 classes
  read-out  picked candidate's centre + argmax offset + parabola on the offset logits (sub-bin)
  loss      coarse CE + pick CE + offset CE (only for the candidate nearest the truth, if within 16 bins) + 0.1 class

Tokenizer: --tok_mode frozen | ft (lr x 0.1) | random (random init, full lr).
Data: original ZWARN=0 (--clean 0), class mix 20% each (data.Spectra default).  Test: the first --n_test spectra of
the validation blocks (never trained on), same filters and mix.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from grids import N_Z, S_LO, N_OBS, CLASSES, bin_to_z
from data import Spectra
from train import ddp_setup, class_ce
from model import topk_peaks
from convtok import rope
from convae import ConvAE, NPAD

CKDIR = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'
CW = 16                                     # fine bins per coarse bin
NC = math.ceil(N_Z / CW)                    # 759
NOFF = 2 * CW + 1                           # offsets -16..+16


def angles(pos, hd):
    om = 2 * np.pi / np.geomspace(16, 32768, hd // 2)
    a = torch.from_numpy(np.outer(pos, om)).float()
    return a.cos(), a.sin(), torch.from_numpy(om).float()


class EncBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, c, s):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, c.to(q.dtype), s.to(q.dtype)), rope(k, c.to(k.dtype), s.to(k.dtype))
        x = x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class DecBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.nq, self.q, self.kv, self.o1 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ns, self.qkv, self.o2 = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.nm, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mem, kc, ks, qc, qs):
        B, K, D = x.shape; H = self.h
        q = rope(self.q(self.nq(x)).view(B, K, H, D // H).transpose(1, 2), qc.unsqueeze(1).to(x.dtype), qs.unsqueeze(1).to(x.dtype))
        k, v = self.kv(mem).view(B, -1, 2, H, D // H).permute(2, 0, 3, 1, 4)
        k = rope(k, kc.to(k.dtype), ks.to(k.dtype))
        x = x + self.o1(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, K, D))
        a, b, c = self.qkv(self.ns(x)).view(B, K, 3, H, D // H).permute(2, 0, 3, 1, 4)
        x = x + self.o2(F.scaled_dot_product_attention(a, b, c).transpose(1, 2).reshape(B, K, D))
        return x + self.mlp(self.nm(x))


class ZTrans(nn.Module):
    def __init__(self, tok_cfg=None, d=768, H=16, L=12, Ldec=2, Kc=8, levels=(7, 7, 7, 5, 5, 5)):
        super().__init__()
        self.tok = ConvAE(levels=levels)
        self.Kc, self.H = Kc, H
        dt, c0 = self.tok.ch[-1], self.tok.ch[0]
        self.nt = NPAD // 8
        self.tin = nn.Linear(dt, d)
        self.fin = nn.Linear(8 * c0, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.nt + 1, d) * 0.02)
        self.enc = nn.ModuleList([EncBlock(d, H) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        hd = d // H
        u = np.concatenate([[0.0], np.arange(self.nt) * 8 + 3.5])           # token centres, 40 km/s bins
        c, s, om = angles(u, hd)
        self.register_buffer('ec', c, persistent=False); self.register_buffer('es', s, persistent=False)
        self.register_buffer('kc', c[1:].clone(), persistent=False); self.register_buffer('ks', s[1:].clone(), persistent=False)
        self.register_buffer('om', om, persistent=False)
        self.coarse = nn.Linear(d, NC)
        self.qemb = nn.Linear(2, d)
        self.dec = nn.ModuleList([DecBlock(d, H) for _ in range(Ldec)])
        self.dnorm = nn.LayerNorm(d)
        self.pick, self.offh = nn.Linear(d, 1), nn.Linear(d, NOFF)
        self.chead = nn.Linear(d, len(CLASSES))
        nn.init.zeros_(self.chead.weight); nn.init.zeros_(self.chead.bias)

    def forward(self, flux, ivar):
        F40, tok, codes, f, n, s = self.tok.encode(flux, ivar)
        B = tok.shape[0]
        fine = F.pad(F40, (0, NPAD - N_OBS)).view(B, -1, self.nt, 8).permute(0, 2, 3, 1).reshape(B, self.nt, -1)
        x = self.tin(tok) + self.fin(fine)
        x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), x], 1) + self.pos
        for b in self.enc:
            x = b(x, self.ec, self.es)
        x = self.norm(x)
        g, mem = x[:, 0], x[:, 1:]
        lc = self.coarse(g).float()                                           # (B, NC)
        _, cb = topk_peaks(lc.detach().unsqueeze(1), self.Kc, sep=2)          # (B, Kc) coarse bins
        cen = cb * CW + (CW - 1) / 2.0                                        # fine-bin centre
        qa = (cen + S_LO).float().unsqueeze(-1) * self.om                     # rest-frame RoPE angle of the candidate
        q = self.qemb(torch.stack([cen.float() / N_Z, torch.log_softmax(lc, -1).gather(1, cb).detach()], -1))
        for b in self.dec:
            q = b(q, mem, self.kc, self.ks, qa.cos(), qa.sin())
        q = self.dnorm(q)
        return dict(coarse=lc, cand_cb=cb, cand_cen=cen, pick=self.pick(q).squeeze(-1).float(),
                    off=self.offh(q).float(), logit_c=self.chead(g).float())


def losses(o, zb, c):
    lcz = F.cross_entropy(o['coarse'], zb // CW)
    d = zb.view(-1, 1).float() - o['cand_cen']                                # truth - centre, fine bins
    tgt = d.abs().argmin(1)
    lp = F.cross_entropy(o['pick'], tgt)
    dt = d.gather(1, tgt.view(-1, 1)).squeeze(1)
    ok = dt.abs() <= CW
    ot = (dt.round().long() + CW).clamp(0, NOFF - 1)
    ofl = o['off'][torch.arange(len(zb), device=zb.device), tgt]
    lo = F.cross_entropy(ofl[ok], ot[ok]) if ok.any() else ofl.sum() * 0
    return lcz, lp, lo, class_ce(o['logit_c'], c)


def readz(o):
    B = o['pick'].shape[0]
    best = o['pick'].argmax(1)
    ofl = torch.log_softmax(o['off'][torch.arange(B, device=best.device), best], -1)
    j = ofl.argmax(1)
    l0 = ofl.gather(1, j.view(-1, 1))[:, 0]
    lm = ofl.gather(1, (j - 1).clamp(0, NOFF - 1).view(-1, 1))[:, 0]
    lq = ofl.gather(1, (j + 1).clamp(0, NOFF - 1).view(-1, 1))[:, 0]
    den = lm - 2 * l0 + lq
    dd = torch.where(den < 0, 0.5 * (lm - lq) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    zbin = o['cand_cen'].gather(1, best.view(-1, 1))[:, 0] + (j - CW).float() + dd
    return bin_to_z(zbin.detach().cpu().numpy())


def selftest():
    torch.manual_seed(0)
    m = ZTrans(d=64, H=16, L=2, Ldec=1, Kc=4)
    m.tok = ConvAE(ch=(8, 16, 24, 32), nb=1); m.tin = nn.Linear(32, 64); m.fin = nn.Linear(64, 64)
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    o = m(fl, iv)
    print('forward:', {k: tuple(v.shape) for k, v in o.items()})
    zb = torch.tensor([3000, 6000]); c = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float)
    L = losses(o, zb, c); sum(L).backward()
    print('init losses coarse %.3f (ln %d %.3f) pick %.3f (ln 4 %.3f) offset %.3f (ln 33 %.3f) class %.3f'
          % (L[0], NC, math.log(NC), L[1], math.log(4), L[2], math.log(NOFF), L[3]))
    print('params without grad:', [k for k, p in m.named_parameters() if p.grad is None])
    print('readz:', readz(o))
    for H in (16, 32):
        print('full size H=%d: head dim %d, params %.2fM (tokenizer %.2fM)'
              % (H, 768 // H, sum(p.numel() for p in ZTrans(H=H).parameters()) / 1e6, sum(p.numel() for p in ConvAE().parameters()) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tok', default=''); ap.add_argument('--tok_mode', choices=['frozen', 'ft', 'random'], default='ft')
    ap.add_argument('--H', type=int, default=16); ap.add_argument('--L', type=int, default=12)
    ap.add_argument('--size', type=int, default=300000); ap.add_argument('--n_test', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=8); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clean', type=int, default=-1); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--tag', default='ZT')
    ap.add_argument('--max_steps', type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tck = torch.load(a.tok, map_location='cpu', weights_only=False)
    levels = tuple(int(x) for x in tck['cfg'].get('levels', '7,7,7,5,5,5').split(','))
    core = ZTrans(H=a.H, L=a.L, levels=levels).to(dev)
    if a.tok_mode != 'random':
        core.tok.load_state_dict(tck['model'])
    if a.tok_mode == 'frozen':
        for p in core.tok.parameters():
            p.requires_grad_(False)
    net = DDP(core, device_ids=[local], find_unused_parameters=True) if world > 1 else core
    tid = {id(p) for p in core.tok.parameters()}
    tokp = [p for p in core.tok.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in core.parameters() if id(p) not in tid], lr=a.lr)]
    if tokp:
        groups.append(dict(params=tokp, lr=a.lr * (0.1 if a.tok_mode == 'ft' else 1.0)))
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01)
    gbs = a.bs * world; steps = max(a.size // gbs, 1)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g_['lr'] for g_ in groups], total_steps=steps, pct_start=0.05)
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, clean=0, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True)
    stem = '%s_%s_H%d_L%d_%d_s%d_cl0_mix' % (a.tag, a.tok_mode, a.H, a.L, a.size, a.seed)
    np_ = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    if is0:
        print('ZT %s | tokenizer %s (%s, %.2fM) | transformer %.2fM: L %d, d 768, %d heads (dim %d) | train %d steps %d gbs %d '
              '| test %d | data: original ZWARN=0, no cleaning, class mix 20%% each'
              % (stem, os.path.basename(a.tok) or '-', a.tok_mode, np_(core.tok), np_(core) - np_(core.tok), a.L, a.H, 768 // a.H,
                 a.size, steps, gbs, a.n_test), flush=True)

    def stream():
        while True:
            for b_ in tr:
                yield b_
            print('  [rank %d] loader pass ended at step %d' % (rank, it), flush=True)
    it = 0; run = []; t0 = time.time(); net.train(); sb = stream()
    if a.tok_mode == 'frozen':
        core.tok.eval()
    while it < steps:
        fl, iv, zb, c, z = next(sb)
        fl, iv, zb, c = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True), zb.to(dev), c.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(fl, iv)
            L_ = losses(o, zb, c)
            loss = L_[0] + L_[1] + L_[2] + 0.1 * L_[3]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0); opt.step(); sched.step()
        run.append([float(v) for v in L_]); it += 1
        if not all(math.isfinite(v) for v in run[-1]):
            raise RuntimeError('non-finite loss at step %d: %s' % (it, run[-1]))
        if is0 and it == 1:
            print('  init  coarse %.3f (ln %d %.3f)  pick %.3f (ln 8 %.3f)  offset %.3f (ln 33 %.3f)  class %.3f (ln 5 %.3f)'
                  % (run[0][0], NC, math.log(NC), run[0][1], math.log(8), run[0][2], math.log(NOFF), run[0][3], math.log(5)), flush=True)
        if is0 and (it % 500 == 0 or it == steps):
            m_ = np.mean(run[-500:], 0); el = time.time() - t0
            print('  step %6d/%d  coarse %.4f  pick %.4f  offset %.4f  class %.4f  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                  % (it, steps, *m_, it * gbs / el, torch.cuda.max_memory_allocated() / 1e9, el, el / it * (steps - it)), flush=True)
    core.eval()
    va = DataLoader(Spectra(1 << 22, seed=0, val=True, rank=rank, world=world, clean=0, native=True),
                    batch_size=32, num_workers=8, pin_memory=True, drop_last=True)
    ZT, ZP, CL, CP, n = [], [], [], [], 0
    with torch.no_grad():
        for fl, iv, zb, c, z in va:
            if n >= a.n_test // world:
                break
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = core(fl.to(dev), iv.to(dev))
            ZT.append(z.numpy()); ZP.append(readz(o)); CL.append(c.numpy()); CP.append(o['logit_c'].argmax(1).cpu().numpy())
            n += len(z)
    loc = tuple(np.concatenate(v) for v in (ZT, ZP, CL, CP))
    if world > 1:
        buf = [None] * world; dist.all_gather_object(buf, loc)
        loc = tuple(np.concatenate([b_[i] for b_ in buf]) for i in range(4))
    if is0:
        zt, zp, cl, cp = loc
        d = (zp - zt) / (1 + zt); bad = np.abs(d) > 0.1
        nm = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        prio = np.full(len(cl), -1)
        for name in ['STAR', 'BGS', 'ELG', 'LRG', 'QSO']:
            prio[cl[:, CLASSES.index(name)] > 0] = CLASSES.index(name)
        res = dict(stem=stem, cfg=vars(a), n_test=len(zt), catastrophic=float(bad.mean()), n_catastrophic=int(bad.sum()),
                   nmad=float(nm(d)), within_0p003=float((np.abs(d) < 0.003).mean()),
                   class_any_bit=float(cl[np.arange(len(cp)), cp].mean()), seconds=time.time() - t0,
                   per_class={c_: dict(n=int((prio == i).sum()), catastrophic=float(bad[prio == i].mean()),
                                       n_catastrophic=int(bad[prio == i].sum()), nmad=float(nm(d[prio == i])))
                              for i, c_ in enumerate(CLASSES)})
        print(json.dumps(res, indent=1), flush=True)
        os.makedirs('results', exist_ok=True); json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        np.savez('results/%s_test.npz' % stem, zt=zt, zp=zp, cls=cl, cls_pred=cp)
        if not a.max_steps:
            torch.save({'model': core.state_dict(), 'cfg': vars(a), 'res': res}, CKDIR + stem + '.pt')
        print('RUN DONE', flush=True)
    if world > 1:
        dist.barrier()
    os._exit(0)


if __name__ == '__main__':
    main()
