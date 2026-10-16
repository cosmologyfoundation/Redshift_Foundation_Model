"""Redshift Transformer reading EVERY output of the dual-scale ConvNeXt tokenizer (dualtok.py).  Owner 2026-10-09:
「下一步是修transformer 要用到所有tokenizer 出來的值來設計」.  Tokenizer frozen (owner: 不微調).

  tokenizer outputs used                          -> in the Transformer
    narrow tokens 1,882 x 384 (160 km/s)            Linear -> 768, + type embedding 'narrow', + learned absolute position
    broad tokens    118 x 768 (2,560 km/s)          Linear -> 768, + type embedding 'broad',  + learned absolute position
    token positions (40 km/s bins)                  RoPE angles for every token (narrow and broad share one axis, so
                                                    attention between a narrow line and a broad line sees their
                                                    wavelength ratio)
    brightness log10 s                              Linear -> 768, added to the [CLS] token
  sequence = [CLS] + 118 broad + 1,882 narrow = 2,001 tokens; encoder L layers, d 768, H heads (default 32 = the better
  of the 16/32 test on the previous tokenizer)
  heads (as ztrans.py): [CLS] -> 759 coarse z bins; top-8 coarse -> 2 decoder blocks with REST-FRAME RoPE over all
  2,000 tokens -> pick + 33 fine offsets (+ parabola); [CLS] -> class.  RoPE periods 8 .. 32,768 bins (narrow tokens are
  4 bins apart, so periods down to 8 bins are resolvable; the old 16-bin floor could not place a line inside 320 km/s).
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
from dualtok import DualTok, NN, NB
import muonopt

CKDIR = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'
CW = 16                                     # fine bins per coarse bin
NC = math.ceil(N_Z / CW)                    # 759
NOFF = 2 * CW + 1                           # offsets -16..+16


def angles(pos, hd):
    om = 2 * np.pi / np.geomspace(8, 32768, hd // 2)
    a = torch.from_numpy(np.outer(pos, om)).float()
    return a.cos(), a.sin(), torch.from_numpy(om).float()


class SwiGLU(nn.Module):
    """down(silu(a) * b), [a, b] = up(x); hidden 8d/3 so the parameters equal the 4d GELU MLP (2 x d x 4d = 3 x d x 8d/3)"""
    def __init__(self, d):
        super().__init__()
        hid = (8 * d + 2) // 3
        self.up, self.down = nn.Linear(d, 2 * hid), nn.Linear(hid, d)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


def mlp(d, swiglu):
    return SwiGLU(d) if swiglu else nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class EncBlock(nn.Module):
    def __init__(self, d, h, swiglu=False):
        super().__init__()
        self.h = h
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), mlp(d, swiglu)

    def forward(self, x, c, s):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, c.to(q.dtype), s.to(q.dtype)), rope(k, c.to(k.dtype), s.to(k.dtype))
        x = x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class DecBlock(nn.Module):
    def __init__(self, d, h, swiglu=False):
        super().__init__()
        self.h = h
        self.nq, self.q, self.kv, self.o1 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ns, self.qkv, self.o2 = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.nm, self.mlp = nn.LayerNorm(d), mlp(d, swiglu)

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
    def __init__(self, d=768, H=32, L=12, Ldec=2, Kc=8, lv_n=(5, 5), tok_swiglu=False, swiglu=False):
        super().__init__()
        self.tok = DualTok(lv_n=lv_n, swiglu=tok_swiglu)
        self.Kc, self.H = Kc, H
        self.n_in, self.b_in, self.s_in = nn.Linear(384, d), nn.Linear(768, d), nn.Linear(1, d)
        self.typ = nn.Parameter(torch.randn(2, d) * 0.02)                   # 0 broad, 1 narrow
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 1 + NB + NN, d) * 0.02)
        self.enc = nn.ModuleList([EncBlock(d, H, swiglu) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        hd = d // H
        u = np.concatenate([[0.0], np.arange(NB) * 64 + 31.5, np.arange(NN) * 4 + 1.5])   # positions, 40 km/s bins
        c, s_, om = angles(u, hd)
        self.register_buffer('ec', c, persistent=False); self.register_buffer('es', s_, persistent=False)
        self.register_buffer('kc', c[1:].clone(), persistent=False); self.register_buffer('ks', s_[1:].clone(), persistent=False)
        self.register_buffer('om', om, persistent=False)
        self.coarse = nn.Linear(d, NC)
        self.qemb = nn.Linear(2, d)
        self.dec = nn.ModuleList([DecBlock(d, H, swiglu) for _ in range(Ldec)])
        self.dnorm = nn.LayerNorm(d)
        self.pick, self.offh = nn.Linear(d, 1), nn.Linear(d, NOFF)
        self.chead = nn.Linear(d, len(CLASSES))
        nn.init.zeros_(self.chead.weight); nn.init.zeros_(self.chead.bias)

    def forward(self, flux, ivar):
        with torch.no_grad() if not any(p.requires_grad for p in self.tok.parameters()) else torch.enable_grad():
            hn, hb, cn, cb, f, n, s = self.tok.encode(flux, ivar)
        B = hn.shape[0]
        c0 = self.cls.expand(B, -1, -1) + self.s_in(torch.log10(s).view(B, 1, 1).to(hn.dtype))
        x = torch.cat([c0.to(hn.dtype), self.b_in(hb) + self.typ[0], self.n_in(hn) + self.typ[1]], 1) + self.pos
        for b in self.enc:
            x = b(x, self.ec, self.es)
        x = self.norm(x)
        g, mem = x[:, 0], x[:, 1:]
        lc = self.coarse(g).float()
        _, cb_ = topk_peaks(lc.detach().unsqueeze(1), self.Kc, sep=2)
        cen = cb_ * CW + (CW - 1) / 2.0
        qa = (cen + S_LO).float().unsqueeze(-1) * self.om
        q = self.qemb(torch.stack([cen.float() / N_Z, torch.log_softmax(lc, -1).gather(1, cb_).detach()], -1))
        for b in self.dec:
            q = b(q, mem, self.kc, self.ks, qa.cos(), qa.sin())
        q = self.dnorm(q)
        return dict(coarse=lc, cand_cb=cb_, cand_cen=cen, pick=self.pick(q).squeeze(-1).float(),
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
    m = ZTrans(d=64, H=32, L=1, Ldec=1, Kc=4)
    for p in m.tok.parameters():
        p.requires_grad_(False)
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    o = m(fl, iv)
    print('forward:', {k: tuple(v.shape) for k, v in o.items()})
    zb = torch.tensor([3000, 6000]); c = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float)
    L = losses(o, zb, c); sum(L).backward()
    print('init losses coarse %.3f (ln %d %.3f) pick %.3f offset %.3f class %.3f' % (L[0], NC, math.log(NC), L[1], L[2], L[3]))
    print('trainable params without grad:', [k for k, p in m.named_parameters() if p.requires_grad and p.grad is None])
    print('sequence length', 1 + NB + NN, '| full size H=32: %.2fM (tokenizer %.2fM)'
          % (sum(p.numel() for p in ZTrans().parameters()) / 1e6, sum(p.numel() for p in DualTok().parameters()) / 1e6))


def per_class_cat(d, cl):
    '''catastrophic |dz/(1+z)|>0.1 overall and per priority class (QSO>LRG>ELG>BGS>STAR), as a dict and a one-line string'''
    bad = np.abs(d) > 0.1
    prio = np.full(len(cl), -1)
    for name in ['STAR', 'BGS', 'ELG', 'LRG', 'QSO']:
        prio[cl[:, CLASSES.index(name)] > 0] = CLASSES.index(name)
    pc = {c_: dict(n=int((prio == i).sum()), n_catastrophic=int(bad[prio == i].sum()),
                   catastrophic=float(bad[prio == i].mean()) if (prio == i).any() else float('nan'))
          for i, c_ in enumerate(CLASSES)}
    txt = 'all %.3f%% (%d/%d) | ' % (100 * bad.mean(), bad.sum(), len(bad)) + '  '.join(
        '%s %.3f%% (%d/%d)' % (c_, 100 * v['catastrophic'], v['n_catastrophic'], v['n']) for c_, v in pc.items())
    return pc, txt


def gather_eval(core, loader, n_max, world, dev):
    """-> (zt, zp, cls, cls_pred, sum of z losses) over all ranks"""
    ZT, ZP, CL, CP, LS, n = [], [], [], [], [], 0
    with torch.no_grad():
        for fl, iv, zb, c, z in loader:
            if n >= n_max // world:
                break
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = core(fl.to(dev), iv.to(dev))
                L_ = losses(o, zb.to(dev), c.to(dev))
            ZT.append(z.numpy()); ZP.append(readz(o)); CL.append(c.numpy()); CP.append(o['logit_c'].argmax(1).cpu().numpy())
            LS.append(float(L_[0] + L_[1] + L_[2]) * len(z)); n += len(z)
    loc = tuple(np.concatenate(v) for v in (ZT, ZP, CL, CP)) + (np.array([sum(LS), n]),)
    if world > 1:
        buf = [None] * world; dist.all_gather_object(buf, loc)
        loc = tuple(np.concatenate([b_[i] for b_ in buf]) for i in range(4)) + (sum(b_[4] for b_ in buf),)
    return loc


nmad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tok', default=''); ap.add_argument('--tok_mode', choices=['frozen', 'ft', 'random'], default='frozen')
    ap.add_argument('--H', type=int, default=32); ap.add_argument('--L', type=int, default=12)
    ap.add_argument('--size', type=int, default=300000); ap.add_argument('--n_test', type=int, default=60000)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--eval_resumed', action='store_true', help='on resume, first print held-out catastrophic per type for the resumed weights')
    ap.add_argument('--eval_ckpts', default='', help='comma list of extra checkpoints to evaluate the same way on resume')
    ap.add_argument('--stop_epoch', type=int, default=0, help='end training after this epoch (schedule and stem still use --epochs); 0 = --epochs')
    ap.add_argument('--patience', type=int, default=2, help='stop after this many epochs with neither the early-stop loss nor NMAD improving')
    ap.add_argument('--n_es', type=int, default=8192, help='early-stop set: validation blocks 400-1300, disjoint from the test blocks')
    ap.add_argument('--resume', default='', help='state file written at the end of every epoch')
    ap.add_argument('--bs', type=int, default=8); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clean', type=int, default=-1); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--tag', default='ZT2')
    ap.add_argument('--mix_train', default='', help='training class shares, e.g. QSO:0.35,ELG:0.35,LRG:0.1,BGS:0.1,STAR:0.1 (held-out and test keep 20%% each)')
    ap.add_argument('--init', default='', help='start the whole model from this checkpoint (fine-tune); a --resume state wins over it')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--swiglu', action='store_true', help='SwiGLU MLPs (parameter-matched) in the encoder, decoder and line-sentence blocks')
    ap.add_argument('--muon', action='store_true', help='Muon on the hidden 2-D matrices (muonopt.TRANSFORMER_HIDDEN), AdamW on the rest')
    ap.add_argument('--muon_lr', type=float, default=None, help='Muon peak lr (match_rms_adamw); default --lr')
    ap.add_argument('--val_blocks', type=int, default=1300, help='main validation blocks held out of training: 0-T test, T-N early stop')
    ap.add_argument('--test_blocks', type=int, default=400, help='T: the test pool is validation blocks 0-T')
    ap.add_argument('--reset_best', action='store_true', help='after --resume: re-evaluate on the early-stop set and restart best/stall from it')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.clean == 0, 'owner rule 2026-10-09: no cleaning, pass --clean 0'
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tck = torch.load(a.tok, map_location='cpu', weights_only=False)
    core = ZTrans(H=a.H, L=a.L, lv_n=tuple(int(x) for x in tck['cfg'].get('lv_n', '5,5,5').split(',')),
                  tok_swiglu=bool(tck['cfg'].get('swiglu', False)), swiglu=a.swiglu).to(dev)
    if a.tok_mode != 'random':
        core.tok.load_state_dict(tck['model'])
    if a.init:
        core.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['model'])
        print('init from', a.init, flush=True) if rank == 0 else None
    if a.tok_mode == 'frozen':
        for p in core.tok.parameters():
            p.requires_grad_(False)
    if a.compile:
        # compile the two heavy parts as FUNCTIONS (not modules), so state_dict keys stay the same and checkpoints
        # resume across compiled / uncompiled runs: the frozen tokenizer's encoder and each Transformer block
        core.tok.encode = torch.compile(core.tok.encode)
        for b in core.enc:
            b.forward = torch.compile(b.forward)
    net = DDP(core, device_ids=[local], find_unused_parameters=True) if world > 1 else core
    tid = {id(p) for p in core.tok.parameters()}
    tokp = [p for p in core.tok.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in core.parameters() if id(p) not in tid], lr=a.lr)]
    if tokp:
        groups.append(dict(params=tokp, lr=a.lr * (0.1 if a.tok_mode == 'ft' else 1.0)))
    mu = []
    if a.muon:
        mu, aw, mu_names = muonopt.split(((n, p) for n, p in core.named_parameters() if id(p) not in tid), muonopt.TRANSFORMER_HIDDEN)
        groups[0]['params'] = aw
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01)
    gbs = a.bs * world; spe = max(a.size // gbs, 1); steps = spe * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g_['lr'] for g_ in groups], total_steps=steps, pct_start=0.05)
    if mu:
        om = muonopt.muon(mu, a.muon_lr or a.lr)
        sm = torch.optim.lr_scheduler.OneCycleLR(om, a.muon_lr or a.lr, total_steps=steps, pct_start=0.05, cycle_momentum=False)
        opt, sched = muonopt.MultiOpt([opt, om]), muonopt.MultiSched([sched, sm])
        if rank == 0:
            print('Muon on %d matrices (%.2fM params, lr %g), AdamW on the rest (%.2fM)'
                  % (len(mu), sum(p.numel() for p in mu) / 1e6, a.muon_lr or a.lr, sum(p.numel() for p in aw) / 1e6), flush=True)
    # training excludes the first 1300 main validation blocks: blocks 0-400 = test, 400-1300 = early-stop set
    mixw = {k: float(v) for k, v in (x.split(':') for x in a.mix_train.split(','))} if a.mix_train else True
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, clean=0, native=True, val_blocks_main=a.val_blocks, mix=mixw),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True)
    es = DataLoader(Spectra(1 << 22, seed=0, val=True, rank=rank, world=world, clean=0, native=True, val_blocks_main=a.val_blocks,
                            val_from_main=a.test_blocks), batch_size=32, num_workers=8, pin_memory=True, drop_last=True)
    stem = '%s_%s_H%d_L%d_%d_e%d_gbs%d_lr%g_tb%d_vb%d_s%d_cl0_mix' % (a.tag, a.tok_mode, a.H, a.L, a.size, a.epochs, gbs, a.lr, a.test_blocks, a.val_blocks, a.seed)
    np_ = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    ep0, best, stall, hist = 0, dict(loss=1e9, nmad=1e9, epoch=0), 0, []
    if a.resume and os.path.exists(a.resume):
        st = torch.load(a.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        ep0, best, stall, hist = st['epoch'], st['best'], st['stall'], st['hist']
    state_fn = CKDIR + stem + '_state.pt'; best_fn = CKDIR + stem + '_best.pt'
    if is0:
        print('ZT2 %s | dual-scale tokenizer %s (%s, %.2fM) | transformer %.2fM: L %d, d 768, %d heads (dim %d) | train %d x up to %d '
              'epochs, %d steps/epoch, gbs %d | early stop: patience %d on loss AND NMAD over %d held-out spectra (blocks %d-%d) | '
              'test %d (blocks 0-%d) | start epoch %d | data: original ZWARN=0, no cleaning, class mix 20%% each'
              % (stem, os.path.basename(a.tok), a.tok_mode, np_(core.tok), np_(core) - np_(core.tok), a.L, a.H, 768 // a.H,
                 a.size, a.epochs, spe, gbs, a.patience, a.n_es, a.test_blocks, a.val_blocks, a.n_test, a.test_blocks, ep0 + 1), flush=True)

    for f_ in ([state_fn] if a.eval_resumed and ep0 > 0 else []) + [x for x in a.eval_ckpts.split(',') if x]:
        # owner 2026-10-10: catastrophic overall AND per type, also for epochs finished before this was added
        m_ = {k: v.clone() for k, v in core.state_dict().items()}; ck = torch.load(f_, map_location='cpu', weights_only=False)
        core.load_state_dict(ck['model']); core.eval()
        zt, zp, cl, cp, ls = gather_eval(core, es, a.n_es, world, dev)
        if is0:
            d = (zp - zt) / (1 + zt)
            print('  EPOCH %d (re-evaluated, %s) held-out loss %.4f  NMAD %.3e | catastrophic by type: %s'
                  % (ck['epoch'], os.path.basename(f_), float(ls[0] / ls[1]), nmad(d), per_class_cat(d, cl)[1]), flush=True)
        core.load_state_dict(m_)
    if a.reset_best:
        # owner 2026-10-10: keep the resumed weights, but judge "best" on the NEW (larger) early-stop set from here on
        core.eval()
        zt, zp, cl, cp, ls = gather_eval(core, es, a.n_es, world, dev)
        d = (zp - zt) / (1 + zt)
        best = dict(loss=float(ls[0] / ls[1]), nmad=float(nmad(d)), epoch=ep0); stall = 0
        if is0:
            torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep0}, best_fn)
            print('  RESET BEST on the new early-stop set (%d spectra): epoch %d loss %.4f  NMAD %.3e  catastrophic %.3f%%'
                  % (len(zt), ep0, best['loss'], best['nmad'], 100 * float((np.abs(d) > 0.1).mean())), flush=True)

    def stream():
        while True:
            for b_ in tr:
                yield b_
    t0 = time.time(); sb = stream()
    for ep in range(ep0, min(a.epochs, a.stop_epoch or a.epochs)):
        net.train()
        if a.tok_mode == 'frozen':
            core.tok.eval()
        run = []
        for it in range(spe):
            fl, iv, zb, c, z = next(sb)
            fl, iv, zb, c = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True), zb.to(dev), c.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(fl, iv)
                L_ = losses(o, zb, c)
                loss = L_[0] + L_[1] + L_[2] + 0.1 * L_[3]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0); opt.step(); sched.step()
            run.append([float(v) for v in L_])
            if not all(math.isfinite(v) for v in run[-1]):
                raise RuntimeError('non-finite loss at epoch %d step %d: %s' % (ep + 1, it, run[-1]))
            if is0 and (it + 1) % 2000 == 0:
                m_ = np.mean(run[-2000:], 0); el = time.time() - t0
                print('  ep %d step %5d/%d  coarse %.4f  pick %.4f  offset %.4f  class %.4f  elapsed %.0fs'
                      % (ep + 1, it + 1, spe, *m_, el), flush=True)
        core.eval()
        zt, zp, cl, cp, ls = gather_eval(core, es, a.n_es, world, dev)
        d = (zp - zt) / (1 + zt)
        e_loss, e_nmad, e_cat = float(ls[0] / ls[1]), float(nmad(d)), float((np.abs(d) > 0.1).mean())
        imp = []
        if e_loss < best['loss'] - 1e-3:
            imp.append('loss')
        if e_nmad < best['nmad'] * 0.99:
            imp.append('NMAD')
        if imp:
            stall = 0
            best = dict(loss=min(best['loss'], e_loss), nmad=min(best['nmad'], e_nmad), epoch=ep + 1)
            if is0:
                torch.save({'model': core.state_dict(), 'cfg': vars(a), 'epoch': ep + 1}, best_fn)
        else:
            stall += 1
        hist.append(dict(epoch=ep + 1, train=np.mean(run, 0).tolist(), es_loss=e_loss, es_nmad=e_nmad, es_catastrophic=e_cat, es_per_class=per_class_cat(d, cl)[0],
                         improved=imp, seconds=time.time() - t0))
        if is0:
            print('  EPOCH %d  train coarse %.4f pick %.4f offset %.4f | early-stop set: loss %.4f  NMAD %.3e  catastrophic %.3f%% | '
                  'improved: %s | stall %d/%d' % (ep + 1, *np.mean(run, 0)[:3], e_loss, e_nmad, 100 * e_cat,
                                                 ','.join(imp) or 'none', stall, a.patience), flush=True)
            print('  EPOCH %d held-out catastrophic by type: %s' % (ep + 1, per_class_cat(d, cl)[1]), flush=True)
            torch.save({'model': core.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep + 1,
                        'best': best, 'stall': stall, 'hist': hist}, state_fn)
        if stall >= a.patience:
            if is0:
                print('  EARLY STOP after epoch %d: neither loss nor NMAD improved for %d epochs; best epoch %d'
                      % (ep + 1, a.patience, best['epoch']), flush=True)
            break
    # test with the best early-stop epoch
    if world > 1:
        dist.barrier()
    core.load_state_dict(torch.load(best_fn, map_location='cpu', weights_only=False)['model'])
    core.eval()
    va = DataLoader(Spectra(1 << 22, seed=0, val=True, rank=rank, world=world, clean=0, native=True, val_blocks_main=a.test_blocks),
                    batch_size=32, num_workers=8, pin_memory=True, drop_last=True)
    zt, zp, cl, cp, _ = gather_eval(core, va, a.n_test, world, dev)
    if is0:
        d = (zp - zt) / (1 + zt); bad = np.abs(d) > 0.1
        prio = np.full(len(cl), -1)
        for name in ['STAR', 'BGS', 'ELG', 'LRG', 'QSO']:
            prio[cl[:, CLASSES.index(name)] > 0] = CLASSES.index(name)
        res = dict(stem=stem, cfg=vars(a), best_epoch=best['epoch'], history=hist, n_test=len(zt),
                   catastrophic=float(bad.mean()), n_catastrophic=int(bad.sum()), nmad=float(nmad(d)),
                   within_0p003=float((np.abs(d) < 0.003).mean()), class_any_bit=float(cl[np.arange(len(cp)), cp].mean()),
                   seconds=time.time() - t0,
                   per_class={c_: dict(n=int((prio == i).sum()), catastrophic=float(bad[prio == i].mean()),
                                       n_catastrophic=int(bad[prio == i].sum()), nmad=float(nmad(d[prio == i])))
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
