"""zfm with the native foundation model as its tokenizer (FM_DESIGN.md 8; owner 2026-10-09:
「依照這個 model 來設計下游 redshift 模型 ... tokenizer 要用這個 foundation model」, 「40km/s 也要先融入」).

  BRZ_FLUX, BRZ_IVAR (7781 px native, raw)
    -> FM: SpecFMNative (specfm_native.py), nothing hidden                 (B, 1+n_patch, 768)
    -> un-patchify: Linear(768 -> p x c_up) per token                      (B, c_up, 7781)   native grid
    -> resample native -> 40 km/s log-lambda (fixed 2-tap linear map)      (B, c_up, 7526)
    -> n_adapt ConvNeXt1D blocks ON THE 40 km/s GRID, then 1x1 -> c_corr  (B, 8, 7526)
    -> zfm Surface (40 archetypes, FFT over 12,140 z bins) -> top-K -> Reranker (+ evidence)
    -> class head on the FM patch states (mean, max) + surface summary

Where 40 km/s enters.  On the log grid a redshift is a pure translation.  The FM itself does not need it
(learned positions, raw pixels, no interpolated noise), so the grid is introduced AFTER the FM, on learned
features: the adapter convolutions are then shift- (= redshift-) equivariant, and the Surface's FFT scan works
exactly as in zfm.  The resample interpolates features, not data, so the noise-correlation problem of
interpolating the input (variance x0.665) does not arise.  Evidence features are the one exception: they are
hand-made from flux*sqrt(ivar) on the log grid and are computed from the native arrays with the same linear
interpolation data.py used (np.interp == LogResample, checked in --selftest).

Two heads, to be compared (owner 2026-10-09: 「下游先做兩個 一個展開一個不展開 分開設計 要做比對」):
  --head unpatch  ZFMN above: tokens are un-patchified back to pixels, then zfm's FFT surface.
  --head token    ZTOK: the tokens are read directly, never un-patchified.  RestAttn = M learned queries that
                  cross-attend to the 973 patch tokens with RoPE on the 40 km/s log-lambda axis: keys rotated by
                  each token's position u_t, queries by the trial redshift's shift, so every attention score is a
                  function of u_t - shift = the token's REST-frame wavelength (in 40 km/s bins).  Redshift-
                  equivariant like the FFT surface, at token resolution.
                    coarse  RestAttn scored at every 16th z bin (759 coarse bins, 640 km/s)  -> coarse log P(z)
                    top-K   K coarse peaks (sep 2 coarse bins)
                    fine    second RestAttn at each candidate -> 256-d vector -> offset (fine bins, +-16) ;
                            evidence at center+offset ; Transformer across candidates -> rank logit
                    read    z = center + offset of the top-ranked candidate (continuous)
                  loss = coarse z NLL + 0.1 class + 1.0 rank + 0.1 offset (smooth L1, candidates within 16 bins)

Arms (FM_DESIGN 8.3): --fm CKPT (fine-tuned, FM lr = --fm_lr_mult x lr), --fm CKPT --freeze, --fm_random.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from grids import N_OBS, N_Z, S_LO, DLOG, CLASSES, LAM_OBS, LAM_NATIVE, OBS_LO, bin_to_z
from data import Spectra
from model import Surface, Reranker, ConvNeXt1D, topk_peaks, n_params
from specfm_native import SpecFMNative, N_NATIVE
from train import ddp_setup, class_ce, mix_nll
import evidence as EV

CKDIR = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt/'


class LogResample(nn.Module):
    """native 0.8 A linear grid (7781) -> 40 km/s log-lambda grid (N_OBS = 7526), 2-tap linear interpolation.
    Identical to np.interp(LAM_OBS, LAM_NATIVE, .) including its end clamping."""

    def __init__(self):
        super().__init__()
        pos = np.clip((LAM_OBS - LAM_NATIVE[0]) / (LAM_NATIVE[1] - LAM_NATIVE[0]), 0, N_NATIVE - 1)
        i0 = np.minimum(np.floor(pos).astype(np.int64), N_NATIVE - 2)
        self.register_buffer('i0', torch.from_numpy(i0), persistent=False)
        self.register_buffer('w', torch.from_numpy((pos - i0).astype(np.float32)), persistent=False)

    def forward(self, e):
        return e[..., self.i0] * (1 - self.w) + e[..., self.i0 + 1] * self.w


class ZFMN(nn.Module):
    def __init__(self, fm_cfg, c_up=64, n_adapt=2, c_corr=8, n_per=8, d_rr=320, rr_layers=8, K=256, sep=64,
                 norm_corr=True, rr_final_norm=True, evidence=True):
        super().__init__()
        self.K, self.sep, self.evidence = K, sep, evidence
        self.fm = SpecFMNative(fm_cfg['d'], fm_cfg['L'], fm_cfg['p'], max(1, fm_cfg['d'] // 64),
                               fm_cfg.get('stem_c', 0), fm_cfg.get('stem_n', 0))
        d, p = fm_cfg['d'], fm_cfg['p']
        self.p, self.c_up = p, c_up
        self.unpatch = nn.Linear(d, p * c_up)
        self.resamp = LogResample()
        self.adapt = nn.ModuleList([ConvNeXt1D(c_up) for _ in range(n_adapt)])
        self.proj = nn.Conv1d(c_up, c_corr, 1)
        self.surf = Surface(n_per, c_corr, norm_corr=norm_corr)
        self.rr = Reranker(d=d_rr, layers=rr_layers, final_norm=rr_final_norm, n_ev=EV.N_FEAT if evidence else 0)
        nc = len(CLASSES)
        n = 2 * d + 2 * nc
        self.chead = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 256), nn.GELU(), nn.Linear(256, nc))
        nn.init.zeros_(self.chead[-1].weight); nn.init.zeros_(self.chead[-1].bias)

    def tokens(self, flux, ivar):
        fm = self.fm
        x, f, n, s = fm.prep(flux, ivar)
        h = fm.encode(x, s, torch.zeros(x.shape[0], fm.np_, dtype=torch.bool, device=x.device))
        return h[:, 1:]                                                     # patch states (B, n_patch, d)

    def logx(self, flux, ivar):
        """the log-grid [flux*sqrt(ivar), sqrt(ivar)] data.py used to build -- for evidence.py only"""
        F_, V_ = self.resamp(flux.float()), self.resamp(ivar.float().clamp_min(0))
        return torch.stack([F_ * V_.sqrt(), V_.sqrt()], 1)

    def forward(self, flux, ivar):
        h = self.tokens(flux, ivar)                                         # (B, n, d)
        B, n, _ = h.shape
        e = self.unpatch(h).view(B, n, self.p, self.c_up).permute(0, 3, 1, 2).reshape(B, self.c_up, n * self.p)
        e = self.resamp(e[..., :N_NATIVE])                                  # (B, c_up, N_OBS) at 40 km/s
        for b in self.adapt:
            e = b(e)
        S = self.surf(self.proj(e))                                         # (B, n_class, N_Z)
        logp_z = torch.log_softmax(torch.logsumexp(S, 1), -1)
        Sd = S.detach().float()
        hf = h.float()
        logit_c = self.chead(torch.cat([hf.mean(1), hf.amax(1), Sd.amax(2), torch.logsumexp(Sd, 2)], 1))
        cls, zb = topk_peaks(S.detach(), self.K, sep=self.sep)
        ev = EV.candidate_features(EV.spectrum_maps(self.logx(flux, ivar)), zb) if self.evidence else None
        logit = self.rr(S, cls, zb, ev)
        return dict(S=S, logp_z=logp_z, logit_c=logit_c, cand_class=cls, cand_z=zb, cand_logit=logit)

    def losses(self, o, zb, c):
        lz = F.nll_loss(o['logp_z'].float(), zb)
        tgt = (o['cand_z'] - zb.view(-1, 1)).abs().argmin(1)
        return dict(z=lz, rank=F.cross_entropy(o['cand_logit'].float(), tgt), off=torch.zeros((), device=zb.device))

    def readz(self, o):
        return subbin_z(o)


def token_u(p):
    """each patch token's centre on the 40 km/s log-lambda axis, in bins from OBS_LO (partial last patch: its own px)"""
    n = math.ceil(N_NATIVE / p)
    lam = np.array([LAM_NATIVE[i * p:min((i + 1) * p, N_NATIVE)].mean() for i in range(n)])
    return np.log(lam / OBS_LO) / DLOG


class RestAttn(nn.Module):
    """M learned queries read the tokens in the rest frame of a trial shift (RoPE on the log-lambda axis).
    <R(a) q, R(c) k> = <q, R(c - a) k>: with keys at c = u_t and queries at a = shift, the score depends only on
    u_t - shift, the token's rest-frame position.  Shift the spectrum and the trial shift by the same number of bins
    and the output is unchanged (selftest).  Periods 4 .. 32768 bins (160 km/s .. the whole rest grid)."""

    def __init__(self, d_tok, M=16, dk=64, dv=64, d_out=256):
        super().__init__()
        self.q = nn.Parameter(torch.randn(M, dk))
        self.k, self.v = nn.Linear(d_tok, dk), nn.Linear(d_tok, dv)
        self.register_buffer('om', torch.from_numpy(2 * np.pi / np.geomspace(4, 32768, dk // 2)).float(), persistent=False)
        self.dk = dk
        self.mlp = nn.Sequential(nn.LayerNorm(M * dv), nn.Linear(M * dv, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    @staticmethod
    def rot(x, ang):
        c, s = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, h, u, shift):
        """h (B,T,d) tokens, u (T,) token positions [bins], shift (B,Z) trial shifts [bins] -> (B,Z,d_out)"""
        with torch.autocast('cuda', enabled=False):
            k = self.rot(self.k(h.to(self.k.weight.dtype)), u.view(1, -1, 1) * self.om)                       # (B,T,dk)
            q = self.rot(self.q.view(1, 1, *self.q.shape), shift.float()[..., None, None] * self.om)   # (B,Z,M,dk)
        v = self.v(h)
        att = torch.einsum('bzmd,btd->bzmt', q.to(v.dtype), k.to(v.dtype)) / math.sqrt(self.dk)
        o = torch.einsum('bzmt,btv->bzmv', att.softmax(-1), v)
        return self.mlp(o.flatten(-2))


class ZTOK(nn.Module):
    CW = 16                      # fine z bins per coarse bin (640 km/s)

    def __init__(self, fm_cfg, M1=16, M2=32, d_rr=320, rr_layers=8, K=256, sep=2, evidence=True):
        super().__init__()
        self.K, self.sep, self.evidence = K, sep, evidence
        self.fm = SpecFMNative(fm_cfg['d'], fm_cfg['L'], fm_cfg['p'], max(1, fm_cfg['d'] // 64),
                               fm_cfg.get('stem_c', 0), fm_cfg.get('stem_n', 0))
        d = fm_cfg['d']
        self.register_buffer('u', torch.from_numpy(token_u(fm_cfg['p'])).float(), persistent=False)
        self.NC = math.ceil(N_Z / self.CW)
        self.register_buffer('ccen', (torch.arange(self.NC) * self.CW + (self.CW - 1) / 2).float(), persistent=False)
        self.coarse = RestAttn(d, M1)
        self.coarse_out = nn.Linear(256, 1)
        self.fine = RestAttn(d, M2)
        self.off = nn.Linear(256, 1)
        self.resamp = LogResample()
        n_ev = EV.N_FEAT if evidence else 0
        self.rr_in = nn.Linear(256 + 2 + n_ev, d_rr)
        lay = nn.TransformerEncoderLayer(d_rr, 8, 4 * d_rr, batch_first=True, norm_first=True, dropout=0.0)
        self.rr = nn.TransformerEncoder(lay, rr_layers, norm=nn.LayerNorm(d_rr), enable_nested_tensor=False)
        self.rr_out = nn.Linear(d_rr, 1)
        n = 2 * d + 2
        self.chead = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 256), nn.GELU(), nn.Linear(256, len(CLASSES)))
        nn.init.zeros_(self.chead[-1].weight); nn.init.zeros_(self.chead[-1].bias)

    tokens, logx = ZFMN.tokens, ZFMN.logx

    def forward(self, flux, ivar):
        h = self.tokens(flux, ivar)                                         # (B, T, d)
        B = h.shape[0]
        sh = lambda b: b + S_LO                                             # z bin -> shift on the log grid
        lc = self.coarse_out(self.coarse(h, self.u, sh(self.ccen).expand(B, -1))).squeeze(-1)  # (B, NC)
        logp_c = torch.log_softmax(lc.float(), -1)
        cls, cb = topk_peaks(logp_c.detach().unsqueeze(1), self.K, sep=self.sep)           # (B, K) coarse bins
        cen = self.ccen[cb]                                                 # (B, K) fine-bin centres
        g = self.fine(h, self.u, sh(cen))                                   # (B, K, 256)
        off = self.CW * torch.tanh(self.off(g).squeeze(-1).float())        # (B, K) in fine bins, +-16
        zf = cen + off
        zi = zf.detach().round().long().clamp(0, N_Z - 1)
        feat = [g, logp_c.detach().gather(1, cb).unsqueeze(-1), (cen / N_Z).unsqueeze(-1)]
        if self.evidence:
            feat.append(EV.candidate_features(EV.spectrum_maps(self.logx(flux, ivar)), zi).to(g.dtype))
        logit = self.rr_out(self.rr(self.rr_in(torch.cat([f.to(g.dtype) for f in feat], -1)))).squeeze(-1)
        hf = h.float()
        logit_c = self.chead(torch.cat([hf.mean(1), hf.amax(1), logp_c.detach().amax(1, keepdim=True),
                                        torch.logsumexp(lc.detach().float(), 1, keepdim=True)], 1))
        return dict(logp_c=logp_c, logit_c=logit_c, cand_cb=cb, cand_cen=cen, cand_off=off, cand_z=zi,
                    cand_logit=logit)

    def losses(self, o, zb, c):
        lz = F.nll_loss(o['logp_c'], zb // self.CW)
        d = zb.view(-1, 1).float() - o['cand_cen']                          # truth - centre, fine bins
        tgt = d.abs().argmin(1)
        near = d.abs() <= self.CW
        lo = (F.smooth_l1_loss(o['cand_off'], d.clamp(-self.CW, self.CW), reduction='none') * near).sum() \
            / near.sum().clamp_min(1)
        return dict(z=lz, rank=F.cross_entropy(o['cand_logit'].float(), tgt), off=lo)

    def readz(self, o):
        best = o['cand_logit'].float().argmax(1)
        zf = (o['cand_cen'] + o['cand_off']).gather(1, best.view(-1, 1)).squeeze(1).detach().cpu().numpy()
        return bin_to_z(np.rint(zf)), bin_to_z(zf)


HEADS = dict(unpatch=ZFMN, token=ZTOK)


def make_head(head, fm_cfg, a):
    if head == 'unpatch':
        return ZFMN(fm_cfg, a['c_up'], a['n_adapt'], d_rr=a['d_rr'], evidence=not a['no_evidence'])
    return ZTOK(fm_cfg, d_rr=a['d_rr'], evidence=not a['no_evidence'])


def subbin_z(o):
    """reranker pick + parabola on log P(z) around it (agree_eval.py's read-out) -> (z bin-level, z sub-bin)"""
    best = o['cand_logit'].float().argmax(1)
    b = o['cand_z'].gather(1, best.view(-1, 1)).squeeze(1)
    lp = o['logp_z'].float()
    bm, bp = (b - 1).clamp(0, N_Z - 1), (b + 1).clamp(0, N_Z - 1)
    l0, lm, lpp = (lp.gather(1, q.view(-1, 1))[:, 0] for q in (b, bm, bp))
    den = lm - 2 * l0 + lpp
    d = torch.where(den < 0, 0.5 * (lm - lpp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    bn = b.cpu().numpy()
    return bin_to_z(bn), bin_to_z(bn + d.detach().cpu().numpy())


def build(a, dev):
    if a.fm:
        ck = torch.load(a.fm, map_location='cpu', weights_only=False)
        fm_cfg = ck['cfg']
    else:
        fm_cfg = dict(d=768, L=12, p=8)
    net = make_head(a.head, fm_cfg, vars(a)).to(dev)
    if a.fm and not a.fm_random:
        net.fm.load_state_dict(ck['model'], strict=True)
    return net, fm_cfg


def train(a):
    rank, world, local = ddp_setup(); is0 = rank == 0
    torch.manual_seed(a.seed); torch.cuda.set_device(local); dev = 'cuda:%d' % local
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    # owner 2026-10-09 (supersedes clean-2): no cleaning, the original ZWARN=0 files -> --clean 0, passed explicitly
    assert a.clean == 0, 'owner rule: no cleaning, pass --clean 0'
    core, fm_cfg = build(a, dev)
    if a.freeze:
        for q in core.fm.parameters():
            q.requires_grad_(False)
    net = DDP(core, device_ids=[local], find_unused_parameters=True) if world > 1 else core
    # parameter groups: FM (AdamW, lr x fm_lr_mult, or frozen), Muon on hidden 2-D matrices of the adapter and the
    # reranker Transformer (C4's --muon --muon_lr 1e-3), AdamW on everything else at lr.  Muon is NOT used on the
    # FM: Muon on an AdamW-pretrained model is not measured.
    fmp = [q for q in core.fm.parameters() if q.requires_grad]
    fm_ids = {id(q) for q in core.fm.parameters()}
    mu = [q for nm, q in core.named_parameters()
          if q.ndim == 2 and (nm.startswith('adapt.') or nm.startswith('rr.tf.') or nm.startswith('rr.layers.'))
          and a.muon_lr > 0]
    mu_ids = {id(q) for q in mu}
    rest = [q for q in core.parameters() if id(q) not in fm_ids and id(q) not in mu_ids]
    groups = [dict(params=rest, lr=a.lr)]
    if fmp:
        groups.append(dict(params=fmp, lr=a.lr * a.fm_lr_mult))
    opts = [torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01)]
    peak = [[g['lr'] for g in groups]]
    if mu:
        opts.append(torch.optim.Muon(mu, lr=a.muon_lr, weight_decay=0.01, momentum=0.95, nesterov=True,
                                     adjust_lr_fn='match_rms_adamw'))
        peak.append([a.muon_lr])
    gbs = a.bs * world; steps = max(a.size // gbs, 1)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    scheds = [torch.optim.lr_scheduler.OneCycleLR(o, mx, total_steps=steps, pct_start=0.05, cycle_momentum=False)
              for o, mx in zip(opts, peak)]
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, clean=a.clean, native=True),
                    batch_size=a.bs, num_workers=a.workers, pin_memory=True, drop_last=True)
    arm = 'rand' if a.fm_random or not a.fm else ('fr' if a.freeze else 'ft')
    hd = 'U_cu%d_na%d' % (a.c_up, a.n_adapt) if a.head == 'unpatch' else 'T'
    stem = '%s_%s_%s_%d_d%d_s%d%s%s_cl%d' % (a.tag, hd, arm, a.size, a.d_rr, a.seed, '_noev' if a.no_evidence else '',
                                              '' if a.fm_lr_mult == 0.1 else '_fl%g' % a.fm_lr_mult, a.clean)
    if is0:
        parts = ' '.join('%s %.2fM' % (nm, n_params(m) / 1e6) for nm, m in core.named_children() if nm != 'fm' and n_params(m))
        print('%s %s | FM %s p=%d d=%d L=%d (%s) | params fm %.2fM head %.2fM (%s) | '
              'size %d steps %d gbs %d (%d x %d) lr %.1e fm_lr x%g muon %.1e | data clean-%d native'
              % (type(core).__name__, stem, os.path.basename(a.fm) if a.fm else '-', fm_cfg['p'], fm_cfg['d'], fm_cfg['L'],
                 'random init' if arm == 'rand' else ('frozen' if a.freeze else 'fine-tuned'),
                 n_params(core.fm) / 1e6, (n_params(core) - n_params(core.fm)) / 1e6, parts,
                 a.size, steps, gbs, a.bs, world, a.lr, a.fm_lr_mult, a.muon_lr, a.clean), flush=True)
    t0 = time.time(); net.train()
    if a.freeze:
        core.fm.eval()
    run = []; it = 0
    for fl, iv, zb, c, ztrue in tr:
        if it >= steps:
            break
        fl, iv, zb, c = fl.to(dev, non_blocking=True), iv.to(dev, non_blocking=True), zb.to(dev), c.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(fl, iv)
            L_ = core.losses(o, zb, c)
            lz, lr_, lo = L_['z'], L_['rank'], L_['off']
            lc = class_ce(o['logit_c'].float(), c)
            loss = lz + a.w_class * lc + a.w_rank * lr_ + a.w_off * lo
        for o_ in opts:
            o_.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in core.parameters() if q.requires_grad], 1.0)
        for o_ in opts:
            o_.step()
        for s_ in scheds:
            s_.step()
        run.append([float(lz), float(lc), float(lr_), float(lo)])
        if not all(math.isfinite(v) for v in run[-1]):
            raise RuntimeError('non-finite loss at step %d: %s' % (it, run[-1]))
        if it == 0 and is0:
            # zfm's pre-registered init values: z ~ ln N_Z = 9.40 (norm_corr range 8.3-10.7), class ln 5, rank ln K
            nz = N_Z if a.head == 'unpatch' else core.NC
            print('  init  z %.3f (ln %d %.3f)  class %.3f (ln 5 %.3f)  rank %.3f (ln K %.3f)  offset %.3f'
                  % (run[0][0], nz, math.log(nz), run[0][1], math.log(5), run[0][2], math.log(a.K), run[0][3]), flush=True)
        if it >= 2000 and it % 200 == 0:
            mr = float(np.mean([r[2] for r in run[-200:]]))
            if mr > 2 * np.log(a.K):
                raise RuntimeError('rank loss diverged at step %d: %.3g' % (it, mr))
        it += 1
        if is0 and (it % 200 == 0 or it == steps):
            m = np.mean(run[-200:], 0); el = time.time() - t0
            print('  step %6d/%d  z %.4f  class %.4f  rank %.4f  offset %.4f  %.0f spectra/s  mem %.1f GB  elapsed %.0fs  eta %.0fs'
                  % (it, steps, m[0], m[1], m[2], m[3], it * gbs / el, torch.cuda.max_memory_allocated() / 1e9, el,
                     el / it * (steps - it)), flush=True)
    if is0:
        res = dict(stem=stem, cfg=vars(a), fm_cfg={k: fm_cfg[k] for k in ('d', 'L', 'p')}, steps=it,
                   seconds=time.time() - t0, train_last200=dict(zip(['z', 'class', 'rank', 'offset'], np.mean(run[-200:], 0).tolist())))
        os.makedirs('results', exist_ok=True)
        json.dump(res, open('results/%s.json' % stem, 'w'), indent=1)
        if not a.max_steps:
            torch.save({'model': core.state_dict(), 'cfg': vars(a), 'fm_cfg': fm_cfg, 'res': res}, CKDIR + stem + '.pt')
            print('checkpoint -> %s%s.pt' % (CKDIR, stem), flush=True)
        print('TRAIN DONE %s %.0fs' % (stem, time.time() - t0), flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
        sys.stdout.flush(); os._exit(0)


def test(a):
    """The 60k clean-2 test of C4 (agree_eval.py --n 60000 --clean_val 2): run with EXACTLY 2 ranks, batch 32,
    16 workers, so the stream is the same objects in the same order; zt is checked against pred60k_c2."""
    rank = int(os.environ.get('SLURM_PROCID', 0)); world = int(os.environ.get('SLURM_NTASKS', 1))
    assert world == 2, 'the C4 60k test set was drawn with 2 ranks; run with -n2'
    torch.cuda.set_device(rank % torch.cuda.device_count())
    out = {}
    # --test_clean 0 (owner 2026-10-09: 「用一組不篩選的 60k 測」): the same validation blocks, unfiltered, = C4's pred60k_raw
    ld = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world, clean=a.test_clean, native=True),
                    batch_size=32, num_workers=16, pin_memory=True, drop_last=True)
    nets = {}
    for stem in a.arms.split(','):
        ck = torch.load(CKDIR + stem + '.pt', map_location='cpu', weights_only=False); c = ck['cfg']
        m = make_head(c['head'], ck['fm_cfg'], c).cuda().eval()
        m.load_state_dict(ck['model']); nets[stem] = m
    keys = ['zt', 'cls'] + [s + x for s in nets for x in ('', '_i', '_c')]
    out = {k: [] for k in keys}; n = 0; cap = 60000 // world
    with torch.no_grad():
        for fl, iv, zb, cl, z in ld:
            if n >= cap:
                break
            fl, iv = fl.cuda(), iv.cuda()
            out['zt'].append(z.numpy()); out['cls'].append(cl.numpy())
            for s, m in nets.items():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    o = m(fl, iv)
                zb_, zi = m.readz(o)
                out[s].append(zb_); out[s + '_i'].append(zi)
                out[s + '_c'].append(o['logit_c'].float().argmax(1).cpu().numpy())
            n += len(z)
    out = {k: np.concatenate(v) for k, v in out.items()}
    refname = {2: 'pred60k_c2', 1: 'pred60k_c1', 0: 'pred60k_raw'}[a.test_clean]
    ref = np.load('results/%s_r%d.npz' % (refname, rank))
    same = len(ref['zt']) == len(out['zt']) and np.array_equal(ref['zt'], out['zt'])
    np.savez('results/%s_r%d.npz' % (a.tag, rank), **out)
    print('rank %d: %d objects, same objects and order as C4 %s: %s' % (rank, n, refname, same), flush=True)
    os._exit(0)


def selftest():
    """CPU checks: LogResample == np.interp; labelled native batch shapes; forward shapes; z-shift equivariance of
    the adapter on the 40 km/s grid."""
    torch.manual_seed(0)
    r = LogResample()
    v = np.random.default_rng(0).normal(size=N_NATIVE).astype(np.float32)
    ref = np.interp(LAM_OBS, LAM_NATIVE, v)
    got = r(torch.from_numpy(v)).numpy()
    print('LogResample vs np.interp: max |diff| %.2e over %d points (lam %.1f-%.1f)'
          % (np.abs(got - ref).max(), len(ref), LAM_OBS[0], LAM_OBS[-1]))
    ds = Spectra(64, seed=1, clean=2, native=True)
    fl, iv, zb, c, z = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    print('labelled native batch:', tuple(fl.shape), tuple(iv.shape), zb.tolist(), c.tolist(), [round(float(q), 4) for q in z])
    x = np.stack([np.interp(LAM_OBS, LAM_NATIVE, fl[0].numpy()), np.interp(LAM_OBS, LAM_NATIVE, iv[0].numpy())])
    net = ZFMN(dict(d=64, L=2, p=8), c_up=16, n_adapt=2, d_rr=64, rr_layers=2)
    lx = net.logx(fl[:1], iv[:1])[0].numpy()
    print('evidence input vs data.py log-grid path: max |diff| %.2e' % np.abs(lx - np.stack([x[0] * np.sqrt(x[1]), np.sqrt(x[1])])).max())
    o = net(fl, iv)
    print('forward:', {k: tuple(v.shape) for k, v in o.items()})
    e = torch.randn(1, 16, N_OBS)
    y0, y1 = e, torch.roll(e, 40, -1)
    for b in net.adapt:
        y0, y1 = b(y0), b(y1)
    print('adapter shift test (40 bins = 1600 km/s): max |f(shift x) - shift f(x)| interior %.2e'
          % (torch.roll(y0, 40, -1) - y1)[..., 100:-100].abs().max().item())
    tnet = ZTOK(dict(d=64, L=2, p=8), d_rr=64, rr_layers=2)
    o = tnet(fl, iv)
    print('token head forward:', {k: tuple(v.shape) for k, v in o.items()}, 'losses',
          {k: round(float(v), 3) for k, v in tnet.losses(o, zb, c).items()}, '(coarse z ln %d = %.3f)' % (tnet.NC, math.log(tnet.NC)))
    ra = tnet.fine.double()
    h = torch.randn(2, len(tnet.u), 64, dtype=torch.float64)
    sh = torch.tensor([[3000.0, 5000.0]], dtype=torch.float64).expand(2, -1)
    with torch.no_grad():
        d0 = ra(h, tnet.u.double(), sh); d1 = ra(h, tnet.u.double() + 137, sh + 137)
    print('RestAttn redshift equivariance (tokens and trial shift both +137 bins): max |diff| %.2e' % (d0 - d1).abs().max().item())
    print('tokens %d, u from %.1f to %.1f bins (N_OBS %d)' % (len(tnet.u), tnet.u[0].item(), tnet.u[-1].item(), N_OBS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'test', 'selftest'])
    ap.add_argument('--size', type=int, default=300000)
    ap.add_argument('--fm', default='', help='FM checkpoint (specfm_native.py) path')
    ap.add_argument('--fm_random', action='store_true', help='FM architecture from --fm cfg, random init (N-rand)')
    ap.add_argument('--freeze', action='store_true', help='FM frozen (N-fr)')
    ap.add_argument('--fm_lr_mult', type=float, default=0.1)
    ap.add_argument('--head', choices=['unpatch', 'token'], required=False, default='')
    ap.add_argument('--w_off', type=float, default=0.1, help='token head: offset loss weight')
    ap.add_argument('--c_up', type=int, default=64); ap.add_argument('--n_adapt', type=int, default=2)
    ap.add_argument('--d_rr', type=int, default=320); ap.add_argument('--K', type=int, default=256)
    ap.add_argument('--no_evidence', action='store_true', help='arm E0')
    ap.add_argument('--bs', type=int, default=8, help='per GPU; C4 used global 32')
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--muon_lr', type=float, default=1e-3)
    ap.add_argument('--w_class', type=float, default=0.1); ap.add_argument('--w_rank', type=float, default=1.0)
    ap.add_argument('--clean', type=int, default=-1, help='must be 0 (owner rule 2026-10-09: no cleaning); pass it explicitly')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=0); ap.add_argument('--tag', default='ZN')
    ap.add_argument('--arms', default='', help='test: comma-separated checkpoint stems')
    ap.add_argument('--test_clean', type=int, default=2, help='test set: 2 = clean-2 (default), 0 = unfiltered')
    a = ap.parse_args()
    assert a.mode != 'train' or a.head, 'pass --head unpatch or --head token'
    {'train': train, 'test': test, 'selftest': lambda a: selftest()}[a.mode](a)


if __name__ == '__main__':
    main()
