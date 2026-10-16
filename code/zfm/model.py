"""The model of DESIGN.md 3.

    spectrum (2 x N_OBS, log-lambda)
        |
        +--> ConvNeXt-block encoder            <- v13, measured 5.9s/5.5s
        |
        +--> project to C_CORR channels
        |
        +--> learned rest-frame archetype bank, per class
        |       FFT cross-correlation over ALL shifts = all redshifts at once
        |
        v
      S(class, z)      a 5 x N_Z surface, EVERY class at EVERY redshift
        |
        +--> marginals:  log P(z) = logsumexp_class S     <- the dense loss
        |                log P(class) = logsumexp_z S     <- multi-label, a set
        |
        +--> top-K peaks over the WHOLE surface, classes mixed
                 |
                 +--> cross-attention reranker -> one logit per candidate

The class is a marginal of the same surface that produces z.  It is never a
switch in front of it: measured, gating multiplies the redshift failure rate by
43x, and 85-89% of errors are "right answer among the candidates, ranked wrong".
"""
import math
import numpy as np
import torch
import evidence as EV
import torch.nn as nn
import torch.nn.functional as F

from grids import (N_OBS, N_REST, N_Z, S_LO, CLASSES, DLOG, OBS_LO, REST_LO)

# Calibrated by calib.py at d_enc=384, c_corr=8, n_per=8 so the normalised
# surface starts with a logit std over z near 0.5 -- NLL(init) = ln(N_Z)+0.12.
# It must not be 1.0: that gives std 2.3e-4, and log_scale would have to grow
# by ln(2000)=7.6 to sharpen the surface while AdamW at lr 3e-4 can move it
# ~0.3 in a 937-step run.  The small-D points would then be limited by the
# optimiser reaching the right temperature, not by data -- the same upward
# bias in beta the normalisation was introduced to remove.  train.py asserts
# the init loss, so a change of architecture that invalidates this number
# refuses to run instead of quietly biasing the ladder.
NORM_GAIN = 536.2


# --------------------------------------------------------------- encoder
class ConvNeXt1D(nn.Module):
    """Adopted in v13: depthwise k=7 -> LayerNorm -> 4x inverted bottleneck.
    NO downsampling stages and NO stem patching -- both break translation
    equivariance below their own scale, and redshift IS a translation."""

    def __init__(self, d, k=7, mlp=4, layer_scale=1e-6, swiglu=False):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.norm = nn.LayerNorm(d)
        # swiglu: down(silu(a) * b) with [a, b] = up(t); hidden 2/3 of the GELU
        # width so the parameter count is the same (3 * d * 8d/3 = 2 * d * 4d)
        self.swiglu = swiglu
        hid = (2 * d * mlp) // 3 if swiglu else d * mlp
        self.up = nn.Linear(d, 2 * hid if swiglu else hid)
        self.act = nn.GELU()
        self.down = nn.Linear(hid, d)
        self.gamma = nn.Parameter(layer_scale * torch.ones(d))

    def forward(self, x):
        r = x
        t = self.norm(self.dw(x).transpose(1, 2))
        if self.swiglu:
            a, b = self.up(t).chunk(2, -1)
            t = self.down(F.silu(a) * b)
        else:
            t = self.down(self.act(self.up(t)))
        return r + (t * self.gamma).transpose(1, 2)


class Encoder(nn.Module):
    """d = 384 gives 14.2 M over 12 layers, which is the ~15 M DESIGN.md 3
    asks for.  An earlier version of this file used d = 128 (1.6 M),
    copied from the encoder A/B where that width was chosen only to match
    the dilated arm's parameter count in a small comparison run.  The
    document said 15 M, the code said 1.6 M, and the FLOP accounting of
    DESIGN.md 9 was done at 15 M -- so every cost in it was right for the
    design and wrong for what was actually built."""

    def __init__(self, d=384, layers=12, c_corr=8, swiglu=False):
        super().__init__()
        self.stem = nn.Conv1d(2, d, 7, padding=3)
        self.blocks = nn.ModuleList([ConvNeXt1D(d, swiglu=swiglu) for _ in range(layers)])
        self.proj = nn.Conv1d(d, c_corr, 1)

    def forward(self, x, return_h=False):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return (self.proj(h), h) if return_h else self.proj(h)


# --------------------------------------------- archetype bank + correlation
class Surface(nn.Module):
    """S(class, z) by FFT cross-correlation against learned rest-frame
    archetypes.  Capacity-limited ON PURPOSE: with too many archetypes every
    redshift finds a template and the aliases come back."""

    def __init__(self, n_per=8, c_corr=8, n_class=len(CLASSES),
                 norm_corr=False):
        super().__init__()
        self.n_per, self.c, self.n_class = n_per, c_corr, n_class
        # norm_corr: divide the raw dot product by ||e||*||w|| before the
        # softmax.  Without it the surface's SHARPNESS tracks the object's
        # flux level, not the quality of the match: measured at init the
        # logits have std 4.15 over z on real spectra (0.38 on white noise),
        # and NLL = ln(N_Z) + s^2/2 = 18.02 against a pre-registered 9.404.
        # The cost is a fixed startup penalty that is a LARGER fraction of a
        # short run than a long one, which biases beta upward in L(D).
        self.norm_corr = norm_corr
        self.A = n_class * n_per
        self.arch = nn.Parameter(torch.randn(self.A, c_corr, N_REST) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(self.A))
        self.nfft = 1 << int(np.ceil(np.log2(N_OBS + N_REST)))
        # r[k] = sum_i enc[i] * arch[i-k] pairs LAM_OBS[i] with LAM_REST[i-k],
        # so 1+z = LAM_OBS[i]/LAM_REST[i-k] = (OBS_LO/REST_LO) * exp(k*DLOG),
        # which is exactly grids.Z_GRID at SHIFTS = k.  The lag index IS the
        # shift index and needs NO further offset.  An earlier version added
        # log(OBS_LO/REST_LO)/DLOG = 12063 bins here and the synthetic delta
        # test of check1.py caught it: every redshift came out wrong.

    def forward(self, e, chunk=32):
        """e: (B, c, N_OBS) -> S: (B, n_class, N_Z)"""
        B = e.shape[0]
        E = torch.fft.rfft(e.float(), n=self.nfft)                 # (B,c,F)
        out = []
        for a0 in range(0, self.A, chunk):
            w = self.arch[a0:a0 + chunk]
            W = torch.fft.rfft(w.float(), n=self.nfft)             # (a,c,F)
            P = torch.einsum('bcf,acf->baf', E, W.conj())
            r = torch.fft.irfft(P, n=self.nfft)                    # (B,a,nfft)
            idx = (torch.arange(N_Z, device=e.device) + S_LO) % self.nfft
            out.append(r.index_select(-1, idx))
        S = torch.cat(out, 1)                                      # (B,A,N_Z)
        if self.norm_corr:
            ne = e.float().pow(2).sum((1, 2)).sqrt().view(B, 1, 1)
            nw = self.arch.float().pow(2).sum((1, 2)).sqrt().view(1, -1, 1)
            S = S / (ne * nw + 1e-8) * NORM_GAIN
        S = S * self.log_scale.exp().view(1, -1, 1)
        S = S.view(B, self.n_class, self.n_per, N_Z)
        return torch.logsumexp(S, dim=2)                           # (B,n_class,N_Z)


# ------------------------------------------------------------- reranker
def topk_peaks(S, K=256, sep=64):
    """K is a CEILING on the whole system: the reranker can never recover a
    redshift that is not in the candidate list.

    Measured 2026-09-08 with a hand-built line comb and no training at all --
    a floor, not the trained value -- recall@K on 288 real DESI spectra at a
    2560 km/s tolerance:

        K =   1   20.1%      K =  32   51.7%      K = 128   69.4%
        K =   4   39.2%      K =  64   59.4%      K = 256   76.7%

    Still climbing at 256.  K = 32 was asserted in DESIGN.md 3 and never
    measured; at that value the system would be capped near 52% before the
    reranker sees anything.

    K is NOT free.  At this model's encoder (d=384, 14.24 M) the reranker is
    18.1% of total FLOPs at K=256, and the dense term scales linearly in K.
    An earlier version of this note computed 66% from a 1.6 M encoder that the
    document never specified and the code has since been corrected away from.

    RE-MEASURED on the TRAINED surface, 2026-09-09, D=100k checkpoint,
    30,000 unseen validation spectra, tolerance 8 bins (~1.1e-3 in dz/(1+z)):

        recall@256, all objects        98.95%
        recall@256, catastrophic only  68.78%
        recall@256, non-catastrophic   99.37%

    So of the 1.37% catastrophic rate, 68.8% is a RANKING failure and 31.2% is
    the surface never proposing the answer.  DESIGN.md 3 asserts 85-89%
    ranking; that was measured on the previous architecture and is ~17 points
    too high for this one.  A PERFECT reranker would take catastrophic from
    1.37% to 0.43% -- that 0.94 points is the entire budget the N-ladder is
    competing for, and no part of the remaining 0.43% is reachable from this
    axis.

    Where the correct candidate sat, among catastrophics that had one:
    rank 2 25.2%, top 10 55.3%, top 25 68.4%, top 100 88.3%, top 256 100%.
    Rank 1 is 0.0% by construction.  Cutting to K=25 would discard 31.6% of
    the recoverable errors, so K does not shrink; whether it should GROW
    depends on recall past 256, measured separately in recall_sweep.py.
    """
    """K largest points of the whole surface, classes mixed, with a minimum
    separation in z so the K slots are not spent on one peak's shoulders."""
    B, NC, NZ = S.shape
    flat = S.reshape(B, -1).clone()
    idx = torch.zeros(B, K, dtype=torch.long, device=S.device)
    for k in range(K):
        j = flat.argmax(-1)
        idx[:, k] = j
        c, z = j // NZ, j % NZ
        zz = torch.arange(NZ, device=S.device).view(1, -1)
        mask = (zz - z.view(-1, 1)).abs() < sep                    # (B,NZ)
        blank = torch.zeros(B, NC, NZ, dtype=torch.bool, device=S.device)
        blank[torch.arange(B), c] = mask
        flat = flat.masked_fill(blank.reshape(B, -1), -1e30)
    return idx // NZ, idx % NZ                                     # class, zbin


class SwiGLULayer(nn.Module):
    """Pre-LN transformer layer, nn.TransformerEncoderLayer(norm_first=True,
    dropout=0) with its ReLU FFN replaced by SwiGLU.  Hidden 8d/3 rounded to 8
    matches the 4d ReLU FFN's parameters to 0.3% at d = 320."""

    def __init__(self, d, heads):
        super().__init__()
        hid = int(round(8 * d / 3 / 8)) * 8
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.w12 = nn.Linear(d, 2 * hid)
        self.w3 = nn.Linear(hid, d)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        h = self.norm1(src)
        x = src + self.self_attn(h, h, h, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                 need_weights=False)[0]
        a, b = self.w12(self.norm2(x)).chunk(2, -1)
        return x + self.w3(F.silu(a) * b)


class Reranker(nn.Module):
    """Attention over CANDIDATES, not pixels.  One token per (class, z) peak
    carrying its own score, its local shape, and where it sits -- so candidates
    can be compared against each other, which is the only place the
    non-additivity of the evidence can live in the architecture."""

    def __init__(self, d=768, layers=8, heads=8, n_class=len(CLASSES), win=9,
                 final_norm=False, n_ev=0, swiglu=False):
        super().__init__()
        self.win = win
        self.swiglu = swiglu
        # n_ev: evidence.py's per-candidate features -- the break and every
        # line's significance AT THAT CANDIDATE'S z (DESIGN.md 5).  Without
        # them the reranker saw only the class, z and nine numbers of S, so it
        # could not ask "what else is true at this z", which is its whole job.
        f_in = n_class + 2 + win + n_ev
        self.inp = nn.Linear(f_in, d)
        lay = (SwiGLULayer(d, heads) if swiglu else
               nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                          norm_first=True, dropout=0.0))
        # final_norm: this is a PRE-LN stack (norm_first=True), and a pre-LN
        # transformer normalises only the INPUT to each block -- the residual
        # stream leaving the last block is never normalised unless the stack
        # is given norm=.  Without it the stream grew to 2.34e4 by the 300k
        # checkpoint and 3.90e5 by 1M (sqrt(d) = 17.9), cand_logit reached
        # 4.3e5, the rank gradient on the encoder became 578x the z gradient,
        # and the global clip at 1.0 cut z's update to 4e-4 of its size: z
        # stopped learning halfway through the 1M run.  Inputs were not the
        # cause -- |peak| max was 16 and 18 in the two checkpoints.
        self.tf = nn.TransformerEncoder(
            lay, layers, norm=nn.LayerNorm(d) if final_norm else None)
        self.out = nn.Linear(d, 1)

    def forward(self, S, cls, zb, ev=None, pad=None):
        B, NC, NZ = S.shape
        K = cls.shape[1]
        h = self.win // 2
        off = torch.arange(-h, h + 1, device=S.device).view(1, 1, -1)
        zz = (zb.unsqueeze(-1) + off).clamp(0, NZ - 1)
        loc = S[torch.arange(B).view(-1, 1, 1), cls.unsqueeze(-1), zz]  # (B,K,win)
        peak = loc[..., h:h + 1]
        feat = torch.cat([F.one_hot(cls, NC).float(),
                          zb.float().unsqueeze(-1) / NZ,
                          peak,
                          loc - peak], -1)
        if ev is not None:
            feat = torch.cat([feat, ev.to(feat.dtype)], -1)
        # pad: candidates that are not real (alias.py's masked slots) take no
        # part in attention, so they cannot colour the real candidates' scores
        return self.out(self.tf(self.inp(feat), src_key_padding_mask=pad)).squeeze(-1)  # (B,K)


CLIP = 50.0


class ZFM(nn.Module):
    def __init__(self, d_enc=384, enc_layers=12, c_corr=8, n_per=8,
                 d_rr=768, rr_layers=8, K=256, norm_corr=False,
                 rr_final_norm=False, sep=64, evidence=False, shape=False, sky=None,
                 softclip=False, class_head=False, swiglu_enc=False, swiglu_rr=False,
                 pairs=False):
        super().__init__()
        self.K = K
        # sep is a TRAINING parameter as well as an inference one: it decides
        # which candidates the reranker ever sees.  Measured on the D=1M
        # checkpoint over 100,096 spectra, sep=16 recovers 5.5 points of
        # catastrophic recall over 64 (105 gained / 44 lost, McNemar p=8.9e-7),
        # but collecting that needs a retrain, so the default stays 64 and the
        # ladder points already run remain comparable.
        self.sep = sep
        self.enc = Encoder(d_enc, enc_layers, c_corr, swiglu=swiglu_enc)
        self._enc_args = (d_enc, enc_layers, c_corr)
        self.surf = Surface(n_per, c_corr, norm_corr=norm_corr)
        self.evidence, self.shape = evidence, shape
        # pairs: evidence.pair_features, partner-line consistency (DESIGN.md 5)
        self.pairs = pairs
        # sky: evidence.spectrum_maps' sky-mask ratio (DESIGN.md 6e); None = unmasked,
        # which every checkpoint trained before 2026-09-23 expects
        self.sky = sky
        # softclip: the encoder sees CLIP*asinh(x/CLIP) instead of x clamped to
        # +-CLIP (DESIGN.md 2).  Identical to 1% below |x| = 12, monotone, and a
        # bright star keeps its shape instead of becoming a flat line at +-50.
        # A buffer, so a checkpoint carries its own input transform and every
        # loader gets it right without being told; old checkpoints lack the key
        # and load as clamped (load_state_dict below).
        self.register_buffer('softclip', torch.tensor(float(softclip)))
        self._soft = bool(softclip)
        self._rr_kw = dict(d=d_rr, layers=rr_layers, final_norm=rr_final_norm,
                           n_ev=(EV.N_FEAT if evidence else 0) + (EV.N_SHAPE if shape else 0)
                           + (EV.N_PAIR if pairs else 0))
        self.rr = Reranker(**self._rr_kw, swiglu=swiglu_rr)
        self.d_enc = d_enc
        if class_head:
            self._make_chead()

    def _make_chead(self):
        # class_head (DESIGN.md 2, class read-out): the old class logit is
        # logsumexp_z S[c] - ln N_Z, a FIXED read-out of the surface whose class
        # partition the z loss never constrains -- class_any_bit 0.197-0.846 over
        # six 150k runs, while a linear probe on the same encoders scores
        # 0.952-0.955 in every one (probe_class.py).  So the information is there
        # and only the read-out is broken: a learned head on the pooled encoder
        # state plus the per-class surface summary, with the surface DETACHED so
        # the class loss never pulls the redshift surface.  Last layer zero-init:
        # the class loss starts at exactly ln 5.
        nc = len(CLASSES)
        n = 2 * self.d_enc + 2 * nc
        self.chead = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 256), nn.GELU(), nn.Linear(256, nc))
        nn.init.zeros_(self.chead[-1].weight); nn.init.zeros_(self.chead[-1].bias)
        dev = self.surf.arch.device
        self.chead.to(dev)

    def forward(self, x, rerank=True, extra=None):
        # the clip data.py used to apply: the encoder input is unchanged, bit
        # for bit, while x itself stays whole for evidence.py
        xin = CLIP * torch.asinh(x / CLIP) if self._soft else x.clamp(-CLIP, CLIP)
        if hasattr(self, 'chead'):
            e, h = self.enc(xin, return_h=True)
        else:
            e = self.enc(xin)
        S = self.surf(e)
        logp_z = torch.log_softmax(torch.logsumexp(S, 1), -1)      # (B,N_Z)
        # mean, not sum: logsumexp over N_Z carries a constant +ln(N_Z)=9.40
        # offset, so every class logit starts strongly positive and the BCE
        # spends its early gradient dragging all five down through the SAME
        # parameters the z loss is trying to shape.  Measured: class loss
        # 9.19 at init instead of ln(2)=0.693.
        logit_c = torch.logsumexp(S, 2) - math.log(N_Z)             # (B,n_class)
        if hasattr(self, 'chead'):
            Sd = S.detach().float()
            f = torch.cat([h.float().mean(2), h.float().amax(2), Sd.amax(2), torch.logsumexp(Sd, 2)], 1)
            logit_c = self.chead(f)
        out = dict(S=S, logp_z=logp_z, logit_c=logit_c)
        if rerank:
            cls, zb = topk_peaks(S.detach(), self.K, sep=self.sep)
            if extra is not None:
                # training only: alias.py's hard negatives join the candidate
                # list, each with the class whose score is highest at that z, so
                # the rank loss must learn to reject a one-line explanation
                # whether or not the surface happened to propose it
                eb, em = extra
                ecls = S.detach().gather(2, eb.unsqueeze(1).expand(-1, S.shape[1], -1)).argmax(1)
                cls, zb = torch.cat([cls, ecls], 1), torch.cat([zb, eb], 1)
            ev = (EV.candidate_features(EV.spectrum_maps(x, sky=self.sky), zb, shape=self.shape,
                                          pairs=self.pairs)
                  if self.evidence else None)
            pad = None
            if extra is not None:
                pad = torch.cat([torch.zeros_like(em[:, :0]).new_zeros(em.shape[0], self.K), ~em], 1)
            logit = self.rr(S, cls, zb, ev, pad)
            if extra is not None:
                logit = torch.cat([logit[:, :self.K],
                                   logit[:, self.K:].masked_fill(~em, -1e4)], 1)
                # a masked slot must never be the rank target: the first run
                # took "the candidate nearest the truth" over ALL slots, and a
                # masked slot -- masked precisely because it sat within MIN_SEP
                # of the truth -- was often nearest.  Its logit is -1e4, so the
                # rank loss read 7.70 at step 200 (uniform over 288 is 5.66).
                zb = torch.cat([zb[:, :self.K],
                                zb[:, self.K:].masked_fill(~em, 10 ** 9)], 1)
            out.update(cand_class=cls, cand_z=zb, cand_logit=logit)
        return out

    def load_state_dict(self, sd, strict=True, **kw):
        sd = dict(sd)
        sd.setdefault('softclip', torch.zeros(()))
        # a checkpoint trained with the class head carries chead.* : build it,
        # so every loader that constructs ZFM without the flag still loads it
        if any(k.startswith('chead.') for k in sd) and not hasattr(self, 'chead'):
            self._make_chead()
        # SwiGLU checkpoints rebuild the matching encoder / reranker, so a loader
        # that constructs ZFM without the flags still loads them correctly
        dev = self.surf.arch.device
        # a pairs checkpoint has N_PAIR more reranker inputs: rebuild to match,
        # so loaders that do not pass pairs= still get the right reranker
        if 'rr.inp.weight' in sd and not self.pairs and \
                sd['rr.inp.weight'].shape[1] - self.rr.inp.weight.shape[1] == EV.N_PAIR:
            self.pairs = True
            self._rr_kw['n_ev'] += EV.N_PAIR
            self.rr = Reranker(**self._rr_kw, swiglu=self.rr.swiglu).to(dev)
        sw_enc = sd['enc.blocks.0.up.weight'].shape[0] != sd['enc.blocks.0.down.weight'].shape[1]
        if sw_enc != self.enc.blocks[0].swiglu:
            self.enc = Encoder(*self._enc_args, swiglu=sw_enc).to(dev)
        sw_rr = any('.w12.' in k for k in sd if k.startswith('rr.'))
        if sw_rr != self.rr.swiglu:
            self.rr = Reranker(**self._rr_kw, swiglu=sw_rr).to(dev)
        r = super().load_state_dict(sd, strict=strict, **kw)
        self._soft = bool(self.softclip.item())
        return r


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    torch.manual_seed(0)
    m = ZFM()
    print('encoder   %8.2f M' % (n_params(m.enc) / 1e6))
    print('archetypes %7.2f M   (%d archetypes x %d ch x %d rest)'
          % (n_params(m.surf) / 1e6, m.surf.A, m.surf.c, N_REST))
    print('reranker  %8.2f M' % (n_params(m.rr) / 1e6))
    print('total     %8.2f M' % (n_params(m) / 1e6))
    x = torch.randn(2, 2, N_OBS)
    o = m(x)
    print('S %s  logp_z %s  logit_c %s  cand %s'
          % (tuple(o['S'].shape), tuple(o['logp_z'].shape),
             tuple(o['logit_c'].shape), tuple(o['cand_logit'].shape)))
    print('logp_z sums to %.4f (should be 0 in log space -> 1)'
          % o['logp_z'].exp().sum(-1).mean())
