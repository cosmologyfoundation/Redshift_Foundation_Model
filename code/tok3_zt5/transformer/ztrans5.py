"""ztrans4 (40 km/s detail in every narrow token) + B: a "line sentence" per candidate redshift (owner 2026-10-11:
「把光譜包成 SOS .... [O II] ....Hβ....[O III].....Hα....CIV... EOS」, 「寫B 跟40 km/s一起」).

The name of a line depends on z, so the sentence is built per candidate.  For each of the decoder's 8 candidates:
    SOS   = the candidate's own decoder state (what the model already thinks of this z)
    word  = one per line of the fixed rest-frame list (zfm/evidence.py LINES: Lya .. [S II], 22 lines, vacuum)
            name embedding
          + its 40 km/s window: +-32 bins (+-1,280 km/s) around where the line falls at this z, 67 channels
            (tokenizer input 3 + first stage 64) + an in-spectrum flag, read by a small 1-D conv (keeps the shape:
            [O II] 3727/3729 is ~5 bins apart, the [O III] pair is two words with a fixed ratio)
          + the Transformer encoder's narrow token at that position (context, 160 km/s)
          + the Transformer encoder's broad token at that position (2,560 km/s: broad QSO lines are wider than
            the window)
          + in-spectrum flag; a line outside 3600-9824 A gets only its name and the flag
    EOS   = learned
    -> 2 Transformer layers (d 256, 8 heads) over [SOS, 22 words, EOS] -> SOS -> one score per candidate,
       ADDED to the decoder's pick logit.  So the sentence can only re-rank the candidates; the pick loss
       (cross-entropy over candidates, target = the candidate nearest the Redrock z) trains both.
The line list is atomic physics (rest wavelengths), not Redrock; nothing about which lines "should" be present or
their ratios is given -- the sentence reader learns that.
Everything else is ztrans4 / ztrans2 (heads, losses, data, early stop, per-type report).
"""
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
import ztrans2
from ztrans4 import ZTrans40, encode40, C40
from ztrans2 import topk_peaks, CW, N_Z, S_LO, mlp
from dualtok import NN, NB, NPAD
from grids import N_OBS, DLOG, REST_LO
from evidence import LINES

NL = len(LINES)
WIN = 32                                      # +-32 bins = +-1,280 km/s
REST_PIX = torch.tensor([np.log(l[1] / REST_LO) / DLOG for l in LINES], dtype=torch.float32)


class SentBlock(nn.Module):
    def __init__(self, d, h, swiglu=False):
        super().__init__()
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = mlp(d, swiglu)

    def forward(self, x):
        y = self.n1(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class LineSentence(nn.Module):
    def __init__(self, d_model=768, d=256, h=8, L=2, swiglu=False):
        super().__init__()
        self.name = nn.Parameter(torch.randn(NL, d) * 0.02)
        self.eos = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.inw = nn.Parameter(torch.randn(2, d) * 0.02)                     # 0 outside, 1 inside the spectrum
        self.conv = nn.Sequential(nn.Conv1d(C40 + 1, 128, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU())
        nwin = 2 * WIN + 1
        self.win_out = nn.Linear(128 * ((((nwin + 1) // 2) + 1) // 2), d)
        self.w_norm = nn.GroupNorm(1, C40)
        self.ctx_n, self.ctx_b, self.sos = nn.Linear(d_model, d), nn.Linear(d_model, d), nn.Linear(d_model, d)
        self.blocks = nn.ModuleList([SentBlock(d, h, swiglu) for _ in range(L)])
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)       # starts as "no re-ranking"
        self.register_buffer('rest_pix', REST_PIX, persistent=False)
        self.register_buffer('offs', torch.arange(-WIN, WIN + 1), persistent=False)

    def forward(self, h40, mem_b, mem_n, q, cen):
        """h40 (B,67,7552) 40 km/s maps; mem_b (B,118,D), mem_n (B,1882,D) encoder outputs; q (B,K,D) decoder states;
        cen (B,K) candidate centres in fine z bins -> (B,K) re-ranking scores"""
        B, K = cen.shape
        pos = self.rest_pix.view(1, 1, NL) + (cen.float() + S_LO).unsqueeze(-1)            # (B,K,NL) observed bin
        inside = (pos >= 0) & (pos <= N_OBS - 1)
        pc = pos.round().long()
        idx = pc.unsqueeze(-1) + self.offs                                                 # (B,K,NL,W)
        ok = (idx >= 0) & (idx <= N_OBS - 1)
        h = self.w_norm(h40.float()).to(h40.dtype)
        g = h.gather(2, idx.clamp(0, N_OBS - 1).view(B, 1, -1).expand(-1, C40, -1))           # (B,C,K*NL*W)
        g = g.view(B, C40, K * NL, -1).permute(0, 2, 1, 3)                                   # (B,K*NL,C,W)
        okf = ok.view(B, K * NL, 1, -1).to(g.dtype)
        x = torch.cat([g * okf, okf], 2).reshape(B * K * NL, C40 + 1, -1)
        w = self.win_out(self.conv(x).flatten(1)).view(B, K, NL, -1)
        jn = (pc // 4).clamp(0, NN - 1).view(B, -1)
        jb = (pc // 64).clamp(0, NB - 1).view(B, -1)
        cn = self.ctx_n(mem_n.gather(1, jn.unsqueeze(-1).expand(-1, -1, mem_n.shape[-1]))).view(B, K, NL, -1)
        cb = self.ctx_b(mem_b.gather(1, jb.unsqueeze(-1).expand(-1, -1, mem_b.shape[-1]))).view(B, K, NL, -1)
        ins = inside.unsqueeze(-1).to(w.dtype)
        word = self.name + self.inw[inside.long()] + ins * (w + cn + cb)                    # (B,K,NL,d)
        sent = torch.cat([self.sos(q).unsqueeze(2), word, self.eos.expand(B, K, -1).unsqueeze(2).to(word.dtype)], 2)
        s = sent.reshape(B * K, NL + 2, -1)
        for b in self.blocks:
            s = b(s)
        return self.out(self.norm(s[:, 0])).view(B, K).float()


def encode40_full(tok, flux, ivar):
    """encode40 plus the raw 40 km/s map (B,67,7552) that the sentence windows read"""
    x, f, n, s = tok.prep(flux, ivar)
    x = F.pad(tok.to_log(x), (0, NPAD - N_OBS))
    h = x
    for i, st in enumerate(tok.enc):
        h = st(h)
        if i == 0:
            h40 = torch.cat([x.to(h.dtype), h], 1)
        if i == 2:
            hn = h.transpose(1, 2)
    hb = h.transpose(1, 2)
    B = x.shape[0]
    fine = h40[:, :, :4 * NN].reshape(B, C40, NN, 4).permute(0, 2, 3, 1).reshape(B, NN, 4 * C40)
    return hn[:, :NN], hb, fine, s, h40


class ZTrans5(ZTrans40):
    def __init__(self, d=768, **kw):
        super().__init__(d=d, **kw)
        import types
        self.tok.encode = types.MethodType(encode40_full, self.tok)
        self.sent = LineSentence(d_model=d, swiglu=kw.get('swiglu', False))

    def forward(self, flux, ivar):
        with torch.no_grad() if not any(p.requires_grad for p in self.tok.parameters()) else torch.enable_grad():
            hn, hb, fine, s, h40 = self.tok.encode(flux, ivar)
        B = hn.shape[0]
        c0 = self.cls.expand(B, -1, -1) + self.s_in(torch.log10(s).view(B, 1, 1).to(hn.dtype))
        nar = self.n_in(hn) + self.f_in(self.f_norm(fine.to(hn.dtype))) + self.typ[1]
        x = torch.cat([c0.to(hn.dtype), self.b_in(hb) + self.typ[0], nar], 1) + self.pos
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
        rr = self.sent(h40, mem[:, :NB], mem[:, NB:], q, cen)
        return dict(coarse=lc, cand_cb=cb_, cand_cen=cen, pick=self.pick(q).squeeze(-1).float() + rr, sent=rr,
                    off=self.offh(q).float(), logit_c=self.chead(g).float())


def selftest():
    torch.manual_seed(0)
    m = ZTrans5(d=64, H=32, L=1, Ldec=1, Kc=4)
    for p in m.tok.parameters():
        p.requires_grad_(False)
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    o = m(fl, iv)
    print('forward:', {k: tuple(v.shape) for k, v in o.items()})
    # a line's window centre must be the observed pixel of lambda_rest (1+z) at the candidate z
    from grids import LAM_OBS, bin_to_z
    c = float(o['cand_cen'][0, 0]); z = float(bin_to_z(c))
    p = float(REST_PIX[6]) + c + S_LO                                  # [O II]
    lam = 3600.0 * np.exp(p * DLOG)
    print('candidate z %.4f: [O II] predicted at %.2f A, window centre %.2f A' % (z, LINES[6][1] * (1 + z), lam))
    assert abs(lam / (LINES[6][1] * (1 + z)) - 1) < 2e-4
    zb = torch.tensor([3000, 6000]); cl = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float)
    sum(ztrans2.losses(o, zb, cl)).backward()
    print('sentence out grad %.3e | trainable params without grad: %s'
          % (m.sent.out.weight.grad.norm(), [k for k, p_ in m.named_parameters() if p_.requires_grad and p_.grad is None]))
    full = ZTrans5()
    print('params: sentence reader %.2fM, whole model without tokenizer %.2fM'
          % (sum(p_.numel() for p_ in full.sent.parameters()) / 1e6,
             sum(p_.numel() for p_ in full.parameters()) / 1e6 - sum(p_.numel() for p_ in full.tok.parameters()) / 1e6))


if __name__ == '__main__':
    if '--selftest5' in sys.argv:
        selftest(); sys.exit()
    ztrans2.ZTrans = ZTrans5
    ztrans2.main()
