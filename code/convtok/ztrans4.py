"""ztrans2 + the 40 km/s detail (owner 2026-10-10: 「讓 Transformer 看到 40 km/s 的細節 先加進去」).

ztrans2 reads the narrow tokens (160 km/s, 4 bins each) and the broad tokens (2,560 km/s).  Here every narrow token
also gets its OWN four 40 km/s bins, from two sources, before the Transformer:
    the tokenizer's first ConvNeXt stage   64 ch x 4 bins   (enc[0], full resolution, before any downsampling)
    the tokenizer's input on the log grid   3 ch x 4 bins   (asinh flux, log ivar, measured) -- the raw detail,
                                                              independent of what reconstruction taught the encoder
    -> LayerNorm(268) -> Linear(268 -> 768), added to the narrow token's embedding.
So the [O II] 3727/3729 pair (~220 km/s apart, about 5 bins) and the [O III] 4960/5008 ratio are visible at 40 km/s
inside the tokens that the encoder and the rest-frame decoder attend to.  Sequence length is unchanged (2,001).
Tokenizer stays frozen.  Everything else (heads, losses, data, early stop, per-type report) is ztrans2.py.

--init CKPT: start the Transformer from a trained ztrans2 checkpoint; the new 40 km/s projection starts at zero, so
step 0 equals that checkpoint.  Without --init everything but the tokenizer starts from scratch (clean A/B).
"""
import os, sys, types
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ztrans2
from ztrans2 import ZTrans, topk_peaks, CW, N_Z, S_LO
from dualtok import NN, NPAD
from grids import N_OBS

C40 = 3 + 64                                 # input channels + first-stage channels, per 40 km/s bin


def encode40(tok, flux, ivar):
    """the dual tokenizer's encoder, also returning its 40 km/s maps: (B,1882,384), (B,118,768), (B,1882,4*67), s"""
    x, f, n, s = tok.prep(flux, ivar)
    x = F.pad(tok.to_log(x), (0, NPAD - N_OBS))                             # (B,3,7552)
    h = x
    for i, st in enumerate(tok.enc):
        h = st(h)
        if i == 0:
            h40 = torch.cat([x.to(h.dtype), h], 1)                          # (B,67,7552)
        if i == 2:
            hn = h.transpose(1, 2)
    hb = h.transpose(1, 2)
    B = x.shape[0]
    fine = h40[:, :, :4 * NN].reshape(B, C40, NN, 4).permute(0, 2, 3, 1).reshape(B, NN, 4 * C40)
    return hn[:, :NN], hb, fine, s


class ZTrans40(ZTrans):
    def __init__(self, d=768, **kw):
        super().__init__(d=d, **kw)
        self.f_norm, self.f_in = nn.LayerNorm(4 * C40), nn.Linear(4 * C40, d)
        nn.init.zeros_(self.f_in.weight); nn.init.zeros_(self.f_in.bias)
        # bound to the tokenizer under the name main() compiles (core.tok.encode)
        self.tok.encode = types.MethodType(encode40, self.tok)
        ini = os.environ.get('ZT40_INIT', '')
        if ini:
            sd = torch.load(ini, map_location='cpu', weights_only=False)['model']
            miss, unexp = self.load_state_dict(sd, strict=False)
            assert set(miss) == {'f_norm.weight', 'f_norm.bias', 'f_in.weight', 'f_in.bias'} and not unexp, (miss, unexp)
            print('init from %s (40 km/s projection starts at zero)' % ini, flush=True)

    def forward(self, flux, ivar):
        with torch.no_grad() if not any(p.requires_grad for p in self.tok.parameters()) else torch.enable_grad():
            hn, hb, fine, s = self.tok.encode(flux, ivar)
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
        return dict(coarse=lc, cand_cb=cb_, cand_cen=cen, pick=self.pick(q).squeeze(-1).float(),
                    off=self.offh(q).float(), logit_c=self.chead(g).float())


def selftest():
    torch.manual_seed(0)
    m = ZTrans40(d=64, H=32, L=1, Ldec=1, Kc=4)
    for p in m.tok.parameters():
        p.requires_grad_(False)
    fl = torch.randn(2, 7781) + 1; iv = torch.rand(2, 7781) + 0.5
    o = m(fl, iv)
    hn, hb, fine, s = m.tok.encode(fl, iv)
    print('fine', tuple(fine.shape), '| forward:', {k: tuple(v.shape) for k, v in o.items()})
    # the 4 bins of token t are bins 4t..4t+3 of the 40 km/s map
    x, f, n, s = m.tok.prep(fl, iv); x = F.pad(m.tok.to_log(x), (0, NPAD - N_OBS))
    t = 777; assert torch.allclose(fine[0, t].view(4, C40)[:, :3], x[0, :, 4 * t:4 * t + 4].T), 'token/bin alignment'
    zb = torch.tensor([3000, 6000]); c = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float)
    sum(ztrans2.losses(o, zb, c)).backward()
    print('f_in grad norm %.3e (zero-init weight still receives gradient)' % m.f_in.weight.grad.norm())
    print('trainable params without grad:', [k for k, p in m.named_parameters() if p.requires_grad and p.grad is None])
    print('alignment OK')


if __name__ == '__main__':
    if '--selftest40' in sys.argv:
        selftest(); sys.exit()
    if '--init' in sys.argv:
        i = sys.argv.index('--init'); os.environ['ZT40_INIT'] = sys.argv[i + 1]; del sys.argv[i:i + 2]
    ztrans2.ZTrans = ZTrans40
    ztrans2.main()
