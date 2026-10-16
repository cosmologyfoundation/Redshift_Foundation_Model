"""Can a tokenizer's output see a faint line?  Injection test, no labels (owner 2026-10-12: 「先做注入測試」).

Real spectra from the test pool (validation blocks 0-2200, labelfree, 20% mix).  Each spectrum is used twice:
as it is, and with ONE synthetic feature added at a random observed wavelength:
  [O II]  doublet 3727.09 / 3729.88 A (ratio 1 : 1.3), each Gaussian sigma 80 km/s (50 intrinsic + instrument)
  broad   one Gaussian, FWHM 4,000 km/s (a QSO broad line)
The amplitude is set from the spectrum's own ivar so the feature has a chosen OPTIMAL matched-filter S/N
(S = A * sqrt(sum p^2 ivar)), S in {2, 3, 5, 8}.  An ideal detector that knows the shape and the position reaches
TPR = Phi(S - 2.326) at 1% false positives -- the ceiling printed next to every result.
Probe: the tokenizer outputs the Transformer reads (pre-FSQ), at the injection position only
  [O II]: narrow tokens t-2..t+2 (5 x 384);  broad line: broad tokens j-1..j+1 (3 x 768)
-> standardised -> logistic regression trained on half of the spectra (both copies), scored on the other half.
Reported per tokenizer: AUC and TPR at FPR 1%.  Same spectra, positions and noise draws for every tokenizer.
usage: python inject_probe.py N CKPT_A [CKPT_B ...]
"""
import math, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from data import Spectra
from dualtok import DualTok, NN, NB
from grids import DLOG

C = 299792.458
LAM = np.arange(7781) * 0.8 + 3600.0


def load(n):
    ds = Spectra(1 << 20, seed=0, val=True, labelfree=True, clean=0, native=True, val_blocks_main=2200)
    F, V = [], []
    for fl, iv in DataLoader(ds, batch_size=256, num_workers=8):
        F.append(fl.numpy()); V.append(iv.numpy())
        if sum(len(x) for x in F) >= n:
            break
    return np.concatenate(F)[:n], np.maximum(np.concatenate(V)[:n], 0)


def inject(F, V, kind, S, rng):
    """returns injected flux and the centre wavelength per spectrum"""
    n = len(F); out = F.copy(); lc = np.zeros(n)
    for i in range(n):
        for _ in range(50):
            l0 = rng.uniform(3750, 9700) if kind == 'oii' else rng.uniform(3900, 9550)
            if kind == 'oii':
                sig = l0 * 80 / C
                p = (np.exp(-0.5 * ((LAM - l0) / sig) ** 2)
                     + 1.3 * np.exp(-0.5 * ((LAM - l0 * 3729.875 / 3727.092) / sig) ** 2))
            else:
                sig = l0 * (4000 / 2.3548) / C
                p = np.exp(-0.5 * ((LAM - l0) / sig) ** 2)
            q = (p * p * V[i]).sum()
            win = p > 0.05 * p.max()
            if q > 0 and (V[i][win] > 0).mean() > 0.8:
                break
        out[i] = F[i] + S / math.sqrt(q) * p
        lc[i] = l0
    return out, lc


@torch.no_grad()
def feats(tok, F, V, lc, kind, dev):
    X = []
    for a in range(0, len(F), 128):
        fl = torch.from_numpy(F[a:a + 128]).float().to(dev); iv = torch.from_numpy(V[a:a + 128]).float().to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hn, hb, *_ = tok.encode(fl, iv)
        ilog = np.log(lc[a:a + 128] / 3600.0) / DLOG
        if kind == 'oii':
            t = np.clip(np.floor(ilog / 4).astype(int), 2, NN - 3)
            idx = torch.from_numpy(t[:, None] + np.arange(-2, 3)).to(dev)
            g = hn.float().gather(1, idx.unsqueeze(-1).expand(-1, -1, hn.shape[-1]))
        else:
            j = np.clip(np.floor(ilog / 64).astype(int), 1, NB - 2)
            idx = torch.from_numpy(j[:, None] + np.arange(-1, 2)).to(dev)
            g = hb.float().gather(1, idx.unsqueeze(-1).expand(-1, -1, hb.shape[-1]))
        X.append(g.flatten(1).cpu())
    return torch.cat(X)


def probe(X0, X1, dev, seed=0):
    """logistic regression, trained on the first half of the spectra, scored on the second half"""
    n = len(X0); h = n // 2
    Xtr = torch.cat([X0[:h], X1[:h]]); ytr = torch.cat([torch.zeros(h), torch.ones(h)])
    Xte = torch.cat([X0[h:], X1[h:]]); yte = torch.cat([torch.zeros(n - h), torch.ones(n - h)])
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev)
    torch.manual_seed(seed)
    w = torch.zeros(Xtr.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2)
    ytr_d = ytr.to(dev)
    for _ in range(400):
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr_d) + 1e-3 * w.pow(2).sum()
        l.backward(); opt.step()
    s = (Xte @ w + b).detach().cpu().numpy(); y = yte.numpy()
    neg, pos = s[y == 0], s[y == 1]
    auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    thr = np.quantile(neg, 0.99)
    return float(auc), float((pos > thr).mean())


def main():
    n = int(sys.argv[1]); cks = sys.argv[2:]
    dev = 'cuda'
    F, V = load(n)
    print('spectra %d (test pool, labelfree)' % len(F), flush=True)
    toks = []
    for ck in cks:
        t = DualTok(lv_n=(5, 5)).to(dev).eval()
        t.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
        toks.append((os.path.basename(ck).split('_')[0] + '_' + os.path.basename(ck).split('_')[1], t))
    print('%-6s %4s  %-11s' % ('kind', 'S', 'ideal TPR') + ''.join('  %-26s' % ('%s AUC / TPR@1%%' % nm) for nm, _ in toks))
    for kind in ['oii', 'broad']:
        for S in [2, 3, 5, 8]:
            rng = np.random.default_rng(1000 * S + (0 if kind == 'oii' else 1))
            Fi, lc = inject(F, V, kind, S, rng)
            ideal = 0.5 * (1 + math.erf((S - 2.326) / math.sqrt(2)))
            row = '%-6s %4d  %-11.3f' % (kind, S, ideal)
            for nm, t in toks:
                X0 = feats(t, F, V, lc, kind, dev); X1 = feats(t, Fi, V, lc, kind, dev)
                auc, tpr = probe(X0, X1, dev)
                row += '  %-26s' % ('%.3f / %.3f' % (auc, tpr))
            print(row, flush=True)


if __name__ == '__main__':
    main()
