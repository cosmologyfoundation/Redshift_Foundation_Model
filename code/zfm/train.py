"""Train one point of the scaling grid.

Loss (DESIGN.md 4), in nats per spectrum:

  z      dense cross-entropy over all N_Z bins of log P(z), the marginal of
         S(class, z) over class.  This is THE number the scaling law is fitted
         on, and it is directly comparable to the uniform-posterior cost
         log(N_Z) = 9.404 nats and to the E >= 1.07 floor of DESIGN.md 9.5.
  class  multi-label BCE against the target bitmask.  13.6% of objects carry
         more than one class bit, so a softmax here would teach the head that
         an equally valid answer is wrong.
  rank   cross-entropy over the K reranked candidates, target = the candidate
         nearest the true z.  85-89% of errors live here.

Usage:
  python train.py --size 1000000 --d_rr 960 --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from grids import N_Z, CLASSES, bin_to_z
from data import Spectra
from model import ZFM, n_params
import alias as AL
import evidence as EV


def ddp_setup():
    """Return (rank, world, local_rank).  Works under torchrun and under srun.

    Four A100s with one RUN per GPU was task parallelism: it made the grid of
    eight points four times faster and every single point exactly as slow.
    bench.py settles what to do about that -- the data pipeline delivers
    10,423 spectra/s and the step consumes 72.5, so the run is GPU bound and
    DDP scales it nearly linearly.  Memory forbids the other route: 28.6 GB of
    a 40 GB card at bs=32, and bs=64 OOMs."""
    if 'RANK' in os.environ:
        r, w, l = (int(os.environ['RANK']), int(os.environ['WORLD_SIZE']),
                   int(os.environ.get('LOCAL_RANK', 0)))
    elif 'SLURM_PROCID' in os.environ:
        r, w, l = (int(os.environ['SLURM_PROCID']),
                   int(os.environ['SLURM_NTASKS']),
                   int(os.environ.get('SLURM_LOCALID', 0)))
        # NOT SLURM_LAUNCH_NODE_IPADDR: srun issued from a login node sets
        # that to the LOGIN node, while rank 0 runs on the compute node, so
        # the NCCL rendezvous never completes.  One node -> loopback.
        if int(os.environ.get('SLURM_NNODES', '1')) == 1:
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        else:
            import subprocess
            host = subprocess.check_output(
                ['scontrol', 'show', 'hostnames',
                 os.environ['SLURM_NODELIST']]).decode().split()[0]
            os.environ.setdefault('MASTER_ADDR', host)
        os.environ.setdefault('MASTER_PORT', '29517')
        os.environ['RANK'], os.environ['WORLD_SIZE'] = str(r), str(w)
    else:
        return 0, 1, 0
    # srun --gpus-per-task=1 gives each task its OWN CUDA_VISIBLE_DEVICES with
    # exactly one device, so every task must use ordinal 0; a full-node
    # allocation sees all four and wants rank % 4.  Taking the modulus covers
    # both without the launcher having to say which one it is.
    n = max(torch.cuda.device_count(), 1)
    l = l % n
    if w > 1:
        dist.init_process_group('nccl', rank=r, world_size=w)
    return r, w, l


def class_ce(logit, c, smooth=0.05):
    """Softmax CE against the target bits normalised to sum 1 (DESIGN.md 2).

    The bits are a bitmask (13.6% carry more than one): a multi-bit object
    splits its mass.  Rows with no class bit (2.8%) carry no class loss.
    Smoothing 0.05 absorbs the ~6% of target bits that are wrong.  Target
    bits, never SPECTYPE.
    """
    n = c.sum(1)
    keep = n > 0
    if not keep.any():
        return logit.sum() * 0
    return F.cross_entropy(logit[keep], c[keep] / n[keep].unsqueeze(1), label_smoothing=smooth)


def mix_nll(logp, tgt, eps):
    """-log[(1-eps) p(tgt) + eps/N] averaged over the batch (DESIGN.md 6e item 2).

    About 1% of ZWARN=0 labels are wrong.  Plain cross-entropy pulls toward a
    wrong label harder the more confidently the model disagrees with it; with
    the mixture, once p(label) falls well below eps/N the gradient toward the
    label vanishes, so a confident different answer is no longer pushed back.
    eps = 0 is F.nll_loss exactly.  N is the number of outcomes: N_Z bins for
    the surface, the candidate count for the reranker.
    """
    if eps <= 0:
        return F.nll_loss(logp, tgt)
    lp = logp.gather(1, tgt.view(-1, 1)).squeeze(1)
    floor = math.log(eps / logp.shape[1])
    return -torch.logaddexp(lp + math.log1p(-eps), torch.full_like(lp, floor)).mean()


def evaluate(net, loader, dev, n_val=50000, world=1):
    """n_val = 50,000 is a power calculation, not a round number.

    At 2,048 spectra the loss standard error is ~0.044 nats.  DESIGN.md
    9.4 measured that moving ONE ladder point by +-0.05 nats moves the
    fitted floor E from 1.14 to 1.90 -- so a 2,048-spectrum validation
    set would make the entire scaling law noise.  50,000 puts the SE
    near 0.009 nats, an order of magnitude below the effect fitted.
    """
    net.eval()
    Lz, Lc, Lr, n = [], [], [], 0
    dz, hit5, hitany = [], [], []
    with torch.no_grad():
        for i, (x, zb, c, z) in enumerate(loader):
            if n >= -(-n_val // world):
                break
            x, zb, c = x.to(dev), zb.to(dev), c.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(x)
            lp = o['logp_z'].float()
            Lz.append(float(F.nll_loss(lp, zb)))
            Lc.append(float(F.binary_cross_entropy_with_logits(
                o['logit_c'].float(), c)))
            cand_z, cand_l = o['cand_z'], o['cand_logit'].float()
            tgt = (cand_z - zb.view(-1, 1)).abs().argmin(1)
            Lr.append(float(F.cross_entropy(cand_l, tgt)))
            best = cand_l.argmax(1)
            zhat = torch.tensor(bin_to_z(cand_z.gather(
                1, best.view(-1, 1)).squeeze(1).cpu().numpy()))
            zt = z.float()
            dz.append(((zhat - zt) / (1 + zt)).numpy())
            pc = o['logit_c'].float().argmax(1).cpu()
            hit5.append((F.one_hot(pc, len(CLASSES)).float() * c.cpu()).sum(1).numpy())
            n += len(x)
    if world > 1:
        # pool the RAW per-object arrays, not per-rank means: nmad is a median
        # and catastrophic is a tail fraction, and neither one averages.
        buf = [None] * world
        dist.all_gather_object(buf, (np.concatenate(dz), np.concatenate(hit5),
                                     Lz, Lc, Lr, n))
        dz = [b[0] for b in buf]
        hit5 = [b[1] for b in buf]
        Lz = sum((b[2] for b in buf), [])
        Lc = sum((b[3] for b in buf), [])
        Lr = sum((b[4] for b in buf), [])
        n = sum(b[5] for b in buf)
    d = np.concatenate(dz)
    nmad = 1.4826 * np.median(np.abs(d - np.median(d)))
    return dict(val_loss_z=float(np.mean(Lz)), val_loss_c=float(np.mean(Lc)),
                val_loss_rank=float(np.mean(Lr)),
                nmad=float(nmad),
                catastrophic=float((np.abs(d) > 0.1).mean()),
                within_0p003=float((np.abs(d) < 0.003).mean()),
                class_any_bit=float(np.concatenate(hit5).mean()), n_val=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--d_rr', type=int, default=960)
    ap.add_argument('--rr_layers', type=int, default=8)
    ap.add_argument('--d_enc', type=int, default=384)
    ap.add_argument('--enc_layers', type=int, default=12)
    ap.add_argument('--n_per', type=int, default=8)
    ap.add_argument('--K', type=int, default=256)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w_class', type=float, default=0.1)
    ap.add_argument('--w_rank', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--norm_corr', action='store_true')
    ap.add_argument('--rr_final_norm', action='store_true')
    ap.add_argument('--sep', type=int, default=64)
    ap.add_argument('--zmax', type=float, default=None,
                    help='train only on label z < zmax (owner decision 2026-09-22); validation is never filtered')
    ap.add_argument('--deep_frac', type=float, default=0.0,
                    help='share of training examples drawn from the SV3 deep tail (DESIGN.md 7b-2)')
    ap.add_argument('--degrade', action='store_true', help='degrade deep-tail draws to main depth')
    ap.add_argument('--teacher', default=None, help='npz of teacher labels for the deep tail')
    ap.add_argument('--recall', action='store_true',
                    help="the truth must beat the K-th peak's score, i.e. make the candidate list")
    ap.add_argument('--w_recall', type=float, default=0.5)
    ap.add_argument('--recall_hard', action='store_true',
                    help='average the recall hinge over VIOLATORS only, not over all objects')
    ap.add_argument('--sky', type=float, default=None,
                    help='sky-mask ratio for the evidence maps (DESIGN.md 6e); default: unmasked')
    ap.add_argument('--eps', type=float, default=0.0,
                    help='epsilon-mixture likelihood for the z and rank losses (DESIGN.md 6e item 2): '
                         '-log[(1-eps) p(label) + eps/N]; 0 = plain cross-entropy')
    ap.add_argument('--clean', type=int, nargs='?', const=1, default=0,
                    help='train on the cleaned copies (DESIGN.md 6d-2): --clean = v1 (rule A + DELTACHI2<25), '
                         '--clean 2 = v2 (v1 + rrwrong P_WRONG>0.01)')
    ap.add_argument('--init_enc', default=None,
                    help='encoder weights from pretrain.py (DESIGN.md 7a); everything else starts fresh')
    ap.add_argument('--swiglu_rr', action='store_true', help='SwiGLU FFN in the reranker (parameter-matched)')
    ap.add_argument('--swiglu_enc', action='store_true', help='SwiGLU MLP in the ConvNeXt blocks (parameter-matched)')
    ap.add_argument('--muon', action='store_true',
                    help='Muon on the hidden 2-D matrices (encoder MLPs, reranker attention/FFN), AdamW on the rest')
    ap.add_argument('--muon_lr', type=float, default=None,
                    help='Muon peak lr (match_rms_adamw scaling); default: --lr')
    ap.add_argument('--class_head', action='store_true',
                    help='learned class head + softmax CE on normalised target bits (DESIGN.md 2)')
    ap.add_argument('--softclip', action='store_true',
                    help='encoder input CLIP*asinh(x/CLIP) instead of the +-50 clamp (DESIGN.md 2)')
    ap.add_argument('--shape', action='store_true',
                    help='per-candidate rest-frame continuum shape (evidence.shape_features)')
    ap.add_argument('--pairs', action='store_true',
                    help='partner-line consistency features for the reranker (evidence.pair_features)')
    ap.add_argument('--alias', action='store_true',
                    help='cross-class alias negatives (alias.py; DESIGN.md 4.2)')
    ap.add_argument('--w_alias', type=float, default=0.5, help='weight of the surface margin term')
    ap.add_argument('--evidence', action='store_true',
                    help='feed evidence.py features to the reranker (DESIGN.md 5)')
    ap.add_argument('--val_blocks_main', type=int, default=400,
                    help='validation = the first N main blocks (256 rows) of the fixed seed-777 permutation, sv3 in'
                         ' proportion; training never draws from them. 400 ~ 108k rows; the first 400 are kept for any N')
    ap.add_argument('--n_val', type=int, default=50000, help='spectra scored by the end-of-run validation')
    ap.add_argument('--clean_val', type=int, nargs='?', const=1, default=0,
                    help='validation rows from the cleaned copies: --clean_val = v1, --clean_val 2 = v2 (default: unfiltered)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=os.path.expanduser('~/claude_learning_map/zfm/results'))
    a = ap.parse_args()
    rank, world, local = ddp_setup()
    is0 = rank == 0
    if is0:
        os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.cuda.set_device(local)
    dev = 'cuda:%d' % local

    net = ZFM(a.d_enc, a.enc_layers, 8, a.n_per, a.d_rr, a.rr_layers, a.K,
          norm_corr=a.norm_corr, rr_final_norm=a.rr_final_norm,
          sep=a.sep, evidence=a.evidence, shape=a.shape, sky=a.sky,
          softclip=a.softclip, class_head=a.class_head,
          swiglu_enc=a.swiglu_enc, swiglu_rr=a.swiglu_rr, pairs=a.pairs).to(dev)
    core = net
    if a.init_enc:
        pe = torch.load(a.init_enc, map_location='cpu', weights_only=False)
        core.enc.load_state_dict(pe['enc'], strict=True)
        if is0:
            print('encoder initialised from %s (pre-trained on %s spectra, label-free)'
                  % (a.init_enc, pe.get('cfg', {}).get('size')), flush=True)
    if world > 1:
        net = DDP(net, device_ids=[local])
    npar = dict(enc=n_params(core.enc), surf=n_params(core.surf),
                rr=n_params(core.rr), total=n_params(core))
    # Muon only for hidden 2-D matrices: the ConvNeXt MLPs and the reranker's
    # attention/FFN.  Everything else stays on AdamW -- the archetype bank
    # (3-D, each archetype is a spectrum, not a map between feature spaces),
    # convolutions, norms, biases, layer scales, the reranker's input/output
    # layers and the class head.  match_rms_adamw scales Muon's orthogonal
    # update to AdamW's RMS, so the same lr and schedule apply to both.
    mu = [p for n, p in core.named_parameters()
          if a.muon and p.ndim == 2 and (n.startswith('enc.blocks.') or n.startswith('rr.tf.'))]
    mu_ids = {id(p) for p in mu}
    aw = [p for p in core.parameters() if id(p) not in mu_ids]
    opt = torch.optim.AdamW(aw, lr=a.lr, weight_decay=0.01)
    opts = [opt]
    if mu:
        opts.append(torch.optim.Muon(mu, lr=a.muon_lr or a.lr, weight_decay=0.01,
                                     momentum=0.95, nesterov=True, adjust_lr_fn='match_rms_adamw'))
        if is0:
            print('Muon on %d matrices (%.2fM params), AdamW on %d tensors (%.2fM)' % (
                len(mu), sum(p.numel() for p in mu) / 1e6, len(aw), sum(p.numel() for p in aw) / 1e6), flush=True)
    # GLOBAL batch, not per-GPU: this is the number that has to be equal
    # across every point of the grid or L(D) has two variables moving.
    gbs = a.bs * world
    steps = max(a.size // gbs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps,
                                                pct_start=0.05)
    scheds = [sched]
    if len(opts) > 1:
        # same one-cycle shape; momentum not cycled (Muon's 0.95 is its own)
        scheds.append(torch.optim.lr_scheduler.OneCycleLR(opts[1], a.muon_lr or a.lr, total_steps=steps,
                                                          pct_start=0.05, cycle_momentum=False))
    deep = (dict(frac=a.deep_frac, degrade=a.degrade, teacher=a.teacher)
            if a.deep_frac > 0 else None)
    tr = DataLoader(Spectra(a.size, seed=a.seed, rank=rank, world=world, zmax=a.zmax, deep=deep, clean=a.clean,
                            val_blocks_main=a.val_blocks_main),
                    batch_size=a.bs,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    va = DataLoader(Spectra(1 << 20, seed=0, val=True, rank=rank, world=world,
                            val_blocks_main=a.val_blocks_main, clean=a.clean_val), batch_size=a.bs,
                    num_workers=max(a.workers // 2, 1), pin_memory=True,
                    drop_last=True)
    if is0:
        from design_status import report
        print(report(), flush=True)
        print('size=%d  d_rr=%d L=%d  params enc %.2fM surf %.2fM rr %.2fM total %.2fM'
              ' steps=%d seed=%d' % (a.size, a.d_rr, a.rr_layers,
                                     npar['enc'] / 1e6,
                                     npar['surf'] / 1e6, npar['rr'] / 1e6,
                                     npar['total'] / 1e6, steps, a.seed),
              'gbs=%d (%d/gpu x %d)' % (gbs, a.bs, world), flush=True)
    t0 = time.time()
    net.train()
    run = []
    it = 0
    for x, zb, c, ztrue in tr:
        if it >= steps:
            break
        x, zb, c = x.to(dev, non_blocking=True), zb.to(dev), c.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            extra = (AL.alias_bins(EV.spectrum_maps(x), zb, ztrue.to(dev))
                     if a.alias else None)
            o = net(x, extra=extra)
            lz = mix_nll(o['logp_z'].float(), zb, a.eps)
            lc = (class_ce(o['logit_c'].float(), c) if a.class_head
                  else F.binary_cross_entropy_with_logits(o['logit_c'].float(), c))
            tgt = (o['cand_z'] - zb.view(-1, 1)).abs().argmin(1)
            lr_ = mix_nll(F.log_softmax(o['cand_logit'].float(), -1), tgt, a.eps)
            loss = lz + a.w_class * lc + a.w_rank * lr_
            la = (AL.surface_margin(o['logp_z'].float(), zb, *extra)
                  if a.alias else torch.zeros((), device=dev))
            loss = loss + a.w_alias * la
            # RECALL (DESIGN.md 3): 26-31% of catastrophics are objects whose
            # true z the surface never proposed, and no reranker can recover
            # those.  topk_peaks takes peaks in descending order, so the K-th
            # one's score IS the admission threshold; require the truth to beat
            # it by a margin.  Unlike the dense z loss, which pushes the truth
            # up against the whole grid, this pushes it past exactly the
            # quantity that decides whether the reranker ever sees it.
            if a.recall:
                S = o['S'].float()
                b_ = torch.arange(len(zb), device=dev)
                thr = S[b_, o['cand_class'][:, a.K - 1], o['cand_z'][:, a.K - 1]]
                truth = S[b_, :, zb].max(1).values
                v = 1.0 + thr - truth
                if a.recall_hard:
                    # A mean over objects is the wrong reduction for a term that
                    # exists for a tail: 98% of objects already clear the K-th
                    # peak, so the first version read 0.002 and its gradient was
                    # negligible while the 2% it targets carried no weight
                    # (300k A/B: 143 saved / 147 broken, p = 0.86).  Averaging
                    # over the violators alone keeps the per-violator gradient
                    # at full size however rare they are.
                    hit = v > 0
                    lrec = (F.relu(v) * hit).sum() / hit.sum().clamp_min(1)
                    frac = hit.float().mean()
                else:
                    lrec = F.softplus(v).mean()
                    frac = (v > 0).float().mean()
                loss = loss + a.w_recall * lrec
            else:
                lrec = frac = torch.zeros((), device=dev)
        for o_ in opts:
            o_.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        for o_ in opts:
            o_.step()
        for s_ in scheds:
            s_.step()
        run.append([float(lz), float(lc), float(lr_), float(lrec),
                    float(extra[1].float().sum(1).mean()) if a.alias else float(frac)])
        if it == 0:
            # Pre-registered init values.  A uniform posterior over N_Z is
            # ln(12140)=9.404 and a uniform choice among K is ln(256)=5.545.
            # The first run started at z 18.71 because the surface was an
            # UNNORMALISED dot product fed to a softmax: NLL = ln(N_Z)+s^2/2
            # with s=4.15 measured on real spectra.  That startup penalty is
            # a larger fraction of a short run than a long one, so it biases
            # beta upward in L(D) -- it must not be allowed to pass silently.
            # Range MEASURED, not guessed (initspread.py, 2026-09-15): 10 seeds
            # x 4 rank-0 batches of 8 gave mean 9.5035, sd 0.197, range
            # 9.055-9.829.  The first version, [9.3, 9.9], came from ONE seed's
            # calibration value and refused seed 43 (9.2717) and would have
            # refused seed 47 -- a guard that rejected 2 of 10 healthy starts.
            # mean +- 6 sd = [8.3, 10.7] still sits ~40 sd below the 18.7072 it
            # exists to catch.
            lo, hi = (8.3, 10.7) if a.norm_corr else (9.3, 1e9)
            assert lo < float(lz) < hi, (
                'init z loss %.4f outside [%.1f, %.1f]: the surface is not '
                'starting near-uniform, so NORM_GAIN no longer fits this '
                'architecture -- recalibrate with calib.py' % (float(lz), lo, hi))
            if a.class_head:
                # zero-init last layer: uniform over 5 classes, so ln 5 exactly
                assert abs(float(lc) - math.log(5)) < 0.02, (
                    'init class loss %.4f != ln 5 = 1.609: the class head is not '
                    'starting uniform' % float(lc))
            assert 5.2 < float(lr_) < 6.0, (
                'init rank loss %.4f != ln(K)=5.545: top-K is not returning '
                'K candidates' % float(lr_))
        if it >= 2000 and it % 200 == 0:
            # Abort a diverging run instead of finishing it.  The 1M point's
            # rank loss went 2.1 -> 11.6 -> 31.7 over steps 16k-18k and the run
            # spent its last 75 minutes training nothing useful, because the
            # rank gradient swamped the global clip and z stopped learning.
            # 2*ln(K) is twice the cost of a uniform guess over the K
            # candidates -- no working reranker is ever there.  Raising gives
            # a non-zero exit, so a ladder loop written with '|| break' stops
            # too.  Checked on every rank; torchrun tears the group down.
            mr = float(np.mean([r[2] for r in run[-200:]]))
            if not np.isfinite(mr) or mr > 2 * np.log(a.K):
                raise RuntimeError(
                    'rank loss diverged at step %d: 200-step mean %.4g > '
                    '2 ln K = %.3f' % (it, mr, 2 * np.log(a.K)))
        if it % 200 == 0 and is0:
            m = np.mean(run[-200:], 0)
            print('  step %6d/%d  z %.4f  class %.4f  rank %.4f  recall/alias %.4f (viol %.3f)  %5.0fs'
                  % (it, steps, m[0], m[1], m[2], m[3], m[4], time.time() - t0), flush=True)
        it += 1
    res = dict(size=a.size, d_rr=a.d_rr, rr_layers=a.rr_layers, seed=a.seed,
               norm_corr=a.norm_corr, rr_final_norm=a.rr_final_norm, sep=a.sep,
               evidence=a.evidence, shape=a.shape, pairs=a.pairs, sky=a.sky, alias=a.alias, w_alias=a.w_alias,
               recall=a.recall, w_recall=a.w_recall, recall_hard=a.recall_hard, zmax=a.zmax,
               deep_frac=a.deep_frac, degrade=a.degrade, teacher=a.teacher,
               params=npar, steps=steps, seconds=time.time() - t0,
               bs_per_gpu=a.bs, global_bs=gbs, world=world, clean=a.clean, clean_val=a.clean_val,
               val_blocks_main=a.val_blocks_main,
               train_z=float(np.mean(run[-200:], 0)[0]))
    res.update(evaluate(net, va, dev, n_val=a.n_val, world=world))
    if is0:
        print(json.dumps(res, indent=1), flush=True)
        stem = '%s%d_d%d_s%d%s' % (a.tag, a.size, a.d_rr, a.seed,
                                   ('_nc' if a.norm_corr else '')
                                   + ('_fn' if a.rr_final_norm else '')
                                   + ('' if a.sep == 64 else '_sep%d' % a.sep)
                                   + ('_ev' if a.evidence else '')
                                   + ('_sh' if a.shape else '') + ('_pr' if a.pairs else '')
                                   + ('_sky%g' % a.sky if a.sky else '')
                                   + ('_sc' if a.softclip else '')
                                   + ('_eps%g' % a.eps if a.eps else '')
                                   + ('_ch' if a.class_head else '')
                                   + (('_mu' + ('%g' % a.muon_lr if a.muon_lr else '')) if a.muon else '')
                                   + ('_swe' if a.swiglu_enc else '') + ('_swr' if a.swiglu_rr else '')
                                   + ('_pt' if a.init_enc else '') + (('_cl' if a.clean == 1 else '_cl%d' % a.clean) if a.clean else '')
                                   + ('_zmax%g' % a.zmax if a.zmax else '')
                                   + ('_deep%g' % a.deep_frac if a.deep_frac else '')
                                   + ('_dg' if a.degrade else '') + ('_tch' if a.teacher else '')
                                   + (('_rch' if a.recall_hard else '_rc') if a.recall else '')
                                   + ('_al' if a.alias else '')
                                   + ('' if a.val_blocks_main == 400 else '_vb%d' % a.val_blocks_main)
                                   + ('_cv%d' % a.clean_val if a.clean_val else ''))
        # closed explicitly: the process ends with os._exit below, which does
        # NOT flush open file buffers, so an unclosed handle could leave an
        # empty result file behind a run that otherwise succeeded.
        with open(os.path.join(a.out, stem + '.json'), 'w') as fh:
            json.dump(res, fh, indent=1)
        # Save the WEIGHTS too.  The first three ladder points were run without
        # this and every trained model was discarded at exit: the losses were
        # kept and the thing that produced them was not, so nothing could be
        # plotted, inspected, or asked a second question.  A run that cannot be
        # looked at afterwards has to be repeated to be looked at.
        ck = os.environ.get('SCRATCH', os.path.expanduser('~')) + '/zfm_ckpt'
        os.makedirs(ck, exist_ok=True)
        torch.save({'model': core.state_dict(),
                    'cfg': dict(d_enc=a.d_enc, enc_layers=a.enc_layers,
                                c_corr=8, n_per=a.n_per, d_rr=a.d_rr,
                                rr_layers=a.rr_layers, K=a.K,
                                norm_corr=a.norm_corr,
                                rr_final_norm=a.rr_final_norm, sep=a.sep,
                                evidence=a.evidence, shape=a.shape, sky=a.sky, alias=a.alias,
                                softclip=a.softclip, eps=a.eps, class_head=a.class_head,
                                muon=a.muon, muon_lr=a.muon_lr,
                                swiglu_enc=a.swiglu_enc, swiglu_rr=a.swiglu_rr,
                                init_enc=a.init_enc, clean=a.clean, pairs=a.pairs),
                    'res': res},
                   os.path.join(ck, stem + '.pt'))
        print('checkpoint -> %s' % os.path.join(ck, stem + '.pt'), flush=True)
    if world > 1:
        dist.destroy_process_group()
        # Leave WITHOUT interpreter teardown.  The 1M run wrote its result and
        # checkpoint and then all four ranks hung on idle GPUs holding 9 GB
        # each.  gdb on rank 0: gc_collect_main -> ~ProcessGroupNCCL ->
        # HeartbeatMonitor::join, waiting on a heartbeat thread that never
        # returned.  MOST LIKELY CAUSE, found afterwards in the terminal: that
        # torchrun had been suspended with Ctrl+Z right after launch.  The
        # ranks run in their own session and kept training, but the rendezvous
        # store lives in the torchrun process, so teardown waited on a server
        # that was frozen.  Earlier 4-rank runs, never suspended, exited
        # normally.  An earlier version of this note called it an intermittent
        # PyTorch deadlock; that was not established.  os._exit stays as a
        # guard: by this line every output is closed and on disk, so skipping
        # interpreter teardown costs nothing and cannot leave a finished run
        # holding the node.  Do not Ctrl+Z a torchrun.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
