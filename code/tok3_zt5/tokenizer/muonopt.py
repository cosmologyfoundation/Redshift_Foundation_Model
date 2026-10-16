"""Muon on hidden 2-D matrices, AdamW on everything else (owner 2026-10-13: 「把架構改成 Muon 用在 tokenizer and transformer上面」).

Same recipe as zfm/train.py (--muon, adopted 2026-09-23 on the old zfm; never ported to convtok until now):
  Muon(momentum 0.95, nesterov, weight decay 0.01, adjust_lr_fn='match_rms_adamw') -- the orthogonalised update is
  rescaled to AdamW's RMS, so one lr and one schedule shape serve both optimisers.
Hidden 2-D matrices only.  AdamW keeps: convolutions (3-D), norms, biases, layer scales, embeddings / positions /
learned tokens, the projections INTO the model (token / brightness / 40 km/s / FSQ in-out) and the output heads.
MultiOpt / MultiSched make a list of optimisers look like one, so the training loops and the resume state change only
in what they hold.
"""
import re
import torch

TRANSFORMER_HIDDEN = re.compile(r'^(enc|dec)\.\d+\.(qkv|o|q|kv|o1|o2|mlp\.(\d+|up|down))\.weight$'
                                r'|^sent\.blocks\.\d+\.(att\.in_proj_weight|att\.out_proj\.weight|mlp\.(\d+|up|down)\.weight)$')
CONVNEXT_HIDDEN = re.compile(r'^(enc|dec_n|dec_b)\..*\.(up|down)\.weight$')


def split(named_params, pattern):
    mu, aw, names = [], [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 2 and pattern.search(n):
            mu.append(p); names.append(n)
        else:
            aw.append(p)
    return mu, aw, names


def muon(params, lr):
    return torch.optim.Muon(params, lr=lr, weight_decay=0.01, momentum=0.95, nesterov=True, adjust_lr_fn='match_rms_adamw')


class MultiOpt:
    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        assert isinstance(sd, list) and len(sd) == len(self.opts), 'resume state was saved with a different optimiser set'
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)


class MultiSched:
    def __init__(self, scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def state_dict(self):
        return [s.state_dict() for s in self.scheds]

    def load_state_dict(self, sd):
        for s, x in zip(self.scheds, sd):
            s.load_state_dict(x)
