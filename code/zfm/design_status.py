"""DESIGN.md "Implementation status", in code, so it cannot drift silently.

train.py prints every item that is not built, at startup, on every run.  The
reader's lessons sat in DESIGN.md for weeks while the trained reranker saw only
the class, z and nine numbers of S; a table nobody is shown is prose, and prose
corrections recurred 3x and 11x where enforced ones recurred 0x.
Update this and the DESIGN.md table in the same edit.
"""
STATUS = [
    # (section, item, status)
    ('3',    'joint class x z surface, class never gates',                      'built'),
    ('2',    'learned class head, softmax CE on target bits (--class_head)',    'built'),
    ('2',    'Muon on hidden 2-D matrices, AdamW rest (--muon --muon_lr 1e-3)', 'built'),
    ('2',    'SwiGLU in reranker / encoder (A/B 150k: no effect, not adopted)', 'BUILT'),
    ('3',    'ConvNeXt encoder, no downsampling',                               'built'),
    ('4.6',  'normalised surface (--norm_corr)',                                'built'),
    ('3',    'reranker final LayerNorm (--rr_final_norm)',                      'built'),
    ('3,5',  'reranker reads evidence at each candidate z (--evidence)',        'built'),
    ('4.1',  'dense z loss',                                                    'built'),
    ('4.5',  'multi-label BCE on target bits',                                  'built'),
    ('4.2',  'cross-class alias negatives (--alias)',                           'built'),
    ('3',    'surface recall loss: truth must beat the K-th peak (--recall)',  'built'),
    ('4.3',  'absent-line / unexplained-feature penalty as a loss',             'NOT BUILT'),
    ('4.4',  'bounded epsilon-mixture z loss (A/B 150k: no effect, eps=0)',       'BUILT'),
    ('6e',   'proven-wrong labels removed from the z loss',                     'NOT BUILT'),
    ('3',    'reader-asserted positions as a prior on S',                       'NOT BUILT'),
    ('5',    'per-candidate continuum shape (--shape)',                          'built'),
    ('5',    'confidence outputs: floor, posterior width, corroboration',       'NOT BUILT'),
    ('6e',   'sky mask in evidence maps: built for the flag, NOT yet in training', 'PARTIAL'),
    ('v10',  'class head sees brightness',                                      'NOT BUILT'),
    ('6e',   'detection output P_model(z_Redrock) with a reason',               'NOT BUILT'),
    ('6',    'eval refuses a headline without the stratification table',        'NOT BUILT'),
    ('2',    'encoder input without the +-50 clip (A/B: no effect, clamp kept)',    'BUILT'),
    ('7a',   'label-free pre-training: masked recon built (pretrain.py); shift needs 2 exposures', 'BUILT'),
    ('7b-2', 'self-distillation from depth (SV3 degraded to main depth)',       'NOT BUILT'),
    ('7b-3', 'confidence calibration on a held-out set',                        'NOT BUILT'),
    ('7c',   'fine-tuning with a frozen encoder',                               'NOT BUILT'),
]


def report():
    todo = [s for s in STATUS if s[2] != 'built']
    lines = [f'DESIGN.md: {len(todo)} of {len(STATUS)} design items are not built:']
    lines += [f'  sec {sec:<5} {st:<10} {item}' for sec, item, st in todo]
    return '\n'.join(lines)
