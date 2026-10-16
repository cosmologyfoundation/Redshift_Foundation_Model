#!/bin/bash
# keep the epoch-1 weights (best.pt right after EPOCH 1) for the per-type re-evaluation
B=$SCRATCH/zfm_ckpt/ZT2_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix
until grep -q "EPOCH 1 " ~/claude_learning_map/convtok/logs/zt1m.log; do sleep 20; done
sleep 30; cp ${B}_best.pt ${B}_ep1.pt; python3 -c "print(1)" >/dev/null; ls -la ${B}_ep1.pt
