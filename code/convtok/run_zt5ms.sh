#!/bin/bash
# ZT5MS: ZT5 architecture with SwiGLU MLPs + Muon (hidden matrices, lr 1e-3; AdamW lr 3e-4 on the rest), on the
# Muon+SwiGLU tokenizer DUALTOKMS (frozen).  Owner 2026-10-13: 「Muon 的學習率要用 1e-3  QSO 45%、ELG 45%、其他三類各 3.33%，1M × 6 輪」.
# From scratch, 1M x 6 epochs, early stop patience 2; held-out 200k (blocks 2200-4400), test 200k (blocks 0-2200), 20% each.
# usage: run_zt5ms.sh JOBID   (resumable)
J=$1; BS=40
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
TOK=$SCRATCH/zfm_ckpt/DUALTOKMS_1000000_n55_cl0_mix_e10_bs1024_ep10.pt
ST=$SCRATCH/zfm_ckpt/ZT5MS_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/zt5ms.log
echo "START $(date +%H:%M:%S) job $J" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans5.py --tag ZT5MS --tok $TOK --H 32 --size 1000000 --bs $BS --lr 3e-4 --epochs 6 \
  --swiglu --muon --muon_lr 1e-3 --mix_train QSO:0.45,ELG:0.45,LRG:0.0333,BGS:0.0333,STAR:0.0334 --compile --patience 2 \
  --test_blocks 2200 --val_blocks 4400 --n_es 200000 --n_test 200000 --clean 0 --resume $ST >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
