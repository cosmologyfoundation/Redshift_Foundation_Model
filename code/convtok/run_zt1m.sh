#!/bin/bash
# all-outputs Transformer, 1M train x up to 5 epochs (owner 2026-10-10: 先5輪; schedule still built for 6), early stop patience 2 on loss AND NMAD,
# held-out 200k from validation blocks 2200-4400; test 200k from blocks 0-2200 (owner: 測試 一樣20%);
# gbs 160 (40 x 4, fills the 40 GB A100s; owner: GPU RAM 要塞滿), lr 3e-4, torch.compile, dual tokenizer frozen, H32, original ZWARN=0, class mix 20%.
# Resumable: re-run with a new job id after a node expires.
J=$1
# optional 2nd arg: first re-evaluate per type the resumed epoch and the saved epoch-1 weights (owner: catastrophic 總體還有每個type)
EX=""; [ -n "$2" ] && EX="--eval_resumed --eval_ckpts $SCRATCH/zfm_ckpt/ZT2_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_ep1.pt"
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
TOK=$SCRATCH/zfm_ckpt/DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt
ST=$SCRATCH/zfm_ckpt/ZT2_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/zt1m.log
echo "START $(date +%H:%M:%S) job $J" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans2.py --tok $TOK --H 32 --size 1000000 --bs 40 --lr 3e-4 --epochs 6 --compile --patience 2 --stop_epoch 5 \
  --test_blocks 2200 --val_blocks 4400 --n_es 200000 --n_test 200000 --clean 0 --resume $ST $EX >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
