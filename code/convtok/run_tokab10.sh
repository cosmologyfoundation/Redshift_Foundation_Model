#!/bin/bash
# Tokenizer A/B at convergence (owner 2026-10-12: 「300k 10輪 開始」): ZT5 architecture, frozen tokenizer v2 or v3,
# 300k x up to 10 epochs (early stop patience 2), training mix QSO 45 / ELG 45 / others 3.33 (owner: 「不要每類20%」), gbs 160, lr 3e-4; held-out / test 60k each.
# usage: run_tokab10.sh ARM(v2|v3) JOBID     (resumable from the per-epoch state file)
ARM=$1; J=$2
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
case $ARM in
  v2) TOK=$SCRATCH/zfm_ckpt/DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt ;;
  v3) TOK=$SCRATCH/zfm_ckpt/DUALTOK3_1000000_n55_cl0_q45e45_e5_bs1024_ep5.pt ;;
esac
ST=$SCRATCH/zfm_ckpt/ZT5tok10m45${ARM}_frozen_H32_L12_300000_e10_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/tokab10m45_${ARM}.log
echo "START $(date +%H:%M:%S) job $J arm $ARM" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans5.py --tag ZT5tok10m45$ARM --tok $TOK --H 32 --size 300000 --bs 40 --lr 3e-4 --epochs 10 --mix_train QSO:0.45,ELG:0.45,LRG:0.0333,BGS:0.0333,STAR:0.0334 --compile --patience 2 \
  --test_blocks 2200 --val_blocks 4400 --n_es 60000 --n_test 60000 --clean 0 --resume $ST >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
