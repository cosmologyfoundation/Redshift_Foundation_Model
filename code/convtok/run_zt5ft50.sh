#!/bin/bash
# ZT5ft50: ZT5ft35 fine-tuned again on ELG and QSO ONLY, 50% each (owner 2026-10-11: 「再fine tune 兩輪 只用 ELG QSO 各一半」).  Held-out and test keep 20% each.
# from the ZT5ft35 best checkpoint, 1M x 2 epochs, OneCycle peak lr 1e-4, no control arm.
# held-out 200k (blocks 2200-4400), test 200k (blocks 0-2200).
# usage: run_zt5ft50.sh JOBID [BS]   (resumable from the per-epoch state file)
J=$1; BS=${2:-40}
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
TOK=$SCRATCH/zfm_ckpt/DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt
ST=$SCRATCH/zfm_ckpt/ZT5ft50_frozen_H32_L12_1000000_e2_gbs$((BS*4))_lr0.0001_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/zt5ft50.log
echo "START $(date +%H:%M:%S) job $J bs $BS" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans5.py --tag ZT5ft50 --tok $TOK --H 32 --size 1000000 --bs $BS --lr 1e-4 --epochs 2 --init $SCRATCH/zfm_ckpt/ZT5ft35_frozen_H32_L12_1000000_e2_gbs160_lr0.0001_tb2200_vb4400_s42_cl0_mix_best.pt --mix_train QSO:0.5,ELG:0.5,LRG:0,BGS:0,STAR:0 --compile --patience 2 \
  --test_blocks 2200 --val_blocks 4400 --n_es 200000 --n_test 200000 --clean 0 --resume $ST >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
