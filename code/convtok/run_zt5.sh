#!/bin/bash
# ZT5 = 40 km/s detail + per-candidate line sentences (B).  Owner 2026-10-11: 「寫B 跟40 km/s一起 1M 訓練 6輪試試」.
# 1M train x 6 epochs from scratch (tokenizer frozen), early stop patience 2 on loss AND NMAD over 200k held-out
# (blocks 2200-4400), test 200k (blocks 0-2200); same split, lr, gbs as the ZT2 1M run so the two compare.
# usage: run_zt5.sh JOBID [BS]   (resumable from the per-epoch state file)
J=$1; BS=${2:-40}
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
TOK=$SCRATCH/zfm_ckpt/DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt
ST=$SCRATCH/zfm_ckpt/ZT5_frozen_H32_L12_1000000_e6_gbs$((BS*4))_lr0.0003_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/zt5.log
echo "START $(date +%H:%M:%S) job $J bs $BS" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans5.py --tag ZT5 --tok $TOK --H 32 --size 1000000 --bs $BS --lr 3e-4 --epochs 6 --compile --patience 2 \
  --test_blocks 2200 --val_blocks 4400 --n_es 200000 --n_test 200000 --clean 0 --resume $ST >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
