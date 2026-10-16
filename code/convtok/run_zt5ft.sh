#!/bin/bash
# ZT5ft35: ZT5 fine-tuned with training class mix QSO 35% ELG 35% LRG/BGS/STAR 10% (owner 2026-10-11).  Held-out and test keep 20% each.
# from the ZT5 best checkpoint, 1M x 2 epochs, new OneCycle peak lr 1e-4, early stop as before, no control arm (owner).
# held-out 200k (blocks 2200-4400), test 200k (blocks 0-2200).
# usage: run_zt5ft.sh JOBID [BS]   (resumable from the per-epoch state file)
J=$1; BS=${2:-40}
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
TOK=$SCRATCH/zfm_ckpt/DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt
ST=$SCRATCH/zfm_ckpt/ZT5ft35_frozen_H32_L12_1000000_e2_gbs$((BS*4))_lr0.0001_tb2200_vb4400_s42_cl0_mix_state.pt
L=logs/zt5ft.log
echo "START $(date +%H:%M:%S) job $J bs $BS" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u ztrans5.py --tag ZT5ft35 --tok $TOK --H 32 --size 1000000 --bs $BS --lr 1e-4 --epochs 2 --init $SCRATCH/zfm_ckpt/ZT5_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_best.pt --mix_train QSO:0.35,ELG:0.35,LRG:0.1,BGS:0.1,STAR:0.1 --compile --patience 2 \
  --test_blocks 2200 --val_blocks 4400 --n_es 200000 --n_test 200000 --clean 0 --resume $ST >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
