#!/bin/bash
# Downstream z A/B of the two tokenizers (owner 2026-10-12: 「做 申請新node 跑下游z 300k」).
# Same Transformer (ztrans5 = ZT5 architecture), frozen tokenizer, 300k x 1 epoch, 20% class mix, gbs 160, lr 3e-4,
# held-out 60k (blocks 2200-4400) and test 60k (blocks 0-2200) = 20% of training each.  Only the tokenizer differs.
J=$1
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
L=logs/tokab.log
for arm in v2:DUALTOK_300000_n55_cl0_mix_e5_bs1024_ep5.pt v3:DUALTOK3_1000000_n55_cl0_q45e45_e5_bs1024_ep5.pt; do
  nm=${arm%%:*}; ck=$SCRATCH/zfm_ckpt/${arm#*:}
  echo "ARM $nm $ck $(date +%H:%M:%S)" >> $L
  srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
    python -u ztrans5.py --tag ZT5tok$nm --tok $ck --H 32 --size 300000 --bs 40 --lr 3e-4 --epochs 1 --compile --patience 2 \
    --test_blocks 2200 --val_blocks 4400 --n_es 60000 --n_test 60000 --clean 0 >> $L 2>&1 < /dev/null
  echo "ARM $nm EXIT $? $(date +%H:%M:%S)" >> $L
done
echo "AB DONE $(date +%H:%M:%S)" >> $L
