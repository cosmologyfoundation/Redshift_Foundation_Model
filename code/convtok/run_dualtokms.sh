#!/bin/bash
# Tokenizer v2 objective + Muon (lr 2e-3) + SwiGLU, 20% class mix, 1M x 10 epochs
# (owner 2026-10-13: 「做 tokenizer 1M 10輪 2e-3」 with 「Muon + SwiGLU 5 types 均分」).  AdamW part lr 2e-4 as v2.
# per GPU 128 x accum 2 = gbs 1024 (same global batch as v2 / v3).  usage: run_dualtokms.sh JOBID
J=$1
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
L=logs/dualtokms.log
echo "START $(date +%H:%M:%S) job $J" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u dualtok.py --clean 0 --tag DUALTOKMS --size 1000000 --epochs 10 --bs 128 --accum 2 --lr 2e-4 \
  --muon --muon_lr 2e-3 --swiglu --lv_n 5,5 --compile >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
