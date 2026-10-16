#!/bin/bash
# DUALTOK3: dual-scale tokenizer, masked + denoise + multi-scale objective, train mix QSO 45 / ELG 45 / others 3.33
# (owner 2026-10-12: 「同時改 先練tokenizer」).  300k or 1M x 5 epochs (owner 2026-10-12: A = 1M x 5); per GPU 128 x accum 2 = gbs 1024 (v2: 64 x 4).
J=$1; SIZE=${2:-300000}
module load pytorch/2.11.0
cd /global/homes/s/shunth/claude_learning_map/convtok
L=logs/dualtok3.log
echo "START $(date +%H:%M:%S) job $J" >> $L
srun --jobid=$J --overlap -N1 -n4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
  python -u dualtok3.py --clean 0 --size $SIZE --epochs 5 --bs 128 --accum 2 --compile >> $L 2>&1 < /dev/null
echo "RUN EXIT $? $(date +%H:%M:%S)" >> $L
