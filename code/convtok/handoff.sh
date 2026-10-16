#!/bin/bash
# move the 1M run to a fresh 4 h node right after an epoch whose state is on disk, when the old node cannot fit another epoch
# usage: handoff.sh OLDJOB EPOCH   (EPOCH = the epoch to finish on the old node)
OLD=$1; E=$2
LOG=~/claude_learning_map/convtok/logs/zt1m.log
ST=$SCRATCH/zfm_ckpt/ZT2_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix_state.pt
P=$((E-1))
until grep -q "EPOCH $P " $LOG && [ -f $ST ]; do sleep 60; done
sleep 120; t=$(stat -c %Y $ST)
until grep -q "EPOCH $E " $LOG && [ $(stat -c %Y $ST) -gt $t ]; do sleep 15; done
s0=-1; s1=$(stat -c %s $ST)
while [ "$s0" != "$s1" ]; do sleep 30; s0=$s1; s1=$(stat -c %s $ST); done
echo "HANDOFF epoch $E state saved $(date +%H:%M:%S); requesting a new node" >> $LOG
salloc --no-shell -q interactive -C gpu -N 1 -G 4 -t 4:00:00 -A m5374_g > logs/handoff_salloc.out 2>&1
NEW=$(grep -oE "Granted job allocation [0-9]+" logs/handoff_salloc.out | grep -oE "[0-9]+$")
if [ -z "$NEW" ]; then echo "HANDOFF FAILED: no allocation $(date +%H:%M:%S)" >> $LOG; exit 1; fi
srun --jobid $OLD --overlap -N 1 -n 1 pkill -f "ztrans[2]\.py"
sleep 20; scancel $OLD
echo "HANDOFF new job $NEW, old job $OLD cancelled $(date +%H:%M:%S)" >> $LOG
./run_zt1m.sh $NEW 1
