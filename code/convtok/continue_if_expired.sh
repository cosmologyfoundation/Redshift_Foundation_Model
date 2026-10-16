#!/bin/bash
# if the node expires before the run writes its test result, get a new node and resume (finishes the epoch, or only runs the test)
OLD=$1
R=results/ZT2_frozen_H32_L12_1000000_e6_gbs160_lr0.0003_tb2200_vb4400_s42_cl0_mix.json
LOG=logs/zt1m.log
while squeue -h -j $OLD 2>/dev/null | grep -q .; do sleep 60; done
sleep 30
if [ -f $R ] && [ $R -nt $LOG.started_$OLD ]; then echo "done, no continuation needed"; exit 0; fi
echo "HANDOFF node $OLD ended without a test result $(date +%H:%M:%S); requesting a new node" >> $LOG
salloc --no-shell -q interactive -C gpu -N 1 -G 4 -t 4:00:00 -A m5374_g > logs/continue_salloc.out 2>&1
NEW=$(grep -oE "Granted job allocation [0-9]+" logs/continue_salloc.out | grep -oE "[0-9]+$")
[ -z "$NEW" ] && { echo "HANDOFF FAILED: no allocation $(date +%H:%M:%S)" >> $LOG; exit 1; }
echo "HANDOFF new job $NEW $(date +%H:%M:%S)" >> $LOG
./run_zt1m.sh $NEW
