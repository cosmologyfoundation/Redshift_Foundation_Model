#!/bin/bash
# keep the ZT5ft35 run going across 4 h interactive nodes until its test result exists.
# After every finished epoch (state saved): if the node's remaining time < 1.15 x the last epoch + 15 min, move to a
# fresh node now (no half epoch is lost).  If a node dies anyway, resume on a new one from the last state.
cd /global/homes/s/shunth/claude_learning_map/convtok
BS=${1:-40}
L=logs/zt5ft.log
R=results/ZT5ft35_frozen_H32_L12_1000000_e2_gbs$((BS*4))_lr0.0001_tb2200_vb4400_s42_cl0_mix.json
ST=$SCRATCH/zfm_ckpt/ZT5ft35_frozen_H32_L12_1000000_e2_gbs$((BS*4))_lr0.0001_tb2200_vb4400_s42_cl0_mix_state.pt
T0=$(date +%s); touch $L
secs() { echo "$1" | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2)s+=3600*$(n-2); if(n>3)s+=86400*$(n-3); print s}'; }
while true; do
  [ -f $R ] && [ $(stat -c %Y $R) -gt $T0 ] && { echo "CHAIN DONE $(date +%H:%M:%S)" >> $L; exit 0; }
  grep -q "EARLY STOP" $L && [ -f $R ] && { echo "CHAIN DONE (early stop) $(date +%H:%M:%S)" >> $L; exit 0; }
  salloc --no-shell -q interactive -C gpu -N 1 -G 4 -t 4:00:00 -A m5374_g > logs/chainft_salloc.out 2>&1
  J=$(grep -oE "Granted job allocation [0-9]+" logs/chainft_salloc.out | grep -oE "[0-9]+$")
  [ -z "$J" ] && { echo "CHAIN no allocation, retry in 5 min $(date +%H:%M:%S)" >> $L; sleep 300; continue; }
  echo "CHAIN node $J $(date +%H:%M:%S)" >> $L
  ./run_zt5ft.sh $J $BS &
  RUNPID=$!
  nep=$(grep -c "^  EPOCH [0-9]  train" $L); last=$(date +%s); dur=0
  while kill -0 $RUNPID 2>/dev/null; do
    sleep 60
    n=$(grep -c "^  EPOCH [0-9]  train" $L)
    if [ $n -gt $nep ]; then
      now=$(date +%s); dur=$((now - last))
      nep=$n; last=$now
      sleep 90                                              # the state file is written right after the EPOCH line
      left=$(secs $(squeue -h -j $J -o %L 2>/dev/null || echo 0:0))
      need=$((dur * 115 / 100 + 900))
      if [ $n -lt 2 ] && [ "$left" -lt "$need" ]; then
        echo "CHAIN epoch $n done; node $J has ${left}s left < ${need}s needed: moving $(date +%H:%M:%S)" >> $L
        srun --jobid $J --overlap -N1 -n1 pkill -f "ztrans[5]\.py"; sleep 20; scancel $J
        wait $RUNPID; break
      fi
    fi
  done
  scancel $J 2>/dev/null
done
