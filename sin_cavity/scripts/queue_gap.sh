#!/bin/bash
# Q and V vs cavity length s (edge-to-edge gap) for the McCutcheon & Loncar nanobeam, 64 px/um.
# Waits for any running nanobeam queue first (one 3D job at a time, RAM).
cd "$(dirname "$0")/.."
while pgrep -f "queue_domain.sh" > /dev/null; do sleep 60; done
E=/home/yuma/miniconda3/envs/mp-mpi/bin
for s in 85 90 100 105 80 110; do
  echo "START mcc_edge_s$s $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_nanobeam.py mcc_edge_s$s res=64 gap_mode=edge gap=0.$(printf %03d $s) 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END mcc_edge_s$s $(date '+%F %T')"
done
echo QUEUE_DONE
