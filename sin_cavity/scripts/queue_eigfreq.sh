#!/bin/bash
# solve_eigfreq with a ring-down warm start and loose inner tolerance, plain L3 at 32 px/um.
# Waits for the L3 recheck queue (one 3D job at a time).
cd "$(dirname "$0")/.."
while pgrep -f queue_l3_v2.sh > /dev/null; do sleep 60; done
E=/home/yuma/miniconda3/envs/mp-mpi/bin
echo "START $(date '+%F %T')"
PYTHONPATH=../src PYTHONUNBUFFERED=1 timeout 3600 $E/mpirun -np 12 $E/python scripts/test_eigfreq.py 1.510 32 1e-5 1e-3 200 2>&1 \
  | grep --line-buffered -E "eigfreq|residual|rror"
echo "END $(date '+%F %T')"
