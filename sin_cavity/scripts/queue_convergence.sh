#!/bin/bash
# Resolution convergence of the McCutcheon & Loncar nanobeam (edge-to-edge gap reading).
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
for r in 48 80; do
  echo "START mcc_edge_res$r $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_nanobeam.py mcc_edge_res$r res=$r gap_mode=edge 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END mcc_edge_res$r $(date '+%F %T')"
done
echo QUEUE_DONE
