#!/bin/bash
# McCutcheon & Loncar reproduction at 64 px/um, both readings of the cavity gap s.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
for g in section edge; do
  echo "START mcc_${g}_res64 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_nanobeam.py mcc_${g}_res64 res=64 gap_mode=$g 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END mcc_${g}_res64 $(date '+%F %T')"
done
echo QUEUE_DONE
