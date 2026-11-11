#!/bin/bash
# L3 two-hole shift scan (S1, S2) around the best single shift, 32 px/um.
# 32 and 64 px/um agree within 2 % in Q and 0.2 % in lambda for the plain L3.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
for s1 in 0.20 0.25; do for s2 in 0.05 0.10 0.15 0.20; do
  run l3_s${s1}_${s2}_res32 a=0.28 r_a=0.30 res=32 ring=300 shifts=$s1,$s2
done; done
echo QUEUE_DONE
