#!/bin/bash
# L3 end-hole shift scan (Akahane-type S1, and the Minkov-Savona Si shifts) at 32 px/um.
# 32 and 64 px/um agree within 2 % in Q and 0.2 % in lambda for the plain L3.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
for s in 0.10 0.15 0.20 0.25 0.30; do
  run l3_s1_${s}_res32 a=0.28 r_a=0.30 res=32 ring=300 shifts=$s
done
run l3_ms3_res32 a=0.28 r_a=0.30 res=32 ring=300 shifts=0.327,0.257,0.116
echo QUEUE_DONE
