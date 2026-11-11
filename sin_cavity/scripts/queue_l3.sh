#!/bin/bash
# Plain L3 in the 200 nm SiN membrane (a = 280 nm, r/a = 0.30): coarse check, then 64 px/um.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
run l3_plain_res32 a=0.28 r_a=0.30 res=32 ring=150
run l3_plain_res64 a=0.28 r_a=0.30 res=64
echo QUEUE_DONE
