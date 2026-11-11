#!/bin/bash
# Own Quan-Loncar nanobeam designs (N = 15 and 20 taper segments, 637 nm), then the fabrication
# tolerance of N15 (radius and thickness +-5 nm). 64 px/um. Every run goes through cpu_slot.sh.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
SLOT=../scripts/cpu_slot.sh
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $SLOT $E/mpirun -np 12 $E/python scripts/run_nanobeam.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
run own_N15 design=N15 res=64
run own_N20 design=N20 res=64
run own_N15_dr+5 design=N15 res=64 dr=0.005
run own_N15_dr-5 design=N15 res=64 dr=-0.005
run own_N15_t205 design=N15 res=64 t=0.205
run own_N15_t195 design=N15 res=64 t=0.195
echo QUEUE_DONE
