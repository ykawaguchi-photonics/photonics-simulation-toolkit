#!/bin/bash
# McCutcheon & Loncar nanobeam (edge gap, 64 px/um): Harminv window and PML distance checks.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_nanobeam.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
run mcc_edge_res64_ring1000 res=64 gap_mode=edge ring=1000
run mcc_edge_res64_pad1 res=64 gap_mode=edge pad=1.0
echo QUEUE_DONE
