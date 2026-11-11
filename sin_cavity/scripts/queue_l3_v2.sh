#!/bin/bash
# L3 re-check with the fundamental-mode pick (lowest frequency in the band), then a narrowband
# second pass on that mode for a single-mode V. 32 px/um.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
C="a=0.28 r_a=0.30 res=32 ring=300"
for p in "plain:" "s25:0.25" "s25_20:0.25,0.20" "ms3:0.327,0.257,0.116"; do
  tag=${p%%:*}; sh=${p#*:}
  run l3v2_${tag}_bb $C ${sh:+shifts=$sh}
  run l3v2_${tag}_nb $C ${sh:+shifts=$sh} ref=l3v2_${tag}_bb
done
echo QUEUE_DONE
