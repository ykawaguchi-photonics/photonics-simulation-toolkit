#!/bin/bash
# Fabrication tolerance of the final L3 (shifts 0.35,0.35,0.20, a = 266.6 nm) at 32 px/um:
# nominal, hole radius +-5 nm (r_a 0.30 +- 0.0188), thickness 205 / 195 nm. Via cpu_slot.sh.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
SLOT=../scripts/cpu_slot.sh
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $SLOT $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
C="a=0.2666 res=32 ring=300 shifts=0.35,0.35,0.20"
run l3tol_nom $C r_a=0.30
run l3tol_dr+5 $C r_a=0.3188
run l3tol_dr-5 $C r_a=0.2812
run l3tol_t205 $C r_a=0.30 t=0.205
run l3tol_t195 $C r_a=0.30 t=0.195
echo QUEUE_DONE
