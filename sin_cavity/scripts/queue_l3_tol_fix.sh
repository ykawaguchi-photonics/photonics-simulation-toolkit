#!/bin/bash
# l3tol_dr+5 picked a low-Q mode at 642 nm (lowest-frequency rule); the fundamental is at
# f 1.5799 c/um (633 nm, Q 1.78e3). Narrowband re-run on it for a single-mode V. Via cpu_slot.sh.
cd "$(dirname "$0")/.."
E=/home/yuma/miniconda3/envs/mp-mpi/bin
echo "START l3tol_dr+5_nb $(date '+%F %T')"
/usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 ../scripts/cpu_slot.sh $E/mpirun -np 12 $E/python scripts/run_l3.py l3tol_dr+5_nb a=0.2666 res=32 ring=300 shifts=0.35,0.35,0.20 r_a=0.3188 fcen=1.5799 df=0.03 pick=amp 2>&1 \
  | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
echo "END l3tol_dr+5_nb $(date '+%F %T')"
echo QUEUE_DONE
