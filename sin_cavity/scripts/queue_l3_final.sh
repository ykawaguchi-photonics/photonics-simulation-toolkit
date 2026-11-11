#!/bin/bash
# Final L3: best GME shift set (both scans), lattice scaled to put the mode near 637 nm
# (a = 280 nm gave 669 nm in Meep), 64 px/um, broadband pass then narrowband pass for V.
cd "$(dirname "$0")/.."
while pgrep -f gme_l3_scan.py > /dev/null; do sleep 60; done
SH=$(/home/yuma/miniconda3/envs/legume/bin/python -c "
import numpy as np
scans = [np.load(f'data/l3/gme_scan{n}.npz') for n in ('', '_ext')]
best = max(scans, key=lambda d: np.nanmax(d['Q']))
i, j, k = np.unravel_index(np.nanargmax(best['Q']), best['Q'].shape)
print(f\"{best['S1'][i]},{best['S2'][j]},{best['S3'][k]}\")")
echo "best GME shifts: $SH"
E=/home/yuma/miniconda3/envs/mp-mpi/bin
run() {
  echo "START $1 $(date '+%F %T')"
  /usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 $E/mpirun -np 12 $E/python scripts/run_l3.py "$@" 2>&1 \
    | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
  echo "END $1 $(date '+%F %T')"
}
C="a=0.2666 r_a=0.30 res=64 ring=300 shifts=$SH"
run l3_final_bb $C
run l3_final_nb $C ref=l3_final_bb
echo QUEUE_DONE
