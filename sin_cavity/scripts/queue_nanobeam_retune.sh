#!/bin/bash
# Retune of the own N15 nanobeam to 637 nm: once own_N15 has finished, scale its in-plane
# dimensions (hole x, r and beam width; thickness stays 200 nm) by s with
# lambda ~ s^0.8 (in-plane scaling at fixed thickness shifts lambda less than proportionally),
# and run it if lambda is more than 3 nm off. Submitted early so it sits in the cpu_slot queue.
cd "$(dirname "$0")/.."
while [ ! -f data/nanobeam/own_N15.npz ]; do sleep 60; done
read SKIP S W <<< $(/home/yuma/miniconda3/envs/mp/bin/python -c "
import numpy as np
d = np.load('data/nanobeam/own_N15.npz')
lam = 1e3 / float(d['f']) if 'f' in d.files else 637.0
s = (637.0 / lam) ** (1 / 0.8)
np.save('data/nanobeam/design_N15s.npy', np.load('data/nanobeam/design_N15.npy') * s)
print(int(abs(lam - 637) <= 3), f'{s:.4f}', f'{0.300 * s:.4f}')")
echo "own_N15 retune: skip=$SKIP scale=$S w=$W"
[ "$SKIP" = 1 ] && { echo QUEUE_DONE; exit 0; }
E=/home/yuma/miniconda3/envs/mp-mpi/bin
echo "START own_N15s $(date '+%F %T')"
/usr/bin/time -f "%e s %M kB" env PYTHONPATH=../src PYTHONUNBUFFERED=1 ../scripts/cpu_slot.sh $E/mpirun -np 12 $E/python scripts/run_nanobeam.py own_N15s design=N15s res=64 w=$W 2>&1 \
  | grep --line-buffered -v "Elapsed\|^$\|FloatProgress\|Using MPI"
echo "END own_N15s $(date '+%F %T')"
echo QUEUE_DONE
