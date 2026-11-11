#!/bin/bash
# Run a command only when the machine's MPI slot is free, so that two sessions never put
# two `mpirun -np 12` jobs on the 12 cores at the same time (that more than halves both).
#
#   scripts/cpu_slot.sh mpirun -np 12 python scripts/xxx.py ...
#
# Waits (polling once a minute) until
#   1. no other cpu_slot.sh holds the lock  (flock on $LOCK, FIFO-ish between sessions), and
#   2. no mpirun / mpiexec is running outside this wrapper (e.g. topological/scripts/bg_queue.sh,
#      which has its own idle check and backs off when it sees our mpirun).
# then runs the command while holding the lock. Use it for every 12-rank job; small serial
# checks (a few seconds, 1 core) do not need it.
LOCK=${CPU_SLOT_LOCK:-/home/yuma/meep-course/photonics-simulation-toolkit/logs/cpu_slot.lock}
mkdir -p "$(dirname "$LOCK")"
exec 9>>"$LOCK"
echo "cpu_slot: waiting for lock ($(date '+%F %T')): $*" >&2
flock 9
mpi_running() {
    # count only real mpirun/mpiexec executables: shells, `time`, `env` and this wrapper itself
    # carry "mpirun" in their command lines too. A job suspended with SIGSTOP (state T) is
    # parked by the coordinating session and does not count.
    for p in $(pgrep -x "mpirun|mpiexec|mpiexec.hydra"); do
        [[ "$(ps -o stat= -p "$p" 2>/dev/null)" == T* ]] || return 0
    done
    return 1
}
while mpi_running; do sleep 60; done
echo "cpu_slot: start $(date '+%F %T') pid $$: $*" | tee -a "$LOCK.log" >&2
"$@"
rc=$?
echo "cpu_slot: end   $(date '+%F %T') pid $$ rc $rc" >> "$LOCK.log"
exit $rc
