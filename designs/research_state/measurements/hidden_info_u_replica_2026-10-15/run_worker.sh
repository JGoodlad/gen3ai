#!/usr/bin/env bash
# ONE detached, resumable worker for one checkpoint, under rule 5b while the GPU is leased: the agent cores
# (5-7 + SMT siblings 13-15), nice 19, its own memory-capped scope. It runs from a CODE SNAPSHOT (made by
# snapshot.sh), so it survives the worktree that launched it. Re-runs itself after a crash (the runner skips every
# state with a done-marker and every arm row already written); stops at the first clean exit.
#   run_worker.sh <cp> <code_dir> <out_dir> [hiu.run args...]
set -u
cp="$1"; code="$2"; out="$3"; shift 3
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
MEMCAP="$code/mem_cap.sh"
[ -x "$MEMCAP" ] || MEMCAP=/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh
for attempt in 1 2 3 4 5 6; do
  echo "== attempt $attempt $cp $(date -Is) pid $$"
  PYTHONPATH="$code/hidden_info_replica_2026-10-11:$code/hidden_info_u_replica_2026-10-15" \
  HIU_LEARNSET="$code/gen3_learnset.json" \
    "$MEMCAP" --name "hiu-$cp" 8 taskset -c 5-7,13-15 nice -n 19 "$PY" -m hiu.run --cp "$cp" --out "$out" "$@"
  rc=$?
  echo "== exit $rc $(date -Is)"
  [ "$rc" -eq 0 ] && break
  sleep 120
done
