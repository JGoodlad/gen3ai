#!/usr/bin/env bash
# X66 ExIt-0H: build the honest label bank (PREREG §2) at the SIDE PIN, outcome-blind, in bounded chunks under the
# heavy slots (rule 5b: nice 19 on the agent cores; each chunk <= 44 min, resumable by shard). Stops on the first
# exit that is neither 0 (done) nor 124 / 130 (the chunk's timeout). CPU only; no GPU.
#   bank_build.sh <side-pin worktree>
set -u
PINWT="${1:?the side-pin worktree (git worktree at the x66-pin tag)}"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
STATE=/home/goodlad/gen3ai_archive/sprime_honest_play_2026-11-06
X=/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09
PSTAR=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
mkdir -p "$X"
say() { echo "[$(date -Is)] X66 bank: $*" | tee -a "$X/bank_build.log"; }
cd "$PINWT" || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { say "HOLD: $PINWT is dirty"; exit 2; }
say "pin $(git rev-parse HEAD)"
export PYTHONPATH="$PINWT/src" CUDA_VISIBLE_DEVICES=""
for chunk in $(seq 1 12); do
    say "chunk $chunk: start"
    nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --name x66bank 12 timeout -s INT -k 60 2640 \
        "$PY" -m main.x65.bank build --source sprime_honest --state "$STATE" --out "$X/bank" --workers 3 \
        --pstar "$PSTAR" >> "$X/bank_build.log" 2>&1
    rc=$?
    say "chunk $chunk: rc=$rc"
    [ "$rc" -eq 0 ] && { say "DONE: $(cat "$X/bank/bank_manifest.sha256")"; exit 0; }
    case $rc in 124|130|137) ;; *) say "HOLD: rc=$rc"; exit 1 ;; esac
done
say "HOLD: not finished after 12 chunks"
exit 1
