#!/usr/bin/env bash
# CHUNK 1 of the X26 collapse battery (one heavy slot, <= 45 min, forward passes only, CPU, rule 5b):
#   (a) B1 / B2: policy reads of X26 checkpoints 150M-380M on X26's own bank (collapse_pilot_2026-10-17/bank_x26);
#   (b) B3: policy reads of the five read points on the pre-era Lane S bank (m5_laneS/bank_v1);
#   (c) B9 (alpha): main.belief_roles read of the five read points (+160M) on the Lane S bank.
# Read-only on models/; every output goes under $A (the archive). Resumable: a read whose JSON exists is skipped.
#     chunk1_policy_reads.sh            (from the worktree root; run detached)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
A="${A:-$HOME/gen3ai_archive/x26_collapse_battery_2026-10-20}"
C=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints
P=$ROOT/designs/research_state/measurements/collapse_pilot_2026-10-17
L=$ROOT/designs/research_state/measurements/m5_laneS
cd "$ROOT"
mkdir -p "$A/reads_x26bank" "$A/reads_bankv1" "$A/belief_roles"

# The bank replays need this checkout's `core_events` (a fresh worktree has no target/): build it once, on the
# agent cores, before the reads (the resolver would otherwise build it inside the first read).
echo "=== $(date -Is) cargo build core_events"
( cd src/rust_sim && nice -n 19 taskset -c 5-7,13-15 cargo build --release --bin core_events -q ) || { echo "FAILED build"; exit 3; }

ck() { echo "$C/checkpoint_${1}_steps.zip"; }
declare -A STEP=( [150M]=150000100 [160M]=160000085 [180M]=180000055 [200M]=200000120 [220M]=220000027
                  [230M]=230000074 [240M]=240000104 [260M]=260000235 [280M]=280000170 [300M]=300000189
                  [320M]=320000071 [330M]=330000202 [340M]=340000290 [360M]=360000095 [380M]=380000195 )

args=()
for lab in 150M 160M 180M 200M 220M 230M 240M 260M 280M 300M 320M 330M 340M 360M 380M; do
  [ -e "$A/reads_x26bank/X26@$lab.json" ] || args+=(--ckpt "$(ck "${STEP[$lab]}")=X26@$lab")
done
echo "=== $(date -Is) (a) bank_x26 reads: $(( ${#args[@]} / 2 )) to do"
[ ${#args[@]} -eq 0 ] || "$HERE/run.sh" -m main.policy_spectrum read --bank "$P/bank_x26" --out "$A/reads_x26bank" \
    --threads 3 --workers 1 "${args[@]}" || echo "FAILED (a)"

args=()
for lab in 180M 230M 280M 330M 380M; do
  [ -e "$A/reads_bankv1/X26@$lab.json" ] || args+=(--ckpt "$(ck "${STEP[$lab]}")=X26@$lab")
done
echo "=== $(date -Is) (b) bank_v1 reads: $(( ${#args[@]} / 2 )) to do"
[ ${#args[@]} -eq 0 ] || "$HERE/run.sh" -m main.policy_spectrum read --bank "$L/bank_v1" --out "$A/reads_bankv1" \
    --threads 3 --workers 1 "${args[@]}" || echo "FAILED (b)"

args=()
for lab in 160M 180M 230M 280M 330M 380M; do
  [ -e "$A/belief_roles/X26@$lab.json" ] || args+=(--ckpt "$(ck "${STEP[$lab]}")=X26@$lab")
done
echo "=== $(date -Is) (c) belief_roles reads: $(( ${#args[@]} / 2 )) to do"
[ ${#args[@]} -eq 0 ] || "$HERE/run.sh" -m main.belief_roles read --out "$A/belief_roles" --threads 3 --workers 1 \
    "${args[@]}" || echo "FAILED (c)"
echo "=== $(date -Is) DONE chunk 1"
