#!/usr/bin/env bash
# X5 A/B LOOK 3: purpose metric (1), run ONLY after the matched-steps read is NON-INFERIOR (design_x5_belief_tokens.md
# §7.4 "Fixed sequence", §7.7(b) Amendment 3(b), Decision record 2026-10-06 "mean over EVERY blob run of the look").
#
#   1. `main.belief_roles read` of all 16 finals (8 blob + 8 fixed_mass, 15M `final_model.zip`) on the Lane S bank, CPU
#      forwards through the strict loader. Every blob run is read on its OWN named set E_row; every fixed_mass run is
#      scored on EVERY blob run's set (`--reference fm=blob_s1001,…,blob_s1008`), its value the MEAN.
#   2. `main.belief_roles infer` — the across-seed two-sample t on 14 df against the LOOK-3 boundary 1.874 (never full
#      α at the crossing look). The gate is `intent_logloss_conditional`; every other per_run metric is reported.
#
# THE PIN (bef16d61): executed from the 706fa536 checkout (PIN_DIR), which re-encodes the bank with ITS encoder;
# REFUSES (exit 5) unless `git rev-parse HEAD` there is 706fa536. CPU only (CUDA hidden). Run DETACHED.
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=706fa536ef53d9a680d2461f16613e0d1430d176
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/x5-look3-pin-706fa536}
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
OPS=${OPS:-/home/goodlad/dev/gen3ai/scripts/ops}
OUT=${OUT:-$HERE/purpose}
SEEDS="1001 1002 1003 1004 1005 1006 1007 1008"

cd "$PIN_DIR" || { echo "REFUSED: no pin checkout at $PIN_DIR"; exit 5; }
export PYTHONPATH=$PIN_DIR/src CUDA_VISIBLE_DEVICES=
HEAD=$(git rev-parse HEAD)
echo "=== $(date -Is) purpose metric (1): checkout $PIN_DIR at git rev-parse HEAD = $HEAD (required $PIN)"
[ "$HEAD" = "$PIN" ] || { echo "REFUSED: not at the registered pin"; exit 5; }
[ -z "$(git status --porcelain --untracked-files=no -- src data designs/research_state/measurements/m5_laneS)" ] \
  || { echo "REFUSED: the pin checkout is modified"; exit 5; }

run_of() { case "$1_$2" in fm_1006) echo rb_x5ab_fm_s1006b ;; *) echo "rb_x5ab_$1_s$2" ;; esac; }
CK=(); BL=""
for s in $SEEDS; do CK+=(--ckpt "$MODELS/$(run_of blob "$s")/final_model.zip=blob_s$s"); BL="$BL${BL:+,}blob_s$s"; done
for s in $SEEDS; do CK+=(--ckpt "$MODELS/$(run_of fm "$s")/final_model.zip=fm_s$s"); done
REF=(); for s in $SEEDS; do REF+=(--reference "fm_s$s=$BL"); done

mkdir -p "$OUT"
"$OPS/mem_cap.sh" 32 timeout 3h nice -n 15 "$PY" -m main.belief_roles read --out "$OUT" --threads 4 --workers 2 "${CK[@]}" "${REF[@]}" \
  || { echo "FAILED read ($(date -Is))"; exit 4; }
T=(); C=(); for s in $SEEDS; do T+=("$OUT/fm_s$s.json"); C+=("$OUT/blob_s$s.json"); done
"$PY" -m main.belief_roles infer --treat "${T[@]}" --control "${C[@]}" --boundary 1.874 --json > "$OUT/infer.txt" \
  || { echo "FAILED infer ($(date -Is))"; exit 4; }
echo "=== $(date -Is) DONE"
