#!/usr/bin/env bash
# X63 corpus registration (2026-10-29): ONE heavy-slot job that (1) builds x63_union_v1 from the committed components,
# (2) soaks it on the Rust env core with every label family (utils.team_corpora.soak, 5 rounds = 10 battles a team,
# N1-P's dose), writing x63_union_v1/soak_selfcheck.json, and (3) checks that x62_recomb_v1 and x63_union_v1 reproduce
# byte for byte (--check). Run from the worktree root under scripts/ops/mem_cap.sh, at nice 19 on the agent cores.
set -euo pipefail
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
echo "== build x63_union_v1 $(date -Is)"
"$PY" -m utils.team_corpora.build_x63_union
echo "== soak x63_union_v1 $(date -Is)"
"$PY" -m utils.team_corpora.soak x63_union_v1 --rounds 5 --seed 3802 \
    --out src/utils/team_corpora/x63_union_v1/soak_selfcheck.json | tail -c 1500
echo
echo "== --check x62_recomb_v1 $(date -Is)"
"$PY" -m utils.team_corpora.build_x62_recomb --check
echo "== --check x63_union_v1 $(date -Is)"
"$PY" -m utils.team_corpora.build_x63_union --check
echo "== done $(date -Is)"
