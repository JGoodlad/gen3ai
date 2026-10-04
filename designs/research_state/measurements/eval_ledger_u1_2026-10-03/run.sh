#!/usr/bin/env bash
set -u
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
U1=/home/goodlad/dev/gen3ai-wt/ledger
BASE=/home/goodlad/dev/gen3ai-wt/ledger-base
SCRIPT=$U1/designs/research_state/measurements/eval_ledger_u1_2026-10-03/h2h_digest_proof.py
A=/home/goodlad/dev/gen3ai/models/sizing_A_n48_e10_s1001/final_model.zip
B=/home/goodlad/dev/gen3ai/models/sizing_B_n256_e10_s1001/final_model.zip
CLI="play --player $A --opponent $B --pairs 100 --batch-pairs 50 --seed 7 --device cpu --backend eager --n-envs 16 --threads 4 --torch-threads 4 --label u1_digest_proof"
cd $BASE && PYTHONPATH=$BASE/src $PY $SCRIPT games --out /tmp/u1proof/base_games.json || exit 11
cd $U1 && PYTHONPATH=$U1/src $PY $SCRIPT games --out /tmp/u1proof/u1_games.json || exit 12
cd $BASE && PYTHONPATH=$BASE/src $PY -m main.h2h $CLI --out /tmp/u1proof/base_rows || exit 13
cd $U1 && PYTHONPATH=$U1/src $PY -m main.h2h $CLI --out /tmp/u1proof/u1_root || exit 14
cd $U1 && PYTHONPATH=$U1/src $PY $SCRIPT compare /tmp/u1proof/base_games.json /tmp/u1proof/u1_games.json /tmp/u1proof/base_rows /tmp/u1proof/u1_root > /tmp/u1proof/compare.json
echo compare_exit=$?
