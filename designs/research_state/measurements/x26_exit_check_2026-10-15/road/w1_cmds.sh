# W1 at P_X26 (orchestrator 10-15 ~16:30): run by x26_chain in the clean P_X26 checkout ($PW) with the Training lease token.
set -u
cd "$PW" || exit 9
echo "== (1) verify_compile_gate_cuda.sh"; GEN3AI_GPU_LEASE_TOKEN_FILE="$T" taskset -c 0-4,8-12 scripts/ops/verify_compile_gate_cuda.sh || exit 1
echo "== (2) compiled_first_update_test -m slow -k on_cuda"
GEN3AI_GPU_LEASE_TOKEN_FILE="$T" GEN3AI_TEST_ALLOW_GPU=1 PYTHONPATH="$PW/src" "$OPS/gpu_lock.sh" timeout 1200 "$PY" -m pytest src/agents/model/compiled_first_update_test.py -m slow -k on_cuda -q -p no:cacheprovider -rA || exit 2
echo "== W1 OK"
