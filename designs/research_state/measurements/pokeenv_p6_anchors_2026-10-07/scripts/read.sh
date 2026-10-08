#!/usr/bin/env bash
# usage: read.sh <blocked:0|1> <anchors args...>
# One `python -m main.anchors` read from THIS worktree: its own src on PYTHONPATH, its own release binaries
# (sim_bridge / live_reader / bot_reader), CPU only; blocked=1 runs it under the poke-env import blocker.
WT=${WT:-/home/goodlad/dev/gen3ai/.claude/worktrees/agent-ae1cd62e313f38efc}
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES=""
export POKESIM_SIM_BRIDGE_BIN=$WT/src/rust_sim/target/release/sim_bridge
export POKESIM_LIVE_READER_BIN=$WT/src/rust_sim/target/release/live_reader
export POKESIM_BOT_READER_BIN=$WT/src/rust_env/target/release/bot_reader
export GEN3AI_LIVE_HALT_FILE=${GEN3AI_LIVE_HALT_FILE:-/tmp/p6s2/work/live_halt.json}
cd "$WT" || exit 9
blocked=$1; shift
if [ "$blocked" = 1 ]; then
  exec "$PY" -m utils.poke_env_blocker main.anchors "$@"
elif [ "$blocked" = legacybot ]; then
  # the OLD path of the bot identity proof: the legacy client + the Python bot, streams seeded by the harness
  exec "$PY" "$(dirname "$0")/legacy_seeded_bot.py" "$@"
elif [ "$blocked" = legacyshadow ]; then
  # the OLD path of the --server node proof: the legacy client with the new reader as a SHADOW
  exec "$PY" "$(dirname "$0")/legacy_shadow.py" "$@"
else
  exec "$PY" -m main.anchors "$@"
fi
