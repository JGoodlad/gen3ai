#!/usr/bin/env bash
# Print, for each A/B binary, the checkout path baked in at compile time (CARGO_MANIFEST_DIR —
# the dex data path) and a sha256 prefix. Each binary must name ONLY its own worktree.
A=/home/goodlad/dev/gen3ai/.claude/worktrees/m1bench-A/src/rust_sim/target/release
B=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a35e14fb9d6f01186/src/rust_sim/target/release
for b in "$A/sim_bridge" "$B/sim_bridge" "$A/bridge_replay" "$B/bridge_replay"; do
  echo "$b  sha256=$(sha256sum "$b" | cut -c1-16)"
  strings "$b" | grep -o "/home/goodlad/dev/gen3ai[^ ]*rust_sim" | sort -u | sed 's/^/    baked: /'
done
