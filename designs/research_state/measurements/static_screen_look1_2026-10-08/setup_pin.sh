#!/usr/bin/env bash
# One-time setup of the P_st play checkout (root CLAUDE.md "Git Worktree Setup"): the Showdown submodule's source
# (main.h2h's team validation runs deps/pokemon-showdown through node) + main's build artifacts, guarded.
set -eu
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09}
cd "$PIN_DIR"
echo "pin HEAD $(git rev-parse HEAD)"
# the guarded links made before the submodule existed would block its checkout: drop them only if they are links
for n in dist node_modules; do [ -L "deps/pokemon-showdown/$n" ] && rm "deps/pokemon-showdown/$n"; done
git submodule update --init
for n in dist node_modules; do
  [ -e "deps/pokemon-showdown/$n" ] || ln -s "/home/goodlad/dev/gen3ai/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"
done
ls deps/pokemon-showdown | head -20
git status --porcelain --untracked-files=no -- src data | head
