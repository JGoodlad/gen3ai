#!/usr/bin/env bash
#
# rebuild_main_rust.sh — rebuild MAIN's Rust target (rust_sim + rust_env, release + selfcheck) after a ship
# that touched src/rust_sim or src/rust_env, and verify every artifact's --build-stamp against main's tree.
# Step 9 of /gen3ai-ship. REFUSES (never forces, never kills) while any live process maps or runs main's
# src/rust_*/target artifacts. Detail and exit codes: src/utils/rebuild_main_rust.py.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
rebuild_main_rust.sh — rebuild main's Rust target after a ship and prove it is current

USAGE
    scripts/ops/rebuild_main_rust.sh run        # refusal checks, the 4 cargo builds, stamp verification
    scripts/ops/rebuild_main_rust.sh --check    # refusal checks + verification of what is built now (no build)
    scripts/ops/rebuild_main_rust.sh --dry-run  # print the plan and the 4 commands

Always acts on the MAIN checkout (from a worktree too). Builds run at nice 19 on cores 5-7,13-15 under
mem_cap.sh 8, GEN3AI_BLOCKING set, CARGO_TARGET_DIR unset. Exit: 0 ok; 2 usage; 3 refused (live process
uses the target / target outside main); 4 main dirty; 5 build failed; 6 an artifact failed verification.
On a refusal: report it and rebuild later; never kill the process, never force.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: rebuild_main_rust.sh needs run, --check or --dry-run" >&2; exit 2 ;;
esac

export GEN3AI_PYTHON="${GEN3AI_PYTHON:-$(ops_python)}"
unset CARGO_TARGET_DIR
exec "$GEN3AI_PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.rebuild_main_rust import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" --main "$(ops_main_checkout)" --mem-cap "$(ops_repo_root)/scripts/ops/mem_cap.sh" "$@"
