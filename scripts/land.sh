#!/usr/bin/env bash
#
# land.sh — run the static gates in a worktree, then land its branch on main.
#
#     scripts/land.sh <branch> [worktree-path]
#     scripts/land.sh --help
#
# This is the landing procedure `designs/ops/ORCHESTRATOR_SOP.md` §3 describes, as a script.
# Four steps, in this order, and it stops at the first failure:
#
#   1. GATES, run INSIDE the worktree — ruff (F,E9) · mypy · the `src/*_gate_test.py` statics.
#   2. `git push origin <branch>:main` from the MAIN checkout.
#   3. `git pull --ff-only origin main` — so the main checkout is not left behind its own remote.
#   4. `git worktree remove --force <worktree>` + prune + delete the local branch.
#
# WHY THE GATES ARE NOT PIPED. On 2026-09-06 a `pytest | tail` swallowed a ruff F811 and a
# duplicate definition landed on main as `00772d05`. Every gate here runs unpiped under
# `set -e -o pipefail`, and a failure prints "GATE FAILED — not pushing" and exits 1.
#
# WHY THE PATHS ARE DERIVED. The main checkout is `dirname` of `git rev-parse --git-common-dir`
# (the same derivation `scripts/bootstrap.sh` uses), never a literal — a hardcoded
# `/home/goodlad/dev/gen3ai` is one box's layout, and this script is now in the repo that gets
# cloned elsewhere.
#
# WHY IT MAY RE-EXEC ITSELF. Step 4 deletes the worktree — which, now that this script lives in
# the repo, is usually the directory the script itself is being read from. Linux keeps the open
# inode alive so bash would survive it, but "usually survives" is not a property to rely on in the
# step that lands code, so the script copies itself to a temp file and re-execs when it is about
# to remove its own tree.
#
# The python interpreter is `$GEN3AI_PYTHON` if set, else the box's `gen3ai_stable` env, else
# whatever `python3` is on PATH.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
land.sh — run the static gates in a worktree, then land its branch on main.

USAGE
    scripts/land.sh <branch> [worktree-path]
    scripts/land.sh --help

ARGUMENTS
    <branch>          the branch to push to main (pushed as <branch>:main)
    [worktree-path]   the linked worktree to gate and then remove. Optional: with no
                      worktree the gates are SKIPPED and nothing is removed — use that
                      only for a branch already gated elsewhere.

WHAT IT DOES
    1. gates, inside the worktree   ruff (F,E9) + mypy + src/*_gate_test.py
    2. git push origin <branch>:main        from the main checkout
    3. git pull --ff-only origin main       so main is not left behind
    4. git worktree remove --force + prune + delete the local branch

    Any gate failure exits 1 WITHOUT pushing. It never force-pushes; a rejected
    non-fast-forward push means someone else landed first — rebase the worktree on
    main, resolve, and run this again.

ENVIRONMENT
    GEN3AI_PYTHON     interpreter to run the gates with (default: the gen3ai_stable env)

EXIT
    0  landed        1  a gate failed, or the push was rejected        2  bad usage
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

BRANCH="$1"
WORKTREE="${2:-}"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# The MAIN checkout: `dirname` of the shared git dir. In a linked worktree `--git-common-dir` is
# the main checkout's `.git`; in the main checkout it is `.git` itself. Resolved relative to the
# script's own location, so it works from any cwd.
#
# ⚠️ Carried ACROSS the re-exec below in the environment, never recomputed. The relocated copy
# lives in `/tmp`, where `dirname "$SCRIPT_PATH"/..` is `/` and `git rev-parse` fails outright —
# which left `MAIN_CHECKOUT` empty and the run dead on `cd: null directory`. Resolve it once, in
# the tree that can answer the question.
if [ -n "${_LAND_MAIN_CHECKOUT:-}" ]; then
    MAIN_CHECKOUT="$_LAND_MAIN_CHECKOUT"
else
    MAIN_CHECKOUT="$(cd "$(dirname "$SCRIPT_PATH")/.." \
        && cd "$(git rev-parse --git-common-dir)" && cd .. && pwd)"
fi

PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"

# --- re-exec out of the tree we are about to delete ------------------------------------------
if [ -n "$WORKTREE" ] && [ -z "${_LAND_RELOCATED:-}" ]; then
    WT_ABS="$(cd "$WORKTREE" && pwd)"
    case "$SCRIPT_PATH" in
        "$WT_ABS"/*)
            TMP="$(mktemp -t land.XXXXXX.sh)"
            cp "$SCRIPT_PATH" "$TMP"
            export _LAND_RELOCATED=1
            export _LAND_MAIN_CHECKOUT="$MAIN_CHECKOUT"
            exec bash "$TMP" "$@"
            ;;
    esac
fi
# The relocated copy unlinks itself immediately: bash keeps reading through its open fd, and no
# temp file is left behind however this run ends.
[ -n "${_LAND_RELOCATED:-}" ] && rm -f "$0"

cd "$MAIN_CHECKOUT"

# --- 1. the gates ----------------------------------------------------------------------------
if [ -n "$WORKTREE" ]; then
    (
        cd "$WORKTREE"
        # MANDATORY in a worktree: `pip install -e .` names the MAIN checkout's src/, so without
        # this a worktree's pytest imports main's code and every result is about a tree nobody
        # edited (root CLAUDE.md → Python Environment).
        export PYTHONPATH="${PYTHONPATH:-}:src"
        "$PY" -m ruff check src/agents src/main src/utils \
              --select F,E9 --exclude src/poke_env --exclude src/rust_sim
        "$PY" -m mypy src/agents/model >/dev/null
        "$PY" -m pytest src/*_gate_test.py -q -p no:cacheprovider >/dev/null
        echo "gates: ruff + mypy + src/*_gate_test.py OK"
    ) || { echo "GATE FAILED — not pushing"; exit 1; }
fi

# --- 2-4. land -------------------------------------------------------------------------------
git push -q origin "$BRANCH":main
git pull -q --ff-only origin main
if [ -n "$WORKTREE" ]; then
    git worktree remove --force "$WORKTREE"
fi
git worktree prune
git branch -q -D "$BRANCH" 2>/dev/null || true

git log --oneline -1
echo "dirty-lines: $(git status --short | wc -l)"
