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
#   2. `git push origin <branch>:main` from the MAIN checkout — after the PUSH GUARD
#      (`python -m utils.push_guard`) finds no file the push would change on main that the branch
#      never changed (2026-10-01: a soft reset onto a moved origin/main pushed 43 stale files).
#      Anyone pushing by hand runs it first too: `python -m utils.push_guard` in the worktree.
#   3. `git pull --ff-only origin main` — so the main checkout is not left behind its own remote.
#   4. `git worktree remove --force <worktree>` + prune + delete the local branch — but only
#      after the RUN-DATA GUARD (`python -m utils.worktree_guard`) finds nothing to lose.
#
# WHY STEP 4 HAS A GUARD (incident 2026-09-23). Early launcher worktrees wrote run directories
# INSIDE the worktree and left only a SYMLINK in the main checkout's `models/`; `models/` is
# gitignored, so a clean `git status` hid them and `git worktree remove --force` destroyed eight
# runs. The guard REFUSES the removal when (a) any `models/` entry of the main checkout is a symlink
# resolving into the worktree, or (b) the worktree's untracked + ignored content, minus an
# allowlist of build/cache paths, exceeds 50 MiB. It runs AFTER the push: the code is landed
# either way, and a refusal keeps the worktree and its branch and exits 3. It fails CLOSED — a
# guard that cannot run is a refusal. Tested at the blocked state by
# `src/utils/worktree_guard_test.py`, which drives this script end to end in a throwaway repo.
#
# WHY THE GATES ARE NOT PIPED. On 2026-09-06 a `pytest | tail` swallowed a ruff F811 and a
# duplicate definition landed on main as `00772d05`. Every gate here runs unpiped, each one's exit
# status is checked explicitly (never left to `set -e`, see the gate block), and a failure prints
# "GATE FAILED (<gate>) — not pushing" and exits 1.
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
# The python interpreter is `$GEN3AI_PYTHON` if set, else the box's `gen3ai_torch28` env (torch 2.8,
# the default since 2026-09-30; `gen3ai_stable` is legacy, for old-run resumes only), else
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
    4. RUN-DATA GUARD, then git worktree remove --force + prune + delete the
       local branch. The guard REFUSES (worktree and branch kept, exit 3) when a
       main-checkout models/ symlink resolves into the worktree, or when its
       untracked/ignored content outside the build/cache allowlist exceeds 50 MiB

    Any gate failure exits 1 WITHOUT pushing. It never force-pushes; a rejected
    non-fast-forward push means someone else landed first — rebase the worktree on
    main, resolve, and run this again.

ENVIRONMENT
    GEN3AI_PYTHON     interpreter to run the gates with (default: the gen3ai_torch28 env)

EXIT
    0  landed        1  a gate failed, or the push was rejected        2  bad usage
    3  landed, but the worktree was NOT removed: it holds run data (or the guard
       could not run) — move or delete the data, then remove it by hand
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

BRANCH="$1"
WORKTREE="${2:-}"
# Absolute from here on: step 4 runs after `cd "$MAIN_CHECKOUT"`, where a relative path would name
# a different directory (or none) — and the guard must inspect exactly the tree being removed.
if [ -n "$WORKTREE" ]; then
    WORKTREE="$(cd "$WORKTREE" && pwd)" || { echo "no such worktree: ${2}" >&2; exit 2; }
fi

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

PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
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
# 🚨 Every gate is checked EXPLICITLY (`|| gate_failed`). A `( … ) || { … }` block does NOT stop at
# a failing command: bash IGNORES `set -e` inside any command that is the left side of `||`, so the
# subshell's status was its LAST command's, the closing `echo … OK`. From the script's creation
# until 2026-10-04 no gate here could fail a landing (found when pytest refused a bare worktree
# with a UsageError and the step still printed OK). Pinned by `src/main/ops/land_gates_test.py`.
gate_failed() { echo "GATE FAILED ($1) — not pushing"; exit 1; }
if [ -n "$WORKTREE" ]; then
    cd "$WORKTREE"
    # MANDATORY in a worktree: `pip install -e .` names the MAIN checkout's src/, so without
    # this a worktree's pytest imports main's code and every result is about a tree nobody
    # edited (root CLAUDE.md → Python Environment).
    GATE_PYTHONPATH="${PYTHONPATH:-}:src"
    PYTHONPATH="$GATE_PYTHONPATH" "$PY" -m ruff check src/agents src/main src/utils \
          --select F,E9 --exclude src/poke_env --exclude src/rust_sim || gate_failed ruff
    PYTHONPATH="$GATE_PYTHONPATH" "$PY" -m mypy src/agents/model >/dev/null || gate_failed mypy
    # The statics include Showdown-backed gates (the teambuilder pack guard), so a worktree without
    # the submodule + the two build symlinks is REFUSED by the root conftest's deps guard, with the
    # fix in its message (`./scripts/bootstrap.sh --skip-env`). Never skip that guard here.
    PYTHONPATH="$GATE_PYTHONPATH" "$PY" -m pytest src/*_gate_test.py -q -p no:cacheprovider >/dev/null \
        || gate_failed "src/*_gate_test.py"
    echo "gates: ruff + mypy + src/*_gate_test.py OK"
    cd "$MAIN_CHECKOUT"
fi
# A test seam: stop after the gates (nothing fetched, pushed or removed).
[ -n "${_LAND_GATES_ONLY:-}" ] && exit 0

# --- 2-4. land -------------------------------------------------------------------------------
# THE PUSH GUARD (gen3_push_guard_v1): refuse a push whose tree would change, on main, a file this
# branch never changed — a stale copy (2026-10-01: a soft reset onto a moved origin/main reverted 43
# files of other commits for ~6 minutes). Exit 1, nothing pushed.
git fetch -q origin main
if ! PYTHONPATH="$MAIN_CHECKOUT/src" "$PY" -m utils.push_guard --repo "${WORKTREE:-$MAIN_CHECKOUT}" \
        --branch "$BRANCH" --remote origin/main; then
    echo "PUSH GUARD REFUSED — not pushing"; exit 1
fi
git push -q origin "$BRANCH":main
git pull -q --ff-only origin main
if [ -n "$WORKTREE" ]; then
    # 4a. THE RUN-DATA GUARD — from the MAIN checkout's src/ (just fast-forwarded, so it is at
    # least as new as the branch that landed), via -m and never by file path (see the module).
    if ! PYTHONPATH="$MAIN_CHECKOUT/src" "$PY" -m utils.worktree_guard \
            --main "$MAIN_CHECKOUT" "$WORKTREE"; then
        echo "LANDED, but the worktree was NOT removed: $WORKTREE (branch $BRANCH kept)." >&2
        echo "  Move or delete the data the guard names, then remove it by hand." >&2
        git log --oneline -1
        exit 3
    fi
    git worktree remove --force "$WORKTREE"
fi
git worktree prune
git branch -q -D "$BRANCH" 2>/dev/null || true

git log --oneline -1
echo "dirty-lines: $(git status --short | wc -l)"
