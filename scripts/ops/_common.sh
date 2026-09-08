# shellcheck shell=bash
#
# _common.sh — path and interpreter resolution shared by the scripts/ops/ watchers and reads.
#
# SOURCED, never executed. It defines functions and sets nothing global except what a caller
# asks for, so a caller keeps its own `set -u` / `set -e` discipline.
#
# WHY THE PATHS ARE DERIVED. These scripts were promoted out of a session-scoped temporary
# directory where every path was a literal (`/home/goodlad/dev/gen3ai/models/<run>`). A literal
# is correct on exactly one box and, worse, it is correct on that box even when the caller meant
# a different checkout. The derivation here is the one `scripts/land.sh` uses: the MAIN checkout
# is `dirname` of `git rev-parse --git-common-dir`, which resolves to the main checkout from
# inside a linked worktree as well.
#
# WHY `models/` GETS ITS OWN QUESTION. `models/` is not committed and exists only in the MAIN
# checkout — a worktree has none. This mirrors `utils.paths.main_models_dir()`, INCLUDING its
# contract: `$GEN3AI_MODELS_DIR` overrides and is AUTHORITATIVE (set-but-not-a-directory means
# "no archive", never a quiet fall-back), and the absence of an archive is a REFUSAL with a
# message, never a path that silently globs to nothing.

# The checkout these scripts came from (this file is scripts/ops/_common.sh).
ops_repo_root() {
    (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
}

# The MAIN checkout — the one that owns `models/`. Falls back to this checkout outside git.
ops_main_checkout() {
    local here
    here="$(ops_repo_root)"
    (cd "$here" && cd "$(git rev-parse --git-common-dir 2>/dev/null)" && cd .. && pwd) \
        2>/dev/null || printf '%s\n' "$here"
}

# The run archive, or exit status 1 with nothing on stdout. Never prints a path that is not one.
ops_models_dir() {
    if [ -n "${GEN3AI_MODELS_DIR:-}" ]; then
        # AUTHORITATIVE: set but not a directory means "no archive", not "look elsewhere".
        [ -d "$GEN3AI_MODELS_DIR" ] || return 1
        (cd "$GEN3AI_MODELS_DIR" && pwd)
        return 0
    fi
    local m
    m="$(ops_main_checkout)/models"
    [ -d "$m" ] || return 1
    printf '%s\n' "$m"
}

ops_no_archive_message() {
    cat >&2 <<'EOF'
REFUSING: no run archive — the main checkout has no models/ directory.
  models/ is NOT committed; it exists only on a machine that has trained, and only in the
  MAIN checkout (a linked worktree has none). Set $GEN3AI_MODELS_DIR if the archive lives
  elsewhere. Nothing is read and nothing is concluded.
EOF
}

# ops_resolve_run <run-name|run-dir> -> the run directory on stdout.
# A bare NAME is looked up in the archive; anything containing a slash is taken as a path and
# must exist. A miss is a refusal with the reason, never an empty string a caller can join.
ops_resolve_run() {
    local a="${1:-}"
    if [ -z "$a" ]; then
        echo "REFUSING: no run given. Pass a run NAME (looked up in models/) or a run DIRECTORY." >&2
        return 2
    fi
    if [ -d "$a" ]; then
        (cd "$a" && pwd)
        return 0
    fi
    case "$a" in
        */*|.*)
            echo "REFUSING: no such run directory: $a" >&2
            return 2
            ;;
    esac
    local m
    if ! m="$(ops_models_dir)"; then
        ops_no_archive_message
        return 2
    fi
    if [ ! -d "$m/$a" ]; then
        echo "REFUSING: no run '$a' under $m" >&2
        return 2
    fi
    printf '%s\n' "$m/$a"
}

# The interpreter, same precedence as scripts/land.sh: $GEN3AI_PYTHON, else this box's
# gen3ai_stable env, else whatever python3 is on PATH.
ops_python() {
    local py="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}"
    [ -x "$py" ] || py="$(command -v python3)"
    printf '%s\n' "$py"
}
