#!/usr/bin/env bash
#
# gen3ai bootstrap — fresh clone to a green test run, in one command.
#
#     ./scripts/bootstrap.sh              # everything except the Rust build (asks, if a TTY)
#     ./scripts/bootstrap.sh --with-rust  # everything, no questions
#     ./scripts/bootstrap.sh --dry-run    # print the plan, touch nothing
#
# Design notes, because a setup script that nobody trusts gets bypassed:
#
#   * IDEMPOTENT. Every step tests for its own result first and says "already done" rather
#     than redoing it. Re-running after a failure resumes; re-running after success is cheap
#     and safe. `--force` re-does the conda step specifically.
#   * THE CONDA ENV IS SHARED, SO A WORKTREE NEVER MUTATES IT SILENTLY. Every new run, gate and
#     agent on the box runs `gen3ai_torch28` (torch 2.8, `environment_torch28.yml` — the DEFAULT
#     since 2026-09-30, owner); a `conda env update --prune` under one of them can swap a package
#     out from under a live process. The "env is current" stamp therefore lives in the git COMMON
#     dir (one per clone, keyed by the env file's hash), so a worktree whose env is current does
#     nothing — and a worktree whose env file does NOT match REFUSES (exit 3) unless given
#     `--update-shared-env`; see step 2.
#   * IT NEVER TOUCHES `gen3ai_stable`. That env (torch 2.5.1, `environment.yml`) is LEGACY and
#     FROZEN: kept byte-for-byte so a run trained on it resumes on it (the launcher selects it from
#     the run's recorded torch — src/main/launcher/torch_runtime.py). Nothing here reads
#     `environment.yml` any more, and nothing may edit it.
#   * FAIL-LOUD. `set -euo pipefail` plus a trap that names the step that died and what to try.
#     A half-built environment that reports success is worse than no script.
#   * IT ANNOUNCES COST. Every step prints what it is about to do and roughly what it costs,
#     because the difference between "hung" and "downloading 2 GB of CUDA wheels" is the single
#     most common reason someone kills a bootstrap halfway.
#   * WORKTREE-AWARE. In a linked git worktree the submodule checkout has no build artifacts;
#     this symlinks them from the main checkout instead of rebuilding (see step 5).
#   * `--dry-run` EXISTS SO THIS IS TESTABLE. The conda/npm/cargo steps cannot run in CI or
#     beside a live training run, so every mutating command routes through `run()` and a dry
#     run exercises the whole control flow — detection, skip logic, ordering — for free.
#
# `-E` (errtrace) is load-bearing, not decoration: without it the ERR trap is NOT inherited by
# shell functions, so a failure inside `run()` — which is every mutating command in this script —
# exits silently with a bare status instead of printing the diagnosis below. Verified by
# injecting a failure; without -E the trap produced no output at all.
set -Eeuo pipefail

ENV_NAME="gen3ai_torch28"
#: THE env file. `environment.yml` (gen3ai_stable, torch 2.5.1) is the frozen LEGACY spec — never
#: read here, never updated (see the header).
ENV_FILE="environment_torch28.yml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SHOWDOWN_DIR="$REPO_ROOT/deps/pokemon-showdown"

DRY_RUN=0
FORCE=0
UPDATE_SHARED_ENV=0
SKIP_ENV=0
WITH_RUST="ask"
RUN_CHECK=1
CURRENT_STEP="startup"
TOTAL_STEPS=7

# ------------------------------------------------------------------------------- output helpers
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; RST=$'\033[0m'
else
    B=""; DIM=""; GRN=""; YEL=""; RED=""; RST=""
fi

step()  { CURRENT_STEP="$1"; printf '\n%s[%s/%s] %s%s\n' "$B" "$2" "$TOTAL_STEPS" "$1" "$RST"; }
cost()  { printf '      %scost: %s%s\n' "$DIM" "$1" "$RST"; }
info()  { printf '      %s\n' "$1"; }
skip()  { printf '      %s✓ already done%s — %s\n' "$GRN" "$RST" "$1"; }
did()   { printf '      %s✓ %s%s\n' "$GRN" "$1" "$RST"; }
warn()  { printf '      %s! %s%s\n' "$YEL" "$1" "$RST"; }

# Every mutating command goes through here, so --dry-run covers the whole script by construction
# rather than by remembering to guard each call site.
run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '      %s[dry-run] %s%s\n' "$DIM" "$*" "$RST"
        return 0
    fi
    "$@"
}

on_error() {
    local rc=$?
    printf '\n%s╔══════════════════════════════════════════════════════════════════════╗%s\n' "$RED" "$RST"
    printf '%s║ BOOTSTRAP FAILED%s during: %s (exit %s)\n' "$RED" "$RST" "$CURRENT_STEP" "$rc"
    printf '%s╚══════════════════════════════════════════════════════════════════════╝%s\n' "$RED" "$RST"
    printf '\nNothing was left half-installed on purpose — this script is idempotent, so fix\n'
    printf 'the cause above and re-run it; completed steps will be skipped.\n\n'
    printf 'If the failure is in the conda step, the usual causes are (a) no network, or\n'
    printf '(b) a partially-created env: %sconda env remove -n %s%s and re-run.\n\n' "$B" "$ENV_NAME" "$RST"
    exit "$rc"
}
trap on_error ERR

usage() {
    cat <<'EOF'
gen3ai bootstrap — fresh clone to a green test run.

Usage: ./scripts/bootstrap.sh [options]

  --with-rust     build the Rust simulator binaries (long; see step 6)
  --no-rust       skip them (a first test run then pays for the build itself)
  --dry-run       print every step and what it WOULD run; change nothing
  --force         redo the conda env step even if it looks current
  --update-shared-env
                  from a linked WORKTREE, allow step 2 to update the SHARED conda env
                  (refused by default — see step 2; only when no live or pinned run uses it)
  --skip-env      leave the conda env alone entirely (step 2 is skipped, nothing is checked
                  beyond a warning); the rest of the bootstrap runs as normal
  --no-check      skip the final verification (step 7)
  -h, --help      this

With neither --with-rust nor --no-rust, the script asks if stdin is a terminal and
skips the Rust build otherwise (so unattended runs never block on a prompt).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1 ;;
        --force)    FORCE=1 ;;
        --update-shared-env) UPDATE_SHARED_ENV=1 ;;
        --skip-env) SKIP_ENV=1 ;;
        --with-rust) WITH_RUST="yes" ;;
        --no-rust)  WITH_RUST="no" ;;
        --no-check) RUN_CHECK=0 ;;
        -h|--help)  usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$REPO_ROOT"

printf '%s╔══════════════════════════════════════════════════════════════════════╗%s\n' "$B" "$RST"
printf '%s║ gen3ai bootstrap%s\n' "$B" "$RST"
printf '%s╚══════════════════════════════════════════════════════════════════════╝%s\n' "$B" "$RST"
info "repo:   $REPO_ROOT"
[ "$DRY_RUN" -eq 1 ] && warn "DRY RUN — nothing will be modified"

# Where the stamps live. Inside .git so they are never committed and never in `git status`.
#
# 🚨 In the git COMMON dir — ONE per clone, shared by the main checkout and every linked worktree
# — because the thing the env stamp describes is ONE shared conda env. It used to be the per-
# worktree gitdir, so EVERY fresh worktree saw no stamp and ran `conda env update --prune` against
# the env that every live run, pinned launch and agent on the box was using (2026-09-29: one
# started under a live measurement job). A stamp must be scoped like the thing it stamps.
STATE_DIR="$(git rev-parse --git-common-dir 2>/dev/null || echo .git)/gen3ai-bootstrap"

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

# Is this a LINKED git worktree rather than the main checkout? Two steps need the answer — the
# editable install (step 3, which must NOT be made from a worktree) and the Showdown build
# artifacts (step 5, which symlinks instead of rebuilding) — so it is resolved once, here,
# rather than twice with two chances to disagree.
IS_WORKTREE=0
MAIN_CHECKOUT="$REPO_ROOT"
GIT_DIR="$(git rev-parse --git-dir)"
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
if [ "$(cd "$GIT_DIR" && pwd)" != "$(cd "$GIT_COMMON_DIR" && pwd)" ]; then
    IS_WORKTREE=1
    MAIN_CHECKOUT="$(dirname "$(cd "$GIT_COMMON_DIR" && pwd)")"
fi

# ══════════════════════════════════════════════════════════════ 1. prerequisites
step "Checking prerequisites" 1
cost "instant"
# A plain string, not an array: `${#arr[@]}` on an EMPTY array is an unbound-variable error
# under `set -u` in bash 3.2, which is still what /bin/bash is on macOS.
missing=""
for tool in git conda node npm; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    printf '\n%sMissing required tools:%s%s\n\n' "$RED" "$missing" "$RST"
    printf '  git   — https://git-scm.com\n'
    printf '  conda — https://docs.conda.io/en/latest/miniconda.html (Miniconda is enough)\n'
    printf '  node  — v20+ recommended; npm ships with it. Needed to BUILD the Showdown\n'
    printf '          submodule, which the Rust simulator is validated against.\n\n'
    exit 1
fi
did "git $(git --version | awk '{print $3}') · conda $(conda --version | awk '{print $2}') · node $(node --version) · npm $(npm --version)"

# ══════════════════════════════════════════════════════════════ 2. conda environment
#
# THE ENV IS SHARED; THE CHECKOUT IS NOT. What decides whether to touch it:
#
#   stamp env-<sha256 of $ENV_FILE> in the COMMON dir   → current: do nothing, anywhere.
#   no such stamp, run from the MAIN checkout                 → update in place (as always).
#   no such stamp, run from a linked WORKTREE                 → print what would change and
#                                                               REFUSE (exit 3), unless
#                                                               --update-shared-env is given.
#
# WHY REFUSE rather than warn-and-update or silently skip: an update that happens because someone
# made a worktree is exactly the unasked mutation the standing rule forbids ("never mutate
# the shared env under live or pinned runs"), and a worktree bootstrap cannot see which runs are
# queued or pinned. Silently skipping is no better — the worktree would then test against an env
# that does not match its $ENV_FILE and report green or red for the wrong reason. Refusing
# puts a human (or the orchestrator) in the loop at the one moment a decision is needed, and names
# both ways forward. `--skip-env` exists so that the refusal is never a dead end: without a
# non-mutating way past it, the opt-in flag would become the reflex.
#
# The stamp FILE holds a copy of the $ENV_FILE it was written for, so a refusal can print a
# real diff. One stamp at a time — writing a new one removes the rest, so reverting
# $ENV_FILE to an older version is (correctly) NOT mistaken for current.
step "Conda environment '$ENV_NAME'" 2
cost "~5-15 min on a fresh machine (≈2 GB of wheels, most of it CUDA); seconds when current"
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
ENV_HASH="$(hash_of "$ENV_FILE")"
ENV_STAMP="$STATE_DIR/env-$ENV_HASH"
ENV_CURRENT=0
[ -f "$ENV_STAMP" ] && ENV_CURRENT=1

# MIGRATION from the per-worktree stamps (which lived in $GIT_COMMON_DIR/worktrees/<wt>/…): if
# there is no common-dir stamp yet, and the NEWEST legacy stamp anywhere was written for THIS
# $ENV_FILE, the last successful update was for exactly this file — adopt it rather than
# refuse. The newest, not any: an older matching stamp may predate an update to something else.
if [ ! -d "$STATE_DIR" ] || [ -z "$(ls -A "$STATE_DIR" 2>/dev/null | grep '^env-' || true)" ]; then
    # shellcheck disable=SC2012  # names are env-<hex>; ls -t is the portable newest-first
    newest_legacy="$(ls -t "$GIT_COMMON_DIR"/worktrees/*/gen3ai-bootstrap/env-* 2>/dev/null | head -1 || true)"
    if [ -n "$newest_legacy" ] && [ "$(basename "$newest_legacy")" = "env-$ENV_HASH" ] && [ -d "$ENV_PREFIX" ]; then
        info "adopting the newest per-worktree stamp ($newest_legacy) — it records the last"
        info "successful update, and it was for this exact $ENV_FILE"
        run mkdir -p "$STATE_DIR"
        run cp "$ENV_FILE" "$ENV_STAMP"
        ENV_CURRENT=1
    fi
fi

# What would change: a diff against the $ENV_FILE the current stamp was written for.
show_env_change() {
    local prev
    prev="$(ls "$STATE_DIR"/env-* 2>/dev/null | head -1 || true)"
    if [ -n "$prev" ] && [ -s "$prev" ]; then
        info "$ENV_FILE vs the one $ENV_NAME was last built from ($(basename "$prev" | cut -c1-16)…):"
        diff -u --label "stamped (env is built from this)" --label "this checkout's $ENV_FILE" \
            "$prev" "$ENV_FILE" | sed 's/^/          /' || true
    elif [ -n "$prev" ]; then
        info "the env was last built from $ENV_FILE with hash $(basename "$prev" | cut -c5-20)…,"
        info "whose contents were not recorded (an older stamp) — no diff available."
    else
        info "NO stamp records which $ENV_FILE $ENV_NAME was built from, so what an update"
        info "would change is unknown. This checkout's $ENV_FILE hashes to ${ENV_HASH:0:16}…"
    fi
}

write_env_stamp() {
    run mkdir -p "$STATE_DIR"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '      %s[dry-run] replace the env stamp in %s with env-%s%s\n' "$DIM" "$STATE_DIR" "$ENV_HASH" "$RST"
        return 0
    fi
    find "$STATE_DIR" -maxdepth 1 -name 'env-*' ! -name "env-$ENV_HASH" -delete
    cp "$ENV_FILE" "$ENV_STAMP"
}

if [ "$SKIP_ENV" -eq 1 ]; then
    warn "SKIPPED (--skip-env) — $ENV_NAME is left exactly as it is, current or not."
    if [ "$ENV_CURRENT" -eq 0 ]; then
        warn "NOTE: no stamp says it matches this $ENV_FILE; tests may fail for env reasons."
    fi
elif [ -d "$ENV_PREFIX" ] && [ "$ENV_CURRENT" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
    skip "$ENV_PREFIX is current for this $ENV_FILE (stamp in $STATE_DIR; --force to redo)"
elif [ -d "$ENV_PREFIX" ] && [ "$IS_WORKTREE" -eq 1 ] && [ "$UPDATE_SHARED_ENV" -eq 0 ]; then
    if [ "$FORCE" -eq 1 ]; then
        warn "--force asks to update the SHARED env from a linked worktree."
    else
        warn "$ENV_FILE differs from what the SHARED env $ENV_NAME was built from."
    fi
    show_env_change
    printf '\n%s╔══════════════════════════════════════════════════════════════════════╗%s\n' "$YEL" "$RST"
    printf '%s║ REFUSED: this is a linked WORKTREE, and %s is SHARED%s\n' "$YEL" "$ENV_NAME" "$RST"
    printf '%s╚══════════════════════════════════════════════════════════════════════╝%s\n' "$YEL" "$RST"
    cat <<EOF

Every training run, pinned launch and agent on this box runs $ENV_PREFIX.
\`conda env update --prune\` under one of them can swap a package out from under a live
process, and a worktree bootstrap cannot see which runs are live, queued or pinned.
Standing rule: never mutate $ENV_NAME under live or pinned runs.

Pick one:
  * update it deliberately, once no live or pinned run uses it — from the MAIN checkout:
        cd $MAIN_CHECKOUT && ./scripts/bootstrap.sh
    or from here, naming the intent:
        ./scripts/bootstrap.sh --update-shared-env
  * leave the env alone and finish this worktree's setup (submodule, symlinks, checks):
        ./scripts/bootstrap.sh --skip-env

EOF
    exit 3
elif [ -d "$ENV_PREFIX" ]; then
    if [ "$IS_WORKTREE" -eq 1 ]; then
        warn "--update-shared-env: updating the SHARED env from a linked worktree"
    fi
    warn "$ENV_NAME is SHARED by every run, pin and agent on this box — this update must not"
    warn "run under a live or pinned run (nothing here can check that for you)."
    info "env exists but $ENV_FILE changed (or --force) — updating in place"
    show_env_change
    run conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    write_env_stamp
    did "updated"
else
    # No env at all: creating one cannot disturb a process that uses it, so no refusal here.
    info "creating from $ENV_FILE"
    info "the pip block carries --extra-index-url for the pytorch cu126 wheels; the"
    info "torch download alone is several hundred MB, so this looks idle for a while. It is not."
    run conda env create -f "$ENV_FILE"
    write_env_stamp
    did "created"
fi
PY="$ENV_PREFIX/bin/python3"

# ══════════════════════════════════════════════════════════════ 3. import path
#
# `pip install -e .` writes ONE `.pth` file naming this checkout's `src/`, and that is the
# entire point: `import agents` then works from any directory with nothing exported. It is the
# replacement for `export PYTHONPATH=$PYTHONPATH:src`, and both keep working (PYTHONPATH simply
# outranks the `.pth` — which is deliberate, see src/packaging_gate_test.py).
#
# pyproject.toml declares NO dependencies, so this cannot resolve, upgrade or replace anything
# in the env; it writes a `.pth` and a `dist-info` and stops. `--no-deps` says so a second time,
# and `--no-build-isolation` keeps pip from fetching a build backend it does not need (setuptools
# is already in the env), so this step works offline.
step "Python import path (editable install)" 3
cost "~2 s"
if [ "$IS_WORKTREE" -eq 1 ]; then
    # An editable install records ONE ABSOLUTE PATH. Made from a worktree, it points at a
    # directory that gets deleted — and Python skips a missing .pth entry SILENTLY, so imports
    # later fail for a reason the install never reports. The main checkout's install already
    # covers this worktree's needs for anything PYTHONPATH-free; anything worktree-specific
    # wants PYTHONPATH anyway (that is how the launcher pins a run to its commit).
    warn "SKIPPED — this is a linked worktree, and an editable install must be made from the"
    warn "MAIN checkout only ($MAIN_CHECKOUT). Run this script there once; a worktree needs"
    info "  export PYTHONPATH=\$PYTHONPATH:src"
    info "for anything that must import THIS worktree's code rather than the main checkout's."
# The honest test of "is it installed" is whether a src-rooted import RESOLVES without
# PYTHONPATH — not whether pip lists the distribution, which stays listed after the checkout
# it points at has moved or been deleted.
elif env -u PYTHONPATH "$PY" -c "
import importlib.util, pathlib, sys
s = importlib.util.find_spec('agents')
sys.exit(0 if s and str(pathlib.Path(s.origin).resolve()).startswith('$REPO_ROOT/src') else 1)
" 2>/dev/null; then
    skip "\`import agents\` already resolves to $REPO_ROOT/src with no PYTHONPATH"
else
    info "installing this checkout as an editable package (no dependencies are declared,"
    info "so nothing else in the env is touched)"
    run "$PY" -m pip install -e . --no-deps --no-build-isolation
    did "pip install -e . — \`import agents\` now works from anywhere"
fi

# ══════════════════════════════════════════════════════════════ 4. submodule
step "Showdown submodule (deps/pokemon-showdown)" 4
cost "~30 s first time; instant after"
if [ -f "$SHOWDOWN_DIR/package.json" ]; then
    skip "submodule already checked out"
else
    info "a plain 'git clone' leaves this directory empty"
    run git submodule update --init
    did "checked out"
fi

# The `keep-ours` merge driver `.gitattributes` names for the GENERATED ledger index (the shared
# .git config, so every worktree gets it). The ship skill regenerates the index after a rebase.
if [ "$(git config --get merge.keep-ours.driver || true)" = "true" ]; then
    skip "merge driver keep-ours already configured"
else
    run git config merge.keep-ours.driver true
    run git config merge.keep-ours.name "keep our side (the file is regenerated after the merge)"
    did "merge driver keep-ours configured"
fi

# ══════════════════════════════════════════════════════════════ 5. showdown build artifacts
#
# TWO PATHS, and picking the right one matters. A linked git worktree gets its own (empty)
# submodule checkout, but the main checkout beside it already has `node_modules/` and `dist/`
# built — so a worktree symlinks rather than spending five minutes rebuilding them. Both are
# gitignored inside the submodule, so neither shows up in `git status`.
#
# IS_WORKTREE / MAIN_CHECKOUT are resolved once near the top of the script (step 3 needs the
# same answer). Note this branch may CLEAR IS_WORKTREE below to fall through to a local build —
# that is a decision about artifacts only, and it happens after step 3 has already used it.
step "Showdown build artifacts (node_modules + dist)" 5

if [ "$IS_WORKTREE" -eq 1 ]; then
    cost "instant (symlinks)"
    info "this is a LINKED WORKTREE — linking artifacts from the main checkout instead of rebuilding"
    info "main checkout: $MAIN_CHECKOUT"
    # STICKY: once any artifact cannot be linked we fall through to the local build path for
    # the whole step. A per-iteration flag would be overwritten by a later successful link and
    # leave the missing artifact silently unbuilt.
    need_local_build=0
    for artifact in dist node_modules; do
        target="$MAIN_CHECKOUT/deps/pokemon-showdown/$artifact"
        link="$SHOWDOWN_DIR/$artifact"
        # ⚠️ The [ -e ] guard is NOT optional and is why this loop is not a bare `ln -sf`.
        # If the NAME already exists as a real directory (which it does in the main checkout),
        # `ln -s TARGET dist` creates the link INSIDE it as dist/dist -> its own parent, and
        # `node build` then dies with ELOOP and every websocket-server path stops working.
        # That happened here on 2026-07-23 and went unnoticed for four weeks.
        if [ -e "$link" ]; then
            skip "$artifact already present"
        elif [ ! -e "$target" ]; then
            warn "$target does not exist — build it in the main checkout first, or run this"
            warn "script there. Falling back to a local build for '$artifact'."
            need_local_build=1
        else
            run ln -s "$target" "$link"
            did "linked $artifact -> $target"
        fi
    done
    if [ "$need_local_build" -eq 1 ]; then
        IS_WORKTREE=0   # fall through to the build path below
    fi
fi

if [ "$IS_WORKTREE" -eq 0 ]; then
    cost "~3-6 min first time (npm ci + a TypeScript build); instant after"
    if [ -e "$SHOWDOWN_DIR/node_modules" ]; then
        skip "node_modules present"
    else
        info "installing Showdown's dependencies from its committed package-lock.json"
        # `npm ci` not `npm install`: the submodule ships a lockfile, and ci honours it exactly
        # (and is the faster of the two on a cold tree).
        run npm ci --prefix "$SHOWDOWN_DIR"
        did "npm ci"
    fi
    # `npm run build` in the submodule is `node build` — it compiles the TypeScript sim into
    # dist/. dist/sim/index.js is the file the Python bridge actually loads, so its presence is
    # the honest test of "is this built", not the existence of the directory.
    if [ -f "$SHOWDOWN_DIR/dist/sim/index.js" ]; then
        skip "dist/sim/index.js present"
    else
        info "compiling the simulator (node build)"
        run npm run build --prefix "$SHOWDOWN_DIR"
        did "built dist/"
    fi
fi

# ══════════════════════════════════════════════════════════════ 6. rust simulator (optional)
step "Rust simulator binaries (optional)" 6
cost "~3-10 min of a FULLY SATURATED box on a cold build; near-instant when warm"
if [ "$WITH_RUST" = "ask" ]; then
    if [ -t 0 ] && [ "$DRY_RUN" -eq 0 ]; then
        info "Training runs on the in-process Rust bridge (the only transport), and the FIRST Rust-backed test builds"
        info "these anyway — mid-test, saturating every core, which is a known cause of"
        info "spurious timeout failures. Building now is strictly better if you have the time."
        read -r -p "      Build them now? [y/N] " reply
        case "$reply" in [yY]*) WITH_RUST="yes" ;; *) WITH_RUST="no" ;; esac
    else
        WITH_RUST="no"
        info "not a terminal (or --dry-run) — skipping; pass --with-rust to build"
    fi
fi

if [ "$WITH_RUST" = "no" ]; then
    warn "SKIPPED. Your first Rust-backed test will pay for this build itself. To do it later:"
    info "  cargo build --release --bin sim_bridge --bin search_driver \\"
    info "      --manifest-path src/rust_sim/Cargo.toml"
elif ! command -v cargo >/dev/null 2>&1; then
    warn "cargo not found — skipping. Install Rust from https://rustup.rs and re-run with"
    warn "--with-rust. Training and the Rust-backed tests need the binaries; only the offline tools keep a node transport (--bridge node)."
else
    info "cargo is incremental — a warm tree finishes in seconds"
    run cargo build --release --bin sim_bridge --bin search_driver \
        --manifest-path src/rust_sim/Cargo.toml
    did "sim_bridge + search_driver built"
fi

# ══════════════════════════════════════════════════════════════ 7. verify
step "Verifying" 7
if [ "$RUN_CHECK" -eq 0 ]; then
    warn "skipped (--no-check)"
elif [ "$DRY_RUN" -eq 1 ]; then
    printf '      %s[dry-run] ruff gate, mypy gate, and a ~30 s unit-test smoke%s\n' "$DIM" "$RST"
else
    cost "~30-60 s (the mypy gate is 20 s cold, 0.3 s warm)"

    # FIRST, and deliberately BEFORE the PYTHONPATH export below: does step 3 actually work?
    # Verifying the import path after exporting PYTHONPATH would pass whether or not the
    # editable install landed — the export is exactly the thing it is supposed to replace.
    if [ "$IS_WORKTREE" -eq 0 ]; then
        info "the editable install (no PYTHONPATH — this is the point of step 3) ..."
        env -u PYTHONPATH "$PY" -c "
import agents, main, utils, pathlib
for m in (agents, main, utils):
    assert str(pathlib.Path(m.__file__).resolve()).startswith('$REPO_ROOT/src'), m
"
        did "import agents / main / utils all resolve to $REPO_ROOT/src"
    fi

    export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

    info "the two static gates (ruff + mypy) ..."
    "$PY" -m pytest src/ruff_gate_test.py src/agents/model/mypy_gate_test.py -q
    did "static gates pass"

    info "the import-precedence gates (no poke_env anywhere + PYTHONPATH vs .pth) ..."
    "$PY" -m pytest src/poke_env_absent_gate_test.py src/packaging_gate_test.py -q
    did "no poke_env package in the tree or the env, and PYTHONPATH still outranks the editable install"

    # A SLICE, not the whole tier: ~600 tests over the obs encoder, the action layer, the
    # event-sourced battle layer and the shared utils — the four packages that actually break
    # when the environment is wrong (a missing data file, a half-built submodule, the wrong
    # numpy). ~10 s. The full ~10,000-test inner loop is what you run next, not what you wait for
    # here; a bootstrap that ends in a two-minute test run gets interrupted.
    info "a ~10 s slice of the unit suite (obs / action / battle / utils) ..."
    "$PY" -m pytest src/agents/observation src/agents/action src/agents/battle src/utils \
        -m "not slow and not e2e and not sim and not integration" -q -n 2
    did "unit smoke passes"
fi

# ══════════════════════════════════════════════════════════════ what next
cat <<EOF

$B╔══════════════════════════════════════════════════════════════════════╗$RST
$B║ Bootstrap complete.$RST
$B╚══════════════════════════════════════════════════════════════════════╝$RST

Activate the environment. Nothing else — step 3 installed this checkout as an
editable package, so \`import agents\` works from any directory with no exports:

    conda activate $ENV_NAME

(In a linked WORKTREE step 3 is skipped on purpose — an editable install records
one absolute path, and a worktree's path goes away. There, and on any machine
without the install, the old incantation is the equivalent fallback:
\`export PYTHONPATH=\$PYTHONPATH:src\`.)

Then, in rough order of usefulness:

    # the routine gate — everything cheap, run this before any commit (~4.5 min)
    pytest src/ -m "not slow and not e2e" -q -n 6

    # a 1-minute training smoke: no server, no GPU needed
    python src/main/train_rl_agent.py --debug --steps 10000

    # the forensic replay inspector, on a real run (web UI on :6008)
    python -m main.prober models/<run>

Where to read next:
    CONTRIBUTING.md          how to work in this repo (tests, ports, worktrees)
    docs/RUNNING.md          training, evaluation, the test tiers
    designs/ARCHITECTURE.md  the model, as it is today
EOF
