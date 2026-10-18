#!/usr/bin/env bash
# THE REMOTE BOOTSTRAP for a rented benchmark box (runs ON the box, as root). Idempotent: each step drops a
# marker in $PROV/done/ and is skipped on a re-run; every step's wall time lands in $PROV/times.tsv
# (step, start epoch, end epoch, seconds, rc). Nothing private is used or written here: the repo and the
# Showdown submodule are PUBLIC and cloned over https, and no key or token ever reaches this machine.
#
#   bash remote_bootstrap.sh <step|all|setup> [<commit>]
#
# setup = apt uv python venv pip torch_sanity clone submodule node showdown rust rust_build (no GPU work)
# all   = setup + smoke + gpu_sanity
# bench = THE FROZEN SUITE (python -m agents.training.gpu_bench_suite run; resumable, ~35-60 min on the card);
#         BENCH_TAG / BENCH_META ("--meta dph=0.26 --meta geo=KR") are recorded in its JSON
# Inputs, scp'd next to this script by the driver (tools/cloud/vast/README.md): the pip requirements ($REQS, default
# requirements_cu128.txt; a cu126 box gets requirements_cu126.txt — both derived by tools/cloud/vast/requirements.py)
# and remote_sanity.py.
set -uo pipefail
PROV=${PROV:-/root/prov}
REPO=${REPO:-/root/gen3ai}
VENV=${VENV:-/root/venv}
REPO_URL=${REPO_URL:-https://github.com/JGoodlad/gen3ai.git}
NODE_VERSION=${NODE_VERSION:-v22.22.1}       # the local box's node (2026-10-12)
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.96.0}     # the local box's rustc (2026-10-12)
PY_VERSION=${PY_VERSION:-3.11.15}            # the shared env's exact python (a bare 3.11 gave 3.11.17, 2026-10-12)
HERE="$(cd "$(dirname "$0")" && pwd)"
REQS=${REQS:-$HERE/requirements_cu128.txt}
STEP=${1:?step}
COMMIT=${2:-$(cat "$PROV/commit" 2>/dev/null || echo main)}
mkdir -p "$PROV/done" "$PROV/logs" "$PROV/results"
echo "$COMMIT" > "$PROV/commit"
export PATH="/root/.local/bin:/root/.cargo/bin:/opt/node/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive

timed() {   # timed <name> <cmd...>: run once (marker), log, record the wall time
    local name=$1; shift
    if [ -e "$PROV/done/$name" ]; then echo "[prov] $name: done already"; return 0; fi
    echo "[prov] $name: START $(date -u +%T)"
    local t0 t1 rc
    t0=$(date +%s.%N)
    ( "$@" ) > "$PROV/logs/$name.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    local dt
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b - a}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$t0" "$t1" "$dt" "$rc" >> "$PROV/times.tsv"
    echo "[prov] $name: rc=$rc in $dt s (log $PROV/logs/$name.log)"
    if [ "$rc" -eq 0 ]; then touch "$PROV/done/$name"; else tail -30 "$PROV/logs/$name.log"; fi
    return "$rc"
}

s_apt() {
    apt-get update -q && apt-get install -y -q --no-install-recommends git curl ca-certificates build-essential \
        pkg-config xz-utils >/dev/null && git --version && gcc --version | head -1
}
s_uv() { curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version; }
s_python() { uv python install "$PY_VERSION" && uv venv -p "$PY_VERSION" "$VENV" && "$VENV/bin/python" -V; }
s_pip() {
    uv pip install -p "$VENV/bin/python" --index-strategy unsafe-best-match -r "$REQS" &&
        uv pip check -p "$VENV/bin/python" && uv pip freeze -p "$VENV/bin/python" > "$PROV/results/pip_freeze.txt"
}
s_torch_sanity() { "$VENV/bin/python" "$HERE/remote_sanity.py" --out "$PROV/results/torch_sanity.json"; }
s_clone() {   # a SHALLOW fetch of the pinned commit (the repo's history is not needed to train)
    # GitHub serves a FULL 40-hex SHA (or a ref) to `fetch --depth 1`, never an abbreviation: a short SHA failed with
    # "couldn't find remote ref" on the 2026-10-18 ladder's first box. Refuse it up front with the reason.
    if [[ "$COMMIT" =~ ^[0-9a-f]{7,39}$ ]]; then
        echo "REFUSED: '$COMMIT' is an abbreviated SHA; pass the full 40-hex commit (git rev-parse HEAD)"; return 2
    fi
    rm -rf "$REPO" && git init -q "$REPO" && git -C "$REPO" remote add origin "$REPO_URL" &&
        git -C "$REPO" fetch -q --depth 1 origin "$COMMIT" && git -C "$REPO" checkout -q FETCH_HEAD &&
        git -C "$REPO" log --oneline -1
}
s_submodule() {
    git -C "$REPO" submodule update --init --depth 1 || git -C "$REPO" submodule update --init
    git -C "$REPO" submodule status
}
s_node() {
    rm -rf /opt/node && mkdir -p /opt/node &&
        curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" |
        tar -xJ --strip-components=1 -C /opt/node && node --version && npm --version
}
s_showdown() {   # the team validator's runtime need (root CLAUDE.md: Node's ONLY runtime job)
    npm ci --prefix "$REPO/deps/pokemon-showdown" && npm run build --prefix "$REPO/deps/pokemon-showdown"
}
s_rust() {
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN" &&
        rustc --version && cargo --version
}
s_rust_build() {   # the env core where its loaders look (utils.rust_env.build), + the in-process sim bridge
    (cd "$REPO/src/rust_env" && CARGO_TARGET_DIR=$PWD/target cargo build --release --lib --bin rust_env_proc) &&
        cargo build --release --bin sim_bridge --manifest-path "$REPO/src/rust_sim/Cargo.toml"
}
s_smoke() {   # the repo's CPU smoke (root CLAUDE.md "Smoke Test"), runs into a scratch archive
    cd "$REPO" && mkdir -p /root/runs &&
        PYTHONPATH="$REPO/src" GEN3AI_MODELS_DIR=/root/runs "$VENV/bin/python" src/main/train_rl_agent.py --debug --steps 10000
}
s_gpu_sanity() {   # ONE update-bench call on the end state: proves compile + CUDA on sm_120. SANITY, NOT A BENCHMARK.
    local B="$REPO/designs/research_state/measurements/perf_phase_2026-10-10/scripts"
    cd "$REPO" && PYTHONPATH="$REPO/src" DIAG_SRC="$REPO/src" "$VENV/bin/python" "$B/update_bench.py" --tag E_sanity \
        --argv "$("$VENV/bin/python" "$B/configs.py" --argv E)" --epochs 1 --warm 0 --reps 1 \
        --out "$PROV/results/E_sanity.json"
}

s_bench() {   # THE FROZEN GPU BENCHMARK SUITE (src/agents/training/gpu_bench_suite.py); incremental, so a re-run resumes
    cd "$REPO" && PYTHONPATH="$REPO/src" GEN3AI_BENCH_COMMIT="$COMMIT" "$VENV/bin/python" -m agents.training.gpu_bench_suite \
        run --out "$PROV/results/suite" --tag "${BENCH_TAG:-}" ${BENCH_META:-}
}

SETUP=(apt uv python pip torch_sanity clone submodule node showdown rust rust_build)
case "$STEP" in
    setup) steps=("${SETUP[@]}") ;;
    all) steps=("${SETUP[@]}" smoke gpu_sanity) ;;
    *) steps=("$STEP") ;;
esac
for s in "${steps[@]}"; do
    timed "$s" "s_$s" || { echo "[prov] STOPPED at $s"; exit 1; }
done
echo "[prov] OK: ${steps[*]}"
