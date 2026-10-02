"""The eval cycle's LAUNCH mechanics — split out of ``eval_callback.py`` (2026-10-01).

Shared by BOTH eval callbacks (``PerOpponentEvalCallback``, ``SelfPlayCallback``) so the bot-eval and
self-play-eval cycles spawn identically: the per-cycle manifest (``write_eval_manifest`` — which model,
which regime, which selection rule), the per-process account nonce, the hung-cycle bound, the Rust
eval core's in-process cycle (``launch_rust_eval_cycle``) and the Python worker subprocesses
(``spawn_eval_workers`` / ``kill_eval_workers``). ``eval_callback`` re-exports every public name here;
a test that STUBS ``subprocess.Popen`` for the workers reaches it through THIS module's ``subprocess``.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from agents.training.eval_player import ForensicQuota, _rule_for
from agents.training.trace_selection import SELECTION_SCHEMA
from main.launcher.ipc import send_event
from utils.contention import scale_timeout
from utils.git import get_git_hash

EVAL_MANIFEST_NAME = "eval_manifest.json"
EVAL_SNAPSHOT_NAME = "snapshot.zip"


def _read_run_identity(model_dir: str) -> tuple:
    """(git_hash, arch_signature, config_version) for this run, from its on-disk
    model_config.json / metadata.json (written at run start), with a git fallback."""
    arch_signature = config_version = git_hash = None
    try:
        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)
        arch_signature = cfg.get("arch_signature")
        config_version = cfg.get("config_version")
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(model_dir, "metadata.json")) as f:
            git_hash = json.load(f).get("git_hash")
    except (OSError, ValueError):
        pass
    if not git_hash:
        try:
            git_hash = get_git_hash()
        except Exception:  # noqa: BLE001 — identity is best-effort
            git_hash = None
    return git_hash, arch_signature, config_version


def opponent_pins_of(fixed_opponents) -> dict:
    """``{ext label: the opponent's pinned team(s)}`` for :func:`write_eval_manifest` — EVERY pinned team.

    A multi-team specialist is measured sampling among ALL its pins (``EvalItem.fixed_from_cfg``,
    F-LH-13), so the manifest records all of them (a LIST, fingerprinted per team by ``_sha``); a
    single-team opponent keeps the one-sha shape. Until 2026-09-30 both callbacks passed ``team_str``
    (the FIRST pin) and the manifest misdescribed a multi-team opponent as single-team."""
    out = {}
    for e in fixed_opponents:
        strs = [t for t in (getattr(e, "team_strs", None) or ()) if t]
        out[e.label] = strs if len(strs) > 1 else (strs[0] if strs else getattr(e, "team_str", None))
    return out


def write_eval_manifest(model_dir: str, step: int, *, opponents, n_games: int,
                        snapshot: "str | None" = None,
                        trainee_team_str: "str | list[str] | None" = None,
                        opponent_pins: "dict | None" = None,
                        quota: "ForensicQuota | dict | None" = None) -> dict:
    """Write ``<model_dir>/eval_traces/step_<N>/eval_manifest.json`` — the per-cycle
    record of *exactly which model* produced this cycle's forensic traces.

    `snapshot` is the relative filename of the persisted weight snapshot
    (``snapshot.zip``) when `--keep-eval-snapshots` retained it this cycle, else None;
    the prober uses it to reload the bit-exact model, falling back to the nearest
    persisted checkpoint when absent.

    The manifest also records the EVAL REGIME, so a trace dir is self-describing about how its
    numbers were measured (the OOD-eval era was invisible precisely because this was missing):
    ``matchup_hash`` (the run's declared-matchup tag, read from the run metadata),
    ``trainee_team_sha`` (the pin the trainee piloted — None = the default pool builder), and
    ``opponent_pins`` ({ext label: sha, or a LIST of per-team shas for a multi-team opponent} for
    stable/exploiter opponents measured on their OWN pinned team(s) — the fold-back contract;
    :func:`opponent_pins_of`).

    It ALSO records the trace SELECTION (`gen3_trace_selection_manifest_v1`): `selection_rule`
    names the outcome quota in words, and `selection` is filled in at COLLECT time by
    :func:`record_eval_selection` with the per-opponent played/won/traced counts. It is written
    here as ``null`` rather than omitted, because a cycle that crashed before collecting must
    read as SELECTION UNKNOWN and not as "no traces were kept".
    """
    git_hash, arch_signature, config_version = _read_run_identity(model_dir)
    from agents.model.snapshot import _read_matchup_hash
    def _sha(t):
        """Team fingerprint(s) — one sha for a single pin, a LIST for a multi-team pin.

        A `pin_multi` trainee (`--trainee-teams`, the multi-team exploiter) carries a LIST of team
        exports, not one string. This used to call `.encode()` on the list and crash the eval
        callback — which, because eval fires mid-rollout, took the whole run down in a restart loop
        (the multi-team def-20 arm, 2026-07-26). Emitting the per-team shas keeps the provenance
        field MEANINGFUL for that case rather than collapsing the set to one opaque digest: each
        entry joins the same `pin_sha` keyspace as `matchup_spec` and the archetype table.
        """
        if not t:
            return None
        if isinstance(t, (list, tuple)):
            return [hashlib.sha1(x.encode()).hexdigest()[:10] for x in t if x]
        return hashlib.sha1(t.encode()).hexdigest()[:10]
    d = os.path.join(model_dir, "eval_traces", f"step_{step}")
    os.makedirs(d, exist_ok=True)
    manifest = {
        "step": step,
        "num_timesteps": step,
        "git_hash": git_hash,
        "arch_signature": arch_signature,
        "config_version": config_version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "opponents": list(opponents),
        "n_games": n_games,
        "matchup_hash": _read_matchup_hash(model_dir),
        "trainee_team_sha": _sha(trainee_team_str),
        "opponent_pins": {k: _sha(v) for k, v in (opponent_pins or {}).items() if v},
        "selection_schema": SELECTION_SCHEMA,
        # 🚨 The rule is stated from the quota THIS RUN is configured with, never from the module
        # default: a manifest whose stated rule contradicts what the recorder did is worse than
        # no manifest, because a consumer reweights by it.
        "selection_rule": _rule_for(quota),
        # Filled by record_eval_selection at collect. NULL here on purpose — see the docstring.
        "selection": None,
    }
    with open(os.path.join(d, EVAL_MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# In-flight watchdog: if a cycle's workers don't all finish within this wall-clock
# budget, the cycle is presumed HUNG (e.g. a Showdown battle that never completes —
# a worker blocked on a websocket await), so the parent kills the workers, collects
# whatever results landed, and clears `_pending`. Without this a single hung worker
# pins `_pending` forever and every later eval boundary is silently skipped — i.e. eval
# never recovers for the rest of the run. Set generously above a healthy CPU cycle
# (the full roster — all bots + sentinels at EVAL_GAMES each — runs well under this) so it
# never trips a slow-but-live eval; only a true hang reaches it.
_EVAL_CYCLE_TIMEOUT_SEC = 1800.0


def eval_cycle_timeout() -> float:
    """The hung-cycle bound, scaled by measured CPU contention.

    Eval is the path MOST exposed to contention in the whole system: it deliberately runs
    concurrently with training (that is the point of the subprocess design), so it is under load
    100% of the time — and the docs already note "on CPU an eval can outlast its interval".

    The cost of firing early is not a lost cycle, it is BIASED NUMBERS: `_abort_pending_cycle`
    kills the workers and collects PARTIAL results, which flow into `win_rate_vs_bots` (the
    curriculum ramp), `win_rate_vs_pool` (the promotion gate) and the ELO fit. A truncated sample
    is not a random subsample either — it is whichever shards happened to get scheduled. So a
    merely-slow cycle must never be mistaken for a hung one.

    Read at CALL time (both callbacks go through this), so the bound tracks load as it develops.
    """
    return scale_timeout(_EVAL_CYCLE_TIMEOUT_SEC)


# ── Shared subprocess-eval mechanics (used by BOTH eval callbacks) ─────────────
# These keep the bot-eval and self-play-eval cycles spawning / merging / grooming
# identically, so the two non-blocking paths can't drift.

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_NONCE_COUNTER = 0


def _b36(n: int, width: int) -> str:
    """Fixed-width base-36 encoding of `abs(n)` (low digits; wraps at 36**width)."""
    n = abs(int(n))
    out = []
    for _ in range(width):
        out.append(_B36[n % 36])
        n //= 36
    return "".join(reversed(out))


def eval_run_nonce() -> str:
    """A short (3 base-36 char) per-PROCESS nonce for eval account names.

    Eval account names are ``<prefix><cycle_tag><wid><claim_seq>``; ``cycle_tag`` used to
    be ``step // 100 % 10000``, which is NOT unique across launcher restarts — the resume
    re-eval always fires at ~the same step, so every restart reused the same account names
    and collided with the previous (killed) process's lingering Showdown challenges
    (``There's already a challenge between you and ...``) → the battle never starts and the
    worker hangs forever. Mixing the pid + wall-clock (+ a process-global counter so
    successive calls differ) makes the tag unique per process while staying ≤4 chars, so
    the full account name comfortably fits Showdown's 18-char username cap.
    """
    global _NONCE_COUNTER
    _NONCE_COUNTER += 1
    return _b36(os.getpid() * 1_000_003 + int(time.time()) + _NONCE_COUNTER * 7919, 3)


def launch_rust_eval_cycle(cb, pool, run_dir: str, step: int) -> None:
    """M5 Lane H: play this cycle's plan on the Rust eval core (``rust_eval.launch``), in process and
    blocking, publishing the same shard results a Python worker would. A cycle failure is logged and
    leaves the shard files absent (the collect then reads it as missing, like a crashed worker); a
    lifecycle violation or a missing eval core is RAISED."""
    from agents.training.rust_eval.executor import EvalCoreError
    from agents.training.rust_eval.launch import run_rust_eval_cycle

    t0 = time.monotonic()
    try:
        st = run_rust_eval_cycle(cb, pool=pool, run_dir=run_dir, step=step)
    except EvalCoreError as e:
        print(f"⚠️ [EVAL] step {step:,}: the Rust eval cycle failed — {e}")
        send_event(f"⚠️ Eval @ {step:,}: Rust eval cycle failed ({type(e).__name__})")
        return
    print(f"[EVAL] step {step:,}: Rust eval core played {st['games']:,} games in {time.monotonic() - t0:.1f}s "
          f"({st['trainee_decisions']:,} trainee decisions, {st['traces']} traces, {st['near_ties']} near-ties)")



def kill_eval_workers(procs: list[dict], wait_timeout: float = 5.0) -> None:
    """Kill any still-running eval workers and reap them (so none linger as zombies)."""
    for w in procs:
        if w["proc"].poll() is None:
            w["proc"].kill()
        try:
            w["proc"].wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            pass


def spawn_eval_workers(run_dir: str, base_cfg: dict, n_workers: int) -> list[dict]:
    """Write one config_<wid>.json per worker and Popen ``python -m main.eval_worker`` on it.

    ``base_cfg`` carries everything common to the workers (snapshot, port, opponent pool,
    claim/result dirs, concurrency, device, cycle_tag, and — for self-play — the sentinel
    specs); this only adds ``worker_id``. The launcher's metrics pipe FD is stripped from
    the child env (only the parent publishes to the TUI; that FD number is invalid in the
    child). Returns a list of ``{proc, log, log_path}``.
    """
    worker_env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
    procs = []
    for wid in range(n_workers):
        cfg = {**base_cfg, "worker_id": wid}
        cfg_path = os.path.join(run_dir, f"config_{wid}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        log_path = os.path.join(run_dir, f"worker_{wid}.log")
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "main.eval_worker", cfg_path],
            stdout=logf, stderr=subprocess.STDOUT, env=worker_env,
        )
        procs.append({"proc": proc, "log": logf, "log_path": log_path})
    return procs
