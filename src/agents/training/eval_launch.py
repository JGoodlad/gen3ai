"""The eval cycle's LAUNCH mechanics — split out of ``eval_callback.py`` (2026-10-01).

Shared by BOTH eval callbacks (``PerOpponentEvalCallback``, ``SelfPlayCallback``): the per-cycle
manifest (``write_eval_manifest`` — which model, which regime, which selection rule), the mirrored-pair
game count and the Rust eval core's in-process cycle (``launch_rust_eval_cycle``). ``spawn_eval_workers``
/ ``kill_eval_workers`` are NOT the callbacks' any more (they play in process); they stay for the
standalone callers that run ``main.eval_worker`` directly — the Python oracle of the Rust eval
(``rust_eval.parity`` / ``eval_benchmark``) and ``main.ops.eval_trace_gen``. ``eval_callback``
re-exports every public name here; a test that STUBS ``subprocess.Popen`` for the workers reaches it
through THIS module's ``subprocess``.
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
                        quota: "ForensicQuota | dict | None" = None,
                        mirrored_pairs: bool = False) -> dict:
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
        # THE PAIRING REGIME (`gen3_mirrored_pairs_v1`): were this cycle's games MIRRORED TEAM PAIRS
        # (each team pairing from both sides, one battle seed)? Always written — False is a fact too.
        "mirrored_pairs": bool(mirrored_pairs),
    }
    with open(os.path.join(d, EVAL_MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


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


# ── MIRRORED TEAM PAIRS (`gen3_mirrored_pairs_v1`, `--eval-mirrored-pairs`) — the launch side ─────────
def mirrored_eval_games(n_games: int) -> int:
    """The per-opponent game count under mirrored pairs: EVEN by construction (an odd request is rounded
    UP — a pair is two games, and half a pair is not a measurement)."""
    n = int(n_games)
    return n + (n % 2)
