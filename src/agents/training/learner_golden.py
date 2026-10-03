"""K9(a) THE LEARNER GOLDEN — one eager fp32 PPO update, pinned (M5 Lane K; ``program_rust_core.md`` K9).

WHAT IT PINS. Beside ``reward_golden`` (the number the trainer optimizes) and the obs golden (what the
network reads), this pins WHAT ONE UPDATE COMPUTES: a production-surface learner, built from a fixed
seed, runs ONE real ``InstrumentedMaskablePPO.train()`` — the production loss surface (the win-prob
critic, every belief / intent / win-prob term the production mirror turns on, via the SAME
``apply_training_hparams`` a launch runs) — on a committed small rollout buffer, and the golden is:

* ``post_params_sha256`` — sha256 of every parameter's float32 BYTES after the update, in
  ``named_parameters()`` order (names included). EXACT bytes, not rounded values: rounding to k digits
  still flips on whichever of the 3M values sits near a rounding boundary, so it buys no portability
  and loses teeth; exactness is what "any change to what the update computes" means. Per-module
  group hashes (``group_sha256``) ride beside it so a failure names WHERE the update moved.
* ``losses`` — every loss/objective scalar the update logs (``_is_pinned_key``), stored as exact
  float64 values and compared EXACTLY, so a failure names WHICH term moved.
* ``init_params_sha256`` — the same hash BEFORE the update. The "checkpoint" is not committed (3M
  params = 12 MB); it is REBUILT from a fixed seed at test time, so the init hash separates "the
  model construction / seeding changed" from "the update changed".

DETERMINISM. CPU, eager, fp32, ``torch.set_num_threads(1)`` for the BUILD and the update (a CPU
matmul's reduction order can depend on the thread count, and so does the init: SB3's orthogonal
re-init is a LAPACK QR, so ``build_learner`` pins one thread itself — F-X5-4, it used to rely on the
test conftest's ``OMP_NUM_THREADS=1``), numpy + torch seeded before ``train()`` (the minibatch shuffle
is ``np.random.permutation``). Exact bytes are only promised WITHIN one torch build, so the golden is
KEYED BY ``torch.__version__``: the interpreter the tests run under (``gen3ai_torch28``; HEAD runs
torch >= 2.8 only, so the 2.5.1 entry was dropped 2026-10-02 — its history rows stay) has its own
recorded entry. A missing key FAILS (never skips, never records).

RE-RECORDING IS EXPLICIT. The test never writes. A deliberate change to the update is recorded with

    python -m agents.training.learner_golden record --reason "why the update is meant to change"

which writes the reason, date, commit and torch version into ``learner_golden.json`` and appends an
append-only ``history`` row. Run it under EVERY interpreter that has an entry (the `record` output
says which). The pinned BUFFER changes only when the observation layout / label schema does:

    python -m agents.training.learner_golden rebuild-buffer --reason "..."   # needs the rust_env build

(a real complete-game rollout from the Rust collector at the production surface — real rows, real
labels, the behaviour log-probs of the seeded learner — sliced to ``N_STEPS`` x ``N_ENVS``).

SCOPE LIMITS (what it does NOT pin): the rollout / GAE (the buffer's advantages and returns are
inputs), the KL→LR controller and every other callback (outside ``train()``), CUDA / compiled
numerics (the compile parity gate and K6's canary own those).

THE RECIPE IS READ, NOT COPIED (`golden_recipe`): the loss knobs the production recipe carries come
from K10(a)'s block (`main.train.recipe_surface.production_recipe()` — ``recipe.fresh`` of
``designs/production_config.json``), the SB3 knobs no launch sets come from sb3's own constructor
defaults (what a launch therefore trains with), and only the golden's TEST-SPECIFIC overrides are
written here (`GOLDEN_OVERRIDES`: the scaled-down shape and the documented LR). A recipe change moves
the golden — `learner_golden_test` names the drifted knob before the hashes do.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

SCHEMA = "gen3_learner_golden_v1"
#: Ship BESIDE this module (a module locating its own data files, not repo-root discovery).
GOLDEN_PATH = Path(__file__).with_name("learner_golden.json")
BUFFER_PATH = Path(__file__).with_name("learner_golden_buffer.npz")

#: The buffer's shape: 16 steps x 4 envs = 64 real rows.
N_STEPS, N_ENVS = 16, 4
#: Seeds: the learner's construction + its seeded perturbation (a fresh pointer head is exactly
#: uniform — every ratio would be trivially 1), and the update's own RNG.
MODEL_SEED, PERTURB_SEED, UPDATE_SEED = 0, 1234, 123
#: The rollout the buffer is sliced from (``rebuild-buffer``).
RECORD_N_STEPS, RECORD_RUN_SEED, RECORD_P2_SEED = 32, 17, 5

#: The golden's TEST-SPECIFIC overrides — the only recipe values written here. The SHAPE is scaled
#: down but keeps the live shape's structure: micro 16 x K 3 over 64 rows = one full accumulation group
#: plus a RAGGED short group every epoch (the live 98,304 / 65,536 shape's half step), two epochs. The
#: LR is a DOCUMENTED TEST CONSTANT, deliberately neither `recipe.fresh` (3e-4) nor `recipe.fork`
#: (5.6e-5): the golden pins what the update COMPUTES; K10(a)'s doc gate pins the recipe.
GOLDEN_OVERRIDES: Dict[str, Any] = {"n_epochs": 2, "batch_size": 16, "grad_accum_steps": 3,
                                    "learning_rate": 2.8e-5}
#: Read from the PRODUCTION recipe block (`recipe.fresh`).
_FROM_RECIPE = ("clip_range", "ent_coef")
#: SB3 constructor knobs no launch sets (the trainer passes none of them), so a launch trains with
#: sb3's own defaults — read from its signature, never copied.
_FROM_SB3_DEFAULTS = ("max_grad_norm", "normalize_advantage", "target_kl")


def golden_recipe() -> Dict[str, Any]:
    """The golden update's SB3 recipe: production values (`_FROM_RECIPE`), sb3 defaults
    (`_FROM_SB3_DEFAULTS`), then `GOLDEN_OVERRIDES`."""
    import inspect

    from sb3_contrib import MaskablePPO

    from main.train.recipe_surface import production_recipe
    prod = production_recipe()
    sig = inspect.signature(MaskablePPO.__init__).parameters
    out: Dict[str, Any] = {k: prod[k] for k in _FROM_RECIPE}
    out.update({k: sig[k].default for k in _FROM_SB3_DEFAULTS})
    out.update(GOLDEN_OVERRIDES)
    return out


#: Logged keys never pinned: wall clocks.
_CLOCK_SUFFIXES = ("_ms", "_s")


class LearnerGoldenError(RuntimeError):
    """The pinned buffer is missing or stale, or no golden is recorded for this torch build."""


def _is_pinned_key(k: str) -> bool:
    """A logged scalar the golden compares: every LOSS / objective value and the PPO update's own
    statistics (KL, clip fraction, explained variance). Diagnostics-only families (grad-balance
    shares, ranks, noise scale, calibration bins) are left out, so a telemetry-only change does not
    move the golden — the PARAMETER hash is what catches any change that reaches the weights."""
    if k.startswith("time/") or k.endswith(_CLOCK_SUFFIXES):
        return False
    if k in ("train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/n_updates"):
        return True
    return "loss" in k.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------------------- building
def _spaces() -> Tuple[Any, Any, Any]:
    from agents.training.rust_rollout import testkit as TK

    return TK.production_spaces()


def build_learner(env: Any = None, args: Any = None) -> Any:
    """The seeded production-surface learner (on a dummy env over the production spaces), the
    production training hparams applied exactly as a launch applies them, then `golden_recipe()`.

    ``args`` (a resolved training namespace, e.g. production + one lever) builds THAT surface
    instead — its spaces (`trainee_spaces`), its policy kwargs and its training hparams — with the
    same seeds and recipe overrides (the lever tests' learner; the golden itself passes None).

    🚨 Built at ONE torch thread, whatever the caller's count (F-X5-4): SB3's `_build` re-initialises
    every Linear with `orthogonal_`, a LAPACK QR whose blocked reduction order follows the BLAS thread
    count — the same RNG draws give different init BYTES at 8 threads (15 of 41 parameter groups,
    measured 2026-10-03). The caller's thread count is restored on return."""
    with _one_thread():
        return _build_learner(env, args)


def _build_learner(env: Any, args: Any) -> Any:
    import torch as th

    from agents.training.rust_rollout import testkit as TK
    from agents.training.train_logger import configure
    from agents.training.rust_vec_env import RustVecEnv
    from main.train.production_args import production_args
    from main.train.model_build import apply_training_hparams

    if args is None:
        args, obs, act = _spaces()
        policy_args = None
    else:
        from agents.training.rust_rollout.build import trainee_spaces
        obs, act = trainee_spaces(args)
        policy_args = args
    RECIPE = golden_recipe()
    if env is None:
        env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act, build=lambda m: None)
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=RECIPE["batch_size"], n_epochs=RECIPE["n_epochs"],
                           seed=MODEL_SEED, perturb_seed=PERTURB_SEED, gamma=1.0, gae_lambda=0.8,
                           learning_rate=RECIPE["learning_rate"], ent_coef=RECIPE["ent_coef"],
                           clip_range=RECIPE["clip_range"], max_grad_norm=RECIPE["max_grad_norm"],
                           normalize_advantage=RECIPE["normalize_advantage"], target_kl=RECIPE["target_kl"],
                           policy_args=policy_args)
    apply_training_hparams(model, production_args() if policy_args is None else policy_args,
                           mappings=None)
    TK.unset_to_class_defaults(model)
    model.grad_accum_steps = int(RECIPE["grad_accum_steps"])
    # K9(b) is its own gate (`learner_gates_test`); the golden pins the update's arithmetic.
    model.behaviour_check = "off"
    model._logger = configure(None, [])
    del args
    th.manual_seed(UPDATE_SEED)
    return model


_BUFFER_FIELDS = ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns",
                  "action_masks")


def load_buffer_into(model: Any, path: Path = BUFFER_PATH) -> Dict[str, Any]:
    """Fill ``model.rollout_buffer`` from the committed buffer (a stale / missing file REFUSES)."""
    if not path.exists():
        raise LearnerGoldenError(f"the learner golden buffer {path} is missing — rebuild it with "
                                 "`python -m agents.training.learner_golden rebuild-buffer --reason ...`")
    rb = model.rollout_buffer
    rb.reset()           # a previous train() flattened it; reset re-allocates the [n_steps, n_envs] layout
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    want = {k: tuple(v.shape) for k, v in rb.observations.items()}
    have = {k[4:]: tuple(data[k].shape) for k in data if k.startswith("obs:")}
    if want != have:
        raise LearnerGoldenError(
            "the learner golden buffer is STALE: its observation keys/shapes differ from the production "
            f"surface's (missing {sorted(set(want) - set(have))}, extra {sorted(set(have) - set(want))}, "
            f"reshaped {sorted(k for k in set(want) & set(have) if want[k] != have[k])}). Rebuild it "
            "(`rebuild-buffer --reason ...`) and re-record.")
    for k in want:
        rb.observations[k][...] = data["obs:" + k]
    for f in _BUFFER_FIELDS:
        getattr(rb, f)[...] = data[f]
    rb.full, rb.pos = True, N_STEPS
    model._current_progress_remaining = 1.0
    return data


# ---------------------------------------------------------------------------------------- hashing
def params_sha256(model: Any) -> str:
    h = hashlib.sha256()
    for name, p in model.policy.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def group_sha256(model: Any, depth: int = 2) -> Dict[str, str]:
    """One hash per parameter GROUP (the first ``depth`` name components) — the WHERE of a failure."""
    groups: Dict[str, Any] = {}
    for name, p in model.policy.named_parameters():
        g = ".".join(name.split(".")[:depth])
        h = groups.setdefault(g, hashlib.sha256())
        h.update(name.encode())
        h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return {g: h.hexdigest()[:16] for g, h in sorted(groups.items())}


def buffer_sha256(path: Path = BUFFER_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def _one_thread() -> Iterator[None]:
    import torch as th

    prev = th.get_num_threads()
    th.set_num_threads(1)
    try:
        yield
    finally:
        th.set_num_threads(prev)


def compute(model: Optional[Any] = None,
            before_train: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """Build (unless given), load the pinned buffer, run ONE ``train()``; return the fingerprint.

    ``before_train`` runs after the update's pinned seeding and immediately before ``train()`` — where
    a caller FREEZES the learner (K6): the seeding is the harness's startup, and a global seed after the
    freeze is a `GlobalReseedError` (`global_rng_guard`, gen3_no_global_reseed_v1)."""
    import torch as th

    with _one_thread():
        if model is None:
            model = build_learner()
        load_buffer_into(model)
        init = params_sha256(model)
        np.random.seed(UPDATE_SEED)
        th.manual_seed(UPDATE_SEED)
        if before_train is not None:
            before_train()
        model.train()
        logged = dict(model.logger.name_to_value)
    losses = {k: float(v) for k, v in sorted(logged.items())
              if _is_pinned_key(k) and isinstance(v, (int, float, np.floating, np.integer))}
    return {"init_params_sha256": init, "post_params_sha256": params_sha256(model),
            "group_sha256": group_sha256(model), "losses": losses}


# ---------------------------------------------------------------------------------------- compare
def torch_key() -> str:
    import torch as th

    return str(th.__version__)


def load_golden(path: Path = GOLDEN_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise LearnerGoldenError(f"{path} is missing — record it with "
                                 "`python -m agents.training.learner_golden record --reason ...`")
    return json.loads(path.read_text())


def diff(recorded: Dict[str, Any], now: Dict[str, Any]) -> List[str]:
    """Every way ``now`` differs from ``recorded`` (empty = identical), most informative first."""
    out: List[str] = []
    if recorded["init_params_sha256"] != now["init_params_sha256"]:
        out.append("the INIT moved (model construction / seeding / perturbation changed — the update "
                   "itself may or may not have): init_params_sha256 "
                   f"{recorded['init_params_sha256'][:16]} -> {now['init_params_sha256'][:16]}")
    rl, nl = recorded["losses"], now["losses"]
    for k in sorted(set(rl) | set(nl)):
        if k not in nl:
            out.append(f"loss key GONE: {k} (was {rl[k]!r})")
        elif k not in rl:
            out.append(f"loss key NEW: {k} = {nl[k]!r}")
        elif not (rl[k] == nl[k] or (rl[k] != rl[k] and nl[k] != nl[k])):
            out.append(f"loss {k}: {rl[k]!r} -> {nl[k]!r} (delta {nl[k] - rl[k]:.3g})")
    rg, ng = recorded.get("group_sha256", {}), now["group_sha256"]
    moved = sorted(g for g in set(rg) | set(ng) if rg.get(g) != ng.get(g))
    if moved:
        out.append(f"parameter groups whose post-update bytes moved: {moved}")
    if recorded["post_params_sha256"] != now["post_params_sha256"]:
        out.append("post_params_sha256 "
                   f"{recorded['post_params_sha256'][:16]} -> {now['post_params_sha256'][:16]}")
    return out


def check() -> Tuple[List[str], Dict[str, Any]]:
    """``(differences, fingerprint)`` against the committed golden for THIS torch build."""
    g = load_golden()
    if g.get("schema") != SCHEMA:
        raise LearnerGoldenError(f"{GOLDEN_PATH}: schema {g.get('schema')!r}, expected {SCHEMA!r}")
    if g["buffer_sha256"] != buffer_sha256():
        raise LearnerGoldenError(
            f"{BUFFER_PATH.name} does not match the sha256 the golden was recorded on — the pinned INPUT "
            "changed. Rebuild + re-record deliberately (`rebuild-buffer`, then `record`, each with --reason).")
    key = torch_key()
    entry = g["entries"].get(key)
    if entry is None:
        raise LearnerGoldenError(
            f"no learner golden recorded for torch {key} (recorded: {sorted(g['entries'])}). Exact bytes are "
            "promised within one torch build only; record this one deliberately with "
            "`python -m agents.training.learner_golden record --reason ...` under this interpreter.")
    now = compute()
    return diff(entry, now), now


# ---------------------------------------------------------------------------------------- recording
def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return str(get_git_hash())
    except Exception:  # noqa: BLE001 — provenance only
        return "unknown"


def record(reason: str) -> Dict[str, Any]:
    """Write THIS torch build's entry (module docs). Keeps the other builds' entries; appends history."""
    if not reason or not reason.strip():
        raise SystemExit("record needs --reason: a re-record is a statement that the update is MEANT to change")
    g = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {
        "schema": SCHEMA, "entries": {}, "history": []}
    g["schema"] = SCHEMA
    bsha = buffer_sha256()
    if g.get("buffer_sha256") not in (None, bsha):
        # a new buffer invalidates every build's entry: they were computed on different input
        g["entries"] = {}
    g["buffer_sha256"] = bsha
    g["buffer"] = {"file": BUFFER_PATH.name, "n_steps": N_STEPS, "n_envs": N_ENVS}
    g["recipe"] = golden_recipe()
    g["seeds"] = {"model": MODEL_SEED, "perturb": PERTURB_SEED, "update": UPDATE_SEED}
    fp = compute()
    import torch as th

    meta = {"recorded_at_commit": _commit(), "recorded_on": _dt.date.today().isoformat(),
            "reason": reason.strip(), "torch": str(th.__version__), "numpy": np.__version__,
            "python": sys.version.split()[0]}
    prev = g["entries"].get(torch_key())
    g["entries"][torch_key()] = {**fp, **meta}
    g.setdefault("history", []).append({**meta, "post_params_sha256": fp["post_params_sha256"],
                                        "changed": diff(prev, fp) if prev else ["(first entry)"]})
    GOLDEN_PATH.write_text(json.dumps(g, indent=1, sort_keys=False) + "\n")
    return fp


def rebuild_buffer(reason: str) -> None:
    """Record a real complete-game rollout on the Rust collector and pin its first ``N_STEPS`` steps."""
    if not reason or not reason.strip():
        raise SystemExit("rebuild-buffer needs --reason")
    import torch as th

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    TK.build_selfcheck()
    _a, obs, act = _spaces()
    decl = RustEnvDecl(n_envs=N_ENVS, threads=2, front="ffi", profile="selfcheck", trigger="complete_game",
                       n_steps=RECORD_N_STEPS, micro_batch=RECORD_N_STEPS, device="cpu", backend="eager",
                       run_seed=RECORD_RUN_SEED, gamma=1.0, gae_lambda=0.8)
    p2 = TK.RandomP2(RECORD_P2_SEED)
    env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    prev_threads = th.get_num_threads()
    try:
        # The SAME seeded learner the golden rebuilds, so the stored log-probs are its behaviour policy —
        # built at one thread as `build_learner` is (F-X5-4: the init bytes follow the thread count).
        with _one_thread():
            model = TK.fresh_model(env, n_steps=RECORD_N_STEPS, batch_size=RECORD_N_STEPS, seed=MODEL_SEED,
                                   perturb_seed=PERTURB_SEED)
        col = env.startup(model)
        if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
            raise LearnerGoldenError("the collector stopped")
        rb = model.rollout_buffer
        out: Dict[str, np.ndarray] = {f"obs:{k}": np.ascontiguousarray(np.asarray(v)[:N_STEPS])
                                      for k, v in rb.observations.items()}
        for f in _BUFFER_FIELDS:
            out[f] = np.ascontiguousarray(np.asarray(getattr(rb, f))[:N_STEPS])
    finally:
        env.close()
        th.set_num_threads(prev_threads)
    np.savez_compressed(BUFFER_PATH, **out)
    print(f"wrote {BUFFER_PATH} ({BUFFER_PATH.stat().st_size} bytes, {N_STEPS}x{N_ENVS} rows); reason: "
          f"{reason.strip()}\nNow re-record the golden under every torch build: record --reason ...")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "rebuild-buffer"):
        sp = sub.add_parser(name)
        sp.add_argument("--reason", required=True, help="why the pinned update / buffer is MEANT to change")
    sub.add_parser("check", help="compare against the committed golden (what the test does), print the diff")
    a = ap.parse_args(argv)
    if a.cmd == "rebuild-buffer":
        rebuild_buffer(a.reason)
        return 0
    if a.cmd == "record":
        fp = record(a.reason)
        g = load_golden()
        print(f"recorded torch {torch_key()}: post {fp['post_params_sha256'][:16]}, {len(fp['losses'])} losses. "
              f"Entries now: {sorted(g['entries'])} — re-record under each of them if the update changed.")
        return 0
    d, _now = check()
    print("IDENTICAL" if not d else "DIFFERS:\n  " + "\n  ".join(d))
    return 0 if not d else 1


if __name__ == "__main__":
    raise SystemExit(main())
