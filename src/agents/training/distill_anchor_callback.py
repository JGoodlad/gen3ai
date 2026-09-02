"""WHERE THE OFF-SLICE ANCHOR'S PARENT COMES FROM (`gen3_distill_offslice_anchor_v1`).

The anchor (`instrumented_ppo/distill_anchor.py`) regularises the student toward the policy it
STARTED the fold as. That makes "which checkpoint is the parent?" the one question this feature can
get silently, catastrophically wrong — so it is answered here, once, and pinned by tests.

🚨 **THE RESTART RULE.** A launcher run is restarted every few hours, and an idempotent FORK swaps
its `--model` to the fork's OWN latest checkpoint on every relaunch
(`main.launcher.checkpoint.resolve_fork_resume_model`) so progress is not discarded. Anchoring to
"the checkpoint we just loaded" would therefore re-anchor to a DRIFTED policy at every restart: the
trust region would follow the student around, ratchet by ratchet, and after a handful of restarts it
would constrain nothing while still reading as ON. So the parent is always **re-loaded from the
ORIGINAL FORK-PARENT PATH**, resolved in this order:

  1. ``--distill-anchor-parent PATH`` — an explicit pin, and the only route that needs no inference.
  2. ``<run_dir>/metadata.json`` → ``original_command`` → its ``--model`` value. `original_command`
     is IMMUTABLE by construction (`agents.model.snapshot.save_model_snapshot`: the existing value
     always wins), so it still names the fork parent after any number of restarts. This is the route
     that carries a live run.
  3. this process's own ``--model`` — correct on a fork's FIRST launch, where the run dir is fresh
     and holds no `metadata.json` yet (`resolve_launch_run_dir` refuses to fork onto an existing
     run, so route 2 cannot fire early with the wrong answer).

The resolved path and the route it came from are PRINTED at startup, because a wrong parent is a
silent no-op rather than a crash, and the only defence against a silent no-op is a loud statement of
what was chosen.

The frozen parent is never persisted (`_excluded_save_params`): re-loading from a path is the whole
correctness argument, and a pickled copy would be a second, wrong answer that survives restarts.

🚨 **THE MOVING REFERENCES ARE RUN STATE, AND THEY *MUST* BE PERSISTED — the mirror of the rule
above.** `--distill-anchor-ref ema` and `periodic` do not have a path to re-read: the reference is a
function of this run's own trajectory. Re-initialising it from the parent on every launcher restart
would reset the trust region to fold start every few hours, which is a different failure from the one
above and just as silent. So the reference is written as a **state_dict SIBLING of the checkpoint** —
`<checkpoint>_anchor_ref.pt` beside `<checkpoint>.zip`, at every site that records a resumable
checkpoint (the periodic callback, the SIGUSR1 forced save, the SIGTERM abort save, the final save) —
and restored from the sibling of this launch's `--model`.

**THE MOVING REFERENCE IS A SECOND, INDEPENDENT LOAD OF THE PARENT — never a copy of the live
student.** `load_parent(path)` is called TWICE: once for the frozen parent that `collateral_kl_vs_parent`
reads, once for the reference that then starts moving. That is not thrift avoidance — a
`copy.deepcopy(model.policy)` is actively unsafe here on two counts. The extractor carries a
per-forward `ExtractorStashes` full of NON-LEAF tensors, which `deepcopy` refuses outright; and
`--compile-trainer` patches the BOUND `fe.forward` as an INSTANCE attribute, which `deepcopy` treats
as ATOMIC — so the copy's `forward` would still be closed over the LIVE extractor and every "frozen
reference" logit would silently be the student's own, reading a KL of exactly 0 forever while every
meter looked healthy. A second load has neither problem, is arch-identical by construction, and
starts at exactly the weights all three modes are supposed to start at.

**Beside the checkpoint rather than at the run root, for one reason: consistency.** A restart rewinds
the policy to a checkpoint; a run-level file would be whatever the reference was when the process
died, i.e. AHEAD of the weights it is supposed to be a trust region for. The sibling is the reference
as of exactly those weights.

**A restore is REFUSED unless the blob's `run_dir` and resolved `parent_path` match this launch's**,
because a FORK off a fold's `final_model.zip` would otherwise silently inherit the fold's average as
its own starting reference. A refused or missing restore falls back to initialising from the frozen
parent and SAYS SO on stdout + the launcher event stream — never silently, for the same reason the
resolved parent path is printed.
"""
import json
import os
import shlex
from typing import Callable, Optional, Tuple

import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from agents.training.instrumented_ppo.distill_anchor import ANCHOR_REFS
from agents.training.lineage import fork_parent as _lineage_fork_parent

#: The resolution routes, in precedence order — the value printed at startup and asserted in tests.
ANCHOR_PARENT_ROUTES = ("explicit", "lineage", "original_command", "cli_model")

_MODEL_FLAGS = ("--model", "--model_path", "--model-path")


def parse_model_arg(command: str) -> Optional[str]:
    """The ``--model`` value inside a recorded shell command, or ``None``.

    Handles both spellings argparse accepts (``--model X`` and ``--model=X``) and the underscore
    alias, via `shlex` so a quoted path with spaces survives. Total: a malformed command yields
    ``None`` rather than raising, because a resolution FALL-THROUGH is recoverable and an exception
    inside callback assembly is not.
    """
    try:
        toks = shlex.split(command or "")
    except ValueError:
        return None
    for i, tok in enumerate(toks):
        for flag in _MODEL_FLAGS:
            if tok == flag and i + 1 < len(toks):
                return toks[i + 1]
            if tok.startswith(flag + "="):
                return tok[len(flag) + 1:]
    return None


def read_original_command(run_dir: str) -> Optional[str]:
    """``<run_dir>/metadata.json``'s immutable ``original_command``, or ``None`` if absent/unreadable."""
    path = os.path.join(run_dir or "", "metadata.json")
    try:
        with open(path, encoding="utf-8") as f:
            val = json.load(f).get("original_command")
    except (OSError, ValueError):
        return None
    return val if isinstance(val, str) and val.strip() else None


def resolve_anchor_parent(*, explicit: Optional[str], run_dir: Optional[str],
                          cli_model: Optional[str]) -> Tuple[Optional[str], str]:
    """``(parent_path, route)`` per the precedence documented at the top of this module.

    Pure but for one `metadata.json` read. Returns ``(None, "unresolved")`` when nothing names a
    parent — the caller turns that into a refusal, never into "anchor off": an anchor that silently
    did not attach is exactly the failure the loud startup line exists to prevent.
    """
    if explicit:
        return explicit, "explicit"
    # gen3_run_lineage_v1: the parent comes from metadata.json's first-class `lineage` block; a
    # legacy run (pre-lineage) derives it from `original_command` through the ONE parser in
    # `agents.training.lineage`, which warns. `.path` is the path AS GIVEN, byte-identical to what
    # the old inline parse returned, so the loader's behaviour is unchanged.
    got = _lineage_fork_parent(run_dir, warn=True) if run_dir else None
    if got is not None:
        return got.path, ("original_command" if got.derived else "lineage")
    if cli_model:
        return cli_model, "cli_model"
    return None, "unresolved"


#: The sibling a resumable checkpoint carries when the anchor's reference is a MOVING one.
#: `<run>/checkpoints/checkpoint_900_steps.zip` -> `..._steps_anchor_ref.pt`.
ANCHOR_REF_SUFFIX = "_anchor_ref.pt"

#: Bumped only on a BREAKING change to the blob. A blob whose schema this build does not know is
#: REFUSED (and re-initialised from the parent, loudly) rather than half-read — the same refusal
#: convention `cf_label_buffer`'s schema gate uses, and for the same reason.
ANCHOR_REF_SCHEMA = 1


def anchor_ref_path(checkpoint_path: str) -> str:
    """The reference sibling for a checkpoint zip. Total; never touches the filesystem."""
    base = checkpoint_path[:-4] if checkpoint_path.endswith(".zip") else checkpoint_path
    return base + ANCHOR_REF_SUFFIX


def ema_window(tau: float) -> float:
    """The Polyak average's nominal window in TRAIN() CALLS: ``1 / (1 - tau)``.

    This is the number to quote when sizing `--distill-anchor-ema-tau`, and it has to be read in
    ENV STEPS to mean anything: at the production shape (`--n-envs 48 --n-steps 2048`) one rollout —
    and therefore one `train()` call, and therefore one EMA update — is **98,304 env steps**, so the
    default tau 0.99 is a window of 100 calls ~ **9.8M env steps**, i.e. most of a generation. tau
    0.9 is 10 calls ~ 983k steps; tau 0.999 is 1000 calls ~ 98M, longer than any run here and
    effectively `parent`.
    """
    return float("inf") if tau >= 1.0 else 1.0 / (1.0 - tau)


def assert_reference_matches_student(ref_policy, student_policy) -> None:
    """RAISE unless the two policies share ONE `state_dict` — the same keys at the same shapes.

    A Polyak average is only defined between objects that agree tensor for tensor, and on a genuine
    fold they do: the student was LOADED FROM this parent. What the check buys is that a mismatch
    fails at startup with the diagnosis, rather than as a `polyak_update_` that quietly skips the
    keys it cannot find and reports a trust region against a partly-frozen reference.
    """
    r, s = ref_policy.state_dict(), student_policy.state_dict()
    missing = sorted(set(s) - set(r))[:6]
    extra = sorted(set(r) - set(s))[:6]
    shape = [k for k in (set(r) & set(s)) if tuple(r[k].shape) != tuple(s[k].shape)][:6]
    if missing or extra or shape:
        raise ValueError(
            "the anchor's MOVING reference does not match the student: "
            f"{len(set(s) - set(r))} key(s) only the student has {missing}, "
            f"{len(set(r) - set(s))} only the reference has {extra}, "
            f"{len(shape)} shape mismatch(es) {shape}. A moving reference is a Polyak average of "
            "the student's own parameters, so the two must be the same architecture — which they "
            "are on any genuine fold. Use --distill-anchor-ref parent, or fix the parent.")


def polyak_update_(ref_policy, student_policy, tau: float) -> None:
    """``ref <- tau*ref + (1-tau)*student``, in place, KEY-MATCHED.

    Matched on `state_dict` KEYS rather than zipped `parameters()` order, because the two policies
    are separate objects and a key-matched update fails loudly on a mismatch where a zip would
    silently average the wrong tensors together.

    NON-float entries (an integer counter such as `num_batches_tracked`) are COPIED, not averaged —
    an average of two integer counters is not a counter. Float BUFFERS *are* averaged along with the
    parameters: they are part of the function the reference computes, and treating them differently
    would make the reference a policy no snapshot of this run ever was.
    """
    ref_sd, stu_sd = ref_policy.state_dict(), student_policy.state_dict()
    with th.no_grad():
        for k, rv in ref_sd.items():
            sv = stu_sd.get(k)
            if sv is None:
                continue
            if rv.dtype.is_floating_point:
                rv.mul_(tau).add_(sv.detach().to(device=rv.device, dtype=rv.dtype), alpha=1.0 - tau)
            else:
                rv.copy_(sv)


def save_anchor_ref_beside(model, checkpoint_path: str) -> Optional[str]:
    """Write the anchor's MOVING reference beside ``checkpoint_path``; return the path, or ``None``.

    ``None`` covers every case in which there is nothing to persist — no anchor, the default fixed
    `parent` reference (which is re-resolved from a path and must never be pickled), or a write that
    failed. TOTAL by design: this is called from the checkpoint sites, including a signal handler,
    and a diagnostic's persistence must never be what loses a checkpoint.
    """
    writer = getattr(model, "_distill_anchor_ref_writer", None)
    if writer is None:
        return None
    try:
        return writer.save_reference(checkpoint_path)
    except Exception as e:  # noqa: BLE001
        print(f"[DistillAnchor] WARNING: could not write the reference sibling for "
              f"{checkpoint_path}: {e}")
        return None


class DistillAnchorCallback(BaseCallback):
    """Attach the anchor's REFERENCE (and the anchor hparams) before `learn()` takes its first step,
    and — under a moving reference — advance it once per rollout and persist it beside every
    checkpoint.

    A callback rather than a line in `apply_training_hparams` for one reason that is also the
    feature's correctness argument: `_on_training_start` runs on EVERY launch, which is exactly the
    cadence at which the parent must be re-read from its own path (and the moving reference restored
    from its sibling). `load_parent(path)` is injected so this module needs no `mappings` and no
    `main.train` import, and so the resolution logic can be tested without a checkpoint on disk.

    A failed load is FATAL (`TrainExitCode.FATAL_CONFIG`), matching the `--distill-teacher` load: a
    run that silently trains with no trust region while its command says otherwise produces a
    result that is worse than no result, because it looks like a measurement.

    🚨 **THE UPDATE CADENCE IS ONE PER `train()` CALL, and `_on_rollout_end` is what provides it.**
    SB3's `learn()` is `collect_rollouts -> callback.on_rollout_end() -> train()`, one train per
    rollout, and there is NO hook after `train()` at all — so this is the only per-`train()` cadence
    a callback can have, and taking it inside `ppo.py` instead would have bought a
    per-optimizer-step reference at `n_epochs x n_minibatches` times the cost, for a quantity nobody
    reads at that resolution. The PHASE is therefore "the reference used inside `train()` k is the
    average of the policies that produced the data", which is the right side of that boundary: the
    anchor is a trust region on the policy being updated, not on the update itself.
    """

    def __init__(self, *, parent_path: str, route: str, coef: float, mode: str, monitor: bool,
                 load_parent: Callable[[str], object], ref: str = "parent", ema_tau: float = 0.99,
                 refresh_every: int = 8, run_dir: Optional[str] = None,
                 resume_model: Optional[str] = None, expect_restore: bool = False,
                 proj_samples: int = 16, verbose: int = 0):
        super().__init__(verbose)
        self.parent_path = parent_path
        self.route = route
        self.coef = float(coef or 0.0)
        self.mode = str(mode)
        self.monitor = bool(monitor)
        # gen3_distill_grad_project_v1: `m` for `--distill-anchor-mode grad_project`. It rides THIS
        # callback because the mode does, and because grad_project is the one mode that needs the
        # frozen parent attached at coefficient 0 — the projection's whole readout is
        # `distill/collateral_kl_vs_parent`, which only exists when this callback ran.
        self.proj_samples = max(1, int(proj_samples or 16))
        self._load_parent = load_parent
        self.ema_tau = float(ema_tau)
        self.refresh_every = int(refresh_every)
        # `periodic` at a 0 cadence IS `parent` — never refreshed. Collapsed ONCE, here, so no
        # reader downstream carries the special case (and so the flag's documented "0 = never =
        # parent" is a fact about the object rather than a promise in a help string).
        self.ref = str(ref)
        if self.ref == "periodic" and self.refresh_every <= 0:
            self.ref = "parent"
        if self.ref not in ANCHOR_REFS:
            raise ValueError(f"unknown --distill-anchor-ref {ref!r}; expected one of {ANCHOR_REFS}")
        self.run_dir = run_dir
        self.resume_model = resume_model
        self.expect_restore = bool(expect_restore)
        self._ref_model = None           # the moving reference MODEL (None under `parent`)
        self._ref_policy = None          # its policy — what moves, and what is persisted
        self._rollouts = 0               # rollouts since the reference was refreshed / initialised
        self._refreshes = 0              # `periodic` refresh count, for the record
        self._restore_note = ""          # what the startup line says about the restore

    # ---- the reference's life ----------------------------------------------------------------
    def _fatal(self, why: str) -> None:
        """Exit `FATAL_CONFIG`, never a crash. The launcher GIVES UP on this code rather than
        restart-looping, which is the right response to a config that cannot work — a run that
        silently trained with no trust region while its command said otherwise would produce a
        result worse than no result, because it would look like a measurement."""
        from main.exit_codes import TrainExitCode
        import sys
        print(f"\n[DistillAnchor] FATAL: {why}")
        sys.stdout.flush()
        os._exit(int(TrainExitCode.FATAL_CONFIG))

    def _publish_age(self) -> None:
        """`distill/anchor_ref_age_rollouts` — WHAT the anchor is currently anchored to.

        Under `ema` the reference has no age (every past policy contributes, geometrically), so the
        honest number is the NOMINAL WINDOW; under `parent` and `periodic` it is rollouts since the
        reference was last set, which for `parent` rises for the life of the fold and is exactly
        the point.
        """
        self.model.distill_anchor_ref_age = (ema_window(self.ema_tau) if self.ref == "ema"
                                             else float(self._rollouts))

    def _restore_reference(self) -> bool:
        """Restore the moving reference from the sibling of this launch's `--model`.

        Returns True on success. Every refusal sets `_restore_note` and returns False, so the
        caller re-initialises from the parent and the startup line states which happened — a
        reference that silently reset to fold start on every launcher restart is the exact failure
        this persistence exists to prevent, and it would read as ON throughout.
        """
        if not self.resume_model:
            self._restore_note = "no --model to restore from (fold start)"
            return False
        path = anchor_ref_path(self.resume_model)
        if not os.path.exists(path):
            self._restore_note = (f"NO reference sibling at {path} — "
                                  + ("EXPECTED one (this is a RESTART of a run that should have "
                                     "written it); re-initialising from the PARENT, so the trust "
                                     "region has been RESET to fold start"
                                     if self.expect_restore else
                                     "initialising from the PARENT (a fork's first launch)"))
            return False
        try:
            blob = th.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001 — a corrupt sidecar must not take down a run
            self._restore_note = f"reference sibling {path} unreadable ({e}); re-init from PARENT"
            return False
        why = self._reject_blob(blob)
        if why:
            self._restore_note = f"reference sibling {path} REFUSED ({why}); re-init from PARENT"
            return False
        try:
            self._ref_policy.load_state_dict(blob["state_dict"], strict=True)
        except Exception as e:  # noqa: BLE001 — the reference is left AT THE PARENT, not half-loaded
            self._restore_note = (f"reference sibling {path} does not fit this policy ({e}); "
                                  f"re-init from PARENT")
            return False
        self._rollouts = int(blob.get("rollouts_since_refresh", 0) or 0)
        self._refreshes = int(blob.get("refreshes", 0) or 0)
        self._restore_note = (f"RESTORED from {path} "
                              f"(saved at {blob.get('num_timesteps')} steps, "
                              f"{self._rollouts} rollouts since its last refresh)")
        return True

    def _reject_blob(self, blob) -> Optional[str]:
        """``None`` if the blob may be used, else the one-line reason it may not.

        The `run_dir` + `parent_path` checks are the FORK guard: a fork launched off a fold's
        `final_model.zip` would otherwise pick up that fold's average as its own starting reference,
        which is a different run's trajectory wearing this run's name.
        """
        if not isinstance(blob, dict):
            return "not a dict"
        if blob.get("schema") != ANCHOR_REF_SCHEMA:
            return f"schema {blob.get('schema')!r} != {ANCHOR_REF_SCHEMA}"
        if blob.get("ref") != self.ref:
            return f"saved under ref {blob.get('ref')!r}, this launch is {self.ref!r}"
        if blob.get("parent_path") != self.parent_path:
            return (f"saved against parent {blob.get('parent_path')!r}, this launch anchors to "
                    f"{self.parent_path!r}")
        _saved_run = blob.get("run_dir")
        _this_run = os.path.abspath(self.run_dir) if self.run_dir else None
        if _saved_run != _this_run:
            return f"belongs to run {_saved_run!r}, this launch is {_this_run!r}"
        if not isinstance(blob.get("state_dict"), dict):
            return "no state_dict"
        return None

    def save_reference(self, checkpoint_path: str) -> Optional[str]:
        """Write the moving reference beside ``checkpoint_path``; ``None`` under a fixed `parent`."""
        if self._ref_policy is None:
            return None
        path = anchor_ref_path(checkpoint_path)
        tmp = path + ".tmp"
        th.save({
            "schema": ANCHOR_REF_SCHEMA,
            "ref": self.ref,
            "ema_tau": self.ema_tau,
            "refresh_every": self.refresh_every,
            "rollouts_since_refresh": int(self._rollouts),
            "refreshes": int(self._refreshes),
            "run_dir": os.path.abspath(self.run_dir) if self.run_dir else None,
            "parent_path": self.parent_path,
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in self._ref_policy.state_dict().items()},
        }, tmp)
        os.replace(tmp, path)          # crash-safe: a reader never sees a half-written blob
        return path

    # ---- SB3 hooks ---------------------------------------------------------------------------
    def _on_training_start(self) -> None:
        from main.launcher.ipc import emit
        # The hparams first: they are read by `train()` off the model, and a load failure below
        # exits the process, so there is no state in which they are set and the parent is not.
        self.model.distill_anchor_coef = self.coef
        self.model.distill_anchor_mode = self.mode
        self.model.distill_anchor_proj_samples = self.proj_samples
        self.model.distill_anchor_monitor = self.monitor
        self.model.distill_anchor_ref = self.ref
        try:
            parent = self._load_parent(self.parent_path)
        except Exception as e:  # noqa: BLE001 — a bad/incompatible parent must not crash-restart-loop
            self._fatal(f"could not load the fold parent {self.parent_path!r} "
                        f"(resolved via {self.route}): {e}")
        self.model._distill_anchor_parent = parent
        # THE REFERENCE. `parent` aliases the frozen parent itself, so `distill_anchor_step` takes
        # its `ref is parent` fast path and the default arm runs exactly one frozen forward, as it
        # always has. The moving forms are restored from the checkpoint sibling when this launch is
        # a restart, and INITIALISED FROM THE PARENT otherwise — so at fold start all three modes
        # hold the same reference and only diverge as the student moves.
        if self.ref == "parent":
            self.model._distill_anchor_ref = parent
            self._ref_model = None
            self._ref_policy = None
            self._rollouts = 0
            self._restore_note = "fixed at the fold parent"
        else:
            # A SECOND, INDEPENDENT load — see the module docstring for why this is not a deepcopy
            # of the student. It starts AT the parent, so all three modes coincide at fold start.
            try:
                ref_model = self._load_parent(self.parent_path)
                ref_model.policy.set_training_mode(False)
                for _p in ref_model.policy.parameters():
                    _p.requires_grad_(False)
                assert_reference_matches_student(ref_model.policy, self.model.policy)
            except Exception as e:  # noqa: BLE001 — same class of refusal as the parent load
                self._fatal(f"could not build the MOVING reference for --distill-anchor-ref "
                            f"{self.ref}: {e}")
            self._ref_model = ref_model
            self._ref_policy = ref_model.policy
            self._restore_reference()          # may overwrite those weights from the sibling
            self.model._distill_anchor_ref = ref_model
        # The checkpoint sites reach the writer through the model, so they need no callback handle.
        self.model._distill_anchor_ref_writer = self
        self._publish_age()
        _ref_desc = {"parent": "FIXED at the fold parent (LwF)",
                     "ema": f"POLYAK average of the student, tau={self.ema_tau:g} "
                            f"(~{ema_window(self.ema_tau):.0f} train() calls; ACER)",
                     "periodic": f"RE-SNAPSHOT from the student every {self.refresh_every} "
                                 f"rollouts"}[self.ref]
        emit(f"⚓ [DISTILL-ANCHOR] parent = {self.parent_path} (via {self.route}) | "
             f"coef={self.coef:g} mode={self.mode} ref={self.ref}"
             f"{' | MONITOR-ONLY (no loss term)' if self.coef == 0.0 else ''} — "
             f"reference: {_ref_desc}; {self._restore_note}. "
             f"The PARENT is re-loaded from the PARENT PATH on every restart, never from the "
             f"current checkpoint. Watch distill/collateral_kl_vs_parent (accumulated displacement, "
             f"emitted in EVERY mode) against distill/teacher_agreement_on_slice (content).")

    def _on_rollout_end(self) -> None:
        """Advance the reference — ONE update per rollout, i.e. one per `train()` call."""
        if self.ref == "ema":
            polyak_update_(self._ref_policy, self.model.policy, self.ema_tau)
            self._rollouts += 1
        elif self.ref == "periodic":
            self._rollouts += 1
            if self._rollouts >= self.refresh_every:
                self._ref_policy.load_state_dict(self.model.policy.state_dict(), strict=True)
                self._rollouts = 0
                self._refreshes += 1
        else:                                  # `parent`: nothing moves, but the age still rises
            self._rollouts += 1
        self._publish_age()

    def _on_step(self) -> bool:
        return True
