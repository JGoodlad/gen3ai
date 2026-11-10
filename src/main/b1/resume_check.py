"""BOUNDARY-1's stop-and-resume check: is "phase 1 to N, then RESUMED to 2N" the run "one launch to 2N" is?

    python -m main.b1.resume_check static  --tree <a git checkout holding the pin's commit> --pin <sha>
    python -m main.b1.resume_check compare --run-through <run A> --phased <run B> --stop <B as phase 1 left it> \\
                                           --logs <dir holding a.log, b1.log, b2_dryrun.log, b2.log, b3.log>
    python -m main.b1.resume_check smoke   --models-dir <an EMPTY scratch dir> [--arm wb] [--half 4000]

Each prints its evidence, then ``PASS`` or the list of differences, and exits non-zero on a difference, on a tree
that cannot be read, and on a run dir or log that is missing (an unreadable input is a difference, never a pass).
The pin is the PREREG's §3 and is always passed (``main.b1.arms`` carries none).

WHAT IS COMPARED, AND AGAINST WHAT. "One launch to 2N" is not one process: the launcher restarts the child every
six hours in the RESUME role (``main.launcher.checkpoint.resume_child_args``). So the question is whether the phase
boundary is anything a six-hour restart is not. Two parts:

``static`` (no run needed), for all four arms on the RESOLVED config:

* the namespace at the phase-1 target and at the phase-2 target differ in ``steps`` and nothing else;
* every knob through which the step TARGET or the fraction of it trained could reach training is at the value
  that switches it off (``TARGET_KNOBS``), and the progress fraction is read only through constant functions;
* the source inventory: which training modules read the step target or the progress fraction
  (``TARGET_READERS``, ``PROGRESS_READERS``). A module outside the list is a difference: it was not audited;
* the phase-2 command is a same-run RESTART for the launcher's one predicate, carries no fresh-only flag and
  holds the pin.

``compare`` (the two-segment smoke's three run dirs and logs):

* the sidecars (``model_config.json``; ``metadata.json``'s ``cli_args`` less the step target; the lineage role,
  which must stay ``fresh``, never ``fork``; one pin row; the era);
* ``num_timesteps`` continuity (the resume starts at the stop's step count and ends where the run-through ends);
* the optimizer state the resume LOADS, through the trainer's own loader and realignment, against what the stop
  SAVED in its zip (tensor-equal), and the weights with it;
* the schedule values each run recorded at the steps both have (learning-rate bounds, clip range, entropy
  coefficient, the team-mix share);
* the self-play pool across the stop (every snapshot and the ledger the stop left are still there after phase 2);
* the launcher's child argv on the phase-2 resume against its own six-hour-restart child argv, both resolved
  against the stopped run's record: the same namespace but for the step target;
* the launcher took the resume of a run that ENDED at its target into the same folder (exit 0, RESTART role), and
  a second launch of the same command on the finished run trains nothing.

WHAT THE PHASE BOUNDARY IS THAT A SIX-HOUR RESTART IS NOT (``BOUNDARY_NOTES``, printed by ``static``): nothing that
reaches the weights' update rule. The differences are in bookkeeping, each stated there.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from main.b1 import arms
from utils.era import prefixed

#: dest -> (the value that keeps the step target out of training, why it matters). Checked on every arm's
#: RESOLVED namespace at both phase targets.
TARGET_KNOBS: Dict[str, Tuple[Any, str]] = {
    "anneal_lr_start_steps": (None, "the two-phase LR anneal is the ONE schedule that reads --steps "
                                    "(main/train/callbacks.py: TwoPhaseLRCallback(total_steps=args.steps)); "
                                    "None = the KL-adaptive controller, which reads no target"),
    "adaptive_batch": ("off", "the adaptive accumulation controller restores a per-checkpoint value on a resume; "
                              "off = grad_accum_steps is the recorded constant"),
    "clip_range": (0.15, "read as clip_range(progress); the trainer installs a constant function of this value"),
    "clip_range_vf": (None, "None = no value clipping, so no second function of progress"),
    "ent_coef": (0.05, "a constant: no entropy schedule exists; a resume re-applies the namespace's value"),
    "n_epochs": (10, "re-applied from the namespace on a resume"),
    "overlap_collect": (False, "the loop collects the NEXT rollout during an update except on the last iteration "
                               "before the target (instrumented_ppo/loop.py); off = never, at any target"),
    "checkpoint_every_steps": (arms.CHECKPOINT_EVERY, "checkpoints fall on GLOBAL multiples of this "
                                                      "(main/train/constants.py: checkpoint_due), not on steps "
                                                      "since the process started"),
}

#: The non-test modules under ``src/main/train`` and ``src/agents`` that read the step target (``args.steps``, a
#: namespace's ``steps``, or the loop's ``total_timesteps``), with what each does with it. Audited at the pin; a
#: module outside the list is a difference. (The scan is by name: an alias of the target under another name would
#: escape it, which is why the two-segment smoke is the second half of the evidence.)
TARGET_READERS: Dict[str, str] = {
    "src/main/train/callbacks.py": "TwoPhaseLRCallback(total_steps=args.steps), built only in annealing mode "
                                   "(anneal_lr_start_steps set); off in every arm",
    "src/main/train/model_build.py": "remaining = args.steps - num_timesteps, passed to learn(); the "
                                     "'Continuing Training' / 'already complete' lines; the anneal's print",
    "src/main/train/combination_checks.py": "a launch check (the anneal's start must be below --steps); no anneal "
                                            "in any arm",
    "src/main/train/disk_preflight.py": "the launch's disk estimate (checkpoints and eval cycles up to --steps): a "
                                        "preflight, read before anything trains",
    "src/agents/training/instrumented_ppo/loop.py": "the loop's stop condition (num_timesteps < target), the "
                                                    "progress fraction (below), and 'no overlap collection on the "
                                                    "last iteration' (overlap_collect is off in every arm)",
    "src/agents/training/metrics_exporter_callback.py": "the metric NAME time/total_timesteps, whose value is "
                                                        "num_timesteps; not the target",
    "src/agents/training/capacity_overhead_benchmark.py": "a benchmark's own learn() call; not a training path",
    "src/agents/training/rust_env_opponents_benchmark.py": "a benchmark's own --steps; not a training path",
}
#: ... and the modules that read the fraction of the target trained (``_current_progress_remaining``).
PROGRESS_READERS: Dict[str, str] = {
    "src/agents/training/instrumented_ppo/loop.py": "sets it each iteration (1 - num_timesteps / target)",
    "src/agents/training/instrumented_ppo/ppo.py": "clip_range(progress), clip_range_vf(progress): constant "
                                                   "functions (model_build installs `lambda _: args.clip_range`)",
    "src/agents/training/instrumented_ppo/train_setup.py": "the same two calls, for the micro-step's statics",
    "src/agents/training/rust_rollout/consistency.py": "clip_range(progress), for the rollout consistency check",
    "src/agents/training/learner_golden.py": "sets it to 1.0 for the golden learner fixture (not a training path)",
}
_TARGET_RE = re.compile(r"\b(?:args|a|ns)\.steps\b|\b_?total_timesteps\b|getattr\(\s*\w+,\s*[\"']steps[\"']")
_PROGRESS_RE = re.compile(r"\b_current_progress_remaining\b")
#: Fixed strings one of which every match of the two patterns contains (the pin's objects are pre-filtered by them).
_NEEDLES = (".steps", "total_timesteps", '"steps"', "'steps'", "_current_progress_remaining")
_INVENTORY_ROOTS = ("src/main/train", "src/agents")
#: What resolves an argv to a config and trains it: must be the pin's in the checkout that runs this check.
_CODE_PATHS = ("src/main/train", "src/main/launcher", "src/agents", "src/main/checkargs.py",
               "designs/production_config.json")

#: What a phase boundary is that a six-hour restart is not. None reaches the update rule; each is bookkeeping.
BOUNDARY_NOTES: Tuple[str, ...] = (
    "a target stop TRAINS on its last rollout and runs the end-of-training hooks (the cycle ledger shard is "
    "closed, the metrics pipe shut) before it saves final_model.zip; a six-hour restart saves "
    "final_model_interrupted.zip at a rollout end WITHOUT training on that rollout "
    "(agents/training/graceful_restart_callback.py) and skips the hooks. The stop is the cleaner boundary",
    "the phase-2 launch starts a new six-hour clock, so the count of interval restarts over 100M steps can differ "
    "by one from a run-through's; a run-through's own restart points are wall-clock, so they are not "
    "reproducible either",
    "every resume, of either kind, rebuilds the environments (battles in flight are dropped), reseeds the learner "
    "from --seed (gen3_learner_segment_seed_v1) and the collector from (--seed, the step the process starts at) "
    "(rust_env_setup.segment_seed: teams, battle seeds and keyed draws of the new segment), and reloads the "
    "optimizer from the zip; the phased run does this once more than a run-through, at a step fixed by the target "
    "instead of by the clock. No two launches of one argv play the same games past their first restart either",
    "the phase-2 child argv is not token-equal to a restart's: the launcher passes every recorded recipe knob "
    "TYPED on a --model launch (recipe_surface.materialize_recorded_recipe) and a restart leaves them to the "
    "child's own resolver. `compare` shows both resolve to one namespace",
    "the RESUME branch's final save writes final_model.zip, model_config.json, metadata.json and latest.txt but "
    "NOT a per-checkpoint final_model.json: the one phase 1 wrote (fresh branch) stays, describing the phase-1 "
    "stop. Read a finished run's step count from metadata.json, never from final_model.json",
    "the step at which phase 1 stops is the first rollout end at or past the target, a step a run-through also "
    "passes; the checkpoint and the evaluation due at the target are taken inside phase 1, as in a run-through "
    "(both fire on GLOBAL multiples: main/train/constants.py checkpoint_due, selfplay_callback's eval cadence)",
)

class Report:
    """Evidence lines and differences, in the order found."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.differences: List[str] = []

    def ok(self, text: str) -> None:
        self.lines.append(f"  ok    {text}")

    def note(self, text: str) -> None:
        self.lines.append(f"  note  {text}")

    def diff(self, text: str) -> None:
        self.lines.append(f"  DIFF  {text}")
        self.differences.append(text)

    def head(self, text: str) -> None:
        self.lines.append(f"-- {text}")

    def expect(self, cond: bool, text: str, detail: str = "") -> bool:
        if cond:
            self.ok(text)
        else:
            self.diff(text + (f": {detail}" if detail else ""))
        return bool(cond)

    def render(self) -> str:
        verdict = ("PASS: no difference" if not self.differences
                   else f"FAIL: {len(self.differences)} difference(s)\n"
                        + "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(self.differences)))
        return "\n".join(self.lines + [verdict])

    @property
    def exit_code(self) -> int:
        return 1 if self.differences else 0


# ----------------------------------------------------------------------------------------------- static
def classify_readers(files: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    """``{"target": [...], "progress": [...]}``: of ``(repo-relative path, text)`` pairs, the non-test modules that
    read the step target / the progress fraction, sorted."""
    out: Dict[str, List[str]] = {"target": [], "progress": []}
    for rel, text in files:
        name = rel.rsplit("/", 1)[-1]
        if not name.endswith(".py") or name.endswith("_test.py") or name == "conftest.py":
            continue
        if _TARGET_RE.search(text):
            out["target"].append(rel)
        if _PROGRESS_RE.search(text):
            out["progress"].append(rel)
    return {k: sorted(v) for k, v in out.items()}


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)


def pin_files(repo: str, pin: str) -> List[Tuple[str, str]]:
    """``(path, text)`` of every file under ``_INVENTORY_ROOTS`` AT THE COMMIT ``pin`` that mentions either name
    (read from the commit's objects, so a dirty or moved checkout cannot change the answer)."""
    r = _git(repo, "grep", "-l", "-F", *[t for n in _NEEDLES for t in ("-e", n)], pin, "--", *_INVENTORY_ROOTS)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git grep at {pin} in {repo} failed: {r.stderr.strip()[:200]}")
    out = []
    for line in r.stdout.splitlines():
        rel = line.split(":", 1)[1]
        show = _git(repo, "show", f"{pin}:{rel}")
        if show.returncode != 0:
            raise RuntimeError(f"git show {pin}:{rel} failed: {show.stderr.strip()[:200]}")
        out.append((rel, show.stdout))
    return out


def check_inventory(inv: Dict[str, List[str]], rep: Report) -> None:
    for kind, audited in (("target", TARGET_READERS), ("progress", PROGRESS_READERS)):
        what = "the step target (args.steps)" if kind == "target" else "the progress fraction"
        extra = sorted(set(inv[kind]) - set(audited))
        gone = sorted(set(audited) - set(inv[kind]))
        rep.expect(not extra, f"every module reading {what} is an audited one ({len(inv[kind])} found)",
                   "NOT audited: " + ", ".join(extra))
        rep.expect(not gone, f"every audited reader of {what} is still one",
                   "audited, but no longer a reader (the audit is of another tree): " + ", ".join(gone))
        for rel in inv[kind]:
            if rel in audited:
                rep.note(f"{rel}: {audited[rel]}")


def check_code_is_the_pins(rep: Report, pin: str) -> None:
    """The config is resolved by the code this process IMPORTED: its training, launcher and model code and the
    production config must be the pin's, byte for byte (the kit itself is younger than the pin)."""
    from utils.paths import repo_root
    repo = str(repo_root())
    d = _git(repo, "diff", "--name-only", pin, "--", *_CODE_PATHS)
    u = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS)
    if d.returncode != 0 or u.returncode != 0:
        rep.diff(f"cannot compare {repo} with the pin: {(d.stderr or u.stderr).strip()[:200]}")
        return
    moved = sorted(set(d.stdout.split()) | {ln[3:] for ln in u.stdout.splitlines()})
    rep.expect(not moved, f"the code that resolved the config ({repo}: {', '.join(_CODE_PATHS)}) is the pin's",
               f"{len(moved)} file(s) differ from {pin[:12]}: {', '.join(moved[:6])}")


def check_resolved(rep: Report, pin: Optional[str] = None) -> None:
    """Every arm's resolved namespace: one difference between the phase targets, and the target knobs off."""
    n1, n2 = arms.PHASE_STEPS
    for arm in arms.ARMS:
        seed = arms.SEEDS[arm][0]
        a = arms.resolved_namespace(arms.build_argv(arm, seed, pin, steps=n1))
        b = arms.resolved_namespace(arms.build_argv(arm, seed, pin, steps=n2))
        d = arms.namespace_diff(a, b)
        rep.expect(d == ["steps"], f"arm {arm}: the resolved config at {n1:,} and at {n2:,} differs in 'steps' only "
                                   f"({len(a)} keys)", f"differs in {d}")
        for dest, (want, _why) in TARGET_KNOBS.items():
            rep.expect(dest in a and a[dest] == want and b.get(dest) == want,
                       f"arm {arm}: {dest} = {want!r}", f"resolved {a.get(dest)!r} / {b.get(dest)!r}")


def check_phase2_role(rep: Report, pin: Optional[str] = None) -> None:
    from main.train.combination_checks import fresh_only_flags
    from main.train.fork_lr import is_same_run_checkpoint
    for arm in arms.ARMS:
        for seed in arms.SEEDS[arm]:
            argv = arms.resume_argv(arm, seed, pin)
            name = arms.run_name(arm, seed)
            model = argv[argv.index("--model") + 1]
            bad = []
            if not is_same_run_checkpoint(model, f"models/{name}"):
                bad.append("not a same-run checkpoint")
            if "--run-name" in argv or "--run-dir" in argv:
                bad.append("names a run dir (a fork)")
            if set(argv) & set(fresh_only_flags()):
                bad.append("carries a fresh-only flag")
            if "--pin-commit" not in argv or argv[argv.index("--pin-commit") + 1] != pin:
                bad.append("does not hold the pin")
            if set(argv) & (set(arms.WIDE_LEVER) | set(arms.BREADTH_LEVER) | {"--seed"}):
                bad.append("types a lever or the seed (they are inherited from the run's record)")
            if bad:
                rep.diff(f"{name}: the phase-2 command {'; '.join(bad)}")
    if not rep.differences:
        rep.ok(f"all {len(arms.LAUNCH_ORDER)} phase-2 commands: a same-run RESTART "
               "(fork_lr.is_same_run_checkpoint), no fresh-only flag, no run name, no lever or seed typed, the pin "
               "held by --pin-commit")


def static_report(tree: Optional[str] = None, pin: Optional[str] = None) -> Report:
    rep = Report()
    rep.head("the resolved config of every arm at both phase targets")
    check_resolved(rep, pin)
    rep.head("the phase-2 command's role")
    check_phase2_role(rep, pin)
    if tree is not None:
        the_pin = arms._pin(pin)
        rep.head(f"the code at the pin {the_pin[:12]} (objects read from {tree})")
        check_code_is_the_pins(rep, the_pin)
        try:
            check_inventory(classify_readers(pin_files(tree, the_pin)), rep)
        except RuntimeError as e:
            rep.diff(f"the source inventory could not be read, so no reader was audited: {e}")
    rep.head("what the phase boundary is that a six-hour restart is not (bookkeeping; none is a difference)")
    for n in BOUNDARY_NOTES:
        rep.note(n)
    return rep


# ----------------------------------------------------------------------------------------------- child argvs
def phase2_child_argv(launcher_argv: Sequence[str]) -> Tuple[List[str], str]:
    """``(child argv, run dir)`` of a ``--model`` launch, by the launcher's own pure resolvers in
    ``_prepare_session``'s order: anchor, run dir, ``--run-dir``, the recorded recipe passed TYPED."""
    from main import checkargs
    from main.launcher.checkpoint import (_insert_or_replace_run_dir_arg, archive_anchored_args,
                                          resolve_launch_run_dir)
    from main.train.recipe_surface import materialize_recorded_recipe
    child = archive_anchored_args(checkargs.forwarded_argv(list(launcher_argv)))
    run_dir = resolve_launch_run_dir(child, "unused")
    child = _insert_or_replace_run_dir_arg(child, run_dir)
    child, _added = materialize_recorded_recipe(child, run_dir)
    return child, run_dir


def restart_child_argv(fresh_launcher_argv: Sequence[str], checkpoint: str, run_dir: str) -> List[str]:
    """The child argv of the launcher's own same-run restart of a FRESH launch (``resume_child_args``)."""
    from main import checkargs
    from main.launcher.checkpoint import resume_child_args
    child, _stripped = resume_child_args(checkargs.forwarded_argv(list(fresh_launcher_argv)), checkpoint, run_dir)
    return child


def child_namespace(child_argv: Sequence[str]) -> Dict[str, Any]:
    """The namespace the trainer resolves a CHILD argv to: its parser, then ``resolve_config`` (which restores the
    untyped knobs of a ``--model`` launch from the run's record), then ``inherit_team_mix`` (the mix's own route)."""
    import contextlib

    from main.train.config import resolve_config
    from main.train.parser import build_parser
    parser = build_parser()
    ns = parser.parse_args(list(child_argv))
    from main.train.matchup_setup import inherit_team_mix
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(ns, parser)
        inherit_team_mix(ns, emit=lambda *_a, **_k: None)   # the mix has its own route (launch_flags.OWN_ROUTE)
    return {k: v for k, v in vars(ns).items() if not k.startswith("_")}


# ----------------------------------------------------------------------------------------------- optimizer
def _raw_zip_state(zip_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(optimizer state_dict, policy state_dict)`` exactly as the zip holds them."""
    import torch
    with zipfile.ZipFile(zip_path) as z:
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=False)
        pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    return opt, pol


def loaded_vs_saved(zip_path: str, ns: Dict[str, Any]) -> Dict[str, Any]:
    """Load ``zip_path`` the way the trainer's resume does (``load_model_snapshot`` under the resolved namespace's
    version, then the by-name optimizer realignment) and compare what is in memory with what the zip holds.

    Returns counts and the list of mismatches (empty = tensor-equal)."""
    import contextlib

    import torch

    from agents.model.snapshot import load_model_snapshot
    from agents.training.reward_config import RewardConfig
    from main.train.checkpoint_state import _validate_or_reset_optimizer_state
    from main.train.fork_arch import resume_version_build

    args = SimpleNamespace(**ns)
    saved_opt, saved_pol = _raw_zip_state(zip_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reward_config = RewardConfig.from_args(args)
        version, _layout, _pk = resume_version_build(args, reward_config=reward_config)
        model = load_model_snapshot(
            zip_path, env=None, current_version=version, device="cpu", enforce_vf_coef=args.vf_coef,
            enforce_reward_config=reward_config, enforce_belief_grad_mode=args.belief_grad_mode,
            enforce_oracle_reveal=args.oracle_reveal)
        _validate_or_reset_optimizer_state(model, zip_path)
    bad: List[str] = []
    said = buf.getvalue()
    if "[Resume]" in said:
        bad.append("the optimizer realignment did not carry the state verbatim: "
                   + "; ".join(ln for ln in said.splitlines() if "[Resume]" in ln)[:300])
    loaded_opt = model.policy.optimizer.state_dict()
    loaded_pol = model.policy.state_dict()
    n_tensors = 0
    steps = set()
    if sorted(loaded_opt["state"]) != sorted(saved_opt["state"]):
        bad.append(f"optimizer state entries: loaded {len(loaded_opt['state'])}, saved {len(saved_opt['state'])}")
    for idx in sorted(set(loaded_opt["state"]) & set(saved_opt["state"])):
        lo, sa = loaded_opt["state"][idx], saved_opt["state"][idx]
        if sorted(lo) != sorted(sa):
            bad.append(f"optimizer state[{idx}] keys {sorted(lo)} vs {sorted(sa)}")
            continue
        for key in sorted(sa):
            lv, sv = lo[key], sa[key]
            if torch.is_tensor(sv):
                n_tensors += 1
                if key == "step":
                    steps.add(float(sv))
                if not (torch.is_tensor(lv) and lv.shape == sv.shape and torch.equal(lv.cpu(), sv.cpu())):
                    bad.append(f"optimizer state[{idx}][{key}] differs")
            elif lv != sv:
                bad.append(f"optimizer state[{idx}][{key}]: loaded {lv!r}, saved {sv!r}")
    groups = []
    for gi, (lg, sg) in enumerate(zip(loaded_opt["param_groups"], saved_opt["param_groups"])):
        for key in sorted(set(lg) | set(sg)):
            if lg.get(key) != sg.get(key):
                bad.append(f"optimizer param_groups[{gi}][{key}]: loaded {lg.get(key)!r}, saved {sg.get(key)!r}")
        groups.append({k: sg[k] for k in ("lr", "betas", "eps", "weight_decay") if k in sg})
    if len(loaded_opt["param_groups"]) != len(saved_opt["param_groups"]):
        bad.append("optimizer param group count differs")
    if sorted(loaded_pol) != sorted(saved_pol):
        bad.append(f"policy keys: loaded {len(loaded_pol)}, saved {len(saved_pol)}")
    n_weights = 0
    for key in sorted(set(loaded_pol) & set(saved_pol)):
        n_weights += 1
        if not torch.equal(loaded_pol[key].cpu(), saved_pol[key].cpu()):
            bad.append(f"policy tensor {key} differs")
    return {"mismatches": bad, "optimizer_entries": len(saved_opt["state"]), "optimizer_tensors": n_tensors,
            "adam_steps": sorted(steps), "param_groups": groups, "policy_tensors": n_weights,
            "num_timesteps": int(model.num_timesteps)}


# ----------------------------------------------------------------------------------------------- the smoke
#: The smoke's shape: a ``--debug`` CPU run, checkpoints every 2,000 steps, and a TYPED rollout shape (one update
#: per 4,096 completed-game rows, micro-batches of 256, one epoch). Typed because a run that took ``--debug``'s
#: automatic shape (2,304 rows / batch 384) cannot be resumed at all, by a six-hour restart or by phase 2: the
#: resume restores the recorded 2,304 while ``--batch-size`` falls back to the parser's 4,096 (inert on a resume),
#: and the parse-time check refuses 2,304 as off the lcm(4,096, 256) quantum (seen 2026-11-09, both forms). 4,096
#: is the smallest target on both the fresh quantum and the resume-side one. No registered run is ``--debug``.
SMOKE_CHECKPOINT_EVERY = 2000
SMOKE_SHAPE = ("--rollout-target-samples", "4096", "--batch-size", "256", "--n-epochs", "1")
SMOKE_LOGS = ("a.log", "b1.log", "b2_dryrun.log", "b2.log", "b3.log")


def _swap(argv: Sequence[str], flag: str, value: Optional[str]) -> List[str]:
    """``argv`` with ``flag``'s value replaced (``value`` None: the flag and its value dropped)."""
    out = list(argv)
    i = out.index(flag)
    out[i:i + 2] = [] if value is None else [flag, value]
    return out


def smoke_fresh_argv(arm: str, name: str, steps: int) -> List[str]:
    """The kit's phase-1 command for ``arm`` with the smoke's substitutions and nothing else: no pin (the checkout
    that runs the smoke IS the code under test), the CPU, ``--debug`` with the smoke's typed rollout shape
    (``SMOKE_SHAPE``), a scratch name and target."""
    argv = arms.build_argv(arm, arms.SEEDS[arm][0], "0" * 40, steps=arms.PHASE_STEPS[0])
    argv = _swap(argv, "--pin-commit", None)
    argv = _swap(argv, "--device", "cpu")
    argv = _swap(argv, "--checkpoint-every-steps", str(SMOKE_CHECKPOINT_EVERY))
    argv = _swap(argv, "--steps", str(steps))
    argv = _swap(argv, "--run-name", name)
    return [*argv, "--no-pin", "--debug", *SMOKE_SHAPE]


def smoke_resume_argv(arm: str, run_dir: str, steps: int) -> List[str]:
    """The kit's phase-2 command (``arms.resume_argv``) with the same substitutions: ``--model`` names the scratch
    run's ``final_model.zip``, no lever, no seed, no run name."""
    argv = arms.resume_argv(arm, arms.SEEDS[arm][0], "0" * 40, steps=arms.PHASE_STEPS[1])
    argv = _swap(argv, "--pin-commit", None)
    argv = _swap(argv, "--device", "cpu")
    argv = _swap(argv, "--model", os.path.join(run_dir, "final_model.zip"))
    argv = _swap(argv, "--steps", str(steps))
    return [*argv, "--no-pin", "--debug"]


def run_smoke(models_dir: str, arm: str = "wb", half: int = 4000, python: str = sys.executable) -> Dict[str, str]:
    """Run the two-segment smoke under ``models_dir`` (which must not hold a run yet): run A straight to
    ``2 * half``; run B to ``half``, copied aside as the stop left it, then resumed to ``2 * half`` by the phase-2
    command form, then that same command once more (it must train nothing). Returns the paths ``compare`` takes.

    The launches are this process's own foreground children, through ``python -m main.launcher``; A and B's first
    segment run side by side."""
    import shutil
    root = Path(models_dir)
    runs, logs, stop = root / "runs", root / "logs", root / "stop"
    if runs.exists() and any(runs.iterdir()):
        raise SystemExit(f"{runs} already holds a run: the smoke takes an empty directory")
    for d in (runs, logs, stop):
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GEN3AI_MODELS_DIR=str(runs), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("GEN3AI_TEST_ALLOW_GPU", None)
    from utils.paths import repo_root
    cwd = str(repo_root())
    env["PYTHONPATH"] = os.path.join(cwd, "src")
    a_name, b_name = prefixed(f"b1smoke_a_{arm}"), prefixed(f"b1smoke_b_{arm}")
    a_dir, b_dir = runs / a_name, runs / b_name

    def launch(argv: Sequence[str], log: str, extra: Sequence[str] = ()) -> subprocess.Popen:
        fh = open(logs / log, "w")
        return subprocess.Popen([python, "-m", "main.launcher", *argv, *extra], stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, cwd=cwd)

    rcs: Dict[str, int] = {}
    pa = launch(smoke_fresh_argv(arm, a_name, 2 * half), "a.log")
    pb = launch(smoke_fresh_argv(arm, b_name, half), "b1.log")
    rcs["b1"] = pb.wait()
    if rcs["b1"] == 0:                       # phase 2 runs beside what is left of run A
        shutil.copytree(b_dir, stop / b_name, symlinks=True)
        resume = smoke_resume_argv(arm, str(b_dir), 2 * half)
        rcs["b2_dryrun"] = launch(resume, "b2_dryrun.log", ["--dry-run"]).wait()
        rcs["b2"] = launch(resume, "b2.log").wait()
        rcs["b3"] = launch(resume, "b3.log").wait()
    rcs["a"] = pa.wait()
    (logs / "exit_codes.json").write_text(json.dumps(rcs, indent=1, sort_keys=True) + "\n")
    return {"run_through": str(a_dir), "phased": str(b_dir), "stop": str(stop / b_name), "logs": str(logs)}


# ----------------------------------------------------------------------------------------------- compare
#: ``metadata.json`` ``cli_args`` keys that differ between a recorded namespace and the one the registered argv
#: resolves to (``arms.resolved_namespace(arms.build_argv(...))``), each with why. Everything else must be equal,
#: after the fresh process, after a six-hour-restart child and after the phase-2 resume (``compare`` shows it).
#: The ``metadata.json:cli_args`` keys that differ from the registered (fresh) namespace WITHOUT the run differing:
#: ``cli_args`` is the LATEST process's resolved namespace, and a ``--model`` launch deliberately does not carry
#: these (``main.train.launch_flags.PER_LAUNCH``, ``recipe_surface.INERT_ON_RESUME``) or re-labels where a value
#: came from (``*_source``). Everything else in the namespace is restored on a same-run resume and is comparable.
_INERT = ("inert on a resume (recipe_surface.INERT_ON_RESUME): SB3 restores the checkpoint's own and the resume's "
          "namespace keeps the parser's default (4096 / 2048 against production's 2048 / 384). The run's true value "
          "is metadata.json's TOP-LEVEL field of the same name")
_SOURCE = ("provenance of a tri-state flag: 'default' on the fresh process, 'inherited' on every resume (the value "
           "itself is inherited and stays compared)")
_SUGAR = ("umbrella sugar (launch_flags.PER_LAUNCH, fresh-only): 'both' under --arch production on the fresh "
          "process, the parser's 'off' on every resume; its components are recorded fields of model_config.json")
CLI_ARGS_VOLATILE: Dict[str, str] = {
    "steps": "this launch's target (PER_LAUNCH identity): phase 2 types the higher one",
    "model": "None on the fresh process; the checkpoint path on every resume child (a restart's and phase 2's)",
    "run_dir": "None in the registered namespace; the launcher passes EVERY child the run's absolute --run-dir",
    "run_name": "the fresh argv types it and a six-hour-restart child keeps it; the phase-2 child has none (naming "
                "a --model launch makes it a fork) and the run's --run-dir, which outranks a name, places both",
    "arch": "fresh-only umbrella: the launcher strips --arch from every resume child (None); the architecture a "
            "resume trains is model_config.json's",
    "unified_moves": _SUGAR,
    "unified_damage": _SUGAR,
    "allow_nonproduction_arch": "a FRESH run's consent (PER_LAUNCH): True on the wide arms' fresh process and in a "
                                "six-hour-restart child (which keeps the typed flag), False after the phase-2 "
                                "resume (its command does not type it); read only by the fresh-run arch report",
    "eval_mirrored_pairs_source": _SOURCE,
    "eval_sentinel_greedy_source": _SOURCE,
    "promote_threshold_source": _SOURCE,
    "promotion_sprt_source": _SOURCE,
    "batch_size": _INERT,
    "n_steps": _INERT,
}
#: The third inert flag, ``lr``, is NOT volatile here: the parser's default is the registered 0.0003, so the key
#: compares equal after a resume, but vacuously (the live rate is the KL controller's, in ``current_lr``).
INERT_KEYS = ("batch_size", "n_steps", "lr")
#: What the FRESH process's record may differ from the registered namespace in (the larger set is a resume's).
FRESH_VOLATILE = ("run_dir", "steps")

CHILD_LOG = "launcher_child.log"
_CONTINUING_RE = re.compile(r"Continuing Training \(Steps: ([\d,]+) remaining of ([\d,]+), LR: (\S+) \(saved=(\S+) ")
_RESUMING_RE = re.compile(r"Resuming at LR (\S+), epochs (\d+) \(checkpoint LR=([^)\s]+)\)")
_DRY_STEPS_RE = re.compile(r"steps\s+: --steps ([\d,]+) vs checkpoint at ([\d,]+) \(sidecar\) → \+([\d,]+) steps")
_ZERO_RESTARTS = "Restarts   : 0"
_ENV_UP = "[RUST ENV] core up"
#: The collector's seeds in a checkpoint sidecar's ``env_core`` block: a hash of ``--seed`` and the step the PROCESS
#: started at (``rust_env_setup.segment_seed`` / ``bot_route_seed``), so they change at every resume of either kind.
SEGMENT_SEED_KEYS = ("bot_seed", "run_seed")
_COMPLETE_RE = re.compile(r"Training already complete \(([\d,]+) / ([\d,]+) steps\)")
#: The constants a schedule could have moved (tensorboard tag -> the resolved-namespace key that sets it).
SCHEDULE_TAGS = {"train/clip_range": "clip_range", "hparams/ent_coef": "ent_coef"}
LR_TAG = "train/learning_rate"
#: Written once per rollout, at the step the update is taken: the run's update boundaries.
UPDATE_TAG = "rollout/collect_decisions"
#: Per-checkpoint sidecar keys that may differ between two runs at one step without the recipe differing: the
#: learning rate the KL controller has reached (and the dose block that quotes it), and the pin row's last step.
SIDECAR_MOVING = ("current_lr", "dose", "pin_history")
_MATCHUP_RE = re.compile(r"\[MATCHUP ([0-9a-f]+)\]")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tb_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Every scalar tag the run's own event files hold, as ``(step, value)`` rows in file then step order."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: Dict[str, List[Tuple[int, float]]] = {}
    for f in sorted((run_dir / "tb").glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((int(e.step), float(e.value)) for e in ea.Scalars(tag))
    return out


def recorded_vs_registered(recorded: Dict[str, Any], registered: Dict[str, Any]) -> List[str]:
    """The public keys in which a recorded ``cli_args`` and a resolved namespace differ (JSON-normalised: the record
    went through ``json``), sorted. Keys starting with ``_`` are the trainer's private bookkeeping, not arguments."""
    norm = json.loads(json.dumps(registered, default=str))
    keys = {k for k in set(recorded) | set(norm) if not k.startswith("_")}
    return sorted(k for k in keys if recorded.get(k, "<absent>") != norm.get(k, "<absent>"))


def compare_report(run_through: str, phased: str, stop: str, logs: str, arm: str = "wb") -> Report:
    """The two-segment smoke's evidence: run A (one launch to 2N), run B (to N, then resumed to 2N), B as the stop
    left it, and the five launcher logs."""
    rep = Report()
    a_dir, b_dir, s_dir, log_dir = Path(run_through), Path(phased), Path(stop), Path(logs)
    rep.head("the inputs")
    need = [d / f for d in (a_dir, b_dir, s_dir) for f in ("metadata.json", "model_config.json", "final_model.zip")]
    need += [log_dir / n for n in (*SMOKE_LOGS, "exit_codes.json")]
    need += [d / CHILD_LOG for d in (a_dir, b_dir, s_dir)]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        rep.diff(f"{len(missing)} missing input(s), nothing can be compared: " + ", ".join(missing))
        return rep
    meta = {k: _read_json(d / "metadata.json") for k, d in (("a", a_dir), ("b", b_dir), ("s", s_dir))}
    conf = {k: _read_json(d / "model_config.json") for k, d in (("a", a_dir), ("b", b_dir), ("s", s_dir))}
    rcs = _read_json(log_dir / "exit_codes.json")
    text = {n: (log_dir / n).read_text(errors="replace") for n in SMOKE_LOGS}
    child = {k: (d / CHILD_LOG).read_text(errors="replace") for k, d in (("a", a_dir), ("b", b_dir), ("s", s_dir))}
    if any(not isinstance(v, dict) for v in (*meta.values(), *conf.values(), rcs)):
        rep.diff("a sidecar or exit_codes.json is not a JSON object")
        return rep
    target, half = int(meta["a"]["cli_args"]["steps"]), int(meta["s"]["cli_args"]["steps"])
    rep.ok(f"run-through {a_dir.name} to {target:,}; phased {b_dir.name} to {half:,} then {target:,}; "
           f"exit codes {json.dumps(rcs, sort_keys=True)}")
    rep.expect(target == 2 * half and int(meta["b"]["cli_args"]["steps"]) == target,
               "the phased run's second target is the run-through's, twice its first",
               f"{half} / {meta['b']['cli_args']['steps']} / {target}")

    # ---- the launcher took the resume -------------------------------------------------------------------------
    rep.head("the launcher accepts the phase-2 command on a run that ENDED at its target")
    for seg in ("a", "b1", "b2", "b3"):
        rep.expect(rcs.get(seg) == 0, f"segment {seg} exited 0", f"exit {rcs.get(seg)!r}")
    dry = [ln.strip() for ln in text["b2_dryrun.log"].splitlines() if re.match(r"\s*role\s+:", ln)]
    rep.expect(rcs.get("b2_dryrun") == 0 and len(dry) == 1 and "RESTART" in dry[0] and "FORK" not in dry[0],
               "the dry run of the phase-2 command resolves to a same-run RESTART and would launch",
               f"exit {rcs.get('b2_dryrun')!r}, role line(s) {dry}")
    for ln in dry:
        rep.note(f"dry run: {ln}")
    # The trainer's own stdout is the run's launcher_child.log, appended to by every child of the run.
    rep.expect(child["b"].startswith(child["s"]),
               f"{CHILD_LOG}: what phase 1 wrote ({len(child['s'].splitlines())} lines) opens the phased run's "
               f"({len(child['b'].splitlines())} lines), unchanged")
    later = child["b"][len(child["s"]):] if child["b"].startswith(child["s"]) else child["b"]
    cont = _CONTINUING_RE.findall(later)
    done = _COMPLETE_RE.findall(later)
    stop_steps = int(meta["s"]["num_timesteps"])
    rep.expect(len(cont) == 1 and len(_RESUMING_RE.findall(text["b2.log"])) == 1 and _ZERO_RESTARTS in text["b2.log"],
               "phase 2 was ONE child: one 'Continuing Training' line in the run's child log, one 'Resuming at LR' "
               "event and 'Restarts : 0' in the launcher's", f"{len(cont)} line(s): {cont}")
    rep.expect(not _CONTINUING_RE.findall(child["s"] + child["a"]) and not _COMPLETE_RE.findall(child["s"] + child["a"])
               and not _RESUMING_RE.findall(text["b1.log"] + text["a.log"]),
               "phase 1 and the run-through trained from scratch (no resume line in either)")
    rep.expect(len(done) == 1 and _int(done[0][1]) == target and _int(done[0][0]) == int(meta["b"]["num_timesteps"])
               and later.index("Training already complete") > later.index("Continuing Training")
               and _ENV_UP in text["b2.log"] and _ENV_UP not in text["b3.log"],
               "the same command AGAIN on the finished run trains nothing: 'Training already complete "
               f"({meta['b']['num_timesteps']:,} / {target:,} steps)', no environment built, exit 0",
               f"line(s) {done}")

    # ---- the step count ---------------------------------------------------------------------------------------
    rep.head("num_timesteps across the stop")
    a_end, b_end = int(meta["a"]["num_timesteps"]), int(meta["b"]["num_timesteps"])
    rep.expect(half <= stop_steps < target, f"phase 1 stopped at {stop_steps:,}: at or past its target {half:,}, "
                                            f"short of {target:,}")
    if cont:
        remaining, of, lr_now, lr_saved = _int(cont[0][0]), _int(cont[0][1]), cont[0][2], cont[0][3]
        rep.expect(of == target and of - remaining == stop_steps,
                   f"phase 2 started at the stop's own count ({of:,} - {remaining:,} remaining = {stop_steps:,})",
                   f"{of} - {remaining} != {stop_steps}")
    drys = _DRY_STEPS_RE.findall(text["b2_dryrun.log"])
    rep.expect(len(drys) == 1 and [_int(x) for x in drys[0]] == [target, stop_steps, target - stop_steps],
               f"the dry run reads the same: --steps {target:,} vs checkpoint at {stop_steps:,} -> "
               f"+{target - stop_steps:,} steps", f"line(s) {drys}")
    tb = {k: tb_scalars(d) for k, d in (("a", a_dir), ("b", b_dir), ("s", s_dir))}
    ups = {k: sorted({st for st, _v in tb[k].get(UPDATE_TAG, [])}) for k in tb}
    rep.expect(bool(ups["s"]) and ups["b"][:len(ups["s"])] == ups["s"] and ups["s"][-1] == stop_steps
               and all(st > stop_steps for st in ups["b"][len(ups["s"]):]) and ups["b"][-1] == b_end,
               f"update boundaries: phase 1's {ups['s']} open the phased run's {ups['b']}; the last is the stop "
               f"({stop_steps:,}), phase 2's follow it and end at the run's final count ({b_end:,}); the third launch "
               "added none", f"stop {ups['s']}, phased {ups['b']}, run-through {ups['a']}")
    rep.note(f"the run-through's update boundaries: {ups['a']}")
    for k, end in (("a", a_end), ("b", b_end)):
        steps = sorted({st for st, _v in tb[k].get(UPDATE_TAG, [])})
        before = [st for st in steps if st < end]
        prev = max(before) if before else -1
        rep.expect(end >= target and 0 <= prev < target,
                   f"run {k.upper()} ended at {end:,}: the first update boundary at or past {target:,} "
                   f"(the one before: {prev:,})", f"ended at {end}, target {target}, earlier updates at {before}")
    rep.note(f"A ended at {a_end:,} and B at {b_end:,}: an update boundary is a count of COMPLETED games' rows, so "
             "the two need not be the same step (nor would two run-throughs' be)")

    # ---- the sidecars -----------------------------------------------------------------------------------------
    rep.head("the sidecars")
    for k, other in (("a", "the run-through"), ("s", "its own phase-1 stop")):
        d = sorted(key for key in set(conf["b"]) | set(conf[k]) if conf["b"].get(key) != conf[k].get(key))
        rep.expect(not d, f"model_config.json: the phased run's equals {other}'s ({len(conf['b'])} keys)",
                   f"differs in {d}")
    d = sorted(key for key in set(meta["a"]["cli_args"]) | set(meta["b"]["cli_args"])
               if meta["a"]["cli_args"].get(key) != meta["b"]["cli_args"].get(key))
    d = [key for key in d if not key.startswith("_")]
    rep.expect(set(d) <= set(CLI_ARGS_VOLATILE),
               f"metadata.json cli_args ({len(meta['b']['cli_args'])} keys; the LATEST process's namespace): the "
               f"phased run's public keys equal the run-through's but for the volatile {d}",
               f"differs outside the volatile set in {sorted(set(d) - set(CLI_ARGS_VOLATILE))}")
    for k in ("a", "s", "b"):
        role = (meta[k].get("lineage") or {}).get("role")
        rep.expect(role == "fresh" and (meta[k]["lineage"].get("fork_parent") is None),
                   f"lineage role of {k.upper()}: fresh, no fork parent", f"role {role!r}, "
                   f"fork_parent {(meta[k].get('lineage') or {}).get('fork_parent')!r}")
    rep.expect(meta["b"].get("lineage") == meta["s"].get("lineage")
               and meta["b"].get("original_command") == meta["s"].get("original_command"),
               "the lineage block and original_command are, after phase 2, what phase 1 wrote (immutable)")
    for k in ("a", "b"):
        rows = meta[k].get("pin_history") or []
        rep.expect(len(rows) == 1 and rows[0].get("first_step") == 0
                   and rows[0].get("last_step") == meta[k]["num_timesteps"]
                   and rows[0].get("git_hash") == meta[k].get("git_hash"),
                   f"pin_history of {k.upper()}: ONE row, steps 0 to {meta[k]['num_timesteps']:,}, the run's git_hash",
                   json.dumps(rows))
    rep.expect(meta["a"].get("git_hash") == meta["b"].get("git_hash") == meta["s"].get("git_hash"),
               f"one code version in all three ({str(meta['b'].get('git_hash'))[:12]})")
    rep.expect(meta["a"].get("era") == meta["b"].get("era") == meta["s"].get("era") and bool(meta["b"].get("era")),
               f"the era block: {json.dumps(meta['b'].get('era'), sort_keys=True)}")
    training = [key for key, v in sorted(meta["a"].items()) if isinstance(v, (int, float, str, type(None)))
                and key not in ("saved_at", "num_timesteps", "launcher_command", "original_command")]
    d = [key for key in training if meta["a"].get(key) != meta["b"].get(key)]
    rep.expect(not d, f"metadata.json's {len(training)} scalar training fields (batch_size, clip_range, ent_coef, "
                      "gamma, n_steps, optimizer, vf_coef, weight_decay, versions ...) are equal", f"differ: {d}")

    # ---- the recorded namespace against the registered one -----------------------------------------------------
    rep.head("the recorded cli_args against the namespace the kit's argv resolves to (volatile keys: "
             + ", ".join(CLI_ARGS_VOLATILE) + ")")
    registered = arms.resolved_namespace(smoke_fresh_argv(arm, b_dir.name, half))
    resume_launch = smoke_resume_argv(arm, str(s_dir), target)
    child2, run_dir2 = phase2_child_argv(resume_launch)
    ns2 = child_namespace(child2)
    child_r = restart_child_argv(smoke_fresh_argv(arm, b_dir.name, target), str(s_dir / "final_model.zip"), str(s_dir))
    ns_r = child_namespace(child_r)
    for label, rec in (("after the FRESH process (the stop's record)", meta["s"]["cli_args"]),
                       ("after the PHASE-2 resume (the phased run's record)", meta["b"]["cli_args"]),
                       ("the run-through's record", meta["a"]["cli_args"]),
                       ("a SIX-HOUR-RESTART child's namespace (resolved against the stop; not a recorded one)", ns_r),
                       ("the PHASE-2 child's namespace (resolved against the stop)", ns2)):
        d = recorded_vs_registered(rec, registered)
        rep.expect(set(d) <= set(CLI_ARGS_VOLATILE), f"{label}: differs from the registered namespace in {d}",
                   f"keys outside the volatile set: {sorted(set(d) - set(CLI_ARGS_VOLATILE))}")
    d = recorded_vs_registered(meta["s"]["cli_args"], registered)
    rep.expect(set(d) <= set(FRESH_VOLATILE), f"the FRESH process's record differs in no more than {FRESH_VOLATILE}: "
                                              "every other volatile key is a resume's doing", f"differs in {d}")
    for key in INERT_KEYS[:2]:
        rep.expect(meta["b"].get(key) == meta["s"].get(key) == meta["a"].get(key) is not None,
                   f"the run's TRUE {key} (metadata.json top level) is {meta['s'].get(key)!r} in A, at the stop and "
                   f"after phase 2, whatever cli_args says after the resume ({meta['b']['cli_args'].get(key)!r})",
                   f"A {meta['a'].get(key)!r}, stop {meta['s'].get(key)!r}, B {meta['b'].get(key)!r}")

    # ---- the two child argvs ----------------------------------------------------------------------------------
    rep.head("the phase-2 child argv against the launcher's own six-hour-restart child argv")
    d = arms.namespace_diff(ns2, ns_r)
    typed_inert = sorted({f.lstrip("-").replace("-", "_") for f in SMOKE_SHAPE[::2]} & set(INERT_KEYS))
    rep.expect(set(d) <= {"run_name", "allow_nonproduction_arch", *typed_inert},
               f"both resolve, against the stopped run's record, to one namespace ({len(ns2)} keys; target "
               f"{ns2.get('steps'):,}, model {os.path.basename(str(ns2.get('model')))}) but for {d}: a restart child "
               "still carries the fresh argv's --run-name beside the --run-dir that outranks it "
               "(run_io._resolve_model_dir and matchup_setup._planned_run_dir read --run-dir first) and, on a wide "
               "arm, the fresh run's typed consent --allow-nonproduction-arch, which only a FRESH run's arch report "
               "reads (main/train/config.py: arch_surface.report(fresh=not args.model, allowed=...))"
               + (f", and the smoke's typed {typed_inert}, inert on a resume, which a restart child keeps typed and "
                  "the phase-2 command does not carry (no registered argv types an inert flag)" if typed_inert else ""),
               f"differ in {d}")
    rep.expect(Path(run_dir2) == s_dir, "the phase-2 command resolves to the run's OWN folder", str(run_dir2))
    rep.note(f"token-different, as stated: the phase-2 child types {len(child2)} tokens, the restart child "
             f"{len(child_r)}")

    # ---- what the resume loads --------------------------------------------------------------------------------
    rep.head("the optimizer and weights the resume LOADS against what the stop SAVED")
    got = loaded_vs_saved(str(s_dir / "final_model.zip"), ns2)
    rep.expect(not got["mismatches"],
               f"tensor-equal: {got['optimizer_entries']} optimizer entries ({got['optimizer_tensors']} tensors, Adam "
               f"step {got['adam_steps']}), {got['policy_tensors']} policy tensors, param groups "
               f"{json.dumps(got['param_groups'])}", "; ".join(got["mismatches"][:5]))
    rep.expect(got["num_timesteps"] == stop_steps, f"the zip's num_timesteps is the sidecar's ({stop_steps:,})",
               str(got["num_timesteps"]))
    saved_lr = got["param_groups"][0]["lr"] if got["param_groups"] else None
    if cont and saved_lr is not None:
        rep.expect(lr_now == lr_saved == f"{saved_lr:.2e}" and "Clamping resume LR" not in text["b2.log"],
                   f"phase 2 resumed at the saved optimizer LR, unclamped ({lr_now} = saved {lr_saved} = the zip's "
                   f"{saved_lr!r})", f"now {lr_now}, saved {lr_saved}, zip {saved_lr!r}")

    # ---- the schedules ----------------------------------------------------------------------------------------
    rep.head("schedule values the runs recorded")
    for tag, dest in SCHEDULE_TAGS.items():
        want = registered[dest]
        for k in ("a", "b"):
            vals = sorted({round(v, 6) for _st, v in tb[k].get(tag, [])})
            rep.expect(vals == [round(float(want), 6)],
                       f"{tag} in run {k.upper()}: {want!r} in each of its {len(tb[k].get(tag, []))} rows",
                       f"values {vals}")
    lo = registered["min_lr"]
    for k in ("a", "b"):
        rows = tb[k].get(LR_TAG, [])
        rep.expect(bool(rows) and all(lo - 1e-12 <= v for _st, v in rows),
                   f"{LR_TAG} in run {k.upper()}: {len(rows)} rows, every one at or above min_lr {lo!r} "
                   f"(range {min((v for _s, v in rows), default=None)!r} .. "
                   f"{max((v for _s, v in rows), default=None)!r}; the KL controller moves it, no schedule does)")
    s_lr, b_lr = tb["s"].get(LR_TAG, []), tb["b"].get(LR_TAG, [])
    rep.expect(bool(s_lr) and b_lr[:len(s_lr)] == s_lr,
               f"the {len(s_lr)} learning-rate rows phase 1 wrote open the phased run's record, unchanged")
    cks = {k: {p.name: _read_json(p) for p in sorted((d / "checkpoints").glob("checkpoint_*_steps.json"))}
           for k, d in (("a", a_dir), ("b", b_dir))}
    both = sorted(set(cks["a"]) & set(cks["b"]), key=lambda n: int(n.split("_")[1]))
    rep.expect(bool(both), f"checkpoint sidecars at steps both runs have: {[int(n.split('_')[1]) for n in both]}")
    from main.train.rust_env_setup import bot_route_seed, segment_seed
    seed = int(registered["seed"])
    for name in both:
        ca, cb = cks["a"][name] or {}, cks["b"][name] or {}
        step = int(name.split("_")[1])
        ea, eb = dict(ca.get("env_core") or {}), dict(cb.get("env_core") or {})
        seeds = {k: {key: e.pop(key, None) for key in SEGMENT_SEED_KEYS} for k, e in (("a", ea), ("b", eb))}
        d = sorted(key for key in set(ca) | set(cb) if key != "env_core" and ca.get(key) != cb.get(key))
        rep.expect(set(d) <= set(SIDECAR_MOVING) and ea == eb and bool(ea),
                   f"{name}: A's and B's are equal in {len(set(ca) | set(cb)) - len(d)} keys (ent_coef "
                   f"{cb.get('ent_coef')!r}, clip_range {cb.get('clip_range')!r}, n_epochs {cb.get('n_epochs')!r}, "
                   f"batch_size {cb.get('batch_size')!r}, n_steps {cb.get('n_steps')!r}, matchup "
                   f"{cb.get('matchup_hash')!r}, env_core but for its segment seeds); differing: {d}; current_lr A "
                   f"{ca.get('current_lr')!r} B {cb.get('current_lr')!r}",
                   f"differ outside {SIDECAR_MOVING}: {sorted(set(d) - set(SIDECAR_MOVING))}; env_core keys "
                   f"{sorted(key for key in set(ea) | set(eb) if ea.get(key) != eb.get(key))}")
        for k, start in (("a", 0), ("b", 0 if step <= stop_steps else stop_steps)):
            want = segment_seed(seed, start)
            rep.expect(seeds[k] == {"run_seed": want, "bot_seed": bot_route_seed(want)},
                       f"{name} of {k.upper()}: the collector's seeds are segment_seed(--seed {seed}, the process's "
                       f"start step {start:,}) = {want}", json.dumps(seeds[k]))
    hashes = {seg: sorted(set(_MATCHUP_RE.findall(text[seg]))) for seg in ("a.log", "b1.log", "b2.log")}
    rep.expect(len({tuple(v) for v in hashes.values()}) == 1 and all(len(v) == 1 for v in hashes.values()),
               f"the matchup spec (who plays which teams, the team-mix share in it) has ONE hash in the run-through, "
               f"phase 1 and phase 2: {hashes['b2.log']}", json.dumps(hashes))
    if "--team-mix" in arms.ARMS[arm]:
        mix = [ln.strip() for ln in text["b2.log"].splitlines() if "[TEAM-MIX]" in ln]
        rep.expect(any("same-run resume: restoring the run's recorded --team-mix" in ln for ln in mix)
                   and not any("FATAL" in ln for ln in mix),
                   "phase 2 restored the run's recorded --team-mix (the '[TEAM-MIX] same-run resume' line)",
                   json.dumps(mix))
        for ln in mix:
            rep.note("b2.log: " + re.sub(r"^\[[\d:]+\]\s*", "", ln)[:200])

    # ---- the pool and the checkpoints -------------------------------------------------------------------------
    rep.head("the self-play pool and the checkpoints across the stop")
    for sub, pat in (("snapshots", "*.zip"), ("checkpoints", "*.zip")):
        kept = sorted((s_dir / sub).glob(pat))
        bad = [p.name for p in kept if not (b_dir / sub / p.name).is_file() or _sha(b_dir / sub / p.name) != _sha(p)]
        rep.expect(not bad, f"{sub}/: the {len(kept)} file(s) the stop left are in the phased run, byte-identical "
                            f"({', '.join(p.name for p in kept) or 'none'})", f"missing or changed: {bad}")
    every = int(meta["s"]["cli_args"]["checkpoint_every_steps"])
    saved = sorted(int(m.group(1)) for p in (s_dir / "checkpoints").glob("checkpoint_*_steps.zip")
                   if (m := re.fullmatch(r"checkpoint_(\d+)_steps\.zip", p.name)))
    due = list(range(every, stop_steps + 1, every))
    missing = [m for m in due if not any(m <= st < m + every for st in saved)]
    rep.expect(bool(due) and not missing and len(saved) == len(due),
               f"the stop left one checkpoint per multiple of {every:,} it crossed (at {saved}; the boundaries are "
               "GLOBAL multiples, named by the first callback step at or past each)",
               f"checkpoints at {saved}, boundaries crossed {due}")
    if half % every == 0:
        hit = [st for st in saved if half <= st <= stop_steps]
        rep.expect(len(hit) == 1,
                   f"the stop left checkpoints/checkpoint_{hit[0] if hit else '?'}_steps.zip, the first callback step "
                   f"at or past the target {half:,}: the file the look reads exists before the run stops",
                   f"no checkpoint in [{half}, {stop_steps}]: {saved}")
        if hit and hit[0] < stop_steps:
            rep.note(f"that checkpoint was saved DURING the last rollout (step {hit[0]:,}); the last update of phase 1 "
                     f"came after it (the stop is at {stop_steps:,}), so it holds the policy one update before "
                     "final_model.zip")
    else:
        rep.note(f"this smoke's target {half:,} is not a multiple of its checkpoint interval {every:,}, so it cannot "
                 "show the checkpoint AT the target (the registered 50,000,000 is a multiple of 1,000,000)")
    return rep


# ----------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.resume_check", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static", help="the resolved config, the phase-2 command's role, the source inventory")
    s.add_argument("--tree", required=True, help="a git checkout that holds the pin's commit (its objects are read)")
    s.add_argument("--pin", required=True, help="the registered pin (PREREG.md §3): a full sha")
    c = sub.add_parser("compare", help="the two-segment smoke's run dirs and logs")
    m = sub.add_parser("smoke", help="run the two-segment CPU smoke, then compare (minutes; a heavy job)")
    for p in (c,):
        p.add_argument("--run-through", required=True)
        p.add_argument("--phased", required=True)
        p.add_argument("--stop", required=True)
        p.add_argument("--logs", required=True)
    m.add_argument("--models-dir", required=True, help="an EMPTY scratch directory (never the real models/)")
    m.add_argument("--half", type=int, default=4000)
    for p in (c, m):
        p.add_argument("--arm", choices=sorted(arms.ARMS), default="wb")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "static":
            rep = static_report(a.tree, a.pin)
        elif a.cmd == "compare":
            rep = compare_report(a.run_through, a.phased, a.stop, a.logs, a.arm)
        else:
            paths = run_smoke(a.models_dir, a.arm, a.half)
            print("smoke: " + json.dumps(paths, sort_keys=True))
            rep = compare_report(arm=a.arm, **paths)
    except arms.ArmError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(rep.render())
    return rep.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
