"""ONE fresh cell against ONE banked BOUNDARY-1 cell: plan, check, play and read the 3 x 3 head-to-head cross.

    python -m main.b1.fresh describe   --arm <fresh_arm.json>                       # the fresh argvs (and resumes)
    python -m main.b1.fresh lever-check --arm <fresh_arm.json> --pin-tree <PIN_DIR>  # FRAME §2 item 2; exit 3 = FAIL
    python -m main.b1.fresh check-run  --arm <J> --run <name> --pin-tree <PIN_DIR> [--step N]   # ONE fresh run, early
    python -m main.b1.fresh plan  --arm <J> --look 50M --pin-tree <PIN_DIR> --b1-look <B1 W>/look.json --out <W>
    python -m main.b1.fresh play  --work <W> --pin-dir <PIN_DIR> --device cuda      # four panels; exit 4 = a panel failed
    python -m main.b1.fresh read  --work <W>                                       # W/result/{inputs,verdict}.json
    python -m main.b1.fresh recheck --work <W>                                     # re-decide from inputs.json alone
    python -m main.b1.fresh bootstrap --work <W>                                   # REPORTED: the panels' teams resampled

The registration is ``designs/research_state/measurements/fresh_cell_frame_2026-11-17/FRAME.md`` (§2, §3, §7), adopted
by a card's ``PREREG.md``. The statistic and every class are the ``main.reads`` rule ``b1_fresh_cell``
(:mod:`main.b1.fresh_rule`). This module reuses BOUNDARY-1's kit and never changes it: the argvs are
:func:`main.b1.arms.build_argv`'s, the banked runs are guarded by :func:`main.b1.look.check_run` itself, the checkpoint,
pin, K9(b) and row rules are :mod:`main.b1.look`'s functions, the panels, pairs and draw rule are
:mod:`main.b1.cross`'s, the team bootstrap is :mod:`main.b1.team_bootstrap`'s helpers.

**The fresh arm** (:func:`load_arm`, schema :data:`ARM_SCHEMA`): one JSON file a card registers, ``{schema, card,
pin, banked_arm, lever, declared_keys, fresh: [{seed, run}], own_control?: [{seed, run}], prereg?}``. ``lever`` is the
tokens typed beyond the banked arm's argv (inserted before ``--checkpoint-every-steps``, FRAME §2 item 2);
``declared_keys`` the resolved-config keys the lever moves. Without ``own_control`` the banked cell is B1's arm
(``C`` or ``W``) and the pin must be B1's launch pin, :data:`B1_LAUNCH_PIN` (FRAME §2 item 1). With it (FRAME §7, the
arm BUYS ITS OWN CONTROL), the control runs type the banked arm's argv at the arm's pin, with their own seeds and
names, and are guarded exactly as the fresh runs are, with no lever.

**The lever check** (:func:`lever_check`, FRAME §2 item 2): every fresh argv and the banked arm's first-seed argv
(or the first control run's) are resolved by THE PIN's parser and ``resolve_config``, in a clean detached checkout of
the pin (:func:`pin_tree_resolver`, a subprocess running the pin's own :func:`main.b1.arms.resolved_namespace`), at
both step targets. It PASSES iff each fresh argv differs from the control's in exactly ``declared_keys`` beyond
``seed`` and ``run_name`` (an own-control argv in nothing beyond them), and resolves its own seed and name. The plan
refuses on any other result.

**A fresh (or own-control) run counts** (:func:`check_cell_run`) iff, as for a B1 run (PREREG §10): its look
checkpoint exists with a matching sidecar step; ONE pin entry, the arm's pin, from step 0 (:func:`main.b1.look.
pin_problems`); it is FRESH with no fork parent; its ``original_command`` is its registered argv token for token at a
registered step target; ``ent_coef`` 0.0125 in its record, its last process and its checkpoint; its last process
resolved every declared key, every one of B1's lever keys (:data:`main.b1.look.CONFIG_KEYS`) and ``n_epochs`` as its
own argv resolves them at the pin (the card's epochs, not B1's ten); and its K9(b) read through the look checkpoint is
PASS. **A banked B1 run counts** iff :func:`main.b1.look.check_run` passes it at the look's step, with its lever keys
from the pin's parser, AND (FRAME §2 item 6) its look checkpoint is byte-identical to what B1's own look read
(``--b1-look``: B1's ``look.json`` for that step; the plan refuses without it). **Across the six runs**: one
``torch_version`` and one ``init_num_threads``; every run's last process records the same public ``cli_args`` as the
first banked run's, except its own keys (:data:`main.b1.look.RUN_OWN_KEYS`), a fresh run's declared keys, and the
keys a resume rewrites between processes of different kinds (:data:`main.b1.look.RESUME_REWRITES`); and the same
rollout shape at ``metadata.json``'s top level (:data:`main.b1.look.TRUE_KEYS`) unless the lever declares it.

**The cells** are the fresh side against the banked side, ``n x n`` (9 at three runs a side), each
``[fresh .zip, banked .zip]``, the same list for every panel; B1's panels and pairs (:data:`main.b1.cross.PAIRS`), one
ledger family a card and panel and one request a card, look and panel (:func:`family`, :func:`request`).

**Play** (:func:`play`) is ``drive_look.sh``'s play loop in Python, unchanged in what it runs: in the PIN checkout
(verified at the plan's pin, clean under ``src/`` and ``data/``), each panel's family is registered and its cells are
played by ``main.h2h play-many`` (both sides greedy, ``--seed 0``, ``--oracle-reveal-mode off``, purpose ``ab``,
batches of 500 pairs), on cuda under ``gpu_lock.sh`` + ``mem_cap.sh 32`` (the caller holds the lease), on cpu at nice
19 on the agent cores under ``mem_cap.sh 24``. ``main.sidecar_audit --strict`` and ``main.checkargs`` on every run are
REPORTED in ``run.log``, never a gate, as in B1.

**A lost fresh run** (a registered card's PREREG §4, as B1's §10): the card has ONE spare seed, and the replacement
is a dated amendment that edits the lost row of the card's arm file (its seed and its run name), written with its
cause; there is no switch. The kit holds the amended file to that rule (:data:`REGISTERED_CELLS`,
:func:`replaced_fresh_seed`): a registered card's rows are its registered seeds in order, at most ONE of them the
spare's, each named ``rb_<card>_s<seed>``; any other row set is refused, so a second replacement or another seed
cannot be planned. The spare run is then a fresh run like the others (from step 0, its own registered argv, the
same guards), and the look doc records ``replaced_fresh_seed``. The cause (files lost or unreadable, or a wrong
argv; never a run "lost to its lever") is the amendment's to state: the kit cannot see it.

**Not built** (FINDINGS of this build): the REPORTED cross between an own-control cell and B1's banked cell across
two pins (FRAME §7); and Q5's REPORTED matched-GPU-time cross (its learning-rate and KL report is
``python -m main.b1.lr_report --fresh-arm``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import fresh_rule as FR
from main.b1 import look as K

#: B1's LAUNCH pin (its PREREG §3, the re-pin record of 2026-11-12): every banked run trained at it, so a fresh run
#: reads against a banked cell only at it (FRAME §2 item 1). An older commit, so this tree may name it.
B1_LAUNCH_PIN = "15136b61634b667bb9da5ccc6647ff9265238829"
ARM_SCHEMA = "gen3_b1_fresh_arm_v1"
LOOK_SCHEMA = "gen3_b1_fresh_look_v1"
#: The B1 arms a banked cell may be (FRAME §1: C at width 128, or W at 256).
BANKED_ARMS: Tuple[str, ...] = ("C", "W")
LOOK_STEPS: Dict[str, int] = {"50M": 50_000_000, "100M": 100_000_000}
#: The resolved keys a fresh argv differs in beyond its declared keys: its own seed and name.
OWN_KEYS: Tuple[str, ...] = ("run_name", "seed")
#: Flags no lever may type (beside every flag the banked argv already types): a resume, a torch switch.
FORBIDDEN_FLAGS: Tuple[str, ...] = ("--model", "--allow-torch-switch", "--fork-lr", "--fork-lr-freeze")
#: The keys a run's last process must resolve as its own argv does at the pin, beyond the declared keys.
CHECKED_KEYS: Tuple[str, ...] = tuple(sorted({*K.CONFIG_KEYS, "n_epochs"}))
#: card -> (its REGISTERED fresh seeds in arm-file order, its ONE spare seed): §4 of each registered card's PREREG
#: (``q6n_policy_readout_fresh_2026-11-20``, ``q5n_epochs_fresh_2026-11-20``), held against the committed arm files and
#: PREREGs by ``fresh_test.py``. A card absent here is a draft: its rows are whatever its file says.
REGISTERED_CELLS: Dict[str, Tuple[Tuple[int, ...], int]] = {"q6n": ((5211, 5212, 5213), 5219),
                                                            "q5n": ((5201, 5202, 5203), 5209)}
_CARD = re.compile(r"^[a-z][a-z0-9]{0,23}$")
_RUN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")

Resolver = Callable[[Sequence[Sequence[str]]], List[Dict[str, Any]]]
RunGuard = Callable[[str, Path, "Run", int], Tuple[List[str], Dict[str, Any]]]


class FreshError(ValueError):
    """A malformed fresh-arm file, look, pin checkout or argument."""


@dataclass(frozen=True)
class Run:
    seed: int
    run: str


@dataclass(frozen=True)
class ArmSpec:
    """One card's fresh arm (module docstring). Build it with :func:`arm_from_doc`, which validates it."""

    card: str
    pin: str
    banked_arm: str
    lever: Tuple[str, ...]
    declared_keys: Tuple[str, ...]
    fresh: Tuple[Run, ...]
    own_control: Optional[Tuple[Run, ...]] = None
    prereg: Optional[str] = None
    #: The REGISTERED seed the card's spare replaced (:func:`replaced_fresh_seed`); never written to the arm file.
    replaced_fresh: Optional[int] = None
    sha256: Optional[str] = field(default=None, compare=False)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema": ARM_SCHEMA, "card": self.card, "pin": self.pin, "banked_arm": self.banked_arm,
                               "lever": list(self.lever), "declared_keys": list(self.declared_keys),
                               "fresh": [{"seed": r.seed, "run": r.run} for r in self.fresh]}
        if self.own_control is not None:
            out["own_control"] = [{"seed": r.seed, "run": r.run} for r in self.own_control]
        if self.prereg is not None:
            out["prereg"] = self.prereg
        return out


def _runs(rows: Any, where: str) -> Tuple[Run, ...]:
    if not isinstance(rows, list) or not rows:
        raise FreshError(f"{where}: a non-empty list of {{seed, run}}")
    out: List[Run] = []
    for r in rows:
        if not isinstance(r, dict) or set(r) != {"seed", "run"}:
            raise FreshError(f"{where}: each entry is exactly {{seed, run}}, got {r!r}")
        seed, name = r["seed"], r["run"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            raise FreshError(f"{where}: seed {seed!r} is not a positive integer")
        if not isinstance(name, str) or not _RUN.match(name) or not name.endswith(f"_s{seed}"):
            raise FreshError(f"{where}: run name {name!r} is not a run name ending in _s{seed}")
        out.append(Run(seed, name))
    return tuple(out)


def replaced_fresh_seed(card: str, fresh: Sequence[Run]) -> Optional[int]:
    """The REGISTERED seed a registered card's spare replaced, or ``None`` (its registered rows; a card with no entry
    in :data:`REGISTERED_CELLS`). Any other row set of a registered card is refused (module docstring)."""
    if card not in REGISTERED_CELLS:
        return None
    seeds, spare = REGISTERED_CELLS[card]
    got = tuple(r.seed for r in fresh)
    lost = [s for s, g in zip(seeds, got) if g != s]
    if len(got) != len(seeds) or len(lost) > 1 or any(g not in (s, spare) for s, g in zip(seeds, got)):
        raise FreshError(f"card {card}: fresh seeds {got} are not its registered {seeds} with at most ONE row replaced "
                         f"by its spare {spare} (its PREREG §4: one replacement, in the lost row; a second loss is "
                         "INCONCLUSIVE, never a second replacement)")
    named = [r.run for r in fresh if r.run != f"rb_{card}_s{r.seed}"]
    if named:
        raise FreshError(f"card {card}: {named} are not its registered run names rb_{card}_s<seed>")
    return lost[0] if lost else None


def arm_from_doc(doc: Mapping[str, Any], sha256: Optional[str] = None) -> ArmSpec:
    """Validate a fresh-arm document (module docstring); every refusal is a :class:`FreshError`."""
    known = {"schema", "card", "pin", "banked_arm", "lever", "declared_keys", "fresh", "own_control", "prereg", "note"}
    if not isinstance(doc, Mapping) or doc.get("schema") != ARM_SCHEMA:
        raise FreshError(f"not a {ARM_SCHEMA} document")
    extra = sorted(set(doc) - known)
    if extra:
        raise FreshError(f"unknown fields {extra}")
    card, pin, arm = doc.get("card"), doc.get("pin"), doc.get("banked_arm")
    if not isinstance(card, str) or not _CARD.match(card):
        raise FreshError(f"card {card!r}: lower-case letters and digits, a letter first (it names ledger families)")
    if not isinstance(pin, str) or not _SHA.match(pin):
        raise FreshError(f"pin {pin!r} is not a full 40-hex commit")
    if arm not in BANKED_ARMS:
        raise FreshError(f"banked_arm {arm!r} is not one of {BANKED_ARMS}")
    lever = doc.get("lever")
    if not isinstance(lever, list) or not lever or not all(isinstance(t, str) and t for t in lever):
        raise FreshError("lever: a non-empty list of argv tokens")
    typed = {t for t in A.build_argv(K.KIT_ARM[arm], A.SEEDS[K.KIT_ARM[arm]][0], pin) if t.startswith("--")}
    clash = sorted({t.split("=", 1)[0] for t in lever if t.startswith("--")} & (typed | set(FORBIDDEN_FLAGS)))
    if clash:
        raise FreshError(f"lever types {clash}: the banked argv already types them, or no lever may")
    keys = doc.get("declared_keys")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys) \
            or len(set(keys)) != len(keys) or set(keys) & set(OWN_KEYS):
        raise FreshError(f"declared_keys: a non-empty list of distinct resolved keys, none of {OWN_KEYS}")
    fresh = _runs(doc.get("fresh"), "fresh")
    own = _runs(doc["own_control"], "own_control") if doc.get("own_control") is not None else None
    if own is None and pin != B1_LAUNCH_PIN:
        raise FreshError(f"pin {pin[:12]} is not B1's launch pin {B1_LAUNCH_PIN[:12]}: a banked B1 cell is a "
                         "control only at its own pin (FRAME §2 item 1); another pin buys its own control "
                         "(own_control, FRAME §7)")
    n_banked = len(own) if own is not None else len(A.SEEDS[K.KIT_ARM[arm]])
    if len(fresh) != n_banked:
        raise FreshError(f"{len(fresh)} fresh runs against a banked cell of {n_banked}")
    mine = [*fresh, *(own or ())]
    b1_seeds = set(A.all_seeds())
    b1_names = set(A.all_run_names())
    if len({r.seed for r in mine}) != len(mine) or len({r.run for r in mine}) != len(mine):
        raise FreshError("a seed or a run name appears twice")
    taken = sorted({r.seed for r in mine} & b1_seeds) + sorted({r.run for r in mine} & b1_names)
    if taken:
        raise FreshError(f"{taken} are BOUNDARY-1's own seeds or run names")
    prereg = doc.get("prereg")
    if prereg is not None and not isinstance(prereg, str):
        raise FreshError("prereg: a path")
    return ArmSpec(card=card, pin=pin, banked_arm=arm, lever=tuple(lever), declared_keys=tuple(sorted(keys)),
                   fresh=fresh, own_control=own, prereg=prereg, replaced_fresh=replaced_fresh_seed(card, fresh),
                   sha256=sha256)


def load_arm(path: "str | Path") -> ArmSpec:
    from utils.hashing import sha256_file

    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise FreshError(f"{path}: cannot be read ({type(e).__name__}: {e})") from e
    return arm_from_doc(doc, sha256_file(str(path)))


# ------------------------------------------------------------------------------------------------- the argvs
def _set(argv: List[str], flag: str, value: str) -> List[str]:
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


def control_argv(spec: ArmSpec, steps: int, run: Optional[Run] = None) -> List[str]:
    """The banked arm's B1 argv at the arm's pin (its first seed's), or with ``run``'s seed and name."""
    kit = K.KIT_ARM[spec.banked_arm]
    argv = A.build_argv(kit, A.SEEDS[kit][0], spec.pin, steps)
    if run is None:
        return argv
    return _set(_set(argv, "--seed", str(run.seed)), "--run-name", run.run)


def fresh_argv(spec: ArmSpec, run: Run, steps: int) -> List[str]:
    """FRAME §2 item 2: the control's argv with the lever inserted before ``--checkpoint-every-steps``."""
    argv = control_argv(spec, steps, run)
    k = argv.index("--checkpoint-every-steps")
    return argv[:k] + list(spec.lever) + argv[k:]


def resume_argv(spec: ArmSpec, run: Run) -> List[str]:
    """Phase 2, B1's resume form (FRAME §2 item 3): ``--model <run>/final_model.zip --steps 100000000``."""
    return ["--restart-interval-hours", A.RESTART_HOURS, "--pin-commit", spec.pin,
            "--model", f"models/{run.run}/final_model.zip", "--steps", str(A.PHASE_STEPS[1]),
            "--device", "cuda", "--log-level", "periodic"]


def banked_runs(spec: ArmSpec, replace: Optional[int] = None) -> Tuple[Run, ...]:
    """The banked cell: B1's arm (``replace``: the REGISTERED seed B1's spare replaced) or the own control."""
    if spec.own_control is not None:
        if replace is not None:
            raise FreshError("--replace names a B1 seed; an own-control cell has none")
        return spec.own_control
    kit = K.KIT_ARM[spec.banked_arm]
    return tuple(Run(s, A.run_name(kit, s)) for s in K.design_seeds(replace, (spec.banked_arm,))[spec.banked_arm])


def find_run(spec: ArmSpec, name: str) -> Tuple[str, Run]:
    for side, runs in (("fresh", spec.fresh), ("banked", spec.own_control or ())):
        for r in runs:
            if r.run == name:
                return side, r
    raise FreshError(f"{name!r} is not a fresh or own-control run of card {spec.card}")


# --------------------------------------------------------------------------------------- the pin's resolver
def tree_problems(tree: Path, pin: str) -> List[str]:
    """``[]`` iff ``tree`` is a checkout at ``pin`` with clean tracked ``src/`` and ``data/``."""
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if head.returncode != 0:
        return [f"{tree} is not a git checkout"]
    if head.stdout.strip() != pin:
        return [f"{tree} is at {head.stdout.strip()[:12]}, not the pin {pin[:12]}"]
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no", "--", "src", "data"],
                           capture_output=True, text=True, check=False)
    if dirty.returncode != 0 or dirty.stdout.strip():
        return [f"{tree}: src/ or data/ is dirty (or unreadable)"]
    return []


_RESOLVE = """
import json, sys
from main.b1.arms import resolved_namespace
argvs = json.load(open(sys.argv[1]))
json.dump([json.loads(json.dumps(resolved_namespace(a), default=str)) for a in argvs], open(sys.argv[2], "w"))
"""


def pin_tree_resolver(tree: "str | Path", pin: str, python: Optional[str] = None) -> Resolver:
    """A :data:`Resolver` that runs THE PIN's parser and ``resolve_config`` (its ``main.b1.arms.resolved_namespace``) in
    the checkout ``tree``, CUDA hidden, in one subprocess a call. Refuses a tree that is not clean at ``pin``."""
    root = Path(tree).resolve()
    bad = tree_problems(root, pin)
    if bad:
        raise FreshError("; ".join(bad))
    py = python or os.environ.get("GEN3AI_PYTHON") or sys.executable

    def resolve(argvs: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "argvs.json", Path(d) / "ns.json"
            src.write_text(json.dumps([list(a) for a in argvs]))
            env = dict(os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="")
            r = subprocess.run([py, "-c", _RESOLVE, str(src), str(dst)], cwd=str(root), env=env, capture_output=True,
                               text=True, check=False)
            if r.returncode != 0:
                raise FreshError(f"the pin's resolver failed in {root}: {r.stderr.strip()[-800:]}")
            return list(json.loads(dst.read_text()))

    return resolve


def in_process_resolver(argvs: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    """THIS checkout's parser (tests, and nothing a plan reads: a plan resolves at the pin)."""
    return [json.loads(json.dumps(A.resolved_namespace(list(a)), default=str)) for a in argvs]


def _json(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def lever_check(spec: ArmSpec, resolver: Resolver) -> Dict[str, Any]:
    """FRAME §2 item 2 (module docstring). ``expected`` holds, per fresh or own-control run, the values its last
    process must record on the declared keys and :data:`CHECKED_KEYS`; ``banked_levers`` B1's lever keys for the
    banked arm's runs (:func:`main.b1.look.check_run`'s ``levers``)."""
    probs: List[str] = []
    targets: Dict[str, Any] = {}
    expected: Dict[str, Dict[str, Any]] = {}
    banked_levers: Dict[str, Any] = {}
    controls = spec.own_control or ()
    for steps in A.PHASE_STEPS:
        c = control_argv(spec, steps)
        argvs = [c] + [fresh_argv(spec, r, steps) for r in spec.fresh] + [control_argv(spec, steps, r) for r in controls]
        ns = resolver(argvs)
        if len(ns) != len(argvs):
            raise FreshError(f"the resolver returned {len(ns)} namespaces for {len(argvs)} argvs")
        nc, rest = ns[0], ns[1:]
        rows: Dict[str, Any] = {}
        for r, argv, na in zip([*spec.fresh, *controls], argvs[1:], rest):
            diff = sorted(k for k in set(nc) | set(na) if _json(nc.get(k)) != _json(na.get(k)))
            lever = sorted(set(diff) - set(OWN_KEYS))
            want = [] if r in controls else sorted(spec.declared_keys)
            if lever != want:
                probs.append(f"{r.run} @ {steps:,}: its resolved config differs from the control's in {lever} "
                             f"beyond its seed and name, not {want}")
            if (na.get("seed"), na.get("run_name")) != (r.seed, r.run):
                probs.append(f"{r.run} @ {steps:,}: it resolves seed {na.get('seed')!r} and run_name "
                             f"{na.get('run_name')!r}, not its own")
            rows[r.run] = {"argv": shlex.join(argv), "diff": {k: [nc.get(k), na.get(k)] for k in diff}}
            if steps == A.PHASE_STEPS[0]:
                expected[r.run] = {k: _json(na.get(k)) for k in sorted({*CHECKED_KEYS, *spec.declared_keys})}
        if steps == A.PHASE_STEPS[0]:
            banked_levers = {k: _json(nc.get(k)) for k in K.CONFIG_KEYS}
        targets[str(steps)] = {"control_argv": shlex.join(c), "runs": rows}
    return {"ok": not probs, "problems": probs, "declared_keys": list(spec.declared_keys), "targets": targets,
            "expected": expected, "banked_levers": banked_levers}


# ------------------------------------------------------------------------------------------------ the guards
def check_cell_run(run_dir: Path, run: Run, *, pin: str, step: int, argvs: Sequence[Sequence[str]],
                   expected: Mapping[str, Any], k9b: Optional[K.K9bReader] = None) -> Tuple[List[str], Dict[str, Any]]:
    """``(problems, facts)`` for one fresh or own-control run at one look step (module docstring, "A fresh run
    counts"): :func:`main.b1.look.check_run`'s guards with the run's own registered argvs and expected values."""
    probs: List[str] = []
    facts: Dict[str, Any] = {"seed": run.seed, "run": run_dir.name}
    try:
        md = json.loads((run_dir / "metadata.json").read_text())
    except (OSError, ValueError) as e:
        return [f"metadata.json cannot be read ({type(e).__name__})"], facts
    zip_path, n, why = K.find_checkpoint(run_dir, step)
    if zip_path is None or n is None:
        return [why or "no checkpoint"], facts
    try:
        side = json.loads(zip_path.with_suffix(".json").read_text())
    except (OSError, ValueError) as e:
        return [f"{zip_path.name}'s sidecar cannot be read ({type(e).__name__})"], facts
    if side.get("num_timesteps") != n:
        probs.append(f"{zip_path.name}'s sidecar says num_timesteps {side.get('num_timesteps')!r}")
    probs.extend(K.pin_problems(md.get("pin_history"), side, zip_path.name, pin, n))
    lin = md.get("lineage") or {}
    if lin.get("role") != "fresh" or lin.get("fork_parent") is not None:
        probs.append(f"lineage is not a FRESH run (role {lin.get('role')!r})")
    try:
        got = shlex.split(str(md.get("original_command") or ""))[1:]
    except ValueError:
        got = None
        probs.append("original_command cannot be parsed")
    if got is not None and not any(got == list(a) for a in argvs):
        probs.append(f"original_command is not the registered argv of {run.run}: {' '.join(got)!r}")
    cli = md.get("cli_args") or {}
    for where, v in (("metadata.ent_coef", md.get("ent_coef")), ("cli_args.ent_coef", cli.get("ent_coef")),
                     (f"{zip_path.name} ent_coef", side.get("ent_coef"))):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or abs(float(v) - K.ENT_COEF) > 1e-12:
            probs.append(f"{where} {v!r} != {K.ENT_COEF}")
    for k, want in sorted(expected.items()):
        if _json(cli.get(k)) != want:
            probs.append(f"cli_args.{k} {cli.get(k)!r} is not its argv's {want!r} at the pin")
    rec = dict((k9b or K.k9b_read)(run_dir, n))
    bad = K.k9b_problem(rec)
    if bad:
        probs.append(bad)
    facts.update({"checkpoint": str(zip_path), "steps": n, "k9b": rec, "torch_version": md.get("torch_version"),
                  "init_num_threads": md.get("init_num_threads"),
                  "_cli": {k: v for k, v in cli.items() if not str(k).startswith("_")},
                  "_true": {k: md.get(k) for k in K.TRUE_KEYS}})
    return probs, facts


def default_guard(spec: ArmSpec, lc: Mapping[str, Any], k9b: Optional[K.K9bReader] = None) -> RunGuard:
    """The registered per-run guard: a banked B1 run by :func:`main.b1.look.check_run`, any other by
    :func:`check_cell_run` against its own argv."""

    def guard(side: str, run_dir: Path, run: Run, step: int) -> Tuple[List[str], Dict[str, Any]]:
        if side == FR.BANKED and spec.own_control is None:
            return K.check_run(run_dir, spec.banked_arm, run.seed, spec.pin, step, lc["banked_levers"], k9b)
        if side == FR.FRESH:
            argvs = [fresh_argv(spec, run, s) for s in A.PHASE_STEPS]
        else:
            argvs = [control_argv(spec, s, run) for s in A.PHASE_STEPS]
        return check_cell_run(run_dir, run, pin=spec.pin, step=step, argvs=argvs, expected=lc["expected"][run.run],
                              k9b=k9b)

    return guard


def cross_problems(spec: ArmSpec, facts: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    """The cross-run guards (module docstring, "Across the six runs"). Pops each run's private ``_cli`` / ``_true``."""
    probs: List[str] = []
    every = [f for s in FR.SIDES for f in facts[s] if "checkpoint" in f]
    for what in ("torch_version", "init_num_threads"):
        seen = sorted({str(f.get(what)) for f in every})
        if len(seen) > 1:
            probs.append(f"{what} differs across the runs: {seen}")
    clis = [(s, f["run"], f.pop("_cli"), f.pop("_true")) for s in FR.SIDES for f in facts[s] if "_cli" in f]
    control = next((c for c in clis if c[0] == FR.BANKED), None)
    if control is None:
        return probs
    for side, run, cli, true in clis:
        if run == control[1]:
            continue
        allowed = set(K.RUN_OWN_KEYS) | (set(spec.declared_keys) if side == FR.FRESH else set())
        if K.process_kind(cli) != K.process_kind(control[2]):
            allowed |= set(K.RESUME_REWRITES)
        drift = sorted(k for k in set(cli) | set(control[2]) if k not in allowed
                       and _json(cli.get(k)) != _json(control[2].get(k)))
        if drift:
            probs.append(f"{run}: its last process differs from {control[1]}'s in {drift} (beyond its own keys"
                         f"{' and the declared lever' if side == FR.FRESH else ''})")
        moved = sorted(k for k in K.TRUE_KEYS if _json(true.get(k)) != _json(control[3].get(k))
                       and not (side == FR.FRESH and k in spec.declared_keys))
        if moved:
            probs.append(f"{run}: metadata.json's {moved} differ from {control[1]}'s")
    return probs


def b1_look_problems(b1_look: Mapping[str, Any], spec: ArmSpec, look: str,
                     banked: Sequence[Mapping[str, Any]]) -> List[str]:
    """FRAME §2 item 6: each banked run's look checkpoint is the file B1's own look read (same step, same sha256)."""
    probs: List[str] = []
    if b1_look.get("look") != look or b1_look.get("pin") != spec.pin or b1_look.get("survivor"):
        return [f"the B1 look given is the {b1_look.get('look')!r} look at {str(b1_look.get('pin'))[:12]} "
                f"(survivor {b1_look.get('survivor')!r}), not B1's full {look} look at {spec.pin[:12]}"]
    theirs = {f.get("run"): f for f in (b1_look.get("runs") or {}).get(spec.banked_arm) or ()}
    for f in banked:
        t = theirs.get(f["run"])
        if t is None:
            probs.append(f"{f['run']}: not in B1's {look} look")
        elif (t.get("steps"), t.get("sha256")) != (f.get("steps"), f.get("sha256")):
            probs.append(f"{f['run']}: its {look} file is step {f.get('steps')} sha {str(f.get('sha256'))[:12]}, B1 "
                         f"read step {t.get('steps')} sha {str(t.get('sha256'))[:12]}")
    return probs


# ---------------------------------------------------------------------------------------------- names, plan
def family(card: str, panel: str) -> str:
    K.family(panel)  # validates the panel
    return f"b1_fresh_{card}_h2h_{panel}"


def request(card: str, look: str, panel: str) -> str:
    if look not in LOOK_STEPS:
        raise FreshError(f"look {look!r} is not one of {tuple(LOOK_STEPS)}")
    return f"b1fresh_{card}_{look.lower()}_{panel}"


def family_rule(card: str, panel: str, pin: str) -> str:
    return (f"fresh_cell_frame_2026-11-17/FRAME.md §3 (card {card}, pin {pin[:8]}): the fresh cell against the banked "
            f"cell on panel {panel}, {X.PAIRS[panel]} mirrored pairs a cell, one request a look; the rule is "
            f"main.reads' b1_fresh_cell (main.b1.fresh_rule: main.b1.cross.simple_effect on N, within-side "
            f"variances floored at {X.FLOOR_PP} pp, t on 2(n-1) df, {int(round(100 * X.LEVEL))} % interval)")


def names(spec: ArmSpec, look: str) -> Dict[str, Dict[str, Any]]:
    return {p: {"family": family(spec.card, p), "request": request(spec.card, look, p), "pairs": X.PAIRS[p],
                "rule": family_rule(spec.card, p, spec.pin)} for p in X.PANELS}


def plan(models: Path, spec: ArmSpec, look: str, resolver: Resolver, *, b1_look: Optional[Mapping[str, Any]] = None,
         replace: Optional[int] = None, k9b: Optional[K.K9bReader] = None, guard: Optional[RunGuard] = None,
         cross: Optional[Callable[[ArmSpec, Mapping[str, Sequence[Mapping[str, Any]]]], List[str]]] = None,
         step: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """``(look doc, [])`` or ``(None, problems)`` (module docstring). TESTS ONLY, never the CLI: ``guard`` replaces the
    per-run guard, ``cross`` the cross-run guards, ``step`` the look's step and ``k9b`` the K9(b) reader; the
    rehearsal uses them to pose B1 runs as a fresh cell, which the registered guards refuse."""
    from utils.hashing import sha256_file

    if look not in LOOK_STEPS:
        raise FreshError(f"look {look!r} is not one of {tuple(LOOK_STEPS)}")
    at = LOOK_STEPS[look] if step is None else step
    lc = lever_check(spec, resolver)
    if not lc["ok"]:
        return None, [f"lever check: {p}" for p in lc["problems"]]
    check = guard or default_guard(spec, lc, k9b)
    sides = {FR.FRESH: spec.fresh, FR.BANKED: banked_runs(spec, replace)}
    probs: List[str] = []
    facts: Dict[str, List[Dict[str, Any]]] = {s: [] for s in FR.SIDES}
    for side, runs in sides.items():
        for r in runs:
            run_dir = models / r.run
            if not run_dir.is_dir():
                probs.append(f"{r.run}: no run folder under {models}")
                continue
            p, f = check(side, run_dir, r, at)
            probs.extend(f"{r.run}: {x}" for x in p)
            facts[side].append(dict(f, run=r.run, seed=r.seed))
    probs.extend((cross or cross_problems)(spec, facts))
    if probs:
        return None, probs
    for fs in facts.values():
        for f in fs:
            f["sha256"] = sha256_file(f["checkpoint"])
    shas = [f["sha256"] for fs in facts.values() for f in fs]
    if len(set(shas)) != len(shas):
        return None, ["two runs' checkpoints have the same sha256"]
    identity: Dict[str, Any] = {"checked": False, "why": "own control: no B1 look reads these runs"}
    if spec.own_control is None:
        if b1_look is None:
            return None, ["no B1 look.json given (--b1-look): the banked cell's files cannot be shown to be the bytes "
                          "B1 read (FRAME §2 item 6)"]
        bad = b1_look_problems(b1_look, spec, look, facts[FR.BANKED])
        if bad:
            return None, bad
        identity = {"checked": True, "b1_look": b1_look.get("look"), "pin": b1_look.get("pin")}
    design = FR.design_of({s: [f["run"] for f in facts[s]] for s in FR.SIDES})
    zip_of = {f["run"]: f["checkpoint"] for fs in facts.values() for f in fs}
    return {"schema": LOOK_SCHEMA, "look": look, "step": at, "card": spec.card, "pin": spec.pin,
            "arm": spec.as_dict(), "arm_sha256": spec.sha256, "banked_arm": spec.banked_arm,
            "own_control": spec.own_control is not None, "replaced_seed": replace,
            "replaced_fresh_seed": spec.replaced_fresh, "lever_check": lc,
            "banked_identity": identity, "runs": facts,
            "cells": [[zip_of[p], zip_of[o]] for p, o in design.cells()], "names": names(spec, look)}, []


# ----------------------------------------------------------------------------------------------------- play
PlayRunner = Callable[[Sequence[str], Mapping[str, str], Path, Path], int]


def _subprocess_runner(argv: Sequence[str], env: Mapping[str, str], cwd: Path, log: Path) -> int:
    with log.open("a") as fh:
        return subprocess.run(list(argv), cwd=str(cwd), env=dict(env), stdout=fh, stderr=subprocess.STDOUT,
                              check=False).returncode


def _log(log: Path, line: str) -> None:
    from datetime import datetime

    msg = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {line}"
    print(msg, file=sys.stderr)
    with log.open("a") as fh:
        fh.write(msg + "\n")


def play_commands(doc: Mapping[str, Any], work: Path, pin_dir: Path, device: str, *, py: str,
                  ledger: Optional[Path] = None, max_s: Optional[int] = None, agent_cpus: Optional[str] = None,
                  pairs: Optional[Mapping[str, int]] = None) -> List[Tuple[str, str, List[str]]]:
    """``[(panel, what, argv)]``: each panel's family-register, then its play-many (``drive_look.sh``'s, verbatim in
    what it plays). ``pairs`` (tests only) plays fewer pairs a cell than the registration."""
    if device not in ("cpu", "cuda"):
        raise FreshError(f"device {device!r} is not cpu or cuda")
    from utils.paths import repo_path

    ops = repo_path("scripts/ops")
    out: List[Tuple[str, str, List[str]]] = []
    for panel in X.PANELS:
        nm = doc["names"][panel]
        n = int((pairs or {}).get(panel, nm["pairs"]))
        root = ["--root", str(ledger)] if ledger else []
        out.append((panel, "family-register", [py, "-m", "main.eval_ledger", "family-register", *root, "--id",
                                               nm["family"], "--decision-kind", "ab_verdict", "--rule", nm["rule"],
                                               "--protocol", K.PROTOCOL, "--commit", str(doc["pin"])]))
        game = [py, "-m", "main.h2h", "play-many", "--cells", str(work / "cells_plan.json"), "--pairs", str(n),
                "--batch-pairs", str(min(500, n)), "--panel", panel, "--seed", "0", "--device", device,
                "--purpose", "ab", "--family", nm["family"], "--request", nm["request"], "--label", nm["request"],
                "--oracle-reveal-mode", "off", *(["--out", str(ledger)] if ledger else [])]
        cap = [str(ops / "mem_cap.sh"), *(["--max-s", str(max_s)] if max_s else [])]
        if device == "cuda":
            argv = [str(ops / "gpu_lock.sh"), *cap, "32", *game]
        else:
            if not agent_cpus:
                raise FreshError("cpu play needs the agent cores (utils.core_split agent-cpus)")
            nt = str(max(1, len(_cpus(agent_cpus)) // 2))  # one thread a physical core
            argv = ["nice", "-n", "19", "taskset", "-c", agent_cpus, *cap, "24", *game, "--threads", nt,
                    "--torch-threads", nt]
        out.append((panel, "play-many", argv))
    return out


def _cpus(text: str) -> List[int]:
    from utils.core_split import parse_cpus

    return parse_cpus(text)


def play(work: Path, pin_dir: Path, device: str, *, ledger: Optional[Path] = None, max_s: Optional[int] = None,
         runner: PlayRunner = _subprocess_runner, pairs: Optional[Mapping[str, int]] = None) -> int:
    """Play a planned look (module docstring, "Play"). Exit 0 played; 4 a panel's family or play failed; 5 the pin
    checkout is missing, not at the plan's pin, or dirty. ``runner`` and ``pairs`` are for tests."""
    work = work.resolve()
    log = work / "run.log"
    doc = json.loads((work / "look.json").read_text())
    pin_dir = pin_dir.resolve()
    bad = tree_problems(pin_dir, str(doc["pin"])) if pin_dir.is_dir() else [f"no pin checkout at {pin_dir}"]
    if bad:
        _log(log, "HOLD: " + "; ".join(bad))
        return 5
    py = os.environ.get("GEN3AI_PYTHON") or sys.executable
    env = dict(os.environ, PYTHONPATH=str(pin_dir / "src"), GEN3AI_SHARED_COMPILE_CACHE="0")
    agent = None
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        from utils.core_split import agent_cpus

        agent = agent_cpus()
    cmds = play_commands(doc, work, pin_dir, device, py=py, ledger=ledger, max_s=max_s, agent_cpus=agent, pairs=pairs)
    runs = sorted({str(Path(f["checkpoint"]).parent.parent) for s in FR.SIDES for f in doc["runs"][s]})
    _log(log, f"card {doc['card']} look {doc['look']}: {len(doc['cells'])} cells, play at pin {doc['pin']} in "
              f"{pin_dir} on {device}")
    rc = runner([py, "-m", "main.sidecar_audit", "--strict", *runs], env, pin_dir, log)
    _log(log, f"REPORTED: main.sidecar_audit --strict exit {rc}")
    for r in runs:
        rc = runner([py, "-m", "main.checkargs", r], env, pin_dir, log)
        _log(log, f"REPORTED: main.checkargs {Path(r).name} exit {rc}")
    for panel, what, argv in cmds:
        _log(log, f"{panel}: {what}")
        if runner(argv, env, pin_dir, log) != 0:
            _log(log, f"HOLD: {what} failed on {panel} (see {log})")
            return 4
    _log(log, "played")
    return 0


# ----------------------------------------------------------------------------------------------------- read
#: The look's read: one panel's A/B family at the h2h protocol, greedy on both sides, mirrored, across runs (the terms of
#: B1's :data:`main.b1.look.LOOK_READ`, declared here because the reader gate wants each reader's own declaration).
FRESH_READ = L.ReaderDecl(
    name="main.b1.fresh.family", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol=K.PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


def ledger_rows(root: Path, doc: Mapping[str, Any]) -> Tuple[Dict[str, List[Any]], Dict[str, List[Mapping[str, Any]]]]:
    """``(each panel's cells, each panel's rows)`` of the look from the eval COUNT ledger, by :data:`FRESH_READ` (the
    terms of B1's :data:`main.b1.look.LOOK_READ`, under this reader's own name) and the plan's own request ids."""

    cells: Dict[str, List[Any]] = {}
    rows: Dict[str, List[Mapping[str, Any]]] = {}
    for panel in X.PANELS:
        req = str(doc["names"][panel]["request"])
        try:
            got = L.read(FRESH_READ, root=root, family=str(doc["names"][panel]["family"]))
            cells[panel] = list(dict(L.looks(got)).get(req, []))
            rows[panel] = [r for r in got.rows if (r.get("request") or {}).get("id") == req]
        except (ValueError, OSError) as e:
            print(f"NOTE: {panel}: no cells read ({type(e).__name__}: {e})", file=sys.stderr)
            cells[panel], rows[panel] = [], []
    return cells, rows


def read_ledger(root: Path, doc: Mapping[str, Any], *, pairs: Mapping[str, int] = X.PAIRS
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(inputs, verdict)`` of a played look from the ledger at ``root`` (``pairs``: tests only)."""
    from main.h2h import panel as PN

    cells, rows = ledger_rows(root, doc)
    reasons = K.row_problems(rows, str(doc["pin"]), {p: PN.panel_team_set_id(p) for p in X.PANELS})
    return FR.read(doc, cells, reasons, pairs=pairs)


def recheck(work: Path) -> Tuple[bool, Dict[str, Any]]:
    """``(same, re-decided verdict)``: ``decide`` run again on ``result/inputs.json`` alone."""
    doc = json.loads((work / "result" / "inputs.json").read_text())
    got = FR.decide(doc, {"look": doc["look"]})
    had = json.loads((work / "result" / "verdict.json").read_text())
    return _json(got) == had, got


# ------------------------------------------------------------------------------------------ the team bootstrap
def bootstrap(doc: Mapping[str, Any], cells_by_panel: Mapping[str, Sequence[Any]],
              rows_by_panel: Mapping[str, Sequence[Mapping[str, Any]]], *, draws: int = 10_000, seed: int = 20261110,
              row_reasons: Sequence[str] = (), pairs: Mapping[str, int] = X.PAIRS) -> Dict[str, Any]:
    """REPORTED, never deciding: B1's team-cluster bootstrap (:mod:`main.b1.team_bootstrap`, the same tallies, draws
    and combined interval) applied to the fresh-cell statistic on every view."""
    import math

    import numpy as np

    from main.b1 import team_bootstrap as TB

    if draws < 100:
        raise FreshError(f"--draws {draws}: at least 100")
    _inputs, verdict = FR.read(doc, cells_by_panel, row_reasons, pairs=pairs)
    head = {"what": "team-cluster bootstrap of the fresh-cell statistic", "status": TB.REPORTED, "decides": None,
            "card": doc["card"], "look": doc["look"], "pin": doc["pin"], "draws": draws, "seed": seed}
    if verdict["decision"]["outcome"] == FR.INCONCLUSIVE:
        return {**head, "outcome": FR.INCONCLUSIVE, "reasons": list(verdict["decision"]["reasons"])}
    design = FR.design_of({s: [f["run"] for f in doc["runs"][s]] for s in FR.SIDES})
    run_of = {f["sha256"]: f["run"] for s in FR.SIDES for f in doc["runs"][s]}
    cells = design.cells()
    flat = {c: 50.0 for c in cells}
    base = FR.stat(design, flat).estimate
    weights = {"effect": {c: FR.stat(design, {**flat, c: 51.0}).estimate - base for c in cells}}
    rng = np.random.default_rng(seed)
    by_panel: Dict[str, Dict[X.Cell, float]] = {}
    star: Dict[str, Dict[str, Any]] = {}
    n_teams: Dict[str, int] = {}
    for panel in X.PANELS:
        tallies = TB.cell_tallies(rows_by_panel.get(panel, ()), run_of)
        TB.check_tallies(panel, tallies, cells_by_panel.get(panel, ()), run_of, design)
        by_panel[panel] = {c: TB.tally_score(t) for c, t in tallies.items()}
        star[panel], n_teams[panel] = TB.resample(tallies, weights, draws, rng)
    lo_q, hi_q = 50.0 * (1.0 - X.LEVEL), 100.0 - 50.0 * (1.0 - X.LEVEL)
    views: Dict[str, Any] = {}
    for view, pw in X.VIEWS.items():
        st = FR.stat(design, X.blend(design, by_panel, view), view)
        point = math.fsum(w * math.fsum(weights["effect"][c] * (by_panel[p][c] - 50.0) for c in cells)
                          for p, w in pw.items())
        if abs(point - st.estimate) > TB.TOL_PP:
            raise FreshError(f"on {view}: the tallies give {point!r} pp, the registered statistic {st.estimate!r} pp")
        d = sum(w * star[p]["effect"] for p, w in pw.items())
        tv = float(np.var(d, ddof=1))
        comb = TB._combined(st, tv)
        views[view] = {"estimate_pp": st.estimate,
                       "registered": {"se_pp": st.se, "df": st.df, "interval_pp": [st.lo, st.hi],
                                      "class": FR.classify(doc["look"], st)["outcome"]},
                       "team": {"se_pp": math.sqrt(tv), "bias_pp": float(np.mean(d)) - st.estimate,
                                "percentile_interval_pp": [float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))]},
                       "combined": {"se_pp": comb.se, "interval_pp": [comb.lo, comb.hi],
                                    "class": FR.classify(doc["look"], comb)["outcome"],
                                    "team_share_of_variance": tv / comb.v_hat}}
    v = views[FR.DECIDING_VIEW]
    return {**head, "n_per_side": design.n, "teams": n_teams, "read_on": FR.DECIDING_VIEW,
            "outcome_registered": verdict["decision"]["outcome"], "class_with_team_variance": v["combined"]["class"],
            "class_moves": v["registered"]["class"] != v["combined"]["class"], "views": views,
            "notes": ["REPORTED: no class is read from this file; the look's verdict is result/verdict.json",
                      "the combined interval adds the team variance at the registered df: an upper bound (both terms "
                      "hold the game noise)", "one-way clusters (the winning pilot's team); the runs are fixed"]}


# ------------------------------------------------------------------------------------------------------ CLI
def _models() -> Path:
    from utils.paths import main_models_dir

    m = main_models_dir()
    if m is None:
        raise FreshError("no models/ archive here (set GEN3AI_MODELS_DIR)")
    return m


def _ledger_root(a: argparse.Namespace) -> Path:
    return Path(a.ledger) if a.ledger else _models() / "_ledger"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.fresh", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="the fresh (and own-control) argvs at both targets, and the resumes")
    lv = sub.add_parser("lever-check", help="FRAME §2 item 2 at the pin (exit 3 = FAIL)")
    c = sub.add_parser("check-run", help="ONE fresh or own-control run at its first checkpoint (exit 3 = a problem)")
    p = sub.add_parser("plan", help="check the six runs and write the look's cells (exit 3 = REFUSED)")
    for q in (d, lv, c, p):
        q.add_argument("--arm", required=True, help="the card's fresh-arm file (schema gen3_b1_fresh_arm_v1)")
    for q in (lv, c, p):
        q.add_argument("--pin-tree", required=True, help="a clean detached checkout of the arm's pin (its parser)")
    c.add_argument("--run", required=True)
    c.add_argument("--step", type=int, default=A.CHECKPOINT_EVERY)
    p.add_argument("--look", required=True, choices=tuple(LOOK_STEPS))
    p.add_argument("--out", required=True, help="the look's work directory")
    p.add_argument("--b1-look", help="B1's look.json at the same step (FRAME §2 item 6; required for a banked B1 cell)")
    p.add_argument("--replace", type=int, help="the REGISTERED B1 seed B1's spare replaced (B1 PREREG §10)")
    y = sub.add_parser("play", help="play a planned look's four panels in the pin checkout")
    y.add_argument("--work", required=True)
    y.add_argument("--pin-dir", required=True, help="a clean checkout of the plan's pin, Rust built")
    y.add_argument("--device", required=True, choices=("cpu", "cuda"))
    y.add_argument("--max-s", type=int, help="mem_cap's short lane for each panel (a job of <= 10 min)")
    for name in ("read", "recheck", "bootstrap"):
        q = sub.add_parser(name)
        q.add_argument("--work", required=True)
        if name != "recheck":
            q.add_argument("--ledger", help="the ledger root (default: <models>/_ledger)")
    y.add_argument("--ledger", help="the ledger root (default: <models>/_ledger)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "describe":
            spec = load_arm(a.arm)
            if spec.replaced_fresh is not None:
                print(f"# REPLACEMENT: the spare {REGISTERED_CELLS[spec.card][1]} stands in the row of registered "
                      f"seed {spec.replaced_fresh} (PREREG §4: the dated amendment and its cause come first)",
                      file=sys.stderr)
            for r in spec.fresh:
                for steps in A.PHASE_STEPS:
                    print(f"{A.LAUNCHER} {shlex.join(fresh_argv(spec, r, steps))}")
                print(f"{A.LAUNCHER} {shlex.join(resume_argv(spec, r))}   # phase 2")
            for r in spec.own_control or ():
                print(f"{A.LAUNCHER} {shlex.join(control_argv(spec, A.PHASE_STEPS[0], r))}   # own control")
            return 0
        if a.cmd in ("lever-check", "check-run", "plan"):
            spec = load_arm(a.arm)
            resolver = pin_tree_resolver(a.pin_tree, spec.pin)
            if a.cmd == "lever-check":
                got = lever_check(spec, resolver)
                print(json.dumps(got, indent=1, default=str))
                return 0 if got["ok"] else 3
            if a.cmd == "check-run":
                side, run = find_run(spec, a.run)
                lc = lever_check(spec, resolver)
                if not lc["ok"]:
                    print("REFUSED: lever check: " + "; ".join(lc["problems"]), file=sys.stderr)
                    return 3
                probs, facts = default_guard(spec, lc)(side, _models() / run.run, run, a.step)
                facts.pop("_cli", None)
                facts.pop("_true", None)
                print(json.dumps({"run": run.run, "side": side, "ok": not probs, "problems": probs, **facts},
                                 indent=1, default=str))
                return 0 if not probs else 3
            b1 = json.loads(Path(a.b1_look).read_text()) if a.b1_look else None
            doc, probs = plan(_models(), spec, a.look, resolver, b1_look=b1, replace=a.replace)
            if doc is None:
                print("REFUSED: " + "; ".join(probs), file=sys.stderr)
                return 3
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "cells_plan.json").write_text(json.dumps(doc["cells"], indent=1))
            (out / "look.json").write_text(json.dumps(doc, indent=1, default=str))
            print(len(doc["cells"]))
            return 0
        work = Path(a.work)
        if a.cmd == "play":
            return play(work, Path(a.pin_dir), a.device, ledger=Path(a.ledger) if a.ledger else None, max_s=a.max_s)
        if a.cmd == "recheck":
            same, got = recheck(work)
            print(json.dumps(got["decision"], default=str))
            return 0 if same else 3
        doc = json.loads((work / "look.json").read_text())
        if a.cmd == "read":
            inp, verdict = read_ledger(_ledger_root(a), doc)
            (work / "result").mkdir(parents=True, exist_ok=True)
            (work / "result" / "inputs.json").write_text(json.dumps(inp, indent=1, default=str))
            (work / "result" / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
            print(json.dumps(verdict["decision"], default=str))
            return 0
        from main.h2h import panel as PN

        cells, rows = ledger_rows(_ledger_root(a), doc)
        reasons = K.row_problems(rows, str(doc["pin"]), {p: PN.panel_team_set_id(p) for p in X.PANELS})
        rep = bootstrap(doc, cells, rows, row_reasons=reasons)
        (work / "result").mkdir(parents=True, exist_ok=True)
        (work / "result" / "team_bootstrap.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({k: rep.get(k) for k in ("outcome", "outcome_registered", "class_with_team_variance")}))
        return 3 if rep.get("outcome") == FR.INCONCLUSIVE else 0
    except (FreshError, K.LookError, A.ArmError, X.CrossInputError, OSError, KeyError, ValueError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
