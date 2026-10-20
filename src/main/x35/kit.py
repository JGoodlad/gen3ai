"""X35's LAUNCH KIT, the Python half of ``launch_x35.sh`` (``design_compute_efficiency.md`` §5.6.1-§5.6.3, §5.6.8;
STAGED_PLAN §2-§4). It BUILDS argvs and CHECKS preconditions; it never launches anything.

EVERY ARM'S ARGV comes from the frozen call record (``main.x35.refpop``'s file for the arm's side: P* or T*, its
step, A1's archive with labels), so the launch cannot drift from what R froze. The forms are the staging's validated
stand-ins (``argv_*_standin.txt``, each ``checkargs`` / ``launcher --dry-run`` checked at ``9ebda190``), in one
canonical flag order (the order carries no meaning to the trainer's parser):

* production (``--behaviour-check warn``, F14; +20M): ``c0``; ``a1`` (the 8 archive nodes, share 0.5);
  ``a2`` (``--pfsp-scale 2.5 --n-sentinels 10``); ``d1`` (X38, the pool WIDENING arm of
  ``design_diversity_teams_opponents.md`` §4.2: ``--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both``, read
  against C0 by ``main.x35.d1``'s own rule, not R1);
* the TWO-PHASE kits. Phase 1 (+5M, F16's ``--no-self-play`` where the parent self-plays): ``a3_mx`` -- a main
  exploiter of P* (seeds 3511-3513) -- and ``a6_lx`` -- a league exploiter of P* + the archive, share 1.0, stable
  PFSP (seeds 3521-3523). Between the phases, ``phase1-record`` writes ONE write-once record of the three
  exploiter finals (path, step, sha256), refusing a missing or short final. Phase 2 (+20M): ``a3`` / ``a6`` carry
  those finals as ``--stable-opponents`` at share 0.4 with ``--stable-opponent-pfsp`` (the registered round-1 B
  recipe), seeds 3501-3505, b-OFF = C0;
* the G-E READERS (``main.x35.ge``, F27 closed for the readers, 2026-10-20; +5M): ``ge_a1`` / ``ge_a2`` / ``ge_d1`` /
  ``ge_c0`` -- one fresh best responder of the arm's seed-3501 FINAL (``--model F --exploiter F --no-self-play``, the
  pinned offense 5-team archetype, ``--fork-lr 5.6e-05 --fork-lr-freeze``), launched only when the arm's primary reads
  GAIN (``stage-check``); the kit seed is the TARGET's, the reader trains at seed 3531;
* toy (``--behaviour-check warn`` under X34's registered GIGO ceilings, C3, orchestrator 2026-10-19; +70M; the X34
  toy extras): ``toy_c0``, ``toy_a1``, ``toy_a2``, and ``toy_x47`` -- X47 / depth growth (``design_bot_ideas_screen.md``
  §2.1), the toy C0 argv + the ONE lever ``--grow trunk_layers=4`` (seeds 3601-3603), read against ``toy_c0``.

    python -m main.x35.kit describe --arm a1 --seed 3501 --call <R_production.json> [--phase1 <record>]
    python -m main.x35.kit describe --arm ge_a1 --seed 3501 --call <R_production.json> [--models-dir D]   # a G-E reader
    python -m main.x35.kit phase1-record --arm a3 --call <R_production.json> --out <stage>/phase1_a3.json
    python -m main.x35.kit verify-phase1 <stage>/phase1_a3.json
    python -m main.x35.kit live                    # exit 1 while any training run is live (pre-check 1)
    python -m main.x35.kit disk --required-gib 20.68 [--thinning]   # pre-check 8 (F22)
    python -m main.x35.kit validate-run --arm toy_x47 --seed 3601 --call <R_toy.json> --dryrun F --checkargs F \\
        --rc-dryrun 0 --rc-checkargs 0     # ``validate_batch1.sh``'s verdict on one argv's transcripts (exit 1 = FAIL)

THE 2026-10-19 AMENDMENTS, and AM-8 / AM-9 of 2026-10-20 (``main.x35.batch``; ``launch_x35.sh`` pre-checks 2b / 2c / 9 /
10 / 11):

    python -m main.x35.kit pin-check --side toy --pin <sha> --stage <stage> [--write --t62 NA_TOY]   # AM-1: ONE pin PER SIDE
    python -m main.x35.kit required-check --side production --pin <sha> --stage <stage>   # AM-8: the side's FP32_ALT commit
    python -m main.x35.kit pool-check --arm c0 --call <R> --stage <stage> [--write]       # POOL: one seeded set
    python -m main.x35.kit gate <six_finals_S.json> --stage <stage> --write              # AM-2: G1, exit 1 = HOLD
    python -m main.x35.kit stage-check --arm d1 --seed 3501 --stage <stage>               # AM-9 / AM-3 / AM-4 / AM-8
    python -m main.x35.kit k9b --run-dir models/<run> --stage <stage>                     # AM-6, at each arm's end
    python -m main.x35.kit k9b-batch --stage <stage>                                      # AM-6 pre-check
    python -m main.x35.kit corpus-check --arm d1 --pin <sha>                              # D1-CORPUS: pre-check 2d
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.x35 import batch as _B
from main.x35 import ge as _GE
from main.x35.batch import STAGES as BATCH_STAGES
from main.x35.batch import BatchError
from main.x35.refpop import RefPopError, sha256_file
from main.x35.refpop import verify as verify_call

L_PROD = 20_000_000
L_EXPLOITER = 5_000_000
L_TOY = 70_000_000
CKPT_EVERY = "1000000"
#: F22: the launcher's REQUIRED headroom for one 20M production fork (dry-run, 2026-10-17).
FORK_GIB = 20.68
PHASE1_SCHEMA = "x35_phase1_exploiters_v1"
TOY_EXTRAS = ["--snapshot-ladder-games", "0", "--supply-starve-cycles", "self_play_pool=0"]
#: D1 / X38's ONE lever (``design_diversity_teams_opponents.md`` §4.2): half of each side's draws from the pin-safe
#: forum corpus (``utils.team_corpora``, tier ABC = 1,982 teams since AMENDMENT D1-CORPUS), both sides; eval stays on
#: the pool.
D1_LEVER = ("--team-mix", "w_forum_v1@ABC:0.5", "--team-mix-sides", "both")
#: AMENDMENT D1-CORPUS (2026-10-20, BATCH1_LAUNCH_SHEET.md §0, before any D1 launch): the corpus D1_LEVER names IS this
#: one. It is ``w_forum_v1`` rebuilt after X54's sealed confirmation panel ``forum216b`` was carved out of its AB tier
#: (``designs/research_state/measurements/forum216b_x54_2026-10-20/``). It was 2,424 teams, tier sha ``7bcdc835c107``.
#: The PIN must carry it (:func:`corpus_check`, launch pre-check 2d): a pin from before the carve would train D1 on the
#: panel X54 confirms on. The pin must also carry a PASSing crash soak OF THIS corpus (``soak_selfcheck.json`` naming its
#: data sha256): the rebuild re-selected the C tier, and 323 of its teams were in no soaked corpus.
D1_CORPUS = {"corpus": "w_forum_v1", "tier": "ABC", "n": 1982,
             "tier_sha256": "73a76e808431f5c430edf7effafc121a0b4822c6dbd19d33b82b6bb60570bdc4",
             "data_sha256": "7f42d95f7f08f667738fe6e8f2af33ede93f5dec78af6fa48ed3f72f3d231336"}
#: The arms whose argv names a team corpus, and the identity their pin must carry.
ARM_CORPUS = {"d1": D1_CORPUS}
#: X47's ONE lever (``design_bot_ideas_screen.md`` §2.1; B-GROW, T64): the toy's 2 attention rounds become 4 at the
#: fork, the 2 new rounds identity-init, so the grown network starts as T*'s function bit for bit (the child refuses
#: to start otherwise: ``FATAL_CONFIG``). ``--dry-run`` prints ``grow : DECLARED trunk_layers 2 -> 4``.
X47_LEVER = ("--grow", "trunk_layers=4")
#: What ``checkargs`` / ``--dry-run`` print for it ("DECLARED <this> (parent from its model_config.json)").
X47_GROWTH = "trunk_layers 2 -> 4"
#: C3 (orchestrator, 2026-10-19): the toy arms run ``--behaviour-check warn`` (K9(b)), as T*'s own run X34 does
#: (``x34_toy_deep_stage_2026-10-17/STAGED_LAUNCH.md`` §7, c9516bac / 96c663d4). The mechanism is the flag plus
#: ceilings READ BY HAND (``main.ops.tb_read``, at every restart and daily); the trainer enforces none: STOP the arm on
#: any update with judged max >= 1e-3, p99 >= 1e-4 or a non-finite value; within ``rule8_rel`` of a ceiling is
#: recorded, no fire (standing rule 8). The toy has no belief tokens, so the production tau class cannot occur and
#: X26's Amendment 6 does not apply. ``describe`` carries them for the toy arms only (production's are the sheet's).
K9B_TOY_CEILINGS: Dict[str, Any] = {"judged_max": 1e-3, "p99": 1e-4, "non_finite": "STOP", "rule8_rel": 1e-9}


@dataclass(frozen=True)
class Arm:
    name: str
    side: str            # the call record's side: production | toy
    length: int
    seeds: Tuple[int, ...]
    run: str             # run-name format
    phase: int           # 0 = single phase, 1 = exploiters, 2 = folded in, 3 = a G-E reader (main.x35.ge)
    behaviour: str
    phase1_of: Optional[str] = None  # phase 2: the phase-1 arm whose record it reads


PROD_SEEDS = tuple(range(3501, 3506))
TOY_SEEDS = tuple(range(3601, 3606))
X47_SEEDS = TOY_SEEDS   # X47: look 1 at 3601-3603, look 2 at 3604-3605 (AM-4, orchestrator 2026-10-19), against toy C0 at the same seeds
ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("c0", "production", L_PROD, PROD_SEEDS, "rb_x35_c0_s{seed}", 0, "warn"),
    Arm("a1", "production", L_PROD, PROD_SEEDS, "rb_x35_a1arch_s{seed}", 0, "warn"),
    Arm("a2", "production", L_PROD, PROD_SEEDS, "rb_x35_a2pfsp_s{seed}", 0, "warn"),
    Arm("d1", "production", L_PROD, PROD_SEEDS, "rb_x35_d1wide_s{seed}", 0, "warn"),
    Arm("a3_mx", "production", L_EXPLOITER, (3511, 3512, 3513), "rb_x35_mx_s{seed}", 1, "warn"),
    Arm("a3", "production", L_PROD, PROD_SEEDS, "rb_x35_a3mx_s{seed}", 2, "warn", phase1_of="a3_mx"),
    Arm("a6_lx", "production", L_EXPLOITER, (3521, 3522, 3523), "rb_x35_lx_s{seed}", 1, "warn"),
    Arm("a6", "production", L_PROD, PROD_SEEDS, "rb_x35_a6lx_s{seed}", 2, "warn", phase1_of="a6_lx"),
    Arm("toy_c0", "toy", L_TOY, TOY_SEEDS, "rb_x35t_c0_s{seed}", 0, "warn"),
    Arm("toy_a1", "toy", L_TOY, TOY_SEEDS, "rb_x35t_a1arch_s{seed}", 0, "warn"),
    Arm("toy_a2", "toy", L_TOY, TOY_SEEDS, "rb_x35t_a2pfsp_s{seed}", 0, "warn"),
    Arm("toy_x47", "toy", L_TOY, X47_SEEDS, "rb_x35t_x47d4_s{seed}", 0, "warn"),
)}
#: ARM ALIASES: the ONE table of alternative spellings a launch sheet or an operator may type; every kit subcommand
#: that takes ``--arm`` resolves it through :func:`canonical_arm` (argparse ``type=``), so ``x47`` == ``toy_x47``
#: everywhere (``describe``, ``validate-run``, ``stage-check``, ``pool-check``, ``pool-seed-check``).
ARM_ALIASES: Dict[str, str] = {"x47": "toy_x47"}


def canonical_arm(name: str) -> str:
    """The kit arm a typed name stands for (an alias resolves; every other string passes through for the caller's
    own refusal)."""
    return ARM_ALIASES.get(name, name)


#: The G-E READERS (``main.x35.ge``): the kit seed is the TARGET's seed; the run is ``rb_x35_ge_<target tag>_s<seed>``.
ARMS.update({r: Arm(r, ARMS[t].side, _GE.L_READER, _GE.TARGET_SEEDS,
                    "rb_x35_ge_" + ARMS[t].run[len("rb_x35_"):], _GE.READER_PHASE, "warn")
             for r, t in _GE.READER_ARMS.items()})
assert all(v in ARMS for v in ARM_ALIASES.values()) and not set(ARM_ALIASES) & set(ARMS)
#: Batch 1 + X47, in GPU order: AM-9 (orchestrator, 2026-10-20; it moved the toy block ahead of production, which waits
#: for the FP32_ALT build), stage by stage in ``main.x35.batch.STAGES``: T1 toy C0 + X47 (C2's interleave, toy A1 out) ->
#: T1b X47 look 2 with toy C0 s3604-s3605 (AM-4: X47 look 1 CONTINUE) -> P1 the A1 / C0 ABBA (s3501-s3503) -> [G1,
#: AM-2] -> P2 C0 s3504-s3505 -> P3 A1 s3504-s3505 (A1 look 1 CONTINUE) -> P4 D1 (G1 PASS) -> T2 toy A1 (AM-3:
#: production's FINAL class); A2 / toy A2 DROPPED (AM-RP-c, 2026-10-20: ``batch.DROPPED``). The conditional stages launch only when ``batch.stage_status`` says ALLOWED
#: (``launch_x35.sh`` pre-check 9), and no production fork before its FP32_ALT commit is named (AM-8).
BATCH1: Tuple[Tuple[str, int], ...] = tuple(f for _, forks in BATCH_STAGES for f in forks)


class KitError(ValueError):
    """A precondition of the kit does not hold (named; the launch script HOLDs on it)."""


def _arm(name: str, seed: int) -> Arm:
    name = canonical_arm(name)
    if name not in ARMS:
        raise KitError(f"arm {name!r} is not one of {sorted(ARMS)}")
    a = ARMS[name]
    if seed not in a.seeds:
        raise KitError(f"seed {seed} is not registered for {name} ({list(a.seeds)})")
    return a


def archive_spec(call: Mapping[str, Any]) -> str:
    """A1's ``--stable-opponents`` value: N_8 … N_1 as explicit ``<zip>:<label>`` (§5.6.3: 8 unique labels)."""
    arch = sorted(call["archive"], key=lambda e: -int(e["k"]))
    return ",".join(f"{e['path']}:{e['label']}" for e in arch)


def run_name(arm: str, seed: int) -> str:
    return _arm(arm, seed).run.format(seed=seed)


def build_argv(arm: str, seed: int, call: Mapping[str, Any], phase1: Optional[Mapping[str, Any]] = None,
               target: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The trainer argv (the launcher's own flags excluded) of one arm × seed. A G-E reader needs its ``target``
    (:func:`reader_target`)."""
    a = _arm(arm, seed)
    if call.get("side") != a.side:
        raise KitError(f"arm {arm} reads the {a.side} call record, got side {call.get('side')!r}")
    if a.phase == _GE.READER_PHASE:
        if target is None:
            raise KitError(f"arm {arm} is a G-E reader: it needs its target final (reader_target)")
        return _GE.build_reader_argv(arm, seed, target, behaviour=a.behaviour, ckpt_every=CKPT_EVERY)
    p = call["pstar"]
    start, step = str(p["path"]), int(p["step"])
    head = ["--model", start]
    lever: List[str] = []
    if arm == "a3_mx":
        head += ["--exploiter", start, "--no-self-play"]
    elif arm == "a6_lx":
        prefix = str(p.get("label") or "x26_pstar")
        lever = ["--stable-opponents", f"{start}:{prefix}," + archive_spec(call),
                 "--stable-opponent-selfplay-share", "1.0", "--stable-opponent-pfsp"]
    elif arm in ("a1", "toy_a1"):
        lever = ["--stable-opponents", archive_spec(call), "--stable-opponent-selfplay-share", "0.5"]
    elif arm in ("a2", "toy_a2"):
        lever = ["--pfsp-scale", "2.5", "--n-sentinels", "10"]
    elif arm == "d1":
        lever = list(D1_LEVER)
    elif arm == "toy_x47":
        lever = list(X47_LEVER)
    elif a.phase == 2:
        if phase1 is None:
            raise KitError(f"arm {arm} is phase 2: it needs the {a.phase1_of} phase-1 record")
        if phase1.get("arm") != a.phase1_of or phase1.get("pstar_sha256") != p["sha256"]:
            raise KitError(f"the phase-1 record is for {phase1.get('arm')} from P* "
                           f"{str(phase1.get('pstar_sha256'))[:12]}, not {a.phase1_of} from {str(p['sha256'])[:12]}")
        lever = ["--stable-opponents", ",".join(f"{e['path']}:{e['label']}" for e in phase1["exploiters"]),
                 "--stable-opponent-selfplay-share", "0.4", "--stable-opponent-pfsp"]
    out = head + ["--behaviour-check", a.behaviour, "--steps", str(step + a.length),
                  "--checkpoint-every-steps", CKPT_EVERY]
    if a.side == "toy":
        out += TOY_EXTRAS
    return out + ["--device", "cuda"] + lever + ["--seed", str(seed), "--run-name", run_name(arm, seed)]


def _run_dir_of(path: str) -> str:
    p = Path(path)
    return str(p.parent.parent if p.parent.name == "checkpoints" else p.parent)


def parent_run_dir(call: Mapping[str, Any]) -> str:
    """The run the fork's dry-run must name: the start node's run directory."""
    return _run_dir_of(str(call["pstar"]["path"]))


def reader_target(arm: str, seed: int, call: Mapping[str, Any], *, models_dir: Optional[str] = None,
                  standin: Optional[str] = None) -> Dict[str, Any]:
    """A G-E reader's target: the target arm's FINISHED ``final_model.zip`` under ``models_dir`` (at least P*'s step
    + the arm's length, from the call record), or -- VALIDATION ONLY -- a ``standin`` model file (its own step)."""
    a = _arm(arm, seed)
    if a.phase != _GE.READER_PHASE:
        raise KitError(f"arm {arm} is not a G-E reader")
    if standin:
        return _GE.standin_target(standin, _final_step(standin))
    t = ARMS[_GE.READER_ARMS[arm]]
    try:
        return _GE.target_final(arm, seed, models_dir or "", min_step=int(call["pstar"]["step"]) + t.length)
    except _GE.ReaderError as exc:
        raise KitError(str(exc)) from exc


def describe(arm: str, seed: int, call: Mapping[str, Any],
             phase1: Optional[Mapping[str, Any]] = None, target: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    a = _arm(arm, seed)
    argv = build_argv(arm, seed, call, phase1, target)
    parent = _run_dir_of(str(target["path"])) if a.phase == _GE.READER_PHASE and target else parent_run_dir(call)
    return {"arm": a.name, "seed": seed, "side": a.side, "phase": a.phase, "run_name": run_name(arm, seed),
            "parent_run_dir": parent, "delta_steps": a.length, "delta_text": f"+{a.length:,} steps",
            "behaviour": a.behaviour, "k9b_ceilings": dict(K9B_TOY_CEILINGS) if a.side == "toy" else None,
            "target": dict(target) if target else None,
            "argv": argv, "argv_shell": " ".join(shlex.quote(x) for x in argv)}


# ------------------------------------------------------------------------------------------- the phase-1 record
def _final_step(zip_path: str) -> int:
    from agents.training.fixed_opponent_pool import resolve_model_ref

    try:
        r = resolve_model_ref(zip_path, warn=False)
    except (FileNotFoundError, ValueError) as exc:
        raise KitError(f"{zip_path}: does not resolve as a model ({exc})") from exc
    if r.num_timesteps is None:
        raise KitError(f"{zip_path}: no recorded step")
    return int(r.num_timesteps)


def phase1_record(arm: str, call: Mapping[str, Any], models_dir: str, out: str,
                  now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Freeze the three exploiter finals of a phase-1 arm (``a3_mx`` / ``a6_lx``); write-once."""
    a = ARMS.get(arm)
    if a is None or a.phase != 1:
        raise KitError(f"{arm!r} is not a phase-1 arm ({[n for n, x in ARMS.items() if x.phase == 1]})")
    if os.path.exists(out):
        raise KitError(f"{out} exists: a phase-1 record is written ONCE")
    p = call["pstar"]
    want = int(p["step"]) + a.length
    ex: List[Dict[str, Any]] = []
    finals = [(run_name(arm, s), Path(models_dir) / run_name(arm, s) / "final_model.zip") for s in a.seeds]
    for rn, z in finals:
        if not z.is_file():
            raise KitError(f"{rn}: no final_model.zip (an interrupted or running exploiter is not a phase-1 final)")
    for rn, z in finals:
        step = _final_step(str(z))
        if step < want:
            raise KitError(f"{rn}: final at {step:,} < the registered {want:,} (P* + {a.length:,})")
        ex.append({"label": rn.replace("rb_x35_", ""), "run": rn, "path": str(z.resolve()), "step": step,
                   "sha256": sha256_file(z)})
    t = now or dt.datetime.now(dt.timezone.utc)
    doc = {"schema": PHASE1_SCHEMA, "arm": arm, "pstar_sha256": p["sha256"], "pstar_step": int(p["step"]),
           "written_at": t.isoformat(), "exploiters": ex}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def verify_phase1(path: str) -> List[str]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: unreadable ({exc})"]
    probs = [] if doc.get("schema") == PHASE1_SCHEMA else [f"schema {doc.get('schema')!r}"]
    if len(doc.get("exploiters") or []) != 3:
        probs.append("a phase-1 record holds exactly 3 exploiters")
    for e in doc.get("exploiters") or []:
        if not os.path.exists(e["path"]):
            probs.append(f"{e['path']}: missing")
        elif sha256_file(e["path"]) != e["sha256"]:
            probs.append(f"{e['path']}: sha256 changed since the record")
    return probs


# ------------------------------------------------------------------------------ validating one argv's transcripts
_DRY_K9B = re.compile(r"^\s+K9\(b\) check\s*: (\w+)", re.M)
_CHK_K9B = re.compile(r"^\s+K9\(b\) behaviour check\s*: (\w+)", re.M)
_DRY_GROW = re.compile(r"^\s+grow\s+: DECLARED (.+?) \(parent", re.M)
_CHK_GROW = re.compile(r"^\s+declared growth \(--grow\)\s*: (.+?)\s+\(parent", re.M)
_DRY_GIB = re.compile(r"([0-9.]+) GiB REQUIRED")


def check_validation(d: Mapping[str, Any], dryrun: str, checkargs: str, rc_checkargs: int,
                     rc_dryrun: int) -> Tuple[bool, str]:
    """``validate_batch1.sh``'s verdict on ONE argv: ``(ok, one line)``. ``d`` is :func:`describe`'s record. PASS iff
    both tools exited 0, the dry run says "would launch" as a FORK of the start node's run with the arm's +steps and
    the arm's K9(b) mode, ``checkargs`` prints the same mode, the trainer's own arch check passed, and the DECLARED
    growth line is there for X47 (``X47_GROWTH``, in BOTH tools) and for no other arm (the always-printed
    "(child-only: a declared growth's …)" note is not a growth line)."""
    probs: List[str] = []
    if rc_checkargs != 0:
        probs.append(f"checkargs exit {rc_checkargs}")
    if rc_dryrun != 0:
        probs.append(f"dry-run exit {rc_dryrun}")
    if dryrun.count("this command would launch") != 1:
        probs.append("the dry-run does not say 'would launch' exactly once")
    if f"role        : FORK of {d['parent_run_dir']}" not in dryrun:
        probs.append(f"the dry-run role is not FORK of {d['parent_run_dir']}")
    if d["delta_text"] not in dryrun:
        probs.append(f"the dry-run step arithmetic is not {d['delta_text']}")
    if "✓ fork arch" not in dryrun:
        probs.append("no '✓ fork arch' line (the trainer's own architecture check)")
    for tool, rx, text in (("dry-run", _DRY_K9B, dryrun), ("checkargs", _CHK_K9B, checkargs)):
        m = rx.search(text)
        if m is None or m.group(1) != d["behaviour"]:
            probs.append(f"{tool} K9(b) mode {m.group(1) if m else None!r} != the arm's {d['behaviour']!r}")
    gd, gc = _DRY_GROW.search(dryrun), _CHK_GROW.search(checkargs)
    want = X47_GROWTH if "--grow" in d["argv"] else None
    for tool, g in (("dry-run", gd), ("checkargs", gc)):
        got = g.group(1) if g else None
        if got != want:
            probs.append(f"{tool} declared growth {got!r} != {want!r}")
    gib = _DRY_GIB.search(dryrun)
    if gib is None:
        probs.append("the dry-run printed no REQUIRED GiB")
    line = (f"{d['arm']}_s{d['seed']} checkargs={rc_checkargs} dryrun={rc_dryrun} "
            f"would_launch={dryrun.count('this command would launch')} delta=[{d['delta_text']}] "
            f"gib=[{gib.group(1) if gib else None}] role=[FORK of {d['parent_run_dir']}] k9b=[{d['behaviour']}] "
            f"grow=[{('DECLARED ' + gd.group(1)) if gd else 'none'}] verdict=" + ("PASS" if not probs else
                                                                                  "FAIL " + "; ".join(probs)))
    return not probs, line


# ---------------------------------------------------------------------------------- the post-spawn pool check
#: Arms whose fork does NOT self-play, so nothing is seeded (F16: the main exploiter types --no-self-play).
NO_POOL_ARMS = frozenset({"a3_mx", *_GE.READER_ARMS})
POOL_SEED_SCHEMA = 2
_FILTER_RE = re.compile(r"\[step filter <= (\d+): (?:DROPPED (\d+) later snapshot\(s\) at steps \[([0-9, ]*)\]"
                        r"|none later)")
PENDING = "PENDING"


def check_pool_seed(arm: str, run_dir: str, call: Mapping[str, Any], log_text: str,
                    fork_num_timesteps: Optional[int],
                    expected_names: Optional[Sequence[str]] = None) -> Tuple[str, List[str]]:
    """T54's post-spawn check (``--dry-run`` cannot preview it: the seed runs in the child). ``(status, problems)``,
    status PASS / FAIL / PENDING (the record is not written yet). FAIL iff ``pool_seed.json`` is not schema 2, its
    ``fork_step`` is NULL (T54 only WARNS and seeds unfiltered when it cannot read the step: a HOLD here) or differs
    from P*'s ``num_timesteps`` (and the call record's step), a seeded snapshot is later than it, or the startup
    line "[step filter <= N: …]" disagrees with the record (or says the filter is OFF), or -- POOL, 2026-10-19 --
    the seeded names differ from ``expected_names`` (the side's recorded set, ``batch.POOL_FILE``)."""
    from agents.training.pool_seed import SEED_RECORD_FILE, pool_dir_for, snapshot_step

    if arm in NO_POOL_ARMS:
        return "PASS", []
    p = Path(pool_dir_for(run_dir)) / SEED_RECORD_FILE
    if not p.is_file():
        return PENDING, [f"{p} not written yet"]
    try:
        rec = json.loads(p.read_text())
    except ValueError as exc:
        return PENDING, [f"{p}: not yet parseable ({exc})"]
    probs: List[str] = []
    if rec.get("schema") != POOL_SEED_SCHEMA:
        probs.append(f"pool_seed.json schema {rec.get('schema')!r} != {POOL_SEED_SCHEMA} (a pin without T54?)")
    fs = rec.get("fork_step")
    want = int(call["pstar"]["step"])
    if fs is None:
        probs.append("fork_step is NULL: T54 could not read the fork step and seeded UNFILTERED")
    else:
        if fork_num_timesteps is None or int(fs) != int(fork_num_timesteps):
            probs.append(f"fork_step {fs} != P*'s num_timesteps {fork_num_timesteps}")
        if int(fs) != want:
            probs.append(f"fork_step {fs} != the call record's P* step {want}")
        late = [n for n in rec.get("snapshots") or [] if (st := snapshot_step(n)) is None or st > int(fs)]
        if late:
            probs.append(f"seeded snapshot(s) later than fork_step {fs} (or unnamed): {late}")
    if expected_names is not None and sorted(rec.get("snapshots") or []) != sorted(expected_names):
        got = set(rec.get("snapshots") or [])
        probs.append(f"the seeded set is not the batch's recorded set: missing {sorted(set(expected_names) - got)}, "
                     f"extra {sorted(got - set(expected_names))}")
    if "step filter OFF" in log_text:
        probs.append("the startup line says the step filter is OFF")
    m = _FILTER_RE.findall(log_text)
    if not m:
        if not probs:
            return PENDING, ["the startup '[step filter <= N: …]' line is not in the log yet"]
    else:
        n, k, steps = m[-1]
        logged = sorted(int(x) for x in steps.replace(" ", "").split(",") if x) if k else []
        if fs is not None and int(n) != int(fs):
            probs.append(f"the startup line's step filter {n} != the record's fork_step {fs}")
        if logged != sorted(rec.get("dropped_steps") or []) or (k and int(k) != len(logged)):
            probs.append(f"the startup line's dropped steps {logged} != the record's {rec.get('dropped_steps')}")
    return ("FAIL" if probs else "PASS"), probs


# ------------------------------------------------------------------------------------------------ pre-checks
def live_runs() -> Dict[str, int]:
    from utils.procfs import live_training_runs

    return live_training_runs()


def disk_check(models_dir: str, required_gib: float, *, thinning: bool,
               existing: Optional[Sequence[str]] = None, free_gib: Optional[float] = None,
               stage: Optional[str] = None) -> Tuple[bool, str]:
    """Pre-check 8 (F22): free disk covers EVERY not-yet-created batch-1 fork at ``required_gib`` each, or the
    operator declares ``thinning`` (finished arms are being thinned to their 10M nodes + final). With ``stage``, a
    conditional fork its records have REFUSED (AM-7: ``batch.stage_status``) is not counted; a PENDING one is (the
    conservative side)."""
    from main.x35.batch import REFUSED, stage_status

    have = set(existing) if existing is not None else {d for d in os.listdir(models_dir)
                                                        if d.startswith("rb_x35")}
    remaining = [run_name(a, s) for a, s in BATCH1 if run_name(a, s) not in have
                 and (stage is None or stage_status(stage, a, s)[0] != REFUSED)]
    free = free_gib if free_gib is not None else shutil.disk_usage(models_dir).free / 2 ** 30
    need = len(remaining) * required_gib
    msg = f"{free:.1f} GiB free; {len(remaining)} batch-1 fork(s) to create x {required_gib:.2f} = {need:.1f} GiB"
    if free >= need:
        return True, msg
    if thinning and free >= required_gib:
        return True, msg + " -- short for the whole batch, THINNING declared (F22): this launch alone fits"
    return False, msg + (" -- not even this launch fits" if free < required_gib else
                         " -- thin finished arms (F22) or pass --thinning")


def corpus_check(arm: str, pin: str, repo: str) -> Tuple[bool, str]:
    """``(ok, line)``: the team corpus ``arm`` names (:data:`ARM_CORPUS`) is, AT ``pin``, the registered identity
    (count, tier sha256 and data sha256, read from the pin's committed manifest). An arm that names no corpus passes."""
    import subprocess

    want = ARM_CORPUS.get(canonical_arm(arm))
    if want is None:
        return True, f"{arm} names no team corpus"
    rel = f"src/utils/team_corpora/{want['corpus']}/manifest.json"
    p = subprocess.run(["git", "-C", repo, "show", f"{pin}:{rel}"], capture_output=True, text=True)
    if p.returncode != 0:
        return False, f"the pin {pin[:12]} holds no {rel}: {p.stderr.strip()[-200:]}"
    man = json.loads(p.stdout)
    tier = man.get("tiers", {}).get(want["tier"], {})
    got = {"n": tier.get("n"), "tier_sha256": tier.get("sha256"), "data_sha256": man.get("data_sha256")}
    bad = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
    if bad:
        return False, (f"the pin {pin[:12]}'s {want['corpus']}@{want['tier']} is not {arm}'s registered corpus "
                       f"(AMENDMENT D1-CORPUS): " + "; ".join(f"{k} {g} != {w}" for k, (g, w) in bad.items()))
    soak_rel = f"src/utils/team_corpora/{want['corpus']}/soak_selfcheck.json"
    p = subprocess.run(["git", "-C", repo, "show", f"{pin}:{soak_rel}"], capture_output=True, text=True)
    soak = json.loads(p.stdout) if p.returncode == 0 else {}
    sc = soak.get("corpus", {})
    if not (soak.get("verdict") == "PASS" and soak.get("core_errors") == [] and sc.get("tier") == want["tier"]
            and sc.get("data_sha256") == want["data_sha256"]
            and soak.get("battles_finished") == soak.get("battles_planned") > 0):
        return False, (f"the pin {pin[:12]} carries no PASSing crash soak of {arm}'s corpus ({soak_rel} names data "
                       f"{str(sc.get('data_sha256'))[:12]}, verdict {soak.get('verdict')}; want data "
                       f"{want['data_sha256'][:12]}): run `python -m utils.team_corpora.soak {want['corpus']} --tier "
                       f"{want['tier']} --rounds 10 --out {soak_rel}` and commit it (AMENDMENT D1-CORPUS)")
    return True, (f"{want['corpus']}@{want['tier']} at the pin {pin[:12]}: {want['n']} teams, tier sha "
                  f"{want['tier_sha256'][:12]} (the registered corpus), soaked: {soak['battles_finished']} battles, "
                  "0 core errors")


def _load(path: str) -> Dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def _cli_target(a: argparse.Namespace, call: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The target of a G-E reader arm on the CLI (None for every other arm)."""
    if ARMS[a.arm].phase != _GE.READER_PHASE:
        return None
    from utils.paths import main_models_dir

    md = a.models_dir or str(main_models_dir() or "")
    return reader_target(a.arm, a.seed, call, models_dir=md, standin=a.target_standin)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.kit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    d.add_argument("--seed", required=True, type=int)
    d.add_argument("--call", required=True, help="the frozen R file of the arm's side (main.x35.refpop)")
    d.add_argument("--phase1", default=None, help="the phase-1 record (phase-2 arms)")
    d.add_argument("--models-dir", default=None, help="G-E readers: where the target's final is (default: main's)")
    d.add_argument("--target-standin", default=None,
                   help="G-E readers, VALIDATION ONLY: a stand-in target model file (never a launch value)")
    d.add_argument("--field", default=None, help="print one field (argv = one token per line, argv_shell, "
                                                 "run_name, parent_run_dir, delta_text, side, phase)")
    r = sub.add_parser("phase1-record")
    r.add_argument("--arm", required=True, choices=("a3", "a6"))
    r.add_argument("--call", required=True)
    r.add_argument("--models-dir", default=None)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verify-phase1")
    v.add_argument("file")
    sub.add_parser("live")
    q = sub.add_parser("pool-seed-check", help="T54's post-spawn check: exit 0 PASS, 1 FAIL, 3 PENDING")
    q.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    q.add_argument("--run-dir", required=True)
    q.add_argument("--call", required=True)
    q.add_argument("--log", action="append", default=[], help="a log holding the startup line (repeatable)")
    q.add_argument("--stage", default=None, help="compare the seeded names with the side's recorded set (POOL)")
    pc = sub.add_parser("pin-check", help="AM-1 (one pin PER SIDE): the pin == the side's X35_PIN_1 and every batch-1 "
                                          "run of that side's recorded pin; exit 0 / 1")
    pc.add_argument("--pin", required=True, help="the full sha")
    pc.add_argument("--side", required=True, choices=_B.SIDES)
    pc.add_argument("--stage", required=True)
    pc.add_argument("--models-dir", default=None)
    pc.add_argument("--write", action="store_true",
                    help="GO only: write x35_pin_batch1_<side>.txt if it does not exist")
    pc.add_argument("--t62", default=None, choices=_B.T62_VERDICTS,
                    help="the side's T62 token (production FP32_ALT, toy NA_TOY): the first GO of a side records it")
    rc = sub.add_parser("required-check", help="AM-8: the side's own required commits (production: FP32_ALT) are real "
                                               "and in the pin; a placeholder refuses; exit 0 / 1")
    rc.add_argument("--side", required=True, choices=_B.SIDES)
    rc.add_argument("--pin", required=True, help="the full sha")
    rc.add_argument("--stage", required=True)
    rc.add_argument("--repo", default=None, help="the git checkout the commits are looked up in (default: this one)")
    cc = sub.add_parser("corpus-check", help="AMENDMENT D1-CORPUS: the arm's team corpus AT the pin is its registered "
                                             "identity (an arm with no corpus passes); exit 0 / 1")
    cc.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    cc.add_argument("--pin", required=True, help="the full sha")
    cc.add_argument("--repo", default=None, help="the git checkout the pin is read from (default: this one)")
    po = sub.add_parser("pool-check", help="POOL: the seeded set == the side's record; exit 0 / 1")
    po.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    po.add_argument("--call", required=True)
    po.add_argument("--stage", required=True)
    po.add_argument("--write", action="store_true", help="GO only: record the set if the side has no record")
    g = sub.add_parser("gate", help="AM-2's G1 from {a1: {seed: S}, c0: {seed: S}}: exit 0 PASS, 1 HOLD")
    g.add_argument("input")
    g.add_argument("--stage", default=None, help="with --write: where x35_gate_g1.json is written (once)")
    g.add_argument("--write", action="store_true")
    sc = sub.add_parser("stage-check", help="AM-7 / AM-3 / AM-4: may this fork launch now? exit 0 ALLOWED, 1 not")
    sc.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    sc.add_argument("--seed", required=True, type=int)
    sc.add_argument("--stage", required=True)
    kb = sub.add_parser("k9b", help="AM-6: K9(b) over the fork's own rows at arm end: exit 0 PASS, 1 VOID, 3 INCONCLUSIVE")
    kb.add_argument("--run-dir", required=True)
    kb.add_argument("--stage", default=None, help="write the record <stage>/k9b/<run>.json")
    kq = sub.add_parser("k9b-batch", help="AM-6 pre-check: every finished batch-1 arm's K9(b) record PASSes; exit 0 / 1")
    kq.add_argument("--stage", required=True)
    kq.add_argument("--models-dir", default=None)
    vr = sub.add_parser("validate-run", help="the verdict on one argv's checkargs + dry-run transcripts: exit 0 PASS, 1 FAIL")
    vr.add_argument("--arm", required=True, type=canonical_arm, choices=sorted(ARMS))
    vr.add_argument("--seed", required=True, type=int)
    vr.add_argument("--call", required=True)
    vr.add_argument("--phase1", default=None)
    vr.add_argument("--models-dir", default=None)
    vr.add_argument("--target-standin", default=None, help="G-E readers: the stand-in the transcripts were built on")
    vr.add_argument("--dryrun", required=True, help="the --dry-run transcript")
    vr.add_argument("--checkargs", required=True, help="the checkargs transcript")
    vr.add_argument("--rc-dryrun", required=True, type=int)
    vr.add_argument("--rc-checkargs", required=True, type=int)
    k = sub.add_parser("disk")
    k.add_argument("--required-gib", type=float, default=FORK_GIB)
    k.add_argument("--models-dir", default=None)
    k.add_argument("--thinning", action="store_true")
    k.add_argument("--stage", default=None, help="skip the conditional forks the stage's records REFUSED (AM-7)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "describe":
            call = _load(a.call)
            probs = verify_call(a.call)
            if probs:
                raise KitError(f"the call record {a.call} does not verify: {probs}")
            doc = describe(a.arm, a.seed, call, _load(a.phase1) if a.phase1 else None, _cli_target(a, call))
            if a.field == "argv":
                print("\n".join(doc["argv"]))  # one token per line: the launch script's mapfile
            else:
                print(doc[a.field] if a.field else json.dumps(doc, indent=1))
            return 0
        if a.cmd == "phase1-record":
            from utils.paths import main_models_dir

            md = a.models_dir or str(main_models_dir() or "")
            doc = phase1_record(f"{a.arm}_{'mx' if a.arm == 'a3' else 'lx'}", _load(a.call), md, a.out)
            print(f"[x35 kit] phase-1 record {a.out}: " + ", ".join(e["label"] for e in doc["exploiters"]))
            return 0
        if a.cmd == "verify-phase1":
            probs = verify_phase1(a.file)
            for x in probs:
                print(f"[x35 kit] {x}", file=sys.stderr)
            return 0 if not probs else 1
        if a.cmd == "pool-seed-check":
            from agents.training.lineage import checkpoint_num_timesteps

            call = _load(a.call)
            text = "".join(Path(x).read_text(errors="replace") for x in a.log if os.path.isfile(x))
            names = _B.recorded_pool_names(a.stage, ARMS[a.arm].side) if a.stage else None
            if a.stage and names is None:
                print(f"[x35 kit] pool seed: FAIL: no recorded {ARMS[a.arm].side} set in {a.stage} (POOL)")
                return 1
            status, probs = check_pool_seed(a.arm, a.run_dir, call, text,
                                            checkpoint_num_timesteps(str(call["pstar"]["path"])), names)
            print(f"[x35 kit] pool seed: {status}" + (f": {probs}" if probs else ""))
            return {"PASS": 0, "FAIL": 1}.get(status, 3)
        if a.cmd == "validate-run":
            vcall = _load(a.call)
            ok, line = check_validation(describe(a.arm, a.seed, vcall, _load(a.phase1) if a.phase1 else None,
                                                 _cli_target(a, vcall)),
                                        Path(a.dryrun).read_text(errors="replace"),
                                        Path(a.checkargs).read_text(errors="replace"), a.rc_checkargs, a.rc_dryrun)
            print(line)
            return 0 if ok else 1
        if a.cmd in ("pin-check", "k9b-batch"):
            from utils.paths import main_models_dir

            md = a.models_dir or str(main_models_dir() or "")
            if a.cmd == "k9b-batch":
                ok, lines = _B.k9b_batch(a.stage, md)
            else:
                ok, lines = _B.pin_check(a.pin, a.stage, md, a.side, a.t62)
                if ok and a.write:
                    lines.append(_B.pin_write_if_absent(a.pin, a.stage, a.side, a.t62))
            for x in lines:
                print(f"[x35 kit] {a.cmd}: {x}")
            return 0 if ok else 1
        if a.cmd == "required-check":
            from utils.paths import repo_root

            ok, lines = _B.required_check(a.stage, a.side, a.pin, a.repo or str(repo_root()))
            for x in lines:
                print(f"[x35 kit] required-check {a.side}: {x}")
            return 0 if ok else 1
        if a.cmd == "corpus-check":
            from utils.paths import repo_root

            ok, line = corpus_check(a.arm, a.pin, a.repo or str(repo_root()))
            print(f"[x35 kit] corpus-check: {'PASS' if ok else 'FAIL'}: {line}")
            return 0 if ok else 1
        if a.cmd == "pool-check":
            call = _load(a.call)
            ok, line = _B.pool_check(a.stage, ARMS[a.arm].side, call, parent_run_dir(call), write=a.write)
            print(f"[x35 kit] pool-check: {'PASS' if ok else 'FAIL'}: {line}")
            return 0 if ok else 1
        if a.cmd == "gate":
            doc = _load(a.input)
            g1 = _B.gate_g1(doc)
            print(json.dumps(g1.as_dict(), indent=1))
            if a.write:
                if not a.stage:
                    raise KitError("gate --write needs --stage")
                print(f"[x35 kit] G1 written to {_B.write_g1(a.stage, g1, doc)}")
            return 0 if g1.verdict == _B.PASS else 1
        if a.cmd == "stage-check":
            _arm(a.arm, a.seed)
            st, why = _B.stage_status(a.stage, a.arm, a.seed)
            print(f"[x35 kit] stage-check {a.arm} s{a.seed}: {st}: {why}")
            return 0 if st == _B.ALLOWED else 1
        if a.cmd == "k9b":
            k = _B.k9b_read(a.run_dir)
            print(json.dumps(k.as_dict(), indent=1))
            if a.stage:
                print(f"[x35 kit] K9(b) record {_B.k9b_record(a.stage, k)}")
            return {"PASS": 0, "VOID": 1}.get(k.verdict, 3)
        if a.cmd == "live":
            live = live_runs()
            for rd, pid in sorted(live.items()):
                print(f"LIVE {rd} pid {pid}")
            return 1 if live else 0
        from utils.paths import main_models_dir

        ok, msg = disk_check(a.models_dir or str(main_models_dir() or "."), a.required_gib, thinning=a.thinning,
                             stage=a.stage)
        print(f"[x35 kit] disk: {msg}")
        return 0 if ok else 1
    except (KitError, BatchError, RefPopError, FileNotFoundError, ValueError) as exc:
        print(f"[x35 kit] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
