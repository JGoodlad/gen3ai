"""BOUNDARY-1's ARMS: twelve fresh production runs, trunk width x team breadth, three seeds a cell (PREREG §2-§4).

    python -m main.b1.arms describe --pin <sha>                      # the launch block: phase 1, then phase 2
    python -m main.b1.arms describe --pin <sha> --block fallback     # every launch straight to 100M (no stop at 50M)
    python -m main.b1.arms describe --pin <sha> --arm wb --seed 5107 # one launch argv (--phase 2: its resume argv)
    python -m main.b1.arms check --tree <a checkout of the pin> --pin <sha>   # seeds / names, corpus, levers; --params

THE PIN is not in this file: a tree cannot name the commit that holds it. It is registered in the PREREG's §3 and every
entry point takes it (``--pin``; a call without one is refused).

THE BASE is a FRESH ``--arch production`` launch (``designs/production_config.json`` supplies the architecture and
the recipe; ten epochs) plus :data:`BASE`, the two knobs EVERY run types (PREREG §18 Amendment 2): ``--ent-coef
0.0125`` (the owner's choice for this experiment after X67; the production default stays 0.05 and is not this
kit's to move) and ``--behaviour-check warn`` (a 100M fresh run under ``fatal`` can stop on a K9(b) threshold
crossing; the judged read is ``main.b1.look.k9b_read`` at every look). The tail every arm shares is the launcher's
6 h restart, the pin, a checkpoint every 1,000,000 steps (the 50M / 75M / 100M looks and the equal-compute read
need that series), the step target, the seed, the device, the log level and the run name.

THE ARMS (each one lever from the control; WB carries both):

* ``c``  -- control: the base and nothing else. Width 128, both sides draw from the usual team pool.
* ``w``  -- wide: :data:`WIDE_LEVER`, the trunk at 256 with its base FFN (2 x the width; a lower value is refused) and
  the consent a non-production architecture needs.
* ``br`` -- breadth: :data:`BREADTH_LEVER`, on each side every fresh team draw is a ``n1_proc_v1`` team with
  probability 0.5, else the side's usual pool draw (``gen3_team_mix_v1``). Eval stays on the pool.
* ``wb`` -- wide + breadth: both levers.

THE SEEDS are one per run, so no two runs share dice: 5101-5108 are each arm's first two (the original
registration's) and 5110-5113 each arm's third (Amendment 1: three runs an arm from the start). :data:`SPARE_SEED`
(5109) is the replacement rule's: the replaced run's own arm takes it, from step 0.

THE SCHEDULE is two phases in :data:`LAUNCH_ORDER`: every run to 50,000,000 steps, the 50M look, then every run
RESUMED in its own folder to 100,000,000 (:func:`resume_argv`, the form ``main.b1.resume_check`` exercises). The
fallback block launches each run straight to 100M in the same order.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.era import prefixed

CORPUS = "n1_proc_v1"
CORPUS_TIER = "ALL"
BREADTH_SHARE = "0.5"
#: The corpus the breadth arms draw from, as it verifies at the pin (``check`` refuses a tree that differs).
CORPUS_IDENT: Dict[str, Any] = {
    "corpus": CORPUS, "tier": CORPUS_TIER, "n": 12000, "schema": "gen3_team_corpus_v1",
    "data_sha256": "99e3425aa687e7d0663525e4ab1df2a3cb52c8d6bf162df35da0bccf7c20bad6",
    "tier_sha256": "d59f44f0b33bd3532a2434b4b9cc4d77da6b871e4a642dd71a9b38d3d875314f",
    "manifest_sha256": "4db822ba1e0cf2e21d3ab7780ab42017ae3b02b15b3687661961b1c8e191e067",
}

#: What EVERY run types beside ``--arch production`` (PREREG §18 Amendment 2): the experiment's entropy coefficient
#: and the K9(b) mode. A base knob, never a lever: the control carries it too, so no arm differs from it in either.
ENT_COEF = "0.0125"
BEHAVIOUR_CHECK = "warn"
BASE: Tuple[str, ...] = ("--ent-coef", ENT_COEF, "--behaviour-check", BEHAVIOUR_CHECK)
WIDE_LEVER: Tuple[str, ...] = ("--d-model", "256", "--trunk-ffn-dim", "512", "--allow-nonproduction-arch")
BREADTH_LEVER: Tuple[str, ...] = ("--team-mix", f"{CORPUS}@{CORPUS_TIER}:{BREADTH_SHARE}", "--team-mix-sides", "both")
#: arm -> the tokens it types beyond the control's.
ARMS: Dict[str, Tuple[str, ...]] = {"c": (), "w": WIDE_LEVER, "br": BREADTH_LEVER, "wb": WIDE_LEVER + BREADTH_LEVER}
ARM_LABELS: Dict[str, str] = {"c": "C control", "w": "W wide", "br": "Br breadth", "wb": "WB wide + breadth"}
#: The resolved-config keys each arm may differ from the control in (``check`` and the tests compare the namespaces).
LEVER_DESTS: Dict[str, Tuple[str, ...]] = {
    "c": (), "w": ("allow_nonproduction_arch", "d_model", "trunk_ffn_dim"), "br": ("team_mix", "team_mix_sides"),
    "wb": ("allow_nonproduction_arch", "d_model", "team_mix", "team_mix_sides", "trunk_ffn_dim")}
SEEDS: Dict[str, Tuple[int, ...]] = {
    "c": (5101, 5102, 5110), "w": (5103, 5104, 5111), "br": (5105, 5106, 5112), "wb": (5107, 5108, 5113)}
#: The replacement rule's seed: the replaced run's own arm takes it, from step 0.
SPARE_SEED = 5109
#: The order of launches within a phase: one seed of every arm before any arm's next (round k = every arm's k-th
#: seed, the arms in :data:`ARMS` order), so a campaign cut short still holds whole rounds.
LAUNCH_ORDER: Tuple[Tuple[str, int], ...] = tuple(
    (arm, SEEDS[arm][k]) for k in range(len(SEEDS["c"])) for arm in ARMS)
#: The step target of phase 1 and of phase 2.
PHASE_STEPS: Tuple[int, int] = (50_000_000, 100_000_000)
CHECKPOINT_EVERY = 1_000_000
RESTART_HOURS = "6"
#: Fresh production learners at width 128 / 256 (``designs/ARCHITECTURE.md``, MEASURED on CPU 2026-11-09).
EXPECTED_PARAMS: Dict[str, int] = {"c": 2_609_183, "w": 6_258_079, "br": 2_609_183, "wb": 6_258_079}
LAUNCHER = "python -m main.launcher"


class ArmError(ValueError):
    """An unknown arm, seed or step target, a missing pin, or a failed check."""


def _pin(pin: Optional[str]) -> str:
    """The pin as given: a full sha. The kit carries no default (see the module docstring)."""
    if not pin:
        raise ArmError("the pin is registered in PREREG.md §3; pass --pin")
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ArmError(f"the pin must be a full 40-hex commit sha, not {pin!r}")
    return pin


def run_name(arm: str, seed: int) -> str:
    """``rb_b1<arm>_s<seed>``: the era prefix, the arm and the seed (one of its own, or the spare)."""
    if arm not in ARMS:
        raise ArmError(f"unknown arm {arm!r} (known: {sorted(ARMS)})")
    if seed not in SEEDS[arm] and seed != SPARE_SEED:
        raise ArmError(f"seed {seed} is not arm {arm!r}'s (its seeds: {SEEDS[arm]}; the spare: {SPARE_SEED})")
    return prefixed(f"b1{arm}_s{seed}")


def build_argv(arm: str, seed: int, pin: Optional[str] = None, steps: int = PHASE_STEPS[0]) -> List[str]:
    """The FRESH launcher argv of one run (after ``python -m main.launcher``) to the step target ``steps``."""
    name = run_name(arm, seed)
    if steps not in PHASE_STEPS:
        raise ArmError(f"--steps {steps} is not a registered target ({PHASE_STEPS})")
    return ["--restart-interval-hours", RESTART_HOURS, "--pin-commit", _pin(pin), "--arch", "production", *BASE,
            *ARMS[arm], "--checkpoint-every-steps", str(CHECKPOINT_EVERY), "--steps", str(steps), "--seed", str(seed),
            "--device", "cuda", "--log-level", "periodic", "--run-name", name]


def resume_argv(arm: str, seed: int, pin: Optional[str] = None, steps: int = PHASE_STEPS[1]) -> List[str]:
    """The PHASE-2 launcher argv: continue the run in its OWN folder to a higher target.

    A ``--model`` with no ``--run-name`` is a same-run resume (a ``--run-name`` would make it a fork); ``--arch`` is
    fresh-only and refused beside ``--model``. Everything the argv does not type (the seed, the levers, the checkpoint
    cadence, the recipe, :data:`BASE`'s two knobs) is restored from the run's own record. ``--pin-commit`` says the pin again: a same-run resume
    refuses a pin that differs from the checkpoint's, so this line cannot move the run onto other code."""
    name = run_name(arm, seed)
    if steps != PHASE_STEPS[1]:
        raise ArmError(f"the resume target is {PHASE_STEPS[1]}, not {steps}")
    return ["--restart-interval-hours", RESTART_HOURS, "--pin-commit", _pin(pin),
            "--model", f"models/{name}/final_model.zip", "--steps", str(steps),
            "--device", "cuda", "--log-level", "periodic"]


def _without(tokens: Sequence[str], flags: Sequence[str]) -> List[str]:
    out, skip = [], False
    for t in tokens:
        if skip:
            skip = False
        elif t in flags:
            skip = True
        else:
            out.append(t)
    return out


def lever_diff(arm: str, pin: Optional[str] = None, steps: int = PHASE_STEPS[0]) -> Tuple[str, ...]:
    """The tokens ``arm``'s argv holds beyond the control's, once each run's own seed and name are set aside.
    Raises unless the control's remaining tokens are all in the arm's, in order."""
    own = ("--seed", "--run-name")
    a = _without(build_argv(arm, SEEDS[arm][0], pin, steps), own)
    c = _without(build_argv("c", SEEDS["c"][0], pin, steps), own)
    extra, rest = [], list(c)
    for t in a:
        if rest and t == rest[0]:
            rest.pop(0)
        else:
            extra.append(t)
    if rest:
        raise ArmError(f"arm {arm!r} lacks the control's tokens {rest}")
    return tuple(extra)


def resolved_namespace(argv: Sequence[str]) -> Dict[str, Any]:
    """The training namespace a launcher argv resolves to: the trainer's own parser, then ``resolve_config``
    (what the launch runs). Private bookkeeping keys (``_explicit_flags`` ...) are dropped."""
    import contextlib
    import io

    from main import checkargs
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    ns = parser.parse_args(checkargs.forwarded_argv(list(argv)))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(ns, parser)
    return {k: v for k, v in vars(ns).items() if not k.startswith("_")}


def namespace_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """The keys two resolved namespaces differ in, sorted."""
    return sorted(k for k in set(a) | set(b) if repr(a.get(k)) != repr(b.get(k)))


def check_levers(pin: Optional[str] = None) -> List[str]:
    """Each arm one lever from the control, in the argv AND in the resolved config, and the control resolving
    :data:`BASE` as typed (so every arm does); ``[]`` when all hold."""
    bad: List[str] = []
    own = ["run_name", "seed"]
    for steps in PHASE_STEPS:
        base = resolved_namespace(build_argv("c", SEEDS["c"][0], pin, steps))
        got_base = (base.get("ent_coef"), base.get("behaviour_check"))
        if got_base != (float(ENT_COEF), BEHAVIOUR_CHECK):
            bad.append(f"c @ {steps}: the base resolves to (ent_coef, behaviour_check) = {got_base}, "
                       f"not ({float(ENT_COEF)}, {BEHAVIOUR_CHECK!r})")
        for arm, lever in ARMS.items():
            if lever_diff(arm, pin, steps) != lever:
                bad.append(f"{arm} @ {steps}: argv lever {lever_diff(arm, pin, steps)} != {lever}")
            got = [k for k in namespace_diff(base, resolved_namespace(build_argv(arm, SEEDS[arm][0], pin, steps)))
                   if k not in own]
            if got != sorted(LEVER_DESTS[arm]):
                bad.append(f"{arm} @ {steps}: resolved config differs from the control's in {got}, "
                           f"not {sorted(LEVER_DESTS[arm])}")
    return bad


# ---------------------------------------------------------------------------------------------------------------
# The seed / run-name collision scan
# ---------------------------------------------------------------------------------------------------------------

def all_seeds() -> Tuple[int, ...]:
    return tuple(sorted({s for ss in SEEDS.values() for s in ss} | {SPARE_SEED}))


def all_run_names() -> Tuple[str, ...]:
    """The registered names in launch order, then each arm's replacement name."""
    return tuple([run_name(a, s) for a, s in LAUNCH_ORDER] + [run_name(a, SPARE_SEED) for a in ARMS])


def scan_roots() -> List[Path]:
    """Where runs live: the main checkout's ``models/`` and the archive of finished read states."""
    from utils.paths import gen3ai_archive_dir, main_models_dir

    roots = [main_models_dir(), gen3ai_archive_dir()]
    return [r for r in roots if r is not None and r.is_dir()]


_SIDECAR_GLOBS = ("*/metadata.json", "*/*/metadata.json", "*/*/*/metadata.json")
_DIR_GLOBS = ("*", "*/*")


def registered_run(run_dir: Path, pin: Optional[str]) -> bool:
    """Is ``run_dir`` THE registered run of its name: a folder named as one of ours whose immutable
    ``original_command`` is, token for token, that run's fresh argv at ``pin`` (to either registered target) and whose
    recorded seed is the name's? Anything unreadable or different is not."""
    by_name = {run_name(a, s): (a, s) for a in ARMS for s in (*SEEDS[a], SPARE_SEED)}
    if not pin or run_dir.name not in by_name:
        return False
    arm, seed = by_name[run_dir.name]
    try:
        doc = json.loads((run_dir / "metadata.json").read_text())
        got = shlex.split(doc["original_command"])[1:]
        return doc["cli_args"]["seed"] == seed and got in [build_argv(arm, seed, pin, n) for n in PHASE_STEPS]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def scan_collisions(roots: Sequence[Path], pin: Optional[str] = None) -> Dict[str, Any]:
    """Every ``metadata.json`` up to three directories below each root, and every directory name up to two below.

    A HIT is a sidecar whose ``cli_args.seed`` is one of ours, a sidecar that cannot be read (its seed cannot be
    verified), a directory named as one of our runs, or a directory whose name carries one of our seeds
    (``..._s5103``). With ``pin``, a folder that IS the registered run of its name (:func:`registered_run`) is the
    experiment's OWN: listed under ``own``, never a hit, so the scan can be re-run after the first launch. A foreign
    run under one of our names or seeds stays a hit. Sorted, so the same tree gives the same answer."""
    seeds, names = set(all_seeds()), set(all_run_names())
    own: List[str] = []
    seed_in_name = re.compile(r"_s(%s)(?!\d)" % "|".join(str(s) for s in sorted(seeds)))
    hits: List[str] = []
    n_sidecars = n_dirs = 0
    for root in roots:
        for pat in _SIDECAR_GLOBS:
            for p in sorted(root.glob(pat)):
                n_sidecars += 1
                try:
                    doc = json.loads(p.read_text())
                except (OSError, ValueError) as e:
                    hits.append(f"UNREADABLE sidecar {p}: {type(e).__name__}")
                    continue
                if registered_run(p.parent, pin):
                    continue
                cli = doc.get("cli_args") if isinstance(doc, dict) else None
                seed = cli.get("seed") if isinstance(cli, dict) else None
                if isinstance(seed, int) and not isinstance(seed, bool) and seed in seeds:
                    hits.append(f"SEED {seed} is taken: {p}")
        for pat in _DIR_GLOBS:
            for p in sorted(root.glob(pat)):
                if not p.is_dir():
                    continue
                n_dirs += 1
                if registered_run(p, pin):
                    own.append(str(p))
                elif p.name in names:
                    hits.append(f"RUN NAME {p.name} exists: {p}")
                elif seed_in_name.search(p.name):
                    hits.append(f"SEED in a run name: {p}")
    return {"roots": [str(r) for r in roots], "sidecars": n_sidecars, "dirs": n_dirs, "hits": hits, "own": own}


# ---------------------------------------------------------------------------------------------------------------
# The corpus at the pin, and the parameter counts
# ---------------------------------------------------------------------------------------------------------------

def tree_head(tree: str) -> Optional[str]:
    r = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def check_corpus_at(tree: str) -> Dict[str, Any]:
    """Load ``n1_proc_v1`` from ``tree``'s own ``utils.team_corpora`` and refuse an identity that is not the
    registered one (:data:`CORPUS_IDENT`)."""
    from main.x61 import arms as x61

    if (x61.CORPUS, x61.CORPUS_TIER) != (CORPUS, CORPUS_TIER):
        raise ArmError(f"main.x61.arms checks {x61.CORPUS}@{x61.CORPUS_TIER}, not {CORPUS}@{CORPUS_TIER}")
    try:
        ident = dict(x61.check_corpus_at(tree))
    except x61.ArmError as e:
        raise ArmError(str(e)) from e
    wrong = {k: (ident.get(k), v) for k, v in CORPUS_IDENT.items() if ident.get(k) != v}
    if wrong:
        raise ArmError(f"{CORPUS}@{CORPUS_TIER} at {tree} is not the registered corpus (got, registered): {wrong}")
    return ident


def param_counts(pin: Optional[str] = None) -> Dict[str, int]:
    """Parameters of each arm's FRESH learner, built on CPU from the arm's own resolved config (a few seconds)."""
    import argparse as _ap

    from main.fresh_checkpoint import build_fresh_model

    out: Dict[str, int] = {}
    for arm in ARMS:
        ns = _ap.Namespace(**resolved_namespace(build_argv(arm, SEEDS[arm][0], pin)))
        model, _args, _pk = build_fresh_model(0, args=ns)
        out[arm] = sum(int(p.numel()) for p in model.policy.parameters())
    return out


# ---------------------------------------------------------------------------------------------------------------
# The launch block
# ---------------------------------------------------------------------------------------------------------------

def arms_table() -> List[str]:
    """arm, its runs, the lever it types (last: the WB lever is long)."""
    runs = {arm: ", ".join(run_name(arm, s) for s in SEEDS[arm]) for arm in ARMS}
    w = max(len(v) for v in runs.values())
    rows = [f"{'arm':<18} {'runs':<{w}}  lever typed"]
    for arm, lever in ARMS.items():
        rows.append(f"{ARM_LABELS[arm]:<18} {runs[arm]:<{w}}  {shlex.join(lever) or 'nothing beyond the base'}")
    return rows


def _commands(title: str, argvs: Sequence[Tuple[str, int, List[str]]]) -> List[str]:
    out = ["", title]
    for i, (arm, seed, argv) in enumerate(argvs, 1):
        out += [f"# [{i}] {run_name(arm, seed)}  ({ARM_LABELS[arm]}, seed {seed})",
                f"{LAUNCHER} {shlex.join(argv)}",
                f"#     dry run: {LAUNCHER} --dry-run {shlex.join(argv)}"]
    return out


def launch_block(pin: Optional[str] = None, block: str = "phased") -> str:
    """The whole launch block as text: ``phased`` (phase 1, the look, phase 2) or ``fallback`` (straight to 100M)."""
    pin = _pin(pin)
    p1, p2 = PHASE_STEPS
    head = [f"# BOUNDARY-1 launch block ({block}): python -m main.b1.arms describe --pin {pin}"
            + ("" if block == "phased" else f" --block {block}"),
            f"# pin {pin}; run every command from the MAIN checkout; one run at a time, in this order",
            f"# every run types the base {shlex.join(BASE)} (PREREG §18 Amendment 2), then its arm's lever",
            "#", *("# " + r for r in arms_table()),
            f"# spare seed {SPARE_SEED}: a replaced run's own arm takes it, from step 0 "
            f"(names {', '.join(run_name(a, SPARE_SEED) for a in ARMS)})"]
    if block == "fallback":
        return "\n".join(head + _commands(
            f"# FALLBACK: {len(LAUNCH_ORDER)} FRESH launches straight to {p2:,} steps (the {p1:,} look reads each run's first "
            f"checkpoint at or after {p1:,})",
            [(a, s, build_argv(a, s, pin, p2)) for a, s in LAUNCH_ORDER])) + "\n"
    if block != "phased":
        raise ArmError(f"unknown block {block!r} (phased, fallback)")
    body = _commands(f"# PHASE 1: {len(LAUNCH_ORDER)} FRESH launches to {p1:,} steps",
                     [(a, s, build_argv(a, s, pin, p1)) for a, s in LAUNCH_ORDER])
    body += ["", f"# THE {p1:,} LOOK (the harm stop) is read here, before any phase-2 command."]
    body += _commands(f"# PHASE 2: each run RESUMED in its own folder to {p2:,} steps (no --run-name: a resume, not a "
                      f"fork; no --arch: fresh-only)",
                      [(a, s, resume_argv(a, s, pin, p2)) for a, s in LAUNCH_ORDER])
    return "\n".join(head + body) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------

def _check(tree: str, pin: Optional[str], params: bool, roots: Optional[Sequence[Path]] = None) -> int:
    pin = _pin(pin)
    failed = 0
    scan = scan_collisions(scan_roots() if roots is None else roots, pin)
    print(f"(a) seeds {all_seeds()[0]}-{all_seeds()[-1]} and {len(all_run_names())} run names: scanned "
          f"{scan['sidecars']} metadata.json and {scan['dirs']} directory names under {', '.join(scan['roots'])}")
    for o in scan["own"]:
        print(f"    own: {o} is the registered run of its name (original_command = the registered argv at the pin)")
    for h in scan["hits"]:
        print(f"    REFUSED: {h}")
    failed += bool(scan["hits"])
    print("    " + ("FAIL" if scan["hits"] else "ok: no seed and no run name is taken by a run that is not ours"
                    + (f" ({len(scan['own'])} of our own exist)" if scan["own"] else "")))

    head = tree_head(tree)
    print(f"(b) corpus {CORPUS}@{CORPUS_TIER} at {tree} (HEAD {head})")
    if head != pin:
        print(f"    FAIL: the tree's HEAD is not the pin {pin}")
        failed += 1
    try:
        print("    ok: " + json.dumps(check_corpus_at(tree), sort_keys=True))
    except ArmError as e:
        print(f"    FAIL: {e}")
        failed += 1

    bad = check_levers(pin)
    print(f"(c) the control resolves the base ({shlex.join(BASE)}) and each arm's argv and resolved config differ "
          f"from the control's by its lever alone (at {PHASE_STEPS[0]:,} and {PHASE_STEPS[1]:,} steps)")
    for b in bad:
        print(f"    FAIL: {b}")
    failed += bool(bad)
    if not bad:
        for arm in ARMS:
            print(f"    ok: {arm:<3} {shlex.join(ARMS[arm]) or '(nothing)'}")

    if params:
        got = param_counts(pin)
        print("(d) parameters of each arm's fresh learner (CPU build)")
        for arm, n in got.items():
            ok = n == EXPECTED_PARAMS[arm]
            print(f"    {'ok' if ok else 'FAIL'}: {arm:<3} {n:,}" + ("" if ok else f" (registered {EXPECTED_PARAMS[arm]:,})"))
            failed += not ok
    else:
        print("(d) parameter counts: not run (pass --params; a CPU build of four learners)")
    print("CHECK " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.arms", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="the launch block, or one argv with --arm and --seed")
    d.add_argument("--block", choices=("phased", "fallback"), default="phased")
    d.add_argument("--arm", choices=sorted(ARMS))
    d.add_argument("--seed", type=int)
    d.add_argument("--phase", type=int, choices=(1, 2), default=1, help="with --arm: 1 the fresh argv, 2 the resume")
    d.add_argument("--steps", type=int, help="with --arm --phase 1: the step target (default: phase 1's)")
    d.add_argument("--pin", required=True, help="the registered pin (PREREG.md §3): a full sha")
    c = sub.add_parser("check", help="seeds and names free, the corpus at the pin, the levers; --params counts")
    c.add_argument("--tree", required=True, help="a checkout of the pin")
    c.add_argument("--pin", required=True, help="the registered pin (PREREG.md §3); the tree's HEAD must be it")
    c.add_argument("--params", action="store_true", help="also build each arm's fresh learner on CPU and count it")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            return _check(a.tree, a.pin, a.params)
        if a.arm is None and a.seed is None:
            sys.stdout.write(launch_block(a.pin, a.block))
        elif a.arm is None or a.seed is None:
            raise ArmError("--arm and --seed go together")
        elif a.phase == 2:
            print(shlex.join(resume_argv(a.arm, a.seed, a.pin)))
        else:
            print(shlex.join(build_argv(a.arm, a.seed, a.pin, a.steps or PHASE_STEPS[0])))
    except ArmError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
