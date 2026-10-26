"""X61's ARMS: the toy recipe from step 0, with and without N1's procedural corpus (the PREREG's §2).

    python -m main.x61.arms describe --arm toy_n1 --seed 6101        # the launch argv, one line
    python -m main.x61.arms check --tree <a checkout of the pin>      # the corpus verifies at the pin's tree

THE TOY RECIPE is ``rb_toy_ent_s4002``'s ``original_command`` (the T-ENT toy, 1.42M parameters, the run behind T*)
with its run-specific tokens (``--steps``, ``--seed``, ``--run-name``) re-typed and the 17 flags the 2026-10-15 legacy
deletions removed dropped: every one of them was the OFF value of a damage-operator family (``--no-damage-op`` makes each
vacuous; ``designs/deleted_flags.md`` rows U1 / U3), so the dropped tokens change no module the toy builds. The toy's
launcher flags are the fresh-run defaults (``--restart-interval-hours 6``) plus ``--pin-commit`` = :data:`PIN`, and
:data:`TOY_EXTRAS` adds C3's ``--behaviour-check warn`` (the toy arms' K9(b) convention).

THE ARMS (one lever apart, the same seed per pair, so the dice match):

* ``toy_c0f`` -- the FRESH toy C0: the recipe, the pool only. The earlier toy C0 (X35's ``rb_x35t_c0_*`` at
  ``4609c155``, forks of T* with fp64 tau) is NOT this control: a different pin, a fork, a different step range.
* ``toy_n1`` -- the recipe + :data:`N1_LEVER`: on each side, every fresh team draw is a ``n1_proc_v1`` team with
  probability 0.7, else the side's usual pool draw (``gen3_team_mix_v1``), so the pool keeps a 0.3 minority share,
  the survey's design ("the pool is kept at a minority share (e.g. 0.3)"). Eval stays on the pool.

N4 (a WIDER network from step 0) is NOT an arm: the trunk width is the architecture constant ``D_MODEL``
(``agents/model/arch_constants.py``), with no flag; a width flag is its own build (PREREG FINDING).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: The pin both arms launch at (tag ``x61-n1-pin-de269480`` on origin): the commit that landed the corpus, so the
#: trainer's ``utils.team_corpora`` at the pin holds ``n1_proc_v1``.
PIN: Optional[str] = "de2694804a67086be56c7ae7d8f827f09ec604fa"
PIN_TAG = "x61-n1-pin-de269480"
CORPUS = "n1_proc_v1"
CORPUS_TIER = "ALL"
N1_SHARE = "0.7"
N1_LEVER: Tuple[str, ...] = ("--team-mix", f"{CORPUS}@{CORPUS_TIER}:{N1_SHARE}", "--team-mix-sides", "both")
STEPS = 100_000_000
#: Look 1 runs seed 6101 on both arms; look 2 (on the PREREG's CONTINUE) adds 6102 on both.
SEEDS = (6101, 6102)
LOOK1_SEEDS = (6101,)
ARMS = ("toy_c0f", "toy_n1")
#: ``rb_toy_ent_s4002``'s ``original_command`` after its launcher flags, the deleted tokens dropped and the
#: run-specific ones (``--steps``, ``--seed``, ``--run-name``) removed; ``arms_test.py`` re-derives it.
TOY_RECIPE: Tuple[str, ...] = tuple(shlex.split(
    "--arch production --allow-nonproduction-arch --unified-moves off --no-damage-op --no-damage-outgoing "
    "--damage-matrices off --damage-topk 0 --move-belief-mode off --no-move-prior-fusion --no-move-latent "
    "--opp-belief-cls-k 0 --no-item-belief --no-spread-belief --no-spread-belief-nature --no-t0-species-prior "
    "--no-species-prior-fusion --entity-topk-seats 0 --no-entity-tail-seats --edge-bias-families off "
    "--no-history-events --no-value-entity-pool --no-value-entity-pool-full --mon-hazard-cost off --trunk-layers 2 "
    "--move-set-closure off --no-op-drop-renders --no-op-believed-lean --opp-belief-aux-coef 0 --opp-intent-coef 0 "
    "--move-belief-coef 0 --move-belief-latent-coef 0 --spread-belief-coef 0 --item-belief-coef 0 "
    "--hp-type-belief-coef 0 --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 --device cuda "
    "--supply-starve-cycles self_play_pool=0"))
#: C3 (orchestrator, 2026-10-19): toy arms run K9(b) as ``warn`` under X34's hand-read ceilings
#: (``main.x35.kit.K9B_TOY_CEILINGS``; the PREREG's V-K9). Both arms carry it, so it is no part of the contrast.
TOY_EXTRAS: Tuple[str, ...] = ("--behaviour-check", "warn")
#: The toy run whose command the recipe is (``arms_test.py`` re-derives :data:`TOY_RECIPE` from its command, holding the
#: list of the deleted flags: a bare deleted-flag literal in a module here would read as live CLI surface to the
#: freshness gate).
TOY_SOURCE_RUN = "rb_toy_ent_s4002"


class ArmError(ValueError):
    """An unknown arm or seed, or a launch asked for before the pin exists."""


def run_name(arm: str, seed: int) -> str:
    return f"rb_x61t_{arm[len('toy_'):]}_s{seed}"


def build_argv(arm: str, seed: int, pin: Optional[str] = None) -> List[str]:
    """The arm's full launcher argv (``python -m main.launcher <argv>``)."""
    if arm not in ARMS:
        raise ArmError(f"unknown X61 arm {arm!r}; arms: {', '.join(ARMS)}")
    if seed not in SEEDS:
        raise ArmError(f"seed {seed} is not registered for X61 (seeds {SEEDS})")
    pin = pin or PIN
    if not pin:
        raise ArmError("X61's pin is not written yet (main.x61.arms.PIN)")
    argv = ["--restart-interval-hours", "6", "--pin-commit", pin, *TOY_RECIPE, *TOY_EXTRAS,
            "--steps", str(STEPS), "--seed", str(seed), "--run-name", run_name(arm, seed)]
    if arm == "toy_n1":
        argv += list(N1_LEVER)
    return argv


def lever_diff(seed: int, pin: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """``(tokens only in toy_n1, tokens only in toy_c0f)`` after dropping the run name: one lever, nothing else."""
    a = build_argv("toy_n1", seed, pin)
    b = build_argv("toy_c0f", seed, pin)
    a = [t for t in a if t != run_name("toy_n1", seed)]
    b = [t for t in b if t != run_name("toy_c0f", seed)]
    return [t for t in a if t not in b], [t for t in b if t not in a]


def check_corpus_at(tree: str) -> Dict[str, object]:
    """Load and verify ``n1_proc_v1@ALL`` from the ``utils.team_corpora`` package of the source tree ``tree`` (a pin's
    worktree): what the trainer at that pin will load. Raises on any mismatch."""
    import importlib.util
    from pathlib import Path

    init = Path(tree) / "src" / "utils" / "team_corpora" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_x61_pin_team_corpora", init)
    if spec is None or spec.loader is None:
        raise ArmError(f"{init}: not a team_corpora package")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # a dataclass resolves its module through sys.modules
    try:
        spec.loader.exec_module(mod)
        c = mod.load_dir(Path(init).parent / CORPUS, CORPUS_TIER, CORPUS)
    finally:
        sys.modules.pop(spec.name, None)
    if len(c) < 10_000:
        raise ArmError(f"{CORPUS}@{CORPUS_TIER} at {tree} holds {len(c)} teams; N1 needs >= 10,000")
    return c.ident()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X61's arm argvs (never launches)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--arm", required=True, choices=ARMS)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--pin", default=None)
    c = sub.add_parser("check")
    c.add_argument("--tree", required=True, help="a checkout of the pin (git worktree add --detach <dir> <pin>)")
    a = ap.parse_args(argv)
    if a.cmd == "describe":
        print(shlex.join(build_argv(a.arm, a.seed, a.pin)))
        return 0
    import json

    print(json.dumps(check_corpus_at(a.tree), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
