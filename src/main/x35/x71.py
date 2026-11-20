"""X71, THE EXPLOITABILITY PROBE: its eight RESPONDERS as kit arms (PREREG
``designs/research_state/measurements/x71_exploitability_probe_2026-11-16/PREREG.md`` §2-§4; its §4.3 "OWED").

A responder is a fresh best responder of ONE target final, R0's / the G-E reader's argv (``main.x35.ge``) cut to +5M,
with the control recipe TYPED so every responder is the same learner whatever its target's record says:

    --model <T> --exploiter <T> --no-self-play --trainee-teams <the five offense teams>
    --fork-lr 5.6e-05 --fork-lr-freeze --behaviour-check warn --steps <T's step + 5,000,000>
    --checkpoint-every-steps 1000000 --device cuda --ent-coef 0.05 --magnet-kl-coef 0
    --seed <7111 | 7112> --run-name rb_x35_x71_<tag>_s<seed>

The registration's rule file (``x71_rule.py`` beside the PREREG) is a FROZEN record, never imported by a tool, so this
module restates its tables; ``x71_kit_test.py`` pins every one of them, and the argv token for token, to that file and
to the six argvs banked at the registration.

* **Arms** (:data:`X71_ARMS`): ``x71_c0`` / ``x71_x67`` / ``x71_swap`` / ``x71_x70``, one per target; the kit seed is
  the responder's TRAINING seed (7111, 7112); all at the CE-3d training pin (``batch.CE3D_PIN``, PREREG §3).
* **The target** (:func:`target`) is the final of the target arm's lowest registered seed. One that exists is judged
  in FULL: the three banked finals by their registered sha256 and step (the PREREG's G2), X70's by its length and its
  recorded recipe (``ent_coef`` 0, no magnet: G5). A missing final is a refusal.
* **X70's target does not exist yet.** For ``arm_chain plan`` ONLY (``kit describe --plan``) a missing X70 final
  becomes a DECLARED PLACEHOLDER (:func:`placeholder_target`: every value known now filled in -- the registered run,
  the path its final lands at, the step floor -- and its hash a declared marker), judged for shape only and marked
  ``TARGET-PLACEHOLDER`` in the describe. It is the pattern X35-A3's phase 2 used for its unfinished phase 1
  (``kit.describe_for_plan`` / ``kit.phase1_placeholder_record``, commit e2c4498fb): a placeholder is never a launch
  argv (:func:`build_x71_argv` refuses it without ``plan``), and once the final exists the same call judges it in full.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

X71_PHASE = 5
X71_PREREG = "designs/research_state/measurements/x71_exploitability_probe_2026-11-16/PREREG.md"
#: kit arm -> the PREREG's target tag
X71_ARMS: Dict[str, str] = {"x71_c0": "c0", "x71_x67": "x67", "x71_swap": "swap", "x71_x70": "x70"}
#: tag -> (the target's kit arm, the target run = that arm's LOWEST registered seed)  [x71_rule.TARGETS]
TARGETS: Dict[str, Tuple[str, str]] = {
    "c0": ("c0", "rb_x35_c0_s3501"),
    "x67": ("x67", "rb_x35_x67ent_s6701"),
    "swap": ("ce3d_swap", "rb_x35_ce3dswap_s7321"),
    "x70": ("x70", "rb_x35_x70ent0_s7331"),
}
#: the banked targets' finals at the registration: sha256 [x71_rule.TARGET_SHA256] and step [PREREG §2's table]; X70's
#: final does not exist yet (None: judged by its floor and its recorded recipe, its sha256 reported).
TARGET_SHA256: Dict[str, Optional[str]] = {
    "c0": "09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303",
    "x67": "94f173f943d4a930ea5a523f6874d6cec7cb90d88c7646b923c7829c2d79f128",
    "swap": "c068bb1a0df29afe3ebdaba4e15452c42df76020f02e1d48871ba68216100cb7",
    "x70": None,
}
TARGET_STEP: Dict[str, Optional[int]] = {"c0": 300_061_940, "x67": 300_021_355, "swap": 300_061_272, "x70": None}
#: the responders' training seeds, the same two for every target [x71_rule.SEEDS]
SEEDS = (7111, 7112)
#: a responder's length [x71_rule.L_RESP]
L_RESP = 5_000_000
#: all eight train at the CE-3d training pin [x71_rule.PIN = batch.CE3D_PIN]
PIN = "9e97cad5709da21a1f9d4d8846096c87eb17b1d0"
FORK_LR = "5.6e-05"
#: the learner's recipe, TYPED on every responder at production's values [x71_rule.RECIPE]
RECIPE = ("--ent-coef", "0.05", "--magnet-kl-coef", "0")
RUN_FMT = "rb_x35_x71_{tag}_s{{seed}}"
#: the declared marker a PLACEHOLDER target carries where a real one carries its final's sha256
PLACEHOLDER_SHA = "<sha256 of the target's final, judged by the kit when rb_x35_x70ent0_s7331 ends>"
PLACEHOLDER_MODE = "TARGET-PLACEHOLDER"


class X71Error(ValueError):
    pass


def tag_of(arm: str) -> str:
    if arm not in X71_ARMS:
        raise X71Error(f"arm {arm!r} is not an X71 responder arm ({sorted(X71_ARMS)})")
    return X71_ARMS[arm]


def run_format(arm: str) -> str:
    """The kit's run-name format of ``arm`` (``{seed}`` left to fill)."""
    return RUN_FMT.format(tag=tag_of(arm))


def target_run(arm: str) -> str:
    return TARGETS[tag_of(arm)][1]


def target_arm(arm: str) -> str:
    return TARGETS[tag_of(arm)][0]


def _final(arm: str, models_dir: str) -> Path:
    return Path(models_dir).resolve() / target_run(arm) / "final_model.zip"


def placeholder_target(arm: str, models_dir: str, floor: int) -> Dict[str, Any]:
    """The DECLARED PLACEHOLDER of a target whose final does not exist yet (X70's only): the registered run, the
    path its final lands at, the step floor as its step, the hash a declared marker. For planning only."""
    tag = tag_of(arm)
    if TARGET_SHA256[tag] is not None:
        raise X71Error(f"{arm}: its target {target_run(arm)} is BANKED (a registered sha256): it takes no placeholder")
    return {"run": target_run(arm), "path": str(_final(arm, models_dir)), "step": int(floor),
            "sha256": PLACEHOLDER_SHA, "standin": False, "placeholder": True}


def _recorded_recipe_problems(run_dir: Path) -> List[str]:
    """G5: X70's target is the ent-0 arm with no magnet, by its own record."""
    try:
        md = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{run_dir.name}: metadata.json is unreadable ({exc})"]
    probs = []
    ent = md.get("ent_coef")
    if ent is None or float(ent) != 0.0:
        probs.append(f"{run_dir.name}: recorded ent_coef {ent!r}, not 0 (G5)")
    if float(md.get("magnet_kl_coef") or 0.0) != 0.0:
        probs.append(f"{run_dir.name}: recorded magnet_kl_coef {md.get('magnet_kl_coef')!r}, not 0 or absent (G5)")
    return probs


def target(arm: str, models_dir: str, *, floor: int, plan: bool = False) -> Dict[str, Any]:
    """``arm``'s target: ``{run, path, step, sha256, standin, placeholder}``. ``floor`` is the step a finished target
    arm reaches at least (P*'s step + the target arm's length, from the call record). A final that exists is judged
    in full and a problem is a refusal. A missing one is a refusal too, except X70's under ``plan``
    (:func:`placeholder_target`)."""
    from main.x35.kit import KitError, _final_step
    from main.x35.refpop import sha256_file

    tag, run, z = tag_of(arm), target_run(arm), _final(arm, models_dir)
    if not z.is_file():
        if plan and TARGET_SHA256[tag] is None:
            return placeholder_target(arm, models_dir, floor)
        raise X71Error(f"{run}: no final_model.zip under {models_dir} (a responder forks a FINISHED target only"
                       + ("; `describe --plan` builds it from a declared placeholder, never a launch" if
                          TARGET_SHA256[tag] is None else "") + ")")
    try:
        step = _final_step(str(z))
    except KitError as exc:
        raise X71Error(str(exc)) from exc
    sha = sha256_file(z)
    probs: List[str] = []
    if step < floor:
        probs.append(f"{run}: final at {step:,} < {floor:,} (the arm did not reach its registered length)")
    if TARGET_SHA256[tag] is None:
        probs += _recorded_recipe_problems(z.parent)
    else:
        if sha != TARGET_SHA256[tag]:
            probs.append(f"{run}: final_model.zip has sha256 {sha[:12]}, not the registered "
                         f"{str(TARGET_SHA256[tag])[:12]} (PREREG G2: the file moved)")
        if step != TARGET_STEP[tag]:
            probs.append(f"{run}: final at {step:,}, not the registered {TARGET_STEP[tag]:,} (PREREG §2)")
    if probs:
        raise X71Error("; ".join(probs))
    return {"run": run, "path": str(z), "step": step, "sha256": sha, "standin": False, "placeholder": False}


def build_x71_argv(arm: str, seed: int, target: Mapping[str, Any], *, run_name: str, behaviour: str = "warn",
                   ckpt_every: str = "1000000", plan: bool = False) -> List[str]:
    """The trainer argv (the launcher's own flags excluded) of one responder. A PLACEHOLDER target is taken only
    with ``plan``: an argv built from it is never a launch."""
    from main.x35.ge import offense_teams

    tag_of(arm)
    if seed not in SEEDS:
        raise X71Error(f"seed {seed} is not registered for {arm} ({list(SEEDS)})")
    if target.get("run") != target_run(arm):
        raise X71Error(f"{arm}: its target is {target_run(arm)}, not {target.get('run')!r}")
    if target.get("placeholder") and not plan:
        raise X71Error(f"{arm}: a PLACEHOLDER target is for planning only, never a launch argv "
                       f"({target_run(arm)} has no final yet)")
    f = str(target["path"])
    return ["--model", f, "--exploiter", f, "--no-self-play", "--trainee-teams", offense_teams(),
            "--fork-lr", FORK_LR, "--fork-lr-freeze", "--behaviour-check", behaviour,
            "--steps", str(int(target["step"]) + L_RESP), "--checkpoint-every-steps", ckpt_every,
            "--device", "cuda", *RECIPE, "--seed", str(int(seed)), "--run-name", run_name]
