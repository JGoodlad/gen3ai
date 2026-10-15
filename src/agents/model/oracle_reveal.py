"""The ORACLE REVEAL's Python side: the spec reader, the recorded-mode reader and the refusal an offline tool takes.

`--oracle-reveal <spec>` is a DIAGNOSTIC observation mode: the Rust encoder writes graded facts of the opponent's TRUE
team into the opponent block of the observation (`src/rust_sim/src/encoder/oracle.rs`). Since the information-lever
spec (`designs/endstate/design_information_levers.md` §6) it takes ANY spec the training observation honours
(`agents.info_levers.consumers.OBS`): a preset (`ki_moves`, …), a level (`off` / `species` / `full`, X32,
`designs/endstate/design_x5_belief_tokens.md` §7.6) or a clause list. What a run RECORDS (`model_config.json`'s
`oracle_reveal`, the env spec's JSON) is the spec's LABEL — the level name for the three X32 levels, so a run that
recorded one reads exactly as before, else the canonical clause text; `metadata.json` carries the full
`InfoReveal.record()` block (label, canonical text, hash, levers).

A checkpoint trained under a reveal is a different function of the observation than a production one, and the
reveal is made by the ENV CORE, so a tool that plays or reads a checkpoint through any other observation source (the
websocket client `main.play`, the Lane S bank the prober re-encodes, `main.anchors`) would feed it observations it
never trained on — silently, with every number still printing. Those tools therefore REFUSE a recorded non-`off` mode,
naming it: :func:`refuse_if_revealed`.

`main.h2h` is the one offline tool that PLAYS an oracle checkpoint: its eval cores take a PER-SIDE reveal
(`main.h2h.reveal`; the one-sided clairvoyance and both-sided cells of the X5 A/B, §7.7(a)), and it refuses an oracle
checkpoint under its `off` mode itself (`main.h2h.reveal.RevealModeError`); its `withheld` mode is the one declared
exception, playing the oracle side at `off` BELOW its recorded level (the honest-play read of a knock-in fork).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from utils.rust_env.protocol import ORACLE_REVEAL_LEVELS, oracle_reveal_label

#: The X32 level names, `off` first (the production mode). The flag takes any spec (:func:`label`); these three stay
#: the labels of their specs.
ORACLE_REVEAL_MODES = ORACLE_REVEAL_LEVELS


class OracleRevealRefused(ValueError):
    """An offline tool was handed a checkpoint trained under a diagnostic observation mode it cannot reproduce."""


def label(spec: str) -> str:
    """The LABEL of an `--oracle-reveal` spec text, validated against the training observation's support (a lever it
    cannot honour is refused BY NAME, ``ValueError``): what the run records and the Rust core parses."""
    return oracle_reveal_label(spec)


def same(a: str, b: str) -> bool:
    """Do two spec texts MEAN the same reveal (``full`` and its clause list do)?"""
    from agents.info_levers.spec import same as _same

    return _same(a, b)


def record(spec: str) -> Dict[str, Any]:
    """The spec's ``InfoReveal.record()`` block (label, canonical text, hash, levers): what ``metadata.json`` and a
    measurement record beside the label."""
    from agents.info_levers.consumers import OBS

    return OBS.validate(spec).record()


def describe(spec: str) -> str:
    """One human line: what the observation is told under ``spec`` (for the launch banner and ``checkargs``)."""
    from agents.info_levers.consumers import OBS

    s = OBS.validate(spec)
    if s.is_off:
        return "nothing (the production observation)"
    parts = []
    names = {"species": "SPECIES of the never-seen mons (a tail slot each)", "moves": "MOVES", "item": "ITEM",
             "ability": "ABILITY", "spread": "SPREAD (nature, EVs, IVs)"}
    for lv in s.levers:
        who = "of the seen mons" if lv.part == "seen" else "of the never-seen mons"
        what = names[lv.cls] if lv.cls == "species" else f"{names[lv.cls]} {who}"
        dose = "always" if lv.mass_bp == 10_000 else f"in {lv.mass:.2%} of episodes per mon (a per-episode draw)"
        parts.append(f"{what}, {dose}")
    return "the opponent's TRUE " + "; ".join(parts)


def recorded_oracle_reveal(model_path: str) -> str:
    """The `oracle_reveal` LABEL a checkpoint's `model_config.json` records; `"off"` for a config that predates the
    field (v137) or a checkpoint with no config (a legacy model). Reads the JSON key alone — no architecture
    check, no ModelVersion — so a tool can ask before it loads anything. A recorded text that is not a spec the
    training observation honours is REFUSED (never read as `off`)."""
    from agents.model.snapshot import _run_config_path

    path = os.path.abspath(model_path)
    # `model_config.json` is run-LEVEL: beside the zip, or one level up from <run>/checkpoints/ (`_run_config_path`)
    config_path = _run_config_path(path if os.path.isdir(path) else os.path.dirname(path))
    if not os.path.exists(config_path):
        return "off"
    with open(config_path) as fh:
        mode = json.load(fh).get("oracle_reveal", "off")
    try:
        return label(mode)
    except ValueError as e:
        raise OracleRevealRefused(f"{config_path}: the recorded oracle_reveal {mode!r} is not one of the levels "
                                  f"{ORACLE_REVEAL_MODES} nor a spec the training observation honours ({e})") from None


def refuse_if_revealed(model_path: str, *, tool: str, reason: Optional[str] = None) -> None:
    """Raise :class:`OracleRevealRefused` if `model_path` records a non-`off` oracle mode. `tool` names the
    caller; `reason` says what it would have fed the checkpoint instead."""
    mode = recorded_oracle_reveal(model_path)
    if mode == "off":
        return
    raise OracleRevealRefused(
        f"{tool}: {model_path} was trained under --oracle-reveal {mode} (a DIAGNOSTIC observation mode: "
        f"{describe(mode)} is written into its observation). "
        + (reason or "This tool builds its observations without the reveal, so the checkpoint would be fed "
                     "inputs it never trained on — silently.")
        + " Play it with `python -m main.h2h --oracle-reveal-mode one_sided|both_sided` (the per-side reveal, "
        "X5 A/B §7.7(a)) or `withheld` (the honest-play read, the reveal declared OFF), or read its own in-loop eval "
        "(the run's recorded mode)."
    )
