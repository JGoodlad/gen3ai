"""The ORACLE REVEAL's Python side: the recorded-mode reader and the refusal an offline tool takes.

`--oracle-reveal {off,species}` (gen3_oracle_reveal_v1, config v137; `designs/endstate/design_x5_belief_tokens.md`
§7.6) is a DIAGNOSTIC observation mode: the Rust encoder writes the opponent's true species into the opponent
block of the observation (`src/rust_sim/src/encoder/oracle.rs`). A checkpoint trained under it is a different
function of the observation than a production one, and the reveal is made by the ENV CORE, so a tool that plays or
reads a checkpoint through any other observation source (the websocket client `main.play`, the Lane S bank the
prober re-encodes, `main.h2h`'s `off`-mode eval engine, `main.anchors`) would feed it observations it never trained
on — silently, with every number still printing. Those tools therefore REFUSE a recorded non-`off` mode, naming it:
:func:`refuse_if_revealed`.

The per-SIDE reveal that would let `main.h2h` play an oracle checkpoint (the one-sided clairvoyance cells of the X5
A/B, §7.7(a)) is DEFERRED, not built.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from utils.rust_env.protocol import ORACLE_REVEAL_LEVELS

#: The legal levels, `off` first (the production mode). One list: the Rust `Level` and the parser read it.
ORACLE_REVEAL_MODES = ORACLE_REVEAL_LEVELS


class OracleRevealRefused(ValueError):
    """An offline tool was handed a checkpoint trained under a diagnostic observation mode it cannot reproduce."""


def recorded_oracle_reveal(model_path: str) -> str:
    """The `oracle_reveal` a checkpoint's `model_config.json` records; `"off"` for a config that predates the
    field (v137) or a checkpoint with no config (a legacy model). Reads the JSON key alone — no architecture
    check, no ModelVersion — so a tool can ask before it loads anything."""
    from agents.model.snapshot import _run_config_path

    path = os.path.abspath(model_path)
    # `model_config.json` is run-LEVEL: beside the zip, or one level up from <run>/checkpoints/ (`_run_config_path`)
    config_path = _run_config_path(path if os.path.isdir(path) else os.path.dirname(path))
    if not os.path.exists(config_path):
        return "off"
    with open(config_path) as fh:
        mode = json.load(fh).get("oracle_reveal", "off")
    if mode not in ORACLE_REVEAL_MODES:
        raise OracleRevealRefused(f"{config_path}: oracle_reveal {mode!r} is not one of {ORACLE_REVEAL_MODES}")
    return str(mode)


def refuse_if_revealed(model_path: str, *, tool: str, reason: Optional[str] = None) -> None:
    """Raise :class:`OracleRevealRefused` if `model_path` records a non-`off` oracle mode. `tool` names the
    caller; `reason` says what it would have fed the checkpoint instead."""
    mode = recorded_oracle_reveal(model_path)
    if mode == "off":
        return
    raise OracleRevealRefused(
        f"{tool}: {model_path} was trained under --oracle-reveal {mode} (a DIAGNOSTIC observation mode: the "
        "opponent's true species are written into its observation). "
        + (reason or "This tool builds its observations without the reveal, so the checkpoint would be fed "
                     "inputs it never trained on — silently.")
        + " The per-side reveal an offline engine needs to play it (X5 A/B §7.7(a)) is deferred; "
        "read its own in-loop eval (the run's recorded mode) instead."
    )
