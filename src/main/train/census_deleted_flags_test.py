"""The flags the FLAG CENSUS (deletion pass P11) deleted stay deleted.

Each one is REFUSED by the trainer's parser (`unrecognized arguments`, exit 2) rather than parsed and
ignored, and none leaves a dest on the namespace — the reason `--eval-concurrency` could not be deleted
until the parser stopped abbreviation-matching. Every row is in `designs/deleted_flags.md` with its
citation; `src/claude_md_freshness_gate_test.py` fails a CLAUDE.md that names one as live.
"""
from __future__ import annotations

import pytest

from main.train.parser import build_parser

# (flag, value tokens) — one row per DELETED flag, appended by each P11 batch.
DELETED = [
    ("--eval-concurrency", ["100"]),                                   # B1
    ("--self-play-use-cpu", []),                                       # B2
    ("--no-self-play-use-cpu", []),
    ("--predict-unrevealed-mon-moves", []),
    ("--no-predict-unrevealed-mon-moves", []),
    ("--snapshot-dir", ["/tmp/pool"]),
    ("--allow-nonsample-trainee", []),
    ("--warmstart-consensus", ["models/a,models/b"]),                  # B3
    ("--warmstart-battles", ["200"]),
    ("--warmstart-bc-steps", ["4000"]),
    ("--showdown-port", ["8001"]),                                     # B4
    ("--eval-workers", ["5"]),
    ("--eval-device", ["cpu"]),
    ("--eval-concurrency-per-worker", ["1"]),
    ("--env-core", ["rust"]),                                          # P11b (a): the one-valued flags
    ("--use-bridge", ["rust"]),
    ("--critic", ["winprob"]),                                         # P11b (b)
    ("--gamma", ["1.0"]),                                              # P11b (c): the reward cluster
    ("--victory-value", ["1.0"]),
    ("--draw-penalty", ["0"]),
    ("--terminal-indicator", []),
    ("--no-terminal-indicator", []),
    ("--exploiter-temp-start", ["2.0"]),
    ("--exploiter-temp-end", ["1.0"]),
    ("--exploiter-temp-anneal-frac", ["0.2"]),
    ("--exploiter-temp-mode", ["ratchet"]),
    ("--exploiter-temp-ratchet-wr", ["0.55"]),
    ("--exploiter-temp-ratchet-factor", ["0.9"]),
    ("--exploiter-temp-ratchet-games", ["500"]),
    ("--rollout-target-band", ["49152,196608"]),
    ("--rollout-trigger", ["window"]),
    ("--opponent-sampling", ["generator"]),
    ("--version-pinning", ["per_game"]),
    ("--pair-value-route", []),
    ("--no-pair-value-route", []),
    ("--td-aux-coef", ["1.0"]),
    ("--win-prob-strata-weight", ["0.5"]),
    ("--bait-bot-share", ["0.1"]),                                     # P11d: the four Rust crossings
    ("--bait-bot-p", ["0.6"]),
    ("--progress-decision-tense", []),
    ("--no-progress-decision-tense", []),
    ("--progress-switch-freeze", []),
    ("--no-progress-switch-freeze", []),
]


@pytest.mark.parametrize("flag,value", DELETED, ids=[f for f, _ in DELETED])
def test_a_census_deleted_flag_is_refused(flag, value, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args([flag, *value])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    # ... and the refusal names the REASON (designs/deleted_flags.md, via `ExplainingParser`), not just the typo
    assert f"{flag.split('=')[0]} was DELETED" in err, err[-400:]


def test_no_census_deleted_flag_is_still_an_option():
    live = {o for a in build_parser()._actions for o in a.option_strings}
    assert not [f for f, _ in DELETED if f in live]
