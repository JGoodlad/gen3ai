"""A DELETED flag is refused WITH the reason (`parser.base.ExplainingParser`, flag census P11).

argparse's own `unrecognized arguments: --eval-workers 5` reads like a typo. The trainer's parser appends one line
per deleted flag it sees — the pass that deleted it and the first sentence of why, from `designs/deleted_flags.md`
(the list the freshness gate already reads) — so an old runbook command fails with its own explanation. A typo gets
argparse's line alone, and the explainer never raises on its error path.
"""
from __future__ import annotations

import pytest

from main.train.parser import build_parser
from main.train.parser import base


def _refusal(argv, capsys) -> str:
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


def test_a_deleted_flag_is_refused_with_its_pass_and_reason(capsys):
    err = _refusal(["--steps", "1", "--eval-workers=3", "--hand-shaping"], capsys)
    assert "unrecognized arguments: --eval-workers=3 --hand-shaping" in err          # argparse's own line, unchanged
    assert "--eval-workers was DELETED — deletion pass P11" in err and "eval-worker pool" in err
    assert "--hand-shaping was DELETED — `gen3_shaped_reward_deletion_v1`" in err    # an older pass's row, same shape
    assert "designs/deleted_flags.md" in err


def test_a_typo_gets_argparses_line_alone(capsys):
    err = _refusal(["--steps", "1", "--evall-games", "5"], capsys)
    assert "unrecognized arguments: --evall-games 5" in err and "DELETED" not in err


def test_a_live_flag_with_a_bad_value_is_not_explained_as_deleted(capsys):
    err = _refusal(["--steps", "notanumber"], capsys)
    assert "invalid int value" in err and "DELETED" not in err


def test_the_explainer_never_raises_on_the_error_path(monkeypatch):
    import utils.paths as paths

    def boom(*_a, **_k):
        raise OSError("no designs/ here")

    monkeypatch.setattr(paths, "repo_path", boom)
    assert base.deleted_flag_reasons(["--eval-workers"]) == []


def test_every_deleted_flags_row_is_explainable():
    """The explainer reads the same rows the freshness gate does; a row it cannot parse would refuse silently."""
    import re

    from utils.paths import repo_path

    text = repo_path("designs", "deleted_flags.md").read_text(encoding="utf-8")
    flags = re.findall(r"^\| `(--[a-z0-9][a-z0-9-]*)` \| ", text, flags=re.M)
    assert len(flags) > 100
    got = base.deleted_flag_reasons(flags)
    assert len(got) == len(set(flags)), (len(got), len(set(flags)))
