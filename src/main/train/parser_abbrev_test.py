"""Neither the trainer's parser nor the launcher's abbreviation-matches (deletion pass P11).

argparse resolves an unknown `--token` that is a UNIQUE PREFIX of a known option onto it. After a flag is
deleted that is a silent redirect: P6 kept `--eval-concurrency` alive for exactly this reason, because
deleting it would have made an old `--eval-concurrency 100` set `--eval-concurrency-per-worker 100`. Both
parsers are `allow_abbrev=False`, so a typed flag means that flag or is refused.
"""
from __future__ import annotations

import pytest

from main.launcher.run import build_launcher_parser
from main.train.parser import build_parser


@pytest.mark.parametrize("argv", [["--n-env", "4"], ["--eval-shard", "5"], ["--run-d", "x"],
                                  ["--eval-concurrency", "100"], ["--no-tb-inh"]])
def test_the_trainer_parser_refuses_a_prefix_of_a_flag(argv, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(argv)
    assert e.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_the_trainer_parser_still_takes_the_exact_spellings():
    ns = build_parser().parse_args(["--n-envs", "4", "--eval-shard-games", "5", "--no-tb-inherit"])
    assert (ns.n_envs, ns.eval_shard_games, ns.tb_inherit) == (4, 5, False)


def test_a_deleted_flag_is_not_a_trainer_option():
    assert "--eval-concurrency" not in {o for a in build_parser()._actions for o in a.option_strings}


def test_the_launcher_parser_leaves_a_prefix_of_its_flag_for_the_child_to_refuse():
    known, rest = build_launcher_parser().parse_known_args(["--nic", "5", "--dry-run"])
    assert known.nice == 10 and known.dry_run is True      # `--nic` did NOT become `--nice`
    assert rest == ["--nic", "5"]
