"""`designs/ops/anchors.json` — the one place a box-specific path is allowed to live.

`src/utils/paths_test.py` fails any module under `src/agents`, `src/main`, `src/utils` that uses a
`/home/...` literal as a VALUE, and it is right to: four tests once read `models/` through such a
literal and therefore skipped forever on every other machine — invisible coverage loss, because a
skip that is supposed to happen looks exactly like a skip that is not. Metamon and Foul Play are
checkouts outside this repo, so "where are they" is a real box-specific question; the answer is a
config file plus a per-key env override, and this file pins the three properties that make that
safe:

1. **an env override WINS and is authoritative** — including the whole-file override;
2. **a set-but-missing override RAISES** rather than falling back, because an explicit override
   that silently resolves elsewhere is how a measurement ends up describing a checkout nobody
   chose;
3. **a missing checkout is a NAMED refusal at the point of use**, never a skip and never a zero —
   and `--dry-run` / `--show-config` still work without either checkout, which is what makes the
   tool readable on a fresh box.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.anchors.config import (
    CONFIG_PATH_ENV_VAR,
    ENV_OVERRIDES,
    AnchorConfigError,
    config_path,
    describe,
    load_config,
)


def _write(tmp_path: Path, **over) -> Path:
    blob = {
        "showdown": {"node": "node", "port_range": [9500, 9599]},
        "opponents": {
            "metamon": {
                "checkout": str(tmp_path / "metamon"),
                "python": str(tmp_path / "metamon" / "python"),
                "cache_dir": str(tmp_path / "metamon" / "cache"),
                "agents": {"SmallRL": {"checkpoint": 40}},
                "team_sets": {"home": "gen3ai_pool", "away": "competitive"},
            },
            "foulplay": {
                "checkout": str(tmp_path / "fp"),
                "python": str(tmp_path / "fp" / "python"),
                "team_dirs": {"home": "gen3/ou/pool"},
            },
        },
        "our_team_sources": {"home": {"kind": "pool"}},
    }
    blob.update(over)
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(blob))
    return path


def test_the_committed_default_parses_and_names_both_anchors() -> None:
    """The file that ships. If it stops parsing, every anchor read stops with it."""
    cfg = load_config()
    assert cfg.source == config_path()
    assert {"SmallRL", "SyntheticRLV2"} <= set(cfg.metamon.agents)
    assert cfg.metamon.agents["SmallRL"]["checkpoint"] == 40
    assert cfg.metamon.team_sets["away"] == "competitive"
    assert set(cfg.our_team_sources) == {"home", "away"}
    assert cfg.port_range == [9500, 9599], "9XXX only — 8000/8001 must be unreachable by default"


def test_a_per_key_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path)
    monkeypatch.setenv(ENV_OVERRIDES["opponents.metamon.checkout"], "/elsewhere/metamon")
    cfg = load_config(path)
    assert str(cfg.metamon.checkout) == "/elsewhere/metamon"


def test_the_whole_file_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path)
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(path))
    assert config_path() == path
    assert load_config().source == path


def test_a_set_but_missing_file_override_raises_instead_of_falling_back(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Property 2. A quiet fall-back here is a measurement about the wrong checkout."""
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, "/no/such/anchors.json")
    with pytest.raises(AnchorConfigError) as excinfo:
        config_path()
    assert CONFIG_PATH_ENV_VAR in str(excinfo.value)


def test_a_missing_checkout_is_a_named_refusal_that_states_the_fix(tmp_path: Path) -> None:
    """Property 3 — and the message must carry BOTH the key and its env var, because the reader
    of this failure is someone on a box that has never run an anchor read."""
    cfg = load_config(_write(tmp_path))
    with pytest.raises(AnchorConfigError) as excinfo:
        cfg.metamon.require("SmallRL")
    message = str(excinfo.value)
    assert "opponents.metamon.checkout" in message
    assert ENV_OVERRIDES["opponents.metamon.checkout"] in message
    assert "designs/ops/anchors.json" in message


def test_an_unpinned_metamon_agent_is_refused_by_name(tmp_path: Path) -> None:
    """A policy is only an anchor once its CHECKPOINT is pinned — the CLI may not invent one."""
    cfg = load_config(_write(tmp_path))
    (tmp_path / "metamon" / "cache").mkdir(parents=True)
    (tmp_path / "metamon" / "python").write_text("")
    with pytest.raises(AnchorConfigError) as excinfo:
        cfg.metamon.require("Kakuna")
    assert "Kakuna" in str(excinfo.value) and "SmallRL" in str(excinfo.value)


def test_an_unknown_team_set_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path))
    with pytest.raises(AnchorConfigError):
        cfg.metamon.team_set("nonesuch")
    with pytest.raises(AnchorConfigError):
        cfg.foulplay.team_name("away")
    with pytest.raises(AnchorConfigError):
        cfg.our_team_source("away")


def test_a_relative_foulplay_team_name_resolves_inside_its_own_checkout(tmp_path: Path) -> None:
    """Foul Play's `load_team` does `os.path.join(TEAM_DIR, name)`, which returns an ABSOLUTE
    name unchanged — so both spellings are supported and must resolve differently."""
    cfg = load_config(_write(tmp_path))
    rel = cfg.foulplay.team_dir("home")
    assert rel == tmp_path / "fp" / "fp" / "teams" / "teams" / "gen3/ou/pool"

    absolute = load_config(_write(
        tmp_path,
        opponents={"metamon": json.loads(_write(tmp_path).read_text())["opponents"]["metamon"],
                   "foulplay": {"checkout": str(tmp_path / "fp"),
                                "python": str(tmp_path / "fp" / "python"),
                                "team_dirs": {"home": "/abs/teams"}}}))
    assert str(absolute.foulplay.team_dir("home")) == "/abs/teams"


def test_describe_prints_every_key_with_its_env_var_and_whether_it_exists(tmp_path: Path) -> None:
    """`--show-config` is the first thing anyone runs on a new box; it has to answer 'is it there'."""
    text = describe(load_config(_write(tmp_path)))
    for env_var in ENV_OVERRIDES.values():
        assert env_var in text
    assert "[MISSING]" in text, "a checkout that is not on this box must SAY so"


def test_the_committed_config_holds_the_absolute_paths_src_may_not_hold() -> None:
    """The reason this file exists, stated as a test.

    `src/utils/paths_test.py` fails any module under `src/` that uses an absolute home path as a
    VALUE. The anchor checkouts ARE at absolute paths outside the repo, so the config is where
    they live — and every one of them must be absolute, because a relative path here would resolve
    against whatever directory the caller happened to be standing in.
    """
    cfg = load_config()
    for label, path in (
        ("metamon.checkout", cfg.metamon.checkout),
        ("metamon.python", cfg.metamon.python),
        ("metamon.cache_dir", cfg.metamon.cache_dir),
        ("foulplay.checkout", cfg.foulplay.checkout),
        ("foulplay.python", cfg.foulplay.python),
    ):
        assert Path(path).is_absolute(), f"{label} = {path} is not absolute"
