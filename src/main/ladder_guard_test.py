"""The LADDER policy in code (owner ruling 2026-10-09): allowed for users, never for our agents without the owner's
approval token, only at concurrency 1.

Every test names its own state files (the halt marker, the approval token, the drift record) in a tmp dir; the real
ones are SEALED under pytest. Nothing here opens a socket: the client is driven through its queues, ``main_rust`` is
stubbed, and a fake client stands in where ``play`` builds one. The 'agent' is simulated by planting the marker
variables; the ancestor walk is neutralised in the fixture and tested on its own with real processes.

What fails on revert: the agent refusal (env and ancestor), the token's expiry / 24 h cap / shape, the stale / missing /
drift / wrong-format drift record, the halt, the forced concurrency 1 (the flag refusal, the sequential search loop, the
second-room refusal), a partial scan recording a green, and the human session still being allowed.
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import main.ladder_guard as G
import main.live.client as live_client
import main.play as play
from main import ladder_drift_scan
from main.exit_codes import TrainExitCode
from main.live import halt as H
from main.live.client import BattleResult, ClientConfig, ClientError, LiveClient, SearchRefused

NOW = dt.datetime(2026, 10, 9, 12, 0, tzinfo=dt.timezone.utc)
LADDER_ARGV = ["--mode", "ladder", "--model", "m.zip", "--username", "someone"]


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def write_token(path: Path, *, hours: float = 4.0, purpose: str = "owner-approved smoke") -> None:
    path.write_text(json.dumps({"purpose": purpose, "expires": iso(dt.datetime.now(dt.timezone.utc)
                                                                   + dt.timedelta(hours=hours))}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A HUMAN session with every ladder guard satisfied — a green drift record from a full scan, no halt, no agent
    marker anywhere (the ancestor walk is neutralised: this test process may itself run under an agent harness)."""
    for k in G.AGENT_ENV_MARKERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(G, "_ancestor_markers", lambda: [])
    paths = SimpleNamespace(halt=tmp_path / "halt.json", approval=tmp_path / "approval.json",
                            drift=tmp_path / "drift.json")
    monkeypatch.setenv(H.HALT_ENV, str(paths.halt))
    monkeypatch.setenv(G.APPROVAL_ENV, str(paths.approval))
    monkeypatch.setenv(G.DRIFT_ENV, str(paths.drift))
    G.record_drift_result(0, battle_format="gen3ou", n_replays=200, full=True, showdown_commit="a" * 40,
                          checks=["read"], now=dt.datetime.now(dt.timezone.utc))
    return paths


# -- the state files are sealed under pytest -------------------------------------------------------------------------

def test_the_real_state_files_are_sealed_under_pytest(monkeypatch):
    monkeypatch.delenv(G.APPROVAL_ENV, raising=False)
    monkeypatch.delenv(G.DRIFT_ENV, raising=False)
    assert not G.approval_path().exists() and not G.drift_path().exists()  # a READ never sees the owner's real file
    with pytest.raises(G.LadderStateUnsealed):
        G.drift_path(write=True)
    with pytest.raises(G.LadderStateUnsealed):
        G.record_drift_result(0, battle_format="gen3ou", n_replays=200, full=True, showdown_commit="x", checks=[])


# -- a human session is allowed, at concurrency 1 --------------------------------------------------------------------

def test_a_human_session_is_cleared_at_concurrency_one(env):
    permit = G.check_ladder_policy(battle_format="gen3ou")
    assert permit.concurrency == G.LADDER_CONCURRENCY == 1
    assert permit.agent_reasons == () and permit.approval is None
    assert permit.drift["status"] == "green"


def test_the_respect_notice_names_the_rules(env):
    text = G.respect_notice(G.check_ladder_policy(battle_format="gen3ou"), forfeit_turn_limit=250)
    for needle in ("RESPECT FOR PLAYERS", "concurrency 1", "never chats", "turn 250", "HALTS all play", "drift gate"):
        assert needle in text
    assert "AGENT SESSION" not in text


# -- guard 4: agent sessions need the owner's token ------------------------------------------------------------------

@pytest.mark.parametrize("marker", G.AGENT_ENV_MARKERS)
def test_an_agent_session_is_refused_without_the_owners_token(env, monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    with pytest.raises(G.LadderRefused) as ei:
        G.check_ladder_policy(battle_format="gen3ou")
    msg = str(ei.value)
    assert "agent sessions need the owner's explicit approval" in msg
    assert "the owner creates the token" in msg and "agents must never create it" in msg


def test_an_ancestors_marker_is_an_agent_session_too(env, monkeypatch):
    """`env -u CLAUDECODE python …` clears the process's own variable but not the harness shell's above it."""
    monkeypatch.setattr(G, "_ancestor_markers", lambda: [(4242, "CLAUDECODE")])
    with pytest.raises(G.LadderRefused, match="ancestor process 4242"):
        G.check_ladder_policy(battle_format="gen3ou")
    assert G.agent_session_reasons({}) == []  # an explicit empty environment is a human's, no walk


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="the ancestor walk reads /proc (Linux)")
def test_the_real_ancestor_walk_sees_markers_the_child_unset():
    """No monkeypatching: a REAL parent started WITH the markers spawns a child with every one REMOVED; the child's own
    environment is clean and `agent_session_reasons()` still names the parent."""
    child = textwrap.dedent("""
        import os
        from main import ladder_guard as G
        assert not [k for k in G.AGENT_ENV_MARKERS if k in os.environ]
        print("REASONS=" + repr(G.agent_session_reasons()))
    """)
    parent = textwrap.dedent(f"""
        import os, subprocess, sys
        markers = {G.AGENT_ENV_MARKERS!r}
        env = {{k: v for k, v in os.environ.items() if k not in markers}}
        r = subprocess.run([sys.executable, "-c", {child!r}], env=env, capture_output=True, text=True)
        print(r.stdout, r.stderr, sep="")
    """)
    env = dict(os.environ, **{k: "1" for k in G.AGENT_ENV_MARKERS})
    out = subprocess.run([sys.executable, "-c", parent], env=env, capture_output=True, text=True, timeout=120)
    assert "REASONS=" in out.stdout, out.stdout + out.stderr
    assert "$CLAUDECODE on ancestor process" in out.stdout, out.stdout


def test_a_valid_token_lets_an_agent_session_in_at_concurrency_one(env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    write_token(env.approval, hours=4, purpose="owner-approved: smoke against my own friend")
    permit = G.check_ladder_policy(battle_format="gen3ou")
    assert permit.concurrency == 1
    assert permit.approval is not None and "owner-approved" in permit.approval.purpose
    assert permit.agent_reasons == ("$CLAUDECODE",)
    text = G.respect_notice(permit, forfeit_turn_limit=250)
    assert "AGENT SESSION" in text and "OWNER'S approval" in text and "owner-approved" in text


def test_an_expired_token_refuses(env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    write_token(env.approval, hours=-1)
    with pytest.raises(G.LadderRefused, match="EXPIRED"):
        G.check_ladder_policy(battle_format="gen3ou")


def test_a_token_more_than_a_day_ahead_refuses(env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    write_token(env.approval, hours=G.APPROVAL_MAX_HOURS + 2)
    with pytest.raises(G.LadderRefused, match="more than 24 h ahead"):
        G.check_ladder_policy(battle_format="gen3ou")


@pytest.mark.parametrize("body", [
    "not json",
    "[]",
    json.dumps({"expires": "2999-01-01T00:00:00+00:00"}),                        # no purpose
    json.dumps({"purpose": "  ", "expires": "2999-01-01T00:00:00+00:00"}),      # blank purpose
    json.dumps({"purpose": "p"}),                                                # no expiry
    json.dumps({"purpose": "p", "expires": "tomorrow"}),                         # not a time
    json.dumps({"purpose": "p", "expires": "2999-01-01T00:00:00"}),              # no UTC offset
])
def test_a_malformed_token_refuses(env, monkeypatch, body):
    monkeypatch.setenv("CLAUDECODE", "1")
    env.approval.write_text(body)
    with pytest.raises(G.LadderRefused, match="unreadable"):
        G.check_ladder_policy(battle_format="gen3ou")


def test_no_token_file_refuses_and_says_how_the_owner_writes_one(env, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    assert not env.approval.exists()
    with pytest.raises(G.LadderRefused, match='"purpose"'):
        G.check_ladder_policy(battle_format="gen3ou")


def test_a_human_is_not_asked_for_a_token(env):
    """A token that would be refused is never read in a human session (the guard is the marker, not the file)."""
    env.approval.write_text("not json")
    assert G.check_ladder_policy(battle_format="gen3ou").approval is None


# -- guard 3: the drift record ---------------------------------------------------------------------------------------

def test_a_missing_drift_record_refuses_and_names_the_command(env):
    env.drift.unlink()
    with pytest.raises(G.LadderRefused, match="ladder_drift_scan.py"):
        G.check_ladder_policy(battle_format="gen3ou")


@pytest.mark.parametrize("age_days, ok", [(0.5, True), (G.DRIFT_MAX_AGE_DAYS - 0.5, True),
                                          (G.DRIFT_MAX_AGE_DAYS + 0.5, False), (30, False)])
def test_the_drift_record_goes_stale(env, age_days, ok):
    # the record's age is judged against an injected NOW (half-day margins: no input on a boundary)
    G.record_drift_result(0, battle_format="gen3ou", n_replays=200, full=True, showdown_commit="b" * 40,
                          checks=["read"], now=NOW - dt.timedelta(days=age_days))
    if ok:
        assert G.check_drift_fresh("gen3ou", now=NOW)["status"] == "green"
    else:
        with pytest.raises(G.LadderRefused, match="h old"):
            G.check_drift_fresh("gen3ou", now=NOW)


def test_a_record_from_the_future_refuses(env):
    G.record_drift_result(0, battle_format="gen3ou", n_replays=200, full=True, showdown_commit="b" * 40,
                          checks=["read"], now=NOW + dt.timedelta(days=1))
    with pytest.raises(G.LadderRefused, match="not from the future"):
        G.check_drift_fresh("gen3ou", now=NOW)


def test_a_drift_finding_revokes_an_earlier_green(env):
    G.check_ladder_policy(battle_format="gen3ou")  # green
    G.record_drift_result(1, battle_format="gen3ou", n_replays=5, full=False, showdown_commit=None, checks=["read"])
    with pytest.raises(G.LadderRefused, match="FOUND DRIFT"):
        G.check_ladder_policy(battle_format="gen3ou")


def test_the_record_is_for_one_format(env):
    with pytest.raises(G.LadderRefused, match="format 'gen3ou'"):
        G.check_drift_fresh("gen3uu")


def test_a_corrupt_drift_record_refuses(env):
    env.drift.write_text("{")
    with pytest.raises(G.LadderRefused, match="unreadable"):
        G.check_ladder_policy(battle_format="gen3ou")


def _scan_args(**kw):
    base = dict(offline=False, no_effects=False, no_format_spec=False, format="gen3ou")
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.parametrize("rc, n, args_kw, master, recorded", [
    (0, 200, {}, "c" * 40, "green"),
    (0, G.DRIFT_MIN_REPLAYS - 1, {}, "c" * 40, None),                    # too few replays: a smoke, not a gate
    (0, 200, {"offline": True}, "c" * 40, None),
    (0, 200, {"no_effects": True}, "c" * 40, None),
    (0, 200, {"no_format_spec": True}, "c" * 40, None),
    (0, 200, {}, None, None),                                            # a user-named / cached Showdown checkout
    (2, 200, {}, "c" * 40, None),                                        # no replays: not a verdict
    (1, 3, {"offline": True}, None, "drift"),                            # a drift finding is ALWAYS recorded
])
def test_only_a_full_scan_records_a_green(env, rc, n, args_kw, master, recorded):
    env.drift.unlink()
    ladder_drift_scan.record_result(rc, _scan_args(**args_kw), n_replays=n, master_commit=master)
    if recorded is None:
        assert not env.drift.exists()
    else:
        rec = json.loads(env.drift.read_text())
        assert rec["status"] == recorded and rec["schema"] == G.DRIFT_SCHEMA and rec["format"] == "gen3ou"


# -- guard 2: the T28 halt -------------------------------------------------------------------------------------------

def test_a_halt_marker_refuses_the_ladder(env):
    H.record_halt(reason="planted", entry_point="test")
    with pytest.raises(H.HaltActive):
        G.check_ladder_policy(battle_format="gen3ou")


# -- main.play: the entry point -------------------------------------------------------------------------------------

@pytest.fixture
def stub_rust(monkeypatch):
    """`play.main_rust` replaced by a recorder: no client, no socket, no policy load."""
    seen = []

    async def fake(args):
        seen.append(args)
        return 0
    monkeypatch.setattr(play, "main_rust", fake)
    return seen


def test_a_human_ladder_session_starts_and_prints_the_notice(env, stub_rust, capsys):
    assert play.run(LADDER_ARGV) == 0
    assert [a.mode for a in stub_rust] == ["ladder"]
    out = capsys.readouterr().out
    assert "RESPECT FOR PLAYERS" in out and "concurrency 1" in out
    assert f"turn {play.DEFAULT_FORFEIT_TURN_LIMIT}" in out  # the existing forfeit limit is the one announced


def test_an_agent_session_cannot_start_a_ladder_session_without_the_token(env, stub_rust, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(SystemExit, match="agents must never create it"):
        play.run(LADDER_ARGV)
    assert stub_rust == []


def test_an_agent_session_with_the_owners_token_starts_at_concurrency_one(env, stub_rust, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    write_token(env.approval, purpose="owner says: one game against my alt")
    assert play.run(LADDER_ARGV) == 0
    assert len(stub_rust) == 1
    assert "owner says: one game against my alt" in capsys.readouterr().out


def test_an_expired_token_stops_an_agent_session(env, stub_rust, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    write_token(env.approval, hours=-0.5)
    with pytest.raises(SystemExit, match="EXPIRED"):
        play.run(LADDER_ARGV)
    assert stub_rust == []


def test_a_guard_that_lapses_between_games_ends_the_session_with_the_reason(env, monkeypatch, capsys):
    async def lapses(args):
        raise G.LadderRefused("ladder REFUSED in an agent session: the owner's approval token EXPIRED")
    monkeypatch.setattr(play, "main_rust", lapses)
    assert play.run(LADDER_ARGV) == 1
    assert "EXPIRED" in capsys.readouterr().err


def test_a_halt_marker_stops_a_ladder_session_before_any_connection(env, stub_rust):
    H.record_halt(reason="planted", entry_point="test")
    assert play.run(LADDER_ARGV) == int(TrainExitCode.FATAL_LIVE_PARSE)
    assert stub_rust == []


def test_a_stale_drift_record_stops_a_ladder_session(env, stub_rust):
    G.record_drift_result(0, battle_format="gen3ou", n_replays=200, full=True, showdown_commit="d" * 40,
                          checks=["read"], now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=9))
    with pytest.raises(SystemExit, match="ladder_drift_scan.py"):
        play.run(LADDER_ARGV)
    assert stub_rust == []


@pytest.mark.parametrize("value", ["1", "2", "3"])
@pytest.mark.parametrize("spelling", ["two-arg", "equals"])
def test_a_typed_concurrency_is_refused_with_the_reason(env, stub_rust, value, spelling):
    """Concurrency is FIXED at 1: there is no option to raise it (nor to restate it)."""
    extra = ["--concurrency", value] if spelling == "two-arg" else [f"--concurrency={value}"]
    with pytest.raises(SystemExit) as ei:
        play.run([*LADDER_ARGV, *extra])
    msg = str(ei.value)
    assert "--concurrency was DELETED" in msg and "fixed at concurrency 1" in msg and "no option to raise it" in msg
    assert stub_rust == []


@pytest.mark.parametrize("extra, needle", [
    (["--opponent", "somebody"], "never names an --opponent"),
    (["--public-acceptance"], "not a --public-acceptance series"),
])
def test_a_ladder_session_names_nobody(env, stub_rust, extra, needle):
    with pytest.raises(SystemExit, match=needle):
        play.run([*LADDER_ARGV, *extra])


def test_a_ladder_session_needs_a_model_and_an_account(env, stub_rust, monkeypatch):
    monkeypatch.delenv("PS_USERNAME", raising=False)
    with pytest.raises(SystemExit, match="without --model"):
        play.run(["--mode", "ladder", "--username", "u"])
    with pytest.raises(SystemExit, match="needs --username"):
        play.run(["--mode", "ladder", "--model", "m.zip"])


def test_the_ladder_reaches_the_client_as_a_ladder_config_one_search_loop(env, monkeypatch):
    """The plumbing: `--mode ladder` builds the client with `ladder=True` and calls `.ladder(n)` — and on the PUBLIC
    server it does not fall into the `--server official needs --opponent` refusal that guards the custom-game modes."""
    captured = []

    class FakeClient:
        def __init__(self, cfg, **kw):
            captured.append(cfg)
            self.called = None

        async def connect(self):
            pass

        async def ladder(self, n):
            captured.append(("ladder", n))
            return [BattleResult("battle-gen3ou-1", "p1", "someone", True, 9, 4, False)]

        async def close(self):
            pass

    monkeypatch.setattr(live_client, "LiveClient", FakeClient)
    monkeypatch.setattr(play, "rust_policy", lambda args, seed_offset=0: object())
    args = play.build_parser().parse_args([*LADDER_ARGV, "--server", "official", "--n-battles", "3",
                                           "--forfeit-turn-limit", "77"])
    assert asyncio.run(play.main_rust(args)) == 0
    cfg = captured[0]
    assert cfg.ladder is True and cfg.auth == "official" and cfg.forfeit_turn_limit == 77
    assert captured[1] == ("ladder", 3)


def test_a_custom_game_mode_is_not_a_ladder_client(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, cfg, **kw):
            captured.append(cfg)

        async def connect(self):
            raise RuntimeError("stop")

        async def close(self):
            pass

    monkeypatch.setattr(live_client, "LiveClient", FakeClient)
    monkeypatch.setattr(play, "rust_policy", lambda args, seed_offset=0: object())
    args = play.build_parser().parse_args(["--mode", "challenge", "--model", "m.zip", "--opponent", "y"])
    with pytest.raises(RuntimeError, match="stop"):
        asyncio.run(play.main_rust(args))
    assert [c.ladder for c in captured] == [False]


def test_the_help_text_states_the_ladder_policy():
    text = play.build_parser().format_help()
    for needle in ("concurrency 1", "ladder_owner_approval.json", "agents must never create it", "2026-10-09"):
        assert needle in " ".join(text.split())


# -- main.live.client: the seam that queues --------------------------------------------------------------------------

class _Pick:
    def choose(self, frame):
        return 0


def ladder_client(**kw):
    cfg = ClientConfig(uri="ws://127.0.0.1:1", username="someone", ladder=True, **kw)
    return LiveClient(cfg, policy=_Pick(), team_fn=lambda: "TEAM")


def drain(c):
    out = []
    while not c._outq.empty():
        out.append(c._outq.get_nowait())
    return out


async def settle():
    await asyncio.sleep(0.05)  # let the client's task run to its next await


def result(room):
    return BattleResult(room, "p1", "someone", True, 12, 6, False)


def test_search_commands_exist_only_for_a_ladder_client():
    plain = LiveClient(ClientConfig(uri="ws://x", username="a"), policy=_Pick(), team_fn=lambda: None)
    for cmd in ("|/search gen3ou", "|/cancelsearch"):
        with pytest.raises(ClientError, match="declared command set"):
            plain.send_raw(cmd)
    lad = ladder_client()
    lad.send_raw("|/search gen3ou")
    lad.send_raw("|/cancelsearch")
    assert drain(lad) == ["|/search gen3ou", "|/cancelsearch"]
    for bad in ("|/msg someone, hi", "battle-x|hello there", "|/pm x", "|/reject someone"):
        with pytest.raises(ClientError, match="declared command set"):
            lad.send_raw(bad)  # a ladder client still never chats or PMs


async def test_a_non_ladder_client_cannot_call_ladder():
    plain = LiveClient(ClientConfig(uri="ws://x", username="a"), policy=_Pick(), team_fn=lambda: None)
    with pytest.raises(ClientError, match="needs ClientConfig.ladder=True"):
        await plain.ladder(1)


async def test_the_ladder_plays_one_battle_at_a_time(env):
    """CONCURRENCY 1 is structural: the next `/search` is queued only after the previous battle ENDED. Revert it to
    queue searches up front (or to search again after the room opens) and a second `/search` appears early."""
    c = ladder_client()
    task = asyncio.ensure_future(c.ladder(3))
    for game in (1, 2, 3):
        await settle()
        sent = drain(c)
        assert [s for s in sent if "/search" in s] == ["|/search gen3ou"], (game, sent)
        assert sent[0] == "|/utm TEAM"
        room = f"battle-gen3ou-{game}"
        c._battle_started.put_nowait(room)
        await settle()
        assert drain(c) == []  # the battle is live: nothing else is queued
        c._battle_done.put_nowait(result(room))
    res = await asyncio.wait_for(task, 5)
    assert [r.battle for r in res] == [f"battle-gen3ou-{i}" for i in (1, 2, 3)]


async def test_the_ladder_rechecks_the_guards_before_every_search(env):
    """A halt (or a lapsed approval / stale drift record) appearing mid-session stops it at the next game boundary."""
    c = ladder_client()
    task = asyncio.ensure_future(c.ladder(3))
    await settle()
    assert [s for s in drain(c) if "/search" in s] == ["|/search gen3ou"]
    c._battle_started.put_nowait("battle-gen3ou-1")
    await settle()
    H.record_halt(reason="planted mid-session", entry_point="test")
    c._battle_done.put_nowait(result("battle-gen3ou-1"))
    with pytest.raises(H.HaltActive):
        await asyncio.wait_for(task, 5)
    assert [s for s in drain(c) if "/search" in s] == []


async def test_the_clients_own_ladder_seam_refuses_an_agent_without_the_token(env, monkeypatch):
    """Guarded at the seam that queues, not only in `main.play`: a caller that skips the entry point is refused."""
    monkeypatch.setenv("CLAUDECODE", "1")
    c = ladder_client()
    with pytest.raises(G.LadderRefused, match="agents must never create it"):
        await c.ladder(1)
    assert drain(c) == []


async def test_a_second_live_battle_room_is_refused(env):
    c = ladder_client()
    c.reader_factory = lambda: SimpleNamespace(close=lambda: None)
    c._on_battle("battle-gen3ou-1", ["|init|battle"])
    with pytest.raises(ClientError, match="ladder concurrency is 1"):
        c._on_battle("battle-gen3ou-2", ["|init|battle"])


async def test_a_search_that_finds_nobody_is_cancelled(env):
    c = ladder_client(ladder_search_timeout_s=0.05)
    with pytest.raises(ClientError, match="timed out"):
        await c.ladder(1)
    assert drain(c)[-1] == "|/cancelsearch"


async def test_a_popup_during_the_search_is_the_servers_refusal(env):
    c = ladder_client()
    task = asyncio.ensure_future(c.ladder(1))
    await settle()
    c._on_global("|popup|You are locked from the ladder.")
    with pytest.raises(SearchRefused, match="locked"):
        await asyncio.wait_for(task, 5)
    assert not c._searching
