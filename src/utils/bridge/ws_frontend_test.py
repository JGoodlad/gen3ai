"""Unit gates for the websocket front end — the protocol surface, driven by a SCRIPTED client.

No socket, no bridge child, no battle: every test here drives `ShowdownFrontEnd` through a fake
websocket that records what was sent. That is deliberate, because the things most likely to break
silently are exactly the ones a battle would hide — a `|pm|` with eight fields instead of nine, a
`|request|` whose `rqid` is missing, a name refusal that never reaches the client. Each of those
presents downstream as a HANG (both de-risks recorded it), which is the failure mode a test has to
turn into an error.

The live end-to-end paths have their own files: `ws_frontend_integration_test.py` (a real
websocket, real battles) and `ws_frontend_byte_identity_integration_test.py` (the differential).
"""

from __future__ import annotations

import json

import pytest

from utils.bridge import ws_frontend
from utils.bridge.ws_frontend import (RESERVED_PORTS, ShowdownFrontEnd, _Battle, _Conn,
                                      _SideRequest, to_id)

# A minimal gen3ou `|request|` in the exact shape the sim emits it — no `rqid`, because the sim
# does not produce one (`server/room-battle.ts` injects it). Kept short; the splice does not care.
_SIM_REQUEST = ('|request|{"active":[{"moves":[{"move":"Rock Slide","id":"rockslide","pp":16,'
                '"maxpp":16,"target":"allAdjacentFoes","disabled":false}]}],'
                '"side":{"name":"Alpha","id":"p1","pokemon":[]}}')


class _FakeWS:
    """An async-iterable stand-in for a websocket connection."""

    def __init__(self, lines=()):
        self.inbox = list(lines)
        self.sent = []

    async def send(self, text):
        self.sent.append(text)

    def __aiter__(self):
        async def gen():
            for line in self.inbox:
                yield line
        return gen()


def _front(**kwargs) -> ShowdownFrontEnd:
    kwargs.setdefault("validate_teams", False)
    return ShowdownFrontEnd(impl="node", **kwargs)


async def _named(front: ShowdownFrontEnd, name: str) -> _Conn:
    conn = _Conn(ws=_FakeWS(), guest_n=1)
    await front._cmd_trn(conn, f"{name},0,assertion")
    return conn


def _fake_battle(front: ShowdownFrontEnd, p1: _Conn, p2: _Conn) -> _Battle:
    """A battle record with NO child process — `_write_child` short-circuits on `stdin is None`."""

    class _NoProc:
        stdin = None
        returncode = None

    battle = _Battle(tag="battle-gen3ou-1", fmt="gen3ou", proc=_NoProc(),
                     conns={"p1": p1, "p2": p2}, names={"p1": p1.name, "p2": p2.name},
                     seed=[1, 2, 3, 4], commands=[])
    front.battles[battle.tag] = battle
    return battle


# --------------------------------------------------------------------------------------------
# handshake + login
# --------------------------------------------------------------------------------------------
async def test_a_new_connection_is_greeted_as_a_guest_before_the_challstr():
    """The real server's ORDER, and poke-env's fork depends on it.

    `|updateuser| Guest N` arrives BEFORE `|challstr|`, and the vendored `PSClient` documents at
    length that honouring that greeting as a login is a hang (a passwordless client announced
    `logged_in` while still named "Guest N"). Reproducing the order is what keeps that documented
    guard on the path it was written for.
    """
    front = _front()
    ws = _FakeWS()
    await front.handler(ws)
    assert ws.sent[0].startswith("|updateuser| Guest 1|0|")
    assert ws.sent[1].startswith("|challstr|4|")
    assert len(ws.sent[1].split("|")[3]) == 64


async def test_trn_accepts_any_assertion_and_answers_with_a_space_prefixed_name():
    """UPSTREAM poke-env compares `split[2]` against `" " + username` — the leading space is
    load-bearing, and our fork `.strip()`s so it satisfies both."""
    front = _front()
    conn = await _named(front, "Alpha")
    assert conn.ws.sent[-1].startswith("|updateuser| Alpha|1|")
    assert front.users["alpha"] is conn


@pytest.mark.parametrize("name", ["A" * 19, "A" * 30])
async def test_a_name_longer_than_eighteen_characters_is_refused_with_the_servers_message(name):
    """foul-play de-risk hazard 1, reproduced LOUDLY.

    `server/users.ts:745` refuses a >18-char userid and does NOT log the user in. Foul Play
    ignores `|nametaken|` entirely, logs "Successfully logged in" and then waits forever — so a
    front end that quietly accepted the name would reproduce the hang with no cause in any log.
    """
    front = _front()
    conn = _Conn(ws=_FakeWS(), guest_n=1)
    await front._cmd_trn(conn, f"{name},0,assertion")
    assert conn.ws.sent[0] == "|nametaken||Your name must be 18 characters or shorter."
    assert not conn.named and front.users == {}


async def test_an_eighteen_character_name_is_accepted():
    """The boundary is INCLUSIVE — 18 is fine, 19 is not."""
    front = _front()
    conn = await _named(front, "A" * 18)
    assert conn.named


async def test_a_name_already_in_use_is_refused_rather_than_silently_stealing_the_connection():
    front = _front()
    first = await _named(front, "Alpha")
    second = _Conn(ws=_FakeWS(), guest_n=2)
    await front._cmd_trn(second, "alpha,0,assertion")
    assert second.ws.sent[0].startswith("|nametaken|alpha|Someone is already using the name")
    assert front.users["alpha"] is first


def test_to_id_matches_showdowns_rule():
    assert to_id("Big Bird 3!") == "bigbird3"
    assert to_id("  ") == ""


# --------------------------------------------------------------------------------------------
# teams
# --------------------------------------------------------------------------------------------
async def test_a_rejected_team_pops_up_and_does_not_take_effect(monkeypatch):
    """The server's shape, captured verbatim by the metamon de-risk, AND the consequence: the
    team is NOT set, so the next `/challenge` is refused instead of starting an impossible game."""
    monkeypatch.setattr(ws_frontend, "validate_teams_locally",
                        lambda fmt, teams: [{"valid": False,
                                             "errors": ['The Pokemon "airmureskarmory" '
                                                        'does not exist.']}])
    front = _front(validate_teams=True)
    conn = await _named(front, "Alpha")
    await front._cmd_utm(conn, "Airmure (Skarmory)|||keeneye|protect||||||")
    assert conn.team is None
    assert conn.ws.sent[-1] == ("|popup|Your team was rejected for the following reasons:||||"
                                '- The Pokemon "airmureskarmory" does not exist.')


async def test_a_valid_team_is_kept_and_the_validator_is_consulted_once_per_team(monkeypatch):
    calls = []

    def fake(fmt, teams):
        calls.append(teams[0])
        return [{"valid": True, "errors": []}]

    monkeypatch.setattr(ws_frontend, "validate_teams_locally", fake)
    front = _front(validate_teams=True)
    conn = await _named(front, "Alpha")
    for _ in range(3):
        await front._cmd_utm(conn, "Tyranitar||leftovers|H|rockslide|Adamant|252,252,0,0,4,0||||100|")
    assert conn.team is not None
    assert len(calls) == 1, "the Node validator must be cached, not respawned per battle"


async def test_a_packed_team_is_read_off_the_raw_line_not_a_pipe_split():
    """A packed team is FULL of `|`. Splitting the client line on `|` would truncate it at the
    first field and the battle would start with a one-field team."""
    front = _front()
    conn = await _named(front, "Alpha")
    packed = "Tyranitar||leftovers|H|rockslide,crunch|Adamant|252,252,0,0,4,0||||100|"
    await front._client_line(conn, f"|/utm {packed}")
    assert conn.team == packed


# --------------------------------------------------------------------------------------------
# challenges
# --------------------------------------------------------------------------------------------
async def test_the_challenge_pm_carries_the_nine_fields_both_clients_parse():
    """poke-env reads field 4 (`/challenge…`) and field 5 (the format); Foul Play reads those AND
    requires the message to have EXACTLY nine `|`-fields (`websocket_client.py:144-152`)."""
    front = _front()
    alpha = await _named(front, "Alpha")
    beta = await _named(front, "Beta")
    alpha.team = "packed"
    await front._cmd_challenge(alpha, "Beta, gen3ou")
    pm = beta.ws.sent[-1]
    fields = pm.split("|")
    assert len(fields) == 9, pm
    assert fields[1] == "pm" and fields[2].strip() == "Alpha" and fields[3].strip() == "Beta"
    assert fields[4].startswith("/challenge")
    assert fields[5] == "gen3ou"
    assert front.challenges[("alpha", "beta")][0] == "gen3ou"


async def test_a_challenge_to_someone_who_is_not_connected_pops_up_and_registers_nothing():
    """The real server drops it and the challenger waits forever — the reason `run_cell.sh` starts
    the acceptor first. Say so, rather than leaving a dangling challenge that can never resolve."""
    front = _front()
    alpha = await _named(front, "Alpha")
    alpha.team = "packed"
    await front._cmd_challenge(alpha, "Nobody, gen3ou")
    assert alpha.ws.sent[-1] == "|popup|The user 'Nobody' was not found."
    assert front.challenges == {}


async def test_a_challenge_without_a_team_is_refused():
    front = _front()
    alpha = await _named(front, "Alpha")
    await _named(front, "Beta")
    await front._cmd_challenge(alpha, "Beta, gen3ou")
    assert "select a team" in alpha.ws.sent[-1]
    assert front.challenges == {}


async def test_accepting_a_challenge_that_does_not_exist_pops_up_instead_of_hanging():
    front = _front()
    beta = await _named(front, "Beta")
    beta.team = "packed"
    await front._cmd_accept(beta, "Alpha")
    assert beta.ws.sent[-1] == "|popup|Alpha is not challenging you."


async def test_search_is_refused_with_a_message_that_names_the_alternative():
    front = _front()
    alpha = await _named(front, "Alpha")
    await front._client_line(alpha, "|/search gen3ou")
    assert "not implemented" in alpha.ws.sent[-1]
    assert "/challenge" in alpha.ws.sent[-1]


# --------------------------------------------------------------------------------------------
# the rqid injection — the decision trigger
# --------------------------------------------------------------------------------------------
async def test_every_request_gets_an_rqid_from_one_counter_shared_by_both_slots():
    """`server/room-battle.ts:796` increments a SINGLE `RoomBattle.rqid` for either slot, so the
    two sides' ids interleave into one 1..N sequence. A per-slot counter would hand both sides
    `1`, and Foul Play's `rqid` echo would then match the wrong request."""
    front = _front()
    battle = _fake_battle(front, await _named(front, "Alpha"), await _named(front, "Beta"))
    out = [front._note_side_line(battle, slot, _SIM_REQUEST)
           for slot in ("p1", "p2", "p1", "p2")]
    ids = [json.loads(ln[len("|request|"):])["rqid"] for ln in out]
    assert ids == [1, 2, 3, 4]
    assert battle.req["p1"].rqid == 3 and battle.req["p2"].rqid == 4


async def test_the_splice_leaves_every_other_byte_of_the_request_untouched():
    """The injection is a string splice precisely so the gate's normalization is one field."""
    front = _front()
    battle = _fake_battle(front, await _named(front, "Alpha"), await _named(front, "Beta"))
    out = front._note_side_line(battle, "p1", _SIM_REQUEST)
    assert out == _SIM_REQUEST[:-1] + ',"rqid":1}'
    assert json.loads(out[len("|request|"):])["side"]["name"] == "Alpha"


async def test_a_non_request_line_passes_through_byte_for_byte():
    front = _front()
    battle = _fake_battle(front, await _named(front, "Alpha"), await _named(front, "Beta"))
    line = "|move|p1a: Tyranitar|Rock Slide|p2a: Skarmory"
    assert front._note_side_line(battle, "p1", line) == line


async def test_a_wait_request_leaves_the_slot_with_nothing_to_choose():
    front = _front()
    battle = _fake_battle(front, await _named(front, "Alpha"), await _named(front, "Beta"))
    front._note_side_line(battle, "p1", '|request|{"wait":true,"side":{"id":"p1"}}')
    assert battle.req["p1"].is_wait == "cantUndo"


# --------------------------------------------------------------------------------------------
# choices
# --------------------------------------------------------------------------------------------
async def test_a_choice_before_any_request_is_refused_the_way_the_server_refuses_it():
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    await front._cmd_choose(alpha, battle.tag, "move 1", "")
    assert alpha.ws.sent[-1].endswith("|error|[Invalid choice] There's nothing to choose")
    assert battle.commands == []


async def test_a_live_request_forwards_the_choice_to_the_child():
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    front._note_side_line(battle, "p1", _SIM_REQUEST)
    await front._cmd_choose(alpha, battle.tag, "move earthquake", "1")
    assert battle.commands == ["CHOOSE p1 move earthquake"]


async def test_a_stale_rqid_is_refused_rather_than_replayed_onto_the_new_turn():
    """Foul Play echoes the `rqid` it decided against; the server's whole reason for checking it
    is that a decision from the previous turn must not land on this one."""
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    front._note_side_line(battle, "p1", _SIM_REQUEST)   # rqid 1
    front._note_side_line(battle, "p1", _SIM_REQUEST)   # rqid 2 — the turn moved on
    await front._cmd_choose(alpha, battle.tag, "move 1", "1")
    assert "too late" in alpha.ws.sent[-1]
    assert battle.commands == []


async def test_an_empty_rqid_skips_the_staleness_check_because_poke_env_sends_none():
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    front._note_side_line(battle, "p1", _SIM_REQUEST)
    await front._cmd_choose(alpha, battle.tag, "move 1", "")
    assert battle.commands == ["CHOOSE p1 move 1"]


async def test_an_invalid_choice_error_reopens_the_slot_so_the_client_may_choose_again():
    """poke-env answers `[Invalid choice]` by re-choosing off the SAME request. If the slot stayed
    closed the re-choice would be refused as "nothing to choose" and the battle would wedge."""
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    front._note_side_line(battle, "p1", _SIM_REQUEST)
    await front._cmd_choose(alpha, battle.tag, "move 1", "")
    assert battle.req["p1"].is_wait is True
    front._note_side_line(battle, "p1", "|error|[Invalid choice] Can't move: Invalid target")
    assert battle.req["p1"].is_wait is False
    await front._cmd_choose(alpha, battle.tag, "move 2", "")
    assert battle.commands == ["CHOOSE p1 move 1", "CHOOSE p1 move 2"]


async def test_an_unavailable_choice_does_not_reopen_the_slot():
    """Showdown only re-opens on `[Invalid choice]`; the `[Unavailable choice]` maybe-trapped
    reveal is followed by a FRESH `|request|`, which is what reopens it."""
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    front._note_side_line(battle, "p1", _SIM_REQUEST)
    await front._cmd_choose(alpha, battle.tag, "switch 2", "")
    front._note_side_line(battle, "p1",
                          "|error|[Unavailable choice] Can't switch: The active Pokémon is trapped")
    assert battle.req["p1"].is_wait is True


async def test_forfeit_reaches_the_child_as_the_bridges_own_verb():
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    await front._client_line(alpha, f"{battle.tag}|/forfeit")
    assert battle.commands == ["FORCELOSE p1"]


async def test_leaving_a_room_answers_with_deinit():
    """Both clients block on `|deinit|` after `/leave` — Foul Play's `leave_battle` loops on it."""
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    battle.finished = True
    await front._client_line(alpha, f"|/leave {battle.tag}")
    assert alpha.ws.sent[-1] == f">{battle.tag}\n|deinit"


async def test_a_disconnect_mid_battle_forfeits_that_side_instead_of_stranding_the_opponent():
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    await front._drop(alpha)
    assert battle.commands == ["FORCELOSE p1"]
    assert "alpha" not in front.users


async def test_the_timer_is_a_no_op_because_the_forfeit_is_client_side():
    """No battle timer, by design: the trainer forfeits at its own 250-turn cap, and a second
    server-side clock would be an unrecorded way to lose."""
    front = _front()
    alpha = await _named(front, "Alpha")
    battle = _fake_battle(front, alpha, await _named(front, "Beta"))
    await front._client_line(alpha, f"{battle.tag}|/timer on")
    assert battle.commands == []


async def test_userdetails_is_answered_because_foul_plays_avatar_call_blocks_on_it():
    front = _front()
    alpha = await _named(front, "Alpha")
    await front._client_line(alpha, "|/cmd userdetails Alpha")
    assert alpha.ws.sent[-1].startswith("|queryresponse|userdetails|")
    assert json.loads(alpha.ws.sent[-1].split("|", 3)[3])["name"] == "Alpha"


# --------------------------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------------------------
def test_the_stream_limit_matches_its_local_battle_runner_twin():
    """The constant is restated (not imported) to keep this module poke-env-free; that is only
    safe while something asserts the two cannot drift."""
    from utils.bridge.local_battle_runner import BRIDGE_STREAM_LIMIT as twin

    assert ws_frontend.BRIDGE_STREAM_LIMIT == twin


def test_the_two_reserved_ports_are_the_dev_and_training_servers():
    assert set(RESERVED_PORTS) == {8000, 8001}


def test_serving_on_a_reserved_port_is_refused_before_anything_binds():
    import asyncio

    args = ws_frontend.build_parser().parse_args(["--port", "8001"])
    with pytest.raises(SystemExit, match="8001"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            ws_frontend.serve_forever(args))


def test_a_side_request_starts_with_nothing_to_choose():
    assert _SideRequest().is_wait == "cantUndo"
