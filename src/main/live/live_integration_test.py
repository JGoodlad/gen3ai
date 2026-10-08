"""The live stack end to end: the reader binary, gate (a) on real websocket battles, and the T28 halt.

* the ``live_reader`` binary refuses an unknown battle keyword and stays dead for the battle;
* GATE (a) in miniature — two battles over our ``--server rust`` front end: every live row equals the training
  chain's (``sim_bridge`` core_obs) row, and a single perturbed byte is SEEN (the gate's teeth);
* T28 END TO END — a fake server plants an unknown keyword in a battle: ``main.play`` exits ``FATAL_LIVE_PARSE``
  with the marker written (battle id, the planted line), and the NEXT start is refused before any connection.
"""
import asyncio
import json
import socket
import threading

import pytest

from main.exit_codes import TrainExitCode
from main.live import halt as H
from main.live.reader import LiveReader, ReaderRefusal

pytestmark = pytest.mark.integration


def _free_port() -> int:
    for port in range(9640, 9700):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free 96xx port")


def _team() -> str:
    from main.play import STAR_TSS_TEAM
    from utils.teambuilder import Gen3Teambuilder
    return Gen3Teambuilder(STAR_TSS_TEAM).yield_team()


def test_the_reader_refuses_an_unknown_keyword_and_stays_dead():
    with LiveReader() as r:
        r.open("p1", "P4 A", _team())
        assert r.feed(["|t:|1", "|gametype|singles", "|player|p1|P4 A|1|", "|player|p2|P4 B|2|"]) is None
        with pytest.raises(ReaderRefusal) as ei:
            r.feed(["|zzznotakeyword|p2a: X"])
        assert "zzznotakeyword" in str(ei.value)
        with pytest.raises(ReaderRefusal):
            r.feed(["|"])  # sticky: the battle's reader never reads again
        r.open("p1", "P4 A", _team())  # a NEW battle opens clean
        assert r.feed(["|t:|1"]) is None


def test_gate_a_two_roads_one_row_in_miniature(tmp_path):
    from main.live import two_roads as TR
    port = _free_port()
    assert TR.main(["--battles", "2", "--pairs", "1", "--port", str(port), "--seed", "3",
                    "--out", str(tmp_path / "a")]) == 0
    summary = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert summary["verdict"] == "PASS" and summary["battles"] == 2 and summary["decisions"] > 20

    # TEETH: one flipped byte in one live row is reported, never averaged away.
    v = [json.loads(line) for line in (tmp_path / "a" / "verdicts.jsonl").read_text().splitlines()]
    cap = json.loads((tmp_path / "a" / "captures" / f"{v[0]['tag']}.json").read_text())
    from utils.bridge.sim_bridge_bin import resolve_sim_bridge_bin
    frames, _chunks, _err = TR.replay_core_obs(cap, [resolve_sim_bridge_bin()])
    live = {s: [{"raw": f, "rqid": 1} for f in frames[s]] for s in ("p1", "p2")}
    assert TR.compare_battle(live, cap, [resolve_sim_bridge_bin()])["ok"]
    obj = json.loads(live["p2"][3]["raw"])
    b64 = obj["frame"]["b64"]
    obj["frame"]["b64"] = ("A" if b64[100] != "A" else "B").join([b64[:100], b64[101:]])
    live["p2"][3]["raw"] = json.dumps(obj)
    bad = TR.compare_battle(live, cap, [resolve_sim_bridge_bin()])
    assert not bad["ok"] and bad["sides"]["p2"]["diffs"][0] == {"n": 3, "fields": ["frame"]}


class _PlantingServer:
    """A minimal websocket server: logs anyone in, and answers p4self1's /challenge with a battle whose
    second write carries a PLANTED unknown keyword. Counts connections (the refused restart makes none)."""

    PLANTED = "|zzplanted|p2a: Gengar|p1a: Skarmory"

    def __init__(self, port: int):
        self.port = port
        self.connections = 0
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import websockets
        asyncio.set_event_loop(self.loop)

        async def handler(ws):
            self.connections += 1
            await ws.send("|challstr|4|" + "ab" * 32)
            name = None
            async for msg in ws:
                body = msg.split("|", 1)[1] if "|" in msg else msg
                if body.startswith("/trn "):
                    name = body[5:].split(",")[0]
                    await ws.send(f"|updateuser| {name}|1|1|{{}}")
                elif body.startswith("/challenge "):
                    room = ">battle-gen3ou-77"
                    await ws.send(f"{room}\n|init|battle\n|title|{name} vs. p4self2\n|j|☆{name}\n")
                    await ws.send(f"{room}\n|t:|1\n|gametype|singles\n|player|p1|{name}|1|\n"
                                  f"|player|p2|p4self2|2|\n|gen|3\n{self.PLANTED}")

        async def main():
            self.stop = self.loop.create_future()
            async with websockets.serve(handler, "127.0.0.1", self.port):
                self.ready.set()
                await self.stop

        self.loop.run_until_complete(main())
        self.loop.close()

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(20)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.stop.set_result, None)
        self.thread.join(20)


def test_a_planted_bad_line_halts_and_refuses_the_next_start(tmp_path, monkeypatch, capsys):
    from main.play import run
    marker = tmp_path / "halt.json"
    monkeypatch.setenv(H.HALT_ENV, str(marker))
    port = _free_port()
    with _PlantingServer(port) as srv:
        with pytest.raises(SystemExit) as ei:
            run(["--mode", "selfplay", "--port", str(port), "--username", "p4self", "--connect-timeout", "20"])
        assert ei.value.code == int(TrainExitCode.FATAL_LIVE_PARSE)
        rec = H.read_halt()
        assert rec["battle_id"] == "battle-gen3ou-77"
        assert _PlantingServer.PLANTED in rec["offending_lines"]
        assert _PlantingServer.PLANTED in rec["stream_tail"]
        assert "zzplanted" in rec["reason"] and rec["entry_point"] == "main.play"
        assert "FATAL_LIVE_PARSE" in capsys.readouterr().err
        seen = srv.connections
        # the NEXT start is refused before it touches the socket
        assert run(["--mode", "selfplay", "--port", str(port), "--username", "p4self"]) == \
            int(TrainExitCode.FATAL_LIVE_PARSE)
        assert srv.connections == seen


def test_a_mid_battle_rename_flips_only_a_spectator_reading_never_a_players():
    """P4 gate (b)'s one refusal in 376,410 public replays, root-caused (fixture: that replay, names anonymized).
    The reading's identity is the player NAME (poke-env's ``_player_role`` rule). Read as p1 — the seat whose
    OPPONENT is renamed mid-battle (``|player|p2|PlayerB|``), the live client's case — it reads to the end. Read
    as p2 — the RENAMED seat, which a live client never is (its own name never changes in a battle) — the role
    flips and the reading refuses; the scan classifies exactly that, deterministically, from the seat's names."""
    from pathlib import Path

    from main.live import replay_scan as RS
    fixture = Path(__file__).parent / "testdata" / "viewer_seat_rename.log"
    lines = RS.battle_lines(fixture.read_text())
    assert RS.seat_names(lines) == {"p1": ["PlayerA"], "p2": ["Guest 1", "PlayerB"]}
    with LiveReader() as r:
        res = RS.scan_one(r, str(fixture), ("p1", "p2"), probe=True)
    assert [x["seat"] for x in res["refusals"]] == ["p2"]
    assert res["refusals"][0]["class"] == RS.VIEWER_SEAT_RENAMED
    assert "already has 6" in res["refusals"][0]["message"]
    assert not res["probe_failures"] and res["probes"] > 50
