"""The OTHER side of P4's gates (c) and (d): a poke-env client in its own process, accepting our challenges.

* ``--role bot`` — a scripted gen-3 bot from ``agents.opponents`` (gate (c)'s opponents);
* ``--role shadow`` — gate (d): OUR checkpoint on the LEGACY poke-env client (``main.play``'s ``RLPlayer`` path, the
  client P6 deletes), with a SHADOW :class:`main.live.reader.LiveReader` fed every battle message the poke-env
  client receives, BEFORE poke-env handles it. At each poke-env decision the shadow compares, decision by decision:
  poke-env's observation row against the reader's frame (BYTES), the action mask, the action the model picks on each
  row, and the choice token poke-env sends against the reader's token for that action. The token poke-env actually
  sent is noted on the shadow reader (``CHOOSE``) before the next message, exactly as the live client notes its own.

Run by ``main.live.master_series`` (it starts this process and waits for ``[peer] READY``); one JSON line per
decision goes to ``<out>/shadow.jsonl``. A shadow-side reader refusal is RECORDED (it is the finding), never raised
into poke-env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BOTS = ("Gen3HeuristicV2Player", "Gen3StallerV2Player", "Gen3AggressiveV2Player", "Gen3SetupSweepV2Player",
        "Gen3StallerPlayer", "Gen3AggressivePlayer", "Gen3SetupSweepPlayer")


class Shadow:
    """The Rust reader beside a poke-env player: per battle, the reader's view of the same messages."""

    def __init__(self, out: Path, model: Any) -> None:
        from main.live.reader import LiveReader
        self.out = open(out, "a")
        self.model = model
        self.reader = LiveReader()
        self.battles: Dict[str, Any] = {}
        self.team: Optional[str] = None
        self.username = ""
        self.pending: Dict[str, Any] = {}       # battle -> the reader's open frame (or None)
        self.last_pe: Dict[str, Dict[str, Any]] = {}

    def write(self, rec: Dict[str, Any]) -> None:
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()

    def on_message(self, room: str, lines: List[str]) -> None:
        from main.live.client import LiveBattle
        from main.live.halt import LiveParseHalt
        shadow = self

        class _ShadowBattle(LiveBattle):
            def decide(self, frame, write):  # the reader's decision: held for poke-env's embed to compare
                shadow.pending[self.room] = frame

        b = self.battles.get(room)
        if b is None:
            if not lines or lines[0] != "|init|battle":
                return
            b = _ShadowBattle(room, our_name=self.username, packed_team=self.team, reader=self.reader,
                              policy=None, forfeit_turn_limit=10 ** 9, send=lambda _t: None)  # type: ignore[arg-type]
            self.battles[room] = b
        if getattr(b, "dead", None):
            return
        # A forfeit ends the battle while a decision may still be open on either side (the server announces it at
        # once): from the first `forfeited.` notice on, a frame arriving before poke-env answered is not a race.
        if any(ln.startswith("|-message|") and ln.endswith(" forfeited.") for ln in lines):
            b.forfeit_seen = True
        if not b.ended and not getattr(b, "forfeit_seen", False) and (room in self.last_pe or room in self.pending):
            self.write({"battle": room, "kind": "race", "reason": "a frame arrived before poke-env answered the "
                        "open decision (the shadow's choice note would land late)", "lines": lines[:5]})
        try:
            b.on_lines(lines)
        except LiveParseHalt as exc:
            b.dead = exc.reason
            self.write({"battle": room, "kind": "reader_halt", "reason": exc.reason,
                        "offending": exc.offending_lines[:5]})

    def on_embed(self, tag: str, obs: Dict[str, Any]) -> None:
        frame = self.pending.pop(tag, None)
        row = np.asarray(obs["observation"], dtype=np.float32)
        mask = np.asarray(obs["action_mask"]).astype(np.int8)
        rec: Dict[str, Any] = {"battle": tag, "kind": "decision", "t": time.time()}
        if frame is None:
            rec["diff"] = ["no_reader_frame"]
            self.last_pe[tag] = {"rec": rec, "frame": None}
            return
        diff = []
        if row.tobytes() != np.asarray(frame.row, dtype=np.float32).tobytes():
            d = np.nonzero(row != frame.row)[0]
            diff.append("row")
            rec["row_cells"] = [int(i) for i in d[:20]]
            rec["row_ncells"] = int(len(d))
        if not np.array_equal(mask, frame.mask):
            diff.append("mask")
            rec["mask"] = {"poke_env": mask.tolist(), "reader": frame.mask.tolist()}
        rec.update(turn=frame.turn, n=frame.n, diff=diff)
        rec["reader_argmax"] = self._argmax(frame.row, frame.mask)
        self.last_pe[tag] = {"rec": rec, "frame": frame}

    def _argmax(self, row: np.ndarray, mask: np.ndarray) -> int:
        import torch
        with torch.no_grad():
            o = torch.as_tensor(np.array(row, dtype=np.float32, copy=True)[None])
            m = torch.as_tensor(mask.astype(np.float32)[None])
            logits = self.model.policy.get_distribution({"observation": o, "action_mask": m}).distribution.logits
            return int(torch.argmax(logits + (m - 1.0) * 1e9, dim=1).item())

    def on_choice(self, tag: str, idx: Optional[int]) -> None:
        st = self.last_pe.get(tag)
        if st is not None:
            st["rec"]["poke_env_idx"] = idx

    def on_send(self, room: str, message: str) -> None:
        if message.startswith("/utm "):
            self.team = message[len("/utm "):] if message != "/utm null" else None
            return
        if message.startswith("/forfeit") and room not in self.last_pe:
            # The trainer's STALL forfeit: poke-env forfeits at the turn limit BEFORE it encodes (`_handle_stall`),
            # so there is no poke-env row to compare — the reader's open frame is closed here, recorded as such.
            frame = self.pending.pop(room, None)
            if frame is not None:
                self.write({"battle": room, "kind": "decision", "turn": frame.turn, "n": frame.n, "diff": [],
                            "sent": "/forfeit", "stall_forfeit": True})
            return
        st = self.last_pe.pop(room, None)
        if st is None:
            return
        rec, frame = st["rec"], st["frame"]
        if message.startswith("/choose "):
            token = message[len("/choose "):].split("|", 1)[0]
            rec["sent"] = token
            if frame is not None:
                idx = rec.get("poke_env_idx")
                if idx is not None and rec.get("reader_argmax") != idx:
                    rec["diff"].append("action")
                if idx is not None and frame.tokens.get(idx) != token:
                    rec["diff"].append("token")
                    rec["reader_token"] = frame.tokens.get(idx)
                b = self.battles.get(room)
                if b is not None and not getattr(b, "dead", None):
                    try:
                        self.reader.choose(token)
                    except Exception as exc:  # noqa: BLE001
                        b.dead = str(exc)
                        rec["diff"].append("reader_choose_refused")
        elif message.startswith("/forfeit"):
            rec["sent"] = "/forfeit"
        self.write(rec)


async def run(a: argparse.Namespace) -> int:
    import main.play as play
    from main.play import build_account, resolve_server
    from utils.teambuilder import Gen3Teambuilder

    server = resolve_server("local", a.port)
    account = build_account(a.username, None, "local")
    if a.team_pool:
        from utils.team_loader import TeamLoader
        tb = Gen3Teambuilder(TeamLoader().get_all_teams())
    else:
        tb = Gen3Teambuilder(play.STAR_TSS_TEAM)
    if a.role == "bot":
        import agents.opponents as O
        player = getattr(O, a.bot)(battle_format=a.format, team=tb, server_configuration=server,
                                   account_configuration=account, max_concurrent_battles=1)
    else:
        ns = play.build_parser().parse_args(["--client", "poke-env", "--model", a.model, "--port", str(a.port),
                                             "--format", a.format, "--temperature", "0"])
        player = play.build_model_player(ns, tb, server, account)
        shadow = Shadow(Path(a.out) / "shadow.jsonl", player.model)
        shadow.username = a.username
        # Feed the shadow at the WEBSOCKET FRAME, in arrival order, before poke-env sees it: the fork's listen loop
        # concatenates frames after a request ("SMART FLUSH") and handles each message in its own task, so its
        # handler's view of the write boundaries is not the server's. Patched on the connection CLASS (this
        # process holds exactly one connection).
        import websockets.asyncio.client as wac
        orig_recv = wac.ClientConnection.recv

        async def recv(conn, *args, **kw):
            msg = await orig_recv(conn, *args, **kw)
            from main.live.reader import split_room_message
            room, lines = split_room_message(str(msg))
            if room and room.startswith("battle-"):
                shadow.on_message(room, lines)
            return msg

        wac.ClientConnection.recv = recv
        orig_embed = player.embed_battle

        def embed(battle):
            obs = orig_embed(battle)
            if int(np.asarray(obs["action_mask"]).sum()) > 0:
                shadow.on_embed(battle.battle_tag, obs)
            return obs

        player.embed_battle = embed
        orig_pred = player._predict_best_action

        def pred(battle, *args, **kw):
            idx, probs, mask = orig_pred(battle, *args, **kw)
            shadow.on_choice(battle.battle_tag, idx)
            return idx, probs, mask

        player._predict_best_action = pred
        orig_send = player.ps_client.send_message

        async def send(message, room="", message_2=None):
            shadow.on_send(room, message)
            return await orig_send(message, room, message_2)

        player.ps_client.send_message = send
    deadline = time.time() + 120
    while not player.ps_client.logged_in.is_set():
        if time.time() > deadline:
            print("[peer] login timed out", flush=True)
            return 3
        await asyncio.sleep(0.1)
    print(f"[peer] READY {a.username}", flush=True)
    await player.accept_challenges(a.opponent, a.n)
    print(f"[peer] DONE finished={player.n_finished_battles} won={player.n_won_battles}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.live.gate_peer")
    ap.add_argument("--role", choices=("bot", "shadow"), required=True)
    ap.add_argument("--bot", choices=BOTS, default="Gen3HeuristicV2Player")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--team-pool", action="store_true")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    from main.play import check_port
    check_port(a.port)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
