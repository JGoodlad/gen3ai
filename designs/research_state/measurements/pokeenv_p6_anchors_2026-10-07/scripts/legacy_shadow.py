"""THE OLD PATH of P6's `--server node` proof, in SHADOW mode — a COMPARISON-HARNESS hook, never production.

A Node Showdown mints its own battle seed, so an old-vs-new pair of `--server node` reads cannot play the same
battles. Instead (P4 gate (d)'s method): the LEGACY anchors client (`--our-transport poke-env`: `main.play` →
`RLPlayer`, vendored poke-env + the Python encoder) plays the read, and a SHADOW
:class:`main.live.reader.LiveReader` — the NEW path's reader — is fed every websocket frame that client receives,
in arrival order, before poke-env handles it. At each poke-env decision `main.live.gate_peer.Shadow` compares
poke-env's row against the reader's frame (BYTES), the two masks, the checkpoint's argmax on each row, and the
token poke-env SENT against the reader's token for that action; the sent token is then noted on the shadow
reader. One JSON line per decision goes to `<--out>/<half>/shadow.jsonl`.

The hook wraps `main.anchors.runner.install_our_side` so the RLPlayer each half builds carries the taps
(the same four `gate_peer.run` installs: the connection's `recv`, `embed_battle`, `_predict_best_action`,
`ps_client.send_message`). Nothing else about the old path changes.

    PYTHONPATH=<wt>/src python legacy_shadow.py --our-transport poke-env --server node --model <zip> \\
        --opponent metamon:SmallRL --games N --out <dir>
"""
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])

    import numpy as np
    import websockets.asyncio.client as wac

    import main.anchors.runner as R
    import main.play as play
    from main.live.gate_peer import Shadow
    from main.live.reader import split_room_message

    current = {"shadow": None}
    orig_recv = wac.ClientConnection.recv

    async def recv(conn, *args, **kw):
        msg = await orig_recv(conn, *args, **kw)
        sh = current["shadow"]
        if sh is not None:
            room, lines = split_room_message(str(msg))
            if room and room.startswith("battle-"):
                sh.on_message(room, lines)
        return msg

    wac.ClientConnection.recv = recv
    orig_install = R.install_our_side
    halves = ["ours_challenge", "peer_challenge"]
    calls = {"n": 0}

    def install(*a, **k):
        undo = orig_install(*a, **k)
        half = halves[calls["n"]]
        calls["n"] += 1
        inner = play.build_model_player

        def build(*aa, **kk):
            player = inner(*aa, **kk)
            (out / half).mkdir(parents=True, exist_ok=True)
            shadow = Shadow(out / half / "shadow.jsonl", player.model)
            shadow.username = player.username
            current["shadow"] = shadow
            orig_embed = player.embed_battle

            def embed(battle):
                obs = orig_embed(battle)
                if int(np.asarray(obs["action_mask"]).sum()) > 0:
                    shadow.on_embed(battle.battle_tag, obs)
                return obs

            player.embed_battle = embed
            orig_pred = player._predict_best_action

            def pred(battle, *args, **kw):
                res = orig_pred(battle, *args, **kw)
                shadow.on_choice(battle.battle_tag, res[0])
                return res

            player._predict_best_action = pred
            orig_send = player.ps_client.send_message

            async def send(message, room="", message_2=None):
                shadow.on_send(room, message)
                return await orig_send(message, room, message_2)

            player.ps_client.send_message = send
            print(f"[harness] shadow reader on {player.username} -> {out / half / 'shadow.jsonl'}", flush=True)
            return player

        play.build_model_player = build
        return undo

    R.install_our_side = install
    sys.argv = ["main.anchors"] + argv
    from main.anchors.cli import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
