"""The BOT READER's Python half (P6 of the poke-env retirement), on a FAKE ``bot_reader`` process (no Rust build).

Pinned: the bot is installed ONCE per process before the first battle (``BOT`` with the route seed and the env
index), a decision's reply is ``__BOT__`` + ``__OBS__`` + ``__FED__`` and becomes a :class:`BotFrame` carrying the
bot's choice; a frame without the bot's choice (or a choice without a frame) is a refusal, never a guess; a process
that died mid-series is REFUSED rather than respawned (a respawn would restart the bot's RNG streams silently); and
:class:`BotPolicy` plays the bot's index only when the frame's token at that index is the bot's token.

The real binary on real battles: ``src/main/anchors/live_side_test.py`` (``sim``) and the Rust gate
``src/rust_env/tests/bot_side_test.rs`` (the bot over a side stream = the env core's bot, decision for decision).
"""
import json
import sys
import textwrap

import numpy as np
import pytest

from main.live.bot_reader import BotFrame, BotPolicy, BotReader
from main.live.reader import Frame, ReaderRefusal

_FAKE = textwrap.dedent(r'''
    import base64, json, sys
    mode = sys.argv[1]
    log = open(sys.argv[2], "a")
    row = base64.b64encode(bytes(12)).decode()
    for line in sys.stdin:
        line = line.rstrip("\n")
        log.write(line + "\n"); log.flush()
        cmd = line.split(" ", 1)[0]
        if cmd == "END":
            break
        if cmd in ("BOT", "OPEN", "CHOOSE", "CLOSE"):
            print("__OK__", flush=True); continue
        if cmd == "FEED":
            lines = json.loads(line[5:])
            if any(l.startswith("|request|") for l in lines):
                frame = {"frame": {"dtype": "<f4", "shape": [3], "b64": row}, "mask": [1, 1] + [0] * 9,
                         "tokens": {"0": "move tackle", "1": "switch Blissey"}, "turn": 2, "line": 9,
                         "rqid": 4, "n": 0}
                bot = {"index": 1, "token": "switch Blissey", "choice_words": 2, "protect_words": 0, "branch": 7}
                if mode != "noframe":
                    if mode != "nobot":
                        print("__BOT__ " + json.dumps(bot), flush=True)
                    print("__OBS__ p1 " + json.dumps(frame), flush=True)
                elif mode == "noframe":
                    print("__BOT__ " + json.dumps(bot), flush=True)
            print('__FED__ {"folded":%d,"decided":0,"turn":2}' % len(lines), flush=True)
            if mode == "die":
                sys.exit(0)
''')


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / "fake_bot_reader.py"
    script.write_text(_FAKE)
    log = tmp_path / "commands.log"

    def make(mode="ok"):
        return BotReader("staller", seed=5, env=1, argv=[sys.executable, str(script), mode, str(log)]), log
    return make


def test_the_bot_is_installed_once_before_the_first_battle_and_a_decision_carries_its_choice(fake):
    r, log = fake()
    try:
        r.open("p1", "ours", None)
        f = r.feed(["|request|{}"])
        r.choose("switch Blissey")
        r.open("p1", "ours", None)          # a second battle: the bot is NOT re-installed (its streams carry over)
        assert r.feed(["|turn|1"]) is None
    finally:
        r.close()
    cmds = [ln.split(" ", 1)[0] for ln in log.read_text().splitlines()]
    assert cmds == ["BOT", "OPEN", "FEED", "CHOOSE", "OPEN", "FEED", "END"], cmds
    assert json.loads(log.read_text().splitlines()[0][4:]) == {"bot": "staller", "seed": 5, "env": 1}
    assert isinstance(f, BotFrame) and f.bot["token"] == "switch Blissey" and f.tokens[1] == "switch Blissey"
    assert f.row.shape == (3,) and f.rqid == 4 and r.choices == [f.bot]


@pytest.mark.parametrize("mode, needle", [("nobot", "without the bot's choice"),
                                          ("noframe", "without its decision frame")])
def test_a_half_reply_is_refused(fake, mode, needle):
    r, _ = fake(mode)
    try:
        r.open("p1", "ours", None)
        with pytest.raises(ReaderRefusal, match=needle):
            r.feed(["|request|{}"])
    finally:
        r.close()


def test_a_reader_that_died_mid_series_is_refused_not_respawned(fake):
    r, log = fake("die")
    try:
        r.open("p1", "ours", None)
        r.feed(["|request|{}"])               # the fake exits after this reply
        r._proc.wait(timeout=10)
        with pytest.raises(ReaderRefusal, match="died mid-series"):
            r.open("p1", "ours", None)
    finally:
        r.close()
    assert [ln.split(" ", 1)[0] for ln in log.read_text().splitlines()].count("BOT") == 1, "a respawn re-installed"


def _frame(bot, tokens=None):
    return BotFrame(side="p1", row=np.zeros(3, np.float32), mask=np.asarray([1, 1] + [0] * 9, np.int8),
                    tokens=tokens or {0: "move tackle", 1: "switch Blissey"}, turn=1, line=3, rqid=1, n=0, bot=bot)


def test_the_bot_policy_plays_the_bots_index_only_when_the_token_agrees():
    p = BotPolicy()
    assert p.choose(_frame({"index": 1, "token": "switch Blissey"})) == 1
    with pytest.raises(ValueError, match="frame's token there"):
        p.choose(_frame({"index": 0, "token": "switch Blissey"}))
    with pytest.raises(ValueError, match="no bot choice"):
        p.choose(Frame(side="p1", row=np.zeros(3, np.float32), mask=np.ones(11, np.int8), tokens={0: "move 1"},
                       turn=1, line=1, rqid=None, n=0))
