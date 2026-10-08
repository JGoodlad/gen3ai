"""A SCRIPTED BOT as a live client's reader + policy (poke-env retirement P6; ``main.anchors``' ``bot:<name>``).

:class:`BotReader` is :class:`main.live.reader.LiveReader` over the env core's ``bot_reader`` binary
(``src/rust_env/src/bin/bot_reader.rs``): the SAME reader chain (``pokesim::side_reader::SideReader``) folds our
side's lines, and at every decision the Rust port of the roster bot (``src/rust_env/src/bots/``, gated
action-equal to the Python bots per decision) decides on that side's reading over the decision's own choice
tokens. Its :class:`BotFrame` carries the bot's choice; :class:`BotPolicy` plays it. Nothing here imports
poke-env: a ``bot:`` our-side used to be a poke-env ``Player`` (``agents.opponents`` / the fork's baselines).

RANDOMNESS (the env core's declared rule, ``opponents::stream_seed``): ONE bot per reader process, installed
before the first battle with the route seed ``seed`` and the env index ``env``; its ``choice`` / ``protect``
streams are ``random.Random(stream_seed(seed, env, k))`` and RUN ACROSS the reader's battles (one ``Player`` per
half-series). The Python twin is ``agents.training.rust_env_opponents.bot_stream_seed``. A reader whose process
died mid-series is REFUSED rather than respawned: a fresh process would restart the streams silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from main.live.reader import Frame, LiveReader, ReaderRefusal


@dataclass
class BotFrame(Frame):
    """A :class:`Frame` plus the bot's choice at it (``{"index","token","choice_words","protect_words","branch"}``)."""
    bot: Dict[str, Any] = field(default_factory=dict)


def bot_reader_argv() -> List[str]:
    from utils.rust_env.bot_reader_bin import resolve_bot_reader_bin
    return [resolve_bot_reader_bin()]


class BotReader(LiveReader):
    """One ``bot_reader`` child: the bot ``bot`` (a ``bots::Kind`` name) with route seed ``seed``, env ``env``."""

    def __init__(self, bot: str, *, seed: int, env: int, argv: Optional[Sequence[str]] = None) -> None:
        super().__init__(argv if argv is not None else bot_reader_argv())
        self.bot, self.seed, self.env = str(bot), int(seed), int(env)
        self._installed_pid: Optional[int] = None
        self.choices: List[Dict[str, Any]] = []

    def _install(self) -> None:
        proc = self._spawn()
        if self._installed_pid is None:
            self._send("BOT " + json.dumps({"bot": self.bot, "seed": self.seed, "env": self.env}))
            self._expect("__OK__")
            self._installed_pid = proc.pid
        elif proc.pid != self._installed_pid:
            raise ReaderRefusal(f"the bot_reader process for {self.bot!r} died mid-series; a respawn would restart "
                                "its RNG streams, so the series cannot continue", "process")

    def open(self, side: str, name: str, packed_team: Optional[str]) -> None:
        self._dead = None
        self._install()
        super().open(side, name, packed_team)

    def feed(self, lines: Sequence[str]) -> Optional[Frame]:
        self._alive()
        for ln in lines:
            if "\n" in ln or "\r" in ln:
                raise ReaderRefusal(f"a line holding a newline: {ln[:120]!r}", "input")
        self._send("FEED " + json.dumps(list(lines), ensure_ascii=False))
        frame: Optional[Frame] = None
        bot: Optional[Dict[str, Any]] = None
        while True:
            ln = self._read()
            if ln.startswith("__ERR__ "):
                obj = json.loads(ln[len("__ERR__ "):])
                self._dead = obj.get("message", ln)
                raise ReaderRefusal(obj.get("message", ln), obj.get("kind", "reader"))
            if ln.startswith("__BOT__ "):
                bot = json.loads(ln[len("__BOT__ "):])
            elif ln.startswith("__OBS__ "):
                _, side, payload = ln.split(" ", 2)
                frame = Frame.parse(side, payload)
            elif ln.startswith("__FED__ "):
                st = json.loads(ln[len("__FED__ "):])
                self.folded, self.decided, self.turn = int(st["folded"]), int(st["decided"]), int(st["turn"])
                break
            else:
                raise ReaderRefusal(f"unexpected reply line {ln[:200]!r}", "protocol")
        if frame is None:
            if bot is not None:
                raise ReaderRefusal("a bot choice without its decision frame", "protocol")
            return None
        if bot is None:
            raise ReaderRefusal("a decision frame without the bot's choice", "protocol")
        self.choices.append(bot)
        return BotFrame(**{k: getattr(frame, k) for k in Frame.__dataclass_fields__}, bot=bot)


class BotPolicy:
    """Plays the bot's own choice (``BotFrame.bot``), after checking it is the frame's token for that index."""

    def choose(self, frame: Frame) -> int:
        bot = getattr(frame, "bot", None)
        if not bot or bot.get("index") is None:
            raise ValueError("a bot decision with no bot choice on the frame")
        idx = int(bot["index"])
        if frame.tokens.get(idx) != bot.get("token"):
            raise ValueError(f"the bot chose {bot.get('token')!r} at index {idx}, but the frame's token there is "
                             f"{frame.tokens.get(idx)!r}")
        return idx
