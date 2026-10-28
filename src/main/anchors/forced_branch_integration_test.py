"""The forced branch on REAL in-process battles (the deviation-only live read's B-V1): a raw battle keeps its decisions'
input logs; the same deal is then replayed three times on fresh front ends (same seed base, battle index and teams):
* a BRANCH forcing a non-greedy action at a kept decision: G-BRANCH passes there, the forced action is played, and the
  two games' protocol is byte-identical through that decision's turn (``main.reads.prefix.compare``);
* a NULL control forcing π's own action at the same decision: the WHOLE game replays the raw one byte for byte;
* a spec whose honest-log digest is tampered: the replay fails by name at the decision (G-BRANCH), before any action.
A stub network plays both sides (``search_side_integration_test``'s)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from main.anchors import core_side
from main.anchors import forced_branch as FB
from main.anchors.input_log import InputLogRecorder
from main.anchors.search_side_integration_test import TEAM_A, TEAM_B, _Greedy, _Team
from main.anchors.session import OurSideState
from main.reads.prefix import compare
from main.value_search.live_branch import WHOLE_GAME_TURN

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

BASE, INDEX = 11, 3


def _battle(cap: Path, *, recorder: Any = None, forcer: Any = None) -> Dict[str, Any]:
    async def play() -> Dict[str, Any]:
        from utils.bridge.ws_frontend import ShowdownFrontEnd

        front = ShowdownFrontEnd(impl="rust", seed_base=BASE, validate_teams=False, battle_index_base=INDEX,
                                 capture_dir=str(cap))
        st_a, st_b = OurSideState(), OurSideState()
        a = core_side.make_slot(front, username="Alpha", policy=_Greedy(), team_source=_Team(TEAM_A), state=st_a,
                                forfeit_limit=16, recorder=recorder, forcer=forcer)
        b = core_side.make_slot(front, username="Beta", policy=_Greedy(), team_source=_Team(TEAM_B), state=st_b,
                                forfeit_limit=250)
        try:
            await a.login()
            await b.login()
            tb = asyncio.create_task(b.accept_series("Alpha", 1))
            await asyncio.sleep(0)
            try:
                await asyncio.wait_for(a.challenge_series("Beta", 1, 120.0), timeout=900.0)
                await asyncio.wait_for(tb, timeout=120.0)
            except Exception:                                  # noqa: BLE001 - the slot's failure is the result
                tb.cancel()
        finally:
            await front.close()
        caps = sorted(cap.glob("*.json"))
        return {"failure": a.failure, "record": st_a.records[-1] if st_a.records else None,
                "capture": json.loads(caps[0].read_text()) if caps else None}

    return asyncio.run(play())


def _pick(rows: Any) -> Dict[str, Any]:
    for r in rows:
        legal = [int(k) for k in r["tokens"]]
        if len(legal) >= 2 and r["turn"] >= 4:
            return r
    raise AssertionError("the fixture battle kept no decision with >= 2 legal actions from turn 4")


def test_a_forced_branch_shares_the_raw_prefix_and_a_null_branch_replays_the_raw_game(tmp_path) -> None:
    kept = tmp_path / "input_logs.jsonl"
    raw = _battle(tmp_path / "raw", recorder=InputLogRecorder("2-14", kept))
    assert raw["failure"] is None and raw["capture"] is not None
    rec = _pick([json.loads(x) for x in kept.read_text().splitlines()])
    assert rec["index"] == INDEX and rec["battle"] == raw["capture"]["tag"]
    other = next(int(k) for k in sorted(rec["tokens"], key=int) if int(k) != rec["pi_argmax"])

    spec = FB.spec_from_kept(rec, other, game=INDEX)
    fb = FB.ForcedBranch(spec, tmp_path / "branch" / "branch.jsonl")
    br = _battle(tmp_path / "branch", forcer=fb)
    assert br["failure"] is None, br["failure"]
    s = fb.summary()
    assert s["forced"] == 1 and s["g_branch"] == "pass" and FB.check_half(s, spec, 1) is None
    line = json.loads((tmp_path / "branch" / "branch.jsonl").read_text().splitlines()[-1])
    assert line["forced"] == other and line["pi_argmax"] == rec["pi_argmax"]
    cmp = compare(raw["capture"], br["capture"], int(rec["turn"]))
    assert cmp["ok"] and all(cmp["reached"].values()), cmp
    assert br["record"].our_argmax_matches == br["record"].our_argmax_decisions - 1

    nul = FB.ForcedBranch(FB.spec_from_kept(rec, rec["pi_argmax"], game=INDEX), tmp_path / "null" / "branch.jsonl")
    nb = _battle(tmp_path / "null", forcer=nul)
    assert nb["failure"] is None and nul.summary()["forced"] == 1
    whole = compare(raw["capture"], nb["capture"], WHOLE_GAME_TURN)
    assert whole["ok"], whole
    assert (nb["record"].won, nb["record"].turns) == (raw["record"].won, raw["record"].turns)

    bad = FB.ForcedBranch(dict(spec, log_sha256="0" * 64), tmp_path / "bad" / "branch.jsonl")
    bb = _battle(tmp_path / "bad", forcer=bad)
    failure: Optional[Any] = bb["failure"]
    assert failure is not None and "G-BRANCH" in failure.detail and "log_sha256" in failure.detail
    assert json.loads((tmp_path / "bad" / "branch.jsonl").read_text())["g_branch"] == "fail"
