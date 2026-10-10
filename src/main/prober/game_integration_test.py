"""`/game` end to end on a REAL battle and a CURRENT-architecture checkpoint.

The battles are real Rust-core games written as core traces (`core_trace_integration_test`'s
`_play_core_games` — p1 a seeded random-legal policy, p2 the in-core `heuristic` bot), and the model is
a freshly built, seeded checkpoint at the PRODUCTION surface (`main.fresh_checkpoint`) — the
"current-architecture checkpoint" the owner's 2026-10-08 scope ruling asks for (no trained one exists
at HEAD's config yet). Every check reads the core-encoded obs the trace stored.

Needs the rust env cdylib and `core_events` (built on first use) — `sim`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

SEED = 5


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from main.fresh_checkpoint import save_fresh_checkpoint
    from main.prober.core_trace_integration_test import _play_core_games, build_core_lib

    lib = build_core_lib()
    run_dir = Path(tmp_path_factory.mktemp("game_run"))
    paths = _play_core_games(lib, str(run_dir), 2, turn_limit=250, seed=21, tag="g")
    save_fresh_checkpoint(run_dir, SEED)        # the run's last snapshot: the ladder's "recent" rung
    return str(run_dir), paths


@pytest.fixture(scope="module")
def sess(run):
    from main.prober.session import ProbeSession
    s = ProbeSession(run[0])
    yield s
    s.close()


def test_the_story_carries_every_turns_events_and_board(run, sess):
    sp = run[1][0]
    st = sess.battle_story(sp)
    assert st["n_decisions"] == len(sess._summary(sess._battle(sp))["invocations"])
    kinds = {e["kind"] for t in st["turns"] for e in t["events"]}
    assert {"switch", "move"} <= kinds
    assert all("board" in t for t in st["turns"])
    played = [t for t in st["turns"] if t["decisions"]]
    assert played and all(st["decisions"][i]["turn"] == t["turn"] for t in played for i in t["decisions"])
    assert any(m["active"] for m in st["turns"][-1]["board"]["we"]["mons"])



# The phases a turn may pass through, in the order the sim runs them (designs/prober/
# battle_viewer_ux_2026-10-09.md §5). A `replace` is the one phase with no fixed rank: gen 3 sends a
# replacement in straight after the action that KO'd its predecessor — between two moves, or after
# the end-of-turn residuals when a residual did the KO — so it is checked by what it FOLLOWS (a faint
# on its own side), and every other phase is monotone.
_PHASE_RANK = {"lead": 0, "start": 1, "switch": 2, "move": 3, "residual": 4, "end": 5}


def test_every_turn_of_a_real_battle_reads_as_ordered_beats(run, sess):
    """The phase ORDER inside every turn of real Rust-core battles: start → switches → moves and
    replacements in execution order (numbered 1, 2, … with exactly one "moved first") → end of turn →
    the result; every replacement follows a faint on its own side; every faint names a cause."""
    checked = 0
    for sp in run[1]:
        st = sess.battle_story(sp)
        for t in st["turns"]:
            beats = t["beats"]
            ranks = [_PHASE_RANK[b["phase"]] for b in beats if b["phase"] != "replace"]
            assert ranks == sorted(ranks), (t["turn"], [b["phase"] for b in beats])
            orders = [b["order"] for b in beats if b["phase"] == "move"]
            assert orders == list(range(1, len(orders) + 1)), (t["turn"], orders)
            assert sum(1 for b in beats if b.get("first")) == (1 if orders else 0)
            fainted: set = set()
            for b in beats:
                if b["phase"] == "replace":
                    assert b["side"] in fainted, f"turn {t['turn']}: a replacement with no faint before it"
                    fainted.discard(b["side"])
                for e in b["effects"]:
                    if e["kind"] == "faint":
                        assert e.get("cause"), f"turn {t['turn']}: a faint with no cause"
                        fainted.add(e["side"])
            assert all(e.get("phase") for e in t["events"]), "an event was left outside every beat"
            checked += len(beats)
    assert checked > 20, "the battles were too short to read anything"

def test_the_readout_re_runs_the_policy_and_reads_every_panel(run, sess):
    sp = run[1][0]
    b = sess._battle(sp)
    r = sess.battle_readout(sp)
    npz = sess._npz(b)
    assert r["n_decisions"] == len(npz["obs"])
    assert r["model_resolution"]["path"].endswith("final_model.zip")
    # the probabilities are the policy's own (an independent batched path on the same model)
    model, _ = sess._model_for(b)
    probs = model.action_probs_batch(npz["obs"], npz["action_mask"])
    got = np.array([[a["p"] for a in d["policy"]] for d in r["decisions"]])
    assert np.allclose(got, probs, atol=1e-5)
    assert all(r["has"][k] for k in ("attention", "intent_p", "slot_species", "op", "scores", "win_prob"))
    assert r["has"]["mr_move"] is False                     # move-resolution is off in production
    d = r["decisions"][0]
    assert sum(c["p"] for c in d["intent"]["candidates"]) == pytest.approx(1.0, abs=1e-4)
    assert len(d["hypotheses"]["slots"]) == 6
    assert d["attention"]["n_layers"] == 2 and d["attention"]["rows"]
    assert r["layout"]["ok"] and r["layout"]["n_tokens"] >= 30
    labelled = [x for x in r["decisions"] if x["intent"] and x["intent"]["actual_col"] is not None]
    assert labelled, "a real battle against a bot must carry opponent labels"
    assert r["intent_calibration"]["n"] == len(labelled)
    # `/analyze`'s operator view decodes on the production surface (`op_drop_renders` ON): it used to
    # read the un-serialized matrices past the row's end, and `analyze` swallowed that as "no op"
    dop = model.damage_op_view(npz["obs"][0], npz["action_mask"][0])
    assert dop is not None and dop["outgoing"] is not None and dop["incoming_matrix"] is None
    # the capture leaves no hook behind: the model is exactly as it was
    fe = model._policy.features_extractor
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in fe.modules())


def test_the_attention_map_is_a_distribution_per_live_row(run, sess):
    sp = run[1][0]
    a = sess.decision_attention(sp, 0)
    m = np.asarray(a["matrix"])
    T = len(a["labels"])
    assert m.shape == (T, T)
    live = [i for i in range(T) if not a["masked"][i]]
    assert np.allclose(m[live].sum(-1), 1.0, atol=1e-3)
    one = np.asarray(sess.decision_attention(sp, 0, layer=1, head=3)["matrix"])
    assert one.shape == (T, T) and not np.allclose(one, m)
    with pytest.raises(IndexError):
        sess.decision_attention(sp, 0, layer=9)


def test_the_web_views_render_the_model_panels(run):
    from fastapi.testclient import TestClient

    from main.prober.web.app import create_app

    with TestClient(create_app(run[0], open_access=True)) as c:
        battle = c.get("/api/battles").json()[0]["short_id"]
        page = c.get("/game", params={"battle": battle, "inv": "1"})
        assert page.status_code == 200 and 'class="rail' in page.text and 'data-nav="next"' in page.text
        assert 'class="beats"' in page.text and 'class="board2"' in page.text
        frag = c.get("/partials/game/model", params={"battle": battle, "inv": "1"})
        assert frag.status_code == 200 and 'data-model-state="ok"' in frag.text
        assert "What we expected them to do" in frag.text and "Where the model looked" in frag.text
        # the model's α lands BESIDE our options (out of band), sorted by probability
        assert 'id="game-intent"' in frag.text and 'hx-swap-oob="true"' in frag.text
        import re
        intent = frag.text.split('data-dist="intent"', 1)[1].split("</ul>", 1)[0]
        ps = [float(x) for x in re.findall(r'<span class="dp">(\d+)%</span>', intent)]
        assert ps and ps == sorted(ps, reverse=True), ps
        att = c.get("/partials/game/attention", params={"battle": battle, "inv": "1", "layer": "0"})
        assert att.status_code == 200 and "vega-spec" in att.text
        api = c.get("/api/game/readout", params={"battle": battle}).json()
        assert api["n_decisions"] > 0 and "decisions" in api
        story = c.get("/api/game/story", params={"battle": battle}).json()
        assert story["turns"] and story["decisions"]
        assert os.path.basename(run[1][0]).startswith(("win", "loss", "draw"))
