"""M5 Lane I's GATES on a real build — search on ``successors()`` in process
(``designs/endstate/program_rust_core.md`` §2 M5; progress: ``designs/research_state/measurements/m5_laneI/``).

THE THREE SEARCH GATES for the in-process road (the M2 trio — ``materializer_parity``,
``fork_sharing_parity``, ``one_sided_view_parity`` — was deleted with the roads they compared in
``43712881``; these are their in-process successors):

1. **The depth-3 successor slice** (:mod:`utils.rust_env.successors_parity`): the in-process tree ==
   the ``search_driver`` binary, every arm field and every leaf row BYTE for byte, to depth 3.
   COMMIT tier here (ladder COMMIT teams + a pool battle); MILESTONE (``slow``): the ladder
   milestone tier, the training pool and procedural teams. The binary's rows are themselves held to
   the poke-env replay by ``agents/battle/core_row_parity_fuzz_test.py``, so the in-process rows
   are too (transitively, byte equality being exact).
2. **Clone independence / determinism** (the fork-sharing gate's claim): re-expanding the same arm
   after its siblings reproduces it byte for byte; a stale node id (a previous root) is refused;
   the playout is a function of its inputs and siblings sharing a seed share their dice.
3. **Decision equality** (the materializer gate's claim): the real ``SearchEngine`` decides the same
   seeded decisions identically on ``search_impl="rust"`` (the child) and ``"inproc"`` — every
   per-action score, the chosen action, the fallback and the realized widths — under a scorer that
   is a pure function of the row (a sum hides no changed dimension), with deepening on.

Plus the error / lifecycle contract of the handle.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from utils.paths import src_path
from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.rust_env import successors as S
from utils.rust_env import successors_parity as SP

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the search gates cannot run (install rustup)")
    return cargo


@pytest.fixture(scope="module")
def lib():
    crate = src_path("rust_env")
    r = subprocess.run([_cargo(), "build", "--lib", "--profile", "selfcheck", "--features", "emission-selfcheck",
                        "--manifest-path", str(crate / "Cargo.toml")],
                       env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


@pytest.fixture(scope="module")
def logs(lib):
    core = S.SearchCore(lib=lib)
    try:
        out = SP.make_logs(SP._teams("ladder_commit"), 3, 5, core=core) + SP.make_logs(SP._teams("pool"), 1, 6, core=core)
    finally:
        core.close()
    assert len(out) >= 3, "the corpus produced too few finished battles"
    return out


def _assert_green(cen: SP.Census) -> None:
    print(cen.render())
    assert not cen.diffs, "\n".join(cen.diffs[:20])
    # Non-vacuity: every depth ran, rows were compared, and D10 leaves were among them.
    assert all(cen.by_depth[d] > 0 for d in (1, 2, 3)), dict(cen.by_depth)
    assert cen.rows > 200 and cen.mid > 0, cen.render()


# ------------------------------------------------------------------ gate 1


def test_gate_1_the_depth_3_successor_slice_commit(lib, logs):
    _assert_green(SP.run(logs, seed=5, lib=lib, n_turns=2))


@pytest.mark.slow
@pytest.mark.parametrize("source,n", [("ladder_milestone", 12), ("pool", 6), ("procedural", 6)])
def test_gate_1_the_depth_3_successor_slice_milestone(lib, source, n):
    core = S.SearchCore(lib=lib)
    try:
        lg = SP.make_logs(SP._teams(source), n, 17, core=core)
    finally:
        core.close()
    _assert_green(SP.run(lg, seed=17, lib=lib, n_turns=3))


def test_gate_1_has_teeth(lib, logs):
    """One flipped row byte, one changed token and one changed request are each a difference."""
    rec = S.log_to_record(logs[0], battle_tag="teeth")
    with S.Successors(lib=lib) as a, S.Successors(lib=lib) as b:
        ra = a.open_root(2, record=rec, side="p1")
        b.open_root(2, record=rec, side="p1")  # the twin handle, in lockstep (same node ids)
        from main.search_dividend.alpha import legal_choices_from_request

        ours = [c["token"] for c in legal_choices_from_request(ra.requests["p1"])]
        opp = [c["token"] for c in legal_choices_from_request(ra.requests["p2"])]
        arm = {"node_id": ra.node_id, "p1_action": ours[0], "p2_action": opp[0], "seed": "sodium," + "0" * 31 + "5", "label": 0}
        xa, xb = a.expand_many([arm], side="p1")[0], b.expand_many([arm], side="p1")[0]
        cen = SP.Census()
        SP._cmp_arm(cen, "same", xa, xb)
        assert not cen.diffs and cen.rows == 1
        for mutate in ("row", "tokens", "requests"):
            import dataclasses

            c = dict(xb.core_p1)
            y = xb
            if mutate == "row":
                r = np.array(c["row"])
                r.view(np.uint8)[123] ^= 1
                c["row"] = r
                y = dataclasses.replace(xb, core_p1=c)
            elif mutate == "tokens":
                c["tokens"] = dict(c["tokens"], **{"99": "move x"})
                y = dataclasses.replace(xb, core_p1=c)
            else:
                y = dataclasses.replace(xb, requests={"p1": None, "p2": None})
            cen = SP.Census()
            SP._cmp_arm(cen, mutate, xa, y)
            assert cen.diffs, f"a changed {mutate} was not caught"


# ------------------------------------------------------------------ gate 2


def test_gate_2_clone_independence_and_stale_ids(lib, logs):
    rec = S.log_to_record(logs[1], battle_tag="clone")
    from main.search_dividend.alpha import legal_choices_from_request

    with S.Successors(lib=lib) as ss:
        root = ss.open_root(3, record=rec, side="p2")
        ours = [c["token"] for c in legal_choices_from_request(root.requests["p2"])]
        opp = [c["token"] for c in legal_choices_from_request(root.requests["p1"])]
        sd = "sodium," + "0" * 30 + "77"
        first = ss.expand_many([{"node_id": root.node_id, "p2_action": ours[0], "p1_action": opp[0], "seed": sd, "label": 0}], side="p2")[0]
        sibs = [{"node_id": root.node_id, "p2_action": a, "p1_action": o, "seed": s, "label": i}
                for i, (a, o, s) in enumerate([(a, o, s) for a in ours for o in opp for s in (sd, "original")])]
        ss.expand_many(sibs, side="p2")
        again = ss.expand_many([{"node_id": root.node_id, "p2_action": ours[0], "p1_action": opp[0], "seed": sd, "label": 0}], side="p2")[0]
        assert np.asarray(first.core_p2["row"]).tobytes() == np.asarray(again.core_p2["row"]).tobytes()
        assert (first.outcome, first.requests, first.choices_used, first.core_p2["tokens"]) == \
               (again.outcome, again.requests, again.choices_used, again.core_p2["tokens"])
        assert first.node_id != again.node_id, "every child is a NEW node (ids are monotonic)"
        # A stale id (the previous root) is refused after a new root; the handle stays usable.
        old = root.node_id
        root2 = ss.open_root(3, record=rec, side="p2")
        with pytest.raises(S.SuccessorsError, match="unknown core node"):
            ss.expand_many([{"node_id": old, "p2_action": ours[0], "p1_action": opp[0], "seed": sd, "label": 0}], side="p2")
        ok = ss.expand_many([{"node_id": root2.node_id, "p2_action": ours[0], "p1_action": opp[0], "seed": sd, "label": 0}], side="p2")[0]
        assert np.asarray(ok.core_p2["row"]).tobytes() == np.asarray(first.core_p2["row"]).tobytes()


def _greedy_hash(rows, masks, who):
    """A deterministic 'greedy' policy: the legal action whose index a hash of the row selects."""
    out = np.empty(len(rows), dtype=np.int32)
    for i, (r, m) in enumerate(zip(rows, masks)):
        legal = np.flatnonzero(m)
        out[i] = legal[int(np.frombuffer(r.tobytes(), dtype=np.uint32)[::13].sum()) % len(legal)]
    return out


def test_gate_2_playouts_are_deterministic_with_common_random_numbers(lib, logs):
    log = logs[0]
    at = next(k for k in range(12, len(log["cmds"])) if log["cmds"][k].startswith("CHOOSE p2"))
    seeds = ["sodium," + "0" * 31 + "1", "sodium," + "0" * 31 + "1", "3,1,4,1"]
    with S.SearchCore(lib=lib) as core:
        a = S.play_out(log, at, "p2", policy=_greedy_hash, seeds=seeds, keep_cmds=True, core=core)
        b = S.play_out(log, at, "p2", policy=_greedy_hash, seeds=seeds, keep_cmds=True, core=core)
        stats = core.stats()
    assert a.branches == b.branches, "a playout is a function of its inputs"
    assert len(a.branches) == 3 * len(a.tokens) and a.answered > 0 and a.batches > 0
    for k in range(0, len(a.branches), 3):
        x, y, z = a.branches[k:k + 3]
        assert x["action"] == y["action"] == z["action"]
        assert x["end"] == y["end"] and x["cmds"] == y["cmds"], "one seed = one dice stream (CRN)"
        assert x["cmds"][:at] == log["cmds"][:at], "the branch keeps the banked prefix"
        assert x["end"] is not None
    assert any(a.branches[k]["cmds"] != a.branches[k + 2]["cmds"] for k in range(0, len(a.branches), 3))
    v = a.values()
    assert set(v) == set(a.tokens) and all(-1.0 <= x <= 1.0 for x in v.values())
    assert stats["playout_branches_live"] == 0 and stats["playout_finished"] == 2 * len(a.branches)


# ------------------------------------------------------------------ gate 3


def _decide(impl: str, record, side, turn, tokens, observed, opp_true):
    from main.search_dividend.search import SearchConfig, SearchEngine, WidthCaps

    cfg = SearchConfig(arm="oracle", budget_s=1e9, seed=7, max_depth=2, search_impl=impl,
                       caps=WidthCaps(m_opp=2, k_worlds=1, r_dice=2))
    eng = SearchEngine(model=None, mappings=None, cfg=cfg, pool_packed=[])
    eng._score_batch = lambda obs, masks: (np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")  # type: ignore
    try:
        res = eng.choose(record=record, side=side, turn=turn, our_history=[], our_tokens=dict(tokens),
                         observed_our_lines=list(observed), pub=None, policy_action=next(iter(tokens)),
                         opp_true_packed=opp_true)
        return dict(res.scores or {}), int(res.action), str(res.fallback), res.widths
    finally:
        eng.close()


def test_gate_3_search_decides_identically_in_process(lib, logs):
    from main.search_dividend import determinize as dz
    from main.search_dividend.alpha import legal_choices_from_request
    from utils.bridge.search_session import SearchSession

    compared = deep = 0
    for li, log in enumerate(logs[:3]):
        rec = S.log_to_record(log, battle_tag=f"d{li}")
        last = SP._last_turn(rec)
        for frac in (0.25, 0.6):
            turn = max(2, int(last * frac))
            side = "p1" if (li + int(frac * 4)) % 2 == 0 else "p2"
            other = "p2" if side == "p1" else "p1"
            with SearchSession(rec, impl="rust") as ss:
                try:
                    root = ss.open_root(turn)
                except Exception:  # noqa: BLE001 — past the end
                    continue
            legal = legal_choices_from_request((root.requests or {}).get(side))
            if not legal:
                continue
            tokens = {i: c["token"] for i, c in enumerate(legal)}
            pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
            observed = dz.chunks_to_lines(pfx)
            opp_true = rec.packed_team(other)
            a = _decide("rust", rec, side, turn, tokens, observed, opp_true)
            b = _decide("inproc", rec, side, turn, tokens, observed, opp_true)
            assert a[2] == b[2], f"{li} T{turn}: fallback {a[2]!r} vs {b[2]!r}"
            assert a[0] == b[0], f"{li} T{turn}: per-action scores differ:\n{a[0]}\n{b[0]}"
            assert a[1] == b[1]
            for f in ("arms_expanded", "arms_scored", "arms_terminal", "deep_arms_expanded", "core_arms",
                      "core_arms_intermediate", "depth_realized", "worlds_gated_ok"):
                assert getattr(a[3], f) == getattr(b[3], f), f"{li} T{turn}: widths.{f}"
            compared += 1
            deep += int(b[3].deep_arms_scored)
    assert compared >= 3, f"only {compared} decisions compared — the gate is vacuous"
    assert deep > 0, "no decision deepened — the depth-2 half of the gate is vacuous"


# ------------------------------------------------------------------ the handle's contract


def test_errors_are_typed_and_do_not_poison(lib, logs):
    rec = S.log_to_record(logs[0], battle_tag="err")
    small = S.spec_json(max_nodes=3, max_branches=2)
    with S.Successors(lib=lib, spec=small) as ss:
        with pytest.raises(S.SuccessorsError, match="core"):
            ss.open_root(2, record=rec, core=None)
        with pytest.raises(S.SuccessorsError, match="trackers"):
            ss.open_root(2, record=rec, trackers=False)
        with pytest.raises(S.SuccessorsError, match="invalid turn"):
            ss.open_root(0, record=rec)
        root = ss.open_root(2, record=rec, side="p1")
        with pytest.raises(S.SuccessorsError, match="ROWS"):
            ss.expand_many([], rows=False)
        with pytest.raises(S.SuccessorsError, match="unknown core node"):
            ss.expand_many([{"node_id": "n999", "label": 0}], side="p1")
        arms = [{"node_id": root.node_id, "p1_action": "random", "p2_action": "random", "seed": f"{i},2,3,4", "label": i}
                for i in range(4)]
        with pytest.raises(S.SuccessorsError, match="max_nodes"):
            ss.expand_many(arms, side="p1")
        # still usable
        assert ss.open_root(2, record=rec, side="p1").node_id
        with pytest.raises(S.SuccessorsError, match="max_branches"):
            S.play_out(logs[0], 0, "p1", policy=_greedy_hash, seeds=["1,1,1,1"], core=ss)
    with S.SearchCore(lib=lib) as core:
        with pytest.raises(S.SuccessorsError, match="not legal"):
            S.play_out(logs[0], 0, "p1", policy=_greedy_hash, seeds=["1,1,1,1"], actions=[10], core=core)
        with pytest.raises(S.SuccessorsError, match="shape"):
            S.play_out(logs[0], 0, "p1", policy=lambda r, m, w: np.zeros(0, dtype=np.int32), seeds=["1,1,1,1"], core=core)
        with pytest.raises(S.SuccessorsError, match="max_turns"):
            S.play_out(logs[0], 0, "p1", policy=_greedy_hash, seeds=["1,1,1,1"], core=core, max_turns=1000)
        # still usable after every refusal
        assert S.play_out(logs[0], 0, "p1", policy=_greedy_hash, seeds=["1,1,1,1"], core=core).branches
    with pytest.raises(P.CallerError, match="unknown key"):
        S.SearchCore(lib=lib, spec=json.dumps({"clock": {"decision_tense": False, "switch_freeze": False},
                                                "max_nodes": 1, "max_branches": 1, "x": 1}))
