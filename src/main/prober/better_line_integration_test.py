"""End-to-end better-line SEARCH: record a REAL bridge battle, then search for a better line from an
anchored move decision with a deterministic FAKE model, exercising the full
SearchSession-clone → materialize → V pipeline (depth 1 AND depth 2, the interior opponent).

The decisive gate is MODEL-FREE: the fake model's V = obs.sum(), so the depth-1 CHOSEN action's value
MUST equal the sum of the recorded ``states.npz`` next obs (its ``recorded_exact`` clone reproduces the
real next state bit-for-bit — the value_crn anchor). Depth 2 additionally checks the beam produces a
principal variation of the right length through the (self-model) interior opponent.

Every test runs against BOTH offline search drivers (``impl="node"`` — ``search_driver.js`` — and
``impl="rust"`` — ``src/rust_sim/src/bin/search_driver.rs``), and one test pins them against EACH
OTHER on the same record: the whole point of a byte-compatible port is that the search answers the
same question, so a value that moves when only the engine changes is the failure mode worth
catching. That cross-impl test is the durable regression gate for ``gen3_rust_search_driver_v1``.

Needs a bridge; no server. Run directly or via ``pytest -m integration``."""

import tempfile

import numpy as np
import pytest

from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle
from main.prober.better_line import better_line_decision
from main.prober.engine import _has_state

# gen3 test tiers (MEASURED 2026-08-14): 38.8 s / 8 tests
pytestmark = pytest.mark.sim


class _SumModel:
    """V(s) = obs.sum() (a deterministic function of the obs, so a faithful materialization is
    provable); a deterministic non-uniform policy so interior top-k differs per state."""

    def value(self, obs, mask):
        return float(np.asarray(obs, dtype=np.float64).sum())

    def values_batch(self, obs, masks):
        return np.asarray(obs, dtype=np.float64).reshape(len(obs), -1).sum(1)

    def _probs(self, obs, mask):
        x = np.asarray(obs, dtype=np.float64)
        m = np.asarray(mask, dtype=float)
        sc = np.array([x[(i * 97) % len(x)] for i in range(len(m))])
        sc = np.where(m > 0, sc, -1e9)
        p = np.exp(sc - sc.max())
        return p / p.sum()

    def action_dist(self, obs, mask):
        p = self._probs(obs, mask)
        return p, np.log(p + 1e-12)

    def action_probs_batch(self, obs, masks):
        return np.stack([self._probs(o, m) for o, m in zip(np.asarray(obs), np.asarray(masks))])

    def win_prob_at(self, obs, mask):
        return None

    def value_dist_at(self, obs, mask):
        return None


def _record_one_battle(out_dir: str, record_impl: str = "node"):
    """The SAME real battle every run — see `record_fixture_battle`'s wall-clock-seed note.

    This fixture is the antipattern's FIRST named instance: it seeded from the clock, so it
    played a different battle every run and an unlucky one outran the recorded command list
    the depth-2 beam replays (ledger 2026-08-22). A fixed battle makes that either always
    true or always false, which is the only state a failure can be diagnosed from."""
    return record_fixture_battle(out_dir, key=2, tag="BL", impl=record_impl)


def _mid_anchor(summary, npz):
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and i + 1 < len(invs)
            and _has_state(npz, i) and _has_state(npz, i + 1)]
    return cand[len(cand) // 2] if cand else None


@pytest.mark.integration
@pytest.mark.parametrize("impl", ["node", "rust"])
def test_better_line_depth1_chosen_value_matches_recorded_next_obs(impl):
    with tempfile.TemporaryDirectory(prefix="better_line_d1_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        out = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=1, impl=impl)

        assert out["inv"] == anchor and out["depth"] == 1
        assert out["opp_model_used"] == "recorded@divergence"
        # The declared interior-ply regime rides on the live payload too (task #29).
        assert out["interior_opponent_regime"] == "greedy"
        chosen = next(c for c in out["candidates"] if c["is_chosen"])
        # ASSERTED, not branched on: `_mid_anchor` only returns a decision that HAS a recorded
        # successor, so the chosen move provably did not end the battle and a None value is a
        # clone defect. Branching on it let an unlucky fixture skip this file's decisive gate.
        assert chosen["value"] is not None, (
            "the anchor has a recorded successor, so the chosen line cannot have ended the "
            "battle — a None value is a clone/materialize defect, not an unlucky battle")
        expected = float(np.asarray(npz["obs"][anchor + 1], dtype=np.float64).sum())
        assert abs(chosen["value"] - expected) < 1e-2, (
            "depth-1 chosen value != sum(recorded next obs) — the value_crn anchor broke "
            "(clone recorded_exact ≠ recorded next state)")
        # Every candidate maps to a choice and is scored or terminal; ΔV is relative to the chosen line.
        for c in out["candidates"]:
            assert c["choice"]
            assert (c["value"] is not None) or (c["terminal"] is not None)


@pytest.mark.integration
@pytest.mark.parametrize("impl", ["node", "rust"])
def test_better_line_depth2_beam_produces_principal_variation(impl):
    with tempfile.TemporaryDirectory(prefix="better_line_d2_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        model = _SumModel()
        out = better_line_decision(model, record, summary, npz, anchor,
                                   depth=2, beam=3, top_k=4, opp_model=model, impl=impl)
        assert out["depth"] == 2 and out["opp_model_used"] == "reloaded"
        best = out["best_alternative"]
        # The battle is FIXED, so "did the beam produce a scorable alternative at all?" is a
        # property of the code, not of the draw — assert it rather than skipping past it.
        assert best is not None, "the depth-2 beam returned no alternative to the chosen line"
        assert best["terminal"] is None, (
            "the mid-battle anchor's best alternative ended the battle — the fixture battle "
            "changed shape; re-check the anchor rather than weakening the gate")
        pv = best["principal_variation"]
        assert 1 <= len(pv) <= 2          # a depth-2 line: the divergence ply + (≤1) interior ply
        assert pv[0]["action"] == best["action"]


@pytest.mark.integration
@pytest.mark.parametrize("impl", ["node", "rust"])
def test_better_line_is_deterministic(impl):
    with tempfile.TemporaryDirectory(prefix="better_line_det_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        a = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2,
                                 opp_model=_SumModel(), impl=impl)
        b = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2,
                                 opp_model=_SumModel(), impl=impl)
        assert [c["action"] for c in a["candidates"]] == [c["action"] for c in b["candidates"]]
        assert [c["value"] for c in a["candidates"]] == [c["value"] for c in b["candidates"]]


@pytest.mark.integration
@pytest.mark.parametrize("record_impl", ["node", "rust"])
def test_better_line_node_and_rust_drivers_agree(record_impl):
    """THE cross-impl gate for ``gen3_rust_search_driver_v1``.

    The same record, the same anchor, the same deterministic model — searched once through
    ``search_driver.js`` and once through the Rust binary. Every candidate's action, choice,
    terminal label and V must match, and V is ``obs.sum()``, so an exact match here means the
    two drivers materialized BIT-IDENTICAL successor observations at every ply of the beam.
    That is a strictly stronger statement than "both ran without error", and it is the property
    that lets a rust-driven ``better-line`` be compared against a node-driven one.

    Parametrized over the LIVE bridge that produced the record too: a mixed run (train on rust,
    forensics on node) is the realistic deployment, so a record from either engine must search
    identically on both.
    """
    with tempfile.TemporaryDirectory(prefix="better_line_xi_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir, record_impl=record_impl)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        kw = dict(depth=2, beam=3, top_k=4)
        n = better_line_decision(_SumModel(), record, summary, npz, anchor,
                                 opp_model=_SumModel(), impl="node", **kw)
        r = better_line_decision(_SumModel(), record, summary, npz, anchor,
                                 opp_model=_SumModel(), impl="rust", **kw)

        assert n["opp_model_used"] == r["opp_model_used"]
        assert len(n["candidates"]) == len(r["candidates"]), "candidate COUNT differs"

        # Compare BY ACTION, not by position. The claim this test makes is about CONTENT — same
        # actions, same choices, same terminal labels, same V — and `candidates` is not a ranked
        # list (the caller reads it as a per-action mapping; `best_alternative` is computed
        # separately). Position additionally encodes the order each driver happened to ENUMERATE
        # the candidates it did not expand, which is not a property either driver promises.
        #
        # ⚠️ THIS WAS A ~35% FLAKE and the positional zip was the whole cause. Measured over 20
        # random battles: 19 fully agreed, 1 returned the same actions with IDENTICAL values in a
        # different order, 0 diverged on any value. The order differs only PAST the beam — the
        # first `beam` entries are the expanded ones and always matched:
        #     node [1, 5, 7 | 9, 8, 0, 6, 2]
        #     rust [1, 5, 7 | 0, 6, 2, 8, 9]     (action 9 == 25043.0162 in BOTH)
        # Keying by action keeps every assertion below at full strength — it is strictly STRONGER
        # than the zip, since it also proves the two drivers returned the same action SET rather
        # than merely the same number of rows.
        nmap = {c["action"]: c for c in n["candidates"]}
        rmap = {c["action"]: c for c in r["candidates"]}
        assert set(nmap) == set(rmap), (
            f"candidate ACTION SETS differ: node-only {sorted(set(nmap) - set(rmap))}, "
            f"rust-only {sorted(set(rmap) - set(nmap))}")

        for action in sorted(nmap):
            cn, cr = nmap[action], rmap[action]
            assert cn["choice"] == cr["choice"], (
                f"action {action}: choice differs — node {cn['choice']!r} vs rust {cr['choice']!r}")
            assert cn["terminal"] == cr["terminal"], (
                f"action {action}: terminal node {cn['terminal']} vs rust {cr['terminal']}")
            if cn["value"] is None or cr["value"] is None:
                assert cn["value"] == cr["value"], (
                    f"action {action}: one driver scored it and the other did not")
                continue
            # V = obs.sum() over a 2667-dim float32 vector, so equality here is an obs-level
            # bit-identity claim, not a tolerance. The 1e-6 only absorbs float64 summation order.
            assert abs(cn["value"] - cr["value"]) < 1e-6, (
                f"action {action}: V differs — node {cn['value']!r} vs rust {cr['value']!r} "
                "(the drivers materialized DIFFERENT successor observations)")


if __name__ == "__main__":
    for fn, args in (
        (test_better_line_depth1_chosen_value_matches_recorded_next_obs, ("node",)),
        (test_better_line_depth1_chosen_value_matches_recorded_next_obs, ("rust",)),
        (test_better_line_depth2_beam_produces_principal_variation, ("node",)),
        (test_better_line_depth2_beam_produces_principal_variation, ("rust",)),
        (test_better_line_is_deterministic, ("node",)),
        (test_better_line_is_deterministic, ("rust",)),
        (test_better_line_node_and_rust_drivers_agree, ("node",)),
        (test_better_line_node_and_rust_drivers_agree, ("rust",)),
    ):
        fn(*args)
        print(f"{fn.__name__}[{args[0]}]: PASSED")
