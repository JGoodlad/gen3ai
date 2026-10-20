"""`engine.disagreements` — `find disagree` batched over a battle, each verdict the single-row path's.

Three pins: (1) on a fake model it returns exactly the decisions where `analyze_invocation` says
``has_state and not agrees`` (the unbatched loop it replaced); (2) a near-tie row is re-read on the
single-row path, so a batched float difference cannot flip a verdict (fails if the tie guard is
removed); (3) on a REAL policy, the batched verdicts equal the single-row `ProbeModel.action_dist`
verdicts and the batched probabilities match the single-row ones within a bound far below the guard."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from main.prober.engine import analyze_invocation, disagreements
from main.prober.engine import analyze as A
from main.prober.engine_test import _OBS_LEN, _OFF, FakeProbeModel


def _actions(n_valid: int = 11) -> dict:
    labels = [f"switch:s{i}" for i in range(6)] + ["m0", "m1", "m2", "m3", "struggle"]
    return {k: {"prob": "0.0%", "valid": j < n_valid} for j, k in enumerate(labels)}


def _battle(chosen, obs, has_state=None):
    n = len(chosen)
    invs = [{"i": i + 1, "turn": i + 1, "phase": "move", "chosen": c, "our": {"species": "Zapdos"},
             "opp": {"species": "Steelix"}, "actions": _actions(), "outcome": {"events": []}}
            for i, c in enumerate(chosen)]
    summary = {"meta": {"step": 1, "battle_id": "b", "result": "WIN", "turns": n, "invocations": n},
               "invocations": invs}
    hs = np.ones(n, dtype=np.int8) if has_state is None else np.asarray(has_state, dtype=np.int8)
    return summary, {"obs": obs, "has_state": hs, "values": np.zeros(n, dtype=np.float32)}


class _BatchedFake(FakeProbeModel):
    """The fake plus a batched read that is the single-row read ± ``jitter`` — the shape of a real
    batch-size-dependent float difference."""

    def __init__(self, offsets, jitter=None):
        super().__init__(offsets)
        self.jitter = jitter or {}
        self.batch_rows = 0
        self.single_rows = 0

    def action_dist(self, obs, mask):
        self.single_rows += 1
        return super().action_dist(obs, mask)

    def action_probs_batch(self, obs, masks):
        self.batch_rows += len(obs)
        out = np.stack([FakeProbeModel.action_dist(self, o, m)[0] for o, m in zip(obs, masks)])
        for (r, col), d in self.jitter.items():
            if r < len(out):
                out[r, col] += d
        return out


def _obs(n, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, _OBS_LEN), dtype=np.float32)
    obs[:, _OFF.mm_off:_OFF.mm_off + 4] = rng.choice([0.0, 0.125, 0.25, 0.5], size=(n, 4))
    return obs


def test_the_batched_verdicts_are_the_analyze_loops():
    n = 9
    obs = _obs(n)
    chosen = ["m0", "m1", "m2", "m3", "switch:s0", "m0", "m1", "m2", "m3"]
    has_state = [1, 1, 0, 1, 1, 1, 1, 1, 0]
    summary, npz = _battle(chosen, obs, has_state)
    loop = [i for i in range(n)
            if (a := analyze_invocation(FakeProbeModel(_OFF), summary, npz, i)).has_state and not a.agrees]
    m = _BatchedFake(_OFF)
    got = disagreements(m, summary, npz)
    assert loop and len(loop) < sum(has_state), "vacuous: need both agreeing and disagreeing decisions"
    assert got == loop
    assert m.batch_rows == sum(has_state)                       # every stateful row went through the batch
    assert disagreements(FakeProbeModel(_OFF), summary, npz) == loop  # a model without a batch read: row by row


def test_a_near_tie_is_decided_on_the_single_row_path():
    # Row 0: m0 and m1 exactly tied on the single-row path (argmax = m0, the first). The batched read is
    # nudged by less than the guard so it would pick m1. Chosen = m0 -> the single-row verdict is AGREE.
    obs = np.zeros((2, _OBS_LEN), dtype=np.float32)
    obs[:, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.5, 0.0, 0.0]
    summary, npz = _battle(["m0", "m0"], obs)
    m1 = 7                                                       # action index of m1
    eps = A.DISAGREE_TIE_EPS / 4
    m = _BatchedFake(_OFF, jitter={(0, m1): eps})
    assert disagreements(m, summary, npz) == [] and m.single_rows == 2   # both rows tie -> both re-read
    # without the guard the batched argmax WOULD flip (the test is not vacuous)
    probs = m.action_probs_batch(obs[:1], np.ones((1, 11), dtype=np.int8))
    assert int(np.argmax(probs[0])) == m1
    # a clear margin is NOT re-read
    obs2 = _obs(4, seed=3)
    obs2[:, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.0, 0.0, 0.0]
    s2, n2 = _battle(["m0"] * 4, obs2)
    m2 = _BatchedFake(_OFF)
    assert disagreements(m2, s2, n2) == [] and m2.single_rows == 0 and m2.batch_rows == 4


@pytest.mark.slow  # ship gate (2026-10-20): 17.7 s beside a live run (the real policy's build dominates)
def test_on_a_real_policy_the_batched_read_is_the_single_row_read(monkeypatch) -> None:
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.static_recovery_test import _policy
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.prober.model import ObsOffsets, ProbeModel

    obs, mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    # a batch of 8 over 20 rows: two full batches and a ragged third, at a third of the single-row reads
    # (the 64-row production batch's error is the measured one in `designs/prober/session_reading.md`)
    monkeypatch.setattr(A, "DISAGREE_BATCH", 8)
    obs, mask = obs[:20], mask[:20]
    n = len(obs)
    assert n > A.DISAGREE_BATCH, n
    obs, mask = np.asarray(obs, dtype=np.float32), np.asarray(mask).astype(np.int8)
    pol = _policy(trunk_layers=3)                                 # cached across tests: restore what is changed
    space, was_training = pol.observation_space, pol.training
    pol.observation_space = gym.spaces.Dict({**space.spaces, "action_mask": gym.spaces.MultiBinary(mask.shape[1])})
    pol.eval()
    try:
        model = ProbeModel(pol, ObsOffsets.resolve())
        single = np.stack([model.action_dist(obs[k], mask[k])[0] for k in range(n)])
        batched = np.concatenate([model.action_probs_batch(obs[s:s + A.DISAGREE_BATCH], mask[s:s + A.DISAGREE_BATCH])
                                  for s in range(0, n, A.DISAGREE_BATCH)])
        assert float(np.abs(batched - single).max()) < A.DISAGREE_TIE_EPS / 10
        labels = list(_actions().keys())
        top = [labels[int(np.argmax(single[k]))] for k in range(n)]
        # choose the argmax on even rows and a different VALID label on odd rows -> a known answer
        chosen = []
        for k in range(n):
            if k % 2 == 0:
                chosen.append(top[k])
            else:
                other = next((labels[j] for j in range(11) if mask[k, j] and labels[j] != top[k]), top[k])
                chosen.append(other)
        summary, npz = _battle(chosen, obs)
        for k, inv in enumerate(summary["invocations"]):               # the trace's mask is the row's own
            inv["actions"] = {lab: {"prob": "0.0%", "valid": bool(mask[k, j])} for j, lab in enumerate(labels)}
        expect = [k for k in range(n) if chosen[k] != top[k]]
        assert expect, "vacuous: no row has a second valid action"
        assert disagreements(model, summary, npz) == expect
    finally:
        pol.observation_space = space
        pol.train(was_training)
