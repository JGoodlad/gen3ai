"""Unit tests for the harvest producer, its schema contract, and the meter's statistics.

Pure and fast — no checkpoint, no bridge, no archive. Everything that needs a trace builds a
synthetic one on disk, because the properties under test (the holdout is disjoint; a timeout is not
a loss; a wrong row index is caught) are properties of the CODE, and a test that can only run on
the owner's box is a test that runs nowhere else.

The consumer's own tests live in ``agents/training/winprob_finetune_test.py``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from agents.training.harvest_schema import (
    HARVEST_KIND, HARVEST_SCHEMA_VERSION, HarvestRow, load_obs, obs_b64, obs_digest,
    read_dir, read_rows, validate_row, write_rows,
)
from main import harvest as H
from main import harvest_meter as M

OBS_DIM = 16


# ---------------------------------------------------------------------------
# Fixtures — a synthetic trace tree
# ---------------------------------------------------------------------------

def _write_battle(root: str, run: str, opp: str, name: str, *, turns: int, result: str,
                  n_inv: int, faints_at_end: int = 0, forcelose: bool = True,
                  win_probs=None, seed: int = 0) -> str:
    """One synthetic trace triple (summary / states.npz / reconstruction) under ``root``."""
    d = os.path.join(root, run, "eval_traces", "step_1000", opp)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
    rng = np.random.default_rng(seed)

    invs = []
    for i in range(n_inv):
        turn = max(1, int(round(1 + (turns - 1) * i / max(1, n_inv - 1))))
        events = []
        if faints_at_end and i >= n_inv - faints_at_end:
            events = ["p2a: Foo:fainted"]
        invs.append({"i": i, "turn": turn, "phase": "move_selection",
                     "outcome": {"events": events}})
    with open(base + "_summary.json", "w") as fh:
        json.dump({"meta": {"result": result.upper(), "turns": turns,
                            "invocations": n_inv, "battle_id": name},
                   "invocations": invs}, fh)

    wp = (np.asarray(win_probs, dtype=float) if win_probs is not None
          else rng.random(n_inv))
    np.savez(base + "_states.npz",
             obs=rng.standard_normal((n_inv, OBS_DIM)).astype(np.float32),
             win_probs=wp,
             values=rng.standard_normal(n_inv),
             has_state=np.ones(n_inv, dtype=int),
             actions=rng.integers(0, 9, n_inv))

    cmds = [["p1", "move 1"], ["p2", "move 1"]] * n_inv
    if forcelose:
        cmds.append(["forcelose", "p1"])
    with open(base + "_reconstruction.json", "w") as fh:
        json.dump({"v": 1, "commands": cmds, "input_log": [], "prng_seed": [1, 2, 3, 4],
                   "trainee_username": "RLTest", "battle_tag": name}, fh)
    return base


@pytest.fixture()
def archive(tmp_path, monkeypatch):
    """A models-root with one current-arch run and a spread of battle shapes."""
    root = str(tmp_path / "models")
    run = "ai_v9_test_run"
    os.makedirs(os.path.join(root, run), exist_ok=True)
    from agents.model.model_version import ARCH_SIGNATURE
    with open(os.path.join(root, run, "model_config.json"), "w") as fh:
        json.dump({"arch_signature": ARCH_SIGNATURE}, fh)

    _write_battle(root, run, "staller", "cap_ok", turns=250, result="loss", n_inv=40, seed=1)
    _write_battle(root, run, "staller", "cap_broken", turns=250, result="loss", n_inv=40,
                  forcelose=False, seed=2)
    for i in range(6):
        _write_battle(root, run, "heuristic", f"longloss_{i}", turns=140, result="loss",
                      n_inv=30, seed=10 + i)
    for i in range(4):
        _write_battle(root, run, "heuristic", f"longwin_{i}", turns=140, result="win",
                      n_inv=30, seed=20 + i)
    _write_battle(root, run, "heuristic", "short", turns=20, result="loss", n_inv=10, seed=30)
    return root


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

def _row(**kw) -> HarvestRow:
    obs = np.arange(OBS_DIM, dtype=np.float32)
    base = dict(run="r", battle_tag="r/b", decision_idx=3, turn=90, n_rollouts=32, n_wins=8,
                phi_head=0.8, beta_evidence=14.0, beta_mean=0.6, priority=0.5,
                provenance={"opponent": "staller"}, obs_npz="r/b_states.npz",
                obs_sha1=obs_digest(obs), obs_inline=obs_b64(obs))
    base.update(kw)
    return HarvestRow(**base)


def test_schema_round_trip_preserves_every_pinned_field(tmp_path):
    rows = [_row(), _row(decision_idx=4, n_wins=0)]
    p = write_rows(rows, str(tmp_path / "labels_0000.jsonl.gz"))
    back = list(read_rows(p))
    assert len(back) == 2
    for r, b in zip(rows, back):
        assert b["schema"] == HARVEST_SCHEMA_VERSION and b["kind"] == HARVEST_KIND
        for f in ("run", "battle_tag", "decision_idx", "turn", "n_rollouts", "n_wins",
                  "phi_head", "beta_evidence", "beta_mean", "priority", "provenance"):
            assert b[f] == getattr(r, f), f


def test_read_dir_reads_every_shard_in_order(tmp_path):
    write_rows([_row(decision_idx=1)], str(tmp_path / "labels_0000.jsonl.gz"))
    write_rows([_row(decision_idx=2)], str(tmp_path / "labels_0001.jsonl.gz"))
    got = read_dir(str(tmp_path))
    assert [r["decision_idx"] for r in got] == [1, 2]


def test_label_is_k_over_n():
    assert _row(n_wins=8, n_rollouts=32).label == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [
    {"n_rollouts": 0}, {"n_wins": 33}, {"n_wins": -1},
])
def test_validate_rejects_impossible_counts(bad):
    d = _row(**bad).to_json()
    with pytest.raises(ValueError):
        validate_row(d)


def test_validate_rejects_a_foreign_schema_version():
    d = _row().to_json()
    d["schema"] = 99
    with pytest.raises(ValueError, match="unknown harvest schema"):
        validate_row(d)


def test_write_leaves_no_tmp_file_behind_on_failure(tmp_path):
    p = str(tmp_path / "labels_0000.jsonl.gz")
    with pytest.raises(ValueError):
        write_rows([_row(n_rollouts=0)], p)
    assert not os.path.exists(p + ".tmp") and not os.path.exists(p)


# ---------------------------------------------------------------------------
# load_obs — the indexing contract cf_audit got wrong
# ---------------------------------------------------------------------------

def test_load_obs_inline_and_npz_agree_and_index_by_decision_idx(tmp_path):
    arr = np.arange(5 * OBS_DIM, dtype=np.float32).reshape(5, OBS_DIM)
    npz = tmp_path / "b_states.npz"
    np.savez(npz, obs=arr)
    row = _row(decision_idx=3, obs_npz="b_states.npz",
               obs_sha1=obs_digest(arr[3]), obs_inline=obs_b64(arr[3])).to_json()

    inline = load_obs(row, models_root=str(tmp_path))
    row_ptr = dict(row, obs_inline=None)
    pointed = load_obs(row_ptr, models_root=str(tmp_path))
    assert np.array_equal(inline, arr[3]) and np.array_equal(pointed, arr[3])


def test_load_obs_raises_when_the_row_index_is_wrong(tmp_path):
    """The exact bug cf_audit shipped: obs_npz rows ignoring decision_idx. The digest is what
    makes it loud instead of a silent wrong label."""
    arr = np.arange(5 * OBS_DIM, dtype=np.float32).reshape(5, OBS_DIM)
    np.savez(tmp_path / "b_states.npz", obs=arr)
    row = _row(decision_idx=1, obs_npz="b_states.npz",
               obs_sha1=obs_digest(arr[3]), obs_inline=None).to_json()
    with pytest.raises(ValueError, match="obs digest mismatch"):
        load_obs(row, models_root=str(tmp_path))


def test_load_obs_refuses_a_relative_path_with_no_models_root():
    row = _row(obs_inline=None).to_json()
    with pytest.raises(ValueError, match="RELATIVE obs_npz"):
        load_obs(row)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------

def test_build_candidates_finds_late_decisions_and_censuses_skips(archive):
    cands, skipped = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    assert cands
    tags = {c.battle_tag for c in cands}
    assert not any("short" in t for t in tags), "a 20-turn battle is not a late-game battle"
    assert skipped["battle_too_short"] >= 1
    assert all(c.turn >= 2 for c in cands)


def test_a_cap_battle_without_its_terminal_forfeit_is_skipped_and_counted(archive):
    """The measured blocker: 40 of 48 current-arch cap records carry no `forcelose`, so the
    offline replay driver refuses them. Skipping must be counted, never silent."""
    cands, skipped = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    tags = {c.battle_tag for c in cands}
    assert any("cap_ok" in t for t in tags)
    assert not any("cap_broken" in t for t in tags)
    assert skipped["cap_record_unterminated"] == 1


def test_cap_battles_are_swept_whole_regardless_of_min_turn(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=200)
    cap = [c for c in cands if "cap_ok" in c.battle_tag]
    assert cap and min(c.turn for c in cap) < 200


def test_meter_class_separates_caps_from_long_losses_and_excludes_wins(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    pools = H.meter_battles(cands)
    assert len(pools["cap"]) == 1
    assert len(pools["long_loss"]) == 6
    assert not any("longwin" in t for v in pools.values() for t in v)


# ---------------------------------------------------------------------------
# Holdout hygiene
# ---------------------------------------------------------------------------

def test_holdout_is_stratified_so_the_scarce_cap_class_reaches_both_arms(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    held = set(H.battle_holdout(cands, 0.35, seed=0))
    pools = H.meter_battles(cands)
    assert set(pools["cap"]) & held, "the cap class must be represented in the holdout"
    assert set(pools["long_loss"]) & held


def test_holdout_is_deterministic_under_a_fixed_seed(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    a = H.battle_holdout(cands, 0.35, seed=7)
    b = H.battle_holdout(cands, 0.35, seed=7)
    c = H.battle_holdout(cands, 0.35, seed=8)
    assert a == b
    assert a != c or len(a) <= 1


def test_no_held_out_battle_can_ever_be_selected(archive):
    """Battle-level holdout hygiene, enforced structurally: exclusion happens before ranking."""
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        c.phi_head = 0.9
    held = H.battle_holdout(cands, 0.5, seed=0)
    chosen = H.select(cands, 10_000, seed=0, exclude_battles=held)
    assert not ({c.battle_tag for c in chosen} & set(held))


def test_cap_states_are_taken_first_inside_the_doomed_stratum(archive):
    """Measured on the real frame: a blended ranking over the pooled doomed tails drew ZERO cap
    states in a 240-state pilot — 8 replayable cap battles cannot outrank 245 long losses on
    priority alone. The cap ending is probe O's headline class, so it gets the front of the queue."""
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        # Make the long losses look STRICTLY more attractive on every priority term.
        c.phi_head = 0.2 if c.is_cap else 0.99
    chosen = H.select(cands, 30, seed=0, drag_frac=1.0, max_per_battle=12)
    assert any(c.is_cap for c in chosen), "the scarce headline class was ranked out of the sample"


def test_the_harvest_samples_the_REGION_THE_METER_READS(archive):
    """The failed pilot's lesson, pinned. A purely priority-ranked 200-state draw reached only
    turn 152 while 29.3% of the meter's eval turns were ABOVE that — the head was fit on mid-game
    states it wins 62% of and never shown a losing tail, so it collapsed to a near-constant ~0.6
    and got WORSE in both directions. A label factory that never samples the region its meter
    scores is extrapolating."""
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        # Make EARLY decisions look maximally attractive on priority, so only the tail stratum
        # can rescue the tail — the exact adversarial shape the real frame had.
        c.phi_head = 0.99 if not c.is_tail else 0.01
    chosen = H.select(cands, 40, seed=0, drag_frac=1.0, tail_frac=0.5, max_per_battle=12)
    tails = [c for c in chosen if c.is_tail]
    assert tails, "no state from the last TAIL_K decisions — the meter's region is unsupervised"
    # And the reserved share is honoured, not merely non-zero.
    assert len(tails) >= 0.4 * len(chosen)


def test_tail_rank_counts_move_decisions_from_the_end(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for tag in {c.battle_tag for c in cands}:
        mine = sorted((c for c in cands if c.battle_tag == tag), key=lambda c: c.decision_idx)
        # The last-kept decision of a battle has the smallest rank among those kept.
        assert mine[-1].tail_rank < mine[0].tail_rank
        assert mine[-1].tail_rank == 0 or not mine[-1].is_tail or mine[-1].tail_rank < H.TAIL_K


def test_tail_frac_zero_reproduces_the_old_priority_only_behaviour(archive):
    """The stratum is a knob, not a hard-coded policy — an ablation is one flag away."""
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        c.phi_head = 0.99 if not c.is_tail else 0.01
    chosen = H.select(cands, 20, seed=0, drag_frac=1.0, tail_frac=0.0, max_per_battle=12)
    assert not any(c.is_tail for c in chosen)


def test_selection_never_exceeds_the_per_battle_cap(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        c.phi_head = 0.9
    chosen = H.select(cands, 10_000, seed=0, max_per_battle=3)
    counts: dict = {}
    for c in chosen:
        counts[c.battle_tag] = counts.get(c.battle_tag, 0) + 1
    assert max(counts.values()) <= 3


def test_selection_is_deterministic_under_a_fixed_seed(archive):
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        c.phi_head = 0.75
    key = lambda ch: [(c.battle_tag, c.decision_idx) for c in ch]           # noqa: E731
    assert key(H.select(cands, 25, seed=3)) == key(H.select(cands, 25, seed=3))


def test_the_general_stratum_is_outcome_balanced(archive):
    """Measured on the real frame: without this, a 300-state draw took ONE state from a won
    battle, because |phi - realized| is ~0 on a correctly-read win."""
    cands, _ = H.build_candidates(archive, ["ai_v9_test_run"], min_turn=60)
    for c in cands:
        c.phi_head = 0.95            # confident everywhere ⇒ wins score ~0 on the gap term
    chosen = H.select(cands, 40, seed=0, drag_frac=0.5, general_win_frac=0.5)
    general = [c for c in chosen if not c.is_drag]
    assert general, "the general stratum must not be empty"
    assert any(c.outcome == "win" for c in general), \
        "the control stratum was selected out of existence by the gap term"


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def _cand(**kw) -> H.Candidate:
    base = dict(run="r", battle_tag="r/b", abs_prefix="/tmp/b", decision_idx=0, turn=90,
                action=1, opponent="staller", outcome="loss", battle_turns=140,
                faints_tail=0, is_cap=False, recorded_phi=0.5)
    base.update(kw)
    return H.Candidate(**base)


def test_priority_rewards_a_confident_head_on_a_lost_game():
    confident_wrong = _cand(phi_head=0.99)
    correct = _cand(phi_head=0.02)
    assert H.priority_of(confident_wrong) > H.priority_of(correct)


def test_priority_rewards_confessed_uncertainty():
    """A wide Beta (low precision) is the head nominating its own weak spot."""
    wide = _cand(phi_head=0.5, beta_alpha=0.6, beta_beta=0.6)
    narrow = _cand(phi_head=0.5, beta_alpha=60.0, beta_beta=60.0)
    assert H.priority_of(wide) > H.priority_of(narrow)


def test_priority_falls_back_cleanly_with_no_evidential_head():
    """Absent is not zero: a subject with no Beta head must not score uniformly lower than one
    that has it, or the two harvests' priorities stop being comparable."""
    c = _cand(phi_head=0.99)                      # a long loss ⇒ drag term is 1.0
    assert c.is_drag
    assert H.priority_of(c) == pytest.approx(
        (H.W_GAP * 0.99 + H.W_DRAG * 1.0) / (H.W_GAP + H.W_DRAG))
    # ...and the same candidate WITH a head scores on the full three-term scale, so neither
    # subject is systematically ranked below the other.
    with_head = _cand(phi_head=0.99, beta_alpha=1.0, beta_beta=1.0)
    assert 0.0 <= H.priority_of(with_head) <= 1.0


def test_priority_modes_isolate_their_own_term():
    c = _cand(phi_head=0.9, beta_alpha=1.0, beta_beta=1.0, is_cap=True)
    assert H.priority_of(c, "gap") == pytest.approx(0.9)
    assert H.priority_of(c, "drag") == 1.0
    assert 0.0 <= H.priority_of(c, "evidence") <= 1.0


def test_priority_stays_in_the_unit_interval():
    for phi in (0.0, 0.5, 1.0):
        for a, b in ((0.5, 0.5), (5.0, 5.0), (50.0, 1.0)):
            p = H.priority_of(_cand(phi_head=phi, beta_alpha=a, beta_beta=b, is_cap=True))
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Label math — CRN pairing at the seam, and the timeout bucket
# ---------------------------------------------------------------------------

class _FakeSession:
    """Stands in for ``ProbeSession`` at the exact seam :func:`H.label_one` uses."""

    def __init__(self, outcomes: dict, n_rollouts: int):
        self.outcomes, self.n_rollouts, self.calls = outcomes, n_rollouts, []

    def replay_counterfactual(self, summary, inv, action, *, n_rollouts=1, **kw):
        self.calls.append((summary, inv, action, n_rollouts))
        return {"outcomes": dict(self.outcomes), "n_rollouts": self.n_rollouts,
                "opponent_source": "bot:staller"}


def test_label_one_counts_wins_over_ADJUDICATED_rollouts():
    sess = _FakeSession({"win": 6, "loss": 26}, 32)
    r = H.label_one(sess, "/tmp/b", 12, 3, 32)
    assert (r["k"], r["n"], r["n_timeout"]) == (6, 32, 0)


def test_a_timed_out_rollout_is_its_own_bucket_and_never_a_loss():
    """The convention the contention doc states as a rule: a timeout is not a semantic outcome.
    Folding one into the denominator would make a busy box read as a losing position."""
    sess = _FakeSession({"win": 5, "loss": 15}, 32)          # 12 neither
    r = H.label_one(sess, "/tmp/b", 12, 3, 32)
    assert r["k"] == 5 and r["n"] == 20 and r["n_timeout"] == 12
    assert r["k"] / r["n"] == pytest.approx(0.25)            # not 5/32


def test_label_one_passes_the_recorded_action_at_the_recorded_decision():
    """CRN pairing at the seam: the label is 'this state, the action actually taken, re-diced'.
    Passing a different inv or action would silently label a different state."""
    sess = _FakeSession({"win": 1, "loss": 1}, 2)
    H.label_one(sess, "/tmp/pfx", 77, 9, 2)
    assert sess.calls == [("/tmp/pfx_summary.json", 77, 9, 2)]


def test_build_row_rejects_a_state_where_nothing_adjudicated(tmp_path):
    c = _cand(abs_prefix=str(tmp_path / "b"))
    assert H.build_row(c, {"ok": True, "n": 0, "k": 0}, subject_ckpt="s",
                       sampler_version="v", seed=0, inline_obs=False,
                       models_root=str(tmp_path)) is None
    assert H.build_row(c, {"ok": False, "n": 8, "k": 4}, subject_ckpt="s",
                       sampler_version="v", seed=0, inline_obs=False,
                       models_root=str(tmp_path)) is None


def test_build_row_carries_the_timeout_count_into_provenance(tmp_path):
    np.savez(tmp_path / "b_states.npz",
             obs=np.zeros((3, OBS_DIM), dtype=np.float32))
    c = _cand(abs_prefix=str(tmp_path / "b"), decision_idx=1, phi_head=0.8,
              beta_alpha=4.0, beta_beta=6.0)
    c.priority = 0.42
    row = H.build_row(c, {"ok": True, "n": 20, "k": 5, "n_timeout": 12, "outcomes": {}},
                      subject_ckpt="s", sampler_version="v1", seed=3, inline_obs=True,
                      models_root=str(tmp_path))
    assert row is not None
    assert row.provenance["n_timeout"] == 12
    assert row.n_rollouts == 20 and row.n_wins == 5
    assert row.beta_evidence == pytest.approx(10.0)
    assert row.beta_mean == pytest.approx(0.4)
    validate_row(row.to_json())


def test_the_label_noise_floor_matches_the_binomial_identity():
    """A k/n label's own sampling sd is sqrt(p(1-p)/n) — what makes R=32 worth 32x one bit."""
    p, n = 0.25, 32
    assert np.sqrt(p * (1 - p) / n) == pytest.approx(0.0765, abs=1e-4)
    assert np.sqrt(0.5 * 0.5 / 1) == 0.5


# ---------------------------------------------------------------------------
# The meter's statistics
# ---------------------------------------------------------------------------

def test_metrics_reproduce_probe_o_definitions():
    m = M.metrics_from_phi([0.99, 0.99, 0.99, 0.99, 0.99])
    assert m["miss"] == 1.0 and m["miss_098"] == 1.0
    assert m["detect_le05"] == 0.0 and m["overconf"] == 1.0
    assert m["c3band"] == 0.0                       # 0.99 is above C3's upper edge

    good = M.metrics_from_phi([0.6, 0.5, 0.4, 0.3, 0.2])
    assert good["detect_le05"] == 1.0 and good["miss"] == 0.0


def test_the_registered_detect_composite_saturates_as_probe_o_found():
    """Probe O's criterion defect, pinned: the 'declining' half fires on almost any lost tail,
    so `detect` reads 1.0 where `detect_le05` reads 0.0. Both are reported, never just one."""
    m = M.metrics_from_phi([0.99, 0.98, 0.97, 0.96, 0.95])
    assert m["detect"] == 1.0 and m["detect_le05"] == 0.0


def test_paired_diff_ci_is_signed_and_flags_significance():
    pre = [0.0] * 20
    post = [1.0] * 20
    r = M.paired_diff_ci(pre, post)
    assert r["diff"] == pytest.approx(1.0) and r["significant"]

    null = M.paired_diff_ci([0.5] * 20, [0.5] * 20)
    assert null["diff"] == 0.0 and not null["significant"]


def test_paired_diff_ci_handles_an_empty_population():
    assert M.paired_diff_ci([], [])["n"] == 0


def test_levels_reports_a_pre_only_baseline_with_a_battle_bootstrap_ci():
    """A PRE-only run is a baseline read of the battery, not a broken pre/post — it must still
    print numbers."""
    rows = [{"battle_tag": f"b{i}", "miss": float(i % 2)} for i in range(20)]
    out = M.levels(rows, ["miss"])
    assert out["miss"]["n"] == 20 and out["miss"]["mean"] == pytest.approx(0.5)
    lo, hi = out["miss"]["ci"]
    assert lo < 0.5 < hi


def test_compare_pairs_on_battle_tag_and_ignores_unmatched(archive):
    pre = [{"battle_tag": "a", "miss": 1.0}, {"battle_tag": "b", "miss": 1.0}]
    post = [{"battle_tag": "a", "miss": 0.0}]
    out = M.compare(pre, post, ["miss"])
    assert out["miss"]["n"] == 1 and out["miss"]["diff"] == pytest.approx(-1.0)


def test_verdict_names_the_degenerate_solution_when_the_control_breaks():
    """The failure mode that looks like a success: 'late means lost' scores perfectly on every
    stall metric while being strictly worse than the head it replaced."""
    report = {"populations": {
        "cap": {"n_battles": 20, "compare": {
            "detect_le05": {"pre": 0.4, "post": 0.95, "diff": 0.55, "ci": [0.3, 0.8],
                            "significant": True, "n": 20},
            "miss": {"pre": 0.6, "post": 0.05, "diff": -0.55, "ci": [-0.8, -0.3],
                     "significant": True, "n": 20}}},
        "control": {"compare": {
            "phi_T": {"pre": 0.98, "post": 0.20, "diff": -0.78, "ci": [-0.9, -0.6],
                      "significant": True, "n": 30}}},
    }}
    lines = " ".join(M.verdict_lines(report))
    assert "CONTROL BROKEN" in lines and "FAILED RUN" in lines


def test_verdict_calls_a_flat_result_uninformative_not_negative():
    report = {"populations": {
        "cap": {"n_battles": 6,
                "compare": {"detect_le05": {"pre": 0.5, "post": 0.5, "diff": 0.0,
                                            "ci": [-0.2, 0.2], "significant": False, "n": 6}}},
        "control": {"compare": {}},
    }}
    lines = " ".join(M.verdict_lines(report))
    assert "uninformative rather than negative" in lines
