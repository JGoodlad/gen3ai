"""X59-S's label sources and students (``teacher.py``, ``student.py``): matched noise, restricted sampling, the set
loss, the arms' targets and their VOID guards, and one tiny train → export → reload on synthetic rows.

TIERS (2026-10-26; the end-to-end test measured 14.4 s niced, over the routine tier's 8 s bar: seven production-policy
builds and ten extractor forwards dominate it): the full train → export → reload → second arm → refusal chain is
`slow`; the routine tier keeps three cheap, separately failing checks of what it proves (one chunk trains and
checkpoints, the SAME seed is the SAME initial network, a state trained under another arm is refused)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.human_data import rowset as RS
from main.human_data import student as ST
from main.human_data import teacher as TE


def test_matched_uniform_is_a_pure_function_of_the_rowset_and_row():
    rid = np.array([3, 1, 2], np.int64)
    a = TE.matched_uniform("abc", rid)
    assert np.array_equal(a, TE.matched_uniform("abc", rid))
    assert np.array_equal(a[[1, 2, 0]], TE.matched_uniform("abc", rid[[1, 2, 0]]))   # order-free
    assert not np.array_equal(a, TE.matched_uniform("abd", rid))
    assert ((a >= 0) & (a < 1)).all()


def test_restricted_sampling_never_leaves_the_allowed_set():
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(2000, 11)).astype(np.float32)
    allowed = (rng.random((2000, 11)) < 0.4).astype(np.int8)
    allowed[np.arange(2000), rng.integers(0, 11, 2000)] = 1
    p = TE.masked_probs(lg, allowed)
    assert np.allclose(p.sum(1), 1.0) and (p[allowed == 0] == 0).all()
    u = np.concatenate([rng.random(1998), [0.0, 1.0 - 1e-17]])
    a = TE.inverse_cdf(p, u)
    assert (allowed[np.arange(2000), a] == 1).all()
    # the empirical frequencies follow the renormalised policy
    one = np.tile(p[:1], (20000, 1))
    f = np.bincount(TE.inverse_cdf(one, rng.random(20000)), minlength=11) / 20000
    assert np.abs(f - p[0]).max() < 0.02


def _z(n=6):
    mask = np.zeros((n, 11), np.int8)
    mask[:, [0, 1, 6, 7, 8]] = 1
    cons = mask.copy()
    cons[:, 1] = 0
    ms = np.zeros((n, 11), np.int8)
    ms[:, [6, 7, 8]] = 1
    act = np.array([6, 0, -1, 7, 8, -1], np.int8)[:n]
    hidden = act < 0
    return {"mask": mask, "cons": cons, "moveset": ms, "act": act, "hidden": hidden}


def test_targets_per_arm_and_the_void_guards():
    z = _z()
    t = ST.targets("human", z, None)
    assert t[0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] and (t[2] == z["moveset"][2]).all()
    lab = np.array([7, 0, -1, 6, 6, -1], np.int8)
    tp = ST.targets("pstar", z, lab)
    assert tp[0, 7] == 1 and (tp[2] == t[2]).all() and (tp[5] == t[5]).all()   # hidden rows: the SAME set label
    bad = lab.copy()
    bad[1] = 1                                         # outside the consistent set: the log rules it out
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.targets("pstar", z, bad)
    bad = lab.copy()
    bad[2] = 6                                         # a P* label on a hidden row: the arms would differ there
    with pytest.raises(ST.StudentError, match="hidden"):
        ST.targets("pstar", z, bad)
    with pytest.raises(ST.StudentError):
        ST.targets("pstar", z, None)


def test_the_set_loss_is_the_cross_entropy_on_a_single_label():
    import torch as th

    lg = th.randn(5, 11)
    legal = th.ones(5, 11)
    legal[:, 10] = 0
    a = th.tensor([0, 3, 6, 7, 9])
    t = th.nn.functional.one_hot(a, 11).float()
    ce = -th.log_softmax(lg.masked_fill(legal == 0, float("-inf")), -1).gather(1, a.view(-1, 1)).squeeze(1)
    assert th.allclose(ST.set_nll_t(lg, legal, t), ce, atol=1e-6)
    s = th.zeros(5, 11)
    s[:, :10] = 1                                      # the whole legal set: zero loss
    assert th.allclose(ST.set_nll_t(lg, legal, s), th.zeros(5), atol=1e-6)


def test_the_plan_is_keyed_by_the_seed_alone():
    a = ST.plan([16384] * 12 + [700], 5901, 3)
    assert a == ST.plan([16384] * 12 + [700], 5901, 3) and a != ST.plan([16384] * 12 + [700], 5902, 3)
    assert sum(nb for *_, nb in a) == 3 * sum(-(-n // ST.BATCH) for n in ([16384 * 8, 16384 * 4 + 700]))


def test_lr_warms_up_then_decays():
    assert ST.lr_at(0, 10000) < ST.lr_at(ST.WARMUP - 1, 10000) == pytest.approx(ST.LR_MAX)
    assert ST.lr_at(10000, 10000) == pytest.approx(ST.LR_MIN)


def _tiny_rowset(d: Path, n: int, seed: int, obs: "np.ndarray | None" = None) -> None:
    """A rowset of ``n`` train rows and 64 validation rows; the observations are zeros, or drawn from ``obs``."""
    from agents.battle.core_obs import obs_dim

    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    z = _z(6)

    def arr(k):
        idx = rng.integers(0, 6, k)
        x = np.zeros((k, obs_dim()), np.float32) if obs is None else obs[rng.integers(0, len(obs), k)]
        return {"rows": x, "mask": z["mask"][idx],
                "act": z["act"][idx], "hidden": z["hidden"][idx], "cons": z["cons"][idx], "moveset": z["moveset"][idx],
                "rid": rng.integers(0, 2 ** 62, k), "bkey": np.zeros(k, np.int64), "acct": np.zeros(k, np.int64),
                "seat": np.zeros(k, np.int8), "k": np.zeros(k, np.int16), "turn": np.zeros(k, np.int16),
                "forced": np.zeros(k, bool), "rating": np.zeros(k, np.int16), "quit_dist": np.zeros(k, np.int16),
                "end": np.zeros(k, np.int8)}
    tr = RS._write_shard(d, "train_0000", arr(n))
    va = RS._write_shard(d, "val", arr(64))
    (d / "rows_manifest.json").write_text(json.dumps({"rowset": "t.r1", "train_shards": [tr], "val": va,
                                                      "train_rows": n, "val_rows": 64}))


def _rows_dir(tmp_path, n=300):
    rows = tmp_path / "rows"
    _tiny_rowset(rows, n, 1)
    return rows


def _arm_dir(tmp_path, name, rows):
    out = tmp_path / name
    out.mkdir()
    (out / "rows_path.txt").write_text(str(rows) + "\n")
    return out


def test_a_tiny_train_finishes_and_saves_the_state_a_resume_is_checked_against(tmp_path, restore_torch_globals):
    """ROUTINE half of the end-to-end test below: a row set under one batch is one step, so the chunk finishes (0), the saved state
    is `done` at step 1 with the arm / seed / init identity a resume compares, and a finite validation NLL."""
    import torch as th

    rows = _rows_dir(tmp_path, n=32)               # one step of 32 rows: the forward / backward is the cost
    out = _arm_dir(tmp_path, "s_h", rows)
    assert ST.train("human", rows, None, 5901, out, epochs=1, budget_min=0, threads=1) == 0
    st = th.load(out / "state.pt", weights_only=False)
    assert st["done"] and st["step"] == 1 and st["arm"] == "human" and st["seed"] == 5901
    assert len(st["init_sha256"]) == 64 and st["hist"] and np.isfinite(st["hist"][-1])


def test_one_seed_is_one_initial_network_in_every_arm(restore_torch_globals):
    """ROUTINE half: the init digest `train` records is a function of the seed alone (the end-to-end test compares
    the human and selfplay arms' recorded digests)."""
    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals

    with inference_globals(1, "cpu"):
        a = ST._state_digest(build_fresh_model(5901)[0].policy.state_dict())
        b = ST._state_digest(build_fresh_model(5901)[0].policy.state_dict())
        c = ST._state_digest(build_fresh_model(5902)[0].policy.state_dict())
    assert a == b and a != c


def test_a_state_trained_under_another_arm_is_refused(tmp_path, restore_torch_globals):
    """ROUTINE half: the resume identity check refuses a `state.pt` whose arm differs, without training anything."""
    import torch as th

    rows = _rows_dir(tmp_path)
    other = _arm_dir(tmp_path, "s_s", rows)
    th.save({"arm": "selfplay", "seed": 5901, "epochs": 1}, other / "state.pt")
    with pytest.raises(ST.StudentError, match="another arm"):
        ST.train("human", rows, None, 5901, other, epochs=1, budget_min=0, threads=1)


@pytest.mark.slow
def test_train_export_reload_on_synthetic_rows(tmp_path, restore_torch_globals):
    rows = tmp_path / "rows"
    _tiny_rowset(rows, 300, 1)
    out = tmp_path / "s_h"
    out.mkdir()
    (out / "rows_path.txt").write_text(str(rows) + "\n")
    assert ST.train("human", rows, None, 5901, out, epochs=1, budget_min=0, threads=1) == 0
    side = ST.export(out, {"own": (rows, "human")})
    assert side["steps"] == 1 and side["anchors_bare_load_bit_equal"] and np.isfinite(side["yardsticks"]["own"])
    other = tmp_path / "s_s"
    other.mkdir()
    (other / "rows_path.txt").write_text(str(rows) + "\n")
    assert ST.train("selfplay", rows, None, 5901, other, epochs=1, budget_min=0, threads=1) == 0
    import torch as th

    a = th.load(out / "state.pt", weights_only=False)
    b = th.load(other / "state.pt", weights_only=False)
    assert a["init_sha256"] == b["init_sha256"]        # one seed, one initial network, in every arm
    with pytest.raises(ST.StudentError, match="another arm"):
        ST.train("human", rows, None, 5901, other, epochs=1, budget_min=0, threads=1)


@pytest.mark.slow  # ship gate (2026-10-26): 23 s beside a live run
@pytest.mark.sim
@pytest.mark.integration
def test_the_anchors_bare_load_reads_the_students_function_on_real_rows(tmp_path, restore_torch_globals):
    """X59-S's CUDA pre-flight check 5 FAILED (2026-10-26) on real rows while the zero rows above passed: the anchors
    harness's bare load (``load_checkpoint_strict``) left the policy in TRAINING mode, where ``nn.MultiheadAttention``
    skips its eval-mode fused fast path, a different kernel (~5e-7 apart) from the student's eval-mode logits. Every
    reader now reads in eval mode; on these rows the two modes differ (asserted: the test's teeth), so ``export``'s
    bit-equal bare-load check fails if the loader's eval is reverted."""
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals, policy_logits
    from utils.rust_env.fixture_battles import spread_rows

    obs, _ = spread_rows(48, n_battles=2, seed=5901)
    rows = tmp_path / "rows"
    _tiny_rowset(rows, 300, 1, obs=obs)
    out = tmp_path / "s_h"
    out.mkdir()
    (out / "rows_path.txt").write_text(str(rows) + "\n")
    assert ST.train("human", rows, None, 5901, out, epochs=1, budget_min=0, threads=1) == 0
    st = th.load(out / "state.pt", weights_only=False, map_location="cpu")
    # One step at the warm-up's lr leaves the zero-initialised paths at zero, where the two modes agree (measured on
    # the fresh init); a seeded perturbation stands in for a trained student's weights.
    g = th.Generator().manual_seed(5901)
    st["model"] = {k: v + 0.02 * th.randn(v.shape, generator=g) if th.is_floating_point(v) else v
                   for k, v in st["model"].items()}
    th.save(st, out / "state.pt")
    man, _ = RS.load_manifest(rows)
    probe = RS.load_shard(rows, man["val"], ("rows", "mask"))
    with inference_globals(ST.THREADS):
        model = build_fresh_model(5901)[0]
        model.policy.load_state_dict(st["model"])
        model.policy.eval()
        a = policy_logits(model, probe["rows"][:256], probe["mask"][:256].astype(np.uint8))
        model.policy.train()
        b = policy_logits(model, probe["rows"][:256], probe["mask"][:256].astype(np.uint8))
    assert not np.array_equal(a.view(np.uint32), b.view(np.uint32)), \
        "precondition: on these rows the training-mode forward must differ from the eval-mode one (the test's teeth)"
    side = ST.export(out)
    assert side["anchors_bare_load_bit_equal"] and side["reload_logits_bit_equal"]


def test_the_unrestricted_arm_needs_legal_labels_only():
    z = _z()
    lab = np.array([1, 0, -1, 6, 6, -1], np.int8)            # 1 is legal but outside the consistent set
    t = ST.targets("pstar0", z, lab)
    assert t[0, 1] == 1 and (t[2] == z["moveset"][2]).all()
    bad = lab.copy()
    bad[0] = 3                                          # not a legal action
    with pytest.raises(ST.StudentError, match="legal"):
        ST.targets("pstar0", z, bad)
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.targets("pstar", z, lab)


def test_the_trainer_refuses_unregistered_inputs():
    reg = {"pass": True, "rows_manifest_sha256": "r", "selfplay_manifest_sha256": "s",
           "labels_manifest_sha256": "l", "labels0_manifest_sha256": "l0"}
    ST.check_registration(reg, "human", "r", None)
    ST.check_registration(reg, "selfplay", "s", None)
    ST.check_registration(reg, "pstar0", "r", "l0")
    for arm, rows, lab in (("human", "s", None), ("selfplay", "r", None), ("pstar", "r", "l0"), ("pstar0", "r", "l")):
        with pytest.raises(ST.StudentError, match="VOID"):
            ST.check_registration(reg, arm, rows, lab)
    with pytest.raises(ST.StudentError):
        ST.check_registration({**reg, "pass": False}, "human", "r", None)


def test_the_learned_nothing_guard():
    assert ST.learned_guard({"hist": [1.95, 1.80], "uniform_val_nll": 1.95})["pass"]
    assert not ST.learned_guard({"hist": [1.95, 1.86], "uniform_val_nll": 1.95})["pass"]
    assert not ST.learned_guard({"hist": [1.95, 1.85], "uniform_val_nll": 1.95})["pass"]     # the tie: VOID
    assert not ST.learned_guard({"hist": [], "uniform_val_nll": 1.95})["pass"]
    V = {"mask": np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], np.int8),
         "target": np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], np.int8)}
    assert ST.uniform_nll(V) == pytest.approx((np.log(4) + 0.0) / 2)
