"""Gate ① through the Rust core, and the reader's teeth and determinism (integration).

Gate ①: re-encoding a banked turn reproduces the recorded obs byte-equal. The routine tier replays
a fixed slice of the COMMITTED bank (4 battles per source cycle); the slow tier replays all of it.
While this checkout's encoder identity (the obs golden's hash, the obs width, the arch signature)
equals the one stamped at build, byte-equality is REQUIRED. After a deliberate encoder change the
recorded rows are a property of the RECORDING commit: the test then requires only that every
banked decision replays to the same legal tokens and played choice (the input logs stay valid), and
gate ①'s byte check runs in a worktree pinned at the recording commit.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum.bank_test import BANK_V1

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: The reader-vs-recording probability bar (float reassociation between the recording and the read).
DP_BAR = 1e-4


def _subset(bank, per_source: int):
    seen = {}
    ids = []
    for b in bank.battles:
        k = b.source["label"]
        if seen.get(k, 0) < per_source:
            seen[k] = seen.get(k, 0) + 1
            ids.append(b.battle_id)
    keep = set(ids)
    return B.Bank(bank.manifest, [b for b in bank.battles if b.battle_id in keep],
                  [d for d in bank.decisions if d["battle"] in keep])


def _check(sub):
    from main.policy_spectrum.reader import reencode

    rows, masks, gate = reencode(sub, workers=2)
    same_encoder = B.encoder_identity() == sub.manifest["encoder_identity"]
    assert gate["recorded_rows_checked"] > 0
    if same_encoder:
        assert gate["obs_as_recorded"], gate
    else:
        pytest.skip("the encoder changed since the bank was built — gate ①'s byte check belongs to "
                    "the recording commit; the structural replay above passed")
    return rows, masks, gate


def test_gate_one_on_a_slice_of_the_committed_bank():
    bank = B.load_bank(BANK_V1)
    sub = _subset(bank, per_source=4)
    assert len({b.source["label"] for b in sub.battles}) == len(bank.manifest["sources"])
    _check(sub)


@pytest.mark.slow
def test_gate_one_on_the_whole_committed_bank():
    _check(B.load_bank(BANK_V1))


def test_gate_one_has_teeth():
    """One flipped obs byte in a recording, a dropped decision and a wrong action each FAIL."""
    from main.policy_spectrum.replay import replay

    bank = B.load_bank(BANK_V1)
    b = next(x for x in bank.battles if x.banked_side == x.traced_side)
    res = replay([b.recorded()])[0]
    ds = res.decisions[0 if b.traced_side == "p1" else 1]
    obs = np.stack([d.row for d in ds]).copy()
    masks = np.stack([d.mask.astype(bool) for d in ds])
    acts = np.array([next(i for i, t in d.tokens.items() if t == d.choice) for d in ds])
    rec = {"obs": obs, "action_mask": masks, "actions": acts, "has_state": np.ones(len(ds), np.int8)}
    assert B.check_recorded(b, ds, rec)["decisions"] == len(ds)
    bad = dict(rec, obs=obs.copy())
    bad["obs"].view(np.uint8)[3, 17] ^= 1
    with pytest.raises(B.GateOneFailure, match="obs differs"):
        B.check_recorded(b, ds, bad)
    with pytest.raises(B.GateOneFailure, match="re-encoded decisions"):
        B.check_recorded(b, ds[:-1], rec)
    wrong = dict(rec, actions=acts.copy())
    legal = sorted(ds[0].tokens)
    wrong["actions"][0] = next(i for i in legal if i != acts[0])
    with pytest.raises(B.GateOneFailure, match="recorded action"):
        B.check_recorded(b, ds, wrong)


def test_the_reader_is_deterministic_on_an_x5_checkpoint(tmp_path, restore_torch_globals):
    """Two reads of one fresh production (X5) checkpoint on a slice of the bank are byte-identical (the
    recording policies in the archive predate the X5 version break, so the reader cannot load them at HEAD)."""
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import load_probs, read_checkpoint, reencode
    from main.train.production_args import production_args

    bank = B.load_bank(BANK_V1)
    sub = _subset(bank, 1)
    run = tmp_path / "run_x5"
    run.mkdir()
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(3, args=production_args())
        snap = run / "snapshot_000000001000.zip"
        model.save(str(snap))
    (run / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    rows, masks, gate = reencode(sub)
    r1 = read_checkpoint(sub, rows, masks, gate, snap, "a", tmp_path, threads=2)
    r2 = read_checkpoint(sub, rows, masks, gate, snap, "b", tmp_path, threads=2)
    pa, pb = load_probs(tmp_path, "a", sub), load_probs(tmp_path, "b", sub)
    assert hashlib.sha256(pa.tobytes()).digest() == hashlib.sha256(pb.tobytes()).digest()
    strip = lambda r: {k: v for k, v in r.items() if k != "label"}   # noqa: E731
    assert strip(r1) == strip(r2)


def test_reader_reproduces_the_recording_policy_and_is_deterministic(tmp_path,
                                                                     restore_torch_globals):
    from utils.paths import main_models_dir

    from main.policy_spectrum.reader import load_probs, read_checkpoint, reencode

    md = main_models_dir()
    if md is None:
        pytest.skip("no models/ archive")
    bank = B.load_bank(BANK_V1)
    label = "N0@74M"
    battles = [b for b in bank.battles if b.source["label"] == label][:6]
    ids = {b.battle_id for b in battles}
    sub = B.Bank(bank.manifest, battles, [d for d in bank.decisions if d["battle"] in ids])
    snap = md / battles[0].source["banked_policy"]
    if not snap.exists():
        pytest.skip(f"{snap} is not in the archive")
    rows, masks, gate = reencode(sub)
    from agents.model.model_version import ModelVersionError
    from agents.model.model_version.version_break import LAST_BLOB_COMMIT
    try:
        r1 = read_checkpoint(sub, rows, masks, gate, snap, "a", tmp_path, threads=2, models_root=md)
    except ModelVersionError as e:
        # The recording policy predates the X5 version break (config v144): this code REFUSES it with the pinned
        # fix (asserted: the refusal names the last pre-break commit), and the reproduction claim belongs to a
        # checkout at or before it. Determinism is pinned on a fresh X5 checkpoint
        # (`test_the_reader_is_deterministic_on_an_x5_checkpoint`).
        assert "PRE-GENERATION" in str(e) and LAST_BLOB_COMMIT[:12] in str(e), str(e)[:400]
        pytest.skip(f"the recording policy {snap} predates the X5 version break and is refused at HEAD "
                    f"(asserted); its reproduction check runs pinned at <= {LAST_BLOB_COMMIT[:12]}")
    r2 = read_checkpoint(sub, rows, masks, gate, snap, "b", tmp_path, threads=2, models_root=md)
    agree = r1["recording_agreement"]
    assert agree["decisions"] == len(sub.decisions)
    # Determinism holds whatever the op's semantics: two reads are byte-identical.
    pa, pb = load_probs(tmp_path, "a", sub), load_probs(tmp_path, "b", sub)
    assert hashlib.sha256(pa.tobytes()).digest() == hashlib.sha256(pb.tobytes()).digest()
    strip = lambda r: {k: v for k, v in r.items() if k != "label"}   # noqa: E731
    assert strip(r1) == strip(r2)
    if bank.manifest.get("op_semantics") != B.op_semantics():
        # The recorded logits came from a forward with DIFFERENT op feature semantics (the bank predates
        # gen3_nonformula_damage_v1, which re-prices Seismic Toss / Return / … in every kernel without a
        # signature bump): this checkout cannot reproduce them, and the reader must SAY so, not pass.
        assert agree["max_abs_dp"] >= DP_BAR, agree
        pytest.skip(f"the bank was recorded under op semantics {bank.manifest.get('op_semantics')!r}, "
                    f"this checkout computes {B.op_semantics()!r} — the reproduction check belongs to "
                    f"a checkout at the recording's semantics (read max|dp| {agree['max_abs_dp']:.3g}); "
                    "determinism passed")
    assert agree["max_abs_dp"] < DP_BAR
    # A flipped argmax is legal only at a DECLARED near-tie — Lane E's rule (`judge_flips`): a flip whose
    # larger top-1 / top-2 margin is < 2 x the bar is a tie; any other flip is a real disagreement.
    assert agree["argmax_agree"] == 1.0 or agree["argmax_flip_max_margin"] < 2 * DP_BAR, agree
