"""The two-phase kit's phase boundary (``main.x35.kit``): ``verify-phase1`` accepts only the REGISTERED phase-1
exploiters (T55 gap 2), and ``describe --plan`` -- what ``arm_chain plan --describe`` asks -- builds a phase-2 arm from
declared placeholders before phase 1 has run and judges the record in full from the boundary on (T55 gap 1).

The world is synthetic: a tmp ``models/`` with P* and three exploiter runs written in the shape a real one has (a
recorded lineage block, ``pin_history``, the launcher's ``original_command``, per-checkpoint sidecars with the dose's
inputs), read back through the real ``main.lineage`` / ``main.sidecar_audit`` / ``main.dose``. Only a final's own step
(which lives inside an SB3 zip) is replaced. Nothing launches; nothing is read from the real ``models/``."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35 import refpop

PIN = "ab" * 20
OTHER_PIN = "cd" * 20
ARM = "a3_mx"
PSTAR_STEP = 280_000_170
FLOOR = PSTAR_STEP + 5_000_000
FINAL_STEP = FLOOR + 241
#: The dose a registered exploiter trains at: lr x updates/step = 1.005e-4 x 2.0345e-4 = 2.044e-8 (C0's, PREREG §3).
LR_OK = 1.005e-4
SCRIPT = "/x/src/main/launcher/__main__.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class World:
    """P*, the call record, the stage folder and the three exploiter runs, every piece mutable before it is judged."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.md = tmp / "models"
        self.stage = tmp / "stage"
        self.stage.mkdir()
        pdir = self.md / "rb_x26_s2601"
        (pdir / "checkpoints").mkdir(parents=True)
        self.pstar = pdir / "checkpoints" / f"checkpoint_{PSTAR_STEP}_steps.zip"
        self.pstar.write_bytes(b"P*")
        (pdir / "metadata.json").write_text(json.dumps({"git_hash": PIN}))
        (pdir / "model_config.json").write_text(json.dumps({"arch_signature": "arch_v1"}))
        doc: Dict[str, Any] = {"schema": refpop.SCHEMA, "side": "production", "archive": [],
                               "pstar": {"label": "x26_pstar", "path": str(self.pstar), "step": PSTAR_STEP,
                                         "sha256": _sha(b"P*")}}
        doc["content_sha256"] = hashlib.sha256(refpop._canon(doc)).hexdigest()
        self.call = doc
        self.call_file = self.stage / "reference_population_production.json"
        self.call_file.write_text(json.dumps(doc))
        B.write_pin_file(B.pin_path(str(self.stage), "production"), PIN, "FP32_ALT", "production")
        self.steps: Dict[str, int] = {}
        monkeypatch.setattr(K, "_final_step", lambda z: self.steps.get(str(z), FINAL_STEP))

    def run_dir(self, seed: int) -> Path:
        return self.md / K.run_name(ARM, seed)

    def make_run(self, seed: int, *, lr: float = LR_OK, role: str = "exploiter", parent_sha: Optional[str] = None,
                 lineage: bool = True, git_hash: str = PIN, cmd_pin: str = PIN, extra_argv: Optional[List[str]] = None,
                 num_timesteps: int = FINAL_STEP, final: bytes = b"") -> Path:
        rd = self.run_dir(seed)
        (rd / "checkpoints").mkdir(parents=True)
        (rd / "final_model.zip").write_bytes(final or f"final {seed}".encode())
        (rd / "model_config.json").write_text(json.dumps({"arch_signature": "arch_v1"}))
        ref = {"path": str(self.pstar), "resolved_path": str(self.pstar), "run_dir": str(self.pstar.parent.parent),
               "run_name": "rb_x26_s2601", "git_hash": PIN, "arch_signature": "arch_v1", "num_timesteps": PSTAR_STEP,
               "resolved_file": str(self.pstar), "resolved_num_timesteps": PSTAR_STEP}
        meta: Dict[str, Any] = {
            "git_hash": git_hash, "pin_source": "pin_commit", "num_timesteps": num_timesteps,
            "pin_history": [{"git_hash": git_hash, "pin_source": "pin_commit", "first_step": PSTAR_STEP,
                             "last_step": num_timesteps}],
            "original_command": f"{SCRIPT} --restart-interval-hours 24 --pin-commit {cmd_pin} "
                                + shlex.join(K.build_argv(ARM, seed, self.call) + (extra_argv or []))}
        if lineage:
            meta["lineage"] = {"schema": 1, "role": role, "fork_step": PSTAR_STEP, "teachers": [],
                               "fork_parent": dict(ref, sha256=parent_sha or _sha(b"P*")),
                               "exploiter_target": dict(ref, sha256=None)}
        (rd / "metadata.json").write_text(json.dumps(meta))
        for s in (281_000_000, 283_000_000, FINAL_STEP):
            (rd / "checkpoints" / f"checkpoint_{s}_steps.zip").write_bytes(b"z")
            (rd / "checkpoints" / f"checkpoint_{s}_steps.json").write_text(json.dumps(
                {"num_timesteps": s, "git_hash": git_hash, "lr": lr, "batch_size": 2048, "grad_accum_steps": 32,
                 "n_epochs": 10, "n_steps": 384, "dose": {"rollout_rows": 98304}}))
        return rd

    def make_runs(self, **kw: Any) -> None:
        for s in K.ARMS[ARM].seeds:
            self.make_run(s, **kw)

    def record(self, name: str = "phase1_a3.json") -> Path:
        out = self.stage / name
        K.phase1_record(ARM, self.call, str(self.md), str(out))
        return out

    def verify(self, path: Path) -> List[str]:
        return K.verify_phase1(str(path), call=self.call, models_dir=str(self.md), pin=PIN)


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def _edit(path: Path, fn: Any) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


# ------------------------------------------------------------------------------------------------ gap 2: verify
def test_the_registered_phase_1_record_verifies(w: World) -> None:
    w.make_runs()
    assert w.verify(w.record()) == []


def test_three_other_files_that_hash_are_not_the_registered_exploiters(w: World, tmp_path: Path) -> None:
    """The old check passed ANY three files that hash (the A3 STAND-IN was three R0 checkpoints). Now the record must
    name the three registered runs."""
    w.make_runs()
    rec = w.record()
    stand_in = []
    for i in range(3):
        z = tmp_path / f"r0_{i}.zip"
        z.write_bytes(f"r0 {i}".encode())
        stand_in.append({"label": f"STANDIN_r0_{i}", "run": "STANDIN (rb_x35_r0_s3531, validation only)",
                         "path": str(z), "step": FINAL_STEP, "sha256": _sha(z.read_bytes())})
    _edit(rec, lambda d: d.update(exploiters=stand_in))
    probs = w.verify(rec)
    assert len(probs) == 1 and "not the 3 registered exploiters" in probs[0], probs


def test_the_registered_runs_at_the_wrong_files_are_refused(w: World) -> None:
    """Right run names, but a path that is not the run's own final: another run's final, or an intermediate
    checkpoint of its own."""
    w.make_runs()
    rec = w.record()
    other = json.loads(rec.read_text())["exploiters"][1]
    _edit(rec, lambda d: d["exploiters"][0].update(path=other["path"], sha256=other["sha256"]))
    assert any("rb_x35_mx_s3511: path" in p and "is not its final" in p for p in w.verify(rec))
    ck = str(w.run_dir(3512) / "checkpoints" / f"checkpoint_{FINAL_STEP}_steps.zip")
    _edit(rec, lambda d: d["exploiters"][1].update(path=ck, sha256=_sha(b"z")))
    assert any("rb_x35_mx_s3512: path" in p and "is not its final" in p for p in w.verify(rec))


def test_a_final_that_changed_since_the_record_is_refused(w: World) -> None:
    w.make_runs()
    rec = w.record()
    w.run_dir(3512).joinpath("final_model.zip").write_bytes(b"swapped")
    assert any("rb_x35_mx_s3512" in p and "sha256 changed since the record" in p for p in w.verify(rec))


def test_a_final_whose_own_step_is_not_the_recorded_one_is_refused(w: World) -> None:
    w.make_runs()
    rec = w.record()
    w.steps[str(w.run_dir(3513) / "final_model.zip")] = FINAL_STEP + 1
    assert any("the final's own step" in p and "rb_x35_mx_s3513" in p for p in w.verify(rec))


@pytest.mark.parametrize("kw,needle", [
    ({"role": "fork"}, "lineage role 'fork', not 'exploiter'"),
    ({"parent_sha": "e" * 64}, "its fork parent is"),
    ({"lineage": False}, "its lineage is not a RECORDED block"),
])
def test_a_run_whose_recorded_lineage_is_not_an_exploiter_of_pstar_is_refused(w: World, kw: Dict[str, Any],
                                                                              needle: str) -> None:
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, **kw)
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and needle in p for p in probs), probs
    assert not [p for p in probs if "rb_x35_mx_s3512" in p or "rb_x35_mx_s3513" in p], probs


@pytest.mark.parametrize("kw,needle", [
    ({"git_hash": OTHER_PIN}, "recorded pin(s)"),
    ({"cmd_pin": OTHER_PIN}, "launched with --pin-commit"),
])
def test_a_run_trained_at_another_pin_is_refused(w: World, kw: Dict[str, Any], needle: str) -> None:
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, **kw)
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and needle in p for p in probs), probs


def test_a_run_launched_with_another_argv_is_refused(w: World) -> None:
    """R0's argv (a reader's ``--fork-lr`` and ``--fork-lr-freeze``) on an otherwise right run: not the kit's argv."""
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, extra_argv=["--fork-lr", "5.6e-05", "--fork-lr-freeze"])
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and "not the kit's argv for a3_mx s3511" in p for p in probs), probs


def test_a_run_at_another_dose_is_refused_and_one_within_the_tolerance_passes(w: World) -> None:
    """The registered dose is C0's, 2.044e-8, within 10 % (PREREG §3): R0's frozen 5.6e-5 is 0.55x."""
    w.make_run(3511, lr=5.6e-5)
    w.make_run(3512, lr=LR_OK * 1.05)
    w.make_run(3513, lr=LR_OK * 1.5)
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and "dose rate" in p and "0.56x" in p for p in probs), probs
    assert any("rb_x35_mx_s3513" in p and "dose rate" in p and "1.50x" in p for p in probs), probs
    assert not [p for p in probs if "rb_x35_mx_s3512" in p], probs


def test_a_run_that_never_reached_the_registered_step_is_refused(w: World) -> None:
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, num_timesteps=FLOOR - 1)
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and "the run's own step" in p for p in probs), probs


def test_a_phase_1_arm_with_no_registered_dose_is_never_verified(w: World, monkeypatch: pytest.MonkeyPatch) -> None:
    w.make_runs()
    rec = w.record()
    monkeypatch.delitem(K.PHASE1_DOSE, ARM)
    assert any("no registered dose" in p for p in w.verify(rec))


def test_a_record_of_another_p_star_or_another_arm_is_refused(w: World) -> None:
    w.make_runs()
    rec = w.record()
    _edit(rec, lambda d: d.update(pstar_sha256="f" * 64))
    assert any("is from P*" in p for p in w.verify(rec))
    _edit(rec, lambda d: d.update(pstar_sha256=_sha(b"P*"), arm="a6_lx"))
    assert any("registered exploiters" in p or "is for arm" in p for p in w.verify(rec))
    _edit(rec, lambda d: d.update(arm="a3"))
    assert any("not a phase-1 arm" in p for p in w.verify(rec))
    assert any("unreadable" in p for p in w.verify(w.stage / "absent.json"))


def test_the_verify_command_defaults_to_the_stage_folder_and_exits_by_the_verdict(w: World, capsys: Any) -> None:
    """``kit verify-phase1 <stage>/phase1_a3.json`` (the chain item's and ``launch_x35.sh``'s form, no other flag): the
    call record and the side's pin are read from the record's own folder."""
    w.make_runs()
    rec = w.record()
    assert K.main(["verify-phase1", str(rec), "--models-dir", str(w.md)]) == 0
    w.run_dir(3512).joinpath("final_model.zip").write_bytes(b"swapped")
    assert K.main(["verify-phase1", str(rec), "--models-dir", str(w.md)]) == 1
    assert "sha256 changed since the record" in capsys.readouterr().err
