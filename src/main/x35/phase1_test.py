"""The two-phase kit's phase boundary (``main.x35.kit``): ``verify-phase1`` accepts only the REGISTERED phase-1
exploiters (T55 gap 2), and ``describe --plan`` -- what ``arm_chain plan --describe`` asks -- builds a phase-2 arm from
declared placeholders before phase 1 has run and judges the record in full from the boundary on (T55 gap 1).

The world is synthetic: a tmp ``models/`` with P* and three exploiter runs written in the shape a real one has (a
recorded lineage block, ``pin_history``, the launcher's ``original_command``, per-checkpoint sidecars with the dose's
inputs), read back through the real ``main.lineage`` / ``main.sidecar_audit`` / ``main.dose``. Only a final's own step
(which lives inside an SB3 zip) is replaced. Nothing launches; nothing is read from the real ``models/``."""
from __future__ import annotations

import contextlib
import hashlib
import io
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

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, *, md: Optional[Path] = None,
                 stage: Optional[Path] = None) -> None:
        self.md = md or tmp / "models"
        self.stage = stage or tmp / "stage"
        self.stage.mkdir(exist_ok=True)
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


def test_a_run_at_another_dose_is_warned_never_refused(w: World) -> None:
    """The A3 PREREG registers a dose bound for the phase-2 forks (§9), not the phase-1 exploiters: a dose off C0's
    2.044e-8 (R0's frozen 5.6e-5 is 0.55x) is a WARN line and a PASS, one within 10 % is silent."""
    w.make_run(3511, lr=5.6e-5)
    w.make_run(3512, lr=LR_OK * 1.05)
    w.make_run(3513, lr=LR_OK * 1.5)
    warns: List[str] = []
    probs = K.verify_phase1(str(w.record()), call=w.call, models_dir=str(w.md), pin=PIN, warns=warns)
    assert probs == [], probs
    assert any("rb_x35_mx_s3511" in x and "dose rate" in x and "0.56x" in x for x in warns), warns
    assert any("rb_x35_mx_s3513" in x and "dose rate" in x and "1.50x" in x for x in warns), warns
    assert not [x for x in warns if "rb_x35_mx_s3512" in x], warns


def test_the_verify_command_prints_the_dose_warn_and_still_passes(w: World, capsys: Any) -> None:
    w.make_run(3511, lr=5.6e-5)
    w.make_run(3512)
    w.make_run(3513)
    assert K.main(["verify-phase1", str(w.record()), "--models-dir", str(w.md)]) == 0
    err = capsys.readouterr().err
    assert "WARN (report-only, never a refusal)" in err and "rb_x35_mx_s3511" in err and "0.56x" in err, err
    assert "verify-phase1: PASS (1 warning(s))" in err, err


def test_a_wrong_argv_is_still_refused_beside_a_wrong_dose(w: World) -> None:
    """The dose went report-only; the argv check, which catches a wrongly launched exploiter, stays hard."""
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, lr=5.6e-5, extra_argv=["--fork-lr", "5.6e-05", "--fork-lr-freeze"])
    warns: List[str] = []
    probs = K.verify_phase1(str(w.record()), call=w.call, models_dir=str(w.md), pin=PIN, warns=warns)
    assert any("not the kit's argv for a3_mx s3511" in p for p in probs), probs
    assert not any("dose" in p for p in probs), probs
    assert any("dose rate" in x for x in warns), warns


def test_a_run_that_never_reached_the_registered_step_is_refused(w: World) -> None:
    w.make_run(3512)
    w.make_run(3513)
    w.make_run(3511, num_timesteps=FLOOR - 1)
    probs = w.verify(w.record())
    assert any("rb_x35_mx_s3511" in p and "the run's own step" in p for p in probs), probs


def test_a_phase_1_arm_with_no_dose_reference_is_warned_not_refused(w: World, monkeypatch: pytest.MonkeyPatch) -> None:
    w.make_runs()
    rec = w.record()
    monkeypatch.delitem(K.PHASE1_DOSE, ARM)
    warns: List[str] = []
    assert K.verify_phase1(str(rec), call=w.call, models_dir=str(w.md), pin=PIN, warns=warns) == []
    assert any("no registered dose reference" in x for x in warns), warns


def test_a_record_of_another_p_star_another_arm_or_a_placeholder_is_refused(w: World) -> None:
    w.make_runs()
    rec = w.record()
    _edit(rec, lambda d: d.update(pstar_sha256="f" * 64))
    assert any("is from P*" in p for p in w.verify(rec))
    _edit(rec, lambda d: d.update(pstar_sha256=_sha(b"P*"), arm="a6_lx"))
    assert any("registered exploiters" in p or "is for arm" in p for p in w.verify(rec))
    _edit(rec, lambda d: d.update(arm="a3"))
    assert any("not a phase-1 arm" in p for p in w.verify(rec))
    ph = w.stage / "ph.json"
    ph.write_text(json.dumps(K.phase1_placeholder_record(ARM, w.call, str(w.md))))
    assert any("PLACEHOLDER record is for planning only" in p for p in w.verify(ph))
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


# ------------------------------------------------------------------------------------- gap 1: plan / describe
def _plan(w: World, seed: int = 3501) -> Dict[str, Any]:
    return K.describe_for_plan("a3", seed, w.call, stage=str(w.stage), models_dir=str(w.md))


def _stable(doc: Dict[str, Any]) -> List[str]:
    argv = doc["argv"]
    return argv[argv.index("--stable-opponents") + 1].split(",")


def test_a_phase_2_item_plans_on_placeholders_before_phase_1_has_run(w: World) -> None:
    """Before phase 1 there is no record and no exploiter run; plan used to refuse every phase-2 item ("arm a3 is phase
    2: it needs the a3_mx phase-1 record"). Now the argv is built from the registered paths and says so."""
    doc = _plan(w)
    assert doc["phase1"]["mode"] == "PLACEHOLDER" and doc["phase1"]["finals_present"] == []
    assert doc["run_name"] == "rb_x35_a3mx_s3501" and doc["delta_steps"] == 20_000_000
    assert _stable(doc) == [f"{w.md}/rb_x35_mx_s{s}/final_model.zip:mx_s{s}" for s in (3511, 3512, 3513)]
    f = doc["argv"]
    assert f[f.index("--stable-opponent-selfplay-share") + 1] == "0.4" and "--stable-opponent-pfsp" in f
    assert f[f.index("--steps") + 1] == str(PSTAR_STEP + 20_000_000)
    assert _plan(w) == doc                                      # deterministic: the same filesystem, the same answer
    # a phase-1 arm and every other arm are plain describe: no phase-1 basis to declare
    assert "phase1" not in K.describe_for_plan("a3_mx", 3511, w.call, stage=str(w.stage), models_dir=str(w.md))
    assert "phase1" not in K.describe_for_plan("c0", 3501, w.call, stage=str(w.stage), models_dir=str(w.md))


def test_the_placeholder_argv_is_the_argv_the_record_will_give(w: World) -> None:
    """The placeholders are every value known before phase 1 runs, so once phase 1 has run and its record verifies
    the planned argv is exactly the launch's: nothing the plan judged moves."""
    planned = _plan(w)["argv"]
    w.make_runs()
    w.record()
    after = _plan(w)
    assert after["phase1"]["mode"] == "RECORD" and after["argv"] == planned


def test_phase_1_still_running_keeps_the_placeholders(w: World) -> None:
    """The boundary is all three finals (or the record), not the first run to appear: a planner run mid-phase-1 (one
    exploiter ended, one running, one not started) is still before the boundary."""
    w.make_run(3511)
    w.run_dir(3512).mkdir(parents=True)                          # running: a run dir, no final yet
    doc = _plan(w)
    assert doc["phase1"]["mode"] == "PLACEHOLDER" and doc["phase1"]["finals_present"] == ["rb_x35_mx_s3511"]


def test_at_the_boundary_a_missing_record_refuses_the_plan(w: World) -> None:
    """All three finals exist and nothing is written: placeholders would hide that the step phase 2 waits for was not
    taken."""
    w.make_runs()
    with pytest.raises(K.KitError, match=r"phase 1 \(a3_mx\) has finished and its record is NOT written.*phase1_a3.json"):
        _plan(w)


def test_at_the_boundary_a_wrong_record_refuses_the_plan_and_the_right_one_is_judged_in_full(w: World) -> None:
    """Once the record exists the plan is as strict as the launch: the same ``verify_phase1``, so a wrong record never
    plans OK, and a right one gives the real argv."""
    w.make_runs()
    rec = w.record()
    good = _plan(w)
    assert good["phase1"] == {"mode": "RECORD", "file": str(rec)}
    assert _stable(good) == [f"{e['path']}:{e['label']}" for e in json.loads(rec.read_text())["exploiters"]]
    w.run_dir(3512).joinpath("final_model.zip").write_bytes(b"swapped")
    with pytest.raises(K.KitError, match=r"does not verify: rb_x35_mx_s3512.*sha256 changed since the record"):
        _plan(w)
    w.run_dir(3512).joinpath("final_model.zip").write_bytes(b"final 3512")
    w.run_dir(3513).joinpath("metadata.json").unlink()
    with pytest.raises(K.KitError, match=r"does not verify.*rb_x35_mx_s3513"):
        _plan(w)


def test_a_stand_in_record_at_the_stage_never_plans(w: World, tmp_path: Path) -> None:
    """The A3 registration's STAND-IN (three banked R0 checkpoints, which hash) at the default path is refused in
    plan exactly as in verify-phase1."""
    stand_in = [{"label": f"STANDIN_{i}", "run": "STANDIN (rb_x35_r0_s3531, validation only)",
                 "path": str(w.pstar), "step": FINAL_STEP, "sha256": _sha(b"P*")} for i in range(3)]
    (w.stage / "phase1_a3.json").write_text(json.dumps(
        {"schema": K.PHASE1_SCHEMA, "arm": ARM, "pstar_sha256": _sha(b"P*"), "pstar_step": PSTAR_STEP,
         "exploiters": stand_in}))
    with pytest.raises(K.KitError, match="does not verify: .*not the 3 registered exploiters"):
        _plan(w)


def test_a_placeholder_record_is_never_a_launch_argv_nor_a_record(w: World) -> None:
    ph = K.phase1_placeholder_record(ARM, w.call, str(w.md))
    assert K.phase1_shape_problems(ph, ARM, w.call, str(w.md), placeholder=True) == []
    with pytest.raises(K.KitError, match="PLACEHOLDER phase-1 record is for planning only"):
        K.build_argv("a3", 3501, w.call, ph)
    with pytest.raises(K.KitError, match="PLACEHOLDER phase-1 record is for planning only"):
        K.describe("a3", 3501, w.call, ph)
    f = w.stage / "phase1_a3.json"
    f.write_text(json.dumps(ph))
    assert any("PLACEHOLDER record is for planning only" in p for p in w.verify(f))
    # its shape is judged: a wrong run, label, path, step or a real-looking hash is a problem, never accepted
    bad = json.loads(json.dumps(ph))
    bad["exploiters"][0].update(label="x", path="/elsewhere/final_model.zip", step=1, sha256="0" * 64)
    probs = K.phase1_shape_problems(bad, ARM, w.call, str(w.md), placeholder=True)
    assert len(probs) == 4, probs
    bad["exploiters"][1]["run"] = "rb_x35_other_s1"
    assert "not the 3 registered exploiters" in K.phase1_shape_problems(bad, ARM, w.call, str(w.md), placeholder=True)[-1]


def test_the_describe_command_plans_a_phase_2_item_before_and_after_the_boundary(w: World, capsys: Any) -> None:
    """``kit describe --plan --stage S`` is the form ``arm_chain plan`` runs; --plan takes the record from the stage."""
    base = ["describe", "--arm", "a3", "--seed", "3501", "--call", str(w.call_file), "--models-dir", str(w.md)]
    assert K.main([*base, "--plan", "--stage", str(w.stage)]) == 0
    assert json.loads(capsys.readouterr().out)["phase1"]["mode"] == "PLACEHOLDER"
    assert K.main(base) == 2                                     # no --plan: describe stays strict (a launch is not a plan)
    assert "needs the a3_mx phase-1 record" in capsys.readouterr().err
    assert K.main([*base, "--plan"]) == 2 and K.main([*base, "--plan", "--stage", str(w.stage), "--phase1", "x"]) == 2
    capsys.readouterr()
    w.make_runs()
    assert K.main([*base, "--plan", "--stage", str(w.stage)]) == 2          # the boundary, no record
    assert "has finished" in capsys.readouterr().err
    w.record()
    assert K.main([*base, "--plan", "--stage", str(w.stage)]) == 0
    assert json.loads(capsys.readouterr().out)["phase1"]["mode"] == "RECORD"


def test_arm_chain_plan_takes_a_phase_2_item_through_the_real_kit_before_and_after_the_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """The contract end to end: ``arm_chain plan --describe`` runs the REAL kit's parser on the argv ``describe_offline``
    builds. Before phase 1 the phase-2 item plans OK on placeholders; once the record is on the stage it is judged in
    full; a swapped final or a missing record refuses the plan (it used to refuse BEFORE phase 1 and never judge after)."""
    from main.ops.arm_chain import cli
    from main.ops.arm_chain.testing_fakes import FakeRunner, make_world

    cw = make_world(str(tmp_path / "chain"))
    w = World(tmp_path, monkeypatch, md=Path(cw.models), stage=Path(cw.repo) / "stage")
    item = cw.arm("a3", 3501, cw.pin_a, run="rb_x35_a3mx_s3501")
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps(cw.chain_dict([item])))

    def real_kit(argv: Any, out: str) -> int:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = K.main(list(argv)[list(argv).index("main.x35.kit") + 1:])
        Path(out).write_text(buf.getvalue())
        return rc

    def plan() -> str:
        # describe AND stage-check (``plan`` asks the stage gate of every arm): both the real kit, in process
        runner = FakeRunner().on(lambda a: "main.x35.kit" in a and ("describe" in a or "stage-check" in a), real_kit)
        rc = cli.main(["plan", str(chain), "--describe"], runner=runner)
        out = capsys.readouterr().out
        assert (rc == 0) == out.rstrip().endswith("PLAN OK"), out
        return out

    out = plan()
    assert "PLAN OK" in out and "its finals are PLACEHOLDERS, judged for shape only" in out
    w.make_runs()
    out = plan()
    assert "PLAN REFUSED" in out and "phase 1 (a3_mx) has finished and its record is NOT written" in out
    w.record()
    out = plan()
    assert "PLAN OK" in out and "phase-1 record VERIFIED in full" in out and "PLACEHOLDERS" not in out
    w.run_dir(3512).joinpath("final_model.zip").write_bytes(b"swapped")
    out = plan()
    assert "PLAN REFUSED" in out and "does not verify" in out and "sha256 changed since the record" in out
