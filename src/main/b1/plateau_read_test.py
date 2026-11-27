"""The builder of the candidates' read against P* (``main.b1.plateau_read``): the registered blocks are the PREREG's
own text, the candidates come from the chain's markers, the guards leave out or refuse, and the size is 8,640 games."""
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from main.b1 import plateau_read as P
from utils.paths import repo_path

PIN = "a" * 40
RUNS = ("rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110")
STOPS = {"rb_b1c_s5101": 200_000_000, "rb_b1c_s5102": 250_000_000, "rb_b1c_s5110": 300_000_000}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    """The registry's ``pstar_plateau`` as item 4 registered it (a later re-base must not move these tests)."""
    from agents.training import baselines as B

    _models, run, ckpt = P.PSTAR_PATH.split("/", 2)
    monkeypatch.setattr(B, "get", lambda name, path=None: SimpleNamespace(run=run, checkpoint=ckpt, sha256=P.PSTAR_SHA256))


def _world(root: Path, *, history: Optional[Dict[str, Any]] = None, check: Optional[Dict[str, Any]] = None,
           runs=RUNS) -> Dict[str, Path]:
    """A scratch copy of the chain's layout: a marker and an arm-end check a run, and each run's final + metadata."""
    markers, models = root / "segB", root / "models"
    markers.mkdir(parents=True)
    for run in runs:
        stop = STOPS[run]
        d = models / run
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(f"final of {run}".encode())
        sha = hashlib.sha256(f"final of {run}".encode()).hexdigest()
        hist = (history or {}).get(run, [{"git_hash": PIN, "first_step": 0, "last_step": stop + 62_000,
                                          "torch": "2.8.0+cu126"}])
        (d / "metadata.json").write_text(json.dumps({"pin_history": hist}))
        tag = "NOT-PLATEAUED" if stop == 300_000_000 else "PLATEAU"
        (markers / f"{run}.candidate").write_text("".join(f"{k}={v}\n" for k, v in (
            ("run", run), ("tag", tag), ("stop_step", stop), ("newest_checkpoint", stop + 46),
            ("final", f"models/{run}/final_model.zip"), ("final_sha256", sha),
            ("status", "TIER1_PLATEAU newest=FLAT prev=FLAT"), ("at", "2026-11-28T10:00:00-08:00"))))
        rec = {"run": run, "arm": "C", "seed": int(run[-4:]), "pin": PIN, "step": stop, "arm_end": True, "ok": True,
               "problems": [], "k9b": {"verdict": "PASS", "fires": [], "reasons": []}}
        rec.update((check or {}).get(run, {}))
        (markers / f"{run}_to{stop // 1_000_000}M.arm_end_check.json").write_text(json.dumps(rec))
    return {"markers": markers, "models": models}


def _build(w: Dict[str, Path], **kw: Any):
    return P.build("C", w["markers"], PIN, w["models"], **kw)


def test_the_anchor_and_decision_blocks_are_item_4s_own_text():
    text = repo_path(P.PREREG).read_text()
    a6 = text[text.index("### Amendment 6 "):text.index("### Amendment 7 ")]
    flat = re.sub(r"\s+", " ", a6)
    assert f"`decision {json.dumps(P.DECISION)}`" in flat
    anchor = {P.GROUP: {"opponent": P.OPPONENT, "teamsets": list(P.TEAMSETS), "unit_seeds": P.UNIT_SEEDS, "models": "@"}}
    assert f"`anchor {json.dumps(anchor).replace(chr(34) + '@' + chr(34), '[the four]')}`" in flat
    assert f"`{P.PSTAR_PATH}`, sha256 `{P.PSTAR_SHA256}`, a `pinned` ref" in flat
    assert "`final` refs, `min_step` each run's stop step, `quiet_s` 300" in flat and P.QUIET_S == 300
    assert "8,640 games, ≈ 60 lane-h, ≈ 20 h on three lanes at §12's rate" in flat
    assert "| Kakuna, 25,920 games | — | ≈ 180 lane-h | ≈ 60 h on 3 lanes |" in text and P.GAMES_PER_LANE_HOUR == 144


def test_three_candidates_build_the_registered_spec_and_its_size(tmp_path):
    from main.reads import spec as S

    doc, rep = _build(_world(tmp_path))
    assert list(doc["models"]) == ["b1c_s5101", "b1c_s5102", "b1c_s5110", "pstar"]
    for run in RUNS:
        m = doc["models"][run[3:]]
        assert m == {"arm": "cand", "seed": int(run[-4:]),
                     "ref": {"kind": "final", "run": f"models/{run}", "min_step": STOPS[run], "quiet_s": 300}}
    assert doc["models"]["pstar"] == {"arm": "pstar", "seed": 2601, "ref": {
        "kind": "pinned", "path": P.PSTAR_PATH, "sha256": P.PSTAR_SHA256}}
    assert doc["anchor"] == {"kakuna": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix", "forum72", "p72"],
                                        "unit_seeds": "0-35", "models": list(doc["models"])}}
    assert doc["decision"] == P.DECISION and doc["decision"] is not P.DECISION
    assert (doc["pin"], doc["read"], doc["prereg"], doc["schema"]) == (PIN, P.READ, P.PREREG, P.SCHEMA)
    assert "rb_b1c_s5110 at 300,000,000 (NOT-PLATEAUED)" in doc["note"] and "LEFT OUT" not in doc["note"]
    assert (rep["planned_games"], rep["planned_units"], rep["lane_hours"], rep["hours_on_the_lanes"]) == (8640, 432, 60.0, 20.0)
    assert rep["degrees_of_freedom"] == 2 and rep["excluded"] == []
    parsed = S.parse(doc)                                   # the runner's own parser, the rule's parameter check included
    assert parsed.rule == "b1_kakuna" and parsed.pin == PIN and not parsed.pending
    assert parsed.models["b1c_s5102"].ref.min_step == 250_000_000 and len(parsed.anchor["kakuna"].unit_seeds) == 36
    assert P.dumps(doc) == P.dumps(_build({k: tmp_path / v for k, v in (("markers", "segB"), ("models", "models"))})[0])


def test_a_run_without_its_marker_is_not_ready_and_a_stray_marker_refuses(tmp_path):
    w = _world(tmp_path)
    (w["markers"] / "rb_b1c_s5110.candidate").rename(w["markers"] / "held")
    with pytest.raises(P.PlateauReadError, match=r"not ready: no candidate marker .* \['rb_b1c_s5110'\]"):
        _build(w)
    (w["markers"] / "held").rename(w["markers"] / "rb_b1w_s5103.candidate")
    with pytest.raises(P.PlateauReadError, match="are not of cell C's runs"):
        _build(w)


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.replace("tag=PLATEAU\n", "tag=DONE\n"), "tag 'DONE' is none of"),
    (lambda t: t.replace("status=", "state="), "not the chain's marker"),
    (lambda t: t.replace("stop_step=200000000", "stop_step=200M"), "is not a step"),
    (lambda t: t.replace("newest_checkpoint=200000046", "newest_checkpoint=201000046"), "not within one checkpoint"),
    (lambda t: t.replace("final=models/rb_b1c_s5101/", "final=models/rb_b1c_s5102/"), "not rb_b1c_s5101's"),
    (lambda t: re.sub(r"final_sha256=\w{4}", "final_sha256=zzzz", t), "is not a sha256"),
])
def test_a_marker_that_is_not_the_chains_shape_refuses(tmp_path, edit, match):
    w = _world(tmp_path)
    m = w["markers"] / "rb_b1c_s5101.candidate"
    m.write_text(edit(m.read_text()))
    with pytest.raises(P.PlateauReadError, match=match):
        _build(w)


def test_a_final_rewritten_after_its_marker_refuses(tmp_path):
    w = _world(tmp_path)
    (w["models"] / "rb_b1c_s5102" / "final_model.zip").write_bytes(b"a later leg")
    with pytest.raises(P.PlateauReadError, match="rb_b1c_s5102: .* no longer hashes to its marker"):
        _build(w)


TWO_ENTRIES = [{"git_hash": PIN, "first_step": 0, "last_step": 150_000_000},
               {"git_hash": PIN, "first_step": 150_000_000, "last_step": 200_062_000}]


@pytest.mark.parametrize("history, why", [
    (TWO_ENTRIES, "is not one entry at aaaaaaaa"),
    ([{"git_hash": "b" * 40, "first_step": 0, "last_step": 200_062_000}], "is not one entry at aaaaaaaa"),
    ([{"git_hash": PIN, "first_step": 1000, "last_step": 200_062_000}], "starts at 1000, not step 0"),
    ([{"git_hash": PIN, "first_step": 0, "last_step": 199_999_999}], "before its stop step 200,000,000"),
])
def test_a_candidate_whose_pin_history_is_not_one_entry_is_left_out_and_two_still_read(tmp_path, history, why):
    doc, rep = _build(_world(tmp_path, history={"rb_b1c_s5101": history}))
    assert list(doc["models"]) == ["b1c_s5102", "b1c_s5110", "pstar"] == doc["anchor"]["kakuna"]["models"]
    assert [x["run"] for x in rep["excluded"]] == ["rb_b1c_s5101"] and why in rep["excluded"][0]["why"][0]
    assert (rep["planned_games"], rep["degrees_of_freedom"], rep["lane_hours"]) == (6480, 1, 45.0)
    assert "LEFT OUT by item 4's guards: rb_b1c_s5101" in doc["note"] and doc["decision"] == P.DECISION


def test_a_k9b_void_leaves_the_candidate_out_and_one_candidate_refuses(tmp_path):
    void = {"ok": False, "problems": ["K9(b) is VOID"], "k9b": {"verdict": "VOID", "fires": ["p99 over its ceiling"]}}
    _doc, rep = _build(_world(tmp_path / "a", check={"rb_b1c_s5110": void}))
    assert rep["excluded"][0]["why"] == ["K9(b) VOID through its stop step (p99 over its ceiling)"]
    with pytest.raises(P.PlateauReadError, match="1 candidate.s. pass item 4's guards .*the rule needs two"):
        _build(_world(tmp_path / "b", check={"rb_b1c_s5110": void}, history={"rb_b1c_s5101": TWO_ENTRIES}))


@pytest.mark.parametrize("check, match", [
    ({"ok": False, "problems": ["n_epochs 8 != 10"]}, "a problem item 4's guards do not decide .n_epochs 8 != 10."),
    ({"ok": False, "k9b": {"verdict": "INCONCLUSIVE"}}, "a problem item 4's guards do not decide"),
    ({"pin": "b" * 40}, "not rb_b1c_s5102's arm-end check at the pin and stop step"),
    ({"step": 200_000_000}, "not rb_b1c_s5102's arm-end check"),
    ({"arm_end": False}, "not rb_b1c_s5102's arm-end check"),
])
def test_an_arm_end_check_that_says_anything_else_refuses(tmp_path, check, match):
    with pytest.raises(P.PlateauReadError, match=match):
        _build(_world(tmp_path, check={"rb_b1c_s5102": check}))


def test_a_missing_arm_end_check_refuses_and_names_the_kits_command(tmp_path):
    w = _world(tmp_path)
    (w["markers"] / "rb_b1c_s5102_to250M.arm_end_check.json").rename(tmp_path / "rb_b1c_s5102_to250M.arm_end_check.json")
    with pytest.raises(P.PlateauReadError, match="no readable arm-end check .* main.b1.look check-run .* --step 250000000 --arm-end"):
        _build(w)
    for run in (RUNS[0], RUNS[2]):
        n = f"{run}_to{STOPS[run] // 1_000_000}M.arm_end_check.json"
        (w["markers"] / n).rename(tmp_path / n)
    assert _build(w, checks=tmp_path)[1]["planned_games"] == 8640


def test_pstar_is_named_through_the_registry_and_a_moved_baseline_refuses(monkeypatch):
    from agents.training import baselines as B

    assert P.pstar_ref() == {"kind": "pinned", "path": P.PSTAR_PATH, "sha256": P.PSTAR_SHA256}
    monkeypatch.setattr(B, "get", lambda name, path=None: SimpleNamespace(
        run="rb_b1c_s5101", checkpoint="final_model.zip", sha256="0" * 64))
    with pytest.raises(P.PlateauReadError, match="'pstar_plateau' is now models/rb_b1c_s5101/final_model.zip"):
        P.pstar_ref()


def test_the_pins_and_the_cell_are_checked(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(P.PlateauReadError, match="--pin must be a full 40-hex"):
        P.build("C", w["markers"], "15136b61", w["models"])
    with pytest.raises(P.PlateauReadError, match="unknown cell 'X'"):
        P.build("X", w["markers"], PIN, w["models"])
    other = "c" * 40                     # the read may play at a later commit than the runs trained at (item 4, "The pin")
    assert P.build("C", w["markers"], other, w["models"], run_pin=PIN)[0]["pin"] == other
    with pytest.raises(P.PlateauReadError, match="not rb_b1c_s5101's arm-end check"):
        P.build("C", w["markers"], other, w["models"])


def test_the_files_fixed_figure_is_the_root_of_the_team_variance_alone():
    blk = {"psi_pp": 3.5, "schemes": {"our_team": {"v_t_pp2": 4.0, "v_s_pp2": 9.0}, "dyadic": {"v_t_pp2": 6.25, "v_s_pp2": 9.0}}}
    got = P.files_fixed({"read": P.READ, "decision": {"contrasts": {"candidate_U": blk}}})
    assert got["contrasts"]["candidate_U"] == {"psi_pp": 3.5, "files_fixed_se_pp": {"our_team": 2.0, "dyadic": 2.5},
                                              "wider_scheme": "dyadic"}
    assert got["reported_only"] is True
    with pytest.raises(P.PlateauReadError, match="no contrast"):
        P.files_fixed({"decision": {"contrasts": {}}})


def test_the_cli_prints_the_size_and_writes_only_with_out(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path)
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(w["models"]))
    argv = ["build", "--cell", "C", "--pin", PIN, "--markers", str(w["markers"])]
    assert P.main(argv) == 0
    rep = json.loads(capsys.readouterr().out)
    assert (rep["planned_games"], rep["lane_hours"], rep["spec"]) == (8640, 60.0, None)
    out = tmp_path / P.SPEC_NAME
    assert not out.exists() and P.main([*argv, "--out", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert out.read_text() == P.dumps(_build(w)[0]) and rep["spec_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert rep["next"][1] == f"python -m main.reads plan --spec {out} --state {P.STATE}"
    assert "--lane anchor --concurrency 3" in rep["next"][2]
    out.unlink()
    (w["markers"] / "rb_b1c_s5101.candidate").unlink()
    assert P.main([*argv, "--out", str(out)]) == 2 and not out.exists()
    assert "REFUSED: not ready" in capsys.readouterr().err
