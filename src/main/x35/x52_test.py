"""X35 X52, the practice ceiling (``main.x35.x52``; launch sheet §0 AM-X52): the argv IS C0's at +5M plus the ONE
lever ``--trainee-teams <the 36 practised files>``, the kit argv equals the committed ``checkargs`` transcripts at the
production pin, the lever refuses files that differ from the registered manifest, and the arm is side-pinned outside
the closed ``STAGES``. Nothing launches or plays."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import pytest

from main.x35 import batch as B
from main.x35 import kit as K
from main.x35 import x52 as X
from main.x35.batch_test import PIN, OTHER, _fill_fp32, _run
from main.x35.kit_test import flags, standin_call
from utils.paths import repo_path

HALVES = repo_path(X.HALVES_DIR)


def _manifest() -> dict:
    return json.loads(repo_path(X.MANIFEST).read_text())


# ------------------------------------------------------------------------------------------------------ the argv
@pytest.mark.parametrize("seed", [3501, 3502])
def test_x52_is_c0s_argv_at_plus_5m_plus_the_trainee_teams_lever(seed):
    call = standin_call()
    x52, c0 = K.build_argv("x52", seed, call), K.build_argv("c0", seed, call)
    fx, fc = flags(x52), flags(c0)
    assert {k for k in fx if fx[k] != fc.get(k)} == {"--steps", "--run-name", "--trainee-teams"}
    assert set(fx) - set(fc) == {"--trainee-teams"} and set(fc) <= set(fx)
    assert fx["--steps"] == str(call["pstar"]["step"] + 5_000_000) and fx["--run-name"] == f"rb_x35_x52prac_s{seed}"
    i = x52.index("--trainee-teams")          # C0's tokens in C0's order, the lever after --device cuda
    swap = {fc["--steps"]: fx["--steps"], fc["--run-name"]: fx["--run-name"]}
    assert x52[:i] + x52[i + 2:] == [swap.get(t, t) for t in c0] and x52[i - 2:i] == ["--device", "cuda"]


def test_the_lever_names_exactly_the_registered_practised_half_repo_relative():
    files = X.lever()[1].split(",")
    want = sorted(f"{i}.txt" for i, t in _manifest()["teams"].items() if t["half"] == "practised")
    assert [Path(f).name for f in files] == want and len(files) == 36
    assert all(not Path(f).is_absolute() and f.startswith(X.PRACTISED_DIR + "/") for f in files)
    unpractised = {i for i, t in _manifest()["teams"].items() if t["half"] == "unpractised"}
    assert len(unpractised) == 36 and not unpractised & {Path(f).stem for f in files}


def _transcript_argv(path: Path) -> List[str]:
    line = path.read_text().splitlines()[0]
    return line.split('--argv "', 1)[1].rstrip('"').split()


@pytest.mark.parametrize("seed", [3501, 3502])
def test_the_kit_argv_equals_the_committed_checkargs_transcript_at_the_production_pin(seed):
    p = HALVES / f"checkargs_x52_s{seed}_f4b25238.txt"
    text = p.read_text()
    assert text.rstrip().endswith("[exit 0]") and "unrecognized                   : 0" in text
    assert "PINNED parser @f4b25238" in text
    want = _transcript_argv(p)
    model = want[want.index("--model") + 1]
    assert model.endswith("/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"), model
    call = {"side": "production", "archive": [], "pstar": {"path": model, "step": 280000170, "sha256": "s"}}
    assert K.build_argv("x52", seed, call) == want


# --------------------------------------------------------------------------------------------- F-X52-4: the files
def _copy_halves(tmp_path: Path) -> Path:
    shutil.copytree(HALVES, tmp_path / X.HALVES_DIR, ignore=shutil.ignore_patterns("checkargs_*"))
    return tmp_path


def test_the_committed_files_match_their_manifest_and_a_copy_verifies():
    assert len(X.practised_files()) == 36
    assert X.main(["verify"]) == 0


def test_an_edited_practised_file_is_refused(tmp_path):
    root = _copy_halves(tmp_path)
    f = sorted((root / X.PRACTISED_DIR).glob("*.txt"))[0]
    f.write_text(f.read_text() + "\n")
    with pytest.raises(X.X52Error, match="sha256"):
        X.practised_files(root)


def test_a_missing_or_extra_practised_file_is_refused(tmp_path):
    root = _copy_halves(tmp_path)
    d = root / X.PRACTISED_DIR
    sorted(d.glob("*.txt"))[0].unlink()
    with pytest.raises(X.X52Error, match="differ from the manifest"):
        X.practised_files(root)
    root2 = _copy_halves(tmp_path / "b")
    (root2 / X.PRACTISED_DIR / "sfv2_9999.txt").write_text("x")
    with pytest.raises(X.X52Error, match="differ from the manifest"):
        X.practised_files(root2)


def test_an_unpractised_team_moved_into_the_practised_dir_is_refused(tmp_path):
    root = _copy_halves(tmp_path)
    u = sorted((root / X.HALVES_DIR / "teams_unpractised").glob("*.txt"))[0]
    shutil.copy(u, root / X.PRACTISED_DIR / u.name)
    with pytest.raises(X.X52Error, match="differ from the manifest"):
        X.practised_files(root)


def test_the_kit_turns_a_bad_file_into_a_kit_error(monkeypatch):
    def bad(repo=None):
        raise X.X52Error("sha256 differs")
    monkeypatch.setattr(X, "practised_files", bad)
    with pytest.raises(K.KitError, match="x52: sha256 differs"):
        K.build_argv("x52", 3501, standin_call())


def test_the_draw_reads_its_source_from_the_production_pin():
    """The halves are drawn from the corpus AS COMMITTED AT X35's production pin (AM-X52-a)."""
    pin = (repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
           / "x35_pin_batch1_production.txt").read_text()
    assert f"sha: {_manifest()['source']['commit']}" in pin


# ------------------------------------------------------------------------------------------ the gates (AM-1 / AM-8)
def test_x52_is_side_pinned_and_outside_the_closed_batches():
    assert set(B.X52_ARMS) == set(X.X52_ARMS) and set(B.X52_ARMS) <= set(B.SIDE_PINNED_ARMS)
    assert not {a for _, forks in B.STAGES + B.BATCH2_STAGES + B.R0_STAGES for a, _ in forks} & set(X.X52_ARMS)
    assert not {a for a, _ in K.BATCH1 + K.BATCH2} & set(X.X52_ARMS)
    assert {(a, s) for _, forks in B.X52_STAGES for a, s in forks} == {("x52", 3501), ("x52", 3502)}
    a = K.ARMS["x52"]
    assert (a.side, a.length, a.seeds, a.phase) == ("production", 5_000_000, (3501, 3502), 0)
    assert "x52" not in K.NO_POOL_ARMS          # it self-plays against the side's seeded pool, as C0 does


def test_an_x52_run_on_a_wrong_pin_holds_the_production_pin_check(tmp_path):
    stage, models = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    B.pin_write_if_absent(PIN, str(stage), "production", "FP32_ALT")
    _run(models, "rb_x35_x52prac_s3501", PIN)
    ok, lines = B.pin_check(PIN, str(stage), str(models), "production")
    assert ok, lines
    (models / "rb_x35_x52prac_s3501" / "metadata.json").write_text(json.dumps({"git_hash": OTHER}))
    ok, lines = B.pin_check(PIN, str(stage), str(models), "production")
    assert not ok and any("rb_x35_x52prac_s3501" in x and OTHER[:12] in x for x in lines), lines


def test_x52_waits_for_fp32_alt_then_has_no_other_condition(tmp_path):
    st, why = B.stage_status(str(tmp_path), "x52", 3501)
    assert st == B.PENDING and "FP32_ALT" in why
    _fill_fp32(tmp_path)
    st, why = B.stage_status(str(tmp_path), "x52", 3502)
    assert st == B.ALLOWED and "AM-X52" in why and B.stage_of("x52", 3502) == "X52"
