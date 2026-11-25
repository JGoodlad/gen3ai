"""X72's finals (``main.x72.finals``): the restated tables against the frozen registration, the copy step, and every
way a copy stops being the registered final. Synthetic run dirs only; nothing here touches ``models/`` or the real
archive."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from main.x72 import finals as F
from utils.hashing import sha256_file
from utils.paths import repo_path

DIR = repo_path("designs/research_state/measurements/x72_swap_plateau_2026-11-25")


def make_swap(models: Path, seed: int) -> str:
    """A stand-in banked swap final of lineage ``seed``; returns its sha256."""
    d = models / F.SWAP[seed][0]
    d.mkdir(parents=True)
    (d / "final_model.zip").write_bytes(f"swap final {seed}".encode())
    (d / "metadata.json").write_text(json.dumps({"num_timesteps": F.ROUND0_TARGET + 61_000 + seed}))
    return sha256_file(d / "final_model.zip")


def make_run(models: Path, seed: int, k: int, *, parent_sha: str, over: int = 61_102, **edit) -> Path:
    """A stand-in X72 run dir whose final is lineage ``seed``'s after round ``k``. ``edit`` overrides record fields:
    ``step``, ``pins``, ``ent``, ``cli_ent``, ``kl``, ``refresh``, ``parent``, ``cli_seed``, ``cli_run``."""
    d = models / F.run_name(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_model.zip").write_bytes(f"x72 final {seed} r{k}".encode())
    (d / "final_model.magnet.pt").write_bytes(f"magnet {seed} r{k}".encode())
    md = {"num_timesteps": edit.get("step", F.target(k) + over),
          "pin_history": [{"git_hash": p} for p in edit.get("pins", [F.TRAIN_PIN] * k)],
          "ent_coef": edit.get("ent", 0.0),
          "cli_args": {"ent_coef": edit.get("cli_ent", 0.0), "seed": edit.get("cli_seed", seed),
                       "run_name": edit.get("cli_run", F.run_name(seed))},
          "lineage": {"schema": 1, "role": "fork", "fork_parent": {"sha256": edit.get("parent", parent_sha)}}}
    (d / "metadata.json").write_text(json.dumps(md))
    (d / "model_config.json").write_text(json.dumps({"magnet_kl_coef": edit.get("kl", 0.05),
                                                     "magnet_refresh_steps": edit.get("refresh", 2_000_000)}))
    return d


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """``(models, archive root)`` with three stand-in swap finals REGISTERED (``F.SWAP`` patched to their sha256)."""
    models, root = tmp_path / "models", tmp_path / "archive" / "finals"
    swap = {s: (F.SWAP[s][0], make_swap(models, s)) for s in F.SEEDS}
    monkeypatch.setattr(F, "SWAP", swap)
    return models, root


def train_round(models: Path, k: int, **edit) -> None:
    for s in F.SEEDS:
        make_run(models, s, k, parent_sha=F.SWAP[s][1], **edit)


# ------------------------------------------------------------------------------------- the restated tables
def test_the_restated_tables_are_the_frozen_registrations():
    spec = json.loads((DIR / "read_spec_kakuna.json").read_text())
    for seed, (run, sha) in F.SWAP.items():
        ref = spec["models"][f"s0_s{run[-4:]}"]["ref"]
        assert ref["sha256"] == sha and ref["path"] == f"models/{run}/final_model.zip"
        argv = (DIR / f"argv_x72_r1_s{seed}.txt").read_text().split()
        assert argv[argv.index("--model") + 1].endswith(f"/models/{run}/final_model.zip")
        assert argv[argv.index("--steps") + 1] == str(F.target(1))
        assert argv[argv.index("--seed") + 1] == str(seed)
        assert argv[argv.index("--run-name") + 1] == F.run_name(seed)
    rule = runpy.run_path(str(DIR / "x72_rule.py"))
    assert (rule["N"], rule["MAX_ROUNDS"]) == (len(F.SEEDS), F.MAX_ROUNDS)
    text = " ".join((DIR / "PREREG.md").read_text().split())
    assert [f"{F.target(k):,}" for k in range(5)] == ["300,000,170", "320,000,170", "340,000,170", "360,000,170",
                                                      "380,000,170"]
    for k in range(1, 5):
        assert f"| {F.target(k):,} |" in text
    assert f"Pin `{F.TRAIN_PIN}`" in text and "Seeds 7341 / 7342 / 7343" in text
    assert ("`final_model.zip`, `final_model.magnet.pt`, `model_config.json` and `metadata.json` are COPIED to "
            "`") in text and "/gen3ai_archive/x72_swap_plateau/finals/r<k>/rb_x72_swapplat_s734N/` and" in text
    assert set(F.FILES) == {"final_model.zip", "final_model.magnet.pt", "model_config.json", "metadata.json"}
    assert "`magnet_kl_coef` 0.05 with `magnet_refresh_steps` 2,000,000 in `model_config.json`" in text
    assert "`ent_coef` 0.0 in `metadata.json` and its `cli_args`" in text
    assert (F.ENT_COEF, F.MAGNET_KL_COEF, F.MAGNET_REFRESH_STEPS) == (0.0, 0.05, 2_000_000)


def test_the_archive_root_is_under_the_gen3ai_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(tmp_path))
    assert F.copy_dir(2, 7342) == tmp_path / "x72_swap_plateau" / "finals" / "r2" / "rb_x72_swapplat_s7342"
    for bad in (0, 5, True, "1"):
        with pytest.raises(F.FinalsError):
            F.copy_dir(bad, 7341)
    with pytest.raises(F.FinalsError):
        F.run_name(7321)


# ------------------------------------------------------------------------------------------------ the copy step
def test_archive_copies_the_four_files_records_their_sha256_and_load_final_verifies_them(world):
    models, root = world
    train_round(models, 1)
    got = F.archive(1, root=root, models_dir=models)
    assert [(f.round, f.seed, f.run, f.source) for f in got] == [(1, s, F.run_name(s), "archive") for s in F.SEEDS]
    for f in got:
        d = root / "r1" / f.run
        assert f.path == str(d / "final_model.zip") and f.sha256 == sha256_file(models / f.run / "final_model.zip")
        man = json.loads((d / "finals.json").read_text())
        assert man["schema"] == "x72_finals_v1" and (man["round"], man["seed"], man["step"]) == (1, f.seed, f.step)
        assert set(man["files"]) == set(F.FILES)
        assert all(man["files"][n]["sha256"] == sha256_file(models / f.run / n) for n in F.FILES)
        assert sorted(p.name for p in d.iterdir()) == sorted([*F.FILES, "finals.json"])   # no temp file left
    # the next round overwrites the run dir's final: the round-1 copy still verifies, and round 2 archives beside it
    train_round(models, 2)
    assert [f.sha256 for f in F.load_round(1, root=root, models_dir=models)] == [f.sha256 for f in got]
    r2 = F.archive(2, root=root, models_dir=models)
    assert {f.sha256 for f in r2}.isdisjoint({f.sha256 for f in got}) and all(f.step >= F.target(2) for f in r2)


def test_archive_is_a_no_op_on_the_same_bytes_and_never_overwrites_other_bytes(world):
    models, root = world
    train_round(models, 1)
    first = F.archive(1, root=root, models_dir=models)
    stamp = (root / "r1" / F.run_name(7341) / "finals.json").read_text()
    assert F.archive(1, root=root, models_dir=models) == first
    assert (root / "r1" / F.run_name(7341) / "finals.json").read_text() == stamp
    (models / F.run_name(7342) / "final_model.zip").write_bytes(b"a later final of the same round")
    with pytest.raises(F.FinalsError, match="OTHER bytes"):
        F.archive(1, root=root, models_dir=models)
    assert [f.sha256 for f in F.load_round(1, root=root, models_dir=models)] == [f.sha256 for f in first]


def test_archive_judges_every_lineage_before_it_copies_anything(world):
    models, root = world
    train_round(models, 1)
    make_run(models, 7343, 1, parent_sha=F.SWAP[7343][1], step=F.target(1) - 1)
    with pytest.raises(F.FinalsError, match="rb_x72_swapplat_s7343 r1: final at 320,000,169"):
        F.archive(1, root=root, models_dir=models)
    assert not root.exists()


def test_a_file_that_changes_under_the_copy_is_a_refusal(world, monkeypatch):
    models, root = world
    train_round(models, 1)
    real = F.shutil.copyfile

    def racing(src, dst):
        real(src, dst)
        if Path(src).name == "metadata.json":
            Path(src).write_text(Path(src).read_text() + " ")

    monkeypatch.setattr(F.shutil, "copyfile", racing)
    with pytest.raises(F.FinalsError, match="changed while it was copied"):
        F.archive(1, root=root, models_dir=models)
    d = root / "r1" / F.run_name(7341)
    assert not (d / "finals.json").exists() and not (d / "metadata.json").exists()
    with pytest.raises(F.FinalsError, match="no finals.json"):
        F.load_final(1, 7341, root=root, models_dir=models)


# ------------------------------------------------------------------------------- what a copy's record must show
@pytest.mark.parametrize("edit, why", [
    ({"step": F.ROUND0_TARGET + 20_000_000 - 1}, "final at 320,000,169"),
    ({"step": F.ROUND0_TARGET + 40_000_000}, "final at 340,000,170"),
    ({"pins": []}, "pin_history names nothing"),
    ({"pins": [F.TRAIN_PIN, "333a73ddda7829b9ccdf9e17ef04624eb40edeba"]}, "not only the training pin 9e97cad5"),
    ({"ent": 0.05}, "metadata.json ent_coef is 0.05"),
    ({"ent": None}, "metadata.json ent_coef is None"),
    ({"cli_ent": 0.05}, "cli_args.ent_coef is 0.05"),
    ({"cli_ent": None}, "cli_args.ent_coef is None"),
    ({"kl": 0.0}, "magnet_kl_coef 0.0"),
    ({"refresh": 1_000_000}, "magnet_refresh_steps 1000000"),
    ({"parent": "0" * 64}, "the wrong parent"),
    ({"cli_seed": 7342}, "cli_args.seed is 7342"),
    ({"cli_run": "rb_x72_swapplat_s7342"}, "cli_args.run_name is 'rb_x72_swapplat_s7342'"),
])
def test_each_record_guard_refuses_alone(world, edit, why):
    models, _root = world
    d = make_run(models, 7341, 1, parent_sha=F.SWAP[7341][1], **edit)
    probs = F.record_problems(d, 1, 7341)
    assert len(probs) == 1 and why in probs[0], probs


def test_a_clean_record_passes_and_an_inherited_seed_or_run_name_is_not_a_problem(world):
    models, _root = world
    assert F.record_problems(make_run(models, 7341, 1, parent_sha=F.SWAP[7341][1]), 1, 7341) == []
    d = make_run(models, 7342, 3, parent_sha=F.SWAP[7342][1], cli_seed=None, cli_run=None, over=0)
    assert F.record_problems(d, 3, 7342) == []
    assert F.record_problems(d, 2, 7342) != [] and F.record_problems(d, 4, 7342) != []     # the round is the step's


@pytest.mark.parametrize("name", F.FILES)
def test_a_missing_or_empty_file_is_named(world, name):
    models, _root = world
    d = make_run(models, 7341, 1, parent_sha=F.SWAP[7341][1])
    (d / name).write_bytes(b"")
    assert name in F.record_problems(d, 1, 7341)[0]
    (d / name).unlink()
    assert name in F.record_problems(d, 1, 7341)[0]


@pytest.mark.parametrize("name", F.FILES)
def test_a_copy_that_moved_after_it_was_archived_is_refused(world, name):
    models, root = world
    train_round(models, 1)
    F.archive(1, root=root, models_dir=models)
    p = root / "r1" / F.run_name(7342) / name
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(F.FinalsError, match=f"rb_x72_swapplat_s7342 r1: {name}.*no longer match"):
        F.load_round(1, root=root, models_dir=models)


def test_a_manifest_of_another_round_or_lineage_is_refused(world):
    models, root = world
    train_round(models, 1)
    F.archive(1, root=root, models_dir=models)
    (root / "r2").mkdir()
    (root / "r1" / F.run_name(7341)).rename(root / "r2" / F.run_name(7341))
    with pytest.raises(F.FinalsError, match=r"names \{'round': 1\}"):
        F.load_final(2, 7341, root=root, models_dir=models)
    with pytest.raises(F.FinalsError, match="no finals.json"):
        F.load_final(1, 7341, root=root, models_dir=models)


# ---------------------------------------------------------------------------------------------------- round zero
def test_round_zero_is_the_banked_swap_final_at_its_registered_sha256(world):
    models, root = world
    got = F.load_round(0, root=root, models_dir=models)
    assert [(f.run, f.sha256, f.source) for f in got] == [(*F.SWAP[s], "banked") for s in F.SEEDS]
    assert got[0].path == str(models / "rb_x35_ce3dswap_s7321" / "final_model.zip")
    (models / "rb_x35_ce3dswap_s7322" / "final_model.zip").write_bytes(b"moved")
    with pytest.raises(F.FinalsError, match="rb_x35_ce3dswap_s7322: final_model.zip has sha256 .* not the registered"):
        F.load_round(0, root=root, models_dir=models)


def test_the_cli_exits_three_on_a_refusal_and_prints_the_finals_otherwise(world, capsys):
    models, root = world
    args = ["--archive-root", str(root), "--models-dir", str(models)]
    assert F.main(["verify", "--round", "1", *args]) == 3
    assert "REFUSED" in capsys.readouterr().err
    train_round(models, 1)
    assert F.main(["archive", "--round", "1", *args]) == 0
    assert [f["seed"] for f in json.loads(capsys.readouterr().out)] == list(F.SEEDS)
    assert F.main(["verify", "--round", "1", *args]) == 0 and F.main(["verify", "--round", "0", *args]) == 0
