"""W0's window item (``main.ops.w0_window``): the census it emits is gpu_census.sh's, step for step; no step nests the
GPU lock or the memory cap; the item parses as an arm-chain window; outputs never land in a checkout or ``models/``."""
import json
import os
import re
from pathlib import Path

import pytest

from main.ops import w0_window as W
from main.ops.arm_chain import spec
from utils.paths import repo_path

PY = "/env/bin/python3"


def _census_script_lists():
    text = repo_path(W.CENSUS_DIR, "gpu_census.sh").read_text()
    lists = []
    for m in re.finditer(r"^for at in ([^;]+); do run \"([^\"]+)\"", text, flags=re.M):
        lists.append((tuple(m.group(1).split()), m.group(2)))
    return text, lists


def _fake_models(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    for run, steps in ((W.X26_RUN, (380000195,)), (W.X26C_RUN, (481000001, 489000196, 99000102))):
        ck = models / run / "checkpoints"
        ck.mkdir(parents=True)
        for s in steps:
            (ck / f"checkpoint_{s}_steps.zip").write_bytes(b"z")
        (models / run / "model_config.json").write_text("{}")
        (models / run / "metadata.json").write_text("{}")
    return models


def _fake_home(tmp_path: Path, with_u1000: bool) -> Path:
    home = tmp_path / "home"
    arc = home / W.U1000_ARCHIVE
    arc.mkdir(parents=True)
    if with_u1000:
        for s in (478000208, 479000039):
            (arc / f"checkpoint_{s}_steps.zip").write_bytes(b"u")
    return home


def _item(tmp_path: Path, with_u1000: bool = True):
    models = _fake_models(tmp_path)
    home = _fake_home(tmp_path, with_u1000)
    return W.window_item(worktree="/repo", require_commit="abcdef1", python=PY, out_dir=str(tmp_path / "out"),
                         models_dir=str(models), home=str(home)), models, home


def test_the_census_lists_are_gpu_census_sh_s_own():
    """The script is the census's definition: a checkpoint added to or dropped from it must move W0 too."""
    text, lists = _census_script_lists()
    by_run = {os.path.basename(r.replace('"', "")): ats for ats, r in lists}
    assert by_run.get(W.X26_RUN) == W.X26_AT
    assert by_run.get(W.X26C_RUN) == W.X26C_AT
    assert re.search(r"for at in " + " ".join(W.U1000_AT) + r"; do run \"\$U\"", text), "the u1000 list moved"
    assert W.U1000_ARCHIVE.split("/", 1)[1] in text and W.U1000_RUN in text
    assert "--dither 32 --vf-coef 0.5" in text and "--k 16 --vf-coef 0.5" in text


def test_every_census_checkpoint_is_its_own_gpu_step_in_the_scripts_order(tmp_path):
    item, models, _ = _item(tmp_path)
    census = [s for s in item["steps"] if s["name"].startswith("census_") and s["gpu"] and s["name"] != "census_cost"]
    want = [(W.X26_RUN, at) for at in W.X26_AT] + [(W.X26C_RUN, at) for at in W.X26C_AT] + \
           [(W.U1000_RUN, at) for at in W.U1000_AT]
    got = [(os.path.basename(s["argv"][s["argv"].index("--run-dir") + 1]), s["argv"][s["argv"].index("--at") + 1])
           for s in census]
    assert got == want
    for s in census:
        assert s["argv"][1].endswith(W.CENSUS_DIR + "/census_read.py")
        assert s["argv"][s["argv"].index("--device") + 1] == "cuda"
        assert s["timeout_s"] == 1200 and s["mem_gb"] == 24


def test_no_step_nests_the_gpu_lock_or_the_memory_cap_or_calls_the_census_script(tmp_path):
    """FINDING 5 resolved: the window wraps each GPU step in gpu_lock.sh + mem_cap.sh itself; a step that wrapped
    itself again (or ran gpu_census.sh, which does) would nest both and hide every checkpoint's exit code."""
    item, _, _ = _item(tmp_path)
    for s in item["steps"]:
        joined = " ".join(s["argv"])
        assert "gpu_lock" not in joined and "mem_cap" not in joined and "gpu_census.sh" not in joined, s["name"]
    assert [s["name"] for s in item["steps"][:3]] == ["compile_cell_b8", "x48_dump_replay", "census_prep_u1000"]
    assert item["steps"][2]["gpu"] is False
    assert item["steps"][-1]["name"] == "census_cost"


def test_the_cost_read_takes_the_newest_x26c_checkpoint_by_step(tmp_path):
    item, models, _ = _item(tmp_path)
    cost = item["steps"][-1]["argv"]
    assert cost[cost.index("--ckpt") + 1].endswith("checkpoint_489000196_steps.zip")


def test_without_the_u1000_archive_its_steps_are_left_out(tmp_path):
    item, _, _ = _item(tmp_path, with_u1000=False)
    names = [s["name"] for s in item["steps"]]
    assert "census_prep_u1000" not in names and not any(W.U1000_RUN in n for n in names)


def test_the_item_parses_as_an_arm_chain_window(tmp_path):
    item, _, _ = _item(tmp_path)
    chain = {"schema": spec.SCHEMA, "name": "w0-test", "job_dir": "/tmp/job", "repo": "/repo", "python": PY,
             "backend": {"kind": "x35", "launch_script": "/tmp/launch.sh", "stage": "/tmp/stage",
                         "lease_owner": "trainer", "lease_token_file": "/tmp/token"},
             "records": {"stage": "/tmp/records"}, "items": [item]}
    s = spec.parse(json.loads(json.dumps(chain)))
    w = s.items[0]
    assert w.kind == "window" and w.on_fail == "continue" and w.budget_min == 150
    assert sum(st.gpu for st in w.steps) == len(w.steps) - 1


@pytest.mark.parametrize("inside", ["repo", "models"])
def test_an_out_dir_inside_a_checkout_or_models_is_refused(tmp_path, inside):
    models = _fake_models(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    out = (repo if inside == "repo" else models) / "w0"
    with pytest.raises(W.W0Error, match="inside"):
        W.window_item(worktree=str(repo), require_commit="abcdef1", python=PY, out_dir=str(out),
                      models_dir=str(models), home=str(_fake_home(tmp_path, True)))


def test_prep_u1000_copies_once_and_never_overwrites(tmp_path):
    models = _fake_models(tmp_path)
    home = _fake_home(tmp_path, True)
    out = tmp_path / "out"
    lines = W.prep_u1000(str(out), str(models), str(home / W.U1000_ARCHIVE))
    assert sorted(lines) == sorted(["model_config.json: copied", "metadata.json: copied",
                                    "checkpoint_478000208_steps.zip: copied", "checkpoint_479000039_steps.zip: copied"])
    u = out / W.U1000_RUN
    (u / "checkpoints" / "checkpoint_478000208_steps.zip").write_bytes(b"kept")
    lines = W.prep_u1000(str(out), str(models), str(home / W.U1000_ARCHIVE))
    assert all(x.endswith("present") for x in lines)
    assert (u / "checkpoints" / "checkpoint_478000208_steps.zip").read_bytes() == b"kept"
    assert not (u / "checkpoints").is_symlink()


def test_prep_u1000_refuses_an_archive_without_both_neighbours(tmp_path):
    models = _fake_models(tmp_path)
    home = _fake_home(tmp_path, False)
    with pytest.raises(W.W0Error, match="478M and 479M"):
        W.prep_u1000(str(tmp_path / "out"), str(models), str(home / W.U1000_ARCHIVE))
