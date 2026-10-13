"""`utils.env_file` — the read-only "does this env already satisfy that env file" check behind
`scripts/bootstrap.sh --adopt-env` (the safe way to re-stamp the SHARED conda env after an env-file edit
that adds only pins the env already has).

Every case runs on a synthetic env file and synthetic installed-package maps, so it reads the same under
any interpreter; one case runs the CLI against a fake interpreter's conda-meta + a real subprocess listing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from utils import env_file as ef
from utils.paths import src_root

_YML = """\
name: demo
channels:
  - nvidia/label/cuda-12.6.3
  - defaults
dependencies:
  - python=3.11.15
  - cuda-cccl=12.6.77
  - pip:
      # a comment
      - --extra-index-url https://download.pytorch.org/whl/cu126
      - Jinja2==3.1.6
      - requests[socks]
      - python-socks[asyncio]
      - torch==2.8.0+cu126
      - typing_extensions==4.15.0   # trailing comment
"""

_CONDA_OK = {"python": "3.11.15", "cuda-cccl": "12.6.77"}
_PIP_OK = {"jinja2": "3.1.6", "requests": "2.33.1", "python-socks": "2.7.1", "torch": "2.8.0+cu126",
           "typing-extensions": "4.15.0"}


def test_the_pins_are_read_from_both_blocks_and_flags_are_skipped():
    assert [(n, v) for n, v, _ in ef.conda_pins(_YML)] == [("python", "3.11.15"), ("cuda-cccl", "12.6.77")]
    assert [(n, v) for n, v, _ in ef.pip_pins(_YML)] == [
        ("jinja2", "3.1.6"), ("requests", None), ("python-socks", None), ("torch", "2.8.0+cu126"),
        ("typing-extensions", "4.15.0")]


def test_an_env_that_has_everything_is_satisfied():
    assert ef.unsatisfied(_YML, _PIP_OK, _CONDA_OK) == []


def test_a_conda_pin_is_a_PREFIX_match_like_condas_equals():
    yml = "dependencies:\n  - python=3.11\n"
    assert ef.unsatisfied(yml, {}, {"python": "3.11.15"}) == []
    assert ef.unsatisfied(yml, {}, {"python": "3.110.1"}), "3.110.1 is not 3.11"
    assert ef.unsatisfied(yml, {}, {"python": "3.12.1"})


def test_a_missing_package_a_wrong_version_and_a_wrong_cuda_build_are_each_named():
    pip = dict(_PIP_OK)
    del pip["python-socks"]
    pip["torch"] = "2.8.0+cu128"             # the same release, a DIFFERENT build: not satisfied
    pip["jinja2"] = "3.1.5"
    bad = ef.unsatisfied(_YML, pip, {"python": "3.11.15"})
    text = "\n".join(bad)
    assert "pip python-socks[asyncio]: not installed" in text
    assert "pip torch==2.8.0+cu126: the env has 2.8.0+cu128" in text
    assert "pip Jinja2==3.1.6: the env has 3.1.5" in text
    assert "conda cuda-cccl=12.6.77: not installed" in text
    assert len(bad) == 4


def test_names_compare_under_pep503_normalisation():
    pip = {"typing-extensions": "4.15.0", "jinja2": "3.1.6", "requests": "1", "python-socks": "1",
           "torch": "2.8.0+cu126"}
    assert ef.unsatisfied(_YML, pip, _CONDA_OK) == []
    assert ef.norm("Typing_Extensions") == ef.norm("typing.extensions") == "typing-extensions"


def test_packages_the_file_does_not_name_are_not_a_failure():
    pip = dict(_PIP_OK, scipy="1.17.1", netron="9.1.0")
    assert ef.unsatisfied(_YML, pip, dict(_CONDA_OK, openssl="3.6.3")) == []


def test_version_equality_is_pep440_not_string_equality():
    yml = "dependencies:\n  - pip:\n      - numpy==2.4.3\n"
    assert ef.unsatisfied(yml, {"numpy": "2.4.3"}, {}) == []
    assert ef.unsatisfied(yml, {"numpy": "2.4.3.0"}, {}) == []      # equal under PEP 440
    assert ef.unsatisfied(yml, {"numpy": "2.4.30"}, {})


def test_conda_installed_reads_conda_meta_without_running_conda(tmp_path):
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "python-3.11.15-h1_1.json").write_text(json.dumps({"name": "python", "version": "3.11.15"}))
    (meta / "cuda-crt-dev_linux-64-12.6.85-0.json").write_text(
        json.dumps({"name": "cuda-crt-dev_linux-64", "version": "12.6.85"}))
    (meta / "broken.json").write_text("{not json")
    (meta / "history").write_text("not a package")
    assert ef.conda_installed(str(tmp_path)) == {"python": "3.11.15", "cuda-crt-dev-linux-64": "12.6.85"}


def test_the_cli_exits_1_naming_the_gaps_and_0_when_met(tmp_path):
    """The real CLI against THIS interpreter (so `pip_installed` runs for real) and a synthetic file."""
    ok = tmp_path / "ok.yml"
    ok.write_text(f"dependencies:\n  - pip:\n      - pytest=={_installed('pytest')}\n")
    bad = tmp_path / "bad.yml"
    bad.write_text("dependencies:\n  - pip:\n      - pytest==0.0.1\n      - no-such-package-xyz==1.0\n")
    r = _cli("check", "--env-file", str(ok))
    assert r.returncode == 0 and "nothing to install" in r.stdout, r.stdout + r.stderr
    r = _cli("check", "--env-file", str(bad))
    assert r.returncode == 1 and "no-such-package-xyz==1.0: not installed" in r.stdout, r.stdout + r.stderr
    assert "pytest==0.0.1: the env has" in r.stdout


def test_the_cli_can_inspect_ANOTHER_interpreter(tmp_path):
    f = tmp_path / "f.yml"
    f.write_text(f"dependencies:\n  - pip:\n      - pytest=={_installed('pytest')}\n")
    r = _cli("check", "--env-file", str(f), "--python", sys.executable)
    assert r.returncode == 0, r.stdout + r.stderr


def _cli(*args: str) -> "subprocess.CompletedProcess[str]":
    """The module as a subprocess, importing THIS checkout's `utils` (a worktree's PYTHONPATH rule)."""
    env = dict(os.environ, PYTHONPATH=str(src_root()))
    return subprocess.run([sys.executable, "-m", "utils.env_file", *args], capture_output=True, text=True,
                          timeout=120, env=env)


def _installed(dist: str) -> str:
    import importlib.metadata as md
    return md.version(dist)


@pytest.mark.parametrize("entry", ["--extra-index-url https://x/y", "", "   "])
def test_flags_and_blanks_are_not_pins(entry):
    assert ef._entry_pin(entry, "==") is None
