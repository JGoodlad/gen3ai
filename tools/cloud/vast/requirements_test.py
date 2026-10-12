"""The cu128 requirements derivation — on a synthetic env file AND on the real ``environment_torch28.yml``."""
import pytest

from tools.cloud.vast import requirements as R
from utils.paths import repo_path

ENV = """name: x
dependencies:
  - python=3.11.15
  - cuda-cccl=12.6.77
  - pip:
      # a comment
      - --extra-index-url https://download.pytorch.org/whl/cu126
      - numpy==2.4.3
      - nvidia-cublas-cu12==12.6.4.1   # dropped
      - torch==2.8.0+cu126
      - torchvision==0.23.0+cu126

      - triton==3.4.0
"""


def test_derive_swaps_the_index_and_local_versions_and_drops_the_nvidia_pins():
    assert R.derive(ENV) == ["--extra-index-url https://download.pytorch.org/whl/cu128", "numpy==2.4.3",
                             "torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "triton==3.4.0",
                             *R.UNDECLARED_RUNTIME]


def test_the_undeclared_runtime_imports_are_added_once():
    out = R.derive(ENV.replace("- numpy==2.4.3", "- scipy==9.9.9"))
    assert [e for e in out if e.startswith("scipy")] == ["scipy==9.9.9"]       # the env file's pin wins
    assert "scikit-learn==1.9.0" in out


def test_lock_from_freeze_pins_everything_but_the_cu12_runtime_and_the_repo():
    freeze = ("click==8.4.2\ngen3ai==1.0.0\npip==26.0.1\nnvidia-cublas-cu12==12.6.4.1\ntorch==2.8.0+cu126\n"
              "torchvision==0.23.0+cu126\ntriton==3.4.0\n")
    assert R.lock_from_freeze(freeze) == ["--extra-index-url https://download.pytorch.org/whl/cu128",
                                          "click==8.4.2", "torch==2.8.0+cu128", "torchvision==0.23.0+cu128",
                                          "triton==3.4.0"]
    with pytest.raises(ValueError):
        R.lock_from_freeze("click==8.4.2\n")


def test_conda_only_packages_never_leak_into_the_pip_list():
    assert not any("cuda-cccl" in e or "python=" in e for e in R.derive(ENV))


def test_refuses_a_file_without_torch_and_a_malformed_cuda_tag():
    with pytest.raises(ValueError):
        R.derive("dependencies:\n  - pip:\n      - numpy==1\n")
    with pytest.raises(ValueError):
        R.derive(ENV, cuda="12.8")


def test_the_real_env_file_derives_the_same_torch_version_on_cu128():
    with open(repo_path("environment_torch28.yml"), encoding="utf-8") as fh:
        text = fh.read()
    local = {e.split("==")[0]: e for e in R.pip_block(text) if "==" in e}
    remote = {e.split("==")[0]: e for e in R.derive(text) if "==" in e}
    assert local["torch"].endswith("+cu126") and remote["torch"] == local["torch"].replace("+cu126", "+cu128")
    assert not [k for k in remote if k.startswith("nvidia-")]
    # every OTHER pin is byte-identical: the CUDA build is the ONLY difference between the two stacks
    same = {k for k in local if not k.startswith("nvidia-") and not local[k].endswith("+cu126")}
    assert same, "the env file's pip block parsed to nothing"
    assert {k: remote[k] for k in same} == {k: local[k] for k in same}
    assert any("whl/cu128" in e for e in R.derive(text))
