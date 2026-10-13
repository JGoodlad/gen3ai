"""DERIVE the remote box's pip requirements from ``environment_torch28.yml`` — the CUDA 12.8 variant of OUR env.

Why a derivation, not a second env file: the shared env (``gen3ai_torch28``) is torch 2.8.0+cu126, whose wheels
carry kernels up to sm_90 only — an RTX 5090 (Blackwell, sm_120) needs a CUDA 12.8 build. The remote installs the
SAME torch version from the cu128 index, and every other pin stays byte-identical to the shared env, so the only
difference between the local and remote stacks is the CUDA build (designs/ops/cloud_gpu_benchmark.md "cu126 vs
cu128"). Deriving it at provisioning time means the variant can never drift from the env file.

The transformation (each step unit-tested in ``requirements_test.py``):
- the ``--extra-index-url .../whl/cu126`` line -> ``.../whl/cu128``;
- every ``+cu126`` local version (torch, torchaudio, torchvision) -> ``+cu128``;
- every ``nvidia-*-cu12`` pin is DROPPED: they are the cu126 runtime's exact versions, and torch's cu128 wheel
  declares its own (cu128) nvidia dependencies, which the resolver then pulls;
- the conda-only CUDA header packages (``cuda-cccl`` & co., AOTInductor's headers) are not pip packages and are
  not in the pip block, so nothing to do — the remote's AOT path is out of scope (FINDING in the runbook).

    python -m tools.cloud.vast.requirements [--cuda cu128] [--env environment_torch28.yml] > requirements_cu128.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List


def pip_block(env_yml_text: str) -> List[str]:
    """The entries of the env file's ``- pip:`` block, in order (comments and blank lines dropped)."""
    out: List[str] = []
    in_pip = False
    pip_indent = -1
    for line in env_yml_text.splitlines():
        stripped = line.strip()
        if re.match(r"^\s*-\s*pip:\s*$", line):
            in_pip, pip_indent = True, len(line) - len(line.lstrip())
            continue
        if not in_pip or not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= pip_indent:
            break
        m = re.match(r"^-\s+(.*?)\s*(#.*)?$", stripped)
        if m:
            out.append(m.group(1))
    return out


#: Packages the CODE imports at runtime that ``environment_torch28.yml`` did NOT declare (FINDING 2026-10-12: the shared
#: env got them from its clone of gen3ai_stable, so a fresh build from the file lacked them; the first rented box died in
#: the training path's ``ridealong_terms.rank_auroc`` on ``from scipy.stats import rankdata``). The env file declares them
#: since 2026-10-12 (and ``src/env_files_test.py`` fails on any undeclared import), so this belt is now redundant: ``derive``
#: adds an entry only when the file lacks it. Pinned at the shared env's versions.
UNDECLARED_RUNTIME = ("scipy==1.17.1", "scikit-learn==1.9.0")


def _swap_cuda(e: str, cuda: str) -> str:
    e = re.sub(r"(download\.pytorch\.org/whl/)cu\d{3}", rf"\g<1>{cuda}", e)
    return re.sub(r"\+cu\d{3}\b", f"+{cuda}", e)


def derive(env_yml_text: str, cuda: str = "cu128") -> List[str]:
    if not re.fullmatch(r"cu\d{3}", cuda):
        raise ValueError(f"cuda tag {cuda!r} is not like cu128")
    entries = pip_block(env_yml_text)
    if not any(e.startswith("torch==") for e in entries):
        raise ValueError("no torch== pin in the env file's pip block — is this the right file?")
    out: List[str] = []
    for e in entries:
        if re.match(r"^nvidia-[a-z0-9-]+-cu12==", e):
            continue
        out.append(_swap_cuda(e, cuda))
    have = {re.split(r"[=<>\[]", e, maxsplit=1)[0].lower() for e in out}
    out += [p for p in UNDECLARED_RUNTIME if p.split("==")[0].lower() not in have]
    return out


#: Never pinned from a lock: the editable install of this repo and the installer's own tools.
_LOCK_SKIP = re.compile(r"^(gen3ai|pip|wheel|setuptools-scm|asyncio|py-spy)$", re.I)


def lock_from_freeze(freeze_text: str, cuda: str = "cu128") -> List[str]:
    """EXACT PARITY with a live env: every ``name==version`` of its ``pip list --format=freeze`` (transitive
    deps included — the env file leaves ~15 of them floating, and a fresh resolve drifted click, starlette,
    pydantic & co. on 2026-10-12), torch on ``cuda``, the cu12 nvidia runtime dropped (torch pulls its own)."""
    out = [f"--extra-index-url https://download.pytorch.org/whl/{cuda}"]
    for line in freeze_text.splitlines():
        line = line.strip()
        if "==" not in line or line.startswith("#"):
            continue
        name = line.split("==")[0]
        if _LOCK_SKIP.match(name) or re.match(r"^nvidia-[A-Za-z0-9-]+-cu12$", name):
            continue
        out.append(_swap_cuda(line, cuda))
    if not any(e.startswith("torch==") for e in out):
        raise ValueError("no torch== in the freeze — is this the right interpreter?")
    return out


def main(argv: List[str] | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", default=os.path.join(root, "environment_torch28.yml"))
    p.add_argument("--cuda", default="cu128")
    p.add_argument("--lock-from", default="", metavar="PYTHON",
                   help="pin EVERY package of this interpreter's env instead (exact parity; see lock_from_freeze)")
    a = p.parse_args(argv)
    if a.lock_from:
        import subprocess
        freeze = subprocess.run([a.lock_from, "-m", "pip", "list", "--format=freeze"], check=True,
                                capture_output=True, text=True).stdout
        lines, src = lock_from_freeze(freeze, a.cuda), f"the live env of {a.lock_from}"
    else:
        with open(a.env, encoding="utf-8") as fh:
            lines, src = derive(fh.read(), a.cuda), os.path.basename(a.env)
    sys.stdout.write(f"# DERIVED from {src} by tools/cloud/vast/requirements.py ({a.cuda})\n")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
