"""The launcher's CUDA-VISIBILITY PREFLIGHT: a ``cuda`` launch whose child cannot see a GPU is refused.

THE INCIDENT (2026-10-20 11:50). An X35 launch passed ``--dry-run ... --device cuda``, then its child died at startup
with "No CUDA GPUs are available": the launch env carried ``CUDA_VISIBLE_DEVICES=""``, inherited from a CPU-only
end-of-run chain. The child's env is the launcher's own (``child._build_child_env``), so the launcher can read exactly
what the child will see - from the environment alone. It never initialises CUDA and never runs ``nvidia-smi``: a
launcher that touched CUDA would hold a context for the run's life.

Hiding rule (the CUDA runtime's): ``CUDA_VISIBLE_DEVICES`` is a comma list of device indices or ``GPU-`` / ``MIG-``
UUIDs, read left to right and cut at the FIRST invalid token. Unset = all visible. The empty string, ``-1``,
``NoDevFiles``, or any list whose first token is invalid therefore hides every device.

Only an EXPLICIT ``cuda`` / ``cuda:N`` device is refused. ``--device auto`` resolves to CPU in the trainer when no GPU is
visible (a silent CPU run, which is the caller's call), so it is reported as a note, never a refusal. NOT checked: an
index beyond the machine's device count, and ``cuda:N`` with N past the visible list (no device count without NVML).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Mapping, Optional

ENV_VAR = "CUDA_VISIBLE_DEVICES"
_INDEX = re.compile(r"^\d+$")


def visible_tokens(value: str) -> List[str]:
    """The devices ``CUDA_VISIBLE_DEVICES=value`` leaves visible, per the runtime's left-to-right, stop-at-invalid rule."""
    out: List[str] = []
    for tok in value.split(","):
        tok = tok.strip()
        if _INDEX.match(tok) or tok.startswith(("GPU-", "MIG-")):
            out.append(tok)
        else:
            break
    return out


@dataclass(frozen=True)
class CudaVisibilityVerdict:
    device: str                       # the --device the launch resolves to ("auto" stays "auto")
    value: Optional[str]              # CUDA_VISIBLE_DEVICES as the child will see it (None = unset)
    hidden: bool                      # the env leaves no device visible
    refused: bool

    @property
    def reason(self) -> str:
        return (f"CUDA_VISIBLE_DEVICES={self.value!r} is set in the environment this launch hands the child, and it "
                f"hides every GPU, but the launch is --device {self.device}: the child would die at startup with "
                f"\"No CUDA GPUs are available\". Fix: `unset CUDA_VISIBLE_DEVICES` (or set the device index to use) "
                f"before launching, or launch with --device cpu")

    def lines(self) -> List[str]:
        if self.refused:
            return [f"cuda devices : ✗ REFUSED (FATAL_CONFIG): {self.reason}"]
        if self.hidden:
            return [f"cuda devices : ⚠ CUDA_VISIBLE_DEVICES={self.value!r} hides every GPU; --device {self.device} "
                    "will run on CPU"]
        if self.value is None:
            return ["cuda devices : ✓ CUDA_VISIBLE_DEVICES unset"]
        return [f"cuda devices : ✓ CUDA_VISIBLE_DEVICES={self.value!r} leaves {len(visible_tokens(self.value))} visible"]


def verdict_for_launch(device: Optional[str], child_env: Mapping[str, str]) -> CudaVisibilityVerdict:
    """The verdict for a launch with ``--device device`` whose child gets ``child_env`` (``_build_child_env()``)."""
    dev = str(device or "auto").strip().lower()
    value = child_env.get(ENV_VAR)
    hidden = value is not None and not visible_tokens(value)
    return CudaVisibilityVerdict(dev, value, hidden, refused=hidden and dev.startswith("cuda"))
