"""THE DESKTOP-GPU REFUSAL — a training run does not start while the desktop holds its GPU (T23, P0).

WHY. On 2026-10-05 `gnome-shell` held 811 MiB of the 12 GiB card: about 40 % of the measured D-6
headroom at N = 256 (~2.0 GiB, `measurements/x5_u2_gpu_checks_2026-10-04/`). The memory is simply
gone for the run's whole life, and a desktop frame can push a tight run over the edge. Owner,
2026-10-05: "a basic check to ensure when training runs the GUI is off. Fail loud."

WHAT. For a CUDA run, list EVERY process NVML shows on the card — compute AND graphics clients
(`nvidia-smi --query-compute-apps` lists compute clients only, which is why a `G` client such as
`gnome-shell` hides from it) — and refuse (`FATAL_CONFIG`, so the launcher does not restart into the
same refusal) when one of them is a display server / compositor / display manager
(`DISPLAY_PROCESS_NAMES`, a DECLARED list).

DETECTION. `nvidia-smi -q -x` (the NVML XML report): every `<process_info>` carries `pid`, `type`
(`G` graphics, `C` compute, `C+G`), `process_name` and `used_memory`. No `pynvml` (absent from both env
files) and NO CUDA CONTEXT — reading NVML never creates one, so this is safe to run before the trainer
builds anything.

NEVER SILENT. NVML unreadable (no `nvidia-smi`, a driver error, unparseable XML) is a REFUSAL for a
CUDA run, with the reason stated — an unanswerable check must not read as a clean one.

THE OPT-OUT. `--allow-desktop-gpu` (dev / short runs): the run proceeds and the verdict is RECORDED
in the run's `metadata.json` (`cli_args.allow_desktop_gpu` and `cli_args._desktop_gpu`, which names
the holders that were tolerated). `--debug` resolves to CPU and is exempt on its own; an explicit
`--debug --device cuda` is a CUDA run and is checked like any other.

BOTH SURFACES. The trainer calls :func:`check_for_run` right after the config resolves (before any
directory exists, so a refusal leaves nothing behind); the launcher's `--dry-run` calls the same
function and prints the same verdict.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Process BASENAMES that are the desktop, compared case-insensitively. A display server
#: (`Xorg`, `Xwayland`), a compositor / shell (`gnome-shell`, `kwin_*`, `mutter`, `sway`, `weston`,
#: `plasmashell`, ...) or a display manager (`gdm*`, `sddm*`, `lightdm*`). Declared, not inferred: a
#: name not here is not refused, so a new desktop environment is ADDED here, with a test row.
DISPLAY_PROCESS_NAMES: Tuple[str, ...] = (
    "gnome-shell", "Xorg", "Xwayland", "X", "gdm", "gdm3", "gdm-x-session", "gdm-wayland-session",
    "gdm-session-worker", "kwin_x11", "kwin_wayland", "kwin_wayland_wrapper", "plasmashell",
    "mutter", "muffin", "cinnamon", "budgie-wm", "xfwm4", "sway", "weston", "sddm", "sddm-greeter",
    "lightdm", "lightdm-gtk-greeter",
)
_DISPLAY_LOWER = frozenset(n.lower() for n in DISPLAY_PROCESS_NAMES)

STOP_CMD = "sudo systemctl stop gdm.service"
START_CMD = "sudo systemctl start gdm.service"
SUDOERS_FILE = "/etc/sudoers.d/gen3ai-gpu"
OPT_OUT_FLAG = "--allow-desktop-gpu"

#: Verdict statuses.
EXEMPT_CPU = "exempt-cpu"            # the run resolves to CPU: nothing to check
PASS = "pass"                        # a CUDA run, no display process on any card
OPTED_OUT = "opted-out"              # a CUDA run that WOULD be refused, proceeding under the flag
REFUSED_DESKTOP = "refused-desktop"
REFUSED_NVML = "refused-nvml"


class NvmlUnavailable(RuntimeError):
    """NVML could not be read (no `nvidia-smi`, a driver error, an unparseable report)."""


@dataclass(frozen=True)
class GpuProcess:
    gpu: int
    gpu_name: str
    pid: int
    kind: str                 # NVML's type: "G", "C", "C+G"
    name: str                 # the reported process_name (often a full path)
    vram_mib: Optional[int]   # None when NVML reports "N/A"

    @property
    def basename(self) -> str:
        return os.path.basename(self.name.strip().split(" ")[0]) if self.name.strip() else ""

    @property
    def is_display(self) -> bool:
        return is_display_process(self.name)

    def describe(self) -> str:
        vram = f"{self.vram_mib} MiB" if self.vram_mib is not None else "VRAM n/a"
        return (f"{self.basename or self.name} (pid {self.pid}, type {self.kind}, {vram} "
                f"on GPU {self.gpu} {self.gpu_name})")


def is_display_process(name: str) -> bool:
    """True when ``name`` (a path or a bare name) is on the declared display-process list."""
    base = os.path.basename(str(name).strip().split(" ")[0]).lower()
    return base in _DISPLAY_LOWER


# --- reading NVML --------------------------------------------------------------------------------

_MIB = re.compile(r"^\s*(\d+)\s*MiB\s*$")


def parse_nvidia_smi_xml(xml_text: str) -> List[GpuProcess]:
    """Every process (compute AND graphics) on every GPU of an `nvidia-smi -q -x` report."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise NvmlUnavailable(f"nvidia-smi's XML report did not parse: {e}") from e
    if root.tag != "nvidia_smi_log":
        raise NvmlUnavailable(f"nvidia-smi's report has an unexpected root <{root.tag}>")
    out: List[GpuProcess] = []
    for idx, gpu in enumerate(root.findall("gpu")):
        gname = (gpu.findtext("product_name") or "?").strip()
        for pi in gpu.findall("./processes/process_info"):
            try:
                pid = int((pi.findtext("pid") or "").strip())
            except ValueError:
                continue
            m = _MIB.match(pi.findtext("used_memory") or "")
            out.append(GpuProcess(
                gpu=idx, gpu_name=gname, pid=pid, kind=(pi.findtext("type") or "?").strip(),
                name=(pi.findtext("process_name") or "").strip(),
                vram_mib=int(m.group(1)) if m else None))
    return out


def list_gpu_processes() -> List[GpuProcess]:
    """The NVML process listing, or `NvmlUnavailable` naming why it cannot be had. The seam the test
    suite pins (the root conftest replaces THIS, so no test reads the box's real display state);
    the reader itself is :func:`read_nvidia_smi`."""
    return read_nvidia_smi()


def read_nvidia_smi(timeout_s: float = 20.0) -> List[GpuProcess]:
    """`nvidia-smi -q -x`, parsed."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise NvmlUnavailable("`nvidia-smi` is not on PATH (no NVIDIA driver?)")
    try:
        res = subprocess.run([exe, "-q", "-x"], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise NvmlUnavailable(f"`nvidia-smi -q -x` could not run: {e}") from e
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise NvmlUnavailable(f"`nvidia-smi -q -x` exited {res.returncode}: {tail[0]}")
    return parse_nvidia_smi_xml(res.stdout)


def nvidia_gpu_visible() -> bool:
    """Whether an NVIDIA driver answers at all (for resolving `--device auto` without torch)."""
    return shutil.which("nvidia-smi") is not None


# --- the verdict ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class DesktopGpuVerdict:
    status: str
    device: str                              # the RESOLVED device ("cpu", "cuda", "cuda:1", ...)
    holders: Tuple[GpuProcess, ...] = ()     # the display processes found
    nvml_error: Optional[str] = None
    allow_flag: bool = False

    @property
    def refused(self) -> bool:
        return self.status in (REFUSED_DESKTOP, REFUSED_NVML)

    def lines(self) -> List[str]:
        """The human report: one line for a pass / exemption, the full refusal for a refusal."""
        if self.status == EXEMPT_CPU:
            return [f"desktop GPU : exempt — the run resolves to {self.device} (no CUDA)"]
        if self.status == PASS:
            return ["desktop GPU : ✓ no display process (gnome-shell / Xorg / Xwayland / …) holds the GPU"]
        if self.status == OPTED_OUT:
            why = (f"NVML unreadable ({self.nvml_error})" if self.nvml_error
                   else "; ".join(h.describe() for h in self.holders))
            return [f"desktop GPU : ⚠️  tolerated under {OPT_OUT_FLAG} — {why}. Recorded in "
                    f"metadata.json (cli_args._desktop_gpu); this run has that much LESS VRAM."]
        if self.status == REFUSED_NVML:
            return [
                "desktop GPU : ✗ REFUSED — cannot tell whether the desktop holds the GPU: "
                f"NVML is unavailable ({self.nvml_error}).",
                "              A check that cannot answer does not read as a pass. Fix the driver "
                f"(`nvidia-smi` must work), or pass {OPT_OUT_FLAG} (dev / short runs only).",
            ]
        holders = "; ".join(h.describe() for h in self.holders)
        total = sum(h.vram_mib or 0 for h in self.holders)
        return [
            f"desktop GPU : ✗ REFUSED — the desktop holds the training GPU: {holders}"
            + (f" (~{total} MiB in all)" if total else "") + ".",
            "              That VRAM is gone for the run's whole life (gnome-shell held 811 MiB of "
            "12 GiB, ~40 % of the N=256 headroom).",
            f"              FIX:     {STOP_CMD}     (passwordless via {SUDOERS_FILE} once the owner "
            "has installed it)",
            f"              RESTORE: {START_CMD}   (or reboot) when the run is done.",
            f"              Dev / short runs only: {OPT_OUT_FLAG} (recorded in metadata.json).",
        ]

    def to_record(self) -> Dict[str, Any]:
        """The JSON-safe block stamped into `metadata.json` (`cli_args._desktop_gpu`)."""
        return {
            "status": self.status, "device": self.device, "allow_desktop_gpu": self.allow_flag,
            "holders": [{"name": h.basename or h.name, "pid": h.pid, "type": h.kind,
                         "vram_mib": h.vram_mib, "gpu": h.gpu} for h in self.holders],
            "nvml_error": self.nvml_error,
        }


def resolve_device(device: str, debug: bool, cuda_available: Callable[[], bool]) -> str:
    """The device the run will use, by the trainer's own rules: `--debug` + `auto` is CPU; `auto`
    otherwise is CUDA iff ``cuda_available()``; an explicit value stands."""
    d = str(device or "auto").strip().lower()
    if d != "auto":
        return d
    if debug:
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def check_for_run(
    device: str,
    *,
    debug: bool,
    allow: bool,
    cuda_available: Optional[Callable[[], bool]] = None,
    lister: Optional[Callable[[], Sequence[GpuProcess]]] = None,
) -> DesktopGpuVerdict:
    """The one decision both surfaces read. Never raises for an NVML failure — it is a verdict."""
    resolved = resolve_device(device, debug, cuda_available if cuda_available is not None
                              else nvidia_gpu_visible)
    if not resolved.startswith("cuda"):
        return DesktopGpuVerdict(EXEMPT_CPU, resolved, allow_flag=bool(allow))
    lst = lister if lister is not None else list_gpu_processes
    try:
        procs = list(lst())
    except NvmlUnavailable as e:
        return DesktopGpuVerdict(OPTED_OUT if allow else REFUSED_NVML, resolved,
                                 nvml_error=str(e), allow_flag=bool(allow))
    holders = tuple(p for p in procs if p.is_display)
    if not holders:
        return DesktopGpuVerdict(PASS, resolved, allow_flag=bool(allow))
    return DesktopGpuVerdict(OPTED_OUT if allow else REFUSED_DESKTOP, resolved, holders=holders,
                             allow_flag=bool(allow))
