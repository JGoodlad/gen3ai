"""T23 — a training run refuses to start while the desktop holds the GPU (`utils.desktop_gpu`).

Every NVML listing here is STUBBED (or a fake `nvidia-smi` on PATH), so the verdicts do not depend on
whether the box running the tests has a desktop up. Each test fails on revert of the thing it names:
the detector, the refusal text, the opt-out, the CPU exemption, the NVML-unavailable refusal, the
trainer's call and the record it stamps into `metadata.json`."""
from __future__ import annotations

import asyncio
import json
import stat

import pytest

from main.exit_codes import TrainExitCode
from utils import desktop_gpu as D

GNOME = D.GpuProcess(0, "NVIDIA GeForce RTX 3080 Ti", 7568, "G", "/usr/bin/gnome-shell", 811)
TRAINER = D.GpuProcess(0, "NVIDIA GeForce RTX 3080 Ti", 3511132, "C",
                       "/opt/envs/gen3ai_torch28/bin/python3", 844)

_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:01:00.0">
    <product_name>NVIDIA GeForce RTX 3080 Ti</product_name>
    <processes>
      <process_info><gpu_instance_id>N/A</gpu_instance_id><pid>7568</pid><type>G</type>
        <process_name>/usr/bin/gnome-shell</process_name><used_memory>811 MiB</used_memory></process_info>
      <process_info><pid>3511132</pid><type>C</type>
        <process_name>/opt/x/bin/python3</process_name><used_memory>844 MiB</used_memory></process_info>
      <process_info><pid>9</pid><type>C+G</type><process_name>Xwayland</process_name>
        <used_memory>N/A</used_memory></process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""


def _boom_lister():
    raise AssertionError("NVML must not even be read for this run")


def _unavailable():
    raise D.NvmlUnavailable("`nvidia-smi` is not on PATH (no NVIDIA driver?)")


# ---- the detector ----------------------------------------------------------------------------

def test_the_listing_carries_graphics_AND_compute_clients():
    """`--query-compute-apps` shows only the `C` rows — the `G` row IS the desktop."""
    procs = D.parse_nvidia_smi_xml(_XML)
    assert [(p.pid, p.kind, p.vram_mib) for p in procs] == [
        (7568, "G", 811), (3511132, "C", 844), (9, "C+G", None)]
    assert [p.is_display for p in procs] == [True, False, True]


def test_an_unparseable_report_is_NvmlUnavailable_not_an_empty_listing():
    with pytest.raises(D.NvmlUnavailable):
        D.parse_nvidia_smi_xml("<<not xml")
    with pytest.raises(D.NvmlUnavailable):
        D.parse_nvidia_smi_xml("<something_else/>")


def _fake_smi(tmp_path, monkeypatch, body):
    exe = tmp_path / "nvidia-smi"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))


def test_the_real_reader_runs_nvidia_smi_and_parses_it(tmp_path, monkeypatch):
    (tmp_path / "report.xml").write_text(_XML)
    _fake_smi(tmp_path, monkeypatch, f"/bin/cat {tmp_path}/report.xml\n")
    assert [p.pid for p in D.read_nvidia_smi()] == [7568, 3511132, 9]


def test_no_nvidia_smi_on_PATH_is_NvmlUnavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))          # an empty directory
    with pytest.raises(D.NvmlUnavailable, match="not on PATH"):
        D.read_nvidia_smi()


def test_a_failing_nvidia_smi_is_NvmlUnavailable_naming_its_error(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, "echo 'Failed to initialize NVML: Driver/library version mismatch' >&2\nexit 18\n")
    with pytest.raises(D.NvmlUnavailable, match="exited 18.*version mismatch"):
        D.read_nvidia_smi()


@pytest.mark.parametrize("name", list(D.DISPLAY_PROCESS_NAMES) + ["/usr/bin/gnome-shell", "/usr/lib/xorg/Xorg",
                                                                   "GNOME-SHELL"])
def test_every_declared_display_process_is_recognised(name):
    assert D.is_display_process(name)


@pytest.mark.parametrize("name", ["/opt/x/bin/python3", "python3", "nvidia-smi", "firefox", "chrome", ""])
def test_ordinary_gpu_clients_are_not_the_desktop(name):
    assert not D.is_display_process(name)


# ---- the verdicts ----------------------------------------------------------------------------

def test_gnome_shell_present_is_REFUSED_with_the_process_vram_and_the_fix():
    v = D.check_for_run("cuda", debug=False, allow=False, lister=lambda: [GNOME, TRAINER])
    assert v.status == D.REFUSED_DESKTOP and v.refused
    text = "\n".join(v.lines())
    for needle in ("gnome-shell", "7568", "811 MiB", "sudo systemctl stop gdm.service",
                   "sudo systemctl start gdm.service", "/etc/sudoers.d/gen3ai-gpu",
                   "--allow-desktop-gpu"):
        assert needle in text, (needle, text)
    assert "3511132" not in text, "the refusal names the desktop, not every client on the card"


def test_an_empty_listing_passes():
    v = D.check_for_run("cuda", debug=False, allow=False, lister=lambda: [])
    assert v.status == D.PASS and not v.refused


def test_compute_clients_alone_pass():
    assert D.check_for_run("cuda:0", debug=False, allow=False, lister=lambda: [TRAINER]).status == D.PASS


def test_the_opt_out_passes_and_is_recorded():
    v = D.check_for_run("cuda", debug=False, allow=True, lister=lambda: [GNOME])
    assert v.status == D.OPTED_OUT and not v.refused
    rec = json.loads(json.dumps(v.to_record()))          # JSON-safe: it lands in metadata.json
    assert rec["allow_desktop_gpu"] is True and rec["status"] == "opted-out"
    assert rec["holders"] == [{"name": "gnome-shell", "pid": 7568, "type": "G", "vram_mib": 811, "gpu": 0}]
    assert "tolerated" in "\n".join(v.lines())


def test_debug_on_auto_is_CPU_and_exempt_without_reading_NVML():
    v = D.check_for_run("auto", debug=True, allow=False, cuda_available=lambda: True, lister=_boom_lister)
    assert v.status == D.EXEMPT_CPU and v.device == "cpu"


def test_an_explicit_cpu_device_is_exempt():
    assert D.check_for_run("cpu", debug=False, allow=False, lister=_boom_lister).status == D.EXEMPT_CPU


def test_auto_with_no_cuda_is_a_cpu_run_and_exempt():
    v = D.check_for_run("auto", debug=False, allow=False, cuda_available=lambda: False, lister=_boom_lister)
    assert v.status == D.EXEMPT_CPU


def test_auto_with_cuda_is_a_CUDA_run_and_is_checked():
    v = D.check_for_run("auto", debug=False, allow=False, cuda_available=lambda: True, lister=lambda: [GNOME])
    assert v.status == D.REFUSED_DESKTOP and v.device == "cuda"


def test_an_explicit_debug_cuda_run_is_checked_like_any_other():
    v = D.check_for_run("cuda", debug=True, allow=False, lister=lambda: [GNOME])
    assert v.status == D.REFUSED_DESKTOP


def test_NVML_unavailable_is_REFUSED_stating_it_never_a_silent_pass():
    v = D.check_for_run("cuda", debug=False, allow=False, lister=_unavailable)
    assert v.status == D.REFUSED_NVML and v.refused
    text = "\n".join(v.lines())
    assert "NVML is unavailable" in text and "not on PATH" in text and "--allow-desktop-gpu" in text


def test_NVML_unavailable_under_the_opt_out_proceeds_and_says_so():
    v = D.check_for_run("cuda", debug=False, allow=True, lister=_unavailable)
    assert v.status == D.OPTED_OUT and v.nvml_error and "NVML unreadable" in "\n".join(v.lines())


# ---- the trainer calls it ---------------------------------------------------------------------

class _Reached(Exception):
    """Raised by the stub that stands where the trainer's next phase begins."""


@pytest.fixture
def trainer(monkeypatch):
    """`train_rl_agent.main()` stopped right after the check: a stub stands at the next phase."""
    from main import train_rl_agent

    def _next_phase(*a, **k):
        raise _Reached

    monkeypatch.setattr(train_rl_agent, "build_matchup_and_opponents", _next_phase)

    def run(argv, lister):
        monkeypatch.setattr(D, "list_gpu_processes", lister)
        monkeypatch.setattr("sys.argv", ["train_rl_agent.py", "--steps", "10", "--arch", "production", *argv])
        return asyncio.run(train_rl_agent.main())
    return run


def test_the_TRAINER_exits_FATAL_CONFIG_while_gnome_shell_holds_the_gpu(trainer, capsys):
    with pytest.raises(SystemExit) as ei:
        trainer(["--device", "cuda"], lambda: [GNOME])
    assert ei.value.code == int(TrainExitCode.FATAL_CONFIG) == 3
    err = capsys.readouterr().err
    assert "gnome-shell" in err and "811 MiB" in err and "sudo systemctl stop gdm.service" in err


def test_the_TRAINER_proceeds_on_an_empty_listing(trainer):
    with pytest.raises(_Reached):
        trainer(["--device", "cuda"], lambda: [])


def test_the_TRAINER_proceeds_under_the_flag_and_says_so(trainer, capsys):
    with pytest.raises(_Reached):
        trainer(["--device", "cuda", "--allow-desktop-gpu"], lambda: [GNOME])
    assert "tolerated under --allow-desktop-gpu" in capsys.readouterr().out


def test_the_TRAINER_debug_smoke_is_exempt(trainer):
    with pytest.raises(_Reached):
        trainer(["--debug"], _boom_lister)


def test_the_TRAINER_refuses_when_NVML_is_unavailable(trainer, capsys):
    with pytest.raises(SystemExit) as ei:
        trainer(["--device", "cuda"], _unavailable)
    assert ei.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "NVML is unavailable" in capsys.readouterr().err


def test_the_trainer_stamps_the_verdict_into_metadata_cli_args():
    """`cli_args` (the full namespace) becomes `metadata.json:cli_args`; the verdict rides beside it."""
    import ast
    import main.train_rl_agent as T
    tree = ast.parse(open(T.__file__, encoding="utf-8").read())
    stamped = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "cli_args"
        and isinstance(t.slice, ast.Constant) and t.slice.value == "_desktop_gpu" for t in n.targets)]
    assert len(stamped) == 1


def test_the_flag_is_registered_and_defaults_off():
    from main.train.parser import build_parser
    ns = build_parser().parse_args([])
    assert ns.allow_desktop_gpu is False
    assert build_parser().parse_args(["--allow-desktop-gpu"]).allow_desktop_gpu is True
