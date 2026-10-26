"""The chain file (``arm_chain_v1``): what runs, in what order, at which pin, and what happens at each end.

A chain is an ordered list of ITEMS: an ``arm`` (one arm x seed, launched as a fork through the backend) or a
``window`` (GPU work inserted between two arms). Every value an ad hoc script used to take from a default, an
environment variable or a run-name glob is a REQUIRED field here, validated at load:

* each arm carries its OWN full 40-hex ``pin`` — there is no chain-wide default (the hand-off class bug: the next
  arm launched at the pin an environment variable or a script default happened to hold);
* each arm names its ``run`` — the launch's describe must agree, and every later match is on that exact token;
* the thin policy is a per-arm FIELD (``planner`` or ``none`` + a reason), never a ``case`` on the run's name (the
  AM-R0 / AM-X47 / AM-X52 keep-every-checkpoint arms were each a new glob, and a family the glob missed is thinned);
* ``go_by`` records whose GO the arm runs under (the orchestrator's; a design block is not a launch command).

Loading never touches the GPU, ``models/`` or git; ``check_pins`` (git) is a separate call.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SCHEMA = "arm_chain_v1"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA_PREFIX = re.compile(r"[0-9a-f]{7,40}")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*")
#: Root rule 3: the literal trainer script name never appears in a backgrounded argv (it trips the live-run watchers).
_TRAINER_SCRIPT = "train_rl_agent" + ".py"
THIN_POLICIES = ("planner", "none")
ON_FAIL = ("hold", "continue")


class SpecError(ValueError):
    """The chain file is malformed; the message names the field."""


@dataclass(frozen=True)
class Banner:
    """A startup banner the child log must print (``expect``) and lines that VOID the arm (``void``)."""

    expect: str
    void: Tuple[str, ...]
    within_min: float


@dataclass(frozen=True)
class Throughput:
    """The first-hour gate: a marginal steps/s (checkpoint mtimes) of at least ``min_steps_per_s`` by ``within_min``."""

    min_steps_per_s: float
    within_min: float


@dataclass(frozen=True)
class ArmItem:
    arm: str
    seed: int
    pin: str
    run: str
    go_by: str
    thin: str
    thin_reason: str
    flags: Tuple[str, ...]
    verify: Tuple[Tuple[str, ...], ...]
    banner: Optional[Banner]
    throughput: Optional[Throughput]
    disk_need_gib: float
    kind: str = "arm"

    @property
    def key(self) -> str:
        return f"arm:{self.run}"


@dataclass(frozen=True)
class WindowStep:
    name: str
    argv: Tuple[str, ...]
    gpu: bool
    mem_gb: float
    timeout_s: int
    cwd: str  # "worktree" | "repo"


@dataclass(frozen=True)
class WindowItem:
    name: str
    worktree: str
    require_commit: str
    steps: Tuple[WindowStep, ...]
    budget_min: float
    on_fail: str
    preempt_marker: str
    kind: str = "window"

    @property
    def key(self) -> str:
        return f"window:{self.name}"


Item = Union[ArmItem, WindowItem]


@dataclass(frozen=True)
class Backend:
    kind: str
    launch_script: str
    stage: str
    lease_owner: str
    lease_token_file: str
    max_load: float


@dataclass(frozen=True)
class ChainSpec:
    name: str
    job_dir: str
    status_file: str
    repo: str
    models_dir: str
    python: str
    backend: Backend
    records_stage: str
    submodules: Tuple[str, ...]
    agent_cores: str
    trainer_cores: str
    disk_headroom_gib: float
    items: Tuple[Item, ...]
    source: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state_path(self) -> str:
        return os.path.join(self.job_dir, "arm_chain", f"{self.name}.state.json")

    @property
    def work_dir(self) -> str:
        """Where the chain's own transcripts, scratch stages and records go (inside the job dir, never main)."""
        return os.path.join(self.job_dir, "arm_chain", self.name)

    def arms(self) -> List[ArmItem]:
        return [i for i in self.items if isinstance(i, ArmItem)]


def _req(d: Mapping[str, Any], k: str, where: str) -> Any:
    if k not in d or d[k] in (None, ""):
        raise SpecError(f"{where}: '{k}' is required")
    return d[k]


def _str_list(v: Any, where: str) -> Tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise SpecError(f"{where}: must be a list of strings")
    return tuple(v)


def _no_trainer_name(argv: Sequence[str], where: str) -> None:
    if any(_TRAINER_SCRIPT in a for a in argv):
        raise SpecError(f"{where}: names the trainer script literally (root rule 3: it trips the live-run watchers)")


def _arm(d: Mapping[str, Any], where: str) -> ArmItem:
    pin = str(_req(d, "pin", where))
    if not _SHA40.fullmatch(pin):
        raise SpecError(f"{where}: pin '{pin}' is not a full 40-hex commit (no default pin, no abbreviation)")
    run = str(_req(d, "run", where))
    if not _RUN.fullmatch(run):
        raise SpecError(f"{where}: run '{run}' is not a run name")
    thin = str(_req(d, "thin", where))
    if thin not in THIN_POLICIES:
        raise SpecError(f"{where}: thin must be one of {THIN_POLICIES}, not '{thin}'")
    reason = str(d.get("thin_reason") or "")
    if thin == "none" and not reason:
        raise SpecError(f"{where}: thin 'none' needs a thin_reason (which read needs every checkpoint)")
    flags = _str_list(d.get("flags", []), f"{where}.flags")
    if "GO" in flags:
        raise SpecError(f"{where}.flags: 'GO' is the chain's to add, never a flag")
    verify = tuple(_str_list(v, f"{where}.verify[{i}]") for i, v in enumerate(d.get("verify", [])))
    for i, v in enumerate(verify):
        _no_trainer_name(v, f"{where}.verify[{i}]")
    banner = None
    if d.get("banner"):
        b = d["banner"]
        banner = Banner(expect=str(_req(b, "expect", f"{where}.banner")),
                        void=_str_list(b.get("void", []), f"{where}.banner.void"),
                        within_min=float(b.get("within_min", 20)))
    tp = None
    if d.get("throughput"):
        t = d["throughput"]
        tp = Throughput(min_steps_per_s=float(_req(t, "min_steps_per_s", f"{where}.throughput")),
                        within_min=float(t.get("within_min", 60)))
    seed = d.get("seed")
    if not isinstance(seed, int):
        raise SpecError(f"{where}: seed must be an integer")
    return ArmItem(arm=str(_req(d, "arm", where)), seed=seed, pin=pin, run=run, go_by=str(_req(d, "go_by", where)),
                   thin=thin, thin_reason=reason, flags=flags, verify=verify, banner=banner, throughput=tp,
                   disk_need_gib=float(_req(d, "disk_need_gib", where)))


def _window(d: Mapping[str, Any], where: str) -> WindowItem:
    name = str(_req(d, "name", where))
    if not _SLUG.fullmatch(name):
        raise SpecError(f"{where}: name '{name}' must be a lowercase slug")
    commit = str(_req(d, "require_commit", where))
    if not _SHA_PREFIX.fullmatch(commit):
        raise SpecError(f"{where}: require_commit '{commit}' is not a commit sha")
    on_fail = str(_req(d, "on_fail", where))
    if on_fail not in ON_FAIL:
        raise SpecError(f"{where}: on_fail must be one of {ON_FAIL} (say whether the chain continues past a failure)")
    steps = []
    raw = d.get("steps")
    if not isinstance(raw, list) or not raw:
        raise SpecError(f"{where}: steps must be a non-empty list")
    for i, s in enumerate(raw):
        w = f"{where}.steps[{i}]"
        argv = _str_list(_req(s, "argv", w), f"{w}.argv")
        _no_trainer_name(argv, w)
        cwd = str(s.get("cwd", "worktree"))
        if cwd not in ("worktree", "repo"):
            raise SpecError(f"{w}: cwd must be 'worktree' or 'repo'")
        steps.append(WindowStep(name=str(_req(s, "name", w)), argv=argv, gpu=bool(s.get("gpu", True)),
                                mem_gb=float(_req(s, "mem_gb", w)), timeout_s=int(_req(s, "timeout_s", w)), cwd=cwd))
    return WindowItem(name=name, worktree=str(_req(d, "worktree", where)), require_commit=commit, steps=tuple(steps),
                      budget_min=float(_req(d, "budget_min", where)), on_fail=on_fail,
                      preempt_marker=str(_req(d, "preempt_marker", where)))


def parse(d: Mapping[str, Any], source: str = "") -> ChainSpec:
    if d.get("schema") != SCHEMA:
        raise SpecError(f"schema must be '{SCHEMA}', not {d.get('schema')!r}")
    name = str(_req(d, "name", "chain"))
    if not _SLUG.fullmatch(name):
        raise SpecError(f"chain: name '{name}' must be a lowercase slug")
    job_dir = str(_req(d, "job_dir", "chain"))
    if not os.path.isabs(job_dir):
        raise SpecError("chain: job_dir must be absolute")
    repo = str(_req(d, "repo", "chain"))
    if not os.path.isabs(repo):
        raise SpecError("chain: repo must be absolute (the MAIN checkout)")
    b = _req(d, "backend", "chain")
    backend = Backend(kind=str(_req(b, "kind", "backend")), launch_script=str(_req(b, "launch_script", "backend")),
                      stage=str(_req(b, "stage", "backend")), lease_owner=str(_req(b, "lease_owner", "backend")),
                      lease_token_file=str(_req(b, "lease_token_file", "backend")),
                      max_load=float(b.get("max_load", 64)))
    if backend.kind != "x35":
        raise SpecError(f"backend: kind '{backend.kind}' has no implementation (x35 only)")
    rec = _req(d, "records", "chain")
    items: List[Item] = []
    raw = d.get("items")
    if not isinstance(raw, list) or not raw:
        raise SpecError("chain: items must be a non-empty list")
    for i, it in enumerate(raw):
        kind = it.get("kind")
        if kind == "arm":
            items.append(_arm(it, f"items[{i}]"))
        elif kind == "window":
            items.append(_window(it, f"items[{i}]"))
        else:
            raise SpecError(f"items[{i}]: kind must be 'arm' or 'window', not {kind!r}")
    keys = [i.key for i in items]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SpecError(f"chain: duplicate items {dup} (a run or window appears once per chain)")
    status = str(d.get("status_file", "status.txt"))
    status = status if os.path.isabs(status) else os.path.join(job_dir, status)
    if os.path.normpath(status) != os.path.join(os.path.normpath(job_dir), "status.txt"):
        raise SpecError(f"chain: status_file {status} is not <job_dir>/status.txt, where the x35 kit writes its START / "
                        "END lines (no trigger would ever fire)")
    cpu = d.get("cpu", {})
    return ChainSpec(
        name=name, job_dir=job_dir,
        status_file=status,
        repo=repo, models_dir=str(d.get("models_dir") or os.path.join(repo, "models")),
        python=str(_req(d, "python", "chain")), backend=backend,
        records_stage=str(_req(rec, "stage", "records")),
        submodules=_str_list(rec.get("submodules", []), "records.submodules"),
        agent_cores=str(cpu.get("agent_cores", "5-7,13-15")), trainer_cores=str(cpu.get("trainer_cores", "0-4,8-12")),
        disk_headroom_gib=float(d.get("disk_headroom_gib", 30)), items=tuple(items), source=source,
        extra={k: v for k, v in d.items() if k.startswith("x_")})


def load(path: str) -> ChainSpec:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise SpecError(f"{path}: {e}") from e
    return parse(d, source=os.path.abspath(path))


def check_pins(spec: ChainSpec) -> List[str]:
    """Every arm's pin and every window's required commit names a commit in the main checkout; the problems found."""
    bad = []
    for it in spec.items:
        sha = it.pin if isinstance(it, ArmItem) else it.require_commit
        p = subprocess.run(["git", "-C", spec.repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            bad.append(f"{it.key}: {sha} is not a commit in {spec.repo}")
    return bad


def describe_item(it: Item) -> str:
    if isinstance(it, ArmItem):
        thin = it.thin if it.thin == "planner" else f"none ({it.thin_reason})"
        return (f"ARM {it.arm} s{it.seed} -> {it.run} at pin {it.pin[:12]} (GO by {it.go_by}); thin {thin}; "
                f"flags {' '.join(it.flags) or '-'}; verify x{len(it.verify)}; "
                f"banner {'yes' if it.banner else 'no'}; throughput {it.throughput.min_steps_per_s if it.throughput else '-'}")
    return (f"WINDOW {it.name}: {len(it.steps)} step(s) in {it.worktree} (>= {it.require_commit[:12]}), budget "
            f"{it.budget_min:g} min, on failure {it.on_fail.upper()}, preempt marker {it.preempt_marker}")


def as_dict(it: Item) -> Dict[str, Any]:
    """A JSON-normalised echo of an item (the state file's snapshot; tuples become lists, as after a reload)."""
    from dataclasses import asdict
    return dict(json.loads(json.dumps(asdict(it))))
