"""WHERE the external anchor opponents live — the one module that knows a box-specific path.

Metamon and Foul Play are **third-party checkouts outside this repo**, each with its own
interpreter (Metamon runs UPSTREAM poke-env 0.8.3.3, we vendor a fork; Foul Play runs its own
conda env), so `main.anchors` drives both as SUBPROCESSES and never imports either. That makes
"where are they" a real question with a box-specific answer — and `src/utils/paths_test.py`
forbids a ``/home/...`` literal as a value anywhere under ``src/agents``, ``src/main``,
``src/utils``, for exactly the reason that bit four tests before it: a path literal is correct on
one machine and an invisible permanent skip on every other.

So the answer lives in **`designs/ops/anchors.json`**, found through :func:`utils.paths.repo_path`,
with a per-key env override. Three ways to point the tool somewhere else, in precedence order:

==========================================  ========================================================
1. an env var per key                       ``GEN3AI_METAMON_DIR``, ``GEN3AI_METAMON_PYTHON``,
                                            ``GEN3AI_METAMON_CACHE_DIR``, ``GEN3AI_FOULPLAY_DIR``,
                                            ``GEN3AI_FOULPLAY_PYTHON``, ``GEN3AI_SHOWDOWN_NODE``
2. a whole replacement config file           ``$GEN3AI_ANCHORS_CONFIG=/path/to/anchors.json``
3. the committed default                     ``designs/ops/anchors.json``
==========================================  ========================================================

**A missing checkout is a NAMED refusal, never a skip and never a zero.** :meth:`Opponent.require`
raises :class:`AnchorConfigError` naming the key, the value it resolved to, and the env var that
overrides it — because the failure this tool exists to prevent is a series that reports nothing and
looks like a result.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from main.anchors.peer_scripts import weight_pins as wp
from utils.paths import repo_path

#: Points at a replacement for the whole file. Set-but-missing RAISES rather than falling back to
#: the committed default: an override that silently resolves elsewhere is how a measurement ends up
#: describing a checkout nobody meant to use.
CONFIG_PATH_ENV_VAR = "GEN3AI_ANCHORS_CONFIG"

#: The repo-relative default, committed so a fresh worktree needs no setup to READ the plan.
DEFAULT_CONFIG_RELPATH = ("designs", "ops", "anchors.json")

#: ``(dotted key in the JSON, env var that overrides it)``. Printed by ``--show-config``, so the
#: mapping is discoverable from the CLI rather than only from this docstring.
ENV_OVERRIDES: Dict[str, str] = {
    "showdown.node": "GEN3AI_SHOWDOWN_NODE",
    "opponents.metamon.checkout": "GEN3AI_METAMON_DIR",
    "opponents.metamon.python": "GEN3AI_METAMON_PYTHON",
    "opponents.metamon.cache_dir": "GEN3AI_METAMON_CACHE_DIR",
    "opponents.foulplay.checkout": "GEN3AI_FOULPLAY_DIR",
    "opponents.foulplay.python": "GEN3AI_FOULPLAY_PYTHON",
}


class AnchorConfigError(RuntimeError):
    """A configured path is missing, or a name the CLI accepted has no entry here."""


def _dig(blob: Dict[str, Any], dotted: str) -> Any:
    cur: Any = blob
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _plant(blob: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = blob
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


@dataclass(frozen=True)
class WeightPin:
    """What one Metamon agent is allowed to load: a hub repo at ONE commit and the sha256 of each weight file
    (:mod:`main.anchors.peer_scripts.weight_pins`). ``digest`` is the weight set's one id."""

    repo: str
    revision: str
    weights: Dict[str, str]

    @property
    def digest(self) -> str:
        return wp.weights_digest(self.weights)


@dataclass(frozen=True)
class MetamonOpponent:
    """The Metamon checkout, its interpreter, its weight/team cache, and the policies we name."""

    checkout: Path
    python: Path
    cache_dir: Path
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    team_sets: Dict[str, str] = field(default_factory=dict)
    #: Metamon's KNOWN team sets, ``{name: directory}``, that every read's ``team_overlap`` measures its
    #: team set against (:mod:`main.anchors.team_overlap`). Empty: every non-``gen3ai_*`` set in the cache.
    reference_team_dirs: Dict[str, str] = field(default_factory=dict)
    #: The Hugging Face repo and the FULL commit sha every agent's weights are pinned to (a per-agent
    #: ``hf_revision`` overrides it). Empty = unpinned, which :meth:`pin` refuses.
    hf_repo: str = ""
    hf_revision: str = ""

    kind: str = "metamon"

    def require(self, agent: str) -> None:
        _require_path(self.checkout, "opponents.metamon.checkout", "GEN3AI_METAMON_DIR")
        _require_path(self.python, "opponents.metamon.python", "GEN3AI_METAMON_PYTHON")
        _require_path(self.cache_dir, "opponents.metamon.cache_dir", "GEN3AI_METAMON_CACHE_DIR")
        if agent not in self.agents:
            raise AnchorConfigError(
                f"metamon agent {agent!r} is not in designs/ops/anchors.json "
                f"(known: {sorted(self.agents)}). Add it with its checkpoint number rather than "
                "passing one on the CLI — a policy is only an anchor once its checkpoint is pinned."
            )
        self.pin(agent)

    def pin(self, agent: str) -> WeightPin:
        """The agent's weight pin, validated: a full commit sha, a non-empty ``{path: sha256}`` map, and (for a
        single-checkpoint agent) exactly that checkpoint's file. A Metamon agent with no pin is REFUSED: its weights
        would be whatever the hub's moving ``main`` holds (FR-1 FINDING 3)."""
        entry = self.agents.get(agent, {})
        revision = entry.get("hf_revision", self.hf_revision)
        weights = entry.get("weights")
        try:
            if not self.hf_repo:
                raise wp.WeightPinError("opponents.metamon.hf_repo is not set")
            wp.check_pins(revision, weights, f"metamon agent {agent!r}")
        except wp.WeightPinError as exc:
            raise AnchorConfigError(
                f"{exc}. Every Metamon agent carries a pinned hub revision and the sha256 of each weight file in "
                "designs/ops/anchors.json (EXTERNAL_ANCHORS_SOP.md, 'Weight pins'). A config that carries no pins resolves "
                "them from that registry by agent name (same policy and checkpoint); this agent is not pinned there."
            ) from None
        checkpoint = entry.get("checkpoint")
        if checkpoint is not None and [Path(w).name for w in weights] != [f"policy_epoch_{int(checkpoint)}.pt"]:
            raise AnchorConfigError(
                f"metamon agent {agent!r}: checkpoint {checkpoint} but weights name {sorted(weights)}; a single-"
                f"checkpoint agent pins exactly its policy_epoch_{checkpoint}.pt")
        return WeightPin(repo=self.hf_repo, revision=revision, weights=dict(weights))

    def policy(self, agent: str) -> str:
        """The Metamon PRETRAINED POLICY an agent name plays (the peer's ``get_pretrained_model(<policy>)``):
        the agent's ``policy`` key, else the name itself. An ALIAS pins a second checkpoint of one policy
        under its own name — ``"Kakuna28": {"policy": "Kakuna", "checkpoint": 28}`` (X51, 2026-10-20: Kakuna
        ckpt 28 against ckpt 34) — so a pair cell can play two checkpoints of one policy, each pinned here."""
        policy = self.agents.get(agent, {}).get("policy", agent)
        if not isinstance(policy, str) or not policy:
            raise AnchorConfigError(
                f"metamon agent {agent!r}: its 'policy' must name a Metamon pretrained model, got {policy!r}")
        return policy

    def identity(self, agent: str) -> Tuple[str, Optional[int]]:
        """``(policy, checkpoint)``: what a peer under this name actually PLAYS. Two names with one identity
        are one policy (the self-pair refusal reads this, not the names)."""
        return self.policy(agent), self.agents.get(agent, {}).get("checkpoint")

    def preset(self, agent: str) -> Optional[Tuple[str, str]]:
        """``(preset name, sha256)`` for a Metamon ENSEMBLE agent, else ``None``. An ensemble (``PastaMittens``,
        the Kakuna router) has no checkpoint: its identity is the member preset it routes over, pinned by the
        sha256 of the preset's canonical JSON (``json.dumps(members, sort_keys=True, separators=(",", ":"))``
        over its entry in Metamon's ``ensemble_presets.json``). The peer refuses to play if either differs."""
        entry = self.agents.get(agent, {})
        name, sha = entry.get("preset"), entry.get("preset_sha256")
        if name is None and sha is None:
            return None
        if not (isinstance(name, str) and name and isinstance(sha, str) and len(sha) == 64):
            raise AnchorConfigError(
                f"metamon agent {agent!r}: an ensemble entry needs BOTH 'preset' (a name) and 'preset_sha256' "
                f"(64 hex), got {name!r} / {sha!r}")
        if entry.get("checkpoint") is not None:
            raise AnchorConfigError(f"metamon agent {agent!r}: an ensemble preset has no single checkpoint")
        return name, sha

    def team_set(self, teamset: str) -> str:
        try:
            return self.team_sets[teamset]
        except KeyError:
            raise AnchorConfigError(
                f"no metamon team set configured for --teamset {teamset!r} "
                f"(known: {sorted(self.team_sets)})"
            ) from None

    def team_dir(self, teamset: str, battle_format: str) -> Path:
        """Where the peer's own team files live, so their COUNT can be recorded per cell."""
        return self.cache_dir / "teams" / self.team_set(teamset) / battle_format


@dataclass(frozen=True)
class FoulPlayOpponent:
    """The Foul Play checkout, its own conda interpreter, and its per-team-set directories."""

    checkout: Path
    python: Path
    team_dirs: Dict[str, str] = field(default_factory=dict)

    kind: str = "foulplay"

    def require(self, agent: str = "") -> None:
        _require_path(self.checkout, "opponents.foulplay.checkout", "GEN3AI_FOULPLAY_DIR")
        _require_path(self.python, "opponents.foulplay.python", "GEN3AI_FOULPLAY_PYTHON")

    def team_name(self, teamset: str) -> str:
        """The ``--team-name`` value. Foul Play's ``load_team`` does
        ``os.path.join(TEAM_DIR, name)``, which returns ``name`` unchanged when it is ABSOLUTE —
        so a relative name resolves inside its own ``fp/teams/teams/`` and an absolute one points
        wherever we say. Both forms are supported here, deliberately."""
        try:
            return self.team_dirs[teamset]
        except KeyError:
            raise AnchorConfigError(
                f"no foulplay team directory configured for --teamset {teamset!r} "
                f"(known: {sorted(self.team_dirs)})"
            ) from None

    def team_dir(self, teamset: str, battle_format: str = "") -> Path:
        name = self.team_name(teamset)
        p = Path(name)
        return p if p.is_absolute() else self.checkout / "fp" / "teams" / "teams" / name


@dataclass(frozen=True)
class AnchorsConfig:
    """The whole file, resolved. ``source`` is the file it came from, for the provenance block."""

    source: Path
    node: str
    port_range: List[int]
    metamon: MetamonOpponent
    foulplay: FoulPlayOpponent
    our_team_sources: Dict[str, Dict[str, Any]]

    def opponent(self, kind: str) -> Any:
        if kind == "metamon":
            return self.metamon
        if kind == "foulplay":
            return self.foulplay
        raise AnchorConfigError(f"unknown opponent kind {kind!r} (known: metamon, foulplay)")

    def our_team_source(self, teamset: str) -> Dict[str, Any]:
        try:
            spec = dict(self.our_team_sources[teamset])
        except KeyError:
            raise AnchorConfigError(
                f"no OUR-side team source configured for --teamset {teamset!r} "
                f"(known: {sorted(self.our_team_sources)})"
            ) from None
        if spec.get("kind") == "dir":
            _require_path(Path(spec["path"]), f"our_team_sources.{teamset}.path", None)
        return spec


def _require_path(path: Path, key: str, env_var: Optional[str]) -> None:
    if path.exists():
        return
    hint = f" Override it with ${env_var}." if env_var else ""
    raise AnchorConfigError(
        f"{key} = {path} does not exist on this box. Edit designs/ops/anchors.json (or point "
        f"$GEN3AI_ANCHORS_CONFIG at your own copy).{hint}"
    )


def config_path() -> Path:
    """The file :func:`load_config` will read. Env override wins and is AUTHORITATIVE."""
    override = os.environ.get(CONFIG_PATH_ENV_VAR)
    if override:
        p = Path(override)
        if not p.is_file():
            raise AnchorConfigError(
                f"${CONFIG_PATH_ENV_VAR}={override} is not a file. An explicit override that "
                "silently fell back to the committed default would make a measurement describe a "
                "checkout nobody chose, so this refuses instead."
            )
        return p
    return repo_path(*DEFAULT_CONFIG_RELPATH)


def _fill_pins_from_registry(meta: Dict[str, Any], src: Path) -> None:
    """The weight pins belong to the OPPONENT, not to a panel's config. A config that does not carry them (every
    committed, sha-pinned panel config under ``designs/research_state/measurements/``, which ``main.x35.kakuna`` passes
    through ``$GEN3AI_ANCHORS_CONFIG``) resolves them from the canonical registry, ``designs/ops/anchors.json``
    ``opponents.metamon``, keyed by agent name. Filled: ``hf_repo`` / ``hf_revision`` when absent, and an agent's
    ``weights`` when absent AND the registry's agent of that name plays the same policy and checkpoint (a panel config
    naming Kakuna at another checkpoint gets no borrowed pin and is refused by :meth:`MetamonOpponent.pin`). A pin the
    config carries itself always wins. The panel file is never edited (its sha256 is pinned elsewhere)."""
    canonical = repo_path(*DEFAULT_CONFIG_RELPATH)
    if src.resolve() == canonical.resolve() or not canonical.is_file():
        return
    reg = (_dig(json.loads(canonical.read_text()), "opponents.metamon") or {})
    for key in ("hf_repo", "hf_revision"):
        if not meta.get(key) and reg.get(key):
            meta[key] = reg[key]
    reg_agents = reg.get("agents", {})
    for name, entry in (meta.get("agents") or {}).items():
        canon = reg_agents.get(name)
        if not isinstance(entry, dict) or canon is None or entry.get("weights"):
            continue
        same = all(entry.get(k) == canon.get(k) for k in ("policy", "checkpoint", "preset", "preset_sha256"))
        if same and canon.get("weights"):
            entry["weights"] = dict(canon["weights"])
            if canon.get("hf_revision") and not entry.get("hf_revision"):
                entry["hf_revision"] = canon["hf_revision"]


def load_config(path: Optional[Path] = None) -> AnchorsConfig:
    """Read the config, apply every env override, and return it. Paths are NOT checked here —
    :meth:`Opponent.require` does that at the point a run actually needs one, so ``--dry-run`` and
    ``--show-config`` still work on a box that has neither checkout."""
    src = Path(path) if path is not None else config_path()
    if not src.is_file():
        raise AnchorConfigError(f"anchors config not found at {src}")
    blob = json.loads(src.read_text())
    blob.pop("_README", None)

    for dotted, env_var in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            _plant(blob, dotted, value)

    meta = _dig(blob, "opponents.metamon") or {}
    _fill_pins_from_registry(meta, src)
    fp = _dig(blob, "opponents.foulplay") or {}
    return AnchorsConfig(
        source=src,
        node=str(_dig(blob, "showdown.node") or "node"),
        port_range=list(_dig(blob, "showdown.port_range") or [9500, 9599]),
        metamon=MetamonOpponent(
            checkout=Path(meta.get("checkout", "")),
            python=Path(meta.get("python", "")),
            cache_dir=Path(meta.get("cache_dir", "")),
            agents=dict(meta.get("agents", {})),
            team_sets=dict(meta.get("team_sets", {})),
            reference_team_dirs=dict(meta.get("reference_team_dirs", {})),
            hf_repo=str(meta.get("hf_repo", "")),
            hf_revision=str(meta.get("hf_revision", "")),
        ),
        foulplay=FoulPlayOpponent(
            checkout=Path(fp.get("checkout", "")),
            python=Path(fp.get("python", "")),
            team_dirs=dict(fp.get("team_dirs", {})),
        ),
        our_team_sources=dict(blob.get("our_team_sources", {})),
    )


def describe(cfg: AnchorsConfig) -> str:
    """``--show-config``: every resolved value beside the env var that overrides it."""
    rows = [
        ("source", str(cfg.source), CONFIG_PATH_ENV_VAR),
        ("showdown.node", cfg.node, ENV_OVERRIDES["showdown.node"]),
        ("showdown.port_range", str(cfg.port_range), ""),
        ("metamon.checkout", str(cfg.metamon.checkout), ENV_OVERRIDES["opponents.metamon.checkout"]),
        ("metamon.python", str(cfg.metamon.python), ENV_OVERRIDES["opponents.metamon.python"]),
        ("metamon.cache_dir", str(cfg.metamon.cache_dir),
         ENV_OVERRIDES["opponents.metamon.cache_dir"]),
        ("metamon.agents", ", ".join(sorted(cfg.metamon.agents)), ""),
        ("metamon.hf_pin", f"{cfg.metamon.hf_repo}@{cfg.metamon.hf_revision}", ""),
        ("metamon.team_sets", json.dumps(cfg.metamon.team_sets), ""),
        ("metamon.reference_team_dirs", json.dumps(cfg.metamon.reference_team_dirs), ""),
        ("foulplay.checkout", str(cfg.foulplay.checkout),
         ENV_OVERRIDES["opponents.foulplay.checkout"]),
        ("foulplay.python", str(cfg.foulplay.python), ENV_OVERRIDES["opponents.foulplay.python"]),
        ("foulplay.team_dirs", json.dumps(cfg.foulplay.team_dirs), ""),
        ("our_team_sources", json.dumps(cfg.our_team_sources), ""),
    ]
    width = max(len(k) for k, _, _ in rows)
    out = []
    for key, value, env_var in rows:
        exists = ""
        if key.endswith((".checkout", ".python", ".cache_dir")) or key == "source":
            exists = "  [OK]" if Path(value).exists() else "  [MISSING]"
        out.append(f"  {key:<{width}}  {value}{exists}"
                   + (f"   (${env_var})" if env_var else ""))
    return "\n".join(out)
