"""THE ENGINE'S ARCHITECTURES: which slot group serves which checkpoint (``main.h2h.play.H2HEngine``'s declaration).

One T2 slot group is ONE architecture (one served state-dict signature and one forward fingerprint,
``agents.inference.service.slots``). An engine declares at most :data:`MAX_GROUPS` of them, before it is built
(:func:`declare_engine` over the plan's cells), each with only the slots its cells need, and one eval core per
(player group, opponent group) its cells use: the X5 A/B's cross (``fixed_mass`` seed x ``blob`` seed) is a
two-architecture plan (FINDING F-U6-1, closed here). A single-architecture plan declares exactly the engine
``main.h2h play`` always built: one group of two slots and one core.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from main.h2h.errors import H2HError

if TYPE_CHECKING:
    from main.h2h.play import PlayerRef


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


class CellArchMismatch(H2HError):
    """A cell whose checkpoint the ENGINE cannot serve: its architecture matches neither of the engine's (at most
    :data:`MAX_GROUPS`) slot groups — each group is ONE state-dict signature and ONE forward fingerprint — or its
    (player group, opponent group) combination has no eval core, or its side has no slot in its group."""


def _load_host(ref: PlayerRef) -> Any:
    """``ref``'s checkpoint through the strict loader (``gen3_strict_checkpoint_load_v1``), on the CPU, eval mode.
    ``historical_load_kwargs`` strips the policy / extractor kwargs DELETED since the checkpoint was written (PopArt,
    the value-dist head) and REFUSES an ON one — a run trained at an older pin still loads at HEAD. The host copy is
    only the SOURCE of a slot load (T2 copies it into its slot in place): it is never served."""
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

    m = load_checkpoint_strict(ref.zip_path, device="cpu", **historical_load_kwargs(ref.zip_path))
    m.policy.eval()
    return m


def _historical(ref: PlayerRef) -> List[str]:
    from agents.model.snapshot import historical_load_kwargs

    return sorted((historical_load_kwargs(ref.zip_path).get("custom_objects") or {}))


def _terminal_of(ref: PlayerRef) -> Any:
    """The core's terminal block as the player's reward config declares it (the eval core is opened with it)."""
    from agents.training.reward_config import RewardConfig
    from utils.rust_env import episode as EP

    return EP.terminal_from_reward_config(RewardConfig.from_dict(_read_json(ref.config_path)))


@dataclass(frozen=True)
class EngineArch:
    """ONE architecture an engine serves (one SLOT GROUP), declared from the first checkpoint of the plan that has it:
    the architecture TOGGLES, the model version they build, the served state-dict SIGNATURE and FORWARD FINGERPRINT
    (the slot group's own two identities, ``agents.inference.service.slots``) and the core's TERMINAL — declared from
    the group's first PLAYER (the eval core's terminal is its player's reward config's), ``None`` while the group holds
    no player."""

    source: str
    toggles: Mapping[str, Any]
    version: Any
    signature: Any
    fingerprint: str
    terminal: Any = None

    @classmethod
    def of(cls, ref: PlayerRef, model: Any, side: str = "player") -> "EngineArch":
        from agents.model.snapshot import arch_toggles_from_model, current_model_version
        from agents.observation.state_encoder import load_mappings
        from agents.training.rust_rollout.build import _arch_key

        toggles = arch_toggles_from_model(model)
        sig, fp = _arch_key(model.policy)
        return cls(source=ref.id, toggles=toggles, version=current_model_version(load_mappings(), **toggles),
                   signature=sig, fingerprint=fp, terminal=_terminal_of(ref) if side == "player" else None)

    def differs(self, ref: PlayerRef, model: Any) -> Optional[str]:
        """WHAT differs between ``ref`` (loaded as ``model``) and this architecture, or ``None`` when it is this
        architecture (the terminal is not part of the architecture: :meth:`check` binds it for a player)."""
        from agents.model.snapshot import ModelVersion, arch_toggles_from_model
        from agents.training.rust_rollout.build import _arch_key

        toggles = arch_toggles_from_model(model)
        if toggles != dict(self.toggles):
            diff = sorted(k for k in set(toggles) | set(self.toggles) if toggles.get(k) != self.toggles.get(k))
            return f"the architecture toggles { {k: (toggles.get(k), self.toggles.get(k)) for k in diff} }"
        try:
            self.version.check_opponent_snapshot_compatible(ModelVersion.from_json_file(ref.config_path))
        except Exception as e:                                       # noqa: BLE001 - reported typed
            return f"the model version ({e})"
        sig, fp = _arch_key(model.policy)
        if sig != self.signature:
            return "the served state-dict signature"
        if fp != self.fingerprint:
            return f"the forward fingerprint ({fp[:12]} vs {self.fingerprint[:12]})"
        return None

    def terminal_differs(self, ref: PlayerRef) -> Optional[str]:
        if self.terminal is None:
            return None
        term = _terminal_of(ref)
        return None if term == self.terminal else f"the terminal ({term} vs {self.terminal})"

    def check(self, ref: PlayerRef, model: Any, side: str) -> None:
        """:class:`CellArchMismatch` naming the cell's side, its checkpoint, this group's source and WHAT differs,
        unless ``ref`` (loaded as ``model``) is this architecture. The terminal binds the PLAYER only (the core's
        terminal is the player's)."""
        what = self.differs(ref, model) or (self.terminal_differs(ref) if side == "player" else None)
        if what is not None:
            raise CellArchMismatch(f"cell {side} {ref.id}: {what} differs from the slot group's (declared from "
                                   f"{self.source}) — a slot group serves ONE architecture")


#: Slot groups (architectures) one engine declares at most: the X5 A/B's cross is TWO (``fixed_mass`` x ``blob``).
MAX_GROUPS = 2
#: The slot groups' names, in declaration order (the first is the single-architecture engine's own, unchanged).
GROUP_NAMES = ("eval", "eval_b")
#: A slot's ROLE in its group, in slot order: the player's eval slot, then the opponent's sentinel slot.
ROLES = ("player", "opponent")


@dataclass(frozen=True)
class EngineDecl:
    """What ONE engine serves, declared BEFORE it is built (the declared lifecycle: nothing is added after startup).

    * ``archs`` — one :class:`EngineArch` per slot group (1 or 2), in first-use order: group 0 is the first cell's
      PLAYER's architecture, exactly as the single-architecture engine declares it.
    * ``roles`` — per group, the slots it holds, in slot order: ``"player"`` (an eval slot) when one of its
      checkpoints is a cell's player, ``"opponent"`` (a sentinel slot) when one is a cell's opponent. A group holds
      ONLY the slots its cells need: the X5 cross (every ``fixed_mass`` seed against every ``blob`` seed) declares the
      ``fixed_mass`` group with one player slot and the ``blob`` group with one opponent slot — two slots, as many as
      a single-architecture engine.
    * ``combos`` — the (player group, opponent group) pairs the cells use; ONE eval core each (an eval core's route
      table names the opponent's slot, fixed when the core opens).
    * ``members`` — every checkpoint's group, by content hash.
    """

    archs: Tuple[EngineArch, ...]
    sources: Tuple[PlayerRef, ...]
    roles: Tuple[Tuple[str, ...], ...]
    combos: Tuple[Tuple[int, int], ...]
    members: Mapping[str, int]

    @property
    def n_slots(self) -> int:
        return sum(len(r) for r in self.roles)

    def slot_of(self, group: int, role: str) -> int:
        """The GLOBAL slot id (the service's declaration order: group 0's slots, then group 1's) of ``group``'s
        ``role`` slot."""
        if role not in self.roles[group]:
            raise CellArchMismatch(f"slot group {group} ({GROUP_NAMES[group]}) declares no {role} slot")
        return sum(len(r) for r in self.roles[:group]) + self.roles[group].index(role)

    def local(self, slot: int) -> Tuple[int, int]:
        """``(group, index in the group)`` of a GLOBAL slot id."""
        base = 0
        for gi, r in enumerate(self.roles):
            if slot < base + len(r):
                return gi, slot - base
            base += len(r)
        raise H2HError(f"slot {slot} is not declared (0..{self.n_slots - 1})")

    def block(self) -> Dict[str, Any]:
        return {"groups": [{"name": GROUP_NAMES[i], "source": a.source, "roles": list(self.roles[i]),
                            "fingerprint": a.fingerprint[:12]} for i, a in enumerate(self.archs)],
                "combos": [list(c) for c in self.combos]}


def declare_engine(cells: Sequence[Tuple[PlayerRef, PlayerRef]], host: Callable[[PlayerRef], Any]) -> EngineDecl:
    """Sort every side of every cell into at most :data:`MAX_GROUPS` architectures (``host`` loads a checkpoint; each
    distinct checkpoint is asked for ONCE, in first-use order, the first cell's player first). A checkpoint whose
    architecture matches neither group is a typed :class:`CellArchMismatch` naming it and what differs from each; a
    player whose terminal differs from its group's first player's is refused the same way."""
    if not cells:
        raise H2HError("no cell to declare an engine for")
    sides: Dict[str, Tuple[PlayerRef, set]] = {}
    for p, o in cells:
        sides.setdefault(p.sha256, (p, set()))[1].add("player")
        sides.setdefault(o.sha256, (o, set()))[1].add("opponent")
    archs: List[EngineArch] = []
    sources: List[PlayerRef] = []
    members: Dict[str, int] = {}
    for ref, roles in sides.values():
        model = host(ref)
        gi: Optional[int] = None
        why: List[str] = []
        for i, a in enumerate(archs):
            d = a.differs(ref, model)
            if d is None:
                gi = i
                break
            why.append(f"group {i} (declared from {a.source}): {d}")
        if gi is None:
            if len(archs) >= MAX_GROUPS:
                raise CellArchMismatch(
                    f"cell {'/'.join(r for r in ROLES if r in roles)} {ref.id}: its architecture matches neither of "
                    f"the engine's {MAX_GROUPS} slot groups — {'; '.join(why)}. One engine serves at most "
                    f"{MAX_GROUPS} architectures; play this cell on an engine of its own")
            archs.append(EngineArch.of(ref, model, "player" if "player" in roles else "opponent"))
            sources.append(ref)
            gi = len(archs) - 1
        elif "player" in roles:
            if archs[gi].terminal is None:
                archs[gi] = replace(archs[gi], terminal=_terminal_of(ref))
            else:
                t = archs[gi].terminal_differs(ref)
                if t is not None:
                    raise CellArchMismatch(f"cell player {ref.id}: {t} differs from slot group {gi}'s (declared "
                                           f"from its first player) — one eval core serves ONE terminal")
        members[ref.sha256] = gi
        del model
    group_roles: List[set] = [set() for _ in archs]
    for sha, (_ref, rs) in sides.items():
        group_roles[members[sha]] |= rs
    roles_of = [tuple(r for r in ROLES if r in gr) for gr in group_roles]
    combos: List[Tuple[int, int]] = []
    for p, o in cells:
        c = (members[p.sha256], members[o.sha256])
        if c not in combos:
            combos.append(c)
    return EngineDecl(archs=tuple(archs), sources=tuple(sources), roles=tuple(roles_of), combos=tuple(combos),
                      members=dict(members))
