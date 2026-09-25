"""The Rust Core parity harness — slice O, the OBSERVATION (``gen3_core_parity_obs_v1``, the Rust
Core Program's M4).

At EVERY decision of a recorded battle, for BOTH viewers, the 2501-dim row TRAINING builds —
``Gen3ObservationEncoder.encode`` called exactly as ``Gen3Env.embed_battle`` calls it (the
EpisodeTracker's HP tracker, the legality snapshot, the progress clock, recency, pair history, the
event window and the incremental assembler) over a ``Gen3Battle`` fed the viewer's text — against
the core's (``core_events --obs``: the row the Rust encoder writes on the version from the same
stream, shipped as a wire frame and wrapped with ``np.frombuffer``). The gate is BYTE equality of the
float32 row (stricter than ``np.array_equal``: a ``-0.0`` for a ``0.0`` is a divergence), plus the
11-dim mask. **No allowlist.** A divergence is classified by the obs BLOCK its first differing cell
falls in (``assembler.describe_offset``), and every differing block is counted.

The Python side is driven inside slice T's decision loop (``rust_core_parity_trackers.check_trackers``)
so the tracker fold runs once for both slices.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

_ENCODER: Optional[Any] = None


def encoder():
    """One ``Gen3ObservationEncoder`` per process (its mappings are the facade's)."""
    global _ENCODER
    if _ENCODER is None:
        from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

        _ENCODER = Gen3ObservationEncoder(load_mappings())
    return _ENCODER


def python_row(battle, tracker, legal) -> np.ndarray:
    """The row ``Gen3Env.embed_battle`` builds for the trainee at this decision (after ``record`` and
    ``update_progress_clock``)."""
    enc = encoder()
    return enc.encode(
        battle, hp_tracker=tracker.hidden_power_tracker, legal=legal,
        progress_clock=tracker.progress_clock, recency=tracker.recency,
        pair_history=tracker.pair_history, event_window=tracker.event_window,
        assembler=tracker.obs_assembler(enc.dimension))


_SLOT_FIELDS = None


def block_of(index: int) -> str:
    """The FIELD a flat index names, slot-independent (``our_team moves+4``, ``event_window
    MAGNITUDE``), so one wrong field is one divergence class however many slots it hits."""
    global _SLOT_FIELDS
    from agents.observation import constants as C

    if _SLOT_FIELDS is None:
        names = [n for n in dir(C) if n.startswith("POKEMON_") and n.endswith("_OFFSET")]
        _SLOT_FIELDS = sorted((int(getattr(C, n)), n[len("POKEMON_"):-len("_OFFSET")].lower()) for n in names)
    i = int(index)
    if i < C.OFFSET_CONTEXT:
        side = "our_team" if i < C.OFFSET_OPP_TEAM else "opp_team"
        off = (i - (C.OFFSET_OUR_TEAM if side == "our_team" else C.OFFSET_OPP_TEAM)) % C.POKEMON_FULL_DIM
        base, name = max((o, n) for o, n in _SLOT_FIELDS if o <= off)
        return f"{side} {name}+{off - base}"
    if i < C.OFFSET_GLOBAL:
        k, off = divmod(i - C.OFFSET_CONTEXT, C.ACTIVE_CONTEXT_DIM)
        return f"active_context[{'ours' if k == 0 else 'opp'}] +{off}"
    if i < C.OFFSET_REACTIVE:
        return f"global +{i - C.OFFSET_GLOBAL}"
    if i < C.OFFSET_PAIR_HISTORY:
        return f"board +{i - C.OFFSET_REACTIVE}"
    if i < C.OFFSET_EVENT_WINDOW:
        return f"pair_history cell+{(i - C.OFFSET_PAIR_HISTORY) % C.PAIR_HISTORY_CELL_DIM}"
    col = (i - C.OFFSET_EVENT_WINDOW) % C.EVENT_TOKEN_DIM
    return f"event_window {C.EventCol(col).name}"


@dataclass
class ObsCensus:
    battles: int = 0
    viewers: int = 0
    decisions: int = 0
    rows_equal: int = 0
    #: cells that differ in VALUE (the row compared as float32 numbers) vs only in their BYTES
    value_cells: int = 0
    byte_only_cells: int = 0
    #: NaN cells in a core row — a cell the Rust encoder never wrote (the test build's poison)
    nan_cells: int = 0
    divergences: collections.Counter = field(default_factory=collections.Counter)
    examples: Dict[str, Any] = field(default_factory=dict)
    refused: List[str] = field(default_factory=list)
    #: per obs block, how many decisions had a nonzero cell there (the gate's non-vacuity)
    nonzero_blocks: collections.Counter = field(default_factory=collections.Counter)
    #: choice tokens compared (one per legal action per decision)
    tokens: int = 0

    def diverge(self, key: str, example: Any) -> None:
        self.divergences[key] += 1
        self.examples.setdefault(key, example)

    def render(self) -> str:
        head = (f"{self.battles} battles, {self.viewers} viewers, {self.decisions} decisions, "
                f"{self.rows_equal} rows byte-equal, {self.tokens} choice tokens")
        if not self.divergences and not self.refused:
            return f"✅ slice O: 0 divergences — {head}"
        out = [f"❌ slice O: {sum(self.divergences.values())} divergences in {len(self.divergences)} classes, "
               f"{len(self.refused)} refused — {head}; differing cells: {self.value_cells} by value, "
               f"{self.byte_only_cells} by bytes only, {self.nan_cells} NaN (unwritten)"]
        for k, n in self.divergences.most_common():
            out.append(f"   {n:7d}  {k}\n            e.g. {self.examples[k]}")
        for r in self.refused[:10]:
            out.append(f"   REFUSED {r}")
        return "\n".join(out)


_BLOCKS = None


def _block_spans():
    global _BLOCKS
    if _BLOCKS is None:
        from agents.observation import constants as C

        _BLOCKS = (("our_team", C.OFFSET_OUR_TEAM, C.OFFSET_OPP_TEAM),
                   ("opp_team", C.OFFSET_OPP_TEAM, C.OFFSET_CONTEXT),
                   ("context", C.OFFSET_CONTEXT, C.OFFSET_GLOBAL),
                   ("global", C.OFFSET_GLOBAL, C.OFFSET_REACTIVE),
                   ("board", C.OFFSET_REACTIVE, C.OFFSET_PAIR_HISTORY),
                   ("pair_history", C.OFFSET_PAIR_HISTORY, C.OFFSET_EVENT_WINDOW),
                   ("event_window", C.OFFSET_EVENT_WINDOW, C.OFFSET_EVENT_WINDOW + C.EVENT_WINDOW_DIM))
    return _BLOCKS


def python_tokens(battle, legal, mask) -> Dict[str, str]:
    """``view_successor._choice_map`` over the reference battle: the REAL mapper's choice string
    per legal action index (an index the mapper refuses is absent) — what a search branches on."""
    from agents.action.mapper import Gen3ActionMapper

    out: Dict[str, str] = {}
    for idx in np.flatnonzero(np.asarray(mask)):
        try:
            out[str(int(idx))] = Gen3ActionMapper.action_to_order(
                int(idx), battle, legal=legal).message[len("/choose "):]
        except Exception:                                    # noqa: BLE001 — _choice_map's rule
            continue
    return out


def compare_row(where: str, cap: Mapping[str, Any], py_row: np.ndarray, py_mask: np.ndarray,
                census: ObsCensus, py_tokens: Optional[Mapping[str, str]] = None) -> None:
    """One decision: the core's wire frame + mask (+ choice tokens) against the Python row + mask
    (+ the mapper's tokens)."""
    from agents.battle.core_obs import RowRefused, check_row, wrap_row

    census.decisions += 1
    if "obs" not in cap:
        census.diverge("[ALIGN] the core recorded no obs row at a decision", (where,))
        return
    try:
        core = wrap_row(cap["obs"])
        check_row(np.ascontiguousarray(py_row))
    except RowRefused as e:
        census.diverge("[REFUSED] a row is not the observation", (where, str(e)))
        return
    py = np.asarray(py_row, dtype=np.float32)
    for name, lo, hi in _block_spans():
        if np.any(py[lo:hi] != 0):
            census.nonzero_blocks[name] += 1
    if list(cap.get("mask", [])) != [int(x) for x in py_mask]:
        census.diverge("[MASK] the 11-dim action mask", (where, "core", cap.get("mask"), "python",
                                                         [int(x) for x in py_mask]))
    if py_tokens is not None:
        census.tokens += len(py_tokens)
        if dict(cap.get("tokens") or {}) != dict(py_tokens):
            census.diverge("[TOKENS] the choice string per legal action",
                           (where, "core", cap.get("tokens"), "python", dict(py_tokens)))
    if core.tobytes() == py.tobytes():
        census.rows_equal += 1
        return
    nan = np.isnan(core)
    census.nan_cells += int(nan.sum())
    by_bytes = core.view(np.uint32) != py.view(np.uint32)
    by_value = (core != py) | nan
    census.value_cells += int(by_value.sum())
    census.byte_only_cells += int((by_bytes & ~by_value).sum())
    seen = set()
    for i in np.flatnonzero(by_bytes | nan):
        key = f"[OBS] {block_of(int(i))}"
        if key in seen:
            continue
        seen.add(key)
        census.diverge(key, (where, int(i), "core", float(core[i]), "python", float(py[i])))


# ---------------------------------------------------------------------------
# the obs GOLDEN (`training/golden_obs_fixture.json`) — reproduced by the core
# ---------------------------------------------------------------------------

def golden_battles():
    """Play the obs golden's fixed battle set (``golden_obs_capture``: the deterministic rotation
    policy, the cycling teambuilders, sim seed ``BRIDGE_SEED``) and return ``(the Python vectors the
    capture recorded, the battles' input logs)`` — so the core can re-derive the trainee's rows."""
    import asyncio

    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.battle.gen3_battle import Gen3Battle
    from agents.battle.rust_core_parity import RecordedBattle
    from agents.observation import moves as _moves_enc
    from agents.training import golden_obs_capture as G
    from utils.bridge import reconstruction
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    _moves_enc._CATEGORY_VAL_CACHE.clear()
    pool = (TeamLoader().get_sample_teams() or TeamLoader().get_all_teams())[:G.N_TEAMS]
    common = dict(battle_format=G.BATTLE_FORMAT, server_configuration=LocalhostServerConfiguration,
                  start_listening=False, battle_class=Gen3Battle)
    p1 = G._DetPlayer(record=True, team=G._CyclingTeambuilder(pool),
                      account_configuration=AccountConfiguration("GoldCap", "pw"), **common)
    p2 = G._DetPlayer(team=G._CyclingTeambuilder(pool[1:] + pool[:1]),
                      account_configuration=AccountConfiguration("GoldOpp", "pw"), **common)
    asyncio.run(run_local_battles(p1, p2, G.N_BATTLES, seed=G.BRIDGE_SEED, impl="rust"))
    battles = []
    for tag in p1.battles:
        rec = reconstruction.pop_record(tag)
        if rec is None:
            raise RuntimeError(f"no __RECON__ for {tag}")
        players = rec.players()
        if players["p1"]["name"] != "GoldCap":
            raise RuntimeError(f"{tag}: the trainee is not p1 ({players['p1']['name']})")
        battles.append(RecordedBattle(label=f"golden_{len(battles)}", format_id=rec.format_id,
                                      seed=rec.prng_seed, p1=players["p1"], p2=players["p2"],
                                      commands=[list(c) for c in rec.commands]))
    return p1.vectors, battles


def core_rows(battles, viewer: int = 0) -> list:
    """The core's rows for ``viewer`` at every decision of ``battles``, in order."""
    from agents.battle.core_obs import wrap_row
    from agents.battle.rust_core_parity import run_core

    out = []
    for res in run_core(battles, trackers=True, obs=True):
        if not res["ok"]:
            raise RuntimeError(f"{res['label']}: {res['error']}")
        out.extend(wrap_row(c["obs"]) for c in res["trackers"][viewer] if "obs" in c)
    return out
