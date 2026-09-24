"""ONE-SIDED VIEW parity — the read-models built from the port's payload vs the ones poke-env
builds by replaying the same ply's protocol (`gen3_one_sided_view_v1`).

**This is the point of the exercise.** ``src/rust_sim/src/view.rs`` claims to hand Python the
board a successor's observation needs, so the materializer need not re-derive it by parsing
protocol text. The claim is only worth anything if the two roads arrive at the SAME vector, so
this gate drives both on real gen3ou battles and compares FOUR things:

1. the whole :class:`~agents.battle.live_view.LiveView` graph, FIELD BY FIELD, per mon;
2. the :class:`~agents.battle.live_view.LegalActions` snapshot, field by field;
3. the **2501-dim observation vector**, ``np.array_equal`` on float32, encoded from each — with
   NO trackers and NO assembler on either side;
4. at every BRANCH point, the **FULL tracker-fed successor observation AND its action mask**,
   against the production ``materialize_branches`` row (:func:`check_successor`).

**Why 3 and 4 are both here, and neither replaces the other.** The tracker-less comparison is
SHARP: the tracker-fed blocks (recency, pair history, the event window, the progress clock, the
Hidden-Power block) are ZERO in both vectors, so every surviving difference is attributable to
the READ-MODELS and to nothing else. It is also silent about five blocks of the observation a
real leaf reads, which is what comparison 4 is for — it runs the whole
`gen3_view_successor_v1` road, so a difference there can be the read-models OR the event fold OR
the tracker advance. Comparison 4 runs only where 3 came back clean, because a vector built on a
differing board says nothing about the trackers.

``designs/rust_sim/one_sided_view.md`` holds the deferral list.

🚨 **A MISMATCH IS A FINDING.** The run prints a CENSUS keyed by field path, so a class is named
and counted rather than reduced to one failing assert. Two things it can be: a missed field (the
port did not send something poke-env tracks) or the WALL (the port sent something poke-env has not
been told); the census shows both values, so the direction is readable.

**Exactly ONE class is DECLARED residual** (:data:`DECLARED_RESIDUAL`), naming its deferral in
``designs/rust_sim/one_sided_view.md``, matched by a NARROW path predicate, and PRINTED with its
count on every run. Everything else fails. The narrowness is the point and it is enforced by
construction: the Wish entry matches only the two reactive columns the layout DECLARES (never "an
obs difference"). The second entry this gate used to carry — D6, an opponent's ``current_pp``
judged against Pressure at READ time — is CLOSED (reading rule V3 judges it at USE time).

**TWO ENTRY POINTS, and the split is the project's fuzz rule.** A fuzz SCRIPT wants a new battle
every run; a pytest-collected TEST wants the SAME battle every run. So:

* the **sweep** (:func:`run`) records fresh random battles and is run as a script — that is what
  found every finding above, and three of them turned up only on the second or third seed;
* the **collected test** takes its battle from
  ``obs_roundtrip_fuzz_test.record_fixture_battle(key=…)`` — pinned teams, a pinned per-player
  RNG and a fixed sim seed — so it is the same board on every run and cannot ride main red on a
  draw nobody has seen.

🚨 **Run the sweep on at least TWO fresh seeds before calling it green** — the same rule the rust
parity harnesses carry, and for the same measured reason: three findings appeared only on the
second or third seed.

The three read-model findings this sweep used to surface (the ``status_counter`` drift, an
opponent ability disclosed off a non-``-ability`` line, a missing Baton-Passed ``substitute``) are
CLOSED — root-caused and pinned by the Rust Core parity harness's slice V
(``rust_core_parity_views.py``), which runs the same projection at EVERY decision of hundreds of
battles. ``designs/rust_sim/one_sided_view.md`` §4b has each root cause.

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/battle/one_sided_view_parity_fuzz_test.py [n_battles] [--arms K]
    pytest -m sim src/agents/battle/one_sided_view_parity_fuzz_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

import agents.training.obs_materializer as OM
from agents.battle.event_fold import ViewEventFolder
from agents.battle.live_view import LegalActions, LivePokemon, LiveSide, LiveView
from agents.battle.poke_env_findings import explain, obs_block_explained
from agents.battle.view_adapter import read_models_from_payload
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from agents.training.view_successor import (ViewSuccessorFactory,
                                            intermediate_decisions,
                                            split_at_intermediate)
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.bridge.search_session import SearchSession
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _wish_columns() -> "frozenset":
    """The two obs indices the reactive block DECLARES for the pending-Wish pair — resolved from
    ``get_layout()``, never written as a literal (the positional-binding sweep convicted that
    pattern five times)."""
    from agents.observation.constants import OFFSET_REACTIVE
    from agents.observation.reactive import ReactiveEncoder

    lay = ReactiveEncoder().get_layout()
    return frozenset(
        OFFSET_REACTIVE + lay[k]["offset"] for k in ("wish_floating_our", "wish_floating_opp"))


#: (path predicate, deferral id, why) — the ONLY divergence classes that do not fail. Both are
#: recorded in ``designs/rust_sim/one_sided_view.md``; a class that leaves this list must be
#: deleted from it, and a class that is not on it fails the gate.
DECLARED_RESIDUAL = (
    (lambda p: p == "obs[reactive.wish_floating]", "D3",
     "the reactive Wish pair folds `battle.events`, and THIS comparison threads no event log — "
     "it is the board-only road. D3 is CLOSED on the road that matters: a SUCCESSOR carries the "
     "whole-battle log (the root's plus the ply's) and the existing fold runs unchanged, which "
     "is why `check_successor` compares the Wish columns strictly. The port's own "
     "`SideState::wish_pending` was never needed — the payload was not the problem, the missing "
     "LOG was."),
)
# D6 (Pressure judged at READ time) is CLOSED — each sighting now carries the target's
# ability-event index AT USE TIME (reading rule V3, `designs/rust_sim/one_sided_view.md` §2b), so
# an opponent's `current_pp` is compared strictly like every other field.


def _declared(path: str):
    for pred, did, why in DECLARED_RESIDUAL:
        if pred(path):
            return did, why
    return None

BATTLE_FORMAT = "gen3ou"

#: Branch points to expand per opened root. The task's bar is >= 200 seeded mid-battle branch
#: points overall; the default run (4 battles x 3 turns x 6 arms + the roots) clears it.
DEFAULT_ARMS = 6
DEFAULT_TURNS = 3


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

class Census:
    """Divergences, keyed by FIELD PATH rather than by occurrence.

    A single wrong field produces one census ROW with a count and one example, instead of N
    identical assertion failures — which is the difference between "the port does not send
    `protect_counter`" and "1,400 decisions failed"."""

    def __init__(self) -> None:
        self.rows: Counter = Counter()
        self.examples: Dict[str, str] = {}
        self.compared = 0
        self.deferred: Counter = Counter()
        self.declared: Counter = Counter()
        #: KNOWN poke-env reading findings the CORE road's TRUE reading differs by
        #: (``agents.battle.poke_env_findings``), value-aware: per finding id, the read-model fields
        #: and (``finding:obs``) the successor obs blocks it explained. Never a divergence.
        self.known: Counter = Counter()

    def note(self, path: str, protocol: Any, view: Any, where: str) -> None:
        hit = _declared(path)
        if hit is not None:
            self.declared[f"{hit[0]}  {path}"] += 1
            return
        self.rows[path] += 1
        self.examples.setdefault(
            path, f"{where}: protocol={protocol!r} view={view!r}")

    def defer(self, reason: str) -> None:
        self.deferred[reason] += 1

    def render(self) -> str:
        if not self.rows:
            head = f"✅ no divergence over {self.compared} comparisons"
        else:
            head = (f"❌ {sum(self.rows.values())} divergences in {len(self.rows)} classes "
                    f"over {self.compared} comparisons")
        lines = [head]
        for path, n in self.rows.most_common():
            lines.append(f"   {n:6d}  {path}")
            lines.append(f"           e.g. {self.examples[path]}")
        for key, n in self.declared.most_common():
            lines.append(f"   [DECLARED residual x{n}] {key}")
        for reason, n in self.deferred.most_common():
            lines.append(f"   [deferred x{n}] {reason}")
        for fid, n in self.known.most_common():
            lines.append(f"   [KNOWN poke-env finding x{n}] {fid}")
        for _pred, did, why in DECLARED_RESIDUAL:
            if not any(k.startswith(did) for k in self.declared):
                lines.append(f"   [DECLARED residual {did}: NOT SEEN this run] {why[:70]}…")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Field-by-field comparison of the read-models
# ---------------------------------------------------------------------------

def _cmp_mon(a: LivePokemon, b: LivePokemon, path: str, cen: Census,
             fired: Optional[set] = None) -> None:
    """``fired`` (the CORE road only — it reads the TRUTH): a difference a registered poke-env
    finding explains, value-aware, is recorded there and in ``cen.known``, not as a divergence."""
    for f in dc_fields(LivePokemon):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "hp_fraction":
            if abs(float(x) - float(y)) > 1e-9:
                cen.note(f"{path}.hp_fraction", x, y, path)
            continue
        if f.name == "moves":
            # The SLOTS (which moves are revealed, and their max pp) are compared strictly; only
            # the current_pp VALUE can fall under a declared residual.
            sx = [(m.id, m.max_pp) for m in x]
            sy = [(m.id, m.max_pp) for m in y]
            if sx != sy:
                cen.note(f"{path}.moves[slots]", sx, sy, path)
            elif [m.current_pp for m in x] != [m.current_pp for m in y]:
                cen.note(f"{path}.moves[current_pp]",
                         [m.current_pp for m in x], [m.current_pp for m in y], path)
            continue
        if f.name in ("boosts", "volatiles", "base_stats", "stats"):
            x, y = dict(x), dict(y)
        if x != y and fired is not None:
            fid = explain(f.name, a, b)
            if fid is not None:
                cen.known[fid] += 1
                fired.add(fid)
                continue
        if x != y:
            cen.note(f"{path}.{f.name}", x, y, path)


def _cmp_side(a: LiveSide, b: LiveSide, which: str, cen: Census, fired: Optional[set] = None) -> None:
    if a.team_size != b.team_size:
        cen.note(f"{which}.team_size", a.team_size, b.team_size, which)
    if dict(a.side_conditions) != dict(b.side_conditions):
        cen.note(f"{which}.side_conditions",
                 dict(a.side_conditions), dict(b.side_conditions), which)
    sa = [m.species for m in a.mons]
    sb = [m.species for m in b.mons]
    if sa != sb:
        # ORDER matters: it IS the obs slot assignment.
        cen.note(f"{which}.mons[order]", sa, sb, which)
        return
    for m1, m2 in zip(a.mons, b.mons):
        _cmp_mon(m1, m2, f"{which}.{m1.species}", cen, fired)


def compare_live(a: LiveView, b: LiveView, cen: Census, fired: Optional[set] = None) -> None:
    cen.compared += 1
    if a.turn != b.turn:
        cen.note("turn", a.turn, b.turn, "view")
    for f in ("finished", "won", "lost"):
        if getattr(a, f) != getattr(b, f):
            cen.note(f, getattr(a, f), getattr(b, f), "view")
    for f in ("weather", "is_permanent", "turns_active"):
        if getattr(a.weather, f) != getattr(b.weather, f):
            cen.note(f"weather.{f}", getattr(a.weather, f), getattr(b.weather, f), "view")
    _cmp_side(a.ours, b.ours, "ours", cen, fired)
    _cmp_side(a.opp, b.opp, "opp", cen, fired)


def compare_legal(a: Optional[LegalActions], b: Optional[LegalActions], cen: Census) -> None:
    if a is None or b is None:
        if (a is None) != (b is None):
            cen.note("legal[presence]", a is None, b is None, "legal")
        return
    for f in dc_fields(LegalActions):
        if f.name == "last_request":
            continue  # the raw mirror: wire-truth on both roads by construction
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x != y:
            cen.note(f"legal.{f.name}", x, y, "legal")


# ---------------------------------------------------------------------------
# The two roads
# ---------------------------------------------------------------------------

def _record_one_battle(out_dir: str, impl: str, fixed_key: Optional[int] = None):
    """A real gen3ou battle. ``fixed_key`` is not None ⇒ the REPRODUCIBLE fixture builder (pinned
    teams, per-player RNG and sim seed) the collected test needs; otherwise a fresh random one,
    which is what the sweep wants."""
    if fixed_key is not None:
        from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle

        return record_fixture_battle(out_dir, key=fixed_key, tag="OV", impl=impl)
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"OVt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"OVo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1, impl=impl))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


class _ProtocolRoad:
    """The CURRENT path: feed one-sided protocol text to poke-env and read the battle it built.

    It reaches for ``obs_materializer``'s player builder rather than calling
    :func:`~agents.training.obs_materializer.materialize_decisions`, because that function
    returns the OBS and this gate needs the BATTLE — the `LiveView` and `LegalActions` the obs
    was built from are exactly what is on trial. Same construction, same feed, same cadence;
    the precedent is ``obs_materializer_branch_integration_test.py``."""

    def __init__(self, record, side: str, actions: List[int]) -> None:
        self.tag = OM._next_tag(None, record.format_id)
        self.player, self.client = OM._build_replay_player(
            username=record.username(side), packed_team=record.packed_team(side), side=side,
            actions=actions, battle_format=record.format_id, mappings=None, stall_config=None,
            map_actions_at=None, stop_after_decision=None, encode_only_at=None)
        self._first = True

    def feed(self, chunks) -> None:
        OM._refuse_poke_loop("one_sided_view parity")
        asyncio.run_coroutine_threadsafe(
            OM._feed(self.client, self.player, list(chunks), self.tag, first=self._first),
            OM.POKE_LOOP).result()
        self._first = False

    @property
    def battle(self):
        return self.player._battles.get(self.tag)


def _encode(encoder, battle, legal) -> np.ndarray:
    """Encode with NO trackers and NO assembler — the full rebuild, which is also the oracle."""
    return encoder.encode(battle, legal=legal)


def _anyone_asleep(live: LiveView) -> bool:
    return any(m.status == "slp" for m in (*live.ours.mons, *live.opp.mons))


def check_point(encoder, road: _ProtocolRoad, payload: dict, where: str, cen: Census,
                ledger=None) -> bool:
    """Compare the two roads at ONE decision point. True when the READ-MODELS agreed outright —
    which is also the precondition for the tracker-fed comparison downstream, because a vector
    built on a differing board says nothing about the trackers."""
    battle = road.battle
    if battle is None:
        cen.defer("the protocol replay produced no battle (chunks did not start one)")
        return False
    strict = battle.strict_view()
    live_p, legal_p = strict.live, strict.legal
    # `ledger` (the ply's folded `ViewEventFolder`) is accepted for the callers' symmetry and no
    # longer read: the view road's fainted-mon boost restore was retired with the fork's PE-V10
    # fix (`gen3_pe_reading_fixes_v1`) — a fainted mon's stages are the payload's, none.
    live_v, legal_v, vbattle = read_models_from_payload(payload, battle_tag=live_p.battle_tag)

    before = sum(cen.declared.values())
    compare_live(live_p, live_v, cen)
    compare_legal(legal_p, legal_v, cen)
    if sum(cen.declared.values()) > before:
        # 🚨 A vector CANNOT be compared downstream of an input difference that is already
        # declared. Reporting it as a pass would be false and as a failure would double-count the
        # same finding, so the decision is INCONCLUSIVE and says so — the same discipline the
        # project applies to a timeout.
        cen.defer("obs SKIPPED: a DECLARED read-model residual fired here, so the vector is "
                  "downstream of a known input difference and is not comparable")
        return False

    if _anyone_asleep(live_p):
        # `state_encoder.encode` folds the WHOLE event log for the 3-dim sleep-wake belief when
        # anyone is asleep (`build_sleep_sources`), and the payload carries a board rather than an
        # event log. DEFERRAL D2 — reported, never silently skipped.
        cen.defer("a mon is ASLEEP and this comparison threads NO event log, so the sleep-wake "
                  "belief reads 0 on the view side (D4 is closed only where a log is supplied — "
                  "the SUCCESSOR comparison below does supply one)")
        return False
    obs_p = _encode(encoder, battle, legal_p)
    obs_v = _encode(encoder, vbattle, legal_v)
    if obs_p.shape != obs_v.shape:
        cen.note("obs[shape]", obs_p.shape, obs_v.shape, where)
        return False
    if not np.array_equal(obs_p, obs_v):
        bad = set(int(i) for i in np.flatnonzero(obs_p != obs_v))
        wish = _wish_columns()
        if bad <= wish:
            cen.note("obs[reactive.wish_floating]", 0, 0, where)
            return False
        first = min(bad - wish)
        cen.note(f"obs[{_block_of(encoder, first)}]",
                 float(obs_p[first]), float(obs_v[first]),
                 f"{where} idx={first} ndiff={len(bad)}")
        return False
    return True


def check_successor(factory, encoder, arm_road, payload: dict, chunks, action: int,
                    dec_i: int, where: str, cen: Census) -> bool:
    """The FULL observation — every block, TRACKERS INCLUDED — at ONE branch point.

    🚨 **This is the comparison the tracker-less one above cannot make, and it is what
    `gen3_view_successor_v1` is for.** ``check_point`` deliberately threads no trackers, so the
    recency / pair-history / event-window / progress-clock / Hidden-Power blocks are ZERO in both
    its vectors — sharp about the read-models, and silent about five blocks of the observation a
    real leaf reads. Here the PROTOCOL road's own materialized successor row (the production
    ``materialize_branches`` output: tracker-fed, assembler-warmed) is compared against the VIEW
    road's :class:`~agents.training.view_successor.ViewSuccessorFactory`, which carries the
    root's ``EpisodeTracker`` forward and advances it with the ply folded from the arm's own
    one-sided protocol.

    The MASK is compared too. A leaf's obs is only half of what the materializer returns — the
    action mask is the other half, and a search that read the right vector against the wrong
    legality would still choose wrongly.

    Returns True when the point was actually compared."""
    mats = arm_road.player._materialized
    if len(mats) <= dec_i:
        cen.defer("the protocol successor produced no decision row (no request at this arm)")
        return False
    got = factory.successor(payload, chunks, action)
    if got is None:
        cen.defer("the VIEW successor reports no decision (all-zero mask / finished) where the "
                  "protocol road materialized one")
        return False
    obs_v, mask_v = got.obs, got.mask
    d = mats[dec_i]
    ok = True
    if not np.array_equal(np.asarray(d.mask), mask_v):
        cen.note("successor.mask", list(np.asarray(d.mask)), list(mask_v), where)
        ok = False
    if d.obs is None:
        cen.defer("the protocol successor row carries no obs (encode_only_at skipped it)")
        return False
    if d.obs.shape != obs_v.shape:
        cen.note("successor.obs[shape]", d.obs.shape, obs_v.shape, where)
        return False
    if not np.array_equal(d.obs, obs_v):
        bad = sorted(int(i) for i in np.flatnonzero(d.obs != obs_v))
        # One census row PER BLOCK, so a tracker-block divergence reads as
        # `successor.obs[pair_history]` rather than as one index.
        seen: Dict[str, int] = {}
        for i in bad:
            seen.setdefault(_block_of(encoder, i), i)
        for blk, i in seen.items():
            cen.note(f"successor.obs[{blk}]", float(d.obs[i]), float(obs_v[i]),
                     f"{where} idx={i} ndiff={len(bad)}")
        ok = False
    return ok


def check_core(factory, encoder, road, row_road, core: dict, action: int, dec_i: int,
               where: str, cen: Census) -> bool:
    """The CORE road (`gen3_core_search_v1`) at ONE branch point — the THIRD road.

    The arm's successor is the driver's Rust-core VERSION: its ``present()`` view (the side's own
    stream, every poke-env rule applied in Rust) against the protocol road's battle at the SAME
    decision (``road`` — for a D10 arm the road stopped at the cut, since the core leaf IS the
    version at the intermediate decision), field by field; then the FULL tracker-fed successor obs
    and mask from :class:`~agents.training.core_successor.CoreSuccessorFactory` against
    ``materialize_branches``' own row (``row_road``'s decision ``dec_i``). The arm was expanded
    with the INTEGRITY check on, so the factory also encodes the text path's view and raises on
    any byte difference. No declared residual applies to this road: the version carries the log,
    so the Wish pair and the sleep belief are compared like every other block."""
    from agents.battle.core_view import legal_actions_from_core, live_view_from_core

    battle = road.battle
    if battle is None:
        cen.defer("[core] the protocol replay produced no battle")
        return False
    strict = battle.strict_view()
    live_c = live_view_from_core(core["view"], battle_tag=strict.live.battle_tag)
    legal_c = legal_actions_from_core(core.get("legal"), core.get("request"))
    sub = Census()
    fired: set = set()
    compare_live(strict.live, live_c, sub, fired)
    compare_legal(strict.legal, legal_c, sub)
    cen.compared += sub.compared
    cen.known.update(sub.known)
    for path, n in sub.rows.items():
        cen.rows[f"core.{path}"] += n
        cen.examples.setdefault(f"core.{path}", f"{where}: {sub.examples[path]}")
    if sub.rows:
        return False
    mats = row_road.player._materialized
    if len(mats) <= dec_i:
        cen.defer("[core] the protocol successor produced no decision row")
        return False
    got = factory.successor(core, action, where=where)
    if got is None:
        cen.defer("[core] the CORE successor reports no decision where the protocol road has one")
        return False
    d = mats[dec_i]
    ok = True
    if not np.array_equal(np.asarray(d.mask), got.mask):
        cen.note("core.successor.mask", list(np.asarray(d.mask)), list(got.mask), where)
        ok = False
    if d.obs is None:
        cen.defer("[core] the protocol successor row carries no obs")
        return False
    if not np.array_equal(d.obs, got.obs):
        bad = sorted(int(i) for i in np.flatnonzero(d.obs != got.obs))
        seen: Dict[str, int] = {}
        for i in bad:
            seen.setdefault(_block_of(encoder, i), i)
        for blk, i in seen.items():
            if obs_block_explained(blk, fired):
                # A block a finding that FIRED on this successor's read-model may touch.
                cen.known[f"{'+'.join(sorted(fired))}:obs"] += 1
                continue
            cen.note(f"core.successor.obs[{blk}]", float(d.obs[i]), float(got.obs[i]),
                     f"{where} idx={i} ndiff={len(bad)} findings={sorted(fired)}")
            ok = False
    return ok


def _block_of(encoder, idx: int) -> str:
    """Name the block (and, inside a per-mon slot, the FIELD) an index falls in, resolved from the
    DECLARED layout — never a literal.

    ``get_layout()["parts"]`` carries the top-level ``{start, end, reshape}`` spans and
    ``["pokemon"]`` the per-mon slot's own ``{offset, dim, layout}`` tree, so a failure reads
    ``opp_team[5].moves`` rather than ``?``."""
    lay = encoder.get_layout()
    for name, part in (lay.get("parts") or {}).items():
        start, end = part.get("start"), part.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and start <= idx < end):
            continue
        shape = part.get("reshape")
        if not (isinstance(shape, tuple) and len(shape) == 2):
            return name
        row, col = divmod(idx - start, shape[1])
        # Only a TEAM block's rows are per-mon slots. `context` is (2, ACTIVE_CONTEXT_DIM) — one
        # row per side's active — and naming its columns from the per-mon layout misnamed a boost
        # byte `context[0].species`, which no finding's `context` pattern could match.
        if not name.endswith("_team"):
            return f"{name}[{row}]"
        field = _field_of(lay.get("pokemon") or {}, col)
        return f"{name}[{row}].{field}"
    return "?"


def _field_of(slot_layout: Any, col: int) -> str:
    """The per-mon slot FIELD at column ``col``, from the slot's own declared layout."""
    best, best_off = "?", -1
    if isinstance(slot_layout, dict):
        for name, v in slot_layout.items():
            if not (isinstance(v, dict) and isinstance(v.get("offset"), int)):
                continue
            off, dim = v["offset"], int(v.get("dim", 1))
            if off <= col < off + dim and off > best_off:
                best, best_off = name, off
    return best


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

#: The CORE road's compared branch points (and how many were D10 leaves) in the last :func:`run`.
CORE_POINTS = 0
CORE_D10 = 0

def run(n_battles: int = 2, arms: int = DEFAULT_ARMS, turns: int = DEFAULT_TURNS,
        impl: str = "rust", fixed_key: Optional[int] = None) -> Tuple[Census, int, int, int]:
    cen = Census()
    encoder = get_observation_encoder(load_mappings())
    branch_points = 0
    full_obs_points = 0
    d10_points = 0
    global CORE_POINTS, CORE_D10
    CORE_POINTS, CORE_D10 = 0, 0
    for b in range(n_battles):
        with tempfile.TemporaryDirectory() as td:
            record, summary, npz = _record_one_battle(
                td, impl, None if fixed_key is None else fixed_key + b)
        actions = np.asarray(npz["actions"], dtype=int)
        invs = summary["invocations"]
        side = record.side_of(record.trainee_username)
        other = "p2" if side == "p1" else "p1"
        cand = [i for i, inv in enumerate(invs)
                if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
        if not cand:
            continue
        picks = [cand[int(len(cand) * f)] for f in (0.25, 0.5, 0.75)][:turns]
        with SearchSession(record, impl=impl) as ss, SearchSession(record, impl=impl) as cs:
            for anchor in picks:
                turn = int(invs[anchor]["turn"])
                try:
                    root = ss.open_root(turn)
                except Exception as e:                                # noqa: BLE001
                    cen.defer(f"open_root({turn}) failed: {type(e).__name__}")
                    continue
                if not (root.view_p1 and root.view_p2):
                    raise AssertionError(
                        "the driver returned no view_p1/view_p2 — this gate is comparing "
                        "against nothing. Is POKESIM_SEARCH_DRIVER_BIN pointing at a binary "
                        "built from this tree?")
                pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
                prefix_actions = [int(x) for x in actions[:anchor]]

                # ROOT — the real decision point. This road is ALSO the FORK: its action list
                # has exactly `anchor` entries, so decision `anchor` is materialized and the
                # tracker is NOT advanced — which is precisely the state every arm branches from,
                # and the state `materialize_branches` snapshots.
                road = _ProtocolRoad(record, side, prefix_actions)
                road.feed(pfx)
                check_point(encoder, road, root.view_p1 if side == "p1" else root.view_p2,
                            f"{record.battle_tag}@t{turn}/root", cen)
                factory = (ViewSuccessorFactory.at_fork(
                    road.player._get_tracker(road.battle), road.battle, encoder)
                    if road.battle is not None else None)

                # BRANCH POINTS — one ply forward, `arms` different legal lines.
                # The arms are taken from the REAL mapper's action-index -> choice-string map, so
                # each carries the action INDEX its successor's replay needs. A road fed the
                # branch suffix without that index exhausts its action list AT the branch decision
                # and stops consuming — every arm would then be compared one ply BEHIND its view.
                cmap = _choice_map(record, side, prefix_actions, pfx, anchor)
                if not cmap:
                    cen.defer("no legal arm at the root (wait / ended)")
                    continue
                opp_rec = root.recorded_choices.get(other) or "default"
                picks = sorted(cmap)[:arms]
                expand = [{"node_id": root.node_id, f"{side}_action": cmap[a],
                           f"{other}_action": opp_rec,
                           "seed": f"{turn},{k + 1},{anchor + 7},{k * 13 + 11}", "label": a}
                          for k, a in enumerate(picks)]
                # THE CORE ROAD — the same arms, from a Rust-core VERSION root, integrity-checked.
                croot = cs.open_root(turn, core="typed")
                cexp = [dict(a, node_id=croot.node_id) for a in expand]
                core_of = {int(n.label): n for n in cs.expand_many(cexp, side=side, integrity=1)}
                from agents.training.core_successor import CoreSuccessorFactory

                cfactory = (CoreSuccessorFactory.at_fork(
                    road.player._get_tracker(road.battle), road.battle, encoder)
                    if road.battle is not None else None)
                for node in ss.expand_many(expand):
                    if node.ended or node.stuck:
                        cen.defer("arm ended / stuck (no successor board to compare)")
                        continue
                    cnode = core_of.get(int(node.label))
                    if cnode is None or cnode.ended != node.ended:
                        cen.note("core.arm[presence]", node.ended, getattr(cnode, "ended", None),
                                 f"{record.battle_tag}@t{turn}/arm{node.label}")
                        continue
                    ccore = cnode.core_p1 if side == "p1" else cnode.core_p2
                    payload = node.view_p1 if side == "p1" else node.view_p2
                    suffix = node.p1_chunks if side == "p1" else node.p2_chunks
                    # PAD the action list. `_feed` stops the moment the player is done, and a
                    # ply that ends in a FAINT opens a replacement round the port has already
                    # resolved through its follow-up policy — so a road given exactly
                    # `prefix + [a]` stops at the replacement boundary while the view describes
                    # the NEXT turn, and every field then reads one ply apart. The padding is
                    # sound because this gate encodes with NO trackers: an action index feeds the
                    # tracker fold and nothing the read-models or the vector are built from.
                    arm_road = _ProtocolRoad(record, side,
                                             prefix_actions + [int(node.label)] + [0] * 8)
                    arm_road.feed(list(pfx) + list(suffix))
                    branch_points += 1
                    where = f"{record.battle_tag}@t{turn}/arm{node.label}"
                    clean = check_point(encoder, arm_road, payload, where, cen)
                    n_mid = intermediate_decisions(suffix)
                    if n_mid:
                        # 🚨 **THE D10 GATE** (`gen3_view_at_intermediate_v1`). The ply resolved a
                        # REPLACEMENT round inside itself, so `view_pN` is one decision PAST the
                        # row `materialize_branches` returns — and the port now ALSO sends the
                        # board it had AT that round (`view_pN_at`). This compares the pair the
                        # production road actually uses: that board, folded over the chunk cut
                        # `split_at_intermediate` makes, against the protocol road's own row.
                        #
                        # The row index is unchanged (`anchor + 1`): `mats[anchor + 1]` is the
                        # FIRST decision the arm's suffix produced, which IS the replacement
                        # round. The road's action padding lets it run on past that row; it does
                        # not change the row itself, which is encoded when it is appended.
                        at = node.view_p1_at if side == "p1" else node.view_p2_at
                        head = split_at_intermediate(suffix)
                        if not at or not at[0] or head is None:
                            cen.note("successor.view_pN_at[missing]", f"{n_mid} intermediate",
                                     f"{len(at or [])} boards / head={head is not None}", where)
                            continue
                        # 🚨 A SECOND ROAD, stopped AT the cut. `arm_road` above ran PAST the
                        # replacement (that is what its action padding is for), so its board is
                        # the next turn's and its read-models say nothing about the one the D10
                        # path reads. This road is fed the prefix plus the CUT and nothing else,
                        # which is exactly where `materialize_branches` stands when it emits the
                        # row — so it is both the read-model oracle for `view_pN_at[0]` and the
                        # owner of the successor row to compare against.
                        mid_road = _ProtocolRoad(
                            record, side,
                            prefix_actions + [int(node.label)] + [0] * 8)
                        mid_road.feed(list(pfx) + list(head))
                        mid_board = ViewEventFolder.seed_from(road.battle)
                        mid_board.fold(head)
                        clean_mid = check_point(
                            encoder, mid_road, at[0], where + "/D10mid", cen,
                            ledger=mid_board)
                        if cfactory is not None and ccore is not None:
                            if not ccore.get("mid"):
                                cen.note("core.mid", True, False, where)
                            elif check_core(cfactory, encoder, mid_road, mid_road, ccore,
                                            int(node.label), anchor + 1, where + "/core-D10", cen):
                                CORE_POINTS += 1
                                CORE_D10 += 1
                        if clean_mid and factory is not None and check_successor(
                                factory, encoder, mid_road, at[0], head, int(node.label),
                                anchor + 1, where + "/D10", cen):
                            full_obs_points += 1
                            d10_points += 1
                    else:
                        if clean and factory is not None and check_successor(
                                factory, encoder, arm_road, payload, suffix, int(node.label),
                                anchor + 1, where, cen):
                            full_obs_points += 1
                        if cfactory is not None and ccore is not None:
                            if ccore.get("mid"):
                                cen.note("core.mid", False, True, where)
                            elif check_core(cfactory, encoder, arm_road, arm_road, ccore,
                                            int(node.label), anchor + 1, where + "/core", cen):
                                CORE_POINTS += 1
    return cen, branch_points, full_obs_points, d10_points


def _choice_map(record, side, prefix_actions, pfx, anchor) -> Dict[int, str]:
    """``{action index: sim choice string}`` for every LEGAL action at the branch decision,
    produced by the REAL action mapper — the same primitive the search itself uses, so the gate
    can never branch on an action the mask forbids."""
    mt = OM.materialize_decisions(
        list(pfx), username=record.username(side), packed_team=record.packed_team(side),
        side=side, actions=prefix_actions, battle_format=record.format_id,
        battle_tag=record.battle_tag, map_actions_at=anchor, stop_after_decision=anchor)
    return mt.action_choices or {}


def test_the_one_sided_view_reproduces_the_read_models_and_the_obs():
    """The collected gate — one REPRODUCIBLE battle (see the module header), three turns, five
    arms each. Deterministic by construction, so a failure here is a regression and never a draw.
    The stochastic sweep is the script entry point."""
    cen, branch_points, full_obs_points, _d10 = run(n_battles=1, arms=5, turns=3, fixed_key=0)
    print("\n" + cen.render())
    assert cen.compared >= 8, (
        f"only {cen.compared} comparisons ran — the gate is vacuous "
        f"(branch points: {branch_points})")
    assert branch_points >= 8, (
        f"only {branch_points} BRANCH points ran — a root-only run would never exercise a "
        f"successor board, which is the whole point")
    assert full_obs_points >= 8, (
        f"only {full_obs_points} FULL-observation (tracker-fed) comparisons ran — the D5 half of "
        f"this gate is what licenses `--materializer view`, and a run that never reaches it is "
        f"vacuous about every tracker block (branch points: {branch_points})")
    assert CORE_POINTS >= 8, f"only {CORE_POINTS} CORE-road successors compared — vacuous"
    assert not cen.rows, "\n" + cen.render()


def test_an_intermediate_decision_arm_is_served_from_the_ports_own_board():
    """THE D10 GATE (`gen3_view_at_intermediate_v1`) — a collected, REPRODUCIBLE battle whose
    arms include at least one ply that resolved a REPLACEMENT round inside itself.

    🚨 **This is the one comparison the rest of this module cannot make.** Every other branch
    point here has the two roads standing on the same decision; a D10 arm does not, and that is
    precisely why it used to be DEFERRED. The port now sends ``view_pN_at`` — the board it held
    at each decision it answered internally — and the view road serves the arm from
    ``view_pN_at[0]`` folded over :func:`~agents.training.view_successor.split_at_intermediate`'s
    chunk cut. The assertion is the production one: the successor's obs BYTES and its MASK equal
    ``materialize_branches``' own row for the same arm, trackers included.

    **The fixture key is chosen, not lucky.** Keys 0-19 were swept and this one carries D10 arms
    with no read-model residual; a run of it that reports ``d10 == 0`` means the battle changed
    and the gate has gone vacuous, which is why that is an assertion and not a print."""
    cen, branch_points, full_obs_points, d10 = run(n_battles=1, arms=10, turns=3, fixed_key=8)
    print("\n" + cen.render())
    print(f"branch points: {branch_points}  full-obs: {full_obs_points}  D10 arms: {d10}  "
          f"core: {CORE_POINTS} ({CORE_D10} D10)")
    assert d10 >= 1, (
        f"NO D10 arm was served on the fixture battle ({branch_points} branch points, "
        f"{full_obs_points} full-obs) — this gate is VACUOUS about the intermediate-decision "
        f"path, which is the only thing it exists to hold. Re-sweep for a fixture key whose "
        f"arms KO one of our mons mid-ply.")
    assert full_obs_points >= 8, (
        f"only {full_obs_points} tracker-fed comparisons ran (branch points: {branch_points})")
    assert CORE_D10 >= 1, (
        f"the CORE road served NO D10 leaf on the fixture battle ({CORE_POINTS} core points) — "
        f"its intermediate-decision version is untested")
    assert not cen.rows, "\n" + cen.render()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_battles", nargs="?", type=int, default=2)
    ap.add_argument("--arms", type=int, default=DEFAULT_ARMS)
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--fixed-key", type=int, default=None,
                    help="use the REPRODUCIBLE fixture battle(s) from this key instead of fresh "
                         "random ones — what the collected test runs")
    a = ap.parse_args()
    census, bp, fop, d10 = run(a.n_battles, a.arms, a.turns, a.impl, a.fixed_key)
    print(census.render())
    print(f"branch points compared: {bp}   (FULL tracker-fed obs at {fop} of them; "
          f"{d10} of those were D10 INTERMEDIATE arms served from view_pN_at[0]); "
          f"CORE road: {CORE_POINTS} successors ({CORE_D10} D10), integrity-checked")
    sys.exit(1 if census.rows else 0)
