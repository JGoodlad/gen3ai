# Design: Event-Sourced Battle (`Gen3Battle`)

**Status:** **Steps 1–4 of §9 SHIPPED to main; Steps 5–6 remain.** Scope: Gen 3 OU
**singles** only. This is a foundation piece (training, replay analysis, and v5 MCTS all
benefit). The original sequencing held; the live-state observation was enriched well
beyond the original "drop-in" framing (see Implementation Status below). The detailed
execution handoff for the remaining work lives in
`handoff_turn_delta_reward_replay.md`.

---

## Implementation Status (updated 2026-05-30)

The migration was executed in shippable increments, each fuzz-gated. What follows is the
record of what was built vs. what this doc originally proposed; the design body below is
the original proposal, kept for context.

### Shipped to `main`

| Commit | Step(s) | What landed |
|---|---|---|
| `3e4de10` | §9.1–§9.2 | Event-sourced battle layer: `Gen3Battle(Battle)` (`src/agents/battle/`), the `BattleEvent`/`EventKind` schema, the `MESSAGE_POLICY` completeness registry with `assert_conservation()`, and `TurnView` (the per-turn fold). |
| `c7ecc8d` | §9.3–§9.4 | `LiveView`/`LiveSide`/`LivePokemon` current-board read-model + `Gen3Battle` injection wired into training/eval/replay (`battle_class=Gen3Battle` default on `Gen3Player`/`Gen3Env`). Added the **per-decision window API** `event_cursor` / `events_since(cursor)`. |
| `786127a` | §9.4 (extended) | **Full live-state observation** (`ARCH_SIGNATURE = gen3_live_state_v1`, obs dim 2734 → 2823). |

### What changed vs. the original proposal

1. **`Gen3Battle` is an additive subclass, not a `parse_message` handler refactor.**
   §2/§9.1 proposed refactoring `AbstractBattle.parse_message` into per-type handlers.
   Instead `Gen3Battle.parse_message` wraps `super()` (state tracking stays **verbatim**
   poke-env) and records events around it. Same goal — one parser, two views,
   pre-mutation attribution — with far less churn to the fork.

2. **The observation became enrich-not-just-drop-in (a retrain, by design choice).** The
   original framing kept obs values identical (shape-preserving). We instead **expanded**
   the live-state obs because it was the higher-value move:
   - **Active context 23 → 55:** the volatile block went from a hand-picked 9 to the
     **full source-derived gen3 set** (`VOLATILE_DIM = 41`, `gen3_effects.py`),
     recovering ~30 silently-dropped volatiles (Disable/Encore/Taunt/Destiny Bond/Curse/
     Yawn/Flash Fire/partial-trap/…), with perish/stockpile **counters normalised**.
   - **Global env 13 → 18:** weather is **event-sourced** with cause-aware permanence +
     turns-remaining (ability weather permanent, move weather 5-turn countdown — read
     from the `|-weather|…|[from] ability:` cause, never guessed); dead gen4+ weather
     slot dropped; per-side **Safeguard + Mist** added.
   - This bumped `ARCH_SIGNATURE` to `gen3_live_state_v1` — a deliberate retrain.

3. **Crash-don't-drop is a load-bearing invariant, realised in two places.** §4's
   "no silent drops" for the event log is mirrored in the obs encoders: `encode_volatiles`
   / `normalize_cant_reason` (`gen3_effects.py`) **raise** on any volatile / `|cant|`
   reason not in the source-derived allowlist. This tripwire caught `focuspunch`,
   `struggle`, and `flashfire` as real gen3 volatiles during bring-up (the last only in
   the live training smoke). Allowlists are re-derived from Showdown `moves.ts` **and**
   `abilities.ts` ∩ the gen3 sets ∩ poke-env's `Effect` enum, checked by tests.

4. **`MoveSource` / heuristic retirement is deferred to Step 4 (the TurnDelta fold), not
   done yet** — see below.

### Remaining work

Tracked in detail in **`handoff_turn_delta_reward_replay.md`**. Summary:

- **Step 4 — `TurnDelta` fold (retrain-class, the §6 work).** Fold `TurnDelta` from the
  event log via `TurnView` over the `events_since` per-decision window (NOT
  `events_for_turn`, which slices by protocol turn). Four requirements:
  (1) **multi-KO + cause** — represent that several mons fainted in one window and *why*
  (attack / hazard / weather / status / recoil / self-KO / Leech Seed); the cause exists
  **only** in the event log, which is what forces the fold. (2) **phase** start-of-attack
  vs extended/forced-switch continuation (`phase_is_forced_switch` already present).
  (3) **attempted-action** capture — what we tried to pick, preserved even when it never
  fired (flinch/frozen/cant). (4) **missing → 0**. This also wires the `CANT_DIM = 12`
  one-hot (`gen3_effects.encode_cant_reason`) into the history block and **retires** the
  diff heuristics (`_ko_before_acting`, `_align_effectiveness`, `opp_all_last_move_ids`
  phaze recovery). Gated by the §8.3 equivalence harness + the event-log fuzz.
- **Step 5 — reward manager** onto `TurnView` + `LiveView` (terminal win/loss stays on
  the battle object). Retrain to measure (per project memory).
- **Step 6 — replay recorder** lossless: record move ids, opponent moves, hit/miss/crit,
  and faint→switch chains-with-cause from the event log.

---

## 0. The problem, restated

poke-env is a **state tracker**: every protocol line overwrites "current board"
fields (`active_pokemon`, `last_move`, HP, boosts…). That is the right design for a
bot that only needs to *pick a move*. But RL, reward shaping, and replay analysis
need the opposite — **what happened, in order, between two states**. We have been
reconstructing that event stream by diffing snapshots and recovering facts the
snapshot dropped (`opp_all_last_move_ids`, the `DamagingMoveEvent` faint-recovery,
phaze special-cases, `_ko_before_acting`). The `MoveSource` provenance enum added in
the previous step is, honestly, a **band-aid that names how lossy that approach is**.

The fix is to capture the events **once, at parse time, before any state mutation**,
in the order they were revealed — and let `TurnDelta`, the encoder, and analysis read
that log instead of re-deriving it.

## 1. Core principle — enrich, don't reimplement

Two **views**, one **parser**, one **object** everyone already reads:

- **Current-state view** — keep poke-env's battle engine verbatim. Deriving HP /
  status / boosts / items / abilities / hazards correctly is 730 hard lines we do not
  want to reinvent or fork away from upstream.
- **Revealed-order event log** — new, captured in the *same* parse pass.

The object is the battle. Because every consumer (encoder, `BattleContext`, action
mask, replay recorder) reads the battle and never the env, enriching the battle
propagates everywhere for free.

> **Non-goals (explicit):** no second parser; no clean-room event-sourced engine that
> re-derives state from events; no replacement of current-state (the encoder needs
> "HP now"). The log is **additive**.

## 2. Architecture

```
AbstractBattle            (fork base — state engine + dispatch, refactored into handlers)
   └── Battle              (classic singles — unchanged behavior, upstream-clean)
         └── Gen3Battle    (event-aware: current state + revealed-order event log + knowledge model)
```

- `Gen3Battle` adds `self._events: list[BattleEvent]` (whole-battle, monotonic
  `seq`), `self._knowledge: BattleKnowledge`, and per-turn slicing.
- **Injection seam:** poke-env's `Player` constructs battles internally. Add a
  `battle_class` parameter (default `Battle`) threaded to where battles are created
  (e.g. `_create_battle`). Our players pass `battle_class=Gen3Battle`. One seam; the
  env is untouched.
- **Clearing cadence:** the event log is NOT cleared per turn (it is the full
  battle timeline); `end_turn()` records a `TurnBoundary(turn)` marker so per-turn
  slices are O(1). Per-turn scalar state poke-env clears today (`_this_turn_move_sides`,
  `_current_move_user_side`) is **subsumed** by the log and retired.

## 3. The `BattleEvent` schema

A small, immutable, ordered record. Attribution (`side`, `actor_species`,
`target_species`) is filled **at parse time, before** the line mutates state — this is
the whole point, and why gap=0 can't corrupt it.

```python
class EventKind(IntEnum):
    MOVE = 1; SWITCH = 2; DRAG = 3; FAINT = 4
    DAMAGE = 5; HEAL = 6; BOOST = 7; UNBOOST = 8; SETBOOST = 9; CLEARBOOST = 10
    STATUS = 11; CURESTATUS = 12; CANT = 13
    CRIT = 14; MISS = 15; FAIL = 16
    IMMUNE = 17; RESISTED = 18; SUPEREFFECTIVE = 19
    ITEM = 20; ENDITEM = 21; ABILITY = 22
    WEATHER = 23; FIELD = 24; SIDE = 25            # hazards / screens
    VOLATILE_START = 26; VOLATILE_END = 27         # Substitute, Leech Seed, …
    PREPARE = 28; MUSTRECHARGE = 29; TRANSFORM = 30; FORMECHANGE = 31; SWAP = 32
    TURN_BOUNDARY = 90
    UNKNOWN = 99                                   # never silently dropped — see §4

@dataclass(frozen=True)
class BattleEvent:
    seq: int                       # monotonic across the whole battle
    turn: int                      # game turn this line belongs to
    kind: EventKind
    side: Optional[str]            # "ours" / "opp" / None, resolved at parse time
    actor_species: Optional[str]   # the mon that acted/was affected (pre-mutation)
    target_species: Optional[str]
    value: dict                    # kind-specific payload (move_id, amount, stat, reason, item, …)
    raw: tuple                     # the original split_message — provenance + debugging
```

Design rules:
- **Self-contained.** An event carries enough to be interpreted without re-reading
  battle state (e.g. `MOVE` carries `move_id`, `delegated_from` for Sleep Talk, the
  resolved `target_species`).
- **Order is meaning.** `seq` is the revealed order; consumers that care about
  "who moved first" read order, not a derived flag.
- **Effectiveness/outcome attach to the owning `MOVE`.** `CRIT`/`MISS`/`FAIL`/
  `SUPEREFFECTIVE`/… events carry `side` = the currently-resolving mover (the existing
  `_current_move_user_side` logic), so folding is trivial.

## 4. Completeness — every battle-relevant line hits us, nothing skipped

This is a first-class requirement: **no silent drops.** Today `MESSAGES_TO_IGNORE` is
a silent sink and the final `else` raises `NotImplementedError`. We replace that with a
**classified, audited policy** so every line is accounted for.

### 4.1 Message policy registry

```python
class Policy(IntEnum):
    EVENT = 1        # recorded as a BattleEvent (may also mutate state)
    STATE_ONLY = 2   # mutates state, intentionally not an event (e.g. |-sethp| correction)
    CONTROL = 3      # protocol control, not battle content (|request| |upkeep| |turn| |t:| …)
    COSMETIC = 4     # explicitly irrelevant, with a reason (|-anim|-style, |c| chat, |inactive|)
    UNSUPPORTED = 5  # a real protocol line that CANNOT occur in gen3ou — raise if ever seen

MESSAGE_POLICY: dict[str, tuple[Policy, str]] = {
    "move": (Policy.EVENT, "move use"),
    "switch": (Policy.EVENT, "switch in"),
    "-crit": (Policy.EVENT, "crit on current mover"),
    "upkeep": (Policy.CONTROL, "end-of-turn control"),
    "c": (Policy.COSMETIC, "chat"),
    # Mechanics that don't exist in Gen 3 — classified, and fatal if they appear:
    "-mega": (Policy.UNSUPPORTED, "Mega Evolution — Gen 6+"),
    "-zpower": (Policy.UNSUPPORTED, "Z-move — Gen 7"),
    "-terastallize": (Policy.UNSUPPORTED, "Terastallization — Gen 9"),
    # … one entry per protocol message type, each with a human reason …
}
```

### 4.2 The dispatcher guarantee

Every parsed line: look up `MESSAGE_POLICY[event[1]]`.

- `EVENT` → append to the log (and mutate state). `STATE_ONLY`/`CONTROL`/`COSMETIC` →
  counted, no event.
- `UNSUPPORTED` → **raise `UnsupportedMessageType`.** This is *correct, by design*, not
  graceful degradation. gen3ou's protocol surface is **fixed and finite**: a `-mega` /
  `-zpower` / `-terastallize` / doubles-only line can never legitimately occur. If one
  does, we're parsing the wrong format and any "state" we'd produce is garbage — so we
  stop loudly rather than fabricate. (Contrast the *forensic* `intent_comparable` check,
  which is non-fatal: that one has an irreducible benign-mismatch surface; this one does
  not — an unsupported line is a hard, deterministic "we are in the wrong game.")
- **Missing entirely (unclassified) → also raise** `UnknownMessageType`. A line we've
  never classified means our coverage is stale; fail so the §4.4 audit forces us to
  classify it (as `EVENT`, `COSMETIC`, or `UNSUPPORTED`) before it can ship.

Because `UNSUPPORTED`/`UNKNOWN` raise, they should be **0 across the entire fuzz
corpus** — they are tripwires, not a runtime path. The `UNKNOWN` `EventKind` exists only
as a defensive sentinel for tooling that chooses to catch rather than crash (e.g. an
offline replay scrubber over arbitrary-format logs); the live gen3ou pipeline never
produces one.

### 4.3 The conservation invariant (the real proof)

For every battle, assert:

```
len(_replay_data)  ==  events_recorded  +  state_only  +  cosmetic_skipped  +  control_handled
```

i.e. **every raw line is in exactly one bucket.** `UNSUPPORTED`/`UNKNOWN` never appear
in the sum — they raise (§4.2), so a balanced battle has zero of them. If the sum
doesn't balance, a line fell through silently; if a battle raises, we hit an
unsupported/unclassified type. Either way we *prove* — not hope — that nothing
battle-relevant is skipped.

### 4.4 Registry audit test

A unit test enumerates `MESSAGE_POLICY` against the **full Gen 3 protocol message
list** (from the Showdown sim docs + the observed corpus). Any protocol type not in the
registry → test fails. A registry entry of policy `EVENT`/`STATE_ONLY`/`CONTROL`/
`COSMETIC` never seen across the corpus → reported (don't carry dead/guessed entries);
`UNSUPPORTED` entries are *expected* to be unseen — they are deliberate tripwires for
non-Gen-3 mechanics, so absence is success, not dead code.

## 5. Team knowledge — ground truth vs revealed (make this separation explicit)

There are **two orthogonal axes**, and conflating them is how info-leak bugs happen:

| Axis | Question | Source |
|---|---|---|
| **Ground truth** | What does this mon *actually* have? | a-priori (we built the team / a replay team sheet) or never |
| **Revealed** | What has the protocol *disclosed* so far, and when? | always, from the event log |

### 5.1 The knowledge model

```python
@dataclass
class Fact:                       # one attribute of one mon
    value: Any                    # ground truth if seeded, else the revealed value
    revealed: bool                # has the protocol disclosed it?
    revealed_seq: Optional[int]   # WHEN (event seq) — gives the reveal timeline
    source: Literal["sheet", "request", "protocol", "unknown"]

class MonKnowledge:   # species, item, ability, 4×move, stats/spread, status, …
    ...
class BattleKnowledge:
    def revealed_view(self, side, as_of_seq=None) -> SideView: ...  # what was KNOWN at a point
    def ground_truth(self, side) -> SideView: ...                   # full truth (analysis only)
```

Every `Fact` independently tracks **the value** and **whether/when it was revealed**.
So we *always* hold both "the truth" (if we have it) and "what a fair observer knew."

### 5.2 Two seeding modes (orthogonal to reveal tracking)

- **Omniscient seed** — the full team is provided up front:
  - **our side, live:** poke-env already gets our complete side from the `|request|`
    JSON (all moves, stats, item, ability). That request IS the omniscient seed for our
    side — formalize it as `source="request"`.
  - **replay / analysis:** seed from the saved team sheet (`source="sheet"`). Works for
    self-play replays and any battle where we hold the teams.
  - In omniscient mode, `value` is ground truth from t0, but `revealed` still starts
    **False** for opponent-facing attributes and flips only when the protocol discloses
    them. We know the truth *and* the fog.
- **Fog seed** — no a-priori team (live ladder opponent). `MonKnowledge` starts empty;
  facts fill in as `EVENT`s reveal them (`source="protocol"`).

### 5.3 The non-cheating invariant (load-bearing)

> The observation the model sees for the **opponent** must be built from
> `revealed_view(opp, as_of=now)` — **never** ground truth — even when we have the team
> sheet. Otherwise the policy learns to exploit hidden information and is useless on the
> real ladder.

- **Training:** our side `revealed_view(ours)` is fully known (we built it →
  `spread_known=1`, all move/item/ability `known=1`); opponent `revealed_view(opp)` is
  reveal-gated (`spread_known=0`, attributes `known` only after disclosure). This is
  *exactly* today's `species_known` / `spread_known` / item-`known` / ability-`known`
  flags — the knowledge model is their single, principled home.
- **Replay analysis:** drive the encoder from `revealed_view(side, as_of=turn_k)` to
  reproduce *what the player could legitimately see at turn k*; use `ground_truth` only
  for labels/counterfactuals (e.g. perfect-info value targets for MCTS), clearly on a
  separate code path the policy never reads.

`BattleContext.from_battle` becomes the **single adapter**: it reads
`battle.revealed_view(...)`, so the downstream encoder is unchanged and the gate is
enforced in exactly one place.

## 6. `TurnDelta.build` on the event log

`build` stops being a detective and becomes a **fold over one turn's events**:

```python
events = battle.events_for_turn(t)                 # O(1) slice via TURN_BOUNDARY markers
our_move  = first(e for e in events if e.kind==MOVE and e.side=="ours")
opp_move  = first(e for e in events if e.kind==MOVE and e.side=="opp")
our_faint = any(e.kind==FAINT and e.side=="ours" for e in events)
order     = [e.side for e in events if e.kind==MOVE]      # who moved first = order[0]
# outcome attaches to the owning MOVE; switches/cant/KO-before-acting are just event presence/absence
```

- **What's gone:** `opp_all_last_move_ids`, `_our/opp_last_damaging_move` recovery,
  `_ko_before_acting`, `_align_effectiveness`, the phaze special-case, and most of the
  `MoveSource` enum (everything is `PROTOCOL` now — the log *is* the protocol).
- **What survives:** `intent_comparable`. The *action we pressed* is still not in the
  protocol, so reconciling our intent against the fired move stays — but it's trivial
  now (the log states exactly which of our moves fired).
- **HP-after** still comes from the current-state snapshot (the encoder wants absolute
  "HP now"); **HP deltas/attribution** come from summing `DAMAGE`/`HEAL` events on the
  named target — cleaner than slot-diffing and immune to newly-revealed-slot artifacts.

## 7. Fuzzing — the verification spine

The fuzz is how this earns trust. Five complementary fuzzers, all over a shared corpus
(see §7.6). Each runs real battles and validates against the **raw protocol** it
intercepts via the existing `_handle_battle_message` pattern.

### 7.1 Completeness fuzz
Assert §4.3 conservation per battle (buckets sum to `len(_replay_data)`; no battle
raised `UnsupportedMessageType`/`UnknownMessageType`) and §4.4 (every observed type is
classified). **Fails on the first unclassified, unsupported, or silently-dropped line** —
so a future format/keyword change can't slip through unnoticed.

### 7.2 State-equivalence fuzz (behavior-preserving)
Feed the identical protocol stream to classic `Battle` and `Gen3Battle` line-by-line;
after **every** line assert current-state parity — HP, status, boosts, active, team,
side conditions — `max|Δ| = 0`. Proves the enrichment changed nothing about state.

### 7.3 Event-vs-protocol fuzz (the core one — upgrade the existing tests)
The current `transition_fuzz` / `move_outcome_fuzz` intercept raw lines and re-derive
ground truth independently. Re-point them at the **event log**: for each turn assert the
log == the independent raw re-derivation (move used, by whom, target, order, outcome,
crit/miss/fail, faints, switches). This is strictly stronger than today and replaces the
scalar-by-scalar checks. Coverage asserted: every `EventKind` observed ≥ once across the
corpus (engineered teams force the rare ones).

### 7.4 Knowledge / no-leak fuzz
- `revealed_view(opp)` **never** contains an un-revealed opponent fact (item before
  `|-enditem|`/`|-item|`, a move before it's used, ability before it triggers).
- Omniscient-seeded battles: opponent attrs are `revealed=False` until the disclosing
  event, then flip with the correct `revealed_seq`; our own team is fully `known` from
  t0.
- `known_as_of(seq)` is **monotonic** (knowledge only grows forward).

### 7.5 Replay determinism fuzz
Parse a saved replay twice → identical event logs (deterministic, no `Date/random`
contamination). Parse forward and snapshot `revealed_view` at each turn → reproduces the
fog timeline; seed the same replay omniscient and assert ground truth matches the
eventual reveals.

### 7.6 Corpus
- Live random `gen3ou` battles (volume).
- The downloaded sample-team replays under `data/teams` (realistic play).
- Engineered edge teams (as today's `VARIANCE`/`EDGE`): Explosion/Self-Destruct, Roar/
  Whirlwind (phaze), Rest/Sleep Talk, Baton Pass, Trick, Knock Off, Substitute,
  Sand/hazards, Truant/recharge — to force every `EventKind` and reveal path.

## 8. Testing strategy — unit, integration, and model integration

### 8.1 Unit (no server)
- **Schema:** `seq` monotonic; `events_for_turn` slicing; frozen/immutability.
- **Per-handler table tests:** for each message type, feed a hand-built
  `split_message`, assert (a) the `BattleEvent` emitted (kind, side, actor/target,
  payload) and (b) the state mutation — parametrized across all `Policy.EVENT` types.
- **Completeness:** a test over `MESSAGE_POLICY` vs the protocol type list (§4.4); a
  test that an unknown type raises `UnknownMessageType` in strict mode.
- **Knowledge:** omniscient vs fog seeding; `revealed` flips on the right event;
  `revealed_view` hides unrevealed opp facts; our team known at t0.
- **`TurnDelta.build` fold:** migrate the existing `move_attribution_test`
  cases to feed event logs instead of mock `BattleContext`s — same canonical cases
  (explosion self-faint, KO-before-acting, phaze, Sleep Talk delegation, switch-death),
  now asserting the fold output.

### 8.2 Integration (`*_integration_test.py`, Node bridge, no live server)
- Run scripted battles through `Gen3Battle`; assert §7.2 state-equivalence vs classic
  `Battle` and a well-formed, conservation-balanced event log.

### 8.3 Model integration (the crux — "does it slot into the model")
This is where we prove the event-sourced battle is a **drop-in** that the model can't
tell apart, and that it doesn't leak.

- **Obs parity / drop-in proof.** For a battery of saved battle states, build the obs
  from `Gen3Battle` (via `BattleContext.from_battle` → `revealed_view`) and from the
  current classic path, and assert the **2754-dim vectors are byte-identical** outside
  the known gap=0 corners. A small, *whitelisted* set of corner turns is allowed to
  differ — and there the event-sourced value is asserted to be the *more correct* one
  (validated against raw protocol). This both guarantees "model sees the same thing"
  and **quantifies the churn** (how many turns differ → informs whether a retrain is
  worth it). Obs **shape** is unchanged, so existing checkpoints load either way.
- **Equivalence harness.** Run M turns through BOTH pipelines (classic `Battle` +
  snapshot-diff `TurnDelta`) and (new `Gen3Battle` + event-fold `TurnDelta`); diff the
  resulting obs *and* the encoded turn-history block. Same whitelist discipline.
- **No-leak, end-to-end.** Construct a battle where the opponent holds a hidden item
  and an unused move; assert the obs vector the **model actually receives** marks them
  `known=0` (item/move/spread flags), proving `revealed_view` gates correctly through
  the full encode path — not just in a unit.
- **Forward-pass smoke.** Feed `Gen3Battle`-derived obs into the dual-head model;
  assert `(pi, vf)` shapes and finiteness (no NaN/inf from a malformed slice).
- **Pipeline smoke.** Run `play.py` / the replay recorder against `Gen3Battle`: eval
  win-rates still compute; `battle_N_summary.json` + `states.npz` still emit; the
  `ReplayCallback` path (the one that crashed before) runs clean.
- **Round-trip.** `describe_vector(encode(Gen3Battle))` decodes back to a battle state
  matching `revealed_view` (species/HP/status), closing the encode↔decode loop.

### 8.4 The retrain question, answered by tests
Because obs shape is identical, **checkpoints load unchanged**; values differ only in
the gap=0 corners (strictly more accurate). The equivalence harness (§8.3) *measures*
the diff rate, so the decision to retrain is data-driven, not a guess. Per project
memory, any reward/obs-affecting change is verified by a short retrain comparison before
relying on the improvement.

## 9. Migration sequencing (each step independently verifiable)

1. **Refactor `parse_message`** into per-type handler methods — behavior-preserving;
   gate on §7.2 state-equivalence (`max|Δ|=0`) and the full unit suite. (This alone
   makes the 730-line monster legible.)
2. **Add the event log** + `BattleEvent` schema + `MESSAGE_POLICY` registry + the §4.3
   conservation guard + §7.1 completeness fuzz. No consumer changes yet.
3. **Add `battle_class` injection**; wire `Gen3Battle` into the training/play players.
4. **Add `BattleKnowledge`** (omniscient/fog) + `revealed_view`; route
   `BattleContext.from_battle` through it; gate on §8.3 obs parity + no-leak.
5. **Migrate `TurnDelta.build`** to fold the event log; retire the recovery/provenance
   scaffolding (§10) behind the §8.3 equivalence harness.
6. **(Optional) retrain** to benefit from the cleaner corner data, measured by §8.4.

Steps 1–4 are obs- and reward-neutral (drop-in). Step 5 is the only one that can change
delta values; it carries the equivalence harness as its gate.

## 10. What this retires

`opp_all_last_move_ids`, `_our/opp_last_damaging_move` + pending/promote machinery,
`_ko_before_acting`, `_align_effectiveness`, the phaze recovery branch, the per-turn
scalar captures (`_we_moved_first`, `_this_turn_move_sides`, `_our/opp_move_crit/missed/
failed`), and most of `MoveSource` (kept only if `intent_comparable` still finds the
ACTION-vs-protocol distinction useful — likely collapses to a boolean). The gap=0
reasoning in `TurnDelta.build` disappears: the log is captured pre-mutation, so the
desync it defends against can no longer occur.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hot-path refactor breaks state | §7.2 state-equivalence fuzz (`max|Δ|=0`) + full unit suite, step 1 gates everything |
| **Info leak** (opp ground truth reaches the model) | single `revealed_view` adapter in `from_battle`; dedicated §7.4 + §8.3 no-leak tests; ground truth on a separate analysis-only path |
| Silent line drop | §4.3 conservation invariant as a standing fuzz assertion; unknown types fail tests |
| Upstream-merge friction | keep state logic in base `Battle`; event emission in `Gen3Battle` / thin hooks |
| Obs churn forces a surprise retrain | §8.3 measures the diff rate up front; shape unchanged so checkpoints always load |
| Per-battle log memory | events are small + flat; cap or stream for very long games; per-turn O(1) slices |
| Doubles / non-gen3 formats | explicitly out of scope; `Gen3Battle` guards on singles |

## 12. Open questions

- Should `revealed_view` be **materialized** per turn (simpler consumers, more memory)
  or **computed on demand** from the log (less memory, more compute)? Lean materialized
  for the live encoder, on-demand for replay scrubbing.
- Do we keep `MoveSource` as a thin `PROTOCOL | ACTION` boolean for `intent_comparable`,
  or fold it away entirely? Decide during step 5 once the fold is in.
- Hidden Power typing and spread inference: the knowledge model is the natural home for
  "revealed HP type" / "inferred EVs" — fold the existing `HiddenPowerTracker` and
  spread-inference work into `BattleKnowledge` rather than leaving them parallel.
