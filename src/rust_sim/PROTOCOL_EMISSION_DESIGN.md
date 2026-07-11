# Protocol Emission — Design (level-2)

The design for the **NEXT, serialized** step: hooking the real battle loop to emit the
byte-identical `|...|` protocol stream our poke-env fork parses, so the Rust port is a
drop-in behind the existing bridge (`src/utils/bridge/local_sim_bridge.js`).

**Status: BUILT — historical design record.** Emission Phases 1+2+3 are live and byte-gated
(`tests/protocol_test.rs`: 114 battles / 16115 lines byte-equal, `gen3_protocol_phase3_v1`),
and the endgame `BattleStream::write_line` streaming surface is built + per-write byte-gated
(`tests/writeline_test.rs` vs `harness/gen_writeline_capture.js`, `gen3_writeline_stream_v1`).
The as-built record lives in `CLAUDE.md` → "Protocol emission"; this doc is the original plan
(kept for the rationale — the phase numbering below is the PLAN's, not the shipped phase names).
It is **observation-only** — it must NOT change PRNG consumption (see §e).

**The capture golden** it targets: `tests/vectors/protocol_capture_golden.txt`
(from `harness/gen_protocol_capture.js`) — the raw omniscient stream, verbatim.
**The line-type spec:** `tests/vectors/protocol_inventory.md`.

---

## (a) The `ProtocolBuilder` / emit API (`protocol.rs`)

`protocol.rs` today has the line/choice TYPES (`ProtocolLine`, `Player`, `Choice`) and
the contract note (raw bytes = source of truth). Add a small **append-only builder** the
engine writes lines into during a turn, plus typed constructors for each line type so the
fiddly formatting (HP fractions, tags) lives in ONE place, not scattered through
`turn.rs`.

```rust
/// Accumulates the omniscient protocol stream as the engine runs. The engine holds
/// ONE of these on `BattleState` (a new `log: ProtocolBuilder` field) and pushes lines
/// at the hook points below. `drain()` hands the accumulated lines to the caller
/// (BattleStream::write_line / the bridge) after each committed decision.
pub struct ProtocolBuilder {
    lines: Vec<ProtocolLine>,
}

impl ProtocolBuilder {
    pub fn new() -> Self { Self { lines: Vec::new() } }

    /// Push a raw, already-formatted line (the escape hatch; prefer the typed helpers).
    pub fn push_raw(&mut self, line: impl Into<String>) { … }

    /// Drain everything emitted so far (the per-decision batch the bridge relays).
    pub fn drain(&mut self) -> Vec<ProtocolLine> { std::mem::take(&mut self.lines).into… }

    // ── Typed constructors — the SINGLE home for each line's grammar ──
    // Each returns nothing and pushes one line. HP formatting + tags are centralized.
    pub fn separator(&mut self);                                   // "|"
    pub fn turn(&mut self, n: u32);                                // "|turn|N"
    pub fn upkeep(&mut self);                                      // "|upkeep"
    pub fn r#move(&mut self, user: MonRef, name: &str, target: Option<MonRef>, tags: &[Tag]);
    pub fn switch(&mut self, mon: MonRef, details: &str, hp: HpStatus);   // "|switch|…"
    pub fn drag(&mut self, mon: MonRef, details: &str, hp: HpStatus);     // "|drag|…"
    pub fn damage(&mut self, mon: MonRef, hp: HpStatus, from: Option<Cause>);
    pub fn heal(&mut self, mon: MonRef, hp: HpStatus, from: Option<Cause>);
    pub fn faint(&mut self, mon: MonRef);
    pub fn status(&mut self, mon: MonRef, status: Status, from: Option<Cause>);
    pub fn curestatus(&mut self, mon: MonRef, status: Status, msg: bool);
    pub fn boost(&mut self, mon: MonRef, stat: BoostStat, amount: i8); // -boost / -unboost by sign
    pub fn crit(&mut self, mon: MonRef);
    pub fn supereffective(&mut self, mon: MonRef);
    pub fn resisted(&mut self, mon: MonRef);
    pub fn immune(&mut self, mon: MonRef, from: Option<Cause>);
    pub fn miss(&mut self, user: MonRef, target: Option<MonRef>);
    pub fn fail(&mut self, mon: MonRef, detail: Option<&str>, weak: bool);
    pub fn weather(&mut self, w: Weather, from: Option<Cause>, upkeep: bool);
    pub fn sidestart(&mut self, side: SideRef, effect: &str);
    pub fn start(&mut self, mon: MonRef, effect: &str);
    pub fn end(&mut self, mon: MonRef, effect: &str);
    pub fn activate(&mut self, mon: MonRef, effect: &str, detail: Option<&str>);
    pub fn singleturn(&mut self, mon: MonRef, effect: &str);
    pub fn ability(&mut self, mon: MonRef, ability: &str, detail: Option<&str>);
    pub fn cant(&mut self, mon: MonRef, reason: &str, move_name: Option<&str>);
    pub fn win(&mut self, player_name: &str);   // player-layer (bridge relays)
    pub fn tie(&mut self);
    // init framing (emitted once at start_with_switchins): player/gen/tier/rule/teamsize/start/gametype
}
```

Supporting formatting types (the ONE place the fiddly rules live — see the inventory):

- **`MonRef`** → renders `p<N><pos>: <Nickname>` (gen-3 singles: `pos` always `a`). A
  `SideRef` → `p<N>: <PlayerName>` (side conditions, no position letter).
- **`HpStatus`** → renders the HP field with the three variants from the inventory:
  `x/y` (healthy), `x/y <status>` (space-appended status token), `0 fnt` (fainted). This
  is the #1 correctness point — one `impl Display` computed from `MonState.hp`/`maxhp`/
  `status`, used by every HP-bearing line.
- **`Cause`** → renders `[from] item: <Item>` / `[from] ability: <A>` / `[from] move:
  <M>` / `[from] <bareField>` (e.g. `Sandstorm`, `psn`), and optional `[of] <MonRef>`.
- **`Tag`** → `[miss]` / `[still]` (with the EMPTY target field) / `[msg]` / `[damage]` /
  `[weak]` / `[upkeep]` / `[silent]`.

Display-name resolution: `|move|`/`|-ability|`/`|-start|` use the **Title-Case spaced**
name (Rock Slide, Sand Stream, Substitute), NOT the id. Add a `Dex::move_display_name` /
`ability_display_name` / `item_display_name` accessor (the dex already carries `name`
fields — `SpeciesData.name`, `MoveData` name, etc.).

**Why a builder, not returning `Vec` from each fn:** the emission is interleaved across
deeply-nested calls (a single move fires move→damage→crit→secondary→faint→heal lines
from different functions). A `&mut ProtocolBuilder` threaded on `BattleState` (or passed
down) is far cleaner than plumbing return values up. Put it on `BattleState` as a field
so `run_move`/`run_residuals`/etc. reach it via `self.log`.

---

## (b) Where the emit hooks go (the real functions, by name)

The engine functions already run every event in Showdown's exact order; emission is a
line pushed at each observable step. Below, each hook references the REAL function
(`src/turn.rs` unless noted) and the inventory line it emits. **This section is
documentation only — do NOT edit `turn.rs` in this step.**

### Battle-init framing — `state::BattleState::start_with_switchins` + `event::run_start_switchins`
Emit once, at construction, in the sim's order (see the golden's first ~14 lines):
`|t:|` → `|gametype|singles` → `|player|p1|…` → `|player|p2|…` → `|gen|3` → `|tier|…` →
`|rule|…` → `|` → `|t:|` → `|teamsize|p1|N` → `|teamsize|p2|N` → `|start` →
`|switch|` (each lead, via the switch-in) → the switch-in ability lines → `|turn|1`.
- `event::run_start_switchins` fires each lead's ability `Start`
  (`single_event_ability_start` → `intimidate_on_start`): emit `|-ability|…|Intimidate|
  boost` + `|-unboost|<foe>|atk|1`, and `|-weather|Sandstorm|[from] ability: Sand Stream|
  [of] <mon>` for the weather setters.

### Move execution — `run_move` (turn.rs:1226)
The core per-move bracket. In order:
- On use: `|move|<user>|<Name>|<target>` (+ `[still]` with empty target for a self/no-
  observable move; + `[miss]` on an accuracy fail — paired with `|-miss|`).
- `on_before_move` (turn.rs:2096) aborts → emit `|cant|<mon>|<par|slp|frz|flinch>` (the
  reason), and for confusion the `|-activate|<mon>|confusion` + the self-hit `|-damage|`.
- Immune short-circuit (`move_is_immune`) → `|-immune|<mon>` (ability form carries
  `[from] ability:`); Water/Volt Absorb heal → `|-heal|…` via `apply_absorb_heal`
  (turn.rs:984).
- Crit → `|-crit|<mon>`; effectiveness → `|-supereffective|` / `|-resisted|` (from the
  type multiplier the calc already computes).
- Damage apply — `apply_damage` (turn.rs:2698) → `|-damage|<mon>|<HpStatus>` (or the
  Substitute path `absorb_into_sub` (turn.rs:2724) → `|-activate|<mon>|Substitute|
  [damage]` + `|-end|…|Substitute` on break).
- Secondaries — `apply_secondaries` (turn.rs:2220) / `apply_triattack_secondary`
  (turn.rs:2311) → `|-status|` (para/frz/psn), `|cant|…|flinch` next turn, and
  `apply_secondary_boost` (turn.rs:2553) → `|-unboost|`/`|-boost|`.
- Fire-move thaw tail → `|-curestatus|<defender>|frz`.

### Status / setup / recovery / protect / spikes / phaze — `run_status_move` (turn.rs:1506)
The category-Status router. Its arms:
- Major-status apply (`try_set_status`, turn.rs:2443) → `|-status|<foe>|<status>`; a
  type-immune target → accuracy drawn then `|-immune|` (or `|-fail|`); an already-
  statused/clause block → `|-fail|<foe>|<status>`.
- Self-boost (setup) → `|-boost|<user>|<stat>|<n>` per boosted stat.
- Recovery (`apply_heal`, turn.rs:959) → `|-heal|<user>|<HpStatus>`; full-HP fail →
  `|-fail|<user>`.
- `run_rest` (turn.rs:1937) → `|-status|<user>|slp|[from] move: Rest` then the wake path
  via `on_before_move` emits `|cant|…|slp` and `|-curestatus|…|slp|[msg]`.
- `run_protect` (turn.rs:2005) → `|move|…|Protect||[still]` + `|-singleturn|<mon>|Protect`;
  the block (in `run_move`) → `|-activate|<protector>|Protect`.
- Spikes (`apply_entry_hazards`, turn.rs:3734 in `run_switch`) → `|-sidestart|<side>|
  Spikes` (the move) + `|-damage|<entrant>|<HpStatus>|[from] Spikes` (the switch-in chip).
- Leech Seed (`apply_leech_seed`, turn.rs:1056) → `|-start|<foe>|move: Leech Seed` +
  the residual `|-damage|…|[from] Leech Seed` / `|-heal|…`.

### Switch / drag / faint — `execute_switch` / `drag_in` / `process_faints`
- `execute_switch` (turn.rs:3533) → `|switch|<mon>|<Details>|<HpStatus>`; `run_switch`
  (turn.rs:3693)'s ability `Start` emits the entry ability/weather lines.
- `drag_in` (turn.rs:3519) (Roar/Whirlwind) → `|drag|<mon>|<Details>|<HpStatus>` (same
  grammar as switch).
- `process_faints` (turn.rs:2741) = `faintMessages` → `|faint|<mon>` per newly-fainted
  mon, in the sim's faint order.

### Residuals — `run_residuals` (turn.rs:630)
Per handler, in the residual sort order (the emission mirrors the apply order the fn
already uses): weather chip → `|-weather|<W>|[upkeep]` then `|-damage|<mon>|<HpStatus>|
[from] <Weather>`; Leftovers (`apply_leftovers`, turn.rs:935) → `|-heal|…|[from] item:
Leftovers`; burn/psn DoT → `|-damage|…|[from] <brn|psn|tox>`; then the closing `|` +
`|upkeep|` + `|turn|N`.

### Win / loss — `check_win` (turn.rs:3371) + `turn_loop` (turn.rs:3236)
On `pokemon_left==0`: `|win|<PlayerName>` (name of the winner); a double-KO → `|tie|`.
These are player-layer lines the bridge relays (see the inventory) — still emitted by the
builder, drained at battle end.

**Emission-order invariant:** the builder pushes lines exactly where the engine already
performs the corresponding event, so line order == the golden by construction (the engine
already runs events in Showdown's order — that is the whole point of the RNG-faithful
port). The one place to be careful is the **deferred-faint protocol**: `apply_damage`
zeroes HP (emit `|-damage|…|0 fnt` there) but `process_faints` sets `fainted` and emits
`|faint|` LATER (after the in-tryMoveHit shuffle) — mirror that split so the `|faint|`
line lands at the sim's position, not right after the damage.

---

## (c) The byte-comparison TEST strategy (`tests/protocol_test.rs`)

A new `tests/protocol_test.rs`, analogous to how `e2e_fuzz_test.rs` asserts the seed:
replay the capture golden through the emitting engine and assert **byte-equality per
line**.

1. **Parse the golden** (`tests/vectors/protocol_capture_golden.txt`): per battle read
   `TEAM` (packed p1/p2), `BATTLE` (the `>start` seed), `INIT` (the pre-first-decision
   seed), the `DEC` choice tokens (`m<K>`/`s<N>` → `Choice::Move`/`Switch`, decoded like
   `fullbattle_test.rs`), and the ordered `L` raw lines (the expected stream).
2. **Replay**: `Battle::start_with_switchins` at the `>start` seed, then feed the recorded
   choices via `run_full_battle` WITHOUT re-seeding (exactly the existing full-battle
   replay), draining `log` after each committed decision.
3. **Diff**: concatenate the drained lines and assert they equal the golden's `L` lines,
   **in order, byte-for-byte** — panicking with the first divergent line index + both
   sides (the same first-divergence style as the seed tests). This is the level-2
   differential: a single wrong token / HP fraction / missing tag fails.
4. **Normalization** (the two allow-lists, matching the inventory):
   - **`|t:|`** — the golden stores `|t:|<NORMALIZED>`; the engine's `|t:|` value is
     wall-clock, un-reproducible, and poke-env-ignored. The comparison NORMALIZES both
     sides' `|t:|` lines to the placeholder before diffing (so their POSITION is asserted
     but not the timestamp value).
   - **`|debug|`** — see §d: debug lines are a phasing decision. If the port opts NOT to
     emit `debug`, the comparison FILTERS `|debug|` lines from BOTH the golden and the
     engine output before diffing (they're poke-env-ignored, so dropping them is safe).
     If the port DOES emit them, keep them in the diff.

Deterministic unit gates alongside (like every other layer): pin the fiddly formatters
directly — the `HpStatus` three variants (`x/y`, `x/y slp`, `0 fnt`), the `[from]`/`[of]`
cause rendering, the `[still]`-empty-target `|move|`, `-boost` vs `-unboost` by sign, the
`|-weather|` set-vs-`[upkeep]` forms, the `p1a:` vs `p1:` ref split.

**Interaction with the existing RNG gate:** `protocol_test.rs` is a SUPERSET assertion —
it replays the SAME battles the seed tests do, so if the seed diverges the protocol
diverges too. Keep both: the seed test localizes an RNG bug, the protocol test localizes
a formatting bug (a wrong line at a correct seed).

---

## (d) Phasing plan (which line types first) + covered/deferred

Emit in frequency order (the inventory's counts), so the highest-value lines land first
and the level-2 test goes green incrementally. Each phase = a self-contained set that
makes a subset of the golden's scenarios byte-clean.

**Phase 1 — the framing + core move/damage/switch/faint/turn/win core** (the bulk of the
bytes; makes the `both_switch_distinct` / `post_faint_sweep_win` / `double_faint_replace`
/ `last_mon_double_ko_tie` scenarios byte-clean):
`|t:|` (normalized), `|gametype|`, `|player|`, `|gen|`, `|tier|`, `|rule|`, `|teamsize|`,
`|start`, `|` (separator), `|turn|`, `|upkeep`, `|move|` (+ `[miss]`/`[still]`),
`|switch|`, `|drag|`, `|-damage|` (all HP variants), `|-heal|` (+ `[from] item:`),
`|faint|`, `|-crit|`, `|-supereffective|`, `|-resisted|`, `|-immune|`, `|-miss|`,
`|win|`, `|tie|`.

**Phase 2 — weather + boosts + abilities** (the `sand_intimidate_effectiveness` scenario):
`|-weather|` (set + `[upkeep]` + `[from] ability:`+`[of]`), `|-ability|`, `|-boost|`,
`|-unboost|`.

**Phase 3 — status + cant + fail** (`status_para_and_boost_drop`,
`secondary_status_flinch`, `recover_and_rest`):
`|-status|` (+ `[from] move: Rest`), `|-curestatus|` (+ `[msg]`), `|cant|`, `|-fail|`.

**Phase 4 — volatiles + side conditions + phaze** (`substitute_absorb`, `protect_block`,
`spikes_and_phaze`):
`|-start|`, `|-end|`, `|-activate|` (+ `[damage]`), `|-singleturn|`, `|-sidestart|`, the
`|drag|` phaze path (already grammatical in Phase 1, exercised here).

**Phase 5 — debug (optional)**: `|debug|` free-form lines. Two options (both byte-legal
per the inventory since poke-env ignores `debug`): (i) emit them to match the golden
exactly (most faithful — reproduce Showdown's `debug`/`-debug` calls at the same sites),
or (ii) do NOT emit them and filter them from the golden in `protocol_test.rs` (§c.4).
**Recommendation: option (ii)** — the debug text is sim-internal, poke-env-ignored, and
reproducing arbitrary free-form strings is high-effort/low-value; filter them. Revisit if
a downstream consumer ever needs them.

**Covered by this design (the modeled gen-3 OU surface):** every type in
`protocol_inventory.md`'s inventory tables — all 38 captured types.

**Deferred (mechanic not yet in the engine → its line lands with the mechanic):** the
"NOT in this capture" list in the inventory — `-sethp` (Pain Split), `-cureteam` (Heal
Bell), `-setboost` (Belly Drum), the `-clear*`/`-invertboost`/`-copyboost`/`-swapboost`
Haze family, `-sideend` (Rapid Spin), `-item`/`-enditem` (berries/Knock Off/Frisk),
`-prepare` (two-turn moves), `-mustrecharge` (Hyper Beam), `-transform` (Ditto). Add each
row + its hook when the engine layer for that mechanic is built. (Team-preview
`clearpoke`/`poke` never apply — gen-3 has no team preview.)

---

## (e) Interaction with the bit-for-bit SEED work — emission is OBSERVATION-ONLY

**Emission must NOT change PRNG consumption.** The whole port's value is the
RNG-consumption-order equivalence (`CLAUDE.md` §"Why bit-for-bit is the hard part"). A
protocol line is a *side output* of an event that ALREADY happened — pushing a formatted
string draws NO randomness and reads only already-computed state.

Concretely, the emission layer must:
- **Draw nothing.** No `self.prng` / `Prng::random*` call in `ProtocolBuilder` or in any
  emit hook. The builder only formats already-decided values (HP, status, boost, the
  chosen crit/miss/damage the RNG-faithful engine computed).
- **Read, never mutate, battle state.** Emit hooks take `&self`-reachable state to
  format; the state mutation is the engine's existing job at the same site. (Threading a
  `&mut ProtocolBuilder` on `BattleState` mutates only the log vector, never battle
  fields.)
- **Not reorder events.** Lines are pushed at the existing event sites, so line order
  follows event order — which is already seed-verified. Adding emission changes no
  control flow, no handler sort, no shuffle count.

**The regression guard:** after wiring emission, the ENTIRE existing suite
(`turn_test`, `battle_test`, `fullbattle_test`, `secondary_test`, …, `e2e_fuzz_test`)
must stay green with **identical seed assertions** — that is the proof emission didn't
perturb the PRNG. If any post-decision seed shifts, an emit hook wrongly drew or a
mutation leaked. Run the full seed suite as the emission gate, THEN add
`protocol_test.rs` for the byte layer. (The two are orthogonal: seed parity proves the
engine untouched; byte parity proves the formatting correct.)

**Bridge drop-in contract (the endgame):** once `protocol_test.rs` is green,
`battle.rs`'s `BattleStream::write_line` / `Battle::choose` (currently `todo!()`) return
`Vec<ProtocolLine>` from `log.drain()` — the omniscient stream the bridge's
`getPlayerStreams` equivalent folds per-side. That is the level-2 goal: the port emits
the exact bytes `local_sim_bridge.js` relays to poke-env, with no server.
