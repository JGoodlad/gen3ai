# The volatile vocabulary — source-derived, crash-don't-drop

**Moved from `src/agents/observation/CLAUDE.md` on 2026-10-10.** This doc OWNS its subject and carries the same
ALWAYS-CURRENT obligation as that leaf — update it in the same pass as `gen3_effects.py` /
`gen3_effect_sources.py`. Index: [`README.md`](README.md).

## Moved from the leaf (2026-10-10)

`gen3_effects.encode_volatiles` RAISES on an id it has not classified. That's by design, and it
kept being completed one crash at a time: `doomdesire`, `immunity`, `magmaarmor`, `waterveil`,
then **`healbell`** (2026-09-24, belief-calibration read, Metamon ladder teams; the pool has no
Heal Bell). Why it kept happening (history — the reader is the Rust core's now): poke-env's `-activate` handler calls `start_effect` for ANY
`|-activate|<mon>|<effect>` it has no special branch for, so a ONE-SHOT announcement lands in
`mon.effects` beside the real volatiles and stays until switch-out.

**The class is derived, not curated** (`gen3_effect_sources.py`, off the hot path):

1. **Static half.** It scans every `add('-start'|'-activate'|'-singleturn'|'-singlemove', …)` that
   the gen3 format EXECUTES. That covers `sim/*.ts`, plus `data/{moves,abilities,items,conditions}.ts`
   through the mod chain gen3 → gen4 → … → gen8 → base, resolved the way `sim/dex.ts` merges it. A
   mod key shadows its parent's, so gen3's Quick Claw / Lightning Rod / Synchronize / Aromatherapy
   lines drop out. Entries are kept only for gen3-legal ids (`agents.gen3_data`).
2. **Executed half — on the RUST reader** (`main.live.effect_scan.probe_lines`): every concrete line is fed,
   after a fixed two-mon preamble, to the reader + encoder the live client runs; a refusal or an unclassified
   volatile is a finding. (The Python execution on a `Gen3Battle` is deleted, T27 P6 slice 6d-2; before it went
   both executions judged all 93 concrete lines alike.)

Every hand-written row in that module carries its reason and is checked against the source by the
test: computed arguments (`DYNAMIC_EFFECT_EXPANSIONS`), gen-gated `sim/` lines and conditions,
rule-gated lines (`RULE_GATED_LINES`), and non-obtainable items.

**The classification** (`gen3_effects.py`, each entry with the line that carries its information):

- a **slot** (`GEN3_VOLATILE_TO_SLOT`). `mindreader` is the `lockon` STATE: Showdown adds the
  same `lockon` volatile for both moves.
- **`NOT_A_VOLATILE`**. These encode to nothing, and each one's reason names where its consequence
  arrives:

  | id | Where the consequence arrives |
  |---|---|
  | `healbell` | `-curestatus` |
  | `magnitude` | `-damage` |
  | `spite` | the PP, on the next request |
  | `mist` | the `-sidestart` side condition |
  | `safeguard` | the `-sidestart` side condition |
  | `focusband` | `-damage` |
  | `typechange` | the per-mon TYPE block: the Rust reader applies the `typechange` to the mon's types (`src/rust_sim/src/present/mon.rs`), which the type block and the damage op read |

  It's a named list, not a catch-all: `unknown` and any other id still raise.
- **the field sports** (`gen3_field_sport_slots_v1`, 2026-09-26). gen3 **Mud Sport** / **Water
  Sport** own the LAST two volatile slots (`mudsport`, `watersport`; `VOLATILE_DIM` 44 → 46, so the
  active context is 60). Baton Pass carries both (the gen4 mod sets `noCopy: false`, so a pass carries
  them silently); a switch or faint ends them. Semantics verified on the pinned Showdown sim — `designs/ARCHITECTURE.md`
  §1.5. `PENDING_OWNER_LINES` is now EMPTY (kept as the mechanism); `unknown` still raises. The Rust
  SIM does not implement either move, so the core sees them only on a parsed (ladder) stream.

**Verified NOT to reach the encoder in gen3 OU:**

- **Aromatherapy.** The gen4 mod emits `-cureteam`.
- **Beat Up.** `-activate|…|move: Beat Up` is gated by `Beat Up Nicknames Mod`, which gen3's
  `Standard AG` includes. Without that rule it would land as `unknown` and crash, and 11.4% of
  Metamon teams carry it. The test pins the rule.

**The gates.** Each of these fails on drift:

- `gen3_effects_test.py::test_every_effect_the_gen3_sim_announces_is_classified` plus its siblings
  (dead entries, stale rows, pending lines still raise).
- `main/live/effect_scan_test.py` — every derived line of the pinned Showdown reads and encodes clean on
  the Rust reader, and a fabricated effect is a finding.
- `main/ladder_drift_scan.py` runs the same TEXT derivation against Showdown **master** (the public
  server) and, since P6 of the poke-env retirement, EXECUTES each derived line on the RUST reader
  (`main.live.effect_scan`: a spectator chain + an encode) instead of on `Gen3Battle`; every replay
  is read and encoded per turn by the same reader (`main.live.replay_scan.scan_one`).

This change is **value-neutral on every state that encoded before**: a newly classified id only
touches encodes that used to RAISE. Proof: 400 pool battles (30,943 decisions) were byte-identical
before and after, and the obs goldens pass unchanged. There was no `ARCH_SIGNATURE` bump.

**Reading findings this surfaced** (2026-09-24, found in poke-env and shared by the Rust core's
reading, which mirrored it). All three are FIXED in the Rust reading (`src/rust_sim/src/present/`,
`gen3_live_reader_reading_v1`, 2026-10-19), each pinned on a real Showdown stream
(`present::live_reader_reading_tests`):

- `|-activate|<mon>|item: Focus Band` now DISCLOSES the band (`BoardReading::activate` sets the item,
  public; the band is not consumed). It used to only start an effect, so the opponent's item slot
  stayed unknown after a public reveal.
- `|-activate|<target>|move: Spite|<move>|<n>` now cuts `n` PP from the named move on the reading
  (the opponent's estimate; our own is the next `|request|`'s word anyway).
- Lock-On and Mind Reader both add the sim's `lockon` volatile (`duration: 2`, ended silently at the
  residual of the turn after). The reading now ends LOCK_ON / MIND_READER at the second `|upkeep|`
  after the start (`PMon::upkeep`), so the `lockon` slot is set at the next turn's decision and clear
  at the one after. It used to drop MIND_READER at the next `|turn|` (never visible at a decision)
  and keep LOCK_ON until switch-out. The engine plays none of Spite / Lock-On / Mind Reader (they
  fail loud), so no training row could hold them: the fixes reach only live and replay reads.
- Every one-shot effect stays in `mon.effects` until switch-out, which is why `NOT_A_VOLATILE` has
  to exist at all.
- **Silent ends** (`gen3_volatile_silent_ends_v1`, 2026-10-26 GIGO audit A-C1 / A-C2): the sim ends some volatile SLOTS with
  no protocol line, and the reading used to keep them set for the whole stint. Endure / Snatch / Helping Hand
  (`duration: 1`) now end at `|turn|` like Protect; Destiny Bond, Grudge and Rage end at the holder's next `|move|` or
  `|cant|` (`onBeforeMove` / `onMoveAborted`; a re-used Destiny Bond keeps it); Mean Look's `trapped` ends when its
  trapper leaves the field (not by Baton Pass), and Baton Pass now copies `trapped` (gen 4's `noCopy: false`, inherited by
  gen 3). Pinned by `src/rust_sim/tests/silent_volatile_end_test.rs` and the engine-truth audit (`present/audit.rs`
  `TRUTH_VOLATILES` names `destinybond` and `trapped`).
