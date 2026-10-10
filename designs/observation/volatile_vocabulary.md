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

**poke-env reading findings this surfaced** (history — the reading is the Rust core's now). These were reported, not fixed. poke-env's reading is
not the spec.

- `|-activate|<mon>|item: Focus Band` does NOT disclose the item. The generic branch only starts an
  effect, so the opponent's item slot stays unknown after a public reveal. This reaches the obs
  item block.
- `|-activate|<target>|move: Spite|<move>|<n>` does NOT apply the PP cut to the opponent's estimated
  PP. Our own side is corrected by the next `|request|`.
- MIND_READER is `ends_on_turn`, so it's dropped at the next `|turn|`. LOCK_ON never ends before
  switch-out. The sim's `lockon` lasts 2 turns (through the user's next move) for both. So the
  `lockon` bit is never visible at a turn-start decision after Mind Reader, and stays set after a
  Lock-On is spent.
- Every one-shot effect stays in `mon.effects` until switch-out, which is why `NOT_A_VOLATILE` has
  to exist at all.
