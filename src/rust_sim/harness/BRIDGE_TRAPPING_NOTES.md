# BRIDGE_TRAPPING_NOTES.md — the trapping `|request|` characterization + fuzz plan

This is the design record for the SUBTLEST part of the poke-env `|request|` JSON —
the Arena-Trap / Magnet-Pull SWITCH-legality flags (`maybeTrapped` / `trapped`) and
the `|error|` re-request round — that the Rust `|request|` emitter (a Phase-1
deliverable of the Rust-sim bridge integration) must reproduce byte-for-bit.

It complements the two harnesses added alongside it:

- `probe_bridge_trapping.js` — the CHARACTERIZATION probe (Deliverable A): drives the
  real in-process `BattleStream`, ISSUES a rejected switch to trigger the reveal, and
  asserts the observed `active[0]` flags across the full 9-case matrix (fail-loud, so
  it doubles as a real-sim characterization GATE).
- `gen_bridge_trapping_capture.js` → `tests/vectors/bridge_trapping_golden.txt`
  (Deliverable B): folds the `trapped:true` case into a byte-reproducible capture
  golden (the Phase-0 `bridge_capture_golden.txt` never reaches it — its legal-only
  driver never issues a rejected switch). This is the Phase-1 byte target that
  includes the trapped state machine.

The gen3 trapping MECHANICS (who is trapped) are already modeled bit-for-bit in the
Rust engine — see `src/rust_sim/CLAUDE.md` → "## Trapping" (`gen3_trapping_v1`) + the
`tests/regression_test.rs` pins T1–T5. This note is about the REQUEST-DISPLAY surface
downstream of that fact.

---

## The characterization result (the matrix, measured vs the real sim)

`probe_bridge_trapping.js` (all 9 cases PASS against the omniscient sim):

| # | case | `active[0]` flag | `\|error\|` | switch accepted? |
|---|---|---|---|---|
| 1 | Arena Trap (Dugtrio) vs a grounded foe (+bench) | `maybeTrapped:true` → (reject) → `trapped:true` | `[Unavailable choice] Can't switch: The active Pokémon is trapped` | NO |
| 2a | Arena Trap vs a FLYING foe (Zapdos) — control | neither | (none) | YES |
| 2b | Arena Trap vs a LEVITATE foe (Gengar) — control | neither | (none) | YES |
| 3 | Arena Trap vs a grounded GHOST (Banette) | `maybeTrapped:true` → `trapped:true` | same `\|error\|` | NO |
| 4 | Magnet Pull (Magneton) vs Skarmory (Steel/Flying) | `maybeTrapped:true` → `trapped:true` | same `\|error\|` | NO |
| 5 | Magnet Pull vs a non-Steel foe — control | neither | (none) | YES |
| 6 | Dugtrio MIRROR (both Arena Trap) — each side | `maybeTrapped:true` → `trapped:true` (both) | same `\|error\|` (both) | NO (both) |
| 7 | Magneton MIRROR (both Magnet Pull) — each side | `maybeTrapped:true` → `trapped:true` (both) | same `\|error\|` (both) | NO (both) |
| 8 | trapped mon is the LAST mon (no live bench) | NEITHER flag (both omitted) | (none) | n/a (no bench) |
| 9 | trapper faints / switches out | `maybeTrapped:true` → NEITHER (lifts next request) | (none) | YES (after) |

### The three captured `active[0]` forms (from case 1 / case 8)

**`trapped:true`** (after the rejected switch — the state Phase-0 never captured):
```json
{
  "moves": [
    { "move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "target": "normal", "disabled": false },
    { "move": "Splash",    "id": "splash",   "pp": 64, "maxpp": 64, "target": "self",   "disabled": false }
  ],
  "trapped": true
}
```

**`maybeTrapped:true`** (BEFORE the rejected switch — the `'hidden'`-ability form):
```json
{
  "moves": [ /* IDENTICAL moves array — every move stays legal */ ],
  "maybeTrapped": true
}
```

**NEITHER flag** (last-mon, case 8 — both flags OMITTED even though the ability traps):
```json
{
  "moves": [ /* IDENTICAL moves array */ ]
}
```

The `moves` array is BYTE-IDENTICAL across all three forms (verified: only SWITCH is
disallowed under a trap — every move keeps its `disabled` state). The `\|error\|` text
is a fixed constant on the trapped side only.

---

## The exact state machine the Rust Phase-1 `|request|` emitter must implement

Recompute trap-ness **per request** from the engine's `is_trapped` (`turn.rs`,
`gen3_trapping_v1`) — the flag is NON-STICKY (it is not a one-time patch; it is
recomputed on every `move` request):

1. **A `move` request for an active mon** — determine `is_trapped` (Arena Trap:
   grounded foe; Magnet Pull: Steel foe; both mirrors mutual; a grounded Ghost IS
   trapped; Flying/Levitate escape Arena Trap):
   - `is_trapped == false` → emit **neither** flag.
   - `is_trapped == true` AND the side has **≥1 live, non-active bench mon** AND **no
     switch has been rejected yet this request** → emit `active[0].maybeTrapped: true`
     (the `'hidden'`-ability display: the sim marks it "maybe" until a rejection
     confirms it). The `moves` array is UNCHANGED (switch-only restriction).
   - `is_trapped == true` AND the side has **NO live bench** (this is the last mon) →
     emit **neither** flag (the `getMoveRequestData` `canSwitchIn` / `isLastActive`
     gates: with nothing to switch to, the sim omits both flags).
2. **The side ATTEMPTS a switch** (a `>pN switch K` at a trapped `move` request) →
   - emit exactly `\|error\|[Unavailable choice] Can't switch: The active Pokémon is
     trapped` to THAT side, and
   - RE-REQUEST that side's `active`, now with `active[0].trapped: true` (drop
     `maybeTrapped`; keep the `moves` array identical). The switch is NOT committed
     (draw-free — the engine already models the rejection as draw-free; the `\|request\|`
     re-emit is likewise draw-free).
3. **The trapper leaves** (faints / switches out) → the next `move` request recomputes
   `is_trapped == false` → **neither** flag (per-request recompute, not a sticky
   `trapped:true` that persists).

### The legal-action rule (how poke-env consumes it — the WHY)

`src/poke_env/battle/battle.py::parse_request` derives switch-legality from the flag:

- `active[0].trapped` (any truthy) → `self._trapped = True` → `available_switches` is
  left EMPTY (`if not self.trapped: … append bench mons`). So `trapped:true` makes the
  policy's switch actions illegal.
- `active[0].maybeTrapped` → `self._maybe_trapped = True` — **display-only**;
  `available_switches` is STILL populated (a `maybeTrapped` request keeps switches
  offered; the sim only firms it to `trapped` after a rejected attempt). This is why
  the RL runtime's legal-action mask must treat `maybeTrapped` as "switch still
  offered" — the rejection round is what removes it.
- neither flag → not trapped → `available_switches` = all live non-active bench mons.

So the Rust emitter getting `maybeTrapped` vs `trapped` vs neither RIGHT is exactly
what makes poke-env compute the identical legal switch set.

### What does NOT change under a trap

- Every MOVE stays legal — the `moves` array (ids, `disabled` bits, pp) is unchanged;
  only SWITCH is disallowed (the `\|error\|` is switch-specific).
- A PHAZE (Roar/Whirlwind) still drags a trapped mon out — trapping gates only the
  VOLUNTARY switch (`is_trapped` never consults the drag path; pin T4).
- A fainted mon's FORCED replacement (`forceSwitch` request) is accepted — the trapped
  check is `requestState === 'move'`-only, so a `forceSwitch` request carries no
  trapped flag and its switches are legal.
- The trapping mon itself switches freely (it is not trapped).

---

## The FUZZ oracle (Phase 1 will validate the Rust emitter against this)

The user flagged "we will likely need to fuzz this." The differential gate, once the
Rust `|request|` emitter exists:

**Generator** — random trapping matchups:
- one side carries an Arena-Trap holder (Dugtrio / Diglett) OR a Magnet-Pull holder
  (Magneton / Nosepass) — sometimes BOTH (a mirror);
- the other side is a random gen3 team of varied FOE TYPES (grounded / Flying /
  Levitate / Steel / grounded-Ghost / Steel-Flying), varied GENDERS (pinned
  explicitly — an unspecified gender on a ratio species makes the sim draw a
  construction-time gender `sample`, an unmodeled init draw), and varied BENCH SIZES
  (incl. a last-mon-no-bench case to exercise the both-flags-omitted branch);
- fixed master seed (printed) → reproducible; the choice-RNG is separate + recorded.

**Per foe-decision, the differential record** — drive BOTH engines over the identical
command stream and, at each `move` boundary for the trapped side, ATTEMPT a switch and
record from BOTH:
- the pre-attempt `active[0]` flag form (`maybeTrapped` / `trapped` / neither),
- the `\|error\|` text (or none),
- the post-attempt (re-request) `active[0]` flag form,
- the `moves` array (must be byte-identical to the pre-attempt form — legality
  unchanged),
- and, downstream, poke-env's derived `available_switches` / `trapped` (the ultimate
  consumer — a Python parity leg can assert the derived legal-switch set matches).

**The gate**: the Rust binary must emit the IDENTICAL flags + `\|error\|` + re-request
for the IDENTICAL command stream. Any divergence (a `maybeTrapped` where the sim omits
both flags, a missing `trapped:true` re-request, a wrong `\|error\|`, a `moves` array
that changed under the trap) is a first-divergence FAIL with the case's teams + seed +
command index — a standalone-replayable repro (mirror the A/B fuzzer's repro-dir
pattern in `ab_fuzz.js`, `kind=request`).

**Coverage floors** the fuzz should enforce so every branch realizes: ≥1
`maybeTrapped→trapped` reveal, ≥1 both-flags-omitted last-mon case, ≥1 mirror
(mutual trap), ≥1 control (Flying/Levitate/non-Steel → neither flag, switch accepted),
≥1 trap-lift (trapper leaves → neither next request).

---

## Surprises / honest scope (noted, not chased)

- **gen3 NEVER emits `trapped:true` WITHOUT a prior rejected switch.** The measured
  state machine is strictly `maybeTrapped` (the `'hidden'`-ability display) → a rejected
  attempt patches it to `trapped:true`. So the Rust emitter must NOT emit `trapped:true`
  on the FIRST request for a trapped mon — only after it has rejected that side's switch
  this request. (This is why Phase-0's capture, which never issued a rejected switch,
  only ever saw `maybeTrapped`.)
- **Partial-trap MOVES (Wrap / Bind / Mean Look / Spider Web / Block / Whirlpool) are
  OUT of gen3-OU scope here.** They set a `partiallytrapped` / `trapped` VOLATILE (a
  different trap source than the ability's `'hidden'`), and Mean-Look-class moves are
  DEFERRED in the engine (fail-loud via the unmodeled-status-move guard — see the
  "## Trapping" DEFERRED note). Their request-flag behaviour (a move-sourced trap tends
  to surface as `trapped:true` directly rather than `maybeTrapped`, since the source is
  not "hidden") is NOT characterized here and should be a SEPARATE probe if/when those
  moves are admitted. This note + the matrix + the golden cover ONLY the ability trap
  (Arena Trap + Magnet Pull), the gen3-OU-relevant case.
- **Shadow Tag** (batch-4 modeled ability) traps unconditionally (no grounded/type
  gate; a Flying foe is trapped; the mirror is mutually trapped). Its request-DISPLAY
  is the same `maybeTrapped→trapped` machine as Arena Trap / Magnet Pull (it is also a
  `'hidden'`-ability trap). It is not in this probe's matrix (no gen3-OU sample team
  carries it into the trapping request path), but the same emitter rules apply — a
  future fuzz should fold Wobbuffet/Wynaut in as an additional trapping source.
```
