# The four A/B fuzzers — runbooks, verdict taxonomies, green gates

> Moved VERBATIM from `src/rust_sim/CLAUDE.md` in the 2026-10-10 leaf cleanup (links re-based;
> statements found FALSE against the code were corrected in place, each saying so). The leaf keeps a
> one-line pointer here. Frozen original: `designs/research_state/claude_md_archive/src_rust_sim_CLAUDE_2026-10-10.md`.

### A/B fuzzer (the continuous differential parity hunter)

The e2e capstone is a FIXED 220-battle committed golden; the **A/B fuzzer** is its UNBOUNDED sibling —
`harness/ab_fuzz.js` runs for hours unattended, generating fresh team pairs + seeds + random legal
choices, driving the REAL Showdown sim and the port side by side through `src/bin/ab_replay.rs`, and
saving a **self-contained, standalone-replayable repro** for every divergence. Zero API quota while
running; every future mechanic layer becomes automatically stress-tested.

- **Five team modes** (`--mode`, default `randbats`; `ourandom` and `ladder` below): **`randbats`** — Showdown's OWN gen3
  random-battle generator, with sets adapted at the SET level to be port-replayable (adjustment rate
  logged per chunk); **`random`** — the MODELED-UNIVERSE generator, the coverage multiplier that
  flushes out modeled-predicate ↔ engine drift (195 species / 146 distinct moves on its first smoke,
  vs the pool's 21 / 37); **`pool`** — the filter-clean `data/teams/` teams with fresh seeds/choices.
  Flags: `--battles N` / `--hours H`, `--master-seed S` (defaults from time, **ALWAYS printed** ⇒
  reproducible), `--chunk N`, `--out DIR`, `--keep-chunks`.
- **The verdict taxonomy** `ab_replay` emits per battle, in precedence order:
  `seed` [a draw bug] > `request` > `species` > `state` [hp/maxhp/fainted/left] > `status` > `boost` >
  `confusion` > `spikes` > `firstmover`. Engine PANICS are CAUGHT (`catch_unwind` + a
  message-capturing hook) and reported as `"verdict":"panic"` — the loop never dies.
- **Repros** land in `<out>/divergences/<runid>_<battleid>/` as `battle.txt` (a single-battle chunk,
  standalone FOREVER, independent of generator drift) + `summary.json`. After an engine fix the same
  `ab_replay <dir>` must flip to `ok` — **then pin it** per the law above.
- 🚨 **The tool is FAULT-INJECTION PROVEN** (2026-07-03): a dropped draw ⇒ 6/6 flagged `kind=seed`, a
  +1-damage state error ⇒ 6/6 `kind=state`, a flipped winner ⇒ 5/6 `kind=winner` (the 6th a
  never-ended prefix battle, correctly still ok). A hunter nobody has fault-injected is a hunter that
  may be finding nothing.
- 🚨 **`ab_fuzz_out*` run dirs are gitignored — NEVER commit run output.**
- **Run it:** see the README runbook ("A/B differential fuzzer"). Quick start:
  `node src/rust_sim/harness/ab_fuzz.js --mode randbats --hours 12` (overnight),
  `--mode random --battles 200 --master-seed S` (reproducible bounded hunt);
  replay any repro with `target/selfcheck/ab_replay <repro-dir>` (the self-check build `ab_fuzz.js` builds).
- **The root-causing workhorse** is `harness/probe_repro_simtrace.js` — replay ANY saved repro dir
  through the REAL sim with per-draw PRNG call-site instrumentation, under the repro's own `FMT`
  format (a gen3ou repro replayed as `gen3customgame` is a different battle).

The driver/replayer internals and every closed finding record (the first bounded smoke's fix queue,
the residual tail's 7 engine bugs, fix-queue #4's 3 more):
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).

#### The OMNISCIENT BYTE differential (`--protocol`, `gen3_omniscient_byte_fuzz_v1`)

`--protocol` turns the A/B fuzzer's reconstructed STATE+seed+winner check into a **literal `|...|`
protocol byte differential**: per battle it TEES the real omniscient filtered log into the chunk
golden, `ab_replay --protocol` replays via `run_full_battle_logged`, filters BOTH sides through a
shared DENYLIST (drop `debug`/`error`, normalize `|t:|`), and first-divergence-diffs to a
`kind:"protocol"` verdict — **reported ONLY after state/seed/winner match, so a draw bug still
surfaces as `seed`**. `--format {gen3customgame,gen3ou}` threads the run format (gen3ou = the
clause-shuffle draw path + the OU framing). Genders are pinned (`pinGenders`) so the sim never draws
one at construction. Isolated build + run:

```bash
CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --profile selfcheck --features emission-selfcheck --bin ab_replay
POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/selfcheck/ab_replay \
  node src/rust_sim/harness/ab_fuzz.js --mode pool --protocol \
  --format gen3ou --battles 300
```

FAULT-INJECTION PROVEN (a mangled `[from] item: Leftovers` tag ⇒ 6/6 flagged `kind=protocol` at the
exact `-heal` line). The byte bugs it found and closed — BF1-BF4, the STATUS-MOVE EMISSION-FORM SWEEP
(pool byte-clean 27% -> ~95%, 13 pins) and the WIDE-NET round (BF-F16..BF-F20) — are recorded in
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md). All were
observation-only: the full seed suite stayed BYTE-IDENTICAL and the e2e md5 unchanged.

#### The KNOWN-RESIDUAL ALLOWLIST + the GREEN GATE (`gen3_omniscient_byte_fuzz_v1`)

The byte fuzzer is a **GREEN GATE**: a NEW divergence fails loudly, while a DOCUMENTED,
non-gen3ou-impacting artifact is EXPLICITLY allowlisted — never silently ignored. **These clauses are
the live definition of when the gate may pass; edit them only with an injection proof.**

`ab_replay --protocol` classifies a first-divergence byte diff via
`classify_known_residual(golden_framing, engine_framing, leads_speed_tie)` and adds
`"allowlisted":<reason>` to the per-battle verdict ONLY when the divergence FORM matches a documented
residual; otherwise the field is ABSENT and the divergence is a hard failure. Both live entries share
one root — the unmodeled turn-0 CONSTRUCTION speed-tie Fisher-Yates shuffle, the project-wide seed
convention every committed golden depends on — and both are **seed=None-invisible, so ZERO production
impact under the Rust bridge (the only training transport)** (at `seed=None` the port is the sole oracle, and
`event::run_start_switchins` falls back to a DETERMINISTIC side-order at a raw-Speed tie, drawing
nothing).

- **E1 `turn0-construction-speed-tie-attribution`** — a PURE PERMUTATION of identical-CONTENT framing
  lines. The classifier takes the two FULL framing WINDOWS (everything before the first `|turn|1`),
  sorts both, and allowlists ONLY IF `leads_speed_tie` AND the two windows are an **IDENTICAL
  MULTISET**. 🚨 **A CONTENT change therefore makes the multisets DIFFER ⇒ returns None ⇒ the gate
  FAILS.** That narrowing is the point: the prior coarse per-line-TYPE key SWALLOWED a
  content-different framing divergence at a mirror lead — a potential real bug.
- **A1 `turn0-construction-speed-tie-mirror-of-flip`** (`classify_construction_mirror_of_flip`) — the
  single-line weather-`[of]`-FLIP on a same-species MIRROR lead, which is NOT a pure permutation (one
  line's `[of]` CONTENT changed) so E1 returns None. Allowlisted **ONLY when ALL SIX STRUCTURAL
  CLAUSES hold**, else None ⇒ the gate FAILS: (1) the construction speed-tie (`leads_speed_tie`);
  (2) the two equal-length framing windows differ in EXACTLY ONE line; (3) that line, in BOTH golden
  and engine, is a `-weather`/`-ability` framing line; (4) the two are byte-identical after stripping
  the trailing `|[of] pNa: <name>` clause (same weather/ability, same `[from]`); (5) the two `[of]`
  targets are the two DIFFERENT active slots (one `p1a:`, one `p2a:`); (6) both `[of]` idents map —
  via the framing `|switch|` details' species field — to the SAME species (the sibling mirror). An
  `[of]` to a non-sibling or different-species mon, a different weather/ability prefix, a
  missing/extra framing line, or a non-mirror pair breaks a clause. **GATE-INTEGRITY PROVEN** by two
  mangled-golden injections plus 7 `a1_allowlist_tests` in `ab_replay.rs`, the load-bearing one being
  `clause6_wrong_of_to_a_real_different_species_mon_fails`.

- **`ab_fuzz.js --protocol`** counts `allowlisted` SEPARATELY from `diverged`, reports
  allowlisted-by-reason, and **exits non-zero ONLY on a non-allowlisted `diverged`/`panic`/
  `parse_error`**. Allowlisted repros are saved under `<out>/allowlisted/` (auditable, and a fixture
  source); real divergences under `<out>/divergences/`.
- **`tests/byte_fuzz_corpus_test.rs`** (the `cargo test` gate) enforces "no NEW kinds": each fixture
  resolves to either `ok` (the emission-form fixtures stay byte-clean) OR a `diverged` verdict whose
  `allowlisted` reason EXACTLY equals a `# ALLOWLIST <reason>` header the fixture is tagged with. So
  (i) a residual fixture that stops matching its reason FAILS, and (ii) **nobody can add a
  silently-ignored divergence — every escape is a named allowlist entry backed by a tagged repro.**
  FAULT-INJECTION PROVEN: stripping a fixture's tag makes the corpus test FAIL. To add a fixture, drop
  a clean fuzzer repro `battle.txt` in the folder (see its `README.md`) — the test auto-discovers it.

The R- and T-numbered bugs behind the current green state (R3 IV-derived Hidden Power BP, R2 the
Leftovers `-heal` slot-condition gather order, R13 Encore x Sleep Talk, R15 Sleep Clause x a
self-Rest sleeper, T1 freeze persistence vs Hidden Power Fire, the Endure and Natural-Cure emission
fixes, and the Pressure-Curse PP root of the round-2 tail):
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).

### Bridge / request A/B fuzzer (the per-side + `|request|` parity hunter)

The **PER-SIDE sibling** of `ab_fuzz.js`: `harness/bridge_ab_fuzz.js` verifies, over random teams,
that the crate's PER-SIDE (`p1`/`p2`) streams + the `|request|` JSON — the legal-action
requests, including the maybeTrapped/trapped switch-legality state machine — are BYTE-IDENTICAL to the
real Node `getPlayerStreams`. It is the validation harness for `bridge.rs`. Its taxonomy: `preamble` /
`perside` / `privacy` [HP-fold] / `request` [JSON] / `error` [trapped] / `chunk_count` / `panic`.

- **Modes** (`--mode`, default `trapping`): **trapping** — a coordinated Arena-Trap / Magnet-Pull /
  Shadow-Tag matchup vs varied grounded/Flying/Levitate/Steel/Ghost foes, the one mode where the port
  is already bit-for-bit on the omniscient stream, so ALL divergences are genuine request/per-side
  issues; plus `randbats` / `random` / `pool` reusing `ab_fuzz.js`'s exported providers. **TRAPPING
  PROBES** (`--trap-prob`, default 0.5) issue a REJECTED `switch` first so the `|error|` +
  `trapped:true` re-request round is exercised.
- 🚨 **Isolated build: `CARGO_TARGET_DIR=/tmp/pokesim_target_bridge`** — NEVER the shared `target/`,
  which holds the live `ab_replay`.
- **Fault-injection PROVEN**: drop the firm-trapped flag ⇒ `kind=request`; wrong `|error|` text ⇒
  `kind=error`; wrong HP-fold % under gen3ou ⇒ `kind=privacy`. Each caught, standalone-replayable, and
  restored byte-identical.
- **Three real Phase-1 bugs it found + FIXED** — Shadow Tag's FIRM trap (`trapped:true` on the FIRST
  request with no `maybeTrapped` phase, and a rejected switch draws `|error|` with NO re-request,
  unlike Arena Trap / Magnet Pull's `tryTrap(true)` -> `'hidden'` machine; `state::trap_is_firm`
  distinguishes them), the forced-Struggle `|-activate|<mon>|move: Struggle` OWNER-ONLY `sideupdate`
  line, and the per-side request residual.

- 🚨 **`maybeTrapped` is NOT "trapped but unconfirmed", and `maybeDisabled` is NOT "shares a move".**
  Both are endTurn DISPLAY predicates wider than the restriction they hint at. `maybeTrapped` also
  fires for a mon Transformed into its foe (`knownType` false: Magnet Pull drops its Steel gate,
  Arena Trap its Flying gate) and, in gen3ou only, for a foe whose SPECIES could hold Arena Trap /
  Shadow Tag (a Sand Veil Dugtrio) — `BattleState::is_maybe_trapped`, `gen3_known_type_maybe_trap_v1`.
  Imprison sets `maybeDisabled` + `maybeLocked` on EVERY live foe; `maybeLocked` drops only on a
  refused-MOVE re-request; the trap flag is written LAST; and the engine REFUSES a pick of an
  imprisoned move (`gen3_imprison_maybe_flags_v1`, `gen3_imprison_choice_reject_v1`). The Rust side reader reads
  `maybeTrapped` into the observation (`encoder/slot.rs`). Oracle `harness/probe_maybe_flags.js`; pins
  `tests/bridge_maybe_flags_test.rs` + bridge fixture 25. 🚨 **The fuzzers' pickers mirror the hidden
  disable, so NO fuzzer submits an imprisoned pick** — that path is covered by the pins only. An
  ALL-imprisoned mon is still OFFERED its full list and its pick is Struggle-SUBSTITUTED
  (`BattleState::forced_struggle`, `gen3_imprison_all_struggle_v1` — the request shape keeps reading
  `MonState::must_struggle`, the choice-time sites read `forced_struggle`; oracle
  `harness/probe_rereq_accumulate.js` F rows, pins `tests/bridge_imprison_struggle_test.rs`).
  Successive refusals in ONE decision ACCUMULATE on the outstanding `Request` (the sim's one
  `activeRequest`): `disabled_mask` keeps every slot a refused move flipped, `trapped` stays once a
  refused switch set it, and a refusal that changes nothing is `[Invalid choice]` with NO
  re-request — except a repeated IMPRISONED pick, which re-derives `maybeLocked` and so re-requests
  again (`gen3_rereq_accumulate_v1`). A `move` sent to a FORCED-SWITCH request is refused FIRST
  (`[Invalid choice] Can't move: You need a switch response`, nothing follows —
  `gen3_choice_kind_mismatch_v1`). Oracle `harness/probe_rereq_accumulate.js` A / K rows; pins
  `tests/bridge_rereq_accumulate_test.rs`
  ([`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md) § The
  M6 CUTOVER stress).

- **Honest scope (next phase):** `randbats`/`random` modes surface PRE-EXISTING **omniscient-stream**
  gaps orthogonal to the request/per-side layer — the non-L100 `details` LEVEL display
  (`switch_details`/request `details` omit `, L84`; the port targets L100 gen3ou), a **mid-battle
  Intimidate `|-unboost|…|atk|0`** at the −6 Atk FLOOR (`turn.rs:6667` hardcodes the delta `-1` →
  always `atk|1`; the sim emits the CLAMPED-applied 0 — repro saved, probe-confirmed
  `harness` Intimidate-clamp), and the same Toxic-`[from]`/status-move/Water-Absorb clusters
  `ab_fuzz.js` already tracks. These belong to the omniscient fuzzer's fix-queue (they'd desync the
  raw stream too), not the bridge layer. `trapped:true` coverage is dense; a `gen3ou`-format trapping
  run additionally exercises the OU reframe + HP-privacy fold.

#### The PER-SIDE KNOWN-RESIDUAL ALLOWLIST (`bridge_replay --ab`) — the bridge fuzzer's green gate

`bridge_ab_fuzz.js` exits non-zero on any `diverge`/`panic`/`parse_error` verdict that carries no
`allowlisted` reason (`verdictClass`: only a `diverge` can be allowlisted). **These clauses are the
live definition of when the per-side gate may pass; edit them only with an injection proof.** The
`|request|` keys (`classify_known_perside_residual`: Curse target and `return102`, both DORMANT, plus
the gender/level `details` suffix) are one family. The other three keys are a `kind=perside` first
divergence tried IN ORDER, and all three have the SAME root as the omniscient E1/A1: the unmodelled
turn-0 construction speed-tie order (the sim's tied `runSwitch` insert draws `battle.random`; the
port's `event::run_start_switchins` is deterministic at a tie). All three are seed=None-invisible:
- **B1 `turn0-construction-speed-tie-order-flip`**: a pure permutation of the framing window
  where every moved line is `-ability`/`-weather`.
- **`perside-construction-speed-tie-mirror-of-flip`**: a same-species MIRROR lead's ident flip
  (the single-line `[of]` form, or the Intimidate block permutation).
- **`turn0-construction-speed-tie-switchin-block-swap`** (`src/bin/bridge_replay/switchin_block_swap.rs`):
  a NON-mirror tie whose moved lines include Intimidate's `-unboost` (B1's clause 3 rejects that line;
  the mirror key needs same-species leads). The Zapdos-vs-Salamence cutover-stress repros are the
  example. Allowlisted ONLY IF ALL of these hold: (1) the leads' Speeds tie; (2) on BOTH sides the first
  `|turn|` line is exactly `|turn|1`, at the same index in golden and engine; (3) on BOTH sides everything
  from `|turn|1` to the end is BYTE-IDENTICAL; (4) each side's windows are an identical MULTISET; (5) a
  differing window is exactly ONE swap of two adjacent runs (golden `A++B`, engine `B++A`); (6) `A` and
  `B` are each a well-formed switch-in block of the two DIFFERENT leads. A block is a lone
  `-weather|…|[from] ability: …|[of] pNa`, a lone non-Intimidate `-ability`, or the pair
  `-ability|…|Intimidate|boost` then `-unboost|<foe>|atk|<n>`. A blocked Intimidate (`-fail`/`-immune`
  tail), Forecast, a three-block rotation, or a co-occurring request residual is NOT admitted.
- 🚨 **B1 and the mirror key do NOT check the rest of the battle.** The verdict is first-divergence,
  so an allowlisted B1/mirror battle cannot report a LATER real divergence. The block-swap key's
  clause (3) closes that for itself only. On 2026-09-25 all 10 B1/mirror-allowlisted cutover-stress
  repros were byte-identical after the window, so nothing was hidden in that run. Retrofitting (3)
  onto B1 and the mirror key is still OPEN.
- **Gate integrity:** `node harness/bridge_ab_fuzz.js --selftest` runs 25 cases through the real
  replayer. It replays every tagged fixture to its reason, then applies 14 mangled-golden injections
  that must each still FAIL (the NEGATIVES are the load-bearing half): `atk|1`→`atk|2`, a changed
  ability, a mis-targeted `-unboost`, a dropped line, an extra line, the Intimidate pair internally
  reordered (the multiset is preserved), the lead `|switch|` lines swapped, the other side's window,
  a non-tie lead, a later `-damage` or `|request|` on either side, a truncated golden, and the
  fixture-22 orientation. The classifier's own 26 `#[cfg(test)]` cases are also mutation-checked:
  every clause's revert fails a test, except (4) and the equal-length check, which (5) implies. Tagged
  corpus fixtures: `tests/vectors/bridge_corpus/21_*` and `22_*`. **Run `--selftest` after ANY change
  to a per-side allowlist clause.**

#### The BANKED SPEC QUEUE — probe-settled specs

Each mechanic that reached the engine through a probe left a re-runnable oracle in `harness/` whose
header carries a SETTLED block: the draw model, the exact emission forms, the edges, and the named way
a naive implementation desyncs. 🚨 **Read the probe before implementing; do not re-derive from
source.** The eight queued specs (Safeguard / Recycle / Fake Out / Conversion / Torment / Imprison /
Weather Ball / Skill Swap) are ALL SHIPPED; the queue and the trap that made each one non-obvious are
in [`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).

#### The `ab_replay` SUBSEQUENCE SEED ANCHOR

`gen3_ab_replay_seed_anchor_subsequence_v1`. `align_seed_subsequence` aligns the sim's per-decision
seeds as a SUBSEQUENCE of the port's checkpoints, so a decision-boundary CHECKPOINT offset stops
reading as `kind:"seed"` and costing a full triage; the per-decision STATE checks then run at the
ALIGNED pairs, and verdicts still report the GOLDEN's decision index (what a reader greps for in
`battle.txt`).

🚨 **A sloppy anchor makes the omniscient gate VACUOUS — far worse than the artifact it removes.**
`anchor_tests` is 10 cases and **the NEGATIVES are the load-bearing half**: an injected extra draw, a
missing draw, a REORDERED pair (same values, wrong order), a port stream that ends early, and an empty
port stream must each still FAIL. `POKESIM_DUMP_SEEDS=1` prints both per-decision seed lists, so the
anchor's soundness precondition — the port's checkpoint list really is a SUPERSET of the sim's — is
CHECKABLE on any repro rather than assumed.

**THREE READINGS, AND ONLY THE THIRD IS RIGHT — the durable methodological lesson.** Within one
session this repro was called: (1) "the known segmentation artifact", asserted from a byte-clean
`POKESIM_PROTOCOL_ONLY` replay — under-evidenced; (2) "the port surfaces FEWER requests, possibly
round 26's hypothesis (b), a real draw-free legality bug" — WRONG, and the alarming one; (3) the
one-draw checkpoint offset above. What settled it was comparing draw POSITIONS **within a single
decision**. The earlier reads compared draw COUNTS across differently-scoped windows — `ab_replay`
plays the WHOLE scripted battle before it compares, so its 171-draw trace covers all 33 port
decisions, not the 5 the verdict names. **A count comparison whose two sides cover different windows
is not evidence, and it reads exactly like evidence.**

What the anchor did and did not close on `ab_41_9` (a one-draw checkpoint-placement artifact the
anchor correctly refuses to reconcile — the port's RNG consumption is CORRECT), and why anchoring
against the DRAW stream instead carries its own vacuity risk:
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).

#### `--mode ourandom` — "gen3ou-randbats", the fuzz surface that is actually the one we care about

`gen3_ou_random_teams_v1` (`harness/ou_random_teams.js`). The fuzzers had two team sources and
NEITHER is the training/ladder surface:

| mode | on-surface? | diverse? |
|---|---|---|
| `pool` | **yes** — the 762 real gen3ou teams | **no**: a FIXED human-built set from a narrow meta, and the committed capstone samples only 220 battles of it |
| `randbats` | **no** — non-L100 levels, curated movesets, near-uniform items | yes |
| **`ourandom`** | **yes** | **yes** |
| **`ladder`** | **yes** — real PUBLIC-LADDER teams | **yes**: 22,813 human-built teams (`--ladder-tier commit\|milestone\|full`) |

#### `--mode ladder` — the LADDER-USAGE corpus (`gen3_ladder_usage_corpus_v1`)

All four A/B fuzzers take `--mode ladder` (`harness/ladder_corpus.js`, the JS twin of
`utils.ladder_corpus`): the Metamon `hl_05_26` gen3ou teams, filtered to what the ENGINE plays
(`scan_move_probe` — 22,813 of 22,862 kept; the 49 dropped carry Metronome, Shell Bell, Snore,
Psywave, Fly, Blast Burn, Dig, Grudge or Triple Kick). A team is drawn from the tier with the
fuzzer's seeded team RNG, so a master seed replays; `teamFilterClean` (the JS mirror) is REPORTED
against, never used to drop a team, and typed Hidden Power is pickable (the corpus is gen3ou-valid
and HP is priced at its IV-true BP). The same corpus is a team source of the Python fuzz scripts (`--team-source ladder`) and was the parity harness's third
(`rust_core_parity.play(key, source="ladder")`, deleted in P6 slice 6c).
The chapter: `designs/ops/testing.md` → THREE TEAM SOURCES.

🚨 **ITS FIRST RUN FOUND A LIVE EMISSION BUG THE POOL COULD NEVER SHOW** (`gen3_lockedmove_announce_v1`):
every CONTINUATION turn of a lock — Outrage / Thrash / Petal Dance, Uproar, Rollout / Ice Ball —
announces `|move|<user>|<Move>|<target>|[from] lockedmove` in the sim (`runMove`'s `getLockedMove()`
branch), and the port emitted the BARE line, so poke-env read each continuation as a fresh use and
charged a PP for it. 0 pool teams and 0 gen3 randbats sets carry any of those moves. Pinned by
`protocol_byte_fuzz_test::lockin_continuations_announce_from_lockedmove` (fails on revert). The
same probe found a MECHANICS bug behind it (`gen3_rollout_lock_duration_v1`): the sim's `rollout`
volatile is added on the first use with `duration: 1` and refreshed to 2 only by the
`basePowerCallback` (a turn that computes damage), so a MISS / Protect / a turn the user cannot act
ENDS the lock at that residual — the next use is fresh (PP paid, bp 30). The port kept the lock
across a miss (wrong PP, wrong bp, a wrong `trapped` request), and had no residual handler for the
volatile at all (a NO_ORDER/subOrder-2 duration handler, the Fury Cutter tie group). Pinned by
`protocol_byte_fuzz_test::rollout_lock_ends_on_a_turn_that_does_not_hit`; rust == node on full
per-side streams for Rollout / Ice Ball × {plain, paralysing, Protect} foes over 8 seeds each.

**THE MOTIVATING MEASUREMENT.** Both bugs found on 2026-08-17 (ROUND 42's Trace/forecast, ROUND
43's Substitute/wrap) have **ZERO gen3ou-pool exposure** — 0 of 773 pool files carry Castform, 0
carry a wrap-family move. Training plays pool-vs-pool, so neither could ever have fired there.
That is the round-24 lesson ("fix bugs found on the SURFACE YOU CARE ABOUT") restated as a number,
and it is why a gen3ou-native random generator is worth having.

**Every input is Smogon-derived and already committed** (`gen3_smogon_stats.json` usage,
`gen3_teammate_priors.json`, and the move / item / ability / spread priors) — deliberately NOT
`data/teams/gen3_species_priors.json`, which is POOL-derived, and the whole point is independence from
the pool. **Legality is Showdown's verdict, not a reimplementation**: every generated team goes
through the real `TeamValidator('gen3ou')`, so the banlist and every team-building clause are enforced
by the sim. 🚨 **Coverage is DISCLOSED, not assumed** — move sampling is renormalized over
ENGINE-MODELED moves (so a generated battle ALWAYS plays to completion), and the renormalized mass is
printed in the run banner by `describeCoverage()` on every run.

The Hidden-Power closed form (restricting IVs to {30,31} PINS BP at 70, so the bit-0 pattern alone
selects the type — verified 16/16 by RECOMPUTING type and BP from the emitted IVs), the two bugs the
real data shapes caught, and the first results:
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).

#### The picker's own blind spots (found while measuring the above)

- **STRUGGLE is now pickable.** It is ENGINE-MODELED (`pp_struggle_test.rs` is a full
  STATE+PP+SEED+winner differential) but `isModeledMove` returns false for it, so a mon with every
  slot spent AND no switch had no pickable choice and the whole battle was DROPPED to a prefix.
  That truncated precisely the PP-exhaustion endgames — the deepest, most state-laden turns, and the
  ones gen3ou STALL teams produce most. The live per-side gate already accepted Struggle
  (`id === 'struggle' || isModeledMove(id)`); the two harnesses simply disagreed and the offline one
  was weaker. **A picker predicate that gates a test silently SHRINKS that test** — the same shape
  as ROUND 42's L100 pin.
- **The drop LABEL named an innocent bystander.** `forced-unmodeled-move:<moves[0]>` reported the
  FIRST slot in the request regardless of why the pick failed, so a drop on a mon whose first slot
  happened to be Substitute read as `forced-unmodeled-move:substitute` — and Substitute is modeled,
  so the label sent a reader hunting a bug that does not exist (it did, on 2026-08-17). It now names
  the moves that actually blocked, with a distinct `all-disabled(...)` reason. **A diagnostic that
  names an innocent bystander is worse than one that names nothing.**
- **A HIDDEN disable reads `disabled:false` in the request** (`gen3_picker_hidden_disable_v1`).
  Imprison seals the foe's shared moves with a `'hidden'` disable that the owner's request masks,
  so a request-reading picker submits a doomed move; the picker now mirrors the sim's own
  `moveSlot.disabled` under `maybeDisabled`, as it mirrors `pokemon.trapped` for switches.
- 🚨 **The recorder never re-writes a HELD side** (`gen3_recorder_held_choice_v1`). After one side's
  reject the other side's choice is held, and a re-write is NOT a no-op in the sim — writes apply in
  order, so a held p2's re-write lands on the NEXT turn's request once `>p1` commits (a held p1's
  REPLACES it). The port's script keeps the held choice, so the repro replays one turn off with
  correct draws and reads `kind=seed` (the M6 stress's `rmuggvoke_ab_3_15`). A live client never
  sends that write (only the rejected side is re-asked). A repro recorded before this fix cannot
  flip to `ok`. Re-record it from the sim-APPLIED choices to test the port on it —
  [`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md) § The M6
  CUTOVER stress.

#### THE EXTERNAL-CONSISTENCY GATE (`gen_sim_bridge_diff.js`) — promoted to a green-gated fuzzer

`gen3_simbridge_diff_allowlist_v1`. **This is the strongest correctness gate in the project**, because
it is the only one that compares what a per-side reader ACTUALLY consumes (poke-env when it was
written; the Rust side reader since T27 P6), at the boundary that reader sits on.

**Why it outranks the byte/seed gates.** `ab_fuzz`/`ab_replay` diff the OMNISCIENT log — which poke-env
never sees — and replay a FIXED recorded decision list, so they must ASSUME both engines segment
decisions identically. When they don't, you get a `kind=seed` artifact that cannot be distinguished from
a real legality bug (the round-26 finding: 12 of 13 open repros have ZERO wrong draws). This harness
instead spawns BOTH real bridges, feeds identical stdin, and **discovers boundaries live** (read a
request → choose → compare). The segmentation artifact CANNOT occur by construction, and a genuine extra
request is an unambiguous request-frame mismatch. It also covers the `|request|` JSON — a genuinely
separate observable with its own bug history (PA2's Spikes-under-Pressure PP was INVISIBLE to the
omniscient fuzzer and only diverged in the request's `pp` field).

**THE LAYERING PRINCIPLE.** per-side + request = the CONTRACT (the correctness requirement); the
omniscient log = a LOCALIZER (engine bug vs fold/serializer bug); the PRNG seed = a LEADING INDICATOR
(catches divergence before it is observable). **An outer-layer mismatch is ALWAYS a bug; an inner-layer
mismatch with a clean outer layer is NOT necessarily one** (the turn-0 construction residuals are exactly
that). Inner layers buy detection SPEED and LOCALIZATION, not correctness.

**What landed:**
- **The GREEN GATE + allowlist.** Previously any known-benign residual failed the run — a 10-battle
  *(this bullet was truncated in the leaf before 2026-10-10; the full record is in
  [`ab_fuzzer_findings.md`](ab_fuzzer_findings.md))*

- **`--selftest`** — 14 gate-integrity assertions: 10 allowlist ones whose NEGATIVES are load-bearing
  (a different move; an alias PLUS a residual pp difference; a LEVEL value difference; a GENDER value
  difference; a missing move; a non-request line; identical lines), plus 4 pinning the switch-probe
  content discriminator. **Run it after ANY change to `ALLOWLIST_TRANSFORMS` or the probe path.**
- **THE DRAIN / PROBE CONTRACT.** Every wait on a child is bounded and every bound is CHECKED:
  `assertDrained` on START + per decision, content-based acceptance on the trapped switch probe
  (`probeWasAccepted`, `PROBE_MAX_MS` 750 ms), and a per-battle wall-clock budget (`BATTLE_BUDGET_MS`
  300 s) over the outer loop. 🚨 **`drain_timeouts` must stay 0** — any non-zero value means a child
  went quiet somewhere, even on a path that recovers. Env overrides for investigation:
  `SBD_DRAIN_MAX_MS`, `SBD_PROBE_MAX_MS`, `SBD_BATTLE_BUDGET_MS`, `SBD_TRACE_DRAINS=1`, and
  `POKESIM_SIMBRIDGE_TARGET` (so two concurrent investigations never share one cargo target dir).
- 🚨 **`--persistent` is MANDATORY for a soak** — ~600 battles/hr with it, ~80/hr without (each battle
  otherwise respawns a Node child that reloads the whole Showdown dist), and ~96% of wall time is the
  per-write quiescence settle rather than CPU.

**HONEST SCOPE:** cross-side p1/p2 interleaving is not asserted (a Node scheduler artifact the Python
demux does not depend on); `__RECON__` is excluded (a real rust deferral — `resumeReseed` works,
`gen3_bridge_resume_reseed_v1`, so reconstruction + search paths still require node); unmodeled moves
fail loud, so "clean" is always relative to the modeled universe.

What landed in the green gate + allowlist, and the measured throughput:
[`designs/rust_sim/ab_fuzzer_findings.md`](ab_fuzzer_findings.md).
