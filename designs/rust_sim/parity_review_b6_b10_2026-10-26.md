# Adversarial parity review of B6–B10, BF, and the Struggle and Beat Up fixes (2026-10-26)

**Scope.** The gen-3 full-parity batches B6 (`44480ffe`), BF (`706faa0e`), B7 (`ca12e488`), B8 (`71296a26`),
B10 (`44820459`) and the two follow-up fixes, forced Struggle (`995c04a8`) and Beat Up King's Rock (`8dd36e9d`).
The goldens and fuzzers already show these batches match Showdown on the boards they test. This review looks for
what those boards do not cover. Four reviewers each read one batch family's Rust against Showdown's resolved gen-3
chain (`deps/pokemon-showdown`: `data/mods/gen3` → `gen4` → base, `sim/`) line by line. Every suspected gap was then
played on a hand-built board through the live differential described below.

**Who fixes what.** This is a review only; no Rust was changed. Fixes belong to the parity lead. Each confirmed row
here should become a named pin in `tests/regression_test.rs`, and each pin should be proven by reverting its fix and
watching it fail (`regression_pins.md`).

## The live differential used here

Each board is a pair of teams, a seed, and a choice plan, written the way `harness/gen_b10_moves_golden.js` writes
its boards (`set(...)`, `team(...)`, and `want` as `seq(rows)` or `M(a, b)`; forced switches are answered
automatically). A scratch driver ran each board in three steps:

1. **Play it in Showdown.** The driver plays the board directly with `Battle` to get the encoded choice script.
2. **Record the protocol.** It replays that script through `harness/gen_e2e_fuzz.js` `runBattle(..., {format,
   replayChoices})`, so the post-construction seed, the decisions and the omniscient lines are captured exactly as
   the byte fuzzer captures them. `emitBattle(..., {protocol: true})` then writes a protocol chunk.
3. **Byte-diff the port.** The chunk is replayed by the port's `ab_replay --protocol`, the same replayer
   `ab_fuzz.js --protocol` uses. Its verdicts are checked in precedence order: seed, then state, then protocol
   bytes.

The replayer was main's `target/selfcheck/ab_replay`. Its build stamp matched main at `8dd36e9d`, and no Rust
source has changed since then.

The harness was validated with a positive and a negative control:

- **Positive control:** `ctrl_tp_basic`, a board from the B10 golden, replays `ok`.
- **Negative control (injection):** planned, but not executed (see § Harness validation).

Two short-lane CPU jobs were run in total.

## Findings

OU-reachability uses the pre-B6 carry-rate table in `gen3_full_parity.md`; its git history has the B6–B10 rows,
which leave the live table once a batch ships. Moves modeled before B6 (Sleep Talk, Snatch, Conversion 2, Beat Up,
Eruption, Focus Punch) are not in that table. For those, "OU-reachable" means the combination is legal in `gen3ou`.

### Confirmed divergences (reproduced on a board)

**F1. A Present HEAL skips the target's Hit-event reactions: Rage and Focus Punch.** (B8)
- **Showdown:** `sim/battle-actions.ts` `runMoveEffects` still runs `runEvent('Hit', target, source, move)` after
  Present's successful heal branch. Present is Physical, so two Hit handlers fire:
  - Rage's `onHit` (`data/moves.ts` `rage.condition`) boosts Attack.
  - Focus Punch's `onHit` sets `lostFocus`, so the gen-4 `onTry` (inherited by gen 3) prints `|move|…|Focus
    Punch||[still]` then `|cant|…|Focus Punch|Focus Punch`.
  - A full-HP heal fail or a Substitute fail never reaches the Hit event.
- **Port:** `src/turn/variable_moves.rs` `run_present_heal` heals and returns. It calls neither `rage_on_hit` nor
  the Focus Punch flag that every other hit path sets.
- **Repro:** turn 2 is the heal turn in both boards.
  - Rage: Showdown prints `-heal|p2a: Snorlax|482/482`, then `|-boost|p2a: Snorlax|atk|1`. The port reports a
    `state` divergence at decision 1 (Delibird 162/252 vs Showdown's 142/252 after the boosted Rage).
  - Focus Punch: Showdown prints `cant … Focus Punch`, but the port lands the punch. That is a `seed` divergence at
    decision 1: the port draws a crit roll and a damage roll that Showdown never draws.

  ```js
  pr_heal_rage:       { format: 'gen3customgame', seed: [38,60,90,124], p1: team(set('Delibird','','Hustle',['present','splash'])), p2: team(set('Snorlax','','ThickFat',['rage','splash'])), turns: 3, want: M(1,1) },
  pr_heal_focuspunch: { format: 'gen3customgame', seed: [38,60,90,124], p1: team(set('Delibird','','Hustle',['present','splash'])), p2: team(set('Snorlax','','ThickFat',['focuspunch','splash'])), turns: 3, want: M(1,1) },
  ```
- **OU-reachable: yes.** Present has tier mean 0.001 % and OU 0.000 %, but appears in 0.013 % of OU replays; Focus
  Punch is common in OU. A gen3ou rerun of the Focus Punch board diverges the same way (§ Harness validation).

**F2. A move called by Sleep Talk does not become the user's `lastMoveUsed`.** (Older than B6; Conversion 2
reads the field.)
- **Showdown:** `data/mods/gen3/scripts.ts` `useMoveInner` sets `pokemon.lastMoveUsed = move` on every `useMove`.
  That includes the move Sleep Talk calls, so after Sleep Talk calls Earthquake, `lastMoveUsed` is `earthquake`.
- **Port:** `src/turn/moves.rs` sets `last_move_used` only inside the `!pursuit_strike && !sleep_talk_call` guard,
  so the outer `sleeptalk` stays recorded. The comment there says this matches the sim; it does not.
- **Repro:** on turn 3 Showdown prints `|-start|p1a: Porygon2|typechange|Flying` (it picked from the
  Ground-resisting types). The port prints `typechange|Rock` (from the Normal-resisting types). This is a `protocol`
  divergence at line 38.

  ```js
  conv2_sleeptalk_lastmoveused: { format: 'gen3customgame', seed: [31,47,61,73], p1: team(set('Porygon2','','Trace',['spore','conversion2','splash'])), p2: team(set('Snorlax','','ThickFat',['sleeptalk','earthquake'],{g:'F'})), turns: 3, want: seq([[1,2],[3,1],[2,1]]) },
  ```
- **OU-reachable: yes.** Porygon2's Conversion 2 against any Sleep Talk user is legal in gen3ou. The board's Spore
  on Porygon2 is customgame-only; two gen3ou variants (Rest + Sleep Talk Snorlax) diverge (§ Harness validation).

**F3. A move stolen by Snatch does not become the snatcher's `lastMoveUsed`.** (Older than B6. This confirms the
"open" row in `gen3_full_parity.md`.)
- **Showdown:** the same `useMoveInner` runs for the snatcher's `useMove(stolen)`.
- **Port:** the steal in `src/turn/status_moves.rs` runs the stolen move without setting `last_move_used`.
  Meanwhile B6's `magic_coat_bounce` does set it, so the two nested-move paths now disagree.
- **Repro:** on turn 2 Showdown prints `typechange|Ghost` and the port prints `typechange|Steel` (a `protocol`
  divergence).

  ```js
  conv2_snatch_lastmoveused: { format: 'gen3customgame', seed: [31,47,61,73], p1: team(set('Porygon2','','Trace',['swordsdance','conversion2','splash'])), p2: team(set('Snorlax','','ThickFat',['snatch','splash'],{g:'F'})), turns: 2, want: seq([[1,1],[2,2]]) },
  ```
- **OU-reachable: yes**, for example a Snatch user against a Conversion 2 user whose Recover it steals.

**F4. Beat Up is blocked by a Wonder Guard holder that is not weak to Dark.** (Beat Up is older than B6; it sits
next to B10's Beat Up King's Rock fix.)
- **Showdown:**
  - gen-3 Beat Up's `onModifyMove` sets `move.type = '???'` (`data/mods/gen3/moves.ts`).
  - The gen-4 Wonder Guard `onTryHit`, inherited by gen 3, returns early for a `???` move (`data/mods/gen4/abilities.ts`).
  - So Beat Up hits any Wonder Guard holder.
- **Port:** the Wonder Guard gate in `src/turn/moves.rs` reads the dex type (Dark) before the branch into
  `run_beat_up`. Shedinja itself is weak to Dark, which hides the bug. A Trace user that copied Wonder Guard
  exposes it.
- **Repro:** on turn 2 Showdown prints `|move|p1a: Houndoom|Beat Up|p2a: Porygon2`, two `-activate`/`-damage`
  pairs, then `-hitcount|…|2`. The port blocks the move with `-immune … Wonder Guard` and skips every strike's
  draws. Both boards show a `seed` divergence at decision 1.

  ```js
  bu_traced_wonderguard:    { format: 'gen3customgame', seed: [31,47,61,73], p1: team(set('Shedinja','','WonderGuard',['splash']), set('Houndoom','','FlashFire',['beatup','splash'])), p2: set('Porygon2','','Trace',['splash','tackle']), turns: 3, want: seq([['switch 2', 1], [1, 1]]) },
  bu_traced_wonderguard_ou: { format: 'gen3ou', seed: [31,47,61,73], p1: team(set('Shedinja','','WonderGuard',['splash','shadowball']), set('Sneasel','','InnerFocus',['beatup','splash'])), p2: set('Porygon2','','Trace',['splash','return']), turns: 3, want: seq([['switch 2', 1], [1, 1]]) },
  ```
- **OU-reachable: yes.** The second board is gen3ou and diverges. It is rare: it needs Shedinja, a Trace
  Porygon2 and a Beat Up user on the field together.

**F5. A base power that reaches 0 after the BasePower modifiers is lifted to 1.** (Older than B7; B7's sports
widened it.)
- **Showdown:** `sim/battle-actions.ts` `getDamage` has `if (!basePower) return 0;` right after
  `runEvent('BasePower')`. The event's last `modify()` can floor a sub-1 base power to 0. The function then returns
  before `modifyDamage`, so:
  - the crit roll has already been drawn,
  - no `random(16)` damage roll is drawn,
  - no crit or effectiveness line is printed,
  - the target gets a 0-delta `|-damage|<t>|<unchanged hp>`.
- **Port:** `src/damage.rs` `bp = bp.max(1);` and the damage path carries on with a draw, an effectiveness line and
  at least 1 HP of damage.
- **Which move gets there:** the only gen-3 move that does is Eruption below roughly 1/150 of max HP. Its base power
  is below 1, and Water Sport or Thick Fat halves it to 0.
- **Repro:** turn 1, Endure leaves Camerupt at 1/302. Turn 2, Eruption deals 0 in Showdown
  (`|-damage|p2a: Swampert|362/362`; with Thick Fat, `Snorlax|482/482`). The port shows a `seed` divergence at
  decision 1.

  ```js
  bp0_watersport_eruption: { format: 'gen3customgame', seed: [31,47,61,73], p1: team(set('Camerupt','','MagmaArmor',['endure','eruption'])), p2: team(set('Swampert','','Torrent',['earthquake','watersport'])), turns: 3, want: seq([[1,1],[2,2]]) },
  bp0_thickfat_eruption:   { format: 'gen3customgame', seed: [31,47,61,73], p1: team(set('Camerupt','','MagmaArmor',['endure','eruption'])), p2: team(set('Swampert','','Torrent',['earthquake','splash']), set('Snorlax','','ThickFat',['splash'])), turns: 3, want: seq([[1,1],[2,'switch 2'],[2,1]]) },
  ```
- **OU-reachable: yes, very rarely.** Camerupt (Eruption, Endure) into Snorlax's Thick Fat needs no B7 move at all.
  Water Sport is on Milotic and Whiscash; its tier mean is 0.000 %. A gen3ou rerun of the Thick Fat board diverges
  (§ Harness validation).

### Plausible (not reproducible on the battle log)

**P1. Safeguard is missing from the search snapshot's side conditions.** (Older than B7.)
- `src/search.rs` lists Light Screen, Mist, Reflect and Spikes in `side_conditions`, but not Safeguard, which
  Showdown's `sideConditions` holds.
- Only a search snapshot taken while Safeguard is up would differ; the battle log does not. This needs a search-driver
  differential to confirm.
- **OU-reachable:** yes (Safeguard is OU-legal).

**P2. The order of tied end-of-turn timers.** (B7.)
- The port gathers the Lock-On timer last among a mon's turn-end timers; Showdown uses the order the volatiles were
  added.
- Lock-On's end is silent, and the draw count is unchanged. The reviewer found no board where this shows today. It
  would surface only if a later batch makes one of the tied handlers print or draw.

**P3. Mirror Move after a Magic Coat bounce.** (B6 × B11.)
- Showdown's bounced `useMove` records `attackedBy` on the original user, and gen-3 Mirror Move reads it. Mirror
  Move is unmodeled (B11), so nothing reaches this yet.
- B11 must cover it, along with:
  - the "last move used" rule of F2 and F3 for Metronome-, Assist- and Nature-Power-called moves;
  - Metronome, Assist, Mirror Move and Sketch calling the B10 moves.

### Cleared (the board matched Showdown, or the reading found no divergence)

These played `ok` on the live differential:

- **Magic Coat:**
  - called through Sleep Talk;
  - a bounced Yawn into an already-statused user (`[from] Magic Coat|[still]` then `-fail`);
  - the holder bouncing a move it has Disabled;
  - a bounced Growl into a user under Mist;
  - a bounced Leech Seed into a Liquid Ooze user;
  - Synchronize back through the holder's Substitute.
- **Foresight:** before and after it, Rapid Spin into a Ghost (`-immune`, then the hazard and seed clears).
- **Bide:**
  - a Rough Skin KO of the Bide user on release (no `[silent]` end on a 0-HP mon);
  - the release boosting a Rage target;
  - a faint that cancels the wait turn, after which the release falls back to the replacement;
  - a release that misses a Fly user who charged the previous turn (the cross-turn `[miss]` attribute is a no-op).
- **Endless Battle Clause:** all four boards reach turn 100 or later.
  - Party order after a switch decides the loser test (`Each side's team …` + `|tie`).
  - A fainted bench mon cannot be switched to (`P1's team … thus loses.`).
  - A side trapped by Block, the trap-move path rather than Shadow Tag.
  - A Skill Swap back clears the user's own staleness: the control runs to `|turn|105` without triggering.

The reviewers' reading found no divergence in the following:

- **B6:**
  - Magic Coat's place in the TryHit chain, its PP and `lastMoveUsed` handling, and its turn-top clear;
  - Swagger and Flatter at ±6 (the zero-delta `-boost`), Own Tempo, Safeguard and already-confused;
  - Foresight through every damage, immunity and Wonder Guard path, Triple Kick's re-roll, and Baton Pass (not
    passed).
- **B7:**
  - Stockpile at 3 layers and Spit Up / Swallow at 0 layers, with no draws; Snatch before the layer check; not
    Baton-Passed;
  - Spit Up's cleanup and Pressure order; Swallow's `modify` rounding and the full-HP `-fail|heal`;
  - the sports' place in the BasePower chain (gen 3 has no type-item BasePower handlers) and their holder filter;
  - Lock-On through every accuracy site and every semi-invulnerable site, Minimize, and the OHKO path (BF17);
  - Mist on every foe-drop path, after Clear Body, not on self-drops, its `[silent]`, and the -6 floor.
- **B8:**
  - Bide's internals: no ModifyMove, the miss / immune / Protect / Wonder Guard order, and its store;
  - Present's no-Update outcomes; it is not on the King's Rock list;
  - contact abilities on 0-damage hits;
  - Triple Kick's per-strike re-roll through Lock-On;
  - OHKO with Lock-On, and Sturdy before the Substitute;
  - Psywave's draw order and rounding;
  - Magnitude's `-activate` and its category re-derivation.
- **B10:**
  - **The clause's timing:** after `turn++`, before the `|turn|` line and the Quick Claw roll.
  - **Who counts as stale:**
    - the restorative-berry list matches `sim/pokemon.ts`;
    - pending staleness is set and cleared on every Trick, Thief, Covet, Recycle and Knock Off path;
    - Skill Swap's staleness is set and cleared correctly.
  - **Role Play:** the fail set (Wonder Guard only in gen 3), the End then `-ability` order with no Start, and the
    revert at switch.
  - **Camouflage:** the type reset at switch and at a forme change, and Transform.
  - **Teleport, Follow Me and Helping Hand:** the fail lines, `[notarget]`, PP and priority.
- **The Struggle fix:** priority, recoil, secondaries, King's Rock, the PP and Disable gates, and the Focus Punch,
  Counter and Pursuit setup all key on the synthetic Struggle. No other stale-slot field leaks.
- **Beat Up (apart from F4):**
  - the strike list leaves out statused and fainted mons;
  - Shell Bell, Focus Band, Endure, Mirror Coat, Rage, Destiny Bond and Grudge run per strike;
  - Color Change and contact abilities do not fire, as for a `???` move with no contact flag.

### Doc and comment errors found on the way

- **`src/state.rs` and `src/turn/moves.rs`:** the comments on `last_move_used` say a Sleep-Talk-called move leaves
  the outer move recorded "matching the sim". F2 shows the sim records the called move.
- **`ROLE_PLAY_FAIL_ABILITIES`:** its comment cites `data/mods/gen6/abilities.ts:125`, which is Zen Mode. The list
  itself is right; the B10-0 pin checks it against the dex.

## Harness validation

**Job 1** played all 23 boards in the sections above.
- **Result:** 16 `ok` and 8 diverged (one per repro board in F1–F5).
- **Control:** `ctrl_tp_basic` replayed `ok`.
- **Cross-check:** every board the reading predicted would diverge did, and every board predicted to match did.

**Job 2** reran the confirmed divergences under `gen3ou`.
- **`ou_thickfat_eruption` (F5):** Thick Fat only, no B7 move. Diverges with a `seed` divergence at decision 1.
- **`ou_present_heal_focuspunch` (F1):** a `seed` divergence at decision 1.
- **`ou_conv2_sleeptalk_a` / `_b` (F2):** two seeds. Porygon2 uses Seismic Toss, Snorlax Rests, Sleep Talk calls
  Earthquake, then Conversion 2. Both show a `seed` divergence at decision 2.
  - Showdown picks from the Ground-resisting types (`typechange|Bug` and `typechange|Flying`). The port picks from
    the Normal-resisting types, so Snorlax's next Earthquake draws differently.
  - Boards: `p1: set('Porygon2','','Trace',['seismictoss','conversion2','splash'])`,
    `p2: set('Snorlax','','ThickFat',['rest','sleeptalk','earthquake'],{g:'F'})`,
    `want: seq([[1,1],[3,2],[2,2]])`, seeds `[31,47,61,73]` and `[5,9,13,17]`.

**FINDING: the negative-control injection was NOT executed.**
- **What failed:** the injection step was meant to move one `-damage` HP value on the control board by one. The
  control board's only `-damage` line is a KO (`0 fnt`). In both jobs the mangling regex therefore matched nothing:
  the job-1 copy went in unchanged, and in job 2 the script raised an error. The replayer then correctly read the
  unmangled chunk as `ok`.
- **Why it was not rerun:** the brief allowed two short-lane jobs, and both were used.
- **The evidence that remains:** the harness separated the boards exactly as an independent code reading predicted,
  12 diverged and 16 ok. Two of the divergences were byte-level `protocol` verdicts at the predicted line.
- **What the parity lead should do:** when pinning F1–F5, rerun the boards with a corrected injection (a board
  whose control hit leaves HP above 0).
