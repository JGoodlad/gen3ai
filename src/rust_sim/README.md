# pokesim — Rust Pokémon Showdown sim (Gen 3 OU singles first)

A from-scratch, **bit-for-bit-faithful** Rust reimplementation of the Pokémon
Showdown battle simulator. Goal: given the same seed, teams, and choice
sequence, emit byte-identical protocol output to upstream Showdown — so it can
slot in behind our existing bridge/tooling without changing any results.

> **Status: skeleton, eight modules built.** The PRNG, the dex, the team codec,
> in-battle stat computation, construction-time battle state, the single-hit
> damage calc, the **event-dispatch core + `>start` switch-in events** (Intimidate
> / Sand Stream / Drizzle / Drought), **multi-turn move execution + residuals**
> (`BattleState::run_turn` runs the FULL per-turn cycle — the action-order +
> per-action `eachEvent('BeforeTurn'/'Update'/'Weather')` speed-tie shuffles,
> accuracy/crit/damage rolls in the exact order, immune short-circuit, the
> deferred-faint protocol, the END-OF-TURN RESIDUALS [weather chip / Leftovers /
> burn /8 / poison /8 / Toxic ramp, in gen-3 residualOrder], and the Quick Claw
> roll; `BattleState::run_battle` LOOPS it, stopping at the first faint), and
> **SWITCHING → a FULL battle to WIN/LOSS** (`BattleState::run_full_battle` takes a
> `Choice::Move`/`Choice::Switch` script and plays to game-end — voluntary switches
> sort before moves with the two-switch action-order tie-shuffle; the gen-3 draw-FREE
> switch-in ability `Start`; the `switchIn` position swap; POST-FAINT replacement,
> single + DOUBLE with the double's `insertChoice` splice draw; and win/loss — a side
> out of mons loses, both → a gen-3 tie) are done and each proven against a
> differential golden. The turn layer is proven by a **per-seed CROSS-TURN STATE +
> post-turn-PRNG-SEED differential**, and the full-battle layer by a **per-seed
> PER-DECISION STATE+SEED+winner differential to game-end** — the bit-for-bit
> RNG-consumption-order proof, sustained across switches + replacements until a side
> is out of mons. **SECONDARY effects on damaging moves (the per-move `random(100)`
> after a landed hit — par/frz/flinch/psn, Serene Grace x2 / Shield Dust x0, the
> onTrySetStatus gates) and the onBeforeMove STATUS draws (the new leading draw before
> accuracy — paralysis 25% full-para, sleep wake, freeze 20% thaw, confusion 50%
> self-hit, flinch) are also done**, proven by a per-seed PER-DECISION
> STATE(+status)+SEED+winner differential to game-end where status is inflicted
> IN-ENGINE. **THE CAPSTONE** (`tests/e2e_fuzz_test.rs` + `harness/gen_e2e_fuzz.js`)
> drives BOTH engines over **REAL Showdown teams** (the 770 `data/teams/*.txt`, imported
> + gen3ou-validated + packed) for **complete random battles to game-end**, with each
> decision a random legal choice RESTRICTED to the modeled move/ability/item set —
> asserting per-decision state + status + boosts + confusion + running PRNG seed +
> winner **bit-for-bit** across **all 220/220 battles** (STRICT — `filtered_diverged == 0`
> over every battle, no escape hatch; 10636 decisions at the current BATCH-4-admitted corpus
> (filter-clean teams **719/719 — the ENTIRE real-gen3ou pool**), of which 4210 USE SPIKES, 353
> USE PHAZE, 612 USE EXPLOSION,
> 343 USE SUBSTITUTE, 114 USE TAUNT (0 USE DISABLE — no sample team carries it), and 201 involve a
> TRAPPED mon. **BATCH 4 (`gen3_ability_batch4_v1`) — the FINAL mechanics tail — is WIRED,
> golden-proven + e2e-admitted**: TRUANT (the priority-9 loaf cant + the order-27 residual toggle;
> the last team-carry gap, =4), INNER FOCUS (block-at-the-flinch-APPLY — the roll still draws),
> SHADOW TAG (unconditional draw-free trap, mutual mirror), CUTE CHARM + the ATTRACT volatile
> (the unconditional 1/3 contact roll, gender-gated inside the volatile add; -activate + 1/2
> onBeforeMove cant at priority 2), COLOR CHANGE (`types_override` through the ONE `mon_types`
> choke point), KING'S ROCK (the appended trailing 10% flinch secondary over the
> execution-derived 130-move list; [own]→[KR]→[contact-proc] order) and FOCUS BAND (the
> onDamage 1/10 drawn on EVERY Damage event into the holder; survive-at-1 on a lethal MOVE hit)
> — validated by `gen_ability_batch4_golden.js` → `tests/ability_batch4_test.rs` (1260 game-end
> battles, per-decision STATE+HP+STATUS+TRAPPED+SEED, md5-pinned) + the B4-1..B4-7
> revert-verified pins. The admission grew the pool **712 → 719/719** and the honest taxonomy's
> 300-battle UNFILTERED sweep is 300/300 clean with an EMPTY ability+item gap list; the ONLY
> deferred member is FORECAST (a Castform forme-change, 0 teams). **The BATCH-2 DRAW-BEARING "reactive" ability classes are now MODELED + e2e-admitted**
> (`gen3_ability_batch2_v1`): CONTACT_PROC (Static par / Poison Point psn / Flame Body brn / Effect Spore
> slp|par|psn — an `onDamagingHit` that, on a CONTACT hit into the holder, draws `randomChance(1,3)` [or
> Effect Spore's `random(10)`+`sample(3)`] and statuses the ATTACKER — drawn AFTER the move's own secondary
> `random(100)`), CONTACT-recoil (Rough Skin, maxhp/16 draw-free), BLOCK (Soundproof immune to sound moves /
> Damp cancels Explosion at TryMove with no self-KO / Suction Cups blocks a phaze DRAG with no `sample`), and
> Synchronize (reflect a foe status back to the source, draw-free in customgame). The pool grew **571 → 585**
> (`synchronize` [the #1 taxonomy gap] + `effectspore` the levers), a CLEAN STRICT pass first-try, NO new
> engine bug. **BATCH 3 (`gen3_berry_trace_shedskin_v1`) is now WIRED + golden-proven** — the BERRY item
> classes (ONE eatItem consumption mechanism + 22 data-driven parameter rows in `gen3_items.json`
> `berryEffect`: 7 cure [lum immediate-in-setStatus] + 7 heal [residual subOrder-4, 2*hp<=maxhp; the
> resolved Figy family heals maxhp/8] + 7 pinch [4*hp<=maxhp; Starf's sample; Lansat's focus-energy] +
> Leppa), TRACE (the n=1 randomFoe `sample` + a LIVE current-ability copy, no copied-onStart in gen3), and
> SHED SKIN (randomChance(33,100) per STATUSED residual at subOrder 3, cure before the DoT) — validated by
> `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` (1280 game-end battles, per-decision
> STATE+STATUS+ITEM+BOOSTS+SEED, byte-reproducible) + the BR1-BR6 revert-verified pins (BR6 = the
> exact-equality `<=` boundary on an EVEN-maxhp board — `probe_berry_threshold_boundary.js` settled that
> the sim EATS at hp == maxhp/2 / maxhp/4 exactly). **The e2e admission is DONE**: the 22 berries →
> MODELED_ITEMS + trace/shedskin → MODELED_ABILITIES grew the filter-clean pool **585 → 712 / 719** (the
> biggest admission since Natural Cure; lumberry=64 + salacberry=46 + trace=9 the levers), a CLEAN STRICT
> pass first-try, NO new engine bug; the then-remaining `truant`=4 + `innerfocus`=2 gaps have since
> been closed by batch 4 (above).
> **The BATCH-1 ability classes are also MODELED + e2e-admitted** (`gen3_ability_batch1_v1`):
> CRIT_IMMUNE (Shell Armor / Battle Armor — a hit into the holder NEVER crits, the crit roll DRAWN then
> overridden false), WEATHER_SPEED (Chlorophyll / Swift Swim — ×2 speed in sun/rain), WEATHER_NEGATE
> (Cloud Nine / Air Lock — suppresses the weather's effects), RESIDUAL (Speed Boost +1 spe/turn / Rain
> Dish +maxhp/16 in rain), all DRAW-FREE — plus the class-(a) no-ops Plus/Minus/Lightning Rod/Sticky
> Hold; the pool grew **525 → 571** (Shell Armor the lever), a CLEAN STRICT pass first-try. **A STEP-1
> weather fix shipped with it:** gen3 sun/rain fire the end-of-turn `eachEvent('Weather')` tie-shuffle
> UNCONDITIONALLY (the port used to gate it on Sand|Hail) — a weather-turn speed tie under sun/rain drew
> one fewer call → FIXED (schedule off RAW weather for sun/rain, off `effectiveWeather()` for sand/hail).
> FORECAST is DEFERRED (a Castform forme+TYPE change under rain/sun/hail — not a no-op).
> **NATURAL CURE is MODELED + e2e-admitted** (`gen3_natural_cure_v1`, the sole gen-3
> SWITCH_OUT-cure ability — the holder's major status is CURED when it switches OUT, voluntary OR
> phaze-drag, DRAW-FREE): it was the #1 team-carry gap (naturalcure=254 on Blissey/Starmie/Celebi/
> Miltank/…), so admitting it grew the filter-clean pool **151 → 449** — the BIGGEST single admission
> lever yet — and the enlarged corpus is a CLEAN STRICT pass (no new engine bug surfaced). The former
> residual-vs-faint-under-weather ordering gap is **FIXED** via per-handler `faintMessages` + the cached
> `pokemon.speed` model; the recovery expansion surfaced + FIXED a residual HANDLER GATHER-ORDER bug
> [status DoT gathered before Leftovers per mon, mirroring `findPokemonEventHandlers`]; the PROTECT
> expansion surfaced + FIXED the `protect`/`stall`/`flinch` RESIDUAL duration handlers [duration-bearing
> volatiles register residual handlers that tie in the speed-sort] + the no-delete-on-fail gen3 stall model
> + the `willAct()` switch gate; and the SPIKES expansion surfaced + FIXED the FORCED-REPLACEMENT
> `updateSpeed` on commit [`commitChoices()` refreshes cached speed on EVERY commit incl. a mid-turn forced
> switch — so a foe paralyzed mid-turn drops to its para speed before the resumed Updates, instead of
> spuriously tying the fresh entrant on its stale full speed]). A separate **coverage taxonomy** (`e2e_fuzz_taxonomy.txt`) ranks coverage gaps by
> STATIC TEAM COMPOSITION — which unmodeled ability/item the paired teams CARRY (with the BATCH-1 classes
> now MODELED, **Synchronize + Effect Spore + Trace (the DRAW-BEARING batch-2 procs), and Lum/Salac berries,
> are the new top gaps**; Shell Armor + Natural Cure [naturalcure=254] + Immunity [immunity=97] have DROPPED
> OFF the list;
> Magnet Pull + Arena Trap AND the **ability DMG_MOD
> class** — Torrent/Blaze/Overgrow/Swarm pinch BP ×1.5, Huge/Pure Power Atk ×2, Guts Atk
> ×1.5-statused + burn-suppressed, Marvel Scale Def ×1.5-statused, all data-driven from
> `AbilityData.dmg_mod` — are ENGINE-MODELED **AND e2e-admitted** (`gen3_sun_freeze_immunity_v1`,
> filter-clean teams 22 → 151; the admission-gating SUN-freeze gap + a `wisp` move-alias cascade
> both FIXED — see EDGE_CASES §5) — NOT by observed divergence cause, and
> move-level-blind. **The ACCURACY pipeline is now WIRED** (`gen3_accuracy_pipeline_v1`): the to-hit
> roll is `effAcc = move.accuracy × the acc/eva STAGE TABLE × the accMod item/ability handlers`, then
> the ONE `random(100) < effAcc` draw — DRAW-RELEVANT (a wrong effAcc flips a hit/miss → the seed
> desyncs). Members (all from the RESOLVED `Dex.mod('gen3')` — the mod-chain law): the acc/eva stage
> table `[3/3,4/3,5/3,6/3,7/3,8/3,9/3]`, Bright Powder ×0.9 / Lax Incense ×0.95 (ACCURACY_ITEM), Compound
> Eyes ×1.3 / Sand Veil ×0.8-in-sand (+ its sand-chip immunity) / Hustle ×0.8-physical (its Atk ×1.5
> dmgMod ships too), with the runEvent integer-guard (a stage/direct-multiply float SKIPS the chain
> members). Data-driven via `dex/accmod.rs::AccMod` → `turn.rs::effective_accuracy`; validated by
> `tests/accuracy_test.rs` (per-decision STATE+HP+SEED to game-end) + the AC1-AC5 pins; **~70 Mud-Slap
> acc-stage A/B repros now replay `ok`** (the empty path is byte-identical to the raw roll, so the e2e
> stays STRICT 220/220 — Bright Powder/Sand Veil/Compound Eyes/Hustle are too rare in gen3 OU teams to
> become filter-clean). **FLASH FIRE's ×1.5 fire-boost is now MODELED** (`gen3_flashfire_boost_v1`) —
> the deferred FF gap + the A/B fuzzer's evidence-based **#1 STATE cluster** (fireblast+flamethrower;
> 397/402 fire-move STATE repros carry an FF mon → the sim deals MORE fire damage than the port). It also
> COMPLETES the type-interaction ability class (Levitate immunity, Water/Volt Absorb heal+immunity were
> done — FF's boost was the last gap). PROBE-SETTLED (`probe_flashfire_rng.js` over the RESOLVED gen3 dist):
> ACTIVATION is the `flashfire.onTryHit` — a `flash_fire: bool` volatile ARMED on a LANDED Fire hit
> (DRAW-FREE, a MISSED Fire move does NOT arm it, skips a `frz`-status holder, cleared on switch-out+faint);
> the BOOST is the volatile's `onModifyDamagePhase1 chainModify(1.5)` — a DAMAGE-PHASE fold (the SAME phase
> as screens, category-agnostic, NOT crit-bypassed — NOT an `onModifyAtk`/`onModifySpA` stat mod, which are
> `undefined` in the resolved dist), ACCUMULATED with any screen into ONE Phase1 chain modifier (sequential
> per-mod rounds diverge for ~¼ of baseDamage). It is an ENGINE FLAG (a volatile + a damage fold), not a
> `dmgMod` data row. Validated by the class-sweep golden `gen_flashfire_golden.js` → `flashfire_test.rs`
> (STATE+HP+SEED to game-end + two calc-level EXACT max-roll pins for the ×1.5 and the FF⊗Light-Screen
> combine) + 3 revert-verified `flash_fire_*` pins. The e2e stays **STRICT 220/220 byte-unchanged**
> (flashfire was already modeled for immunity → the 151-team pool is unchanged); **of 200 replayed FF-team
> STATE repros, 185 (92.5%) flip to `ok`**, and a boost-revert re-diverges them.
> **NATURAL CURE is now MODELED** (`gen3_natural_cure_v1`) — the sole gen-3 SWITCH_OUT-cure ability class,
> and the **#1 e2e team-carry gap** (naturalcure=254 on Blissey/Starmie/Celebi/Miltank/…). The holder's
> MAJOR STATUS (any of brn/par/psn/tox/slp/frz; the tox stage + sleep counter reset too) is CURED when it
> SWITCHES OUT. PROBE-SETTLED (`probe_naturalcure_rng.js` over the RESOLVED gen3 dist): the trigger is
> `naturalcure.onSwitchOut` (`onCheckShow` is **undefined** — resolving the long-deferred "NaturalCure
> CheckShow" draw question: there is NO CheckShow gate); it fires in `switchIn`'s `runEvent('SwitchOut')`
> on BOTH a VOLUNTARY pivot AND a phaze-DRAG-out (only `BeforeSwitchOut` is `!isDrag`-gated), BEFORE
> `clearVolatile`; it is a NO-OP on a FAINT (the `status==='fnt'`/empty guard); and it is **DRAW-FREE** —
> the cure + its `[silent]` `-curestatus` reveal consume ZERO PRNG (SEED byte-identical to a non-NC pivot),
> so admitting it is seed-neutral for every pre-existing suite. It is an ENGINE FLAG (a `status = None`
> clear at the switch-out site in `turn.rs::execute_switch`, gated on an ALIVE outgoing `naturalcure`
> holder), NOT a data row — `gen3_abilities.json` is unchanged (obs-neutral; the Python facade + the
> extractor `--check` gate are untouched). Validated by the class-sweep golden `gen_naturalcure_golden.js`
> → `naturalcure_test.rs` (7 scenarios × 40 seeds = 280 game-end battles, per-decision STATE+STATUS+SEED;
> the cure is observable on the ACTIVE-status timeline — an NC mon statused → pivots out CURED → RETURNS
> UNSTATUSED, vs a non-NC control that RETURNS still statused + a phaze-drag scenario + an unstatused/faint
> no-op) + 3 revert-verified `natural_cure_*` pins (NC1 voluntary, NC2 faint no-op, NC3 phaze-drag). It is
> the **BIGGEST single e2e-admission lever yet** — the filter-clean pool grew **151 → 449 / 719** and the
> enlarged corpus is a CLEAN STRICT pass (no new engine bug surfaced).
> **The STATUS_IMMUNE ability class is now MODELED + e2e-admitted** (`gen3_status_immune_v1`, DATA-DRIVEN) —
> the gen-3 abilities that grant immunity to a specific MAJOR status, the **#2 e2e team-carry gap** (immunity=97):
> **Limber** (par) / **Insomnia** + **Vital Spirit** (slp) / **Immunity** (psn,tox) / **Water Veil** (brn)
> block via `onSetStatus`; **Magma Armor** (frz) via `onImmunity` (BEFORE the SetStatus event). Own Tempo
> (confusion) + Oblivious (attract) block a VOLATILE not a status → NOT members (Leaf Guard is num 102 = NOT
> gen-3). It emits a `statusImmune {statuses, phase}` field into `gen3_abilities.json` via the extractor
> (`_GEN3_ABILITY_MECHANICS`, drift-gated by `dump_gen3_mechanics.js --check`, obs-neutral) →
> `dex/abilities.rs::AbilityData.status_immune` → `turn.rs::try_set_status` reads it (the `Immunity` phase
> gates BEFORE `set_status_event_shuffle`, the `SetStatus` phase AFTER). PROBE-SETTLED draw model
> (`probe_statusimmune_{rng,setstatus_event,shuffle_size,magmaarmor,enumerate}.js`): DRAW-FREE in
> gen3customgame (the ability is the SetStatus event's ONLY handler → no shuffle) — so admission is SEED-CLEAN.
> **The SURPRISE that killed the old "size-3 shuffle" fail-loud**: in gen3ou an `onSetStatus`-phase ability
> adds a 3rd SetStatus handler, but it sorts into its OWN speed group (its defined `speed` beats the clauses'
> `undefined`), leaving the 2 clauses a SIZE-2 tie → `shuffle(list,1,3)` draws EXACTLY ONE `random`, identical
> to the control's `shuffle(list,0,2)` (draw COUNT unchanged). Validated by the class-sweep golden
> `gen_statusimmune_golden.js` → `statusimmune_test.rs` (12 scenarios × 40 seeds = 480 game-end battles, the
> block observable on the active-status timeline + a stable-md5 byte-reproducibility gate) + 4 revert-verified
> pins `limber_blocks_paralysis_draw_free` / `insomnia_blocks_sleep_draw_free` [a draw-COUNT pin — a landed
> sleep draws `random(2,6)`] / `magma_armor_blocks_freeze` / `immunity_blocks_tox_but_not_burn`. Admitting the
> 6 members (+ moving `insomnia`/`vitalspirit` OUT of `NOOP_ABILITIES`) grew the filter-clean pool
> **449 → 525 / 719** (+76 teams, immunity=97 the #2 gap); the enlarged corpus is a STRICT
> `filtered_diverged == 0` pass (220/220, 11651 decisions, byte-reproducible). It surfaced + FIXED ONE real
> engine bug (NOT the STATUS_IMMUNE class) — the **EMPTY NATURE**: e2e_8/e2e_73 carry a Suicune with an
> OMITTED nature field, which the sim treats as NEUTRAL (Serious) but the port PANICKED on; `stats.rs` now
> computes the neutral all-1.0 multipliers for an empty nature (VERIFIED vs the sim, pinned
> `empty_nature_computes_the_neutral_stats`). `immunity` DROPPED OFF the taxonomy top-gaps list.
> **Standalone STATUS MOVES are MODELED** (Thunder Wave / Stun Spore /
> Glare / Poison Powder / Poison Gas / Toxic / Will-O-Wisp / the 6 sleep moves),
> **SELF-TARGETING SETUP / STAT-BOOST MOVES** (Calm Mind / Dragon Dance / Swords Dance /
> Agility / Bulk Up / Amnesia / Tail Glow / the +Def & +Atk one-stat moves — 17 in all,
> data-driven from the `selfBoosts` dex field): never-miss → no accuracy draw, a DRAW-FREE
> `boost()` apply, and the +Speed CACHED-SPEED interaction (a Dragon Dance / Agility flips the
> first-mover on the FOLLOWING turn, bit-exact); and now **SELF-HEAL / RECOVERY MOVES too**
> (Recover / Soft-Boiled / Slack Off / Milk Drink → `floor(maxhp/2)`; Moonlight / Synthesis /
> Morning Sun → the gen4-inherited PLAIN-integer weather heal `floor(maxhp/2)` / `floor(maxhp*2/3)`
> sun / `floor(maxhp/4)` sand+rain+hail; **Rest** → full heal + cure + a FIXED `Sleep(3)` whose
> `slp.onStart` STILL draws-then-DISCARDS one `random(2,6)` — the verified draw-COUNT crux):
> never-miss → no accuracy draw, a DRAW-FREE heal, the full-HP FAIL path. Each has its own
> differential golden (`gen_status_move_golden.js`/`status_move_test.rs` in gen3ou for the Sleep
> Clause + SetStatus shuffle; `gen_setup_move_golden.js`/`setup_move_test.rs` for the boost-stage +
> first-mover-flip proof; `gen_recovery_move_golden.js`/`recovery_move_test.rs` for the heal-amount +
> Rest sleep/`random(2,6)` proof), and now **PROTECT / DETECT** (identical full-turn protection):
> the gen-3 consecutive-use STALL draw (the FIRST protect short-circuits with NO draw; a consecutive
> one draws one `randomChance(1, counter)` at the floored 2/4/8 denominator — the gen3 1/8 floor, via
> the gen4-inherited `stall` `counterMax: 8`; the counter resets after one non-protect/switch turn —
> the `stall` volatile's `duration: 2` expiry, modeled at the residual) + the move-BLOCK (a foe move
> TARGETING the protected mon draws its accuracy roll then is blocked BEFORE crit/damage/secondary/
> status, gen-3-`tryMoveHit`-ordered; a self-target move is never blocked). DEFERRED (fail-loud):
> Endure (`volatileStatus:'endure'`, a survive-at-1-HP `onDamage`) + the gen4+ Quick/Wide Guard /
> King's Shield (none in gen3). Its own differential golden (`gen_protect_move_golden.js`/
> `protect_move_test.rs`) asserts the per-decision STALL COUNTER + the block (HP unchanged) + seed,
> and the protect expansion surfaced + FIXED a real residual-model gap — **the `protect`/`stall`
> duration-only volatiles register residual handlers** (gathered by `findPokemonEventHandlers(...,
> 'duration')` to count down their `duration`) that PARTICIPATE in the residual speed-sort, so a
> protecting mon adds 2 tied handlers that change the tie-group shuffle COUNT (a draw-COUNT desync if
> omitted). The **e2e capstone picks the modeled status + setup + recovery + protect + SPIKES + LEECH
> SEED + PHAZE sets** (`isModeledMove`); **SUBSTITUTE + LEECH SEED + EXPLOSION/SELF-DESTRUCT + PHAZING
> (Roar/Whirlwind) are now INCLUDED** (`SUBSTITUTE_E2E_EXCLUDED = false`, `LEECHSEED_E2E_EXCLUDED = false`,
> `EXPLOSION_E2E_EXCLUDED = false`, `PHAZE_E2E_EXCLUDED = false` — 541 explosion / self-KO + **1035
> phaze-DRAG** decisions across the 220-battle strict gate, bit-for-bit). The PHAZE re-enable required
> ROOT-CAUSING the multi-phaze `sample` draw-POSITION desync: gen-3 **Roar / Whirlwind carry the
> `protect: 1` flag**, so a Protect / Detect BLOCKS the phaze at `TryHit` (after the accuracy roll) → NO
> drag → NO `sample` — the port's phaze arm was MISSING that block, so it dragged an EXTRA `sample` into a
> protected foe the sim left in place, shifting every LATER phaze's `sample` position (same total draw
> COUNT, wrong `sample` INDEX, compensated elsewhere → the boundary seed matched while the dragged mon was
> wrong). Fixed by a `protect_blocks` check in the phaze arm (pinned by
> `phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target`).
> The Explosion re-enable required FIXING two STATEFUL desyncs admitting it surfaced (NOT the self-KO): a
> **double-faint → double-replacement → cascade `runSwitch` cancellation** (gen-3 `faintMessages`
> `cancelAction(getAllActive)` drops the OTHER side's pending runSwitch, so the foe entrant is NOT re-chipped
> — the port had kept the stale runSwitch) and a **confusion self-hit × Choice Band** gap (gen-4 confusion
> runs the FULL getDamage, so CB folds in) — both DRAW-FREE, pinned deterministically (see below). The
> Substitute re-enable required
> FIXING a SWITCHING/weather bug — the **`eachEvent('WeatherChange')` switch-in tie-shuffle**: a MID-TURN
> switch-in whose entrant TIES the opposing active AND CHANGES the weather (Sand Stream / Drizzle / Drought)
> draws ONE `random(0,2)` tie-shuffle from `Field.setWeather` (field.ts:87) that the port MISSED (e2e_84
> dec4: a 213-Tyranitar switches in under Sand Stream vs a 213-Suicune → the sim drew 8, the port 7).
> `turn.rs`'s `run_switch` now reports a weather change and `turn_loop` fires the shuffle — pinned by
> `regression_test.rs::switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed` (a true SEED pin). The setup-move expansion also surfaced + FIXED a real-team-only engine
> bug — **Water/Volt Absorb is now accuracy-gated** (a MISSED Water/Electric move no longer heals the
> holder; the absorb is an `onTryHit` that fires only on a HIT). And now **SPIKES — the gen-3 ENTRY
> HAZARD, which introduces the first SIDE CONDITION** (`SideState::spikes`, a per-side persistent
> 0..=3 layer count that survives switches): the Spikes MOVE (`sideCondition:'spikes'`, `target:
> 'foeSide'`) is never-miss + DRAW-FREE and increments the FOE side's layer by 1 capped at 3 (a 4th
> FAILS), and a GROUNDED switch-in takes the gen-3 chip on the `runSwitch`'s `runEvent('EntryHazard')`
> (BEFORE the ability `Start` — a Spikes-KO skips it): `max(floor([_,3,4,6][layers]·maxhp/24),1)` =
> maxhp/8 ÷6 ÷4 for 1/2/3 layers, DRAW-FREE; a Flying-type / Levitate entrant takes ZERO; a spikes-KO
> on entry faints the mon → forces ANOTHER replacement (which ALSO takes spikes), wired through the
> existing faint/replacement machinery with no Quick Claw. Its own differential golden
> (`gen_spikes_golden.js`/`spikes_test.rs`) asserts the per-side SPIKES LAYERS + the switch-in chip
> (HP) + seed to game-end (5 scenarios × 80 seeds, 400 runs), and the e2e capstone now picks it
> (`MODELED_HAZARD_MOVES`) so real Skarmory/Forretress/Cloyster spikers lay Spikes + grounded
> switch-ins take the chip on the filtered gate. DEFERRED (fail-loud): Toxic Spikes / Stealth Rock
> (NOT gen3), Rapid Spin (the hazard-CLEAR move). And now **PHAZING — Roar + Whirlwind** (the gen-3
> `forceSwitch` moves: force the FOE to switch to a RANDOM eligible team member). The draw model,
> verified vs a sim PRNG probe (`probe_phaze_rng.js`): gen-3 Roar/Whirlwind resolve to **`accuracy:
> 100`** (NOT never-miss — the SURPRISE this layer surfaced!), so they DRAW `randomChance(100,100)`
> (always passes but CONSUMES a draw); a SUCCESSFUL phaze (the foe has ≥1 eligible bench mon) then
> draws ONE `sample`/`random(n)` (the random target — EVEN for n==1, the draw gotcha) and `dragIn`s the
> picked mon at the runAction tail (it takes Spikes via the existing `runSwitch` EntryHazard, fires its
> ability `Start`, and a Spikes-KO on entry chains a NORMAL replacement); a phaze with NO eligible foe
> (its last mon) FAILS draw-free (only the accuracy roll). **A Protect / Detect on the target BLOCKS the
> phaze** — gen-3 Roar / Whirlwind carry the `protect: 1` flag, so a Protect blocks the phaze at `TryHit`
> (after the accuracy roll) → NO `forceSwitchFlag` → NO drag → NO `sample` (Substitute does NOT block —
> Roar/Whirlwind carry `bypasssub: 1`). The phazed-OUT mon's boosts/volatiles are
> cleared; the dragged mon does NOT act this turn (priority −6 → the phazer moves last). Its own
> differential golden (`gen_phaze_golden.js`/`phaze_test.rs`) sweeps seeds so DIFFERENT mons get dragged
> (the random-target proof — ≥2 distinct drags per multi-bench scenario), and asserts the per-decision
> active species (which mon was dragged) + HP (the phaze-into-Spikes chip) + spikes layers + seed to
> game-end (7 scenarios × 80 seeds, 560 runs, 10388 seed + 20776 HP/spikes assertions, 2795 drags).
> **Phaze is now INCLUDED in the e2e capstone** (`PHAZE_E2E_EXCLUDED = false` — 1035 phaze-DRAG decisions
> across the 220-battle strict gate, bit-for-bit, `phaze_decisions >= 50` coverage floor) after fixing the
> multi-phaze `sample` draw-POSITION desync (= the MISSING Protect block above: the port dragged an EXTRA
> `sample` into a protected foe the sim left in place, shifting every LATER phaze's `sample` position —
> same total draw COUNT, wrong `sample` INDEX). Pinned by
> `phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target`. DEFERRED (fail-loud): **Haze** (resets
> boosts — a DIFFERENT mechanic, not
> forceSwitch), Perish Song, Roar of Time (not gen3). And now **LEECH SEED** (`leechseed`) — a
> foe-targeting Status move (type Grass, **accuracy 90**) that plants the `leechseed` volatile on the
> FOE; each end-of-turn the seeded mon loses `floor(maxhp/8)` and the SEEDER's CURRENT active heals it.
> The draw model, verified vs a sim PRNG probe (`probe_leechseed_rng.js`): the MOVE DRAWS
> `randomChance(90,100)` (it CAN miss) UNCONDITIONALLY — even into a Grass-immune (`onTryImmunity` →
> `-immune`, no volatile) or already-seeded target (re-seed FAILS, volatile unchanged) — then PLANTS the
> volatile on a landed hit (draw-free). The crux is **the residual**: the gen4-inherited override puts
> Leech Seed at **`onResidualOrder: 10, onResidualSubOrder: 5`** — so the residual ladder is sand
> `[o=8]` → Leftovers `[s=4]` → **LEECH `[s=5]`** → status DoT `[s=6]`, all DRAW-FREE (the leech drain
> `floor(maxhp/8)` clamped to the seeded mon's HP, the seeder's active heals that dealt amount; a
> FAINTED seeder skips the whole drain). Two seeded mons at equal speed TIE at order 10 sub 5 → one
> residual handler-sort shuffle. Its own differential golden (`gen_leechseed_golden.js`/
> `leechseed_test.rs`) sweeps 7 scenarios × 80 seeds (560 runs, 5001 seed + 10002 HP + 10002 leech-state
> assertions, 3838 leech-seeded rows): seed-lands→drain+heal, Grass-immune, already-seeded fail, the
> leech-drain KO, the **leech+Leftovers+sand+burn 4-way residual ORDER** (the risk case), the
> seeder-replaced heal-follows, leech-into-a-real-battle — plus 3 deterministic regression pins. Leech
> Seed is **INCLUDED in the e2e capstone** (`MODELED_LEECH_MOVES`, `LEECHSEED_E2E_EXCLUDED = false`): its
> residual is DRAW-FREE so it can't shift the LCG the way the phaze `sample` does. DEFERRED (fail-loud):
> a **Liquid Ooze** target reverses the drain (rare in gen-3 OU). And now **SUBSTITUTE** (`substitute`) —
> a never-miss self-targeting Status move that spends `floor(maxhp/4)` HP to make a decoy with that much
> HP that ABSORBS incoming foe hits. The draw model, verified vs the sim PRNG probes
> (`probe_substitute_*.js`): the MOVE is DRAW-FREE (never-miss + a draw-free create/fail; FAILS if
> already-subbed or `hp <= floor(maxhp/4)`); a DAMAGING foe move into a sub draws acc+crit+damage as
> normal and the damage hits the SUB's HP (breaking at 0, the excess NOT carried to the mon) — **and the
> per-move secondary `random(100)` is STILL DRAWN** (the gen-3 SURPRISE that CONTRADICTED the build
> task's "one fewer random(100)" assumption: gen-3's `secondaries()` iterates the now-`null` target, so
> the draw fires — the same count as a bare hit — but its EFFECT, incl. any confusion `random(2,6)` /
> Tri-Attack `random(3)` follow-on, is SUPPRESSED). A STATUS / stat-drop move is BLOCKED (accuracy still
> drawn, draw-free past it); a CONFUSION self-hit hits the **MON, not the sub**; PHAZING BYPASSES the sub.
> Its own differential golden (`gen_substitute_golden.js` / `substitute_test.rs`) sweeps 9 scenarios × 80
> seeds (720 runs, 4320 decision rows: create + already-subbed FAIL, the low-HP create boundary, a
> held-sub absorb + secondary-suppression, the break no-carry, blocked status, blocked stat-drop, the
> confusion-self-hit-hits-the-mon, the phaze drag-through, sub-into-a-real-battle) — plus 5 deterministic
> regression pins. Substitute is now **INCLUDED in the e2e capstone** (`SUBSTITUTE_E2E_EXCLUDED = false`,
> 284 substitute-MOVE / 320 sub-up decisions across the 220-battle strict gate, bit-for-bit) after FIXING
> the SWITCHING/weather `eachEvent('WeatherChange')` switch-in tie-shuffle the substitute battle surfaced
> (e2e_84 dec4 — a switch-in-into-a-speed-TIE + freshly-set-weather draw-COUNT bug, NOT a substitute bug;
> see the FIXED note in EDGE_CASES.md + the regression pin).
>
> **EXPLOSION / SELF-DESTRUCT** (`explosion` / `selfdestruct`) — a Normal PHYSICAL move (BP 250 / 200 gen3)
> that HALVES the target's Def and faints the USER as part of the move. The gen-3 self-KO
> (`useMoveInner`:501-503) is UNCONDITIONAL + DRAW-FREE + PRECEDES the hit, so the USER FAINTS THROUGH a
> Protect (blocked), a Ghost (Normal-immune), a Substitute (the damage breaks the sub, no carry), or a
> miss — verified vs `probe_explosion_rng.js`. It draws the same acc `randomChance(100,100)` + crit +
> damage as any damaging move (no secondary); the resulting faint cancels the foe's queued move + draws NO
> Quick Claw; a mutual Explosion (both last mons) is a true double-faint gen-3 TIE. Its own differential
> golden (`gen_explosion_golden.js` / `explosion_test.rs`) sweeps 7 scenarios × 80 seeds (560 runs, 3688
> decision rows, 7376 FAINTED assertions, 880 self-KO rows, 294 sub-break boundaries, 341 wins + 59 ties)
> + 4 deterministic regression pins E1-E4. It is now **INCLUDED in the e2e capstone** (`EXPLOSION_E2E_EXCLUDED
> = false`, bit-for-bit — 544 explosion / self-KO decisions across the 220-battle strict gate,
> `explosion_decisions >= 50` coverage floor) after FIXING the two STATEFUL desyncs admitting it surfaced
> (NEITHER the self-KO): (1) a **double-faint → double-replacement → cascade `runSwitch` cancellation**
> (e2e_9 — when the FIRST runSwitch faints its own entrant on its side's Spikes, gen-3 `faintMessages`
> `cancelAction(getAllActive)` DROPS the OTHER side's pending runSwitch → the foe entrant is NOT re-chipped;
> the port had kept the stale runSwitch → re-chipped it; fixed in `cancel_active_actions`), and (2) a
> **confusion self-hit × Choice Band** gap (e2e_194 — gen-4 confusion runs the FULL `getDamage`, so Choice
> Band ×1.5 folds into the typeless self-hit; the port dropped it; fixed in `apply_confusion_self_hit`).
> Both DRAW-FREE (SEED bit-for-bit), pinned by `double_replacement_cascade_does_not_rechip_the_other_sides_entrant`
> + `confusion_self_hit_applies_choice_band` (revert-verified). See EDGE_CASES.md. And now **FIXED-DAMAGE /
> FIXED-FORMULA MOVES** (`run_fixed_damage_move`,
> routed by `is_fixed_damage_move` in `run_move` BEFORE the `category == Status` branch — these carry
> `basePower:0` so `derive_category` calls them Status) — the `damage:` / `damageCallback` moves that
> BYPASS `getDamage` (NO crit roll, NO 16-way damage roll): **Seismic Toss / Night Shade** (`damage:
> 'level'` → the USER's level), **Sonic Boom** (fixed 20), **Dragon Rage** (fixed 40), **Super Fang**
> (`max(floor(target.hp/2),1)`). The draw model (verified vs `probe_fixeddamage_rng.js`): they draw
> ONLY their accuracy roll — **Seismic Toss / Night Shade / Dragon Rage are acc-100 but NOT never-miss so
> they STILL draw one** (the phaze acc-100 precedent), **Sonic Boom / Super Fang are acc-90 and CAN miss**
> — then apply the FIXED amount (NO crit, NO damage roll, NO secondary). A GHOST is IMMUNE to Seismic Toss
> (Fighting 0×), a NORMAL to Night Shade (Ghost 0×), a GHOST to Sonic Boom / Super Fang (Normal 0×) —
> accuracy-drawn-THEN-`-immune` (same draw count as a landed hit). Into a SUBSTITUTE the fixed NUMBER hits
> the sub (breaks no carry); **Super Fang still halves the MON's hp behind a sub** (its callback reads
> `target.hp` before the sub-intercept). Its own differential golden (`gen_fixeddamage_golden.js` /
> `fixeddamage_test.rs`) sweeps 9 scenarios × 80 seeds (720 runs, 4144 seed + 8288 HP assertions, 2469
> fixed-damage-hit decisions, 720 wins) + 4 deterministic regression pins (FD1-FD4). The DEFERRED
> fixed-damage family (Psywave / the OHKO moves / Counter / Mirror Coat / Bide / Endeavor) is routed here
> too but FAIL-LOUDs (PANICS) rather than silently no-op. The e2e allow-list carries
> `MODELED_FIXED_DAMAGE_MOVES`, but 0 filter-clean teams happen to carry one (the leech-seed situation) —
> so the layer is proven by its dedicated golden + the FD1-FD4 pins, not the e2e. **PROTOCOL EMISSION
> (level-2) Phase 1 + Phase 2 + Phase 3 are now BUILT, and the drop-in `BattleStream::write_line`
> streaming surface with them** —
> `protocol.rs`'s `ProtocolBuilder` (an append-only, PRNG-free line buffer — two sim-mirroring
> retro-edits excepted: `attr_last_move_still` / `attr_last_move_miss`, the ports of
> `attrLastMove('[still]'/'[miss]')`) emits the byte-identical
> OMNISCIENT `|...|` stream. **Phase 1** (the CORE): framing / `turn` / `upkeep` / separator /
> `move`+`[miss]`/`[still]` / `switch` / `drag` / `-damage` [all HP variants + residual `[from]`] /
> `-heal` [Leftovers] / `faint` / `-crit` / `-supereffective` / `-resisted` / `-immune` / `-miss` /
> `win` / `tie`. **Phase 2** ADDS the status / boost / weather / ability / volatile / side-condition
> lines: the STATUS-move `|move|` announce (self-target vs foe-target, + the Spikes/Protect `[still]`
> did-nothing form), `-status` (+ `[from] move: Rest`) / `-curestatus` (+ `[msg]`) / `cant`
> (par/slp/frz/flinch), `-boost`/`-unboost` (by sign), `-weather` (SET `[from] ability:`+`[of]` +
> `[upkeep]` tick), `-ability` (Intimidate `boost`), `-fail` (+ `move: Substitute`+`[weak]`),
> `-sidestart` (Spikes), `-start`/`-end`/`-activate` (Substitute up/break/absorb `[damage]`),
> `-singleturn`/`-activate` (Protect). `run_full_battle_logged` returns `(BattleOutcome,
> Vec<ProtocolLine>)`. It is **OBSERVATION-ONLY** (draws no PRNG, mutates no asserted state) — the
> entire seed suite stays green with BYTE-IDENTICAL assertion counts (e2e 14228 / battle_test 2034 /
> fullbattle 2053) — gated by `tests/protocol_test.rs`, which replays the capture golden and asserts
> byte-equality on the filtered stream (**114 battles, 16115 lines byte-equal** across ALL 19 scenarios —
> up from Phase-2's 66/8721, 63/7223, 51/5630 and Phase-1's 30/1512; **0 scenario deferrals**,
> `DEFERRED_SCENARIOS` empty, **0 battles skipped**). **Phase 3** (`gen3_protocol_phase3_v1`, 8 new
> capture scenarios) closed the deferred long tail: the taunt/disable residual `-end`s + the Disable
> `[miss]`/`[still]` retro-edits, the Trace `-ability` reveal, the Flash Fire `-start`/`-immune`/`-end
> [silent]` cycle, the STATUS_IMMUNE `-immune [from] ability:` block lines (status-move sources only),
> the Synchronize→Lum `-status`/`-enditem`/`-curestatus` interleave (+ LumRest), the MID-BATTLE
> switch-in ability lines (weather SET / Pressure `[silent]` / Intimidate incl. the Clear-Body
> `-fail|unboost` form + the Substitute `-hint`), Leech Seed's full line family, Splash's `-nothing`,
> Pay Day's `-fieldactivate`, and the Rest-at-full-HP `-fail|heal` detail. And **`gen3_writeline_stream_v1`**:
> `battle.rs`'s `BattleStream::write_line` is the streaming drop-in — per-write byte-gated against the
> real Node `BattleStream` by `tests/writeline_test.rs` (`harness/gen_writeline_capture.js`: **38
> battles / 1722 writes, every per-write chunk byte-equal**). The `status_para_and_boost_drop` / `secondary_status_flinch` all-Seismic-Toss
> pair was un-deferred by
> `gen3_forced_replacement_resume_v1`: a **forced-replacement REQUEST-BOUNDARY resume** fix — the
> "phantom" was really an INVALID scripted move slot after a replacement swapped in a mon with FEWER
> moves, which the sim REJECTS drawing 0; the port now validates the move slot (`move_decision_is_legal`)
> and SKIPS it draw-free (mirroring `side.choose`) instead of running a full turn — plus the standalone
> status-move **already-statused `|-fail|` emission** (`foe_status_move_fail`). And the 3 `recover_and_rest`
> CB-Tyranitar Struggle battles were un-skipped by **`gen3_pp_tracking_v1`** — per-move PP counters + the
> forced-Struggle substitution + the Choice-Band lock + the gen-3 Struggle move (typeless '???' 50 BP,
> accuracy 100 → draws accuracy, `max(floor(dmg/4),1)` recoil, emitted via the `damage_of` `[from]
> Recoil|[of]` line). All three fixes are OBSERVATION-ONLY so the seed suite is byte-identical (pinned by
> `forced_replacement_resume_runs_the_post_replacement_move_decision` + the 4 PP/Struggle pins). `debug`
> lines stay poke-env-ignored + filtered. And now **TAUNT + DISABLE** (`gen3_taunt_disable_v1`) — the
> move-SELECTION-restriction layer on top of the PP/Choice-lock brick: **Taunt** (Dark, acc 100 — DRAWS
> `randomChance(100,100)`; the volatile is a FIXED duration 2, NO duration draw — the base onStart's
> `duration++` is SHADOWED by the gen4 mod gen3 inherits through, probe-proven constant 2) makes every
> Status-category slot un-selectable (+ cants a QUEUED status move at execution, `onBeforeMove` priority
> 0, AFTER the para roll; residual tick at order 10/subOrder 15 — gen4's values, NOT the base's order
> 15); **Disable** (Normal, acc 55 — CAN miss) disables the target's lastMove slot for **stored =
> disabler-faster ? random(2,6) : random(2,6)+1** (the gen4-inherited `!willMove → duration++`;
> PROBE-SETTLED — a base-source reading mis-predicts by a constant +1, the refuted-review cautionary
> tale), onTryHit FAILS draw-free with no lastMove, + cants a QUEUED now-disabled move at priority 7
> (BEFORE the para roll — the opposite of taunt); both `protect: 1` + `bypasssub: 1`; a taunt+disable /
> lock+disable mon draws ONE endTurn `runEvent('DisableMove')` tie-shuffle; taunt × disable × Choice
> lock × 0 PP compose into forced Struggle. Its own differential golden (`gen_taunt_disable_golden.js`
> / `taunt_disable_test.rs`) sweeps 9 scenarios × 80 seeds (720 runs, 4723 seed + 8595 taunt + 8595
> disabled-slot assertions) with BOTH disable-duration branches pinned at their exact free-up
> boundaries (the golden gate FAILS on a ±1 perturbation — proven both ways) + 4 revert-verified
> regression pins (TD1-TD4). `taunt`/`disable` are in the e2e modeled set (`MODELED_RESTRICTION_MOVES`);
> sample teams carry Taunt (real e2e coverage) but NONE carries Disable (0 e2e disable decisions — the
> leech-seed situation; proven by the dedicated golden + pins). And now **TRAPPING**
> (`gen3_trapping_v1`) — **Arena Trap + Magnet Pull**, the request-time SWITCH-legality layer (the
> switch mirror of the move-legality gate): `is_trapped` (Arena Trap = grounded foes, Flying/Levitate
> escape — but a grounded GHOST **IS** trapped in Showdown-gen3, no `trapped` type-immunity in the
> gen3 dex; Magnet Pull = Steel foes, groundedness irrelevant) + the draw-free `chooseSwitch`
> rejection (a scripted trapped `Switch` is SKIPPED, boundary open) + the endTurn
> TrapPokemon/MaybeTrapPokemon tie-shuffles (gen3 magnetpull is `onAny*` → the speed-tied MAGNETON
> MIRROR draws 4/endTurn, probe-proven 11-vs-7 against a Sturdy control; the Dugtrio mirror draws 0);
> phaze drags BYPASS trapping, forced replacements are never gated. Its own differential golden
> (`gen_trapping_golden.js` / `trapping_test.rs`) sweeps 8 scenarios × 80 seeds (640 runs, 5771 seed
> + 8346 trapped assertions, 508 mutual-trap rows, 160 phaze-drag rows) + 5 revert-verified pins
> (T1-T5, incl. the grounded-Ghost Showdown-gen3 surprise). `arenatrap`/`magnetpull` are in the e2e
> `MODELED_ABILITIES` (they took filter-clean teams 18 → 22; now 88 trapped-boundary decisions
> bit-for-bit at the DMG_MOD-admitted 151-team corpus); the
> admission's corpus shift surfaced + FIXED a real Intimidate-vs-SUBSTITUTE gap (a mid-battle
> Intimidate switch-in must NOT drop a subbed foe's Atk — probe-proven seed-neutral, pinned by
> `intimidate_into_a_substitute_is_a_noop`). The rest of the live engine (the full
> `runEvent` gather,
> Wish/Heal-Bell status moves, the OHKO/reactive fixed-damage moves, Torment/Imprison/Encore,
> Mean Look / Shadow Tag, the
> request/choice layer, protocol Phases 3-5) is
> next. See `CLAUDE.md` for the architecture, the bit-for-bit contract, and why the hard part is
> RNG-consumption-order + protocol parity, not the math.

## Layout

```
src/rust_sim/
  Cargo.toml              # std-only, zero deps (bit-exact + no-network test)
  src/
    lib.rs                # crate root
    prng/                 # DONE — bit-for-bit port of sim/prng.ts
      mod.rs              #   Prng: random*/sample/shuffle + backend dispatch
      sodium.rs           #   SodiumRNG (hand-rolled ChaCha20) — the default
      gen5.rs             #   Gen5RNG (64-bit LCG) — legacy seeds
    json.rs               # DONE — tiny std-only JSON reader (dex parses with it)
    dex/                  # DONE — static data over data/pokemon/*.json
                          #   items.rs: ItemData + the gen3_item_mechanics_v1 structured
                          #   fields (typeBoost/statMods/onlySpecies/choice/isBerry) + accMod;
                          #   abilities.rs: AbilityData (dmgMod + accMod); accmod.rs: the shared
                          #   AccMod (gen3_accuracy_pipeline_v1 — the to-hit fold reads it). See
                          #   CLAUDE.md "## Data-driven mechanics"
      mod.rs              #   Dex: species/moves/item/ability/nature/chart/learnset
      types.rs            #   Type / MoveCategory / BaseStats
      species.rs moves.rs type_chart.rs   #   per-table parsers
    team.rs               # DONE — Teams.unpack/pack (the packed string from >player)
    stats.rs              # DONE — gen-3 in-battle stat computation
    state.rs              # DONE — battle state + Battle::start / start_with_switchins
    event.rs              # DONE — event-dispatch core (singleEvent + speed_sort tie shuffle)
                          #        + the >start switch-in abilities (Intimidate/weather)
    damage.rs             # DONE — gen-3 single-hit damage calc (two-phase modifier chain)
    turn.rs               # DONE — multi-turn move execution + residuals + SWITCHING + win/loss:
                          #        BattleState::run_turn (FULL per-turn cycle: action-order +
                          #          per-action eachEvent speed-tie shuffles, accuracy/crit/damage,
                          #          immune short-circuit, deferred-faint protocol, end-of-turn
                          #          residuals [weather chip / Leftovers / burn,psn,tox DoT], Quick Claw)
                          #        BattleState::run_battle (scripted multi-turn loop, stops at 1st faint)
                          #        BattleState::run_full_battle (Choice::Move/Switch script → BattleOutcome,
                          #          plays to WIN/LOSS: voluntary switches [order 103<200], the switchIn
                          #          position swap [stable MonState::uid], draw-FREE gen-3 switch-in ability
                          #          Start, execute_switch NATURAL CURE [status=None on an alive outgoing
                          #          naturalcure holder, voluntary + phaze-drag, draw-free — gen3_natural_cure_v1],
                          #          post-faint replacement [single + DOUBLE: insertChoice splice +
                          #          no-op-move tail-skip], Explosion self-KO, win/loss [pokemon_left==0;
                          #          both→gen-3 tie; no Quick Claw on the deciding faint])
                          #        + SECONDARY effects (per-move random(100) after a landed hit: par/frz/
                          #          flinch/psn; CONFUSION = random(100)+random(2,6) duration [gated by
                          #          already-confused/Own Tempo]; foe stat-DROP / self stat-RAISE via the
                          #          structured secondaryBoosts spec [draw-free, Clear Body/Hyper Cutter/
                          #          Keen Eye gates]; Tri Attack = random(100)+sample(3) [fail-loud >1-col
                          #          guard]; Serene Grace x2 / Shield Dust x0; onTrySetStatus gates) +
                          #          onBeforeMove STATUS draws (the NEW LEADING draw before accuracy: sleep
                          #          draw-free counter, freeze randomChance(1,5) thaw, flinch draw-free,
                          #          confusion randomChance(1,2)+self-hit random(16), paralysis
                          #          randomChance(1,4) full-para; gen-3 par speed x0.25)
                          #        + STANDALONE STATUS MOVES (run_status_move: Thunder Wave/Stun Spore/
                          #          Glare [par], Poison Powder/Gas [psn], Toxic [tox], Will-O-Wisp [brn],
                          #          Spore/Sleep Powder/Hypnosis/Sing/Lovely Kiss/Grass Whistle [slp] —
                          #          accuracy-only draw + try_set_status; move-TYPE immunity for TWave→Ground
                          #          + Glare→Ghost [accuracy still drawn → -immune]; sleep random(2,6) duration
                          #          + Early Bird double-decrement; Toxic stage 0 [residual ramps]; gen3ou
                          #          Sleep Clause Mod + ability immunity [Insomnia/Limber/…]; the gen3ou-only
                          #          runEvent('SetStatus') 2-clause handler-sort shuffle [gated by sleep_clause];
                          #          fail-loud on any unmodeled status move)
                          #        + SELF-TARGETING SETUP / STAT-BOOST MOVES (run_status_move self-boost branch:
                          #          Calm Mind / Dragon Dance / Swords Dance / Agility / Bulk Up / Amnesia /
                          #          Barrier / Acid Armor / Iron Defense / Cosmic Power / Tail Glow / Meditate /
                          #          Sharpen / Howl / Harden / Withdraw / Growth — data-driven selfBoosts; never-
                          #          miss → no accuracy draw, DRAW-FREE boost() apply [±6 clamp, own Clear Body
                          #          never blocks self], landed FALSE; +Spe Dragon Dance/Agility raises boosts[4]
                          #          NOW but cached_speed stays stale → the first-mover flips NEXT turn, bit-exact;
                          #          fail-loud excludes Defense Curl/Minimize/Double Team/Belly Drum/Curse)
                          #        + SELF-HEAL / RECOVERY MOVES (run_status_move recovery branch + run_rest:
                          #          Recover/Soft-Boiled/Slack Off/Milk Drink → floor(maxhp/2); Moonlight/Synthesis/
                          #          Morning Sun → gen4-inherited PLAIN-integer weather heal [none floor(maxhp/2) /
                          #          sun floor(maxhp*2/3) / sand+rain+hail floor(maxhp/4)]; Rest → full heal + cure +
                          #          a FIXED Sleep(3) whose slp.onStart STILL draws-then-DISCARDS one random(2,6)
                          #          [the draw-COUNT crux] + the gen3ou SetStatus shuffle ordered shuffle→random(2,6);
                          #          never-miss → no accuracy draw, DRAW-FREE apply_heal, full-HP/heal-0 FAIL path,
                          #          landed FALSE; splash = a draw-free no-op; fail-loud excludes Wish/Heal Bell/
                          #          Aromatherapy/Refresh/Leech Seed)
                          #        + PROTECT / DETECT (run_protect + the foe-move block in run_move:
                          #          NEVER-MISS + priority 3 → resolves before the foe's attack; the gen-3 STALL
                          #          success draw [FIRST protect short-circuits NO draw, a CONSECUTIVE one draws
                          #          randomChance(1, counter) at the floored 2/4/8 denominator via the gen4-inherited
                          #          counterMax 8; success → onStart 2 / onRestart *2; fail → NO-delete (gen3
                          #          resolved gen5-base: counter persists, consecutive fails re-roll same denom)] + the move-BLOCK
                          #          [a foe move TARGETING the protected mon draws its accuracy roll then is blocked
                          #          BEFORE crit/damage/secondary/status, gen-3-tryMoveHit-ordered; self-target never
                          #          blocked]; the stall counter resets after one non-protect/switch turn [the stall
                          #          volatile's duration:2 expiry at the residual]; the protect+stall duration-only
                          #          volatiles register RESIDUAL handlers [findPokemonEventHandlers(..., 'duration')]
                          #          that participate in the residual speed-sort [+2 tied handlers → the tie-shuffle
                          #          COUNT]; new MonState fields protected/protect_counter/stall_duration; fail-loud
                          #          excludes Endure + the gen4+ Quick/Wide Guard/King's Shield)
                          #        + WATER/VOLT ABSORB heal now ACCURACY-GATED (an onTryHit → fires only on a HIT;
                          #          a missed Water/Electric move no longer heals the holder)
                          #        + SPIKES / PHAZING / LEECH SEED / SUBSTITUTE / EXPLOSION move layers (see the
                          #          feature blurbs above + CLAUDE.md for each's draw model + e2e status)
                          #        + FIXED-DAMAGE / FIXED-FORMULA MOVES (run_fixed_damage_move, routed by
                          #          is_fixed_damage_move BEFORE the category==Status branch [bp 0 → derive_category
                          #          calls them Status]: Seismic Toss/Night Shade [damage:'level' → user's level],
                          #          Sonic Boom [20], Dragon Rage [40], Super Fang [max(floor(target.hp/2),1)] —
                          #          accuracy-only draw [acc-100-but-NOT-never-miss STILL draws; acc-90 CAN miss],
                          #          NO crit/damage roll/secondary; accuracy-drawn-THEN-immune type gate [Fighting→
                          #          Ghost, Ghost→Normal, Normal→Ghost]; sub-absorb of the fixed number [Super Fang
                          #          halves the MON's hp behind a sub]; fail-loud on Psywave/OHKO/Counter/Mirror
                          #          Coat/Bide/Endeavor; 0 e2e filter-clean teams carry one → proven by the
                          #          dedicated golden + FD1-FD4 pins)
                          #        + PP TRACKING + STRUGGLE (gen3_pp_tracking_v1): per-move PP (init pp*8/5 via 3
                          #          PP-ups; −1/use draw-free, −2 into Pressure, only when the mon MOVES, PERSISTS
                          #          across switch), the Choice-Band lock, must_struggle() → the forced-Struggle
                          #          substitution, and Struggle (typeless '???' 50 BP physical, accuracy 100 → draws
                          #          accuracy, recoil = max(floor(dmg/4),1) per the gen3 recoil:[1,4] path). Un-skips
                          #          the 3 recover_and_rest Struggle protocol battles; seed suite byte-identical.
                          #        + TAUNT + DISABLE (gen3_taunt_disable_v1): the selection-restriction layer —
                          #          Taunt (acc-100 draw; FIXED duration 2, NO draw; blocks Status slots + cants a
                          #          queued one at priority 0; residual tick order 10/subOrder 15) and Disable
                          #          (acc-55 draw; ONE random(2,6), stored +1 iff the target already moved; blocks
                          #          the lastMove slot + cants at priority 7; residual NO_ORDER/subOrder 2;
                          #          draw-free no-lastMove fail), the endTurn runEvent('DisableMove') tie-shuffle
                          #          for a multi-restriction mon, move_usable/must_struggle composition → forced
                          #          Struggle. In the e2e modeled set (teams carry Taunt; none carries Disable).
                          #        + TRAPPING (gen3_trapping_v1): Arena Trap + Magnet Pull — is_trapped (grounded /
                          #          Steel; a grounded Ghost IS trapped in Showdown-gen3), the draw-free trapped-
                          #          switch rejection in move_decision_is_legal (the switch mirror of the move
                          #          gate; phaze drags + forced replacements never gated), and trap_event_shuffles
                          #          (the endTurn TrapPokemon/MaybeTrapPokemon tie-shuffles: the speed-tied
                          #          Magneton mirror draws 4/endTurn — gen3 magnetpull is onAny*; Dugtrio mirror 0).
                          #          In the e2e modeled set (trapping took teams 18 → 22; now 88 trapped decisions at the 151-team DMG_MOD corpus).
                          #        — the RNG-consumption-order layer, cross-turn AND to game-end
    battle.rs             # Battle::start{,_with_switchins} + state_mut + BattleStream::write_line
                          #   (gen3_writeline_stream_v1 — the per-write streaming drop-in, byte-gated by
                          #   writeline_test.rs vs gen_writeline_capture.js: 38 battles / 1722 writes /
                          #   5510 filtered lines); snapshot/reseed/choose stay todo!()
    protocol.rs           # Player / Choice / ProtocolLine types
                          #        + ProtocolBuilder — the Phase-1 EMIT API (append-only, PRNG-free line buffer on
                          #          BattleState.log; one retro-edit: attr_last_move_still = attrLastMove('[still]')):
                          #          MonRef / HpStatus (x/y | x/y <status> | 0 fnt) / Cause
                          #          ([from] item: … | [from] <bare>) + typed constructors for the core line types;
                          #          run_full_battle_logged emits framing + the |move|/|-damage|/|switch|/|faint|/…
                          #          stream; OBSERVATION-ONLY (draws no PRNG → the seed suite is unchanged)
  tests/
    prng_golden.rs        # differential vs real prng.js (~2900 assertions)
    dex_test.rs           # parity vs agents.gen3_data (~1500 assertions) + smoke
    team_test.rs          # differential vs Showdown Teams.pack/unpack (24 cases) + smoke
    stats_test.rs         # differential vs the sim's own storedStats (18 cases) + smoke
    state_test.rs         # differential vs the sim's construction-time state (12 mons) + smoke
    switchin_test.rs      # differential vs the sim's post-switch-in boosts+weather (5 scenarios) + smoke
    damage_test.rs        # EXACT differential vs the omniscient oracle (31 scenarios) + smoke
    turn_test.rs          # per-seed STATE + post-turn-PRNG-SEED differential (15 scenarios x 60 seeds;
                          #   780 EXACT seed-parity rows + speed-tie first-mover) — the single-turn proof
    battle_test.rs        # per-seed CROSS-TURN STATE+SEED differential (12 scenarios x 40 seeds x several
                          #   turns; ~2034 EXACT post-turn-seed rows incl. the TIE class + residuals)
    fullbattle_test.rs    # per-seed PER-DECISION STATE+SEED+winner differential TO GAME-END (8 scenarios x
                          #   50 seeds; ~2053 EXACT per-decision seed rows: both-switch distinct/tie,
                          #   switch-vs-move, post-faint single + double replace, KO-to-win, last-mon tie)
    residual_faint_test.rs # deterministic regression for the 2 residual-faint BLOCKERS the golden can't
                          #   reach (residual-KO-under-tie skips the trailing draws; chip-KO not revived by Leftovers)
    secondary_test.rs     # per-seed PER-DECISION STATE(+STATUS+BOOSTS+CONFUSION)+SEED+winner differential TO
                          #   GAME-END with REAL secondary moves (status/boosts/confusion inflicted IN-ENGINE →
                          #   onBeforeMove draws fire): 12 scenarios x 80 seeds, ~4328 EXACT per-decision seed
                          #   rows + ~7457 status + ~7457 boost-stage + ~7457 confusion-counter assertions
                          #   (Body Slam par / Ice Beam frz / Thunderbolt par / Rock Slide flinch / Sludge Bomb
                          #   psn; Crunch/Psychic/Shadow Ball -1 SpD foe; Meteor Mash +1 Atk self; Water Pulse
                          #   confusion + random(2,6); Intimidate -1 Atk on entry) — the secondary draw + apply proof
    status_move_test.rs   # per-seed PER-DECISION STATE(+STATUS+sleep/Toxic counter)+SEED+winner differential TO
                          #   GAME-END with STANDALONE STATUS MOVES (gen3ou → Sleep Clause Mod + the gen3ou
                          #   runEvent('SetStatus') handler-sort shuffle ACTIVE): 10 scenarios x 80 seeds —
                          #   Thunder Wave lands/par + Ground-immune, Toxic ramp + Steel-immune, Will-O-Wisp
                          #   lands+miss, Spore sleep + random(2,6) + onBeforeMove wake, Sleep Powder miss,
                          #   Stun-Spore para, a SLEEP CLAUSE block, status-into-a-real-battle. The standalone draw proof
    setup_move_test.rs    # per-seed PER-DECISION STATE+BOOST-STAGE+SEED+first-mover differential TO GAME-END
                          #   with SELF-BOOST SETUP MOVES (gen3customgame): 6 scenarios x 80 seeds — Calm Mind
                          #   climb to +6 cap, Swords Dance +2 Atk, Dragon Dance → first-mover FLIP, Agility →
                          #   first-mover FLIP, a +6-cap no-op, setup-into-a-real-battle. 2667 seed + 4736 boost-
                          #   array + 2549 first-mover assertions. The self-boost + cached-speed-timing proof
    recovery_move_test.rs # per-seed PER-DECISION STATE+HP+STATUS+SEED+winner differential TO GAME-END with
                          #   SELF-HEAL / RECOVERY MOVES (gen3customgame): 8 scenarios x 80 seeds — Recover from
                          #   low HP + at FULL HP (the fail), Rest sleep+wake + Rest curing para, Moonlight in
                          #   none/sun (Groudon Drought)/Synthesis in sand (TTar Sand Stream), recovery-into-a-
                          #   real-battle. ~4468 decision rows. The heal-amount + Rest random(2,6) draw-count proof
    protect_move_test.rs  # per-seed PER-DECISION STATE(+HP+STATUS+STALL-COUNTER)+SEED+winner differential TO
                          #   GAME-END with PROTECT / DETECT (gen3customgame): 6 scenarios x 80 seeds — single
                          #   protect block, CONSECUTIVE protects (the stall denominator BOTH ways), the counter
                          #   reset (protect→non-protect→protect), protect vs a STATUS move, Detect, protect-into-
                          #   a-real-battle. 480 runs, 2772 seed + 5544 HP + 4984 STALL-COUNTER assertions (1102
                          #   blocks, 272 escalated counter>=4). The stall-draw + block + counter-escalation/reset proof
    spikes_test.rs        # per-seed PER-DECISION STATE(+HP+SPIKES-LAYERS)+SEED+winner differential TO GAME-END
                          #   with SPIKES (the entry hazard + the first SIDE CONDITION; gen3customgame): 5 scenarios
                          #   x 80 seeds — lay 1 + grounded switch-in (maxhp/8), stack 2/3 + a Spikes-at-max FAIL +
                          #   the 3-layer chip (maxhp/4), a Flying/Levitate IMMUNE entry (ZERO), a spikes-KO-on-entry
                          #   → forced replacement (which ALSO takes spikes), spikes-into-a-real-battle. 400 runs,
                          #   ~3475 seed + ~6950 HP + ~6950 SPIKES-LAYER assertions (1440 switch-in-chip, 160 spikes-KO
                          #   -on-entry rows). The side-state + grounded switch-in-damage + draw-free proof
    phaze_test.rs         # per-seed PER-DECISION STATE(+HP+SPIKES-LAYERS+DRAG-SPECIES)+SEED+winner differential TO
                          #   GAME-END with PHAZING (Roar / Whirlwind; gen3customgame): 7 scenarios x 80 seeds —
                          #   Roar drags a RANDOM bench (seed sweep → ≥2 distinct mons), the n=1 sample draw, a Roar
                          #   that FAILS (foe's last mon), Whirlwind, Roar INTO Spikes (the dragged mon takes the chip),
                          #   repeated Roar into a stochastic spikes-KO, phaze-into-a-real-battle. 560 runs, 10388 seed +
                          #   20776 HP + 20776 SPIKES-LAYER assertions (2795 drags, 1769 phaze-into-spikes-damage). The
                          #   random-target-draw + accuracy-roll + n=1-sample + phaze-into-Spikes proof. (The exact
                          #   single-drag spikes-KO is ALSO pinned deterministically in regression_test.rs.)
    leechseed_test.rs     # per-seed PER-DECISION STATE(+HP+STATUS+SPIKES-LAYERS+LEECH-SEEDED)+SEED+winner differential
                          #   TO GAME-END with LEECH SEED (gen3customgame): 7 scenarios x 80 seeds — seed lands→drain
                          #   (floor(maxhp/8)) + seeder heal, a GRASS-immune target (accuracy still drawn → -immune, no
                          #   volatile), an already-seeded re-seed FAIL, the leech-drain KO, the leech+Leftovers+sand+burn
                          #   4-WAY RESIDUAL ORDER (sand[o=8]→Lefto[s=4]→LEECH[s=5]→burn[s=6] — the risk case), the
                          #   seeder-replaced heal-follows, leech-into-a-real-battle. 560 runs, 5001 seed + 10002 HP +
                          #   10002 LEECH-STATE assertions (3838 leech-seeded rows). The accuracy-90-draw + draw-free
                          #   residual + subOrder-5 proof. (The 4-way order + the leech handler tie + the seeder-fainted
                          #   gate are ALSO pinned deterministically in regression_test.rs.)
    explosion_test.rs     # per-seed PER-DECISION STATE(+HP+FAINTED+STATUS+SPIKES-LAYERS+SUB-HP)+SEED+winner differential
                          #   TO GAME-END with EXPLOSION / SELF-DESTRUCT (gen3customgame): 7 scenarios x 80 seeds — plain
                          #   self-KO (foe def-halved), into-a-Substitute (breaks it, user faints), into-a-Protect (blocked,
                          #   user faints), into-a-Ghost (immune, user faints), the mutual double-faint TIE, an Explosion-KO
                          #   double replacement, explosion-into-a-real-battle. 560 runs, 3688 rows, 7376 FAINTED + 3688 seed
                          #   assertions (880 self-KO rows, 294 sub-break boundaries), 341 wins + 59 ties. The unconditional
                          #   draw-free self-KO proof. (The Protect/Ghost/sub-break/mutual-TIE edges are ALSO pinned
                          #   deterministically in regression_test.rs E1-E4.) NOTE: Explosion is INCLUDED in the e2e
                          #   capstone (EXPLOSION_E2E_EXCLUDED = false, 544 self-KO decisions) after fixing the
                          #   double-faint→cascade runSwitch-cancellation + confusion-self-hit-Choice-Band bugs (both DR pins).
    pp_struggle_test.rs   # per-seed PER-DECISION STATE(+HP+STATUS+PP)+SEED+winner differential TO GAME-END with
                          #   PP TRACKING + STRUGGLE (gen3_pp_tracking_v1, gen3customgame): 5 scenarios x 80 seeds —
                          #   CB-lock forced Struggle into a Levitate Ghost (immune decrement + Struggle-into-a-Ghost +
                          #   recoil), single-move all-slots Struggle, Pressure −2, PP-persists-across-switch + miss,
                          #   PP-into-a-real-battle to a win. 400 runs, 4424 seed + 8368 PP assertions (1035 forced-
                          #   Struggle, 1035 recoil, 378 Pressure−2, 3292 immune-decrement, 1355 0-PP rows), 400 wins.
                          #   The per-slot PP array + the Struggle recoil (floor(dmg/4)) are the new signals; PP is
                          #   draw-free so a wrong count is a STATE bug, a wrong Struggle draw model a SEED bug.
    taunt_disable_test.rs # per-seed PER-DECISION STATE(+STATUS+TAUNT+DISABLED-SLOT)+SEED+winner differential TO
                          #   GAME-END with TAUNT + DISABLE (gen3_taunt_disable_v1, gen3customgame): 9 scenarios x 80
                          #   seeds — taunt lands→Status slots blocked→frees on the sim window (incl. the taunter-
                          #   SECOND minor-A branch), BOTH disable-duration branches AT their exact free-up turns
                          #   (faster = rolled, slower = rolled+1 — a ±1 FAILS at the boundary, perturb-proven),
                          #   disable-into-no-lastMove fail, taunt+disable forced Struggle, disable-clears-on-switch,
                          #   taunt-into-a-real-battle. 720 runs, 4723 seed + 8595 taunt + 8595 disabled-slot
                          #   assertions (1188 taunted / 803 disabled rows; free-ups taunt 930 / disable 289; 415
                          #   miss / 128 fail / 273 struggle), 720 wins. (The window/duration/struggle edges are
                          #   ALSO pinned deterministically in regression_test.rs TD1-TD4.)
    trapping_test.rs      # per-seed PER-DECISION STATE(+STATUS+per-side TRAPPED)+SEED+winner differential TO
                          #   GAME-END with TRAPPING (gen3_trapping_v1, gen3customgame): 8 scenarios x 80 seeds —
                          #   Arena Trap holds a grounded foe (its script would flee when free → it only ever
                          #   fights), Flying/Levitate switch freely, Magnet Pull holds Steel only (Skarmory
                          #   trapped despite Flying; Snorlax walks), the MAGNETON MIRROR (mutual trap + the
                          #   4-draws-per-endTurn tie-shuffles, para-broken ties pinned both ways), the DUGTRIO
                          #   MIRROR (mutual trap, ZERO draws), Roar drags a trapped mon (phaze bypass), the
                          #   AT-vs-MP cross (one-sided trap + 2 draws), trapping-into-a-real-battle. 640 runs,
                          #   5771 seed + 8346 trapped assertions (2631 trapped rows, 508 mutual), 160 drag rows,
                          #   822 free voluntary switches, 640 wins. (The REJECTION path — a trapped scripted
                          #   Switch skipped draw-free — is pinned deterministically in regression_test.rs T1-T5.)
    regression_test.rs    # DETERMINISTIC named pins for the edge cases each layer surfaced (constructed scenario +
                          #   fixed seed + ground-truth seed from a probe). Phaze pins: phaze_draws_accuracy_then_n1_
                          #   sample_seed (gen-3 acc 100, NOT never-miss + the n=1 sample DRAWS), phaze_fail_draws_only_
                          #   accuracy_no_sample_seed (foe's last mon → no sample), phaze_drag_into_a_spikes_ko_chains_a_
                          #   replacement (the composition). Leech pins: leech_residual_order_leftovers_sand_burn (the
                          #   4-way order: leech is subOrder 5, between Leftovers 4 + burn 6), leech_handler_tie_at_equal_
                          #   speed_draws_one_shuffle (two seeded mons tie → one residual shuffle), leech_seeder_fainted_
                          #   skips_the_drain (a fainted seeder → no drain). Explosion pins (E1-E4): explosion_into_a_
                          #   protect_the_user_still_faints, explosion_into_a_ghost_the_user_still_faints, explosion_breaks_
                          #   a_substitute_and_the_user_still_faints, mutual_explosion_is_a_double_faint_tie (the self-KO is
                          #   unconditional + draw-free + precedes the hit). Double-replacement-cascade pins (DR1-DR2, the
                          #   Explosion e2e re-enable): double_replacement_cascade_does_not_rechip_the_other_sides_entrant
                          #   (a cascade faint's cancelAction drops the FOE's pending runSwitch → the foe entrant is NOT
                          #   re-chipped), confusion_self_hit_applies_choice_band (the confusion self-hit folds Choice Band).
                          #   PP/Struggle pins (gen3_pp_tracking_v1): pp_decrements_on_use_draw_free (−1/use, draw-free
                          #   seed), pressure_decrements_two_pp (−2 into a Pressure holder), no_usable_move_forces_
                          #   struggle_and_struggle_recoil_is_gen3_quarter_damage_dealt (a CB-locked mon out of PP → forced
                          #   Struggle, HITS a Ghost, recoil = floor(dmg/4) NOT round NOT maxhp/4). All 4 revert-verified.
                          #   Taunt/Disable pins (TD1-TD4, gen3_taunt_disable_v1): taunt_blocks_status_move_selection_
                          #   for_the_sim_window_draw_free (queued move cant'd; ONE restricted selection; the freed
                          #   Thunder Wave paralyzes = the free-up proof), disable_duration_stored_per_branch_matches_sim
                          #   (faster = rolled, slower = rolled+1, pinned AT the free-up boundary — trips +1/-1 AND the
                          #   dropped-branch model), taunt_plus_disable_forces_struggle (nothing usable → Struggle, HP +
                          #   floor(15/4)=3 recoil), taunt_and_disable_onbeforemove_priority_vs_paralysis (taunt cants
                          #   AFTER the para roll [+1 draw], disable BEFORE it [no para roll] — the golden doesn't cover
                          #   a paralyzed queued move, so TD4 is the only ordering gate). All revert-verified.
                          #   Plus the 4 substitute pins + the 7 prior engine-bug pins + the switch-tie-weather pin.
    e2e_fuzz_test.rs      # THE CAPSTONE — per-seed PER-DECISION STATE(+STATUS+BOOSTS+CONFUSION+SPIKES)+SEED+winner
                          #   differential TO GAME-END over REAL teams (data/teams/*.txt) with RANDOM modeled
                          #   choices (INCLUDING status moves, SETUP, RECOVERY, PROTECT/DETECT, SPIKES, SUBSTITUTE,
                          #   EXPLOSION/SELF-DESTRUCT, and TAUNT/DISABLE): 220 battles, ALL 220 bit-for-bit clean (STRICT
                          #   filtered_diverged 0, no escape hatch), 11630 decisions of which 5069 USE SPIKES, 581 a
                          #   SUBSTITUTE, 557 an EXPLOSION self-KO, 282 a PHAZE drag, 178 USE TAUNT, and 88 a TRAPPED mon (the new
                          #   coverage + a taunt_decisions >= 50 floor; 0 USE DISABLE — no sample team carries it,
                          #   the honest disclosure). The EXPLOSION re-enable surfaced + FIXED
                          #   the double-faint→double-replacement→cascade runSwitch-cancellation (gen-3 cancelAction over
                          #   getAllActive drops the FOE's pending runSwitch → the foe entrant is NOT re-chipped) + the
                          #   confusion-self-hit Choice-Band fold (gen-4 confusion runs the FULL getDamage). The SPIKES
                          #   expansion surfaced + FIXED the FORCED-REPLACEMENT updateSpeed-on-commit (a foe
                          #   para'd mid-turn must drop to its para speed before the resumed Updates, not spuriously tie
                          #   the fresh entrant on its stale speed). Prior expansions fixed the protect/stall/flinch
                          #   RESIDUAL duration handlers + no-delete-on-fail stall + willAct() gate, the RESIDUAL HANDLER
                          #   GATHER-ORDER bug, Water/Volt Absorb heal-on-miss, + Toxic stage-reset-on-switch.
                          #   gen3customgame. Headline tallies are clean-only (loop breaks at first divergence).
                          #   Ignored: e2e_diag, e2e_trace_one
    protocol_test.rs      # PROTOCOL-EMISSION (level-2, Phase 1+2) BYTE-differential — replays the capture golden
                          #   through run_full_battle_logged, FILTERS both sides to the Phase-1+2 line types (only
                          #   debug + still-deferred-mechanic lines dropped from BOTH; |t:| normalized), asserts
                          #   BYTE-EQUALITY per line in order (a truncated/turn-capped golden is a PREFIX match).
                          #   66 battles / 8721 lines byte-equal across ALL 11 scenarios; 0 deferred + 0 skipped
                          #   (the all-Seismic-Toss pair un-deferred by gen3_forced_replacement_resume_v1; the 3
                          #   recover_and_rest Struggle battles un-skipped by gen3_pp_tracking_v1 — PP + forced-Struggle
                          #   + the Choice-Band lock + the Struggle |move| / |-damage| [from] Recoil|[of] lines).
                          #   A truncated/turn-capped golden is a byte-exact PREFIX match. The OBSERVATION-ONLY proof
                          #   is the seed suite staying green with IDENTICAL counts (e2e 13367 / battle 2034 / full 2053).
    vectors/{prng,dex,team,stats,state,switchin,damage,turn,battle,fullbattle,secondary,status_move,setup_move,recovery_move,protect_move,spikes,phaze,leechseed,substitute,explosion,fixeddamage,pp_struggle}_golden.txt
    vectors/e2e_fuzz_golden.txt     # the CAPSTONE filtered gate (real teams, modeled mechanics, full battles)
    vectors/e2e_fuzz_taxonomy.txt   # coverage map: gaps ranked by STATIC team composition (which unmodeled
                          #   ability/item the paired teams carry) — NOT observed divergence cause, move-blind
  harness/
    prng_reference.js     # dependency-free JS spec the Rust PRNG mirrors
    gen_prng_vectors.js   # cross-checks the spec vs real prng.js, emits vectors
    gen_dex_golden.py     # dumps the agents.gen3_data facade view -> dex golden
    gen_team_golden.js    # captures Showdown Teams.pack/unpack -> team golden
    gen_stats_golden.js   # captures the sim's own computed stats -> stats golden
    gen_state_golden.js   # captures the sim's construction-time state -> state golden
    gen_switchin_golden.js # captures the sim's post-switch-in boosts+weather -> switchin golden
    gen_damage_golden.js  # captures the sim's EXACT damage (max-roll) -> damage golden
    gen_turn_golden.js    # captures the sim's post-turn STATE + before/after PRNG seed -> turn golden
    gen_battle_golden.js  # captures the sim's per-turn STATE + seeds over N turns -> battle golden
    gen_fullbattle_golden.js # drives the sim a FULL battle (move+switch+replacement→win) -> fullbattle golden
    gen_secondary_golden.js # drives the sim a FULL battle with REAL SECONDARY moves (status inflicted
                          #   in-engine → onBeforeMove draws fire) -> secondary golden
    gen_status_move_golden.js # drives the sim (gen3ou) full battles with STANDALONE STATUS MOVES
                          #   (Thunder Wave/Toxic/Will-O-Wisp/Spore/… → accuracy + apply + sleep random(2,6) +
                          #   Sleep Clause + the gen3ou SetStatus shuffle) -> status_move golden; fail-loud per
                          #   scenario if its declared branch didn't realize, + a stall guard
    gen_setup_move_golden.js # drives the sim (gen3customgame) full battles with SELF-BOOST SETUP MOVES
                          #   (Calm Mind/Dragon Dance/Swords Dance/Agility/… → DRAW-FREE boost apply; the +Spe
                          #   cached-speed first-mover FLIP) -> setup_move golden; fail-loud require floors
                          #   (a boost applied / the +6 cap / a first-mover flip) + a stall guard
    gen_recovery_move_golden.js # drives the sim (gen3customgame) full battles with SELF-HEAL RECOVERY MOVES
                          #   (Recover/Soft-Boiled/Slack Off/Milk Drink → floor(maxhp/2); Moonlight/Synthesis/
                          #   Morning Sun → weather heal; Rest → full heal + cure + a FIXED Sleep(3) that
                          #   draws-then-discards one random(2,6)) -> recovery_move golden; fail-loud require
                          #   floors (a heal applied / the full-HP fail / a Rest sleep / a wake) + a stall guard
    gen_protect_move_golden.js # drives the sim (gen3customgame) full battles with PROTECT / DETECT (the gen-3
                          #   STALL draw [first protect NO draw, consecutive randomChance(1,counter) at 2/4/8] +
                          #   the move-BLOCK [accuracy drawn then blocked]) -> protect_move golden; records the
                          #   per-decision STALL COUNTER (volatiles.stall.counter); fail-loud require floors
                          #   (a block / a protect SUCCESS / a consecutive-protect FAILURE) + a stall guard
    probe_protect_rng.js  # the PROTECT PRNG draw PROBE (the crux investigation): monkey-patches the sim PRNG
                          #   to record every draw + its sim/* site over constructed protect scenarios — proved
                          #   the first-protect-no-draw, the 2/4/8 stall denominators, and the accuracy-then-block
    gen_spikes_golden.js  # drives the sim (gen3customgame) full battles with SPIKES (the entry hazard + the
                          #   first SIDE CONDITION → lay 1/stack 2/3 with the increasing maxhp/8 ÷6 ÷4 grounded
                          #   switch-in chip; a Flying/Levitate IMMUNE entry; a Spikes-at-max FAIL; a spikes-KO-on
                          #   -entry → forced replacement) -> spikes golden; records the per-side SPIKES LAYERS;
                          #   fail-loud require floors (a lay / a chip / a FAIL / an immune entry / a spikes-KO) +
                          #   a stall guard
    probe_spikes_rng.js   # the SPIKES PRNG draw PROBE: monkey-patches the sim PRNG to count draws over constructed
                          #   spikes scenarios — proved the Spikes move + the switch-in EntryHazard damage are BOTH
                          #   DRAW-FREE, the maxhp/8 ÷6 ÷4 amounts, the Flying/Levitate ZERO, and the KO-on-entry chain
    gen_phaze_golden.js   # drives the sim (gen3customgame) full battles with PHAZING (Roar / Whirlwind → the
                          #   accuracy roll [acc 100, NOT never-miss] + the random-target sample + the drag → Spikes
                          #   chip / spikes-KO chain) -> phaze golden; sweeps seeds so DIFFERENT mons get dragged
                          #   (records the dragged species); fail-loud require floors (a random drag with ≥2 distinct
                          #   mons / the n=1 drag / a phaze FAIL / a phaze-into-Spikes chip) + a stall guard
    probe_phaze_rng.js    # the PHAZE PRNG draw PROBE (the crux): monkey-patches the sim PRNG to record every draw +
                          #   its sim/* site over constructed phaze scenarios — proved the acc-100 roll (NOT never-miss),
                          #   the n=1 `sample` STILL draws, the FAIL no-sample, and the dragIn → runSwitch order
    probe_phaze_regression_rng.js # captures the GROUND-TRUTH seeds for the deterministic phaze regression tests
                          #   (the acc-then-n1-sample, the fail-no-sample, the phaze-into-a-spikes-KO chain)
    gen_explosion_golden.js # drives the sim (gen3customgame) full battles with EXPLOSION / SELF-DESTRUCT (the
                          #   gen-3 self-KO that precedes the hit) -> explosion golden; records per-side FAINTED + SUB-HP
                          #   (proves the user faints through Protect/Ghost/a-sub-break); reuses the substitute TAB shape
    probe_explosion_rng.js # the EXPLOSION PRNG draw PROBE (the crux): monkey-patches the sim PRNG to count draws +
                          #   snapshot the user faint over constructed edges — proved the self-KO is UNCONDITIONAL +
                          #   DRAW-FREE + precedes the hit (faints through Protect / Ghost immunity / a Substitute)
    probe_explosion_regression_rng.js # captures the GROUND-TRUTH seeds for the deterministic explosion pins E1-E4
                          #   (into-a-Protect, into-a-Ghost, breaks-a-sub, the mutual double-faint TIE)
    gen_pp_struggle_golden.js # drives the sim (gen3customgame) full battles with PP TRACKING + STRUGGLE -> pp_struggle
                          #   golden; records the 4 move slots' PP + the struggle/recoil/pressure2/immune branch flags
    probe_pp_struggle_rng.js # the PP + STRUGGLE PRNG draw PROBE (the crux, SETTLED the wrong hints): counts draws +
                          #   dumps moveSlots[k].pp/.maxpp — proved PP init = pp*8/5 (3 PP-ups), −1/use draw-free (−2
                          #   into Pressure, none on a can't-move turn), forced Struggle at 0 PP, and Struggle's model
                          #   (accuracy 100 NOT never-miss, recoil = max(floor(dmg/4),1) via the gen3 recoil:[1,4] path)
    probe_pp_struggle_regression_rng.js # captures the GROUND-TRUTH seeds/PP/recoil for the 4 deterministic PP/Struggle pins
    gen_taunt_disable_golden.js # drives the sim (gen3customgame) full battles with TAUNT + DISABLE -> taunt_disable
                          #   golden; records per-side TAUNT presence + the DISABLED slot + the branch flags
                          #   (tauntStart/disableStart/miss/fail/struggle); scenarios pin BOTH disable-duration
                          #   branches at their exact free-up boundaries (the ±1 off-by-one gate)
    gen_trapping_golden.js # drives the sim (gen3customgame) full battles with TRAPPING -> trapping golden;
                          #   records per-side sim `pokemon.trapped` at move boundaries + the drag flag; scripts
                          #   respect the sim's trapped flag (a trapped mon fights — the rejection path is
                          #   pin-covered, not golden-scripted)
    probe_trapping_rng.js # the TRAPPING semantics + PRNG PROBE (the crux): request/rejection flow (trapped =
                          #   'hidden'; a rejected switch is draw-free + boundary stays open), grounded/Steel/
                          #   Flying/Levitate/GHOST semantics (a grounded Ghost IS trapped in Showdown-gen3),
                          #   both mirrors (Magneton +4 draws/endTurn vs control; Dugtrio 0), phaze bypass,
                          #   forced replacements un-gated
    probe_trapping_regression_rng.js # captures the GROUND-TRUTH seeds/trapped/species for the 5 deterministic
                          #   trapping pins (T1-T5)
    probe_intimidate_substitute_rng.js # GROUND TRUTH for the Intimidate-vs-SUBSTITUTE gate (sub up -> NO Atk
                          #   drop, seed-neutral) — the e2e-surfaced I1 pin's probe
    probe_taunt_disable_rng.js # the TAUNT + DISABLE PRNG draw PROBE (the crux): counts draws + dumps the volatile
                          #   durations + the request disabled flags — proved taunt = acc-100 draw ONLY (FIXED
                          #   duration 2, no draw), disable = acc-55 + ONE random(2,6) (none on a no-lastMove fail),
                          #   and the endTurn runEvent('DisableMove') tie-shuffle for a taunt+disable mon
    probe_disable_full_lifecycle.js # the SETTLED-ground-truth probe (+ 5 siblings: probe_disable_{onstart,
                          #   duration_branch,duration_direct,willmove_determinant,reviewer_scenario}.js): stored
                          #   post-onStart disable duration per branch — disabler-FIRST = rolled, disabler-SECOND =
                          #   rolled+1 (REFUTED a base-source-reading review; the gen4 mod's onStart is the truth)
    probe_taunt_duration_branch.js # taunt duration per branch: constant 2 in ALL branches (taunter-second-on-turn>=2
                          #   included) — the base onStart's duration++ is SHADOWED by the gen4 mod's replaced onStart
    probe_taunt_disable_onbeforemove_rng.js # the EXECUTION-TIME block probe: a QUEUED status/disabled move is cant'd
                          #   draw-free with NO PP (taunt at priority 0 AFTER the para roll: +1 draw when paralyzed;
                          #   disable at priority 7 BEFORE it: identical draws) + Seismic Toss executes under taunt +
                          #   the taunt residual -end precedes a slower foe's brn -damage (order 10/subOrder 15 proof)
    probe_taunt_disable_regression_rng.js # captures the GROUND-TRUTH seeds/state for the TD1-TD4 taunt/disable pins
    gen_e2e_fuzz.js       # THE CAPSTONE — loads the REAL teams (data/teams/*.txt), pairs + seeds them
                          #   from a fixed MASTER_SEED, drives the sim full battles with RANDOM legal
                          #   modeled choices -> e2e_fuzz_golden.txt (the bit-for-bit gate) +
                          #   e2e_fuzz_taxonomy.txt (the ranked ability/item coverage-gap list)
    trace_turn_rng.js     # instrumented per-turn PRNG draw tracer (single-turn crux investigation)
    trace_multiturn_rng.js # instrumented MULTI-turn PRNG draw tracer (eachEvent shuffles + residuals)
    trace_switch_rng.js   # instrumented SWITCH-turn PRNG draw tracer (switch-phase + post-faint crux)
    trace_status_secondary_rng.js # instrumented STATUS onBeforeMove + SECONDARY draw tracer (this step's crux)
    gen_protocol_capture.js # PROTOCOL-EMISSION (level-2) capture — drives the sim (gen3customgame) over 11
                          #   scenarios x 6 seeds, captures the RAW OMNISCIENT |...| stream verbatim (|t:|
                          #   normalized) -> tests/vectors/protocol_capture_golden.txt (66 battles, 9740 lines,
                          #   38 line types); the byte target protocol_test.rs replays + diffs the Phase-1 subset
```

## Build & test

Needs a Rust toolchain (`rustup`, stable). Then:

```bash
cd src/rust_sim
cargo test            # runs unit smokes + the differential golden test
```

## A/B differential fuzzer — the overnight parity hunter (runbook)

`harness/ab_fuzz.js` runs for **hours unattended**, driving the REAL Showdown sim
and this port side-by-side over generated teams, hunting divergences and saving a
minimized, standalone-replayable repro for each. It reuses the e2e capstone's
recorder (same TAB golden format) + modeled-universe predicates (required from
`gen_e2e_fuzz.js` — one source of truth), and the Rust side replays chunks via
`src/bin/ab_replay.rs` (per-battle JSON verdicts, never panicking on divergence).

### Launch overnight

```bash
cd src/rust_sim
# Showdown's OWN gen3 randbats generator (default mode), 12 hours:
nohup node harness/ab_fuzz.js --mode randbats --hours 12 > /dev/null 2>&1 &
# The modeled-universe generator (widest coverage of the modeled surface):
nohup node harness/ab_fuzz.js --mode random --hours 12 > /dev/null 2>&1 &
# Or a fixed battle count / the e2e's 585 filter-clean real teams:
node harness/ab_fuzz.js --mode pool --battles 500
```

Flags: `--mode randbats|random|pool` (default randbats) · `--battles N` /
`--hours H` (default: run until killed) · `--master-seed S` (default from time —
ALWAYS printed, so any run is reproducible) · `--chunk N` (default 25) ·
`--out DIR` (default `harness/ab_fuzz_out/`) · `--keep-chunks` (default: clean
chunk files are deleted after replay; divergent battles are always saved
standalone). No Showdown server is needed (in-process BattleStream only); the
driver `cargo build --release --bin ab_replay`s once at startup.

### Where results land / how to read them

- `<out>/ab_fuzz.log` — ONE stats line per chunk (append-only):
  `… chunk=41 battles=25 ok=25 diverged=0 cum_battles=1050 cum_ok=1048
  cum_diverged=2 cum_panic=0 … kinds=state=2 species=213 moves=118
  adj_rate=0.008 bph=2450 …`. `kinds=` is the cumulative first-divergence
  taxonomy (`seed|state|status|boost|confusion|spikes|species|firstmover|
  request|decision_count|ended|winner|panic|start_error`); `bph` =
  battles/hour; `adj_rate` (randbats) = sets touched by the item/ability
  adapter / sets fielded.
- `<out>/divergences/<runid>_<battleid>/` — one SELF-CONTAINED repro per
  divergence: `battle.txt` (a single-battle chunk in the e2e TAB format —
  replayable standalone forever, independent of generator drift) +
  `summary.json` (mode, master seed, init/choose/battle seeds, packed teams,
  choice tokens, and the first divergence: kind + decision index +
  expected-vs-got + detail).
- `<out>/summary_<runid>.json` — the end-of-run cumulative summary (tallies,
  divergence kinds, coverage counts, randbats adjustment stats, repro dirs).

### Replaying a repro

```bash
./target/release/ab_replay harness/ab_fuzz_out/divergences/<runid>_<battleid>/
# → {"battle":"ab_3_7","verdict":"diverged","kind":"seed","decision":12,
#    "expected":"…","got":"…","detail":"…"}
```

The replayer accepts a repro dir (reads its `battle.txt`) or any chunk file.
After an engine fix, the same command must print `"verdict":"ok"` — that flip is
the fix's acceptance test.

### Repro → regression pin (the project's edge-case→pin law)

Every REAL engine bug a repro isolates gets a NAMED, DETERMINISTIC pin in
`tests/regression_test.rs` when it is fixed: reduce the repro (its
`summary.json` carries the exact packed teams + init seed + choice tokens —
`battle.txt`'s INIT/DEC lines are the script), reconstruct the minimal
scenario as a constructed-team test (mirror the existing pins), assert the
exact post-turn STATE **and the post-turn SEED** for draw-order bugs, and
verify the pin FAILS with the fix reverted. The repro dir stays as the
provenance record; the pin is the permanent guard.

### SIGINT / reproducibility

Ctrl-C (SIGINT) finishes the current chunk, prints + writes the cumulative
summary, and exits 0 (a second SIGINT aborts immediately) — so an overnight
run can always be stopped cleanly. Every run prints its `--master-seed`; the
same mode + master seed + code replays the same battles. Repro dirs are
additionally self-contained, so they replay even after the generator changes.

### Modes (what each hunts)

- **randbats** (default) — Showdown's OWN `gen3randombattle` team generator
  (`Teams.generate('gen3randombattle', {seed})`, deterministic under the
  master seed). Sets are made port-replayable at the SET level: unmodeled
  item → Leftovers, unmodeled ability → the species' modeled/no-op ability
  (else the team is rejection-sampled — real gen3 species are saturated with
  unmodeled abilities, so expect a high disclosed rejection rate); a missing
  nature is normalized to Hardy (neutral — stat-identical in both engines).
  Movesets/levels/EVs stay untouched; the choice picker simply never PICKS an
  unmodeled move (a battle forced into one is kept as a comparable PREFIX).
  The per-chunk `adj_rate` disclosure marks these teams "randbats-derived".
- **random** — the MODELED-UNIVERSE generator: 6 species sampled from every
  gen3-dex species whose learnset ∩ modeled moves has ≥4 options (≥1 modeled
  DAMAGING move forced per mon), modeled/no-op ability, modeled item, random
  nature/EVs/IVs, level 100. The widest legal coverage of the claimed modeled
  surface — the mode that flushes out modeled-predicate ↔ engine drift the 151
  real teams never exercised.
- **pool** — the e2e capstone's filter-clean `data/teams/` pool (585 teams),
  fresh seeds/choices every run.

## Regenerating the PRNG golden vectors

The golden vectors are captured from the **real** Showdown `prng.js`. Regenerate
after any PRNG change (needs the `deps/pokemon-showdown` submodule's `dist/` +
`node_modules` symlinks — see the root `CLAUDE.md` "Git Worktree Setup"):

```bash
node src/rust_sim/harness/gen_prng_vectors.js
```

This first cross-checks the dependency-free JS reference against the real lib and
**aborts if they ever differ**, so a regenerated `prng_golden.txt` is always a
faithful capture. Then `cargo test` re-pins the Rust against it.

## Regenerating the gen3 mechanics inventory + the item mechanics data

The DATA-DRIVEN MECHANICS FRAMEWORK's class map (`tests/vectors/gen3_mechanics_inventory.md`)
and the committed item/ability mechanics fields are derived from the RESOLVED `Dex.mod('gen3')`
(never a raw data file — the mod-chain law). After a Showdown submodule bump or an extractor
change (needs the submodule `dist/` + `node_modules` symlinks):

```bash
node src/rust_sim/harness/dump_gen3_mechanics.js            # rewrites the inventory .md
node src/rust_sim/harness/dump_gen3_mechanics.js --check    # DRIFT GATE: committed
                                                            # gen3_items/abilities.json
                                                            # vs the resolved dist
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    tools/pokemon_data_extractor/sync.py --datasets items abilities
```

Run `--check` after every regeneration; the Python side is separately pinned by
`src/agents/gen3_data/extractor_parity_test.py`.

## The handler-completeness audit (manifest + gate)

The STATIC completeness guarantee for the at-site dispatch model
(`gen3_handler_audit_v1` — CLAUDE.md "## Data-driven mechanics" → "### Handler-completeness
audit"): every resolved handler key on every REACHABLE effect (MODELED∪NOOP abilities,
MODELED items, the engine-enterable conditions, the isModeledMove moves + struggle) has an
explicit disposition in `tests/vectors/gen3_handler_audit.json` (human census:
`gen3_handler_audit.md`), body-fingerprint-pinned against the resolved dist:

```bash
node src/rust_sim/harness/dump_gen3_handlers.js --audit     # THE GATE (also inside cargo
                                                            # test via tests/handler_audit_test.rs):
                                                            # fails on a NEW handler, a stale
                                                            # row, a body-fingerprint drift,
                                                            # or a dead `implemented` anchor
node src/rust_sim/harness/dump_gen3_handlers.js             # regenerate manifest + census
                                                            # (after triaging in
                                                            # harness/handler_audit_dispositions.js)
node src/rust_sim/harness/dump_gen3_handlers.js --enumerate # raw row dump (triage/debug)
```

Triage rule: a fingerprint drift or a new key means the dist's semantics moved — PROBE first
(the sim is the only oracle), then disposition in `handler_audit_dispositions.js` and
regenerate. Never wave a gap through: a real miss is a latent bug (the first run caught the
Jump Kick/HJK crash + Freeze Clause Mod — see EDGE_CASES.md).

## Regenerating the item-modifier class-sweep golden

Covers EVERY wired item-class member (24 TYPE_BOOST + 6 SPECIES_STAT + Choice Band, with
matching + wrong-type/wrong-species controls) as full battles to game-end — per-decision
STATE+HP+SEED:

```bash
node src/rust_sim/harness/gen_item_mods_golden.js
```

`tests/item_mods_test.rs` replays all 33 scenarios × 30 seeds (990 battles) and enforces
>=10 boosted-hit rows per member. The exact fold math is additionally pinned by the
damage golden's 17 item probes (`node src/rust_sim/harness/gen_damage_golden.js` →
`tests/damage_test.rs`, 48 EXACT max-roll scenarios).

## Regenerating the dex golden vectors

The dex golden is dumped from the `agents.gen3_data` facade (the Python runtime's
source of truth). Regenerate after any data or category-derivation change:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/rust_sim/harness/gen_dex_golden.py
```

`cargo test` then asserts the Rust dex reproduces every line, so the Rust dex and
the Python runtime can't silently drift apart.

## Regenerating the team golden vectors

The team golden is captured from the **real** Showdown `Teams.pack`/`unpack`.
Regenerate after any change to the team codec (needs the submodule `dist/` +
`node_modules` symlinks):

```bash
node src/rust_sim/harness/gen_team_golden.js
```

It aborts if `Teams.pack` isn't idempotent on its own output, and emits
`(IN, UNPACK, PACK)` triples — Showdown-canonical, poke-env lowercase, and raw
edge-case fixtures — that `tests/team_test.rs` pins the Rust codec against.

## Regenerating the switch-in golden vectors

The switch-in golden is captured from a **real** gen3 battle's POST-switch-in
state (omniscient `BattleStream`, no server). Regenerate after any change to the
switch-in event dispatch (needs the submodule `dist/` + `node_modules` symlinks):

```bash
node src/rust_sim/harness/gen_switchin_golden.js
```

It starts 5 scenarios (Sand Stream + Intimidate both directions, the
single-setter Drizzle/Drought, and the order-dependent double-weather case),
reaches the first request, and dumps each lead's Atk boost stage +
`field.weather`, which `tests/switchin_test.rs` pins `Battle::start_with_switchins`
against. The scenarios use distinct-speed leads so the switch-in dispatch draws
nothing — the boosts/weather are seed-independent (the seed is pinned `[1,2,3,4]`
so the Rust builds the same `Prng`).

## Regenerating the turn golden vectors

The turn golden is captured from a **real** gen3 single-turn battle (omniscient
`BattleStream`, no server): both sides use one damaging move, and the harness reads
the post-turn state PLUS the PRNG seed right **before** and **after** the turn.
Regenerate after any change to `turn.rs` or the move-execution draw order (needs the
submodule `dist/` + `node_modules` symlinks):

```bash
node src/rust_sim/harness/gen_turn_golden.js
```

It runs **15 scenarios × 60 seeds** (neutral / STAB / SE / resist / Choice Band /
rain / never-miss / sub-100-accuracy / high-crit / low-level / type-immune /
guaranteed-faint / two speed-ties), GUARDS that each "distinct-speed" scenario's
actives don't silently tie on action speed, and emits one row per (scenario, seed)
with `seed_before` / `seed_after` + the post-turn hp/fainted/crit/miss/moved.
`tests/turn_test.rs` seeds its `BattleState` prng with `seed_before` (sidestepping
the `>start` setup draws this step omits), runs `run_turn`, and asserts the
post-turn state AND — for the distinct-speed rows — the **post-turn PRNG seed equals
`seed_after`**: 780 EXACT seed-parity assertions, the bit-for-bit proof that the
turn consumes the PRNG in Showdown's exact order and count. The speed-tie rows here
assert the action-order shuffle's first-mover (full tie-cycle seed parity is closed
by the multi-turn golden below). The companion `harness/trace_turn_rng.js`
instruments the live sim's PRNG to print the per-turn draw order — the investigation
that pinned this sequence (it is not a golden generator).

### Multi-turn golden (`tests/vectors/battle_golden.txt`)

The battle golden is captured from **real** gen3 MULTI-turn battles (omniscient
`BattleStream`, no server): bulky mons trade a damaging move over several turns so the
end-of-turn residuals fire. Regenerate after any change to `turn.rs`'s cycle/residuals
or draw order (needs the submodule symlinks):

```bash
node src/rust_sim/harness/gen_battle_golden.js
```

It runs **12 scenarios × 40 seeds × several turns** — leftovers (no weather),
sandstorm (Tyranitar Sand Stream + the chip), burn / poison / Toxic-ramp (status
applied by a status MOVE on turn 1, which the port defers, so recording starts at
turn 2 with the status injected — the DoT is draw-free so cross-turn parity holds),
rain (no chip), and SPEED-TIE (snorlax / sand-tie / tauros-no-Leftovers). Per turn it
records `seed_before` / `seed_after` + both actives' hp/maxhp/fainted/status (+ Toxic
stage) + first-mover. `tests/battle_test.rs` seeds a `BattleState` at the first
recorded turn, injects the init status, and runs `run_battle` WITHOUT re-seeding,
asserting per turn the hp/maxhp/fainted/status + first-mover AND the post-turn PRNG
seed == `seed_after`. It runs BOTH a single-seed cross-turn carry AND a **per-turn
re-seed pass** that pinpoints the FIRST diverging turn — ~**2034 EXACT post-turn-seed
assertions**, now including the SPEED-TIE class (FULL prng-state parity, closing the
single-turn step's tie deferral) and the residual phase. The companion
`harness/trace_multiturn_rng.js` instruments the live sim's PRNG to print the
per-turn draw order incl. the `eachEvent` shuffles + residuals (the investigation that
pinned the cycle; not a golden generator).

### Full-battle golden (`tests/vectors/fullbattle_golden.txt`)

The full-battle golden is captured from **real** gen3 battles played to GAME-END
(omniscient `BattleStream`, no server) with scripted MOVE + SWITCH + replacement
choices — so it exercises voluntary switches, the post-faint forced-replacement
request (single + double), and win/loss. Regenerate after any change to `turn.rs`'s
switch phase or draw order (needs the submodule symlinks):

```bash
node src/rust_sim/harness/gen_fullbattle_golden.js
```

It runs **8 scenarios × 50 seeds** to game-end — both-switch distinct/tie,
switch-vs-move, post-faint single + double replace, KO-to-win, and a last-mon
double-KO TIE (all moves are **secondary-free** so no unmodeled `random(100)`
desyncs). It drives the sim by reading `requestState` (move vs `forceSwitch`) per
decision and submitting the scripted `move`/`switch` choice (recorded as a compact
token the Rust replays verbatim — duplicate-species safe). Per DECISION boundary
(each move turn AND each forced-switch sub-step) it records `seed_after` + both
actives' species/hp/maxhp/fainted/status + side `pokemonLeft` + first-mover, and the
final `|win|`/winner. `tests/fullbattle_test.rs` seeds a `BattleState` at the
pre-first-decision seed and runs `run_full_battle` WITHOUT re-seeding, asserting per
decision the active species/hp/fainted/status + pokemon_left + request kind + first
mover AND the post-decision PRNG seed == `seed_after` — ~**2053 EXACT per-decision
seed assertions** to game-end — PLUS the final winner (or tie). The single-seed carry
IS the per-decision pass (each intermediate boundary is asserted, so the first
diverging decision panics with its index). The companion `harness/trace_switch_rng.js`
instruments the live sim's PRNG to print the switch-phase + post-faint draw order (the
investigation that pinned the switch model; not a golden generator).

### Protocol capture golden (`tests/vectors/protocol_capture_golden.txt`)

The **level-2** target: the RAW OMNISCIENT `|...|` protocol stream Showdown emits,
verbatim, per battle — the byte set `protocol.rs`'s `ProtocolBuilder` reproduces.
Captured from the omniscient `BattleStream` (no server, no per-side privacy fold →
full `x/y` HP both sides) over 11 scenarios × 6 seeds; the `|t:|` wall-clock line is
normalized to `|t:|<NORMALIZED>` (un-reproducible + poke-env-ignored). Regenerate
after any change to the scenarios or the sim's line ordering (needs the submodule
symlinks):

```bash
node src/rust_sim/harness/gen_protocol_capture.js
```

`tests/protocol_test.rs` replays each battle through `run_full_battle_logged`,
FILTERS both the golden's lines and the engine's output to the **Phase-1 + Phase-2**
line types (only `debug` [poke-env-ignored] + the still-deferred mechanics' lines are
dropped from BOTH sides — a real subset-equality, not a fake pass), and asserts
BYTE-EQUALITY per line, in order, with a first-divergence panic (a turn-capped/truncated
golden — one with no terminal `|win|`/`|tie|`, e.g. `spikes_and_phaze/2`'s infinite
Spikes-at-cap↔immune-EQ stall — is asserted as a byte-exact PREFIX of the longer engine
output). **51 battles / 5630 lines byte-equal** across 9 scenarios: the 4 design-core +
`sand_intimidate_effectiveness` (Phase 1) PLUS `substitute_absorb` / `protect_block` /
`spikes_and_phaze` / `recover_and_rest` (Phase 2 — the status-move `|move|` announce +
`-status`/`-curestatus`/`cant`/`-boost`/`-weather`/`-ability`/`-fail`/`-sidestart`/
`-start`/`-end`/`-activate`/`-singleturn` lines). STILL deferred: `status_para_and_boost_
drop` + `secondary_status_flinch` — their status/boost lines ARE emitted AND (now) the
Seismic Toss lines replay byte-exact (fixed-damage is modeled), but both all-ST battles
ALSO exercise a **forced-replacement REQUEST-BOUNDARY resume** the port collapses (a
switching-layer nuance, same family as `forced_replacement_recaches_speed_seed`, NOT a
protocol gap) + 3 `recover_and_rest` battles that hit **Struggle** (no PP tracking).
Phase-2 emission is still **observation-only** — it draws no PRNG, so the whole seed suite
(`prng`/`dex`/…/`battle`/`fullbattle`/`secondary`/`e2e_fuzz`) stays green with IDENTICAL
seed-assertion counts (e2e 14228 / battle_test 2034 / fullbattle 2053); that is the
load-bearing proof it didn't perturb the engine. The line grammar + the parse/ignore split
are catalogued in `tests/vectors/protocol_inventory.md`; the design is
`PROTOCOL_EMISSION_DESIGN.md`.
