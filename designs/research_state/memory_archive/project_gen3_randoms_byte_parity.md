---
name: project_gen3_randoms_byte_parity
description: ACTIVE GRIND (started 2026-07-22) — get ALL gen3 random battles byte-for-byte in the rust_sim port (enables random-battle/diverse-team training, the Metamon regime). Measured tail is SMALL/bounded; work queue + models captured here for multi-session continuity. When byte-clean → kick off a 24h background random-battle fuzz.
metadata:
  node_type: memory
  type: project
  originSessionId: fe9aa832-f12b-4c5b-8209-a8b8c4ee882a
  modified: 2026-08-05T02:23:43.274Z
---

> **Archived 2026-09-08** — rust randoms grind; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record. Preserved verbatim; nothing below is current.

**GOAL (user directive):** "get all of gen3 randoms byte-for-byte working, in a quota-aware
manner" — use NODE for team generation (Showdown's `Teams.generate('gen3randombattle')`, already
wired in `ab_fuzz.js`), don't reimplement it. When done: kick off a **24h background random-battle
fuzz**. WHY it matters: the training POOL (`data/teams/`) is ALREADY 100% byte-for-byte (e2e
STRICT 220/220, taxonomy 300/300 ENGINE_GAP=none); this buys FULL-gen3 completeness → unlocks the
random-battle / diverse-team training regime ([[project_plateau_research_2026_06_25]] Metamon hope).

🟢 **STATUS 2026-08-04 — THE MOVE/ABILITY TAIL BELOW IS CLOSED (verified against the live harness,
not docs).** Every move the tail names is now MODELED: psychoboost, transform, the whole wrap family
(wrap/bind/firespin/clamp/whirlpool/sandtomb), haze, trick, volttackle, bonemerang, yawn — 13/13.
Both abilities too (liquidooze, wonderguard), plus forecast (ROUND 35). Full-universe census:
**369 gen3-legal moves → 281 modeled, 88 fail-loud, 0 MISMODELED** (`SCAN_UNIVERSE=1 node
harness/scan_move_coverage.js`, exits non-zero on any silent desync). Pool greedy set-cover tables
are EMPTY. The remaining 88 fail-loud moves are LATENT guards — measured exposure ZERO on both the
pool and the curated randbats movepool.
⇒ **The open question is no longer "what to model" but "does a randbats battle run byte-for-byte
end-to-end"** — i.e. the 24h background fuzz this memory was queued to trigger. Re-measure before
planning from the per-move counts below; they are kept as the historical work-queue.

**MEASURED TAIL (HISTORICAL — static scan of 600 randbats teams / 3600 mons — `/tmp/randbats_tail_scan.js`,
reuses `gen_e2e_fuzz.js` exports isModeledMove/MODELED_ABILITIES/NOOP_ABILITIES/MODELED_ITEMS):**
94.9% of random mons ALREADY fully modeled. The whole tail (by mon-count over 600 teams):
- UNMODELED MOVES: psychoboost(25) transform(24) wrap-family(18) haze(17) trick(13) volttackle(8)
  bonemerang(6) yawn(5). — BUT volttackle (recoilFraction 0.333, like doubleedge) + psychoboost
  (selfDrops {spa:-2}, like overheat) are ALREADY ENGINE-MODELED (data-driven recoil/self_drops);
  just PICKER-rejected (not in MODELED_RECOIL/SELFDROP_MOVES). → admit + byte-verify, no engine work.
- UNMODELED ABILITIES: liquidooze(23, reverse Leech drain — fail-loud) wonderguard(17, SE-only dmg)
  forecast(15, Castform — deferred, reporting surface unprobed).
- UNMODELED ITEMS: whiteherb(22, BOOST_RESTORE) stick(16, CRIT_ITEM, roadmap item 5).

**BATCH PLAN (proven cadence: probe node → model → verify byte-clean via the random fuzz [the
oracle] → revert-verified pin; a dedicated golden is optional for the tail — the fuzz IS the gate):**
- BATCH 7: **multihit family — ✅ DONE (`gen3_move_coverage_batch7_v1`, NOT yet shipped/committed).**
  `turn.rs::run_multihit` (reuses run_move's built ctx; refreshes crit + live defender types per strike
  for Color Change) + data `MoveData::{multihit(MultiHit::Fixed/Range), multiaccuracy}` (extractor
  pass-through, obs-neutral). DRAW MODEL confirmed vs source (battle-actions.ts:748): fixed=NO count
  draw; `[2,5]`=ONE `sample([2,2,2,3,3,3,4,5])`=random(8); per strike crit+random(16)+**per-strike
  secondary**(Twineedle psn)+eachEvent; STOPS at target faint (no Quick Claw). Triple Kick
  (multiaccuracy) FAIL-LOUDS + picker-excluded. **A cross-side ModifyDamagePhase1 fix rode this**
  (`speed.rs::modify_damage_phase1_shuffle`): gen3 screens are `onAnyModifyDamagePhase1` → EVERY screen
  across BOTH sides ties (size-k → k-1 draws); old gate only fired for foe-both-screens → a latent
  single-hit desync the multi-strike amplified + the random fuzz surfaced (both-LS boards).
  volttackle(recoil 1/3, NO gen3 secondary)+psychoboost(selfDrops spa-2) admitted (engine-ready).
  VERIFIED: `harness/probe_batch7_isolated.js` 80/80 node-vs-rust `--protocol` clean (1839 strikes);
  5 revert-verified pins MC108-112 (MC112=the cross-side fix); handler audit 915 rows; full cargo
  test GREEN; e2e md5 UNCHANGED (`3155eb…` — the 220-sample picks no multihit, the leech/disable/snatch
  situation, so isolated harness is the byte gate). PIN GROUND-TRUTH NOTE: pins use the PORT's opts_cg
  values (isolated-validated == sim); a sim `>start` probe can't reproduce the exact seeds (turn-0
  construction gender-sample/QuickClaw window differs from `start_with_switchins` — the project-wide
  seed-convention deferral). Probes kept: `probe_batch7_{multihit,isolated}.js`.
- BATCH 8: haze (boost reset) · trick (item swap) · yawn (delayed sleep) · wrap-family (partial-trap
  multi-turn lock+chip).
- BATCH 9: transform · wonderguard (SE-only damage) · forecast · liquidooze · whiteherb · stick.

**MISMODELED QUEUE (random-mode omniscient byte-fuzz, 250 battles, 371 species/217 moves — the
1st round; repros in `harness/ab_fuzz_out/mismodel_hunt_r1/divergences/`, gitignored, standalone-
replayable via `ab_replay <dir> --ab`):** 11 non-allowlisted, 0 panics = **7 seed (draw-count) + 4
protocol (emission)**. Seed repros' moves: Staryu[Reflect]/Poochyena[Protect] dec13; Suicune-switch/
Baltoy[PsychUp] dec22; Feebas[DragonBreath]/Forretress[RapidSpin] dec36; Geodude[Strength]/Gligar
dec28; Graveler[DoubleEdge]×2/Whiscash[DoubleEdge] dec16; Wingull[SteelWing]/Lombre[Absorb] dec7.
FIRST TRACE (probe_repro_simtrace.js on rmrx1346k_ab_0_16 dec13): the port's seedAfter[13] ==
node's post-residual-shuffle-PRE-Quick-Claw seed — the port drew ONE FEWER draw at that endTurn (a
RESIDUAL fieldEvent speedSort tie-shuffle OR the Quick Claw). Likely a residual-handler-sort or
screen-residual tie the port misses (Reflect side-condition residual?). Root-cause the rest with
probe_repro_simtrace.js (node trace) + an env-gated per-draw backtrace in `prng::next` (port trace)
+ diff — the [[feedback_determinism_means_fixable]] method. Several of the 7 likely CLUSTER.

**THE LOOP (quota-aware, ≤~50%/5h-window per [[feedback_quota_pacing]]):** model a batch / fix a
bug cluster → re-run `ab_fuzz.js --mode random --protocol` (+ `--mode randbats`, + per-side
`bridge_ab_fuzz.js`) → until BYTE-CLEAN (0 panics + 0 non-allowlisted divergences) over a large
sample, BOTH formats → then kick off the 24h fuzz. Tooling: ab_replay isolated build in
`/tmp/pokesim_target_bytefuzz`; the scan/probes in `/tmp/randbats_tail_scan.js`,`/tmp/probe_mh2.js`.

**RANDOM-MODE TAIL (the mismodeled hunt, task #81 — enumerated by `--mode random --protocol` @
master-seed 70701, 250 battles; NOT multihit bugs — the multihit engine is byte-clean isolated 80/80):**
- ✅ **STATUS-MOVE `[miss]` EMIT gap FIXED** (`gen3_status_move_miss_emit_v1`, UNCOMMITTED — accumulating
  for the next /gen3ai-ship). `run_status_move`'s ENCORE / PHAZE(Roar) / SELF-BOOST arms returned draw-free
  on an accuracy MISS but emitted NOTHING; the TAUNT arm emitted `|-miss|` but not the `[miss]` attr. So a
  status move that missed via **Bright Powder / Lax Incense / a defender evasion boost** (acc<100) showed
  `|move|…` without `[miss]`. All 4 arms now emit `attr_last_move_miss()` + `log.miss()` (the working
  Disable-arm pattern). EMISSION-ONLY → all committed seed suites BYTE-IDENTICAL (verified: protocol/
  regression 257/e2e/taunt_disable/phaze/batch6 green). Cleared 3 of the 8 divergences (Taunt 0_13 / Roar
  6_11 / Encore 9_11 → all replay `ok`). Guard: byte_fuzz_corpus fixture
  `34_status_move_miss_emits_miss_attr.txt` (revert-verified: reverting the taunt attr → diverges protocol).
- ✅ **CUTE-CHARM ATTRACT `-end` ORDER FIXED** (`gen3_attract_end_order_v1`, UNCOMMITTED). `execute_switch`
  emitted the attract source-left `|-end|<foe>|Attract|[silent]` BEFORE the `|switch|` line; the sim fires
  it as the `attract.onUpdate` AFTER the switch-in. Now CAPTURED pre-swap + emitted after the `|switch|`
  emit. EMISSION-ORDER-ONLY → all committed goldens BYTE-IDENTICAL (protocol/ability_batch4/regression 257/
  e2e/fullbattle/batch3 green). Clears the Attract class (1_5's first byte-divergence + 2_22).
- 🔧 **NEW DIAGNOSTIC: `ab_replay` `POKESIM_PROTOCOL_ONLY=1`** (UNCOMMITTED) — skips the per-decision SEED +
  STATE + decision_count + ended checkpoints, runs ONLY the protocol byte diff. Distinguishes a real draw
  bug (bytes diverge) from a decision-boundary-segmentation artifact (bytes clean, only the per-`makeRequest`
  seed checkpoint misaligns — the bab A2-anchor class). USED IT: 1_5 → the Attract-order byte bug (fixed),
  then a DOWNSTREAM curse-chip (line 448, a Curse KO the sim doesn't have → symptom of the dec-48 draw bug);
  1_6 → `winner` (port None/tie vs sim P2) + decision_count 60-vs-68, NO byte-line divergence (all common
  lines matched) — the endgame diverges in the last decisions.
- ✅ **ROOT-CAUSED (2026-07-23, fresh-context agent — DECISIVE): the dec-42 (ab_1_6, master-seed 70701
  `--mode random`) bug is a REAL DRAW BUG, NOT a segmentation artifact.** Port total `Prng::next()` = 257
  (60 decisions, ended=false) vs sim 293 (68 decisions, winner P2) — the total-draw-count test settled it.
  THE MECHANIC: **turn 37 both players use Light Screen → both sides have Light Screen up; the sim's
  end-of-turn `fieldEvent("Residual")` gathers the two Light-Screen side-condition DURATION handlers
  (Reflect `onSideResidualOrder=1` / Light Screen `=2`, priority/speed 0), which TIE → ONE size-2
  Fisher-Yates `random(0,2)` handler-sort shuffle** (sim draw #191). The port MISSES it: in
  `src/rust_sim/src/turn/residuals.rs` (~L36-67) the screen countdown is decremented DIRECTLY at the top of
  `run_residuals`, BEFORE the `handlers` list is built, so screens never become sorted residual duration
  handlers. This is the RESIDUAL-side sibling of the batch-7 `modify_damage_phase1_shuffle` (`turn/speed.rs`,
  the PER-HIT both-screens shuffle — separate event). FIX (in progress, fresh-context agent): register
  `ResidualAction::ScreenDuration { side, is_reflect }` (reflect order 1 / ls order 2), gather into
  `handlers`, remove the top decrement, emit `|-sideend|` from the handler-apply → two same-type screens tie
  → the draw fires. HARD: e2e md5 MUST stay `3155eb796cb4bf453c6053d769ba98e5` (observation-neutral unless an
  e2e battle exercises a both-sides-same-screen residual — then STOP + review, don't blind-regen); add a
  revert-verified pin; re-fuzz `--mode random --protocol --battles 60 --master-seed 70701` → ab_1_6 must → ok.
  (Sodium-PRNG trap the agent flagged: the seed after N draws is a PURE FUNCTION of N — a raw seed-stream
  prefix match is VACUOUS, it only re-confirms COUNT; the real signal is what each draw is USED FOR.)
- ✅ **LS FIX SHIPPED (2026-07-23, commit `722e2fb` on main; CLAUDE.md batch-2 screens note updated in the
  same commit).** `ResidualAction::ScreenDuration { side, is_reflect }` added (reflect order 1 / ls order 2,
  suborder 4, speed 0), gathered into the residual `handlers` list per-side-per-screen BEFORE the mon-fainted
  guard (side handler → effectHolder is the SIDE, mirrors `findSideEventHandlers` before
  `findPokemonEventHandlers`); the top-of-fn direct decrement REMOVED; `|-sideend|` emitted from the
  handler-apply on the duration-END (D4 `continue` pattern). 3 tracked files: `turn.rs` (+enum/consts),
  `turn/residuals.rs`, `tests/regression_test.rs`. **ab_1_6 → ok** (fuzz 58→59/60). New revert-verified pin
  `both_sides_light_screen_residual_draws_the_handler_sort_shuffle` (real-Showdown ground truth
  `57388,452,34593,29177` both-sides / `18464,3966,47670,60926` one-side control). MC112 constants
  legitimately updated (its dec0-both-cast-LS scenario now exercises the residual draw; port-derived seed per
  the batch-7 convention). e2e md5 UNCHANGED `3155eb796cb4bf453c6053d769ba98e5`, full cargo test green. Probes
  kept: `harness/probe_screen_residual_{shuffle,regression_rng}.js`. **TODO at ship:** document this in
  `src/rust_sim/CLAUDE.md` (batch-2/batch-7 screen sections) — deferred (uncommitted WIP, consolidated doc
  pass at /gen3ai-ship).
- ✅ **VALIDATION-FUZZ LOOP (2026-07-23, random `--protocol` @ seed 80808, 150 battles):** the 6 mechanics
  were fuzz-validated → the byte-fuzz CAUGHT 3 emission/gate bugs the STATE goldens MISSED (Yawn
  double-`[still]`; White Herb fired at switch-in but the sim fires it via `onAnyAfterMove` after the
  holder's next move; Wonder Guard fixed-damage byte form — Counter→Shedinja) — ALL FIXED + corpus fixtures
  35/36/37, fuzz 6→2 diverged, e2e md5 unchanged, full suite green. **LESSON: run a random-mode `--protocol`
  byte-fuzz per NEW mechanic — STATE+SEED unit goldens do NOT catch emission-byte / novel-combination bugs
  (now baked into the agent briefs as a self-validation step).** The 2 remaining are PRE-EXISTING TAIL BUGS in
  COMMITTED code (NOT batch-8/9) — **NEW QUEUE:** (a) **Soundproof does NOT block DAMAGING sound moves**
  (Hyper Voice/Uproar into a Soundproof mon → port `-damage`, sim `-immune|[from] ability: Soundproof`;
  `move_is_sound` exists but is unused in `run_move`'s damaging path — only the status-move path checks it);
  (b) a **Toxic-application-turn Quick-Claw draw desync** (post-status-turn seed one draw short → flips an
  endgame winner). Both DELICATE (the Soundproof fix RISKS the committed e2e golden → needs a deliberate
  reviewed regen) — triage in a fresh window.
- ✅ **MILESTONE — RANDBATS FUZZ ~99.8% CLEAN (2026-07-24, the "all gen3 randoms byte-for-byte" goal
  essentially MET).** Confidence sweep (random+randbats × 3 seeds × 150 battles, current fixed tree):
  **randbats = 2/3 seeds GREEN (0 divergences), 1 seed 1 protocol div ⇒ ~0.2% residual** — the 24h-fuzz
  target is clean. Random mode still ~2-3% (seed/state MODELED-UNIVERSE edge cases — a DEEPER SEPARATE hunt,
  NOT the randbats goal; the ab_4_6-class turn-0-construction firstmover is OFFLINE-fuzz-ONLY — the production
  bridge models construction via `new_construct_turn0`). ALL FIXES UNCOMMITTED on `b546595`: Forecast
  guard+reject; 3 batch-89 REGRESSIONS (Trace crash / White Herb lead-Intimidate / Yawn re-cast); 4
  pre-existing (Soundproof-vs-damaging-sound / Liechi at-cap emission / Rest-into-sleep-immune / Mimic-Disable
  slot-vs-id). **STRONG /gen3ai-ship candidate** (fixes real crashes/mismodels in shipped code) → recommend
  SHIP then launch the 24h randbats fuzz. ✅ **BOTH DONE (2026-07-24): SHIPPED as `f9175ea`** on main (clean rebase
  onto training commit 9bdb1c5); the **24h randbats `--protocol` fuzz is LAUNCHED** (master-seed 240724,
  `--hours 24`, out `harness/ab_fuzz_out/fuzz24h`, background bary1nzr6) — triage its repros on
  check-in/completion (the standing find→fix loop continues via the 24h finds). **1h RANDOM FUZZ (2026-07-24, seed 100125): 86 div / 4375 battles ≈ 2%, ZERO panics/crashes** —
  kinds 41% PROTOCOL/emission (CLUSTERABLE: `-immune`×8 DOMINANT [likely a Wonder-Guard/Soundproof edge], then
  Substitute×4 / White-Herb×2 / Trace×2 / Mimic×2), 33% seed (draw-count, diffuse), 19% state (damage, diffuse).
  READ: the tail is LOPSIDED-tractable — the ~40% emission half clusters into a few byte-neutral fixes; the
  ~50% seed/state half is the diffuse hard grind. ✅ EMISSION AGENT (a9ab3c0b) fixed 5 emission forms (corpus 42-46: Damp/Explosion `[still]`,
  Substitute-Shedinja `[weak]`, Trace-mirror, SolarBeam-sun-skip, WhiteHerb-residual) — RE-MEASURE (1000 random
  @ 100125): **16→10 diverged (1.6%→1.0%, −37%); protocol 5→1 CLEARED, seed 7→5, state 3→3, firstmover 1 (const
  artifact)** — confirms lopsided-tractable. ✅ SEED/STATE AGENT (a9ee0a4c) fixed 3 bugs (Focus-Band-survive-at-1 emission [corpus 47];
  multihit-per-strike-ctx-rebuild [Poison-Point armed Guts mid-Arm-Thrust, corpus 48]; Mimic×choice-lock
  self-overwrite [corpus 49]) → RE-MEASURE 10→5 (seed 5→3, state 3→1, protocol 1→0). **REFRAME: of the 5
  remaining, 3 = turn-0 construction Quick-Claw FIRSTMOVER ARTIFACT (NOT real bugs — offline-only, bridge
  models it), 2 = real deferred: ab_3_17 (confusion-self-hit × screen ModifyDamagePhase1 shuffle omitted —
  TRACTABLE) + ab_16_24 (accumulated-damage-history — HARD). Random-mode REAL-bug rate ≈0.2% (2/1000), down
  from ~2%.** CUMULATIVE over 2 agents: 16→5 (1.6%→0.5%), the tail now mostly benign construction noise.
  8 total fixes UNCOMMITTED on `f9175ea` (5 emission + 3 seed/state) = strong ship candidate. Frontier options:
  ab_3_17 fix + construction-firstmover ALLOWLIST (clean the random gate) / the 2 real bugs / ship+checkpoint.
  Diffuse tail = DIMINISHING RETURNS; randbats goal MET. ✅ SHIPPED `43b8b43` on main (the 8 fuzz-tail fixes).
  **USER DIRECTIVE (2026-07-24): IGNORE the Quick-Claw construction-window firstmover artifact** (known-benign,
  offline-fuzz-only — the production bridge models construction; FILTER it in triage, do NOT chase/fix/allowlist
  it); **KEEP HUNTING the real random-mode tail + running fuzz; weekly ceiling → 75%.** Next real targets: ab_3_17
  (confusion-self-hit × foe-screen `onAnyModifyDamagePhase1` double-gather shuffle omitted by `apply_confusion_self_hit`
  — TRACTABLE, diagnosed) + ab_16_24 (accumulated-damage-history — hard) + whatever fresh fuzz seeds surface. ✅ HUNT AGENT (a1507dc4) fixed 4 real bugs (ab_3_17
  confusion-self-hit×screen-shuffle [corpus 50]; ab_7_7 Beat-Up-strike-Focus-Band-draw [51]; ab_12_17
  spurious `-end Charge`-on-self-KO [52]; ab_19_1 Endure-wrongly-clamps-Future-Sight-resolve [53, removed a
  class-inferred guess]) → RE-MEASURE (QC artifacts excluded): **REAL bugs 5→1 across seeds 100125+200724
  (200724 now REAL-CLEAN 3→0; 100125 only ab_16_24 [accumulated-damage-history, HARD] + QC artifacts).**
  4 more fixes UNCOMMITTED on `43b8b43`. Random-mode REAL tail now VERY THIN (~0-1 per 1000-1500 at sampled
  seeds). Budget 64% weekly / 53% 5h (ceilings 75%/70%). FRESH-SEED FIND (300724/400724/515151, 3600 battles): 19 div
  ~0.5% — state 7 / seed 6 / protocol 3 / firstmover 3 (QC artifacts, IGNORED per user). NOT as thin as the
  2-seed sample — different seeds surface different real bugs. Tractable PROTOCOL trio (seed 400724): ab_47_23
  Rough-Skin+Recoil DOUBLE-COUNT (engine emits a spurious `-damage [from] Recoil` on a mon already fainted by
  Rough Skin); ab_43_12 Knock-Off×Endure×Deep-Sea-Scale emission; ab_29_5 switch-related. Diffuse seed/state =
  dense-combo grind (Curse/Endure/Rest/Beat-Up/Destiny-Bond). ✅ AGENT a9caa2de fixed 4 root causes → 9 repros
  (ab_47_23 Rough-Skin+Recoil: gen3 `spreadDamage` early-returns 0 with NO emit when the recoiling user is
  already 0-HP; ab_43_12 Knock-Off×Endure: a 0-dealt Endure hit is still a `damagedTargets` member so
  `onAfterHit` item-removal FIRES [gate `dealt>0`→`!absorbed`, ITEM-REMOVAL ONLY — extending it to contact-procs
  REGRESSED ab_34_19: `DamagingHit` gates on the per-target relayVar so Static does NOT fire on a 0-dealt hit];
  ab_29_5 White-Herb `onAnySwitchIn` restore at the SwitchIn step; **CURSE cluster common-cause: switch-out
  clearVolatile never cleared `curse` → a pivoted-out cursed mon got re-chipped maxhp/4 on re-entry** [4 repros]).
  Corpus 54-57, revert-verified. RE-MEASURE @400724: 9→3 div, REAL 7→1. DIAGNOSED: Encore×Mimic-overwritten-slot
  (sibling of `gen3_mimic_disable_self_overwrite_v1`). **24h RANDBATS FUZZ @49k battles: 106 div (0.22%), 0
  PANICS, kinds protocol=93/seed=10/state=2/boost=1 — ~90% EMISSION at scale (clusterable) BUT its binary
  PREDATES 8 fixes so it OVERSTATES the tail; re-measure randbats with the CURRENT binary.**
  ✅ SHIPPED `af68d88` on main (the 8 round-2 tail fixes, corpus 50-57; clean rebase over training c01e70e).
  ⚠️ **KEY MEASUREMENT (2026-07-24, CURRENT binary, randbats 2 seeds × 2500 = 5000 battles): 12 diverged
  (0.24%) — i.e. the 8 round-2 fixes did NOT measurably improve randbats (24h stale-binary was 0.22%).
  MY PREDICTION THAT THE FIXES WOULD CLEAR THE 93 EMISSION FINDS WAS WRONG.** WHY: those fixes came from
  RANDOM-mode repros (dense adversarial combos) and randbats DOESN'T PRODUCE those combos — **the randbats
  tail is a DISTINCT bug set.** LESSON: fix bugs found on the SURFACE YOU CARE ABOUT; random-mode fixes do
  not transfer to randbats. **11 of 12 randbats divergences are PROTOCOL/EMISSION and 8 fall in 3 CLUSTERS:
  (1) Leech-Seed×Protect ×4 [`-activate Protect` form]; (2) White-Herb emission ORDER vs an on-hit proc
  [Poison-Point `-status` / Rough-Skin `-damage` must precede `-enditem White Herb`] ×2; (3) Fire-Punch
  frz-cure BEFORE Static contact-proc ×2. Singles: WonderGuard×SolarBeam, Trace→FlashFire×WoW, LiquidOoze×
  LeechSeed double-faint ORDER, 1 seed.** Several involve mechanics added this session (likely more of my own
  regressions). ✅ AGENT a3703edd FIXED ALL 3 CLUSTERS + ALL 4 SINGLES (corpus 58-64: leech-seed-into-protecting-grass-foe /
  white-herb-after-the-contact-proc / fire-thaw-before-static-proc / wonder-guard-blocks-sun-skipped-solarbeam /
  traced-flash-fire-absorbs-wow / liquid-ooze-leech-double-faint-ORDER / choicelock-lazy-release-after-knock-off),
  full suite GREEN, e2e md5 unchanged, prng clean. Its partial re-measure showed 0 diverged at 575/2500 +
  25/2500. ✅ SHIPPED `71150e9` on main. 🎯 **DEFINITIVE RE-MEASURE (post-fix, shipped binary): randbats
  240724 = 2500/2500 ok, 0 diverged, GREEN-GATE **PASS**; 606060 = 2498 ok, 0 diverged (2 allowlisted turn-0
  construction artifacts), GREEN-GATE **PASS**. 5000 battles, ZERO divergences (was 12).** ⇒ **THE
  "ALL GEN3 RANDOMS BYTE-FOR-BYTE" GOAL IS MET at this sample** — randbats is byte-clean; the 7 emission-cluster
  fixes did it. CAVEAT: 5000 battles / 2 seeds is a SAMPLE not a proof — the 24h fuzz v2 (relaunched on this
  binary, seed 250724, out `ab_fuzz_out/fuzz24h_v2`) gave the REAL number ⇒ **24h FUZZ v2 DONE (70,267 battles, seed 250724, shipped 71150e9 binary): 22
  non-allowlisted div = 0.031% (1 per ~3,200 battles), 0 PANICS, 24 correctly-allowlisted turn-0 construction
  artifacts. vs the OLD stale-binary run's 0.22% ⇒ a 7× IMPROVEMENT.** ⚠️ **TEMPER THE "BYTE-CLEAN" CLAIM: the
  5000-battle 0-divergence sample was LUCKY, consistent with a 0.031% rate — randbats is NOT zero-bug, it is
  ~1-in-3200.** REMAINING 22 = 14 protocol + 8 seed; emission forms (small clusters): `-activate move: Heal Bell`
  ×2, `-heal <mon> tox` ×2, `-resisted` ×2, `-immune` ×2, `-fail`/`-fail heal` ×3, `-end Attract [silent]`,
  `-heal`, `-damage`. Repros: `harness/ab_fuzz_out/fuzz24h_v2/divergences/` (in worktree
  `bridge-cse_01PcqnMsWWniDYtm1yWY95UB`; gitignored, standalone-replayable).
- ✅ **TRIAGE OF THE 22 (2026-07-31).** All 22 STILL reproduce on the current tree (HEAD `59cb14f`, rust_sim
  unchanged since `71150e9`) — verified by rebuilding `ab_replay` (`CARGO_TARGET_DIR=/tmp/pokesim_report_build`)
  and replaying each dir. They CLUSTER into 5 protocol groups + a diffuse seed group; TWO are root-caused to the
  EXACT line, with the fix known:
  - **A (6 repros, 27% — the biggest, ROOT-CAUSED): phaze `-activate Suction Cups` where the sim emits
    `[still]`+`-fail`.** In EVERY repro the TARGET is on its LAST mon (verified from the DEC rows' remaining-count
    column 33/21). Showdown gates the whole force-switch on `canSwitch(target.side)` FIRST — both at
    `battle-actions.ts:1281` (`hitResult = !!canSwitch(...)` → `didSomething=false` → `-fail`+`attrLastMove('[still]')`)
    and again in `forceSwitch()`'s guard at `:1378` — so with no living bench the move fails and **Suction Cups'
    `onDragOut` NEVER runs**. The port checks Suction Cups BEFORE canSwitch: `src/rust_sim/src/turn/status_moves.rs`
    §(2c) ~L1229 returns `-activate` at L1234, and the `eligible_switch_ins` check is only at §(3) ~L1242. The doc
    comment at L1226-1228 asserts the WRONG thing ("this is BEFORE canSwitch matters"). FIX: move the
    `eligible.is_empty()` fail ahead of the Suction-Cups block. Repros: ab_69_3 / ab_1258_19 / ab_1266_19 /
    ab_1720_21 / ab_1726_8 / ab_2059_7.
  - **D (2 repros, ROOT-CAUSED — a ROUND-24 OVER-GENERALIZATION): Rough Skin must fire BEFORE the fire-thaw.**
    Sim: `-damage … Rough Skin` THEN `-curestatus frz [msg]`; port: thaw first. `DamagingHit` is sorted by
    `Battle.compareLeftToRightOrder` (battle.ts:421) = **ascending `order`** (missing order ⇒ 4294967296), then
    priority, then gather `index`. **`roughskin` carries `onDamagingHitOrder: 1`** (abilities.ts:3894) while
    `static`/`poisonpoint`/`flamebody`/`effectspore` and the `frz` handler carry NO order — so Rough Skin sorts
    FIRST (before the thaw) while Static sorts by index AFTER it (status gathered before ability). Round 24's
    `gen3_fire_thaw_before_contact_proc` fix (corpus fixture 60) hard-coded "thaw before ability" and MISSED the
    `onDamagingHitOrder` key — correct for Static, WRONG for Rough Skin. FIX: honor `onDamagingHitOrder` in the
    DamagingHit ordering instead of a flat status-then-ability rule. Repros: ab_70_2 / ab_2293_23.
  - **B (3 repros, hypothesis): Spider Web emits `-activate trapped` where the sim emits `[still]`+`-fail`.**
    The port ALREADY has an already-trapped fail arm (`status_moves.rs` §(3) ~L752), so the port believed the
    target UNTRAPPED — either the port dropped a `trapped_by` link the sim keeps, or the sim fails for another
    reason. All 3 targets are also on their last mon (Steelix/Azumarill/Registeel), so it may share cluster A's
    "no bench" root. NEEDS one probe. Repros: ab_1271_13 / ab_387_12 / ab_943_5.
  - **C (2 repros, LIKELY A HIDDEN BENCH-STATE BUG — the training-relevant one): Heal Bell cures a BENCHED mon
    the sim does not.** Port emits an extra `-curestatus|p1: Stantler|par|[silent]` / `p2: Rapidash|psn`. gen3
    inherits gen4's healbell (`data/mods/gen4/moves.ts:598`) whose `ally.cureStatus(true)` DOES emit the silent
    bench line — so the sim WOULD emit it if the mon were statused. ⇒ the port has a status on a BENCH mon the
    sim doesn't. `ab_replay` only checks the ACTIVE mon's status, so the byte diff is the ONLY detector. **Heal
    Bell is common in gen3ou (Blissey/Miltank/Celebi), so this cluster — unlike A/B/D/E, whose species are
    non-OU — could touch the training surface.** Repros: ab_1126_6 / ab_789_7.
  - **E (1 repro): Attract `-end [silent]` ordering when BOTH sides switch the same turn** — the port emits it
    after the FOE's `|switch|` but before its OWN; sibling of the shipped `gen3_attract_end_order_v1`. ab_400_21.
  - **SEED (8, diffuse, NOT clustered): draw-count desyncs at dec 0/4/8/9/30/42/51/83.** Two are suspiciously
    early (ab_2044_12 @dec0, ab_136_14 @dec4) — check the turn-0-construction artifact filter first (the fuzz
    allowlist caught 24 separately, so these are NOT the known QC form). Root-cause with
    `probe_repro_simtrace.js <repro-dir>` (node per-draw trace, replays RECORDED choices since R9's T2 fix) +
    an env-gated per-draw backtrace in `prng::next`, then diff.
  SUGGESTED ORDER (highest value/effort first): A (6, one-line reorder) → D (2, exact fix known) → C (2, likely a
  real state bug that could reach gen3ou) → B (3, one probe) → E (1) → the seed 8. Then re-run
  `ab_fuzz.js --mode randbats --protocol` at fresh master-seeds to re-measure the ~0.031% rate.
- ✅ **C + D + A FIXED (2026-07-31), 10 of 22 repros → `ok`. UNCOMMITTED** (no /gen3ai-ship). Full suite
  **562 tests / 0 fail**, e2e md5 `3155eb796cb4bf453c6053d769ba98e5` UNCHANGED, `dump_gen3_mechanics.js
  --check` OK (132 items + 76 abilities match the dist). Three corpus fixtures added
  (`byte_fuzz_corpus/65_phaze_no_bench_beats_suction_cups.txt`, `66_rough_skin_order_precedes_the_fire_thaw.txt`,
  `67_heal_bell_skips_a_fainted_ally.txt`), each REVERT-VERIFIED one-to-one (reverting all 3 fixes makes each
  fixture diverge on its OWN line: the Whirlwind announce / the `-curestatus frz` / the `-curestatus par [silent]`).
  - **C `gen3_heal_bell_skips_fainted_v1`** (`turn/status_moves.rs`, the team-cure loop) — MY EARLIER HYPOTHESIS
    ("a hidden bench-STATE divergence") WAS WRONG. The real cause is simpler: the port cured a **CORPSE**. In BOTH
    repros a Pursuit KO'd the statused ally moments earlier; gen4's healbell calls `ally.cureStatus(true)`, which
    early-returns at `if (!this.hp || !this.status)` (pokemon.ts:1676), so the sim skips a 0-HP ally. Added an
    `hp == 0` skip (keyed on HP, not the `fainted` FLAG, to mirror `!this.hp` through the deferred-faint window).
  - **D `gen3_damaging_hit_order_v1`** (data + `dex/abilities.rs` + `turn/{status,moves}.rs`) — a ROUND-24
    OVER-GENERALIZATION. `DamagingHit` sorts by `compareLeftToRightOrder` (battle.ts:421) = ASCENDING `order`
    (missing ⇒ 4294967296), and **roughskin alone in gen3 carries `onDamagingHitOrder: 1`** — so Rough Skin runs
    BEFORE the un-ordered `frz` fire-thaw, while Static & co. run AFTER it (gather order status→ability). Round 24's
    flat "thaw before ability" rule was right for Static, backwards for Rough Skin. Wired DATA-DRIVEN: new
    `damagingHitOrder` field (extractor `_GEN3_ABILITY_MECHANICS` + the `--check` drift gate +
    `AbilityData::damaging_hit_order`) and a new `DamagingHitPhase::{Ordered,Unordered,All}` splitting the
    DamagingHit region into two passes around the thaw (`All` at the fixed-damage tail — no thaw there).
  - **A `gen3_phaze_canswitch_before_dragout_v1`** (`turn/status_moves.rs` phaze arm) — the `canSwitch` gate now
    PRECEDES the Suction-Cups DragOut check, per battle-actions.ts:1281 + :1378 (both gate on
    `canSwitch(target.side)` BEFORE `runEvent('DragOut')`). A phaze into a foe on its LAST MON now emits
    `[still]`+`-fail` and Suction Cups never activates. The old code comment asserted the opposite and was wrong.
  **STILL OPEN (12):** B Spider Web ×3 (ab_1271_13/387_12/943_5), E Attract-order ×1 (ab_400_21), and the 8
  diffuse seed repros. ⚠️ B is NOT the same root as A — it survived the A fix, so the "shared no-bench root"
  hypothesis is REFUTED; it needs its own probe.
- ✅ **CLUSTERS B + F + E FIXED (2026-08-01) — ALL PROTOCOL-CLASS DIVERGENCES ACROSS BOTH RUNS ARE NOW
  CLOSED. UNCOMMITTED** (the `/gen3ai-ship` in that turn covered only C/D/A = `60c9c9f`; these three are
  NOT shipped). Suite **562/0**, e2e md5 `3155eb796cb4bf453c6053d769ba98e5` UNCHANGED, fixtures 68/69/70
  each revert-verified one-to-one. Status: fuzz24h_v2 **13/22 ok**, fuzz_r25 **2/6 ok**; 13 open = 1
  `species` + 12 `seed`.
  - **B `gen3_trap_link_survives_baton_pass_v1`** (`turn/switch.rs`, gate the trap-link clear on `!bp`).
    PROBE-SETTLED by the NEW `harness/probe_spiderweb_link_lifetime.js` — **the trapper NORMAL-switching
    out BREAKS the foe's `trapped` (true→false), but a BATON PASS does NOT (true→true)**. The severing is
    a side effect of the departing mon's `clearVolatile()` (pokemon.ts:1527-1531 → `removeLinkedVolatiles`);
    Baton Pass takes the `copyVolatileFrom` path instead and skips it — even though BOTH `trapped` and
    `trapper` are `noCopy: true` (conditions.ts:208-221), so the entrant does NOT inherit `trapper` and the
    target is left trapped with a DANGLING link. ⚠️ TWO source-read hypotheses died here first (I predicted
    BP re-points the link, then that the link always breaks) — the probe is the only oracle. Cleared 3 of 4:
    ab_387_12 / ab_943_5 / ab_533_11 → ok; **ab_1271_13 advanced past its (masked) protocol bug to a NEW
    `species` divergence at dec152** — a SEPARATE issue, not a regression (suite green incl. trapping_test's
    8346 trapped assertions + T1-T5).
  - **F `gen3_yawn_sub_before_immune_v1`** (`turn/status_moves.rs` — a NEW cluster this session's fuzz found,
    repro ab_707_4). PROBE-SETTLED by the NEW `harness/probe_yawn_fail_precedence.js`: **SUBSTITUTE OUTRANKS
    the sleep-immune ability.** Matrix: sub-only → `[still]`+`-fail`; immune-only → `-immune|[from] ability`;
    **sub+immune → `[still]`+`-fail`**. (A source read predicts the opposite — yawn's own `onTryHit` runs at
    `runEvent('TryHit')`, the sub blocks at the LATER `onTryPrimaryHit`.) The port's `yawn_subbed` gated on
    `!yawn_immune`, which flipped BOTH the branch AND — via `yawn_still` — the ANNOUNCE form. Corrected order:
    Protect > statused > **substitute** > sleep-immune > add.
  - **E `gen3_attract_end_skips_a_corpse_v1`** (`turn/switch.rs`, repro ab_400_21) — the SAME SHAPE as
    cluster C: a missing alive-guard. On a DOUBLE faint (both mons die to residual poison) the port fired the
    attract source-left clear against an ALREADY-FAINTED holder, emitting a phantom
    `|-end|<corpse>|Attract|[silent]`; the sim's corpse already ran `clearVolatile()` silently at faint and
    emits NOTHING. Guarded on `hp > 0`.
  **LESSON (now 3 instances — C, E, and the pre-existing thaw/Focus-Band guards): a "does this effect fire
  on a CORPSE?" alive-guard is a RECURRING bug class in this port.** Worth a sweep of every emit site that
  reads a mon other than the current actor.
- 🎯 **THE SEED CLUSTER IS NOT AN ENGINE DRAW BUG (2026-08-01) — the session's most important finding.**
  ⚠️ SUPERSEDES my earlier "they cluster by board (Kecleon 6 / Deoxys-mirror 4 / Furret 2)" note — the
  species pattern was a red herring; I never established causation and the real answer is orthogonal.
  METHOD: the NEW permanent `POKESIM_PRNG_TRACE=1` (port draws) diffed against `probe_repro_simtrace.js`
  (sim draws) — dedupe the sim's `randomChance` wrapper lines with `awk 'NR==1||$0!=prev'`.
  RESULT over ALL 13 open repros (12 seed + 1 species): **12 of 13 have ZERO port-only draws** — the port's
  stream is a byte-exact PREFIX of the sim's — and for **ab_618_7 (412 vs 412) + ab_262_11 (228 vs 228) the
  streams are IDENTICAL end-to-end** while `ab_replay` still calls them `kind=seed`. (Exception: ab_239_13
  has 1 extra draw at the very end; it is the `forced-unmodeled-move:struggle` non-ended prefix battle.)
  ⇒ **the port's RNG consumption is CORRECT on the whole remaining queue.** These are DRAW-FREE
  decision-stream desyncs: the port segments decision boundaries differently, so its per-decision seed
  CHECKPOINT sits at a different draw index (proved on ab_262_11 — BOTH the `expected` and `got` seeds occur
  EXACTLY ONCE in the port's OWN stream) and it consumes the fixed recorded choice list at a different rate,
  running out before the battle ends (`POKESIM_PROTOCOL_ONLY=1` → bytes CLEAN, only `winner` differs).
  ⚠️ **HONEST LIMIT — two hypotheses remain live and the draw evidence cannot separate them:** (a) pure
  HARNESS boundary segmentation in the offline `ab_replay` path, or (b) a real DRAW-FREE legality bug (the
  port rejecting a decision the sim accepts, via the reject-and-re-request gate). Needs a per-decision
  ACCEPTED/REJECTED comparison, NOT more draw tracing.
  **NEXT STEP (scoped, with a precedent): port the BRIDGE's round-20 A2 fix
  (`gen3_perside_seed_anchor_subsequence_v1` — align the sim's decision-seed list as a SUBSEQUENCE of the
  port's boundary-seed list) from `bridge_replay.rs` to `ab_replay.rs` — held to the SAME gate-integrity bar
  (round 20 shipped 9 `a2_anchor_tests` + an injected-extra-draw proof that a genuine desync is still
  caught). A sloppy anchor makes the omniscient byte gate VACUOUS — far worse than the artifact.**
  ⇒ **The headline "randbats tail = 0.031%" OVERSTATES engine incorrectness**: the seed half (12 of the 22
  v2 repros, 4 of the 6 r25 ones) is measurement, not mismodeling.
- ⚠️ **SELF-CORRECTION: my shipped `gen3_trap_link_survives_baton_pass_v1` (in `887b205`) was HALF-RIGHT
  — superseded by `gen3_trap_link_baton_pass_transfers_v1` (2026-08-01, UNCOMMITTED).** The `!bp` skip
  correctly keeps the trap alive across a Baton Pass, but hands the "who severs it" role to NOBODY, so the
  port's trap then persisted FOREVER: the foe could never switch again and every later request carried a
  spurious `"trapped":true`. Externally visible (poke-env stops offering switches to the policy) and
  DRAW-FREE, so no seed/omniscient gate can see it — **the external-consistency gate caught it within ~35
  battles** (repro `sim_bridge_diff_out/soak_randbats/divergences/sbd_msapcesj_b35`, an Ariados that
  Spider-Webs then Baton-Passes).
  PROBE-SETTLED (`probe_spiderweb_link_lifetime.js` case E — added for this): spiderweb → trapped TRUE;
  trapper BATON-PASSES → **survives** TRUE; **the ENTRANT then NORMAL-switches out → BREAKS false**. ⇒ a
  Baton Pass does NOT cancel the severing, it **DEFERS** it to the entrant. FIX: on a BP, RE-POINT the
  target's `trapped_by` to the ENTRANT's uid (pre-swap `target` index) instead of skipping the clear —
  the ordinary non-BP clear then fires naturally on the entrant's own next switch-out, no new state.
  Suite 567/0, e2e md5 unchanged, corpus fixture 68 still clean (the BP survival is preserved).
  **LESSON: when a probe shows behaviour X survives event A, ALSO probe what ends it later — I stopped at
  "it survives" and shipped a model where it never ends. "Survives A" and "is cancelled by A" are not the
  only two options.**
- ✅ **TRANSFORM FAIL-LOUD (`gen3_transform_failloud_v1`, 2026-08-01, UNCOMMITTED).** The bridge gate's
  other finding (repro `sbd_msapcesj_b22`): node emits `|-transform|p1a: Ditto|p2a: Kyogre`, the port
  emitted NOTHING — Transform is unmodeled and was NOT fail-loud, so a Ditto silently no-op'd and fed the
  policy wrong observations for the rest of the battle. Mirrored the Forecast guard: panic in
  `MonState::from_set` keyed on the **MOVE** (species-agnostic — Mew/Smeargle learn it too; a Ditto
  WITHOUT it is harmless), plus upstream `REJECT_MOVES` in `gen_e2e_fuzz.js::teamFilterClean` and the
  `ab_fuzz.js` randbats adapter so carrier teams are rejection-sampled instead of panicking. Verified ZERO
  of the 722 pool teams carry Transform/Ditto ⇒ the e2e golden CANNOT shift. Pins
  `transform_fails_loud_at_construction` (revert-verified: removing the guard → "test did not panic as
  expected") + the negative control `a_ditto_without_transform_builds_fine` (guards against a future
  over-broad species-keyed guard). Transform itself stays UNIMPLEMENTED (a large state-overlay job).
  **WHY IT SURVIVED EVERY OTHER GATE: the offline fuzzers' pickers filter Transform out, so it was never
  PICKED; the live bridge drives choices off the sim's OWN request, so a randbats Ditto reached it.**
- 🏆 **THE EXTERNAL-CONSISTENCY GATE (2026-08-01, UNCOMMITTED) — `harness/gen_sim_bridge_diff.js` promoted
  from a 120-battle validation harness to a green-gated fuzzer.** It is the STRONGEST gate we have: it
  spawns BOTH real bridges, feeds identical stdin, and **discovers decision boundaries LIVE** (read a
  request → choose → compare), so the `ab_replay` segmentation artifact CANNOT occur by construction, and
  it compares what poke-env ACTUALLY consumes (per-side stream + the `|request|` JSON) rather than the
  omniscient log poke-env never sees.
  **LAYERING PRINCIPLE (worth reusing):** per-side+request = the CONTRACT; omniscient = a LOCALIZER;
  seed = a LEADING INDICATOR. **An outer mismatch is ALWAYS a bug; an inner mismatch with a clean outer
  layer is NOT necessarily one.** Inner layers buy detection speed + localization, not correctness.
  - **Added a GREEN GATE + allowlist** (keys mirror `bridge_replay.rs::classify_known_perside_residual`).
    Before: any known-benign residual failed → a 10-battle smoke read **8/10 "diverged", ALL the same
    `return102` alias** ⇒ unusable. Now allowlisted ones are counted separately + filed under
    `<out>/allowlisted/`; exit non-zero ONLY on a non-allowlisted divergence.
  - **SAFETY = reconcile-then-compare** (any residual byte difference still FAILS).
  - ⚠️ **I WROTE AN UNSAFE TRANSFORM AND CAUGHT IT:** a symmetric strip of `, L<n>`/`, <gender>` from
    `details` would reconcile a genuine VALUE divergence (`L84` vs `L83`, M vs F) into a FALSE PASS — the
    exact "vacuous gate" failure. REMOVED; the run then still went green under `return102` alone, proving
    it was never needed. **LESSON: an alias→canonical transform is safe (a real value difference still
    fails); a symmetric STRIP is not. If a presence-vs-absence residual ever appears, do it PAIRWISE.**
  - **`--selftest`**: 10 gate-integrity assertions, negatives load-bearing (different move / alias+pp /
    LEVEL value / GENDER value / missing move / non-request / identical). Run after any allowlist change.
  - **THROUGHPUT MEASURED: ~600 battles/hr `--persistent` vs ~80/hr without** (a fresh Node child reloads
    the whole Showdown dist per battle). **~96% of wall is the per-write quiescence settle, NOT CPU**
    (`user` 2.3s / 59s wall) ⇒ `--persistent` is mandatory for a soak; the 40 ms settle is the next lever.
  - **SOAK RUNNING:** `--mode randbats --battles 1200 --master-seed 810125 --persistent --out
    harness/sim_bridge_diff_out/soak_randbats`, log `/tmp/sbdiff_soak.log`. It SETTLES the round-26
    (a)-vs-(b) question gating the `ab_replay` anchor: clean ⇒ (a) harness segmentation, anchor safe;
    request mismatches ⇒ (b) a real draw-free legality bug the anchor would have HIDDEN.
  - **NEXT STRENGTHENING:** parse both streams with the real `src/poke_env` fork and diff the resulting
    battle STATE — external consistency at the OBSERVER's abstraction, not the byte level. That dissolves
    the `return102` class outright (poke-env resolves both) instead of allowlisting it on an assumption
    that is currently UNTESTED.
- ✅ **ROUND 30 + 31 (2026-08-02/03, UNCOMMITTED in worktree `bridge-cse_01MVJ22dh5peA8cti4Sgrpgn`).**
  **R30 `gen3_turn0_quick_claw_capture_v1`:** the OFFLINE replay convention resumes at the
  POST-construction seed, so it SKIPPED the turn-0 endTurn that decides TURN 1's Quick Claw
  (`randomChance(1,5)`, read next turn) — unrecoverable from the seed, silently dropped ⇒ ~1 lead in 5
  mis-ordered. Captured as an OPTIONAL trailing `INIT` field on BOTH offline replayers (5th for
  `ab_fuzz`/`ab_replay`, 6th for the bridge sibling; ABSENT ⇒ false ⇒ every old golden/repro still
  replays). Omniscient random-mode gate **3 → 0 / 1000 battles**. **R31 (mine, 2026-08-03) — the
  PER-SIDE gate in `--mode random`, a surface it had NEVER been measured on: 252/400 diverged (63%).
  A 63% rate is a GATE bug, not an engine tail** — and it was 2 gate bugs + 1 real byte:
  (1) `gen3_happiness_bp_alias_any_digits_v1` — the numeric-BP allowlist hardcoded `102`, but the
  suffix is the COMPUTED BP: Frustration at happiness 255 clamps to **`frustration1`, NEVER
  `frustration102`** ⇒ the arm could never match, every Frustration board failed AND dragged its
  co-occurring correct `return102` down with it. Fixed with a digit-agnostic strip that is **PAIRWISE**
  (bails when both sides carry DISAGREEING digits — a symmetric strip would false-pass a mis-parsed
  happiness). SAME hole in `gen_sim_bridge_diff.js` (the strongest gate; its selftest even asserted the
  impossible `frustration102`) → fixed symmetrically. (2) `gen3_curse_reconcile_slot_bounded_v1` — the
  "dormant/legacy" Curse arm searched to END-of-line and rewrote the NEXT slot's `"target":"normal"`
  (a Double Edge), MANUFACTURING divergences. (3) `gen3_mimic_request_no_target_v1` — a REAL port bug:
  gen3 inherits gen4's Mimic, whose slot literal (`data/mods/gen4/moves.ts:868`) OMITS `target`, so the
  sim's request drops the key for a mimicked slot (`pp:5` + full `maxpp` is its signature); the port
  emitted it. **RE-MEASURE 252 → 0 diverged** (384 allowlisted, 0 panics), suite **579/0**, e2e md5
  `3155eb…` unchanged, 3 fixes + 1 corpus fixture `20_mimic_slot_request_omits_target_cg.txt` each
  REVERT-VERIFIED. **LESSONS: (a) re-measure before re-opening a divergence list (R30 began from a
  3-item list already closed by rounds 22/26); (b) the same gate on a NEW TEAM DISTRIBUTION is a
  DIFFERENT TEST — pool/trapping/randbats never pair Frustration+Curse or Mimic a move; (c) a
  "dormant, retained defensively" branch is the 3rd one found actively misfiring — delete or pin it.**
  NEXT: the per-side gate is green in random mode too ⇒ re-run at FRESH master-seeds (+ `--mode
  randbats`) to re-measure the ~0.031% tail, and re-run the `gen_sim_bridge_diff` soak now that its
  Frustration noise is gone. Needs a `/gen3ai-ship`.
- 🔬 **RE-MEASURE FUZZ RUNNING (2026-07-31):** `ab_fuzz.js --mode randbats --protocol --hours 8 --master-seed
  310725 --out harness/ab_fuzz_out/fuzz_r25_randbats` (FRESH seed — do not re-measure on 250724, the seed the
  fixes were derived from), log `/tmp/fuzz_r25_randbats.log`, run_id `rms9nh02e`, ~2650 battles/hr ⇒ ~21k
  battles. First chunk 25/25 ok. NOTE: 21k battles at the prior 0.031% rate expects ~6-7 divergences, so a
  LOW count is weak evidence — the honest read needs the divergence KINDS, not just the total. **QUOTA: hit STOP_WEEKLY at 79% (ceiling 75%) — HANDED BACK to user.**
  OLD 24h run (stale
  binary) final: 60473 battles / 135 div (0.22%) / 0 panics. Repros:
  `harness/ab_fuzz_out/now_rb_{240724,606060}/`. Budget 70% weekly (ceiling 75%) — NO more agents this window. [was: DRIVING the diffuse seed/state class with agents] (quota-gated 70%
  weekly). Repros under `harness/ab_fuzz_out/fuzz1h_random/divergences/`. Random-mode ~2-3%
  modeled-universe tail = a SEPARATE deeper hunt (NOT the randbats goal).
- ✅ **FORECAST GUARD + CURRENT-TAIL FIX (2026-07-23→24, post-`b546595`, UNCOMMITTED):** Forecast
  fail-loud in `state.rs::from_set` (GIGO — panics on a Castform/Forecast mon) + fuzz REJECT all modes
  (`gen_e2e_fuzz.js` REJECT_ABILITIES/REJECT_SPECIES + `ab_fuzz.js` randbats reject) — DONE, green,
  `#[should_panic]` test. Then a broad FIND fuzz (random+randbats @ seed 55667, 200 each) enumerated the tail
  = 9 divergences → surfaced **2 batch-89 REGRESSIONS (in shipped `b546595`, being fixed by a fresh agent):**
  (1) **Trace×Liquid-Ooze PANIC** — liquidooze/wonderguard admitted to MODELED_ABILITIES but NOT the
  `TRACE_COPYABLE` lockstep → a Porygon2 Trace crashes `--use-bridge=rust`; (2) **White Herb lead-Intimidate**
  — my earlier `onAnyAfterMove` fix (fixture 36) over-corrected + broke the LEAD-switch-in restore (sim fires
  White Herb before `|turn|1`, engine doesn't; repros ab_6_13/ab_7_4). **PRE-EXISTING tail (separate queue):**
  a Liechi-berry +6-cap `-boost|atk|0` emission gap (ab_1_6), Mimic/Encore/Toxic draw-count desyncs
  (ab_1_1/3_21/6_16/6_22), a Mimic first-mover (ab_4_6), + the earlier Soundproof-vs-damaging-sound-moves /
  Toxic-QC / ab_2_19 dmg-calc / ab_5_11 Encore-Mimic. **LESSON: per-mechanic self-fuzz MISSES CROSS-mechanic
  combos (Trace×LiquidOoze, lead-Intimidate×WhiteHerb) — a broad MULTI-SEED fuzz is required post-batch, not
  just the single-mechanic self-fuzz.**
- ✅ **BATCH 8/9 IMPL (part 1) — SHIPPED `b546595` on main (2026-07-23) — STICK + HAZE + LIQUID OOZE + WHITE HERB + WONDER GUARD + YAWN + TRICK DONE (Trick = `gen3_trick_v1`, item swap, choice-lock release, pins TR1-TR5,
  SELF-VALIDATION fuzz 0 Trick divergences, e2e md5 unchanged; remaining batch-8/9 = partial-trap + Transform
  + Forecast[defer]; MORE random-mode tail found @ seed 91919 — a damage-calc STATE edge ab_2_19 + an
  Encore-into-Mimicked-move protocol edge ab_5_11 + seed desyncs, all pre-existing/orthogonal, for the
  mismodeled queue) (all VERIFIED green by
  parent; Wonder Guard = `gen3_wonder_guard_v1` SE-only gate + leech-drain `.max(1)` clamp so the residual KOs
  the 1-HP Shedinja, pins WG1-WG4, e2e md5 unchanged, handler-audit 931 rows; see the CLAUDE.md `gen3_*_v1`
  notes for detail) (2026-07-23, fresh-context agent, UNCOMMITTED on
  top of `722e2fb`).** Stick = the whole CRIT_ITEM class wired (`gen3_crit_item_v1`: Scope Lens / Lucky
  Punch / Stick, data-driven `critBoost` in `gen3_items.json` + `effective_crit_ratio` fold, GIGO-proof);
  Haze (`gen3_haze_v1`, `-clearallboost` both actives, golden `haze_test.rs`); Liquid Ooze
  (`gen3_liquid_ooze_v1`, drain+leechseed heal→damage reversal, removed the fail-loud, golden
  `liquidooze_test.rs`). All DRAW-FREE; revert-verified pins CI1/CI2, HZ1/HZ2, LO1-LO3 + **WHITE HERB** (`gen3_white_herb_v1`,
  BOOST_RESTORE — draw-free negative-boost restore at after-move/switch-in, golden `whiteherb_test.rs` + WH
  pins); full suite GREEN (regression 269). **LESSON: the White-Herb agent ENDED PREMATURELY leaving the tree
  RED — it skipped the handler-audit manifest regen after admitting `whiteherb` to MODELED_ITEMS (8 un-manifested
  handlers → `handler_audit_test` FAILED); parent finished it (`node harness/dump_gen3_handlers.js` → 930 rows +
  full `cargo test` green). Batch agents MUST end FOREGROUND with the audit regen + full suite green, and NOT
  wait on a background process (a subagent doesn't get re-invoked).**
  **e2e md5 UNCHANGED `3155eb796cb4bf453c6053d769ba98e5`** (all 3 have 0 gen3ou-team-carry → 0 e2e decisions,
  proven by the dedicated goldens — the leech-seed situation; they WILL be exercised in random/randbats fuzz).
  Handler-audit 897→922. CAVEAT: Liquid-Ooze×Focus-Band (a FB healer at lethal HP) is correct-by-construction
  on the probe-verified `focus_band_damage` primitive but NOT directly golden-exercised — the fuzz would catch
  a draw desync; close with a pin if it surfaces.
- **REMAINING BATCH 8/9** (later windows, heavier / state-carrying): White Herb (BOOST_RESTORE), Wonder Guard
  (`runEffectiveness>0` gate), Yawn (delayed sleep via `try_set_status`), Trick (item swap), partial-trap
  wrap-family (`random(3,7)` duration draw + firm trap), Transform (largest — copy overlay + revert), Forecast
  (DEFER — owner forme-reporting decision). Then: triage ab_1_5 (dec-48 Fury Attack, Cute-Charm, SEPARATE) +
  9_6/2_12 → re-fuzz all modes (random + randbats + per-side bridge) to byte-clean → the 24h background fuzz.
  Fresh-context, quota-paced, ONE heavy agent/window. Budget ceiling now 70% 5h / 50% weekly; work is
  UNCOMMITTED (needs a future `/gen3ai-ship`).
- **STILL OPEN after the LS fix:** ab_1_5 (dec-48, Fury Attack) = a SEPARATE Cute-Charm-downstream case (not
  investigated); 9_6/2_12 unexamined; batches 8/9 unmodeled (research DONE — see below). Then re-fuzz
  random+randbats to byte-clean → the 24h fuzz.
- ✅ **BATCH 8/9 RESEARCH DONE (2026-07-23, fresh-context agent → `src/rust_sim/harness/BATCH89_RESEARCH.md`
  + 7 `probe_batch89_*.js`).** Probe-settled draw models: Haze (draw-free, clears BOTH actives incl. user),
  Trick (1 acc draw + draw-free swap; Mail+berries DO swap; switcheroo is gen4/not-legal), Yawn (draw-free
  cast, sleep `random(2,6)` at RESOLVE — route through `try_set_status`), partial-trap wrap-family (ONE
  `random(3,7)` duration at cast → 2-5 chip turns, draw-free maxhp/16 chip, firm `trapped:true`), Transform
  (draw-free, largest state job), Wonder Guard (`runEffectiveness>0` gate), Forecast (draw-free,
  `effectiveWeather()`-driven, Cloud-Nine composes — OWNER DECISION on the forme-species reporting surface,
  recommend defer), Liquid Ooze (draw-free reversal, Dream Eater EXCLUDED), White Herb (draw-free), Stick
  (species-gated critRatio+2). Build order: Stick→Haze→WhiteHerb→LiquidOoze→WonderGuard→Yawn→Trick→
  PartialTrap→Transform→Forecast(defer).

**STATE @ handoff (Batch 7 window):** Batch 7 (multihit) DONE + verified + docs (rust_sim + tools
CLAUDE.md) — NOT committed (no /gen3ai-ship). Full cargo test GREEN, e2e md5 unchanged. NOT ready for
the 24h fuzz (random tail above + batches 8/9 unmodeled remain). Next: burn the random-mode tail
(mismodeled hunt) + batches 8 (haze/trick/yawn/wrap) + 9 (transform/wonderguard/forecast/liquidooze/
whiteherb/stick), re-fuzz to byte-clean, THEN kick the 24h random-battle fuzz. Relates to
[[project_rust_sim_port]], [[project_rust_bridge_incremental]], [[feedback_edge_case_regression_tests]].
