# Hidden-info study: the independent replication of the fair-world instrument (2026-10-11 / 12)

## In plain language

The hidden-information study (`designs/endstate/design_hidden_info_binding.md`) asks how much better a search
would decide if it knew the opponent's hidden facts (its unseen moves, its never-seen team members, items,
abilities, stat spreads). Its yardstick is **IMP**: how much the search's pick beats the policy's own pick, in
win probability per decision, judged by a referee that played every action out 24 times in the TRUE battle.
The search plays its rollouts either in the true battle (**O**, the oracle) or in **fair worlds**: copies of the
battle where every fact the searched side has not seen is re-drawn from the Smogon usage prior, kept only if
replaying the game so far shows the searched side exactly what it really saw.

The owner's worry was that a bug in the fair-world builder would quietly BE the result. So this folder is a
SECOND builder, written by a different agent from the spec text alone (§13.1), without looking at the first
builder, its scripts or any of its results. Its numbers go to a third party, who compares the two. This folder
does NOT compare them.

**What we found while building it (the short version):** see FINDINGS below. Finding 1 there was found at the
finish, and the third party should read it first: the learnset table both builders can reach does not carry
pre-evolution moves, so this builder's worlds never draw, for example, Tyranitar's Dragon Dance. The most important for the third
party: (1) reusing the true battle's dice seed lets the turn-0 gender dice leak which never-seen species the
opponent has, and we fixed that by aligning the seed; (2) the same seed reuse still lets any hidden fact that
spends dice without printing a line (Focus Band, King's Rock, a speed tie) be "seen" by the gate; (3) the pool
teams use Hidden Power IV patterns and spreads the Smogon top-25 spread table does not hold, so a faithful
builder needs the full spread table and every max-power IV pattern, or it cannot rebuild about half the truths.

## FINAL RESULTS (all 600 + 600 registered states; read 2026-10-13 ~14:10; `result.json`, `imp_per_state.json`)

**COMPUTED, not a verdict.** §12.2 R-REPLICATE (the comparison with the first implementation) belongs to a third
party; until it reads, nothing below is more than "computed". **This folder does not compare these numbers with the
first implementation's.**

The read is the registered one: LEAF-K8, IMP on the win scale, `replica/score.py` UNCHANGED from the interim read.
CIs are battle-clustered percentile bootstraps (2,000 draws, `numpy.random.default_rng(20261011)`), every contrast
paired on the same draws, the pooled read resampling battles within each checkpoint. The common set holds the
states where all five arms built 8 worlds: **end2 545 states / 384 battles, prod 544 / 383.**

| read | states (end2 / prod) | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M |
|---|---|---|---|---|---|---|
| end2 | 545 | +0.0795 [+0.0653, +0.0953] | +0.0219 [+0.0088, +0.0343] | +0.0134 [+0.0003, +0.0274] | +0.0244 [+0.0106, +0.0385] | +0.0425 [+0.0274, +0.0578] |
| prod | 544 | +0.0741 [+0.0583, +0.0914] | -0.0012 [-0.0162, +0.0145] | +0.0000 [-0.0152, +0.0155] | +0.0020 [-0.0129, +0.0187] | +0.0363 [+0.0208, +0.0529] |
| pooled | 545 / 544 | +0.0768 [+0.0662, +0.0881] | +0.0103 [+0.0007, +0.0197] | +0.0067 [-0.0032, +0.0163] | +0.0132 [+0.0026, +0.0237] | +0.0394 [+0.0286, +0.0508] |

| read | G = O - F | G_P = O - P | F+M - F | O - (O-M) | s_M | n_M | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|---|
| end2 | +0.0661 [+0.0503, +0.0827] | +0.0576 [+0.0442, +0.0722] | +0.0109 [-0.0040, +0.0263] | +0.0370 [+0.0246, +0.0507] | +0.17 [-0.07, +0.36] | +0.56 [+0.39, +0.76] | yes | no | yes |
| prod | +0.0740 [+0.0586, +0.0905] | +0.0753 [+0.0613, +0.0910] | +0.0020 [-0.0131, +0.0175] | +0.0377 [+0.0268, +0.0496] | +0.03 [-0.20, +0.22] | +0.51 [+0.37, +0.68] | yes | no | yes |
| pooled | +0.0701 [+0.0591, +0.0811] | +0.0664 [+0.0566, +0.0762] | +0.0065 [-0.0039, +0.0166] | +0.0373 [+0.0290, +0.0463] | +0.09 [-0.06, +0.23] | +0.53 [+0.42, +0.65] | yes | no | yes |

s_M / n_M to three places (`result.json`): end2 s_M +0.165 [-0.072, +0.358], n_M +0.559 [+0.393, +0.764]; prod
s_M +0.027 [-0.196, +0.224], n_M +0.510 [+0.373, +0.679]; pooled s_M +0.092 [-0.060, +0.227], n_M +0.533
[+0.419, +0.653]. **No quantity sits within ±0.0005 of a §12.3 boundary** (`boundary_flags` empty on every read).

What the B rules say here, read mechanically and only for what this replication ran:
- **B2** (G lower > 0.02): holds on end2, prod and pooled.
- **B3** (s_M point ≥ 0.50, lower ≥ 0.25): fails on every read (pooled point +0.092).
- **B4** (n_M point ≥ 0.50, lower ≥ 0.25): holds on every read (pooled +0.533, lower +0.419; prod's point
  +0.510 is 0.0096 above the 0.50 line, outside the ±0.0005 boundary band).
- **B8's two halves, per checkpoint:** IMP_F+M − IMP_F lower > +0.0005 FAILS on both (end2 −0.0040, prod
  −0.0131); IMP_O − IMP_O-M lower > +0.0005 holds on both.
- B1 needs R-GIGO and R-REPLICATE, B5 the rival classes U / I / A / S, B6 / B7 the dose and placebo arms: none of
  those were in this replication's scope (§13.1 items 5-7), so no §12.3 verdict is issued here. Descriptively, the
  pooled pattern (n_M lower +0.419 ≥ 0.25, s_M upper +0.227 < 0.25) is the shape §12.3 names
  MOVES-NECESSARY-NOT-SUFFICIENT; end2 alone does not have it (s_M upper +0.358).

**Own-set IMP** (each arm on all its own ok states, LEAF-K8; `result.json` `own_set`): end2 O +0.0806 (600),
P +0.0279 (598), F +0.0141 (559), F+M +0.0217 (563), O-M +0.0454 (599); prod O +0.0714 (600), P −0.0036 (600),
F −0.0019 (564), F+M +0.0029 (567), O-M +0.0340 (600).

**Other scorers from the same playouts** (common set; reported, not registered):

| scorer | read | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M | G |
|---|---|---|---|---|---|---|---|
| LEAF-K1 | end2 | +0.0325 [+0.0217, +0.0439] | +0.0075 [-0.0017, +0.0166] | +0.0035 [-0.0053, +0.0122] | -0.0022 [-0.0110, +0.0064] | +0.0150 [+0.0050, +0.0253] | +0.0290 [+0.0167, +0.0409] |
| LEAF-K1 | prod | +0.0285 [+0.0148, +0.0433] | +0.0012 [-0.0081, +0.0105] | -0.0044 [-0.0142, +0.0051] | -0.0024 [-0.0131, +0.0091] | +0.0078 [-0.0026, +0.0192] | +0.0329 [+0.0184, +0.0489] |
| LEAF-K1 | pooled | +0.0305 [+0.0215, +0.0398] | +0.0044 [-0.0021, +0.0110] | -0.0004 [-0.0070, +0.0062] | -0.0023 [-0.0096, +0.0047] | +0.0114 [+0.0039, +0.0188] | +0.0310 [+0.0214, +0.0411] |
| LEAF-K4 | end2 | +0.0677 [+0.0532, +0.0842] | +0.0119 [-0.0000, +0.0236] | +0.0062 [-0.0064, +0.0200] | +0.0170 [+0.0053, +0.0293] | +0.0304 [+0.0153, +0.0463] | +0.0614 [+0.0453, +0.0782] |
| LEAF-K4 | prod | +0.0613 [+0.0467, +0.0774] | +0.0080 [-0.0046, +0.0206] | -0.0073 [-0.0212, +0.0072] | +0.0079 [-0.0057, +0.0223] | +0.0229 [+0.0093, +0.0363] | +0.0686 [+0.0531, +0.0850] |
| LEAF-K4 | pooled | +0.0645 [+0.0542, +0.0758] | +0.0099 [+0.0014, +0.0185] | -0.0005 [-0.0098, +0.0091] | +0.0124 [+0.0030, +0.0222] | +0.0267 [+0.0163, +0.0375] | +0.0650 [+0.0543, +0.0765] |

**Exclusions** (states without 8 accepted worlds; every arm under §6.1's 10 % line):

| cp | O | P | F | F+M | O-M |
|---|---|---|---|---|---|
| end2 (of 600) | 0 | 2 (`only_0_worlds`) | 41 (6.8 %, `no_consistent_context`) | 37 (6.2 %, same) | 1 (`only_0_worlds`) |
| prod (of 600) | 0 | 0 | 36 (6.0 %, same) | 33 (5.5 %, same) | 0 |

The builder is seeded per (arm, state) (`builder.rng_for`), so re-running an excluded state reproduces its failure;
the three P / O-M states were not re-run again (P's two had been re-run once already). The common set's 55 / 56
lost states are mostly F / F+M screen failures (a state with no consistent world under the prior), which is a
selection the third party should hold fixed when comparing.

**AUDITS** (`audit_end2.json`, `audit_prod.json`; every registered state, `--rows`; run 2026-10-13 14:09 on the
code that built the rows):

| audit | end2 | prod |
|---|---|---|
| ID: the TRUE opponent team through this builder passes the gate (lines AND rows) | **599 / 600** (1 FAIL, below) | 600 / 600 |
| ID: the rendering is the identity; the playout tool replays it | 599 / 600 | 600 / 600 |
| T1 a revealed item swapped: caught | 542 / 542 | 541 / 541 |
| T2 a past switch to the wrong slot: caught | 537 / 537 | 532 / 532 |
| T3 a used, seen move removed: caught | 591 / 591 | 592 / 592 |
| T4 the aligned seed one draw off: caught | 591 / 600 (9 not) | 580 / 600 (20 not) |
| PARSE: a fact called revealed that the true set lacks | 0 of 3,120 moves, 1,539 items, 444 abilities, 2,302 details | 0 of 2,931 / 1,486 / 415 / 2,201 |
| O vs Phase 1, dice seeds 0-7 (same battle, seeds, continuation) | 4,478 / 4,478 outcomes agree | 4,436 / 4,436 agree |
| pending opponent root choices dropped (A3) | 7 | 1 |

- **The ID failure** (`rb_es_end2_s2001/step_14000347/staller/loss_s3_004#p1#8`) is a RENDERER defect: the true
  log's opponent choice 11 is the NUMERIC `move 1` (the Recharge turn after Hyper Beam at choice 9), and
  `builder.render` rewrites it by name to `move dragondance`, which the engine refuses on a recharge turn. The
  Struggle / Recharge guard covers only the NAMED forms. Every constructed arm (P, F, F+M, O-M) already failed on
  this state, so it is outside the common set and moves no number; no other registered state has a numeric
  opponent move choice (the identity rendering is unchanged on the other 1,199). The interim README's "the
  identity audit passes on every state" is therefore 1,199 / 1,200 on the final audit.
- **The T4 misses are states where the +1 offset does not change the searched side's view.** Re-run per missed
  state at offsets +2, +3, +5, +9, +17, +101: 19 of the 29 (end2 7, prod 12) pass the gate at EVERY offset (the
  prefix shows no dice: mostly the first one or two decisions), and the other 10 pass at +1 but fail at some other
  offset (the view depends on the seed, coarsely enough that +1 coincides). The gate compares the searched side's
  bytes, so a pass means the planted defect was not observable on that state; strictly, though, R-GIGO's "each
  audit FAILS on its planted defect" is not met on those 29 states.

**Marginals** for §12.2's total-variation check: `marginals.json` (per checkpoint and arm, the class U / M / I / A /
S frequency tables over every included state's 8 worlds). **Deliverable rows** (§13.1 item 7: per state and arm the
8 world teams, every action's 8 outcomes, the build bookkeeping): `rows/<cp>_<arm>.jsonl.gz` (`replica.pack`); the
full rows stay in `~/gen3ai_archive/hidden_info_replica/rows/`.

## FINDINGS (for the third party and the red team, §13.2)

1. **The pinned learnset table drops every move a species inherits from its pre-evolution** (egg moves and
   pre-evolution level-up moves): `agents.gen3_data.learnset.is_legal` says False for Breloom Spore, Tyranitar
   Dragon Dance, Jolteon Baton Pass, Cloyster Rapid Spin, Swampert Hydro Pump, and True for Shroomish / Larvitar /
   Eevee / Shellder / Mudkip (checked at the end2 pin AND at current main). This replication's move prior
   (`sampling.move_ok`) filters on it, so its worlds can NEVER hold those moves although the Smogon prior gives
   them heavy mass (Breloom Spore 0.98, Umbreon Wish 0.89, Jolteon Baton Pass 0.79, Cloyster Rapid Spin 0.75,
   Swampert Hydro Pump 0.53, Tyranitar Dragon Dance 0.42). The TRUE opponent team holds at least one such move in
   **560 / 600 end2 states and 561 / 600 prod states** (41 / 44 distinct species-move pairs; Swampert Hydro Pump
   alone in 255 / 260 states, Tyranitar Dragon Dance in 203 / 207). So every arm that RE-DRAWS moves (P, F, O-M,
   and F / F+M's never-seen slots) draws from a prior missing real support, and an unrevealed Dragon Dance on a
   Tyranitar is never in a world. This bears directly on the M contrasts (s_M, n_M). Whether the first
   implementation shares it is for the third party. **Outside this unit, and possibly larger:**
   `agents/model/belief_tables.py` reads the same table (`get_legal_moves`) to give "illegal" moves about zero
   prior probability, so the production model's move-belief prior may hold the same gap. Not verified here.
2. **Renderer defect, one state** (the ID audit's only failure, above): a numeric `move N` opponent choice is
   rewritten by name. Excluded from every constructed arm, so it moves no number.
3. **T4 (seed one draw off) does not fail on 29 / 1,200 states** (above): 19 have a prefix that shows no dice; 10
   coincide at +1 only.
4. **Rows span two builder revisions.** P (both checkpoints, started 03:33-03:34 on 2026-10-12) was built by the
   code of `9bf13b3e`. Every other constructed arm (O-M from 06:53, F / F+M from 09:00) was built after
   `88e9b8df` (05:38), which (a) keeps a NAMED forced Struggle / Recharge unrendered and (b) rejects at build time
   a world the playout tool refuses. Neither change can alter an accepted P world: under the old code (a) makes
   every world of an affected state fail the gate, and (b) makes the state's playout raise. Both would show up as
   a failed P row, and P failed on only 2 states, both re-run under the new code with the same result. Rows carry
   a schema tag (`hir_replica_rows_v1`) but no code commit. This was argued from the diff and the row times; it
   was not re-executed.
5. Carried from the build (see the top and A1 / A2 / A7 / A8): the gender-dice leak (fixed by seed alignment), the
   silent-dice leak (NOT fixed; inherent to replaying on the true seed), and the spread / IV-pattern coverage
   (fixed with the full spread table and every max-power IV pattern).
6. The common set drops 55 (end2) / 56 (prod) states, mostly F / F+M `no_consistent_context` failures: the read is
   conditional on states for which the screen found a consistent world, and any comparison should hold that
   selection fixed.

## INTERIM REGISTRATION (2026-10-12 ~10:00, before any F / F+M outcome was read)

The CPU allowance was cut to 3 workers (a GPU run took the box at 09:29), so the full 600 + 600 states will take
about a day. On the orchestrator's instruction an INTERIM subset is registered here, and its numbers go out first:

- **Subset:** per checkpoint, the first 40 states of each of the pilot's five gap quintiles, in the pilot's own
  registered sampling order (`sha256("value_disc_pilot_2026:" + decision_id)`, `design_value_discrimination.md`
  §7.2): 200 states per checkpoint, listed in `interim_ids_end2.txt` / `interim_ids_prod.txt`
  (`replica/interim.py`). The states file lists states sorted by id, so a plain file-order prefix would hold one
  cycle and a few opponents only; hence the stratified hash order.
- **Read:** `replica/score.py` unchanged, run on these ids for every arm (O, P, F, F+M, O-M), and its
  states-included rule unchanged (states where all five arms built 8 worlds).
- **What had been seen when this was written:** O rows (the 8,914 / 8,914 match with Phase 1's seeds 0-7, no IMP
  computed); P / O-M row COUNTS and failure reasons; F / F+M row counts and failure reasons only (no outcome,
  no IMP, no contrast). The scorer was run once at ~23:30 on partial O rows only (own-set IMP of O; no other arm).
- The final read uses all 600 states per checkpoint when they complete.

## INTERIM RESULTS (the registered subset, read 2026-10-12 17:10; `interim/result.json`, `interim/tables.md`)

The read is LEAF-K8, with IMP on the win scale. CIs are battle-clustered bootstraps (2,000 draws, rng 20261011), with
every contrast paired on the same draws. The common set holds the states where all five arms built 8 worlds:
end2 180 states / 159 battles, prod 181 / 163. **This replication does not compare these numbers with the first
implementation's; a third party does.**

| read | states | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M |
|---|---|---|---|---|---|---|
| end2 | 180 | +0.0569 [+0.0335, +0.0810] | +0.0037 [-0.0178, +0.0264] | +0.0147 [-0.0052, +0.0360] | +0.0116 [-0.0152, +0.0361] | +0.0237 [+0.0013, +0.0461] |
| prod | 181 | +0.0642 [+0.0385, +0.0922] | -0.0218 [-0.0462, +0.0019] | -0.0152 [-0.0387, +0.0088] | +0.0132 [-0.0092, +0.0378] | +0.0363 [+0.0111, +0.0621] |
| pooled | 180 / 181 | +0.0606 [+0.0430, +0.0787] | -0.0091 [-0.0258, +0.0081] | -0.0003 [-0.0159, +0.0157] | +0.0124 [-0.0045, +0.0298] | +0.0300 [+0.0124, +0.0476] |

| read | G = O - F | G_P = O - P | F+M - F | O - (O-M) | s_M | n_M | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|---|
| end2 | +0.0422 [+0.0182, +0.0659] | +0.0532 [+0.0285, +0.0783] | -0.0031 [-0.0252, +0.0180] | +0.0332 [+0.0113, +0.0566] | not computed (B2) | not computed (B2) | no | no | no |
| prod | +0.0794 [+0.0503, +0.1106] | +0.0860 [+0.0592, +0.1110] | +0.0284 [+0.0047, +0.0544] | +0.0280 [+0.0113, +0.0447] | +0.358 [+0.065, +0.650] | +0.352 [+0.153, +0.630] | yes | no | no |
| pooled | +0.0609 [+0.0422, +0.0797] | +0.0697 [+0.0517, +0.0881] | +0.0127 [-0.0036, +0.0295] | +0.0306 [+0.0170, +0.0453] | +0.209 [-0.067, +0.457] | +0.502 [+0.290, +0.758] | yes | no | yes (point 0.5024, only 0.0019 above the 0.5005 boundary) |

LEAF-K1 / K4 and the own-set IMP of every arm are in `interim/tables.md` and `interim/result.json`.
**Exclusions** (no consistent F world found by the screen, all with reason `no_consistent_context`): end2 F 17 / 200,
end2 F+M 11, prod F 13, prod F+M 12. O, P and O-M lost no interim state. Every arm is under §6.1's 10 % line.

## STATUS AND HANDOFF (2026-10-12 ~20:05; the agent stopped under the quota rule; the workers keep running)

> **DONE 2026-10-13** (a second agent finished the unit). The unattended workers all exited 0, none OOM-killed
> (last: end2 O-M 02:41, end2 F 04:00). `replica.remaining` lists 0 remaining states for every (cp, arm); the 600 +
> 600 rows per arm hold no state with two ok rows. Then `replica.audit` (both checkpoints, `--rows`),
> `replica.score`, `replica.report`, `replica.marginals` and `replica.pack` ran with the code UNCHANGED; their
> outputs are FINAL RESULTS above. The text below is the handoff as written.

The full read (600 + 600) is IN PROGRESS. Three unattended, resumable workers run on cores 5-7 / 13-15 at nice 19,
each under `mem_cap`, and write rows to `~/gen3ai_archive/hidden_info_replica/rows/<cp>/<arm>/*.jsonl`:

| worker | now | then |
|---|---|---|
| W1 | prod F (rest) | end2 F, half A of the non-interim states (`interimRestA_ids_end2.txt`) |
| W2 | end2 F+M (rest) | end2 O-M (rest) |
| W3 | prod F+M (rest) | end2 F, half B (`interimRestB_ids_end2.txt`) |

Done at 20:00: O 600 + 600; P 600 + 600 (end2: 2 failed states remain after one re-run); O-M prod 600, end2 316;
F end2 257, prod 346; F+M end2 387, prod 430. At about 1.3 states a minute in all, roughly 15 h remain.
**To finish:**
1. When every arm holds 600 rows, run `replica.remaining` per (cp, arm). Any listed state is unfinished: run it with
   `seq_remaining.sh <new tag> 5-7,13-15 <cp>:<arm>`.
2. Run `replica.audit` per checkpoint with `--rows` (identity + playout replay, teeth, parser, harness vs Phase 1).
3. Run `replica.score` (all states), `replica.report`, `replica.marginals` and `replica.pack --out <this folder>/rows`.
4. Put the final tables here beside the interim ones. Do NOT compare with the first implementation.
A state run twice keeps its ok row (`score.load_rows`); two ok rows for one state is an error by design.

## Blindness (what this agent read and did not read)

- **Read:** `design_hidden_info_binding.md` §1.1, §1.2, §4, §5 (the audit definitions), §6.1-§6.3, §12.1-§12.3,
  §12.10, §12.12 (Amendment 1), §13, §15; `design_value_discrimination.md` §2.1, §5.1, §7.1-§7.4 (the pilot's
  definitions: the referee, the seeds, the checkpoint, the continuation). The registered states
  (`value_disc_pilot_2026-10-11/states_<cp>.json`), the banks (`~/gen3ai_archive/value_disc_pilot/<cp>/bank`) and
  the Phase 1 truth rows (`.../<cp>/rows/`, used only for the referee's outcome strings and, as an audit of this
  harness, their seeds 0-7). General infrastructure at the pins: `utils.rust_env.successors.play_out`,
  `agents.battle.core_replay.run_core`, `main.policy_spectrum.{bank,reader,replay,truth}` (the truth tool's
  `turn_seeds`, `to_log`, `cmd_index`, `MicroBatcher`), `agents.gen3_data.*`, the Rust engine source (gender
  construction, the gen-5 LCG, the move panics) and its `scan_move_probe` coverage oracle.
- **Not read:** `src/main/value_search/`, `src/main/hidden_info_study/`, `determinize.py`, every script and
  result file in `value_disc_pilot_2026-10-11/` and `hidden_info_binding_2026-10-11/` other than the two states
  files, the ledger entries about them, `design_hidden_info_binding.md` §0 and §16 (results), the pilot's §7.10-
  §7.13 (amendments and results), the pilot's `det*` / `qrows` archives, and `~/.cache/gen3ai/vdp/support_*`
  (the first study's engine probe; this folder ran its own: `engine_probe_*.jsonl`).
- The pin checkouts `~/.cache/gen3ai/vdp/pin_{end2,prod}` were used as built, after a byte check that every
  tracked file under `src/` and `data/` equals the pin commit.
- **The finishing agent (2026-10-13)** read: this README, `replica/` (score, audit, report, pack, remaining,
  marginals' header, fair's seeding), the logs and rows under `~/gen3ai_archive/hidden_info_replica/`, and
  `design_hidden_info_binding.md` §12.1-§12.3 and §13.1. It also saw the doc's section HEADINGS once (a heading
  list; §16's headings carry file names but no numbers) and read none of §16's text. It read nothing in the
  forbidden list (`src/main/value_search/`, `src/main/hidden_info_study/`, `src/agents/info_levers/`, the
  measurement folders `value_disc_pilot_2026-10-11/` other than `states_<cp>.json` through the scorer,
  `hidden_info_binding_2026-10-11/`, `hidden_info_by_reveal_count_2026-10-12/`, `info_levers_2026-10-12/`, the
  results sections of `design_hidden_info_binding.md` / `design_value_discrimination.md` / `search_ledger.md`, the
  related ledger entries) and compared nothing with the first implementation.

## What was built (`replica/`)

| file | what it does |
|---|---|
| `env.py` | puts the pin first on the path, names its binaries, loads the registered states |
| `view.py` | reads the SEARCHED side's own protocol lines up to the root: the opponent mons seen, every move a public line names for them, items and abilities named by `-item` / `-enditem` / `-ability` or a `[from] item:` / `[from] ability:` tag (owner resolution written out) |
| `sampling.py` | the Smogon prior filtered to what the pinned engine runs (own probe) and the pinned gen3ou rules allow; conditional-Poisson move draws; the never-seen species draw; IV patterns; legality |
| `builder.py` | per state: the truth replay, the pending opponent choice, seed alignment, world assembly per arm, rendering of past opponent choices, the GATE (searched side's lines AND rows byte-equal) |
| `screen.py` | the F / F+M screen of the seen mons' item, ability, spread, IVs and Hidden Power |
| `fair.py` | the 8 accepted worlds per state and arm (≤ 60 draws) |
| `play.py` | the playouts (every legal action x 8 worlds, world j on dice seed j; O: the true battle on seeds 0-7) |
| `score.py` | the referee, IMP per arm and scorer, the contrasts, §12.3's B2-B4 for M |
| `audit.py` | builder identity, the planted-defect teeth, the parser against the truth, the harness against Phase 1 |
| `marginals.py` | per-class marginals of the drawn worlds (for §12.2's TV check) |

## Spec ambiguities, and the choice made for each (most literal reading; none peeked at the first builder)

| # | where the spec is silent or self-overlapping | this replication's choice |
|---|---|---|
| A1 | worlds reuse the TRUE battle's dice seed (needed for the prefix to replay), but the engine's turn-0 construction draws one die per blank-gender, gendered-species mon (p1's team, then p2's): a world whose never-seen species are genderless where the truth's are not shifts EVERY later die, so the gate would accept only worlds whose never-seen species match the truth's gender pattern (a leak about U) | every world plays with EXPLICIT genders on both teams (ours: as realized; a seen opponent mon: as shown; a never-seen one: M or F with probability 1/2 each, independent of the species, blank for a fixed-gender or genderless species) and with the raw seed advanced past the truth's gender draws (`builder.lcg_advance`, the gen-5 LCG of `rust_sim/src/prng/gen5.rs`). The identity audit passes on every state with it, and fails when the seed is one draw off (tooth T4) |
| A2 | the same seed reuse lets ANY hidden fact that spends dice without printing a line (Focus Band's roll on every hit to its holder, King's Rock, a speed tie's shuffles) change later dice, so the gate "sees" it | NOT fixed (it is inherent to replaying with the true seed; the first builder shares it). Reported as a finding |
| A3 | the opponent's root-turn choice, when the log recorded it before the searched side's (it is pending, not yet executed, so not seen) | DROPPED in every constructed world: the world's greedy opponent answers its root decision, identically across the searched side's actions. Kept in O (the referee's convention). Phase 1's rows say 7 end2 states and 1 prod state are of this kind |
| A4 | slot positions (team order, move-slot order) are not a class of §6.1 | kept from the truth: a seen mon in its true slot, a revealed move in its true move slot; the drawn moves fill the other slots |
| A5 | "past opponent choices that name a re-drawn move or a never-seen species must be rendered" | a `move X` whose X is not in the world's set (a choice that never executed, so never shown) becomes the world's move in X's TRUE slot; a typed Hidden Power becomes the world's Hidden Power; a `switch S` to a re-drawn slot becomes the world's mon in that slot |
| A6 | "species ∝ the teammate prior given the revealed species" with several revealed species | the MEAN over the revealed species of the Smogon P(teammate \| revealed), each never-seen slot in turn, excluding a species whose base is already on the team (Species Clause), banned species, and species with no move / item / ability prior |
| A7 | "every unrevealed fact re-drawn from the Smogon prior": the spread prior | `priors.all_spreads` (every chaos Spreads entry). The top-25 file `priors.spreads` holds only ≈ 49 % of the bank's opponent spreads (2,435 of 4,998), so with it about half the truths could never be rebuilt |
| A8 | Smogon publishes no IVs | the attack IV is "max" or "min" (0, or the lowest value keeping a Hidden Power's two low bits) with prior 1/2 each; with a Hidden Power of type t, every max-power pattern (each IV 30 or 31) whose type is t, uniformly. The pool uses non-canonical patterns (e.g. Hidden Power Ice as Spe 30 alone) and the gate is exact to one HP point |
| A9 | §6.1's classes OVERLAP in gen 3: a Hidden Power's type (M) is a function of the IVs (S) | where S is TRUE (P, O-M) the IVs are kept, so a revealed bare Hidden Power has its true type and an M draw may add a Hidden Power only of the IVs' type; in F an UNREVEALED Hidden Power's presence and type are screened together with S (prior: its conditional-Poisson inclusion probability); in F+M the true moves carry the true type and the IV pattern is drawn among that type's patterns |
| A10 | "each candidate in the prior's support is screened by a replay with the other mons at their F draws" | the support is thousands of spreads per species, so the screen is: a context search (one consistent world, found by replays), per-mon windows over the support at that context, then per-mon REJECTION sampling of joint (spread, IV key, item, ability, Hidden Power) prior draws with the other mons at the context; ≤ 192 draws per mon, else the context alone (flagged `context_only`). Worlds combine the mons' accepted draws and pass the full gate (≤ 60 tries) |
| A11 | which line types "name" a move / item / ability for a mon | written out in `view.py`: `move` (not a `[from]` bounced / called move, not Struggle / Recharge), `cant` with a move, `-singleturn`, `-singlemove`, Spite's `-activate`, Disable's `-start`; items: `-item` (Thief / Trick handled), `-enditem`, `[from] item:`, a positional `item:`; abilities: `-ability` (Trace handled), `[from] ability:` with the owner resolved by who can hold it |
| A12 | legality of a world team | the pinned format's rules (`team_legality`, ladder-only rules excluded), ≤ 1 Hidden Power, learnset, Hidden Power type = the IVs' type, engine support (own `scan_move_probe` run). A problem the TRUE team also has (our learnset table rejects some pool moves, e.g. Dragon Dance Tyranitar) is excused |
| A13 | the scorers | LEAF-K8 as registered (LEAF-K1 / K4 reported from the same playouts). The critic and TRUNC scorers are not run: §13.1 names only LEAF-K8 |
| A14 | the referee's tie / truncation | an outcome '0' in the Phase 1 strings is 1/2 |
| A15 | "F+M (F with the true seen moves)" | the TRUE full movesets of the seen mons (class M revealed), everything else as F |
| A16 | public details beyond the species | a seen mon's shown gender, shiny flag and level are carried into every world (a shiny seen mon otherwise fails every world's gate) |
| A17 | §12.3 reads R-MOVES "on POOLED shares over the three checkpoints", but §13.1 hands over registered states for two (end2, prod) | the pooled read pools those two, battles resampled within each checkpoint, states weighted equally |
| A18 | §12.3's "≥ 0.50" / "lower ≥ 0.25" under rule 8's boundary band | a threshold counts as met only when clear of the ±0.0005 band (`x ≥ thr + 0.0005`); none of this read's quantities fall in a band |
| A19 | the common set when an arm lost a state | every contrast on the states where ALL five arms built 8 worlds (so a contrast never mixes state sets); each arm's own-set IMP reported beside it |
| A20 | a state whose world build failed deterministically | excluded with its reason (§6.1's 10 % line per arm), never re-drawn with a new seed; the builder's seed is a function of (arm, state) |

## Reproduce

```bash
D=designs/research_state/measurements/hidden_info_replica_2026-10-11
$D/run_chain.sh <cp> <shard> <nshards> <cpus> O P F F+M O-M        # rows -> ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.audit --cp <cp> --out $D --rows ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.score --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
PYTHONPATH=$D python -m replica.marginals --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
python $D/replica/report.py $D/result.json                                  # the markdown tables above
PYTHONPATH=$D python -m replica.pack --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D/rows
```
