# TIER 1 of the hidden-information binding study (2026-10-13)

The DECISIVE, pre-registered experiment of [`design_hidden_info_binding.md`](../../../endstate/design_hidden_info_binding.md)
(§6, §12, Amendment 1 §12.12, Amendment 2 §12.13, §15). **Status: the scripts and the sealed arm key are committed
BEFORE any arm output exists (§12.10); the results sections below are filled when read.** The blind replica's code,
results and archive were not read (owner's rule; a third party compares).

Bulk: `~/gen3ai_archive/hidden_info_tier1/<cp>/` (worlds, rows, banks, the sealed key). This folder: the scripts, the
committed ids, the blinded summary, the result.

## 1. What runs (the registered Tier 1 + Tier-1S, and nothing else)

| arm | what | states |
|---|---|---|
| O | the truth (Phase 1): the pilot's rows on the registered 600; NEW rows on the supplement | all |
| F | the fully honest world (`off`) | all |
| F+M / O−M | the seen mons' moves told / hidden | all |
| F+U / O−U | the never-seen mons told / hidden (promoted by Amendment 2) | all |
| FN | F with species and move draws from the NETWORK's own belief (the build item of §12.13.3) | all |
| D50, D90 | the move dose `moves:seen@p` | registered 600 |
| W90 | the wrong-moves placebo at 0.9 | registered 600 |
| MAP100 | the honest sharpening (the prior's own top moves) | registered 600 |
| POOLSETS | F with the TRAINING POOL's per-species set distribution | registered 600 |
| POOL | worlds = pool teams consistent with the observation | registered 600 |
| DK family | M8 / M5 / M5W / M5X re-weightings of O and F (§6.9): no playout | all |
| HE@O, HE@P, HE@F | Foul Play's hand evaluator (§6.5) in the true, the pilot's v2 (`det2`) and the F worlds: truncated re-plays | registered 600 |

States: the pilot's registered 600 per checkpoint (`value_disc_pilot_2026-10-11/states_<cp>.json`) + the k-stratified
SUPPLEMENT (§12.13.2; `supplement_<cp>.json`, `states_supp_<cp>.json`: end2 +200, prod +220; every k bin 160 per
checkpoint, prod k = 1–2 180). Checkpoints `rb_es_end2_s2001` (pin `62085750`), `rb_es_prod_s2001` (pin `95d014fa`).

**NOT run (Tier 2 / Tier 3 / third party; each a named consequence in §6):** F+I, F+A, F+S, O−I, O−A, O−S, F+Usp; D25,
D75, W50, W100, MAP90; P16 (R-WIDTH); the second referee (H-REFEREE); `rb_es_end2_s2002`; the trajectory arm (M-STABLE,
the V8 guard); the away bank; R-REPLICATE (the independent implementation, compared by a third party).

## 2. Where each step runs, and why that is sound

- **Worlds are BUILT at this study's HEAD** (`build_worlds.py` via `run_head.sh`): the information-lever builder
  `main.value_search.worlds` (honest by default; `182d7c57` and later), whose gate replays at HEAD's engine and uses the
  engine's GENDER DRAW PAD (`98146ddb`).
- **Everything is PLAYED at the checkpoint's own pin** (`run_pin.sh`): the truth, the census, the supplement bank and the
  network-belief dump at the UNMODIFIED pin checkouts; every WORLD at a **PAD PIN** = the pin + the code of `98146ddb`
  (cherry-picked, docs dropped: end2 `67f3f7fa`, prod `9a698801`; `~/.cache/gen3ai/hib_t1/pin_<cp>_pad`). Between the pins
  and HEAD the engine source differs only by the pad, comments, and the oracle-reveal encoder (byte-inert at `off`):
  `git diff 67f3f7fa HEAD -- src/rust_sim/src src/rust_env/src` touches `encoder/oracle.rs`, `encoder/slot.rs`,
  `core/pool.rs`, `core/spec.rs` (oracle reveal) and comment-only lines in `lib.rs`, `search.rs`, `turn/forecast.rs`.
- **Validity checks of that split (no arm outcome):**
  - `identity_read.py` — `oracle` worlds through the HEAD builder, stored in the det format, played at the pad pin, must
    reproduce the pilot's Phase 1 rows (outcomes exact; values within one rounding unit): **PASS** on 12 + 12 states
    (`identity_V_oracle.json`: 0 outcome mismatches over 88 / 93 actions; max |ΔV| 1e-5, |Δq| 1e-6).
  - `pin_regate.py` — every sampled world of an arm passes the strict gate again at the PAD PIN's own `core_events`
    (rows + protocol byte-equal to the true prefix), with a planted seen-mon item swap that must fail and the pad marker
    stripped (descriptive: dice move).
  - `world_audit.py` — R-GIGO for the Tier-1 worlds: the information-lever audit's own provenance, leak and dose checks
    and the legality re-check over EVERY stored world, plus the two reads above.
  - k = 6 identities: F+U ≡ F and O−U ≡ O (seeds 0–7) where no mon is unseen (`unblind_t1.validity`).

## 3. Blinding (§12.10) and its limits

Arm outputs carry an opaque code `Tnn` = the rank of `sha256("hib_arm_v1:" + label)` among the Tier-1 labels
(`t1_common.arm_codes`). The key lives in the archive until every arm output exists; its sha256 is committed now
(`arm_key.sha256`). `score_blind.py` reads arms by code only and freezes the blinded table (`blinded_table.sha256`,
`result_blinded.json`) before `unblind_t1.py` (committed now, with every rule as code) maps codes to labels.
**Disclosure (as §16.5):** the code scheme is public and the generation harness knows each arm's label, so the author
knows the codes; the blinding protects the frozen scripts, rules and table, not the author's knowledge.

## 4. The prior builds (README of the build: the sub-agent's report, summarised when read)

- **FN** (`fn_belief.py` at the pin: the pilot's Phase 2c belief read, one CPU forward at the root; `fn_worlds.py` at
  HEAD): never-seen species ∝ the network's X5 species presence; seen mons' unrevealed moves from the network's per-mon
  move presence; a never-seen mon of a hypothesis species takes that hypothesis' move presence; everything else F's (the
  seen mons' item / ability / spread from F's own screen — the same draws even in a fresh process).
- **POOLSETS** (`poolsets_worlds.py`): F's structure with the training pool's per-species moves / item / ability /
  spread frequencies (`Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams`, 719 teams); its own screen.
- **POOL** (`pool_worlds.py`): a world = a pool team containing every revealed species and every revealed move, placed
  on the true slots, gated by the builder's own strict gate; uniform among candidates; an exclusion when none passes.

## 5. Scripts

| script | where | what |
|---|---|---|
| `t1_common.py` | — | constants: pins, seeds, stage bins, the supplement targets, the arms, the codes, paths |
| `census.py`, `select_supplement.py`, `make_supp_bank.py`, `t1_supp.py` | pin | the census (k, turn, material of every pool state), the supplement selection, its bank |
| `phase1_supp.py`, `phase1_supp.sh` | pin | the supplement's truth (the pilot's `run_playouts.one_state`) |
| `build_worlds.py` | HEAD | every arm's 8 worlds per state (`--validation <spec>` for non-arm checks) |
| `play_worlds.py` | pad pin | the pilot's `run_det_playouts.one_state` on this unit's worlds |
| `fn_belief.py`, `fn_worlds.py`, `poolsets_worlds.py`, `pool_worlds.py`, `pool_teams.py` | pin / HEAD | FN, POOLSETS, POOL |
| the hand evaluator | pin / pad pin | `../hidden_info_binding_2026-10-11/hand_eval_playouts.py` unchanged (`--world O`; `--world P --worlds-dir …/det2`; `--world F --worlds-dir <F's worlds>`), outputs under `<archive>/<cp>/hand_eval/<H-code>/` |
| `identity_read.py`, `pin_regate.py`, `world_audit.py` | — / pad pin / HEAD | validity (§2) |
| `t1_data.py`, `t1_stats.py`, `score_blind.py`, `unblind_t1.py` | pure | the read: arrays, bootstrap + rule 8, the blinded table, the rules |
| `t1_analysis_test.py`, `rehearsal.py` | pure | unit tests on planted effects; a dress rehearsal on the pilot's ALREADY-READ P rows (every arm a copy of P) |

## 6. Readings fixed BEFORE any arm output existed (where the registration leaves a choice)

1. **Primary sets.** R-MOVES, R-RIVALS, R-RECOVER, R-EVAL and the condition read: the registered 600 per checkpoint (the
   natural mix), pooled over the TWO checkpoints run (the registration's "three" includes `end2_s2002`, Tier 3) and per
   checkpoint where a rule says so. The stage rules (§12.13.4): every Tier-1S state, unweighted per bucket.
2. **Consequences of what Tier 1 does not run:** B1 needs R-REPLICATE, which is not available, so a pattern meeting
   B2–B8 reads "B2–B8 HOLD; MOVES-BINDING PENDING B1 …", never "MOVES-BINDING"; B5, R3 and NON-ADDITIVE use the class U
   only (I, A, S are Tier 2); B6's slope is fitted over the Tier-1 dose points p ∈ {0 (F), 0.5, 0.9, 1.0 (F+M)} (D25 /
   D75 are Tier 2); R-RIVALS is read for U only; the V8 guard is UNRESOLVED (M-STABLE needs the trajectory); CONDITION-
   ROBUST is read on the opponent class (bot / sentinel) and the game phase only.
3. **Pairing and exclusions.** Every contrast is on the states every involved source holds (an exclusion — a state the
   builder could not fit — removes the state from that contrast only, counted). §6.1's 10 % rule is applied to F.
4. **Bootstrap.** Per checkpoint: the pilot's battle-clustered bootstrap (2,000 draws, seed 20261011, percentile).
   Pooled: battles resampled WITHIN each checkpoint (§6.3). A share is a ratio of per-draw sums.
5. **Rule 8** (`t1_stats`): a boundary of ±0.0005 already carries the margin; any other boundary c is clear only beyond
   c ± 0.0005; "inside ±m" = lower > −m + 0.0005 and upper < m − 0.0005. Applied to shares and p-units alike.
6. **The hand evaluator.** Every HE scorer on Foul Play's own leaf scale σ(0.0125·(h − h_root)), a terminal 1 / ½ / 0
   (one scale, because a K-mean mixes terminals with evaluations); the critic side of each R-EVAL contrast is the SAME
   re-play's V (`VT-TRUNC-4-K8`, `VT-CRITIC`; equal to the arm's own scorers by construction, checked). R-EVAL is read
   in F (primary) and in P (`det2`, Amendment 1 (b)) separately; the verdict string names where it holds.
7. **The dose family.** M5X's "most wrong" world: the fewest correct hidden facts (the seen mons' true unrevealed moves
   held + the never-seen species that are true never-seen species; `build_worlds.correct_counts`), ties to the lowest
   world index; its light part = the 35 four-subsets of the other 7 worlds (as M5W). TODAY = the p-equivalent of FN on
   the 0.05 grid (its upper bound the 97.5th percentile over draws; above the curve's p = 1 value reads 1.0, flagged);
   Π_b uses M8 LEAF-K8 on a 0.01 grid at TODAY(b) (rounded, capped at 0.9) and TODAY(b) + 0.10, TODAY(b) fixed at its
   point estimate. P-STAR's third comparator (today's critic search) = M8 CRITIC at p = 0 (F's honest critic).
8. **Stage rules.** The slope of R-STAGE-VOI is on the axis z-scored over the frame's states; FLAT's b_max / b_min =
   the buckets with the largest / smallest point G; "replicated" = the same sign of the point estimate on each
   checkpoint. R-STAGE-CLASS: the σ- and ν-contrasts each name a direction (M, U or none); LED when both name the same;
   SIDES-DISAGREE when exactly one names one or they name opposite ones; then MIXED, NEITHER, NOT DETECTED. R-STAGE-
   SEARCH's pooled 99.6 % level = the (0.2, 99.8) percentiles.
9. **The opponent's root-turn choice.** Every source uses the HEAD builder's default `root_choice_in_log=False`: when the
   opponent's root-turn choice precedes the searched side's in the log (about 1 % of states), the world keeps the
   choice by SLOT (the world's move there), while O plays the true choice. Counted and read as a sensitivity (those
   states dropped).

10. **Amendment 3 (§12.14, registered 2026-10-13 before any outcome was unblinded; arm outputs for part of the states
    existed, unread):** per-class dose curves for M and U by re-weighting F+X and F worlds (`unblind_t1.class_dose`; the
    blinded table now also stores every world source's per-world arrays in policy order and the builder's per-state hit
    counts, `score_blind.fact_hits`). TODAY per class is a labelled hit-rate PROXY; CLASS-FIRST(b) compares Π_M and Π_U
    from the proxy point and from p = 0.
11. **Operations (no read).** `queue_play.py` keeps the playout workers busy in the registered priority order;
    `build_chain.sh` / `build_after.sh` run one world builder at a time (3 workers beside the live GPU run: 1 builder +
    2 playout shards); `pin_regate.py`'s REQUIRED tooth is a seen mon's REVEALED move replaced (always observable; the
    item swap is descriptive: an item that never acted is invisible, 18 of 19 caught on the first read).

## 7. Results (read 2026-10-15; the table frozen at `c6f71d9a`, sha `fc3aced3…`, then unblinded)

Full tables: [`result_t1.md`](result_t1.md) (rendered from `result_t1.json`); the key: [`arm_key.json`](arm_key.json) (its sha256
matches the sealed `arm_key.sha256`). Every read is PROVISIONAL "at 15M, current architecture" (§10.5 of the registration)
and COMPUTED, not replicated: R-REPLICATE (the blind replica) is a third party's comparison.

**In plain language.** Hidden information about the opponent's team costs honest search 7 points of win probability per
decision, as the pilot said, and the fully honest builder (which also hides items, abilities and spreads) did not make
the gap larger. Most of it is the opponent's NEVER-SEEN Pokémon, not the seen ones' hidden moves: telling the search the
never-seen Pokémon closes a third of the gap and hiding them opens four fifths of it, against an eighth and two fifths for
the seen Pokémon's moves. By the registered rules **"the moves are THE binding constraint" is REFUTED** (rule R3: the
never-seen class dominates on both sides). The moves still matter (a confident wrong move set HURTS; the move dose
helps a little), and the Bayes posterior of our own team pool recovers 80 % of the gap (it nearly identifies the team, which
the ladder would not allow). Foul Play's hand evaluator is no better than our critic in honest worlds. No stage of the game
has an honest search that clearly beats the critic on both checkpoints.

**Validity (no outcome):** the cross-commit identity (`identity_V_oracle.json`) PASS; every world re-gated at the engine that
played it (`pin_regate_*.json`: 16,800 worlds, 0 failures, every planted revealed-move swap caught); R-GIGO for the Tier-1
worlds (`world_audit.json`) PASS: 0 provenance / legality violations, every leak and dose check |z| ≤ 1.4; the k = 6
identities F+U ≡ F and O−U ≡ O hold on every state (0 of 276 / 320 differ); the hand-eval re-plays' critic scorers equal
the arms' own (0 of 1,097 / 1,200 differ). Exclusions (states the honest builder cannot fit): F 8.8 % (< §6.1's 10 %),
MAP100 27.6 %, POOLSETS 5.1 %, O−M / POOL 0. Size rules (§12.10, §12.13.2): not triggered (s_M / n_M half-widths 0.133 /
0.113; G_F per k bin ≤ 0.024).

| registered rule | verdict | the numbers (pooled end2 + prod, 95 %) |
|---|---|---|
| **R-MOVES** (§12.3) | **MOVES-REFUTED (R3)** | G = IMP_O − IMP_F **+0.070** [+0.060, +0.081]; s_M **0.13** [−0.02, 0.25], n_M **0.42** [0.31, 0.54]; s_U 0.31 [0.18, 0.43], n_U **0.82** [0.70, 0.96]; s_U − s_M +0.19 [+0.03, +0.35], n_U − n_M +0.41 [+0.26, +0.55] (R3); B2, B6 (D90 − F +0.011 [+0.001, +0.021]; slope over p ∈ {0, .5, .9, 1} +0.010 [+0.001, +0.019]) and B7 (D90 − W90 +0.025 [+0.013, +0.036]; W90 − F **−0.013** [−0.024, −0.003]) hold; B1, B3, B4, B5, B8 fail |
| R-RIVALS U (§12.4) | **U-SHARED** | B2, B4, B5 (vs M), B8 hold (F+U − F end2 +0.015 [+0.001, +0.030], prod +0.029 [+0.015, +0.043]; O − O−U +0.046 / +0.066); B3 fails (s_U point 0.31) |
| R-RECOVER (§12.5) | **BELIEF-RECOVERABLE (on this distribution)**; PRIOR-MISMATCH no; HONEST-SHARPENING-HELPS no; p½ none | r_POOL **0.80** [0.73, 0.87]; r_POOLSETS −0.02 [−0.19, 0.13]; MAP100 − F −0.002 [−0.014, +0.009]; r_D50 0.11, r_D90 0.16, r_F+M 0.13 |
| R-EVAL (§12.6) | **not EVALUATOR-BOUND** (F and P); EVALUATOR-NOT-BINDING not shown; **INFORMATION-BOUND (support)** | F: HE-TRUNC-4-K8 − TRUNC-4-K8 +0.007 [−0.001, +0.016], HE-1PLY − CRITIC −0.002 [−0.011, +0.006]; P: +0.003 / −0.005; g_HE +0.028 [+0.018, +0.037] vs g_CRIT +0.037, ratio 0.75 [0.52, 1.06] |
| NON-ADDITIVE | no | Σ s (M + U) 0.45 [0.22, 0.65] |
| CONDITION (§10.4, partial) | moot (no class reaches s ≥ 0.5) | s_M by opponent class / phase 0.08–0.23 |
| V8 guard (§10.4) | UNRESOLVED | needs the trajectory arm (Tier 3) |
| **R-STAGE-VOI** (§12.13.4) | k **TREND-DOWN**, turn **TREND-DOWN**, material **TREND-UP** (each replicated in sign) | G by k 1–2 / 3 / 4 / 5 / 6: +0.087 / +0.093 / +0.076 / +0.061 / +0.025; by turn ≤ 7 … ≥ 22: +0.095 / +0.078 / +0.069 / +0.033 |
| R-STAGE-AXIS | **turn CARRIES**; k and material do not once the others are in | partial slopes per SD: turn −0.014 [−0.024, −0.003], k −0.010 [−0.023, +0.003], material +0.003 [−0.009, +0.014] |
| R-STAGE-MECH | NOT DETECTED on all three (Δc clear, Δs not bounded within ±0.04) | k: c 0.78 → 0.42, s 0.116 → 0.103 (Δs +0.013 [−0.021, +0.045]); split Δc·s̄ +0.039 vs c̄·Δs +0.008: frequency carries the difference descriptively |
| R-STAGE-CLASS | **U-LED** at k = 1–2, 3, 4 and turn ≤ 7, 8–14; SIDES-DISAGREE at k = 5 and every material bucket; NOT DETECTED at turn ≥ 15 | two-point Shapley φ (pooled all states): see the CLASS table in `result_t1.md`; on the registered 600: φ_M = ½(σ + ν) = 0.019, φ_U = 0.040 (of G 0.070) |
| **R-STAGE-SEARCH** | **FRONTIER EMPTY**; k = 6 GATE-WEAK (pooled 95 % only) | k = 6: TRUNC-4-K8 − CRITIC +0.024 [+0.006, +0.044] pooled; per checkpoint n 145 / 131 (< 150 after exclusions) and bounds straddle 0 |
| R-DOSE (§6.9, M8 on F) | CONCAVE; CONTENT-MATTERS; CRITIC-INFO-LIMITED; OVERCONFIDENCE-COSTS no; P-STAR 0.1 (all three comparators); TODAY(FN) 0.009 [≤ 0.032]; WITHIN-REACH; DETERMINIZATION-FIRST k = 4, turn 15–21 | **read with the post-hoc FINDING below: the step to p = 0.1 is TIE-BREAKING** |
| Amendment 3 (§12.14) | M: P-STAR vs policy 0.75, vs critic search NONE, LINEAR; U: P-STAR 0.2 / 0.35, CONCAVE; TODAY proxies M 0.011, U 0.12; CLASS-FIRST from 0: U-FIRST at k 1–2, 3, 4 and alive 10–11, else NOT DETECTED | M curve +0.001 (p 0) … +0.013 (p 0.75–1); U +0.002 … +0.031 (0.75) |

**POST-HOC FINDING (`posthoc_tiebreak.py`, `posthoc_tiebreak.json`; read after unblinding, descriptive).** An 8-world
LEAF-K8 score is a mean of 8 win / loss / draw outcomes, so the honest argmax is TIED in **60 %** [58 %, 63 %] of states and
the registered tie rule picks the policy's first. ANY positive weight on a second component breaks those ties by it: the
M8 curve jumps from +0.002 (p = 0) to **+0.021 at p = 0.001** and is flat to p = 0.1 (+0.0004 more), then rises (+0.032 at
0.2, +0.053 at 0.35, +0.062 at 0.5, +0.072 at 1). So the registered "P-STAR = 0.1", "TODAY = 0.009" and "WITHIN-REACH" sit on a
lexicographic tie-break step, not on a dose: beyond the step the curve needs **p ≈ 0.2–0.35** of weight on the TRUE world
before it rises, half its further rise by p ≈ 0.30, and the network's own belief (FN, +0.006 ≈ F's +0.006) does not even
reach the step. The class curves show the same effect (U +0.007 at 0.001; M none) and rise only from p ≈ 0.2 (U) / 0.75 (M).
The pilot's descriptive curve (§16.7 of the registration) carries the same step.

**Other FINDINGS.** (1) FN, the network-belief worlds on the honest base, buy nothing at LEAF-K8 (+0.006 vs F +0.006; its
4-turn leaf +0.013 vs +0.005): the network's move belief hits 0.513 of the hidden moves vs Smogon's 0.508; its species belief
0.172 vs 0.058 of the never-seen species, and that buys no value either. (2) The fully honest gap equals the pilot's (P, the
v2 builder that told items / abilities / spreads true: IMP_P +0.009 vs IMP_F +0.006): items, abilities and spreads add
≈ 0.003, far less than registered prediction (+0.005 to +0.02). (3) POOL's 0.80 is team IDENTIFICATION within our 719-team
pool, which a ladder opponent does not offer; POOLSETS (the pool's per-species sets) recovers nothing, so the Smogon prior's
per-species mismatch is not the gap. (4) A confidently WRONG move set (W90) costs −0.013 [−0.024, −0.003] against F.
(5) k = 6 cannot be GATE-OPEN as registered: after F's exclusions its per-checkpoint n is 145 / 131 < 150.
(6) Opponent root-turn choice recorded before the root: 8 registered states; dropping them changes nothing (G +0.070, s_M
0.13, n_M 0.41). (7) Operations: the run held 3 processes (1 world builder + 2 playout shards) for ~24 h, within the brief's
3-worker allowance but above root rule 5b's "≤ 2 heavy jobs"; the GPU lease never read FREE, so the run never widened.
