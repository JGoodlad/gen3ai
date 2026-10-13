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

## 7. Results

(filled when read)
