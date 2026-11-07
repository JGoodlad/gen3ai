# S′-HONEST: does P\* + a LEAK-HONEST search beat P\* against the REAL Kakuna on unfamiliar teams? — PREREG (2026-11-06)

**STATUS: REGISTERED 2026-11-06, before any game of this read** (this file and [`read_spec.json`](read_spec.json) in one
commit, on the coordinator's go). Pin `e46e3ddf8`; §15's checks R0–R5 all PASSED at the pin.

**History before registration (no read data existed at any step).** Ported from the unregistered `r0s4_honest@10`
draft (local branch `sprime-honest-prereg`, `a9c9f6499`) to the incremental builder; REVIEW A (§ Review) and its
verification folded in; the deals raised from the brief's 3,840 to **5,100**, accepted by the coordinator
(2026-11-06) on §3's argument: the conservative SE needs ≈ 5,060 for 80 % at +2.30 pp, and the incremental
builder's ≈ 4.9× cost drop makes that ≈ +10.5 h of the box; the read pinned after the our-HP fix `83272a8f9` landed
(the coordinator's ruling: the read runs on code that holds it), every identity reference re-derived at the pin. Written BEFORE any game of this read. Design:
[`design_sprime_honest.md`](../../../endstate/design_sprime_honest.md) (§5 the replication's inputs, §9 the incremental
builder). Spec: [`read_spec.json`](read_spec.json) (`main.reads`, rule `searched_play`). Pin: `e46e3ddf8`. State:
`~/gen3ai_archive/sprime_honest_play_2026-11-06/`. Licensed by S′-LIVE's GO-LEAKY
([`../sprime_live_play_2026-11-01/README.md`](../sprime_live_play_2026-11-01/README.md)), which licenses this
replication and nothing else.

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`), greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy.
- **S′h** = `r0s4_honest_inc@10` (`main.anchors.search_presets`; `main.value_search.honest_inc`): `r0s4_rolls`'s 8
  worlds, TRUNC-4-K8 leaf and piKL λ 0.015, every world built from the searched side's information set alone on
  fresh dice matched segment by segment, and CARRIED from one searched decision of a battle to the next (design §9).
  **S′** = `r0s4_rolls@10`, S′-LIVE's searched arm (its worlds replay the true battle seed).
- **I** = the searched side's information set at a decision: the format, both names, our packed team, every protocol
  line our side was shipped up to the decision (its requests included), and our own commands.
- U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal weight.
- **E6** = the engine `port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130` (stage 1b's; S′-LIVE's E4 was
  `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`).
- **The HP fix** = `gen3_our_hp_exact_v1` (config v165, main `83272a8f9`): every HP of our side reaches P\*'s damage
  operator as an exact integer (a Seismic Toss against exactly 100 HP was read as no KO).
- y = a game's result for us: win 1, tie ½, loss 0. D = y(searched) − y(raw) for one deal. G_U = §5's estimand.
- LB / UB = a 95 % interval's lower / upper bound. pp = percentage points. K = the anchor lane's concurrency.
- GUESS = an unmeasured effect. ESTIMATE = a cost from measured rates.

---

## 1. The question and the classes

**Does P\* + S′h, searching every eligible decision from turn 10 with a search that reads only what the live player
knows, win more games against Kakuna on U than P\* alone, on the same deals?** The truth is two real games per deal
against the real opponent. No referee and no opponent model enter it. This is the verdict that decides whether search
is our path to goal (b): beat Kakuna on unfamiliar teams.

| class | rule | what it licenses | what it does NOT license |
|---|---|---|---|
| **GO-HONEST** | LB(G_U) > 0, with every post-treatment unpaired deal imputed at its worst case for GO (§5) | P\* + S′h, a search that is a function of I alone (§7), beats P\* on U against Kakuna. **Search is a measured path to goal (b)**: the search program for goal (b) is un-paused, and its next registrations (a confirmation if GOAL-B is not reached, cost, the deployable player's design) may be written | any public-server or ladder play (the owner's approval file and the drift scan, root § Playing); any training or league change; a claim against other opponents, home72, other budgets, builders or presets; a size claim beyond the interval |
| ↳ **GOAL-B-REACHED** (a decided flag on GO-HONEST) | GO-HONEST AND the searched arm's pooled-U score LB > 0.50, every post-treatment unpaired deal's missing searched game counted a LOSS (§5) | "**Goal (b) met with search:** P\* + S′h beats Kakuna (checkpoint 34, greedy) on unfamiliar teams (U = H, F, P72)", always quoted with that regime, this n, and ties as non-wins | the same as GO-HONEST's column; "goal (b) met" without its regime |
| **NULL** | not GO-HONEST, and UB(G_U) < +3 pp, with the opposite imputation | S′h at P\* adds less than 3 pp on U (95 %). Search **as built** (`r0s4_honest_inc@10`, §8's builder) is closed as a stand-alone route to goal (b) at P\*: §5's margin. The search ledger enters [LB, UB] beside the other levers. Beside S′-LIVE's GO-LEAKY on other deals, the leak is the likeliest carrier of S′-LIVE's gain; its size is §12's cross-read difference, REPORTED, never decided | excluding a gain of ≤ 3 pp that would stack with other levers; any claim about other budgets, another builder (S′H-15's fix, i.i.d. worlds), a Kakuna-modelled leaf, deeper search, another network or home72; a decided leak share |
| **STOP** (undecided) | otherwise | Nothing: unresolved at the registered n. Search stays paused for goal (b) | a re-run at a larger n without a new registration |

- **The classes are S′-LIVE's**: the same rule, the same imputation, the same +3 pp margin and the same rule-8
  handling. Only the GO class's name differs: the rule spells it by the registered preset's dice (GO-HONEST for
  `fresh` and `fresh_inc`, `rules_searched_play.go_class`), so this read can never print GO-LEAKY.
- **A bound within 0.0005 of its bar is never a crossing** (root rule 8): that class does not hold, and the record is
  flagged `boundary`. GOAL-B's LB within 0.0005 of 0.50 is never reached.
- **Two flags, neither a class.** `go_below_margin` (a GO whose NULL-imputed UB is also below +3 pp: a gain, but not
  by itself enough for goal (b)) is reachable at this n only if the realised SE is near S′-LIVE's (§9).
  `excess_exclusions` (§6) never VOIDs and never changes the class; when set, the record says the arm excluded more
  live than stage 1b's own sample can explain, and §12's cross-read difference is not read as the leak's share alone.

## 2. Why this read, and what it is not

- **It is the replication S′-LIVE's GO-LEAKY licensed** (its PREREG §1: "building and registering a leak-honest
  replication on fresh deals, and that work alone is un-paused"). The honest search was built and measured in two
  stages (design §8 `r0s4_honest`, §9 `r0s4_honest_inc`); this read plays the second.
- **One lever.** S′-LIVE's searched arm `r0s4_rolls@10` becomes `r0s4_honest_inc@10`. Everything else is S′-LIVE's:
  P\*'s bytes, Kakuna, the peer pack, the U panels, lone one-game units paired by deal, G-PREFIX with the `peer_first`
  classifier, the rule, the imputation, the caps, the 10-h watchdog and the identity guard's design.
- **Why the incremental builder, not `r0s4_honest`.** Both are functions of I and pass the leak test (§7). The cold
  builder re-matches the whole of I at every decision: ≈ 30.3 CPU-min a searched game at K = 6, U exclusions ≈ 7.7 %
  (F 14.8 %), and F's longest games projected near the 10-h watchdog (a VOID risk). The incremental builder carries the
  worlds: ≈ 6.2 CPU-min, ≈ 3.4 % (F 5.8 %), its longest game ≈ 1.1 h (§11, stage 1b). It picks `r0s4_honest`'s action
  in 63 % of U decisions where both search (258 / 411, ± ≈ 5 pp), as often as `r0s4_honest` re-drawn under another key
  picks its own (63 %): indistinguishable from the cold builder by pick agreement at that n, at a fifth of the cost.
  That is not a measurement of equal strength. Its own biases (§8) are part of S′h as defined.
- **What else moves, and why it is not a second lever.**
  - The engine stamp, E4 → E6: the Rust sources gained the fresh-dice matcher, its carried trajectory, the `seg`
    damage-roll mode and the reseed schedule, all opt-in. `r0s4_rolls` re-scored S′-LIVE's registered 600 decisions
    byte-identically at E6 (design §9.4).
  - **The HP fix.** It corrects P\*'s own forward (both arms equally): the fixed-damage KO cut at our exact HP, and the
    heal worlds' truncation. Its changelog entry measured it on P\*'s 16,785-row tie-rate population: 1,118 rows (6.7 %)
    change at all, |Δ log π| median 2.9e-6, p99 7.3e-4, max 0.113 (30 rows above 1e-4, every one through the heal
    worlds), |ΔV| ≤ 3.0e-4, **0 top-action changes**. The read must run on code that holds it (the owner, 2026-11-06).
    It moves P\*'s raw play too, so it is not a difference between the arms; it is a difference between this read and
    S′-LIVE, which §12's cross-read difference carries (small by that measurement, not measured on games).
  - Fresh deals and a larger n (5,100 against 2,698: §3, §9).
  - The rule's spelling of GO, two decided flags (GOAL-B, `excess_exclusions`), a reported distillation-field count,
    an identity guard that also compares the honest build record and rebuilds each sampled decision's carry from I,
    and the units' core set and slot count (`kakuna.unit_shape`): code off the game's content (§4).
- **The forking-paths cost, stated.** This read exists because S′-LIVE read GO-LEAKY.
  1. **No pooling.** A GO-HONEST here is never pooled with, or "confirmed by", S′-LIVE's estimate, and the two reads'
     deals differ, so they are never differenced as a pair.
  2. **S′-LIVE's record stands as written** (GO-LEAKY, +4.60 pp [+2.66, +6.48]); it is not re-read.
  3. **The cross-read difference** (S′-LIVE's G_U − this read's) is REPORTED with its unpaired interval (§12), and it
     carries S′-LIVE's selection: that read is here because it cleared its bar. At S′-LIVE's estimate, 4.7 SE above
     its bar, the selection bias is small, but it is not zero.
- **The link to X65-R2** (the X65 ExIt-0 record, `110865d2a`). This read banks every honest searched decision's
  distillation fields (§6, X65-F2): ≈ 140k honest labels at no extra search CPU, the input of an ExIt-0H registration.
  Nothing a learnability probe finds changes this read.
- **No human replay data** (the owner, 2026-11-06). No human DECISION enters: P\* trained with no `--human-data`
  lever, the builder's priors are Smogon's (`main.value_search` reads no replay or ladder source), no human opponent
  model. What is human in the read is what S′-LIVE had (`human_data_architecture.md` §2): the forum-built F teams
  (fine locally, the owner), the replay-DERIVED H teams (teams extracted from Metamon's ladder archive, hidden sets
  filled from usage; no decision), and Kakuna itself (Metamon's, pretrained on human replays: the opponent, not our
  data). **FINDING S′HP-1: whether the H panel's replay-derived team lists fall under "no human replay data" was put
  to the orchestrator: RESOLVED, H stays** (§15 R0).

## 3. The arms, the deals, the pairing

- **P\* + S′h** (`pstar_s`, `load.search: r0s4_honest_inc@10`).
  - S′h decides every one of our decisions at turn ≥ 10 with ≥ 2 legal actions (`main.anchors.search_side`). Every
    other decision is P\*'s greedy choice.
  - A decision whose builder returns fewer than 8 worlds consistent with I in its budget is an EXCLUSION: P\* plays
    it, it is recorded, and it is part of S′h as defined. It is a function of I alone (D8): deployable and
    outcome-blind. After one, the battle is STUCK and its later decisions probe first (design C28).
  - Any other search error fails the game (a named defect, never a fallback): a root guard's `LiveSearchError`, the
    strict gate's RAISE (C13), anything else.
- **P\* raw** (`pstar`), greedy, the same pinned bytes (`spec.search_twins`).
- **The opponent and server.** Kakuna, greedy, peer pack `gen3_peer_showdown_pack_v1`. Our `ws_frontend` serves on a
  9XXX port, through the `main.reads` anchor lane.
- **The deals: U unit seeds 700–784** on H, F and P72, 20 deals a seed: **5,100 deals, 1,700 a panel, 10,200 one-game
  units**. They are disjoint from every deal any registered Kakuna read has played or reserved: S′-LIVE 300–344; dlive
  200–244 (stage 1's 300 states, stage 1b's and INV's fixtures are dlive's or S′-LIVE's states); X65 ExIt-0's second
  look 600–674 (FINDING X65-F3); S-U's gate 100–109; X63 1000–1199; every other read's 0–59. The smoke (§15) played
  home72 seeds 200–201, not a U panel.
  - A battle's seed is `blake2b("<unit seed>:<index>")` (`utils.bridge.seed_spec.derive_seed_from_base`) and its team
    draws are keyed by the unit seed, so a new unit seed is a new deal, never a shifted old one.
- **Why 5,100 and not the brief's default 3,840 (64 seeds)** — argued from stage 1b's numbers, as the brief requires:
  - The design's target is 80 % power at half S′-LIVE's effect (+2.30 pp). 3,840 reaches it only on S′-LIVE's
    REALISED SE (0.974 pp at 2,698 pairs). That SE is 12 % below the iid value the same data give (E[D²] = 0.339,
    √(Var D / n) = 1.118 pp; the unit-cluster SE 0.981 pp; FINDING S′HP-9), and nothing in the design makes it
    structural: every game's two teams are independent draws WITH replacement (`metamon_side.install_seeded_team_draw`'s
    `choice`, `Gen3Teambuilder`), so no stratification shrinks a unit's variance. With 135 units the gap is ≈ 2 SE of
    the SE estimate: likely chance. On the conservative SE √(d / n) (d = 0.341), 3,840 gives P(GO | +2.30) = 0.69 and
    P(STOP) up to 0.29 near +1.5 pp; 5,100 gives 0.80 and at most 0.10, the decisiveness the design promised (§9).
  - **Stage 1b is what makes that affordable.** The draft kept 3,840 for cost: on the cold builder the read was ≈ 14
    days locally and each further 660 deals ≈ 1.5 days. Stage 1b's builder cuts a searched game ≈ 4.9× (30.3 → 6.2
    CPU-min), so 5,100 costs ≈ 577 CPU-h against 3,840's 434: ≈ +10.5 h of the box (§11). Cost is outcome-blind, so
    buying power with it is no forking path.
  - Stage 1b moves no input of the power itself. Its departure rate (38.6 % of decisions, S′-LIVE's live arm 36.6 %)
    cannot move d: 98.4–99.4 % of S′-LIVE's searched games already held a departure (`inputs/cost_model_inc.json`), so
    nearly every pair's trajectories split either way. Fewer exclusions (3.4 % against S′-LIVE's 5.8 %) mean S′h
    searches more of the game, not less.
  - 85 seeds is the smallest whole-seed n reaching 80 % at +2.30 pp on the conservative SE (≈ 5,060 needed).
- **One `per_game` group plays both arms**, deal by deal: game g's raw unit, then its searched unit, on every panel,
  before game g + 1. Unit seeds are the outer loop, so a stopped queue stays balanced by seed and panel.
- **Every unit is ONE game, alone in a fresh process** (`main.anchors --games 20 --deal-game g`), for dlive's FINDING
  DL-7 (Kakuna's in-process divergence), as in S′-LIVE. A fresh process starts with an empty carry store; the
  incremental builder's first searched decision of the battle is built cold (design C25).
- **The shared prefix.** Before turn 10 both arms are the same deterministic play on the same seed and teams.
  G-PREFIX requires each pair's protocol (`|t:|` removed, both sides, and the `START` document) to be byte-identical
  through `|turn|10` (§6).
- **After turn 10** the arms differ by design. GUESS, as in S′-LIVE: residual Kakuna nondeterminism adds variance, not
  bias, ≪ 1 pp.

## 4. The engine, the bytes, the pin, the checkout

- **One engine: E6.** Every counted unit of both arms is on it, or the read is VOID. The life script checks
  `engine_stamp()` against E6 before every supervisor life (§14); the rule VOIDs on any other. The HP fix and S′H-14
  touch no Rust source, so the pin builds E6: checked at registration (§15 R2).
- **The pin `e46e3ddf8`** is S′H-14's main commit (`live_identity`'s carry modes, the rule's leak-honest dice,
  `kakuna.unit_shape`), which sits directly on the HP fix's `83272a8f9`: it holds both. The read plays at the pin and
  the identity re-score runs there.
- **The pin's play path against stage 1b's.** Stage 1b measured at `f7da935ab`. Main's `335355e6f` (S′H-15's record)
  changed `src/main/value_search/honest_inc.py` there, in its docstring and one comment only: the two versions' ASTs
  with docstrings stripped are equal (checked 2026-11-06; §15 R3a). So the reference is `335355e6f`:
  `main.reads.playpath.compare("335355e6f", "e46e3ddf8")` = the 9 files below and `kakuna.py`, nothing else.
  **Required (§15 R3b, PASSED):** exactly the HP
  fix's files under `src/agents/model/` (`damage_kinds`, `damage_op`, `damage_op_blocks`, `damage_op_pairwise`,
  `damage_tables`, `eot_residual`, `model_version/constants`, `model_version/migrations`, `selection_sites`, as the
  branch stands) and `src/main/x35/kakuna.py` (the unit shape, scheduling only); no file of the builder, the search or
  the engine (`src/main/value_search/`, `src/main/anchors/`, `src/rust_sim/src/`, `src/rust_env/src/`). Then:
  - **The builder is stage 1b's, byte for byte.** It reads no network (the chain, the matcher and the gates are
    functions of I and its keyed dice), so every exclusion, every world and every `honest` record field stage 1b
    measured is what the pin computes on the same I; the cost model (§11) transfers exactly.
  - **The search's values are not.** Q̂, log π, root V and so the pick read P\*'s forward, which the HP fix moves on rare
    rows (and by ulps elsewhere). Stage 1b's agreement figures are a description of the pre-fix S′h; nothing in the
    verdict rests on them.
  - **MEASURED at the pin (§15 R4):** the smoke's live records (played before the HP fix) re-scored at the pin in REBUILD
    mode: every builder field (`searched`, `why`, `search_key`, `honest`) must be identical; differences in `played`,
    `q`, `log_pi` or `root_v` are counted and reported as the HP fix's. The smoke holds 3 sampled decisions, so R3b
    carries the weight and R4 is a live end-to-end check of it.
  - A builder file differing in any byte is a STOP before registration (§15 R3b): the read would then need INV and a
    builder tooth re-run at the pin, which this PREREG does not register.
- **The checkout** is a clean detached worktree at the pin, bootstrapped `--with-rust` (DL-4: the env stamp hashes the
  gitignored `Cargo.lock`).
- **One host, locally** (the owner, 2026-11-06: no rented machine). Every unit of both arms and the identity re-score
  run on this box.

## 5. The rule (`searched_play`; the spec's `decision` block)

The rule is S′-LIVE's, unchanged in every estimate and class test; `rules_searched_play`'s docstring is its text.

- **Pairs.** A deal is a PAIR iff both arms' one-game units COUNTED (`results.unit_counts`, the forfeit-win guard
  included) and it is not a peer drop.
- **Unpaired deals**, by kind:
  - PRE-TREATMENT: G-PREFIX peer drops (before turn 10). Excluded, never imputed.
  - POST-TREATMENT: every other finished deal without a pair: one side counted, or neither, after its attempts. Timer
    kills, a recurring search defect, `peer_long_tail_fault` and every other peer fault after turn 10 are this kind.
- **Imputation, inside the unit statistic like a pair:**

  | missing side | for GO-HONEST's LB | for NULL's UB |
  |---|---|---|
  | searched | D = −y_raw | D = 1 − y_raw |
  | raw | D = y_s − 1 | D = y_s |
  | both | D = −1 | D = +1 |

- **Three intervals:** paired-only (descriptive), GO-imputed and NULL-imputed, each from a fresh
  `default_rng(20261106)`.
- **The estimand.** A unit statistic is the mean D over a (panel, unit seed)'s deals, in exact fractions; G_panel the
  mean of its 85 unit statistics; G_U the mean of G_H, G_F and G_P72.
- **The interval:** percentile bootstrap over units within each panel, B = 2,000, panels in the order H, F, P72, for
  each `idx = rng.integers(0, n, size=(B, n))` over its unit statistics in unit-seed order; G_U's replicate b the mean
  of the panels' replicate b; bounds `numpy.percentile(reps, [2.5, 97.5])`.
- **The classes** (§1), compared as exact fractions with `rule8_eps` 0.0005: GO-HONEST on the GO-imputed LB > 0; NULL
  on the NULL-imputed UB < `null_margin_pp` 3.0.
- **GOAL-B** (`decision.goal_b`): the searched arm's pooled-U score (`strat_score`: the equal-weight panel mean of win
  shares, a tie a non-win, its Wilson interval at the effective n) over every finished non-peer-drop deal, a missing
  searched game a loss and a counted one at its result. `reached` iff the class is GO-HONEST and its LB > 0.50, never
  within 0.0005. The Wilson interval treats the games as independent: it is a claim about these panels' teams, not a
  population of teams beyond them.
- **The margin, +3 pp, is S′-LIVE's, kept for symmetry.** Raw P\* scored 0.486 on pooled U in dlive (2,700 games) and
  0.498 in S′-LIVE (2,698). Goal (b) needs a searched-arm LB > 0.50, a point of ≈ 0.514 at this n (score SE ≈ 0.70 pp),
  so ≈ +1.6 to +2.8 pp over those raw scores. A gain whose UB is below +3 pp cannot carry P\* to goal (b) by itself
  except at the top of its interval and the top of P\*'s raw range: "closed as a stand-alone route" means that.

**VOID** (named reasons, before any estimate; `rules_searched_play.void_reasons`).
- **G-PREFIX:** not run on the group; a mismatch the registered classifier does not class `peer_first`; G-PREFIX
  checked a different number of deals than the pairs plus the peer-dropped deals.
- **Engines:** counted units on two engines, an unknown one, or one other than E6.
- **Bytes:** any counted unit or resolved model on bytes other than P\*'s sha256.
- **Regime:** a counted searched unit not stamped `greedy+search=r0s4_honest_inc@10` or not played with that preset's
  config; a counted raw unit carrying a search stamp, or a regime other than greedy.
- **Results:** a counted unit whose `games.jsonl` result disagrees with its summary.
- **Units:** a panel with fewer than 2 units holding a pair.
- **Unfinished deals:** any deal whose units are neither counted nor out of their attempts (never a drop).
- **Fail-closed, not a rule reason:** a unit that exhausted its attempts with no summary at all (a SIGKILL, an OOM at
  the 12 GB cap) leaves the group `unplayed`, and `record` refuses. The operator re-runs that deal's units under a new
  supervisor life; if the deal still cannot play, the read is stopped and reported, never decided.
- **Drops, in any U panel:** pre-treatment above **2 %** of its 1,700 deals (more than 34); post-treatment above
  **0.5 %** (more than 8).
- **The identity guard (§6):** it FAILs; it re-scored fewer than 300 decisions; its output does not cover exactly the
  registered-order first 600 sampled decisions; it did not compare the honest fields (`search_key`, `honest`); or it
  did not run in the registered carry mode (`identity.carry` = `rebuild`); or its recorded checkout is missing, dirty,
  does not play as the pin, or differs from it in the re-score's own code (`inputs.code`,
  `rules_searched_play._rescore_at_pin`).

## 6. The guards and the teeth

**G-PREFIX and the pre-registered peer drop** (S′-LIVE §6, unchanged).
- G-PREFIX (`main.reads.prefix`) compares each pair as soon as both units count; a mismatch HOLDS the lane
  (`by: G-PREFIX`).
- The rule classes each mismatch with `main.reads.peer_divergence.classify_pair`, branch turn 10: **`peer_first`** is a
  PEER DROP (excluded, counted per panel); **anything else** (`ours_first`, `peer_first_late`, `unattributable`, an
  unreadable capture) is VOID.
- **The operator's step:** the read-only, outcome-blind
  `python -m main.reads.peer_divergence prefix --spec SPEC --state STATE --group u` prints each mismatch's deal, class
  and divergence turn. A `peer_first` mismatch is released with `python -m main.reads resume --state STATE --lane
  anchor --by <who> --why "peer_first <deal>"` (in the next life's script); any other class stops the read as VOID.
  `release-guard-hold` is never used for G-PREFIX.
- GUESS on the rate: ≈ 0 (S′-LIVE: 0 of 2,698).

**The choice-level identity guard** (`main.value_search.live_identity rescore`). The S′h that PLAYS must be, decision
for decision, the S′h the pin's code computes offline from the same I.
- **The sample.** A live search-path decision is in it iff `int(sha256(key)[:8], 16) % 16 == 0`, key =
  `su_live_v1|<deal>|t<turn>|n<n>` (outcome-blind). A sampled record keeps its honest log, mask and tokens, and the
  carry it started from (`inc_carry_in`). Expected ≈ 12,200 sampled (≈ 38.4 search-path decisions a game ÷ 16 × 5,100).
- **The re-score.** After the lane ends, the first **600** sampled decisions in the registered order (sha256(key) first
  across every cell, then panel, unit seed, game and attempt dir), re-run OFFLINE at the pin with one torch thread,
  through `sign_gate.GateScorer`'s root guards and the live searcher's own `_search`. Every attempt dir counts.
- **The carry: REBUILD** (`--carry rebuild`, registered as the spec's `identity.carry`). Each re-scored decision runs
  on a searcher whose carry store is EMPTY, so the builder rebuilds its battle's earlier searched decisions from I
  alone (design C19's cache-miss path) before the decision itself. The live process reached the same carry through its
  cache, in decision order; the compared `honest.inc.carry_in` digest then checks that the live carry WAS the pin's
  function of I, not only that the last step reproduces. `seeded` (from the record's own `inc_carry_in`) would check the
  last step only, at ≈ 0.9 CPU-h instead of ≈ 16 (§11): rejected, because the carry is where the incremental builder
  could hide a dependence on process history (a cache, an order, a retry) that the leak test's fixtures never meet.
- **What must be equal, exactly as recorded:** searched-or-excluded, the exclusion's `why`, the pick, Q̂, log π, root
  V, `search_key` (I's digest) and the `honest` build record (the chain's counts, the fails, each world's attempt /
  re-rolls / feeds / rolls, the root command, the world teams, and the `inc` block: the step, `carry_in`,
  `carry_out`, the worlds pending / extended / failed / escalated / new / retired, their ages and the overage).
  Every one is a count, a rounded value written by the same code, or a digest, so the comparison is exact (root rule
  8). It reads no result; it prints counts only.
- **PASS iff** ≥ 300 decisions are re-scored and none differs. **A FAIL is a VOID**: the live player is not the measured
  S′h. Run with `--jobs 14` on the whole box (≈ 2 h, §11), else `--jobs 3` on the agent cores.
- **The re-score ran the pin.** Its JSON records the re-scoring checkout (`inputs.code`: commit, dirty); the rule VOIDs
  a pinned leak-honest read whose re-score checkout is missing, dirty, does not play as the pin
  (`main.reads.playpath`), or differs from the pin in the re-score's own off-path code (`live_identity.py`,
  `sign_gate.py`), so neither a builder edit off the engine stamp nor another comparator can pass unseen.
- **A killed attempt's torn record.** `live_identity` reads every attempt dir, `.partial` ones included. A KILLED
  attempt's decision log (a `.partial` dir, or the last attempt of a unit that never counted: a watchdog, an OOM, a
  stop) may end inside a record; that one torn, newline-less last line is dropped and named (`torn_tails`, reported).
  Any other unreadable line, or a torn line in a counted unit's dir, refuses (exit 2) and the record waits for the
  operator.
- **What the smoke covered, and what it did not.** The smoke (§15) ran this guard live in both carry modes: 3 of 3
  identical, one of them an EXCLUDED decision of a stuck battle's full attempt (40 checkpoints rebuilt). None of its
  sampled decisions came after a stuck stretch that recovered; the checkpoint's re-score (≈ 140 live decisions,
  rebuild mode) is the first wide coverage of that path and of long games, before the bulk is played.

**The exclusion guard** (`max_excluded_frac`, decided as the flag `excess_exclusions`; the checkpoint's hold).
- **The meter:** per panel, the searched arm's exclusions ÷ its search-path decisions (searched + excluded), over every
  counted searched unit (`rules_searched_play.exclusions`), as an exact fraction.
- **The projection** (`inputs/excl_bounds.py` → `excl_bounds.json`): stage 1b's exclusion rate over EVERY decision of
  its 407 battles' sequences (stuck stretches included) per turn bucket (10–19, 20–29, 30–63, 64+), weighted by
  S′-LIVE's per-panel search-path decisions a game in each bucket. Exclusions rise with the turn (F 1.8 % at 10–19,
  15.0 % at 64+; P72 0.3 % → 20.5 %), so a raw rate from early states would trip on the turn mix alone. **Its own error
  is large**: the late buckets rest on the 14 F and 11 P72 battles that reached turn 64, and resampling stage 1b's
  battles (exclusions cluster by battle) gives H 0.29 % [0.05, 0.62], **F 5.75 % [1.45, 12.86]**, **P72 4.02 % [0.58,
  8.59]** (95 %).
- **The bound, per panel: the larger of twice the projection and the projection's own 95 % upper end, never below 5 %,
  rounded up to a whole percent: H 0.05, F 0.13, P72 0.09.** At 1,700 games a panel the live share's own sampling
  noise is small, so what the bound must absorb is the projection's error: above it, the arm excluded more live than
  stage 1b's sample can explain AND at least twice what it projects (the live distribution or the long game defeats
  the builder), a less-searched S′h than registered. The 5 % floor: H's projection is 0.3 %, and an exclusion share
  under 5 % dilutes the treatment by under 5 % of its decisions: immaterial.
- **Why a flag and not a VOID.** Exclusions are part of S′h as defined, and a function of I: a GO-HONEST at a high
  exclusion rate is still a deployable gain, and a VOID would discard it. What a high rate harms is the reading of a
  NULL or of the leak share; the flag narrows those (§1).
- **At the cost checkpoint** (below) the per-panel shares are REPORTED, not held on: with 20 searched games a panel a
  battle-resampled share exceeds its bound with probability F 0.13 and P72 0.25 even when the builder behaves exactly
  as stage 1b measured. The checkpoint holds only on the POOLED share (all three panels' exclusions ÷ their search-path
  decisions) above **0.14**, its 99th percentile under the same resampling at 20 games a panel (`excl_bounds.json`).

**The distillation fields** (X65-F2; `rules_searched_play.distill_report`, REPORTED). Every searched record must carry
`q`, `log_pi`, `root_v`, `frame_sha256`, `log_sha256` and `search_key`; every excluded one `frame_sha256`,
`log_sha256` and `search_key`; every counted searched unit its capture. The incremental builder writes them all (its
records are `r0s4_honest`'s plus `honest.inc`); the smoke verified it live (§15). A missing field never VOIDs the read;
it HOLDS the lane at the checkpoint, because the labels are the read's second product.

**The cost checkpoint** (operational, never a stop on outcome).
- **When:** the queue's first 120 units: unit seed 700's 60 deals on all three panels, both arms (life 0,
  `run --max-units 120`).
- **What the operator reads:** `main.reads search-cost` (CPU, decisions, exclusions, deviation rate, G-PREFIX); each
  panel's and the pooled exclusion share (§14's one-liner); the builder report (`inputs/inc_report.py`: modes, stuck share, overage,
  cold-build CPU); the identity re-score of the checkpoint's sample (rebuild, `--min 1`); the distillation report;
  nothing else.
- **The lane HOLDs (life 1 is not started) if:** the checkpoint's re-score FAILs; G-PREFIX holds; a unit used up its
  attempts on any class but `peer_long_tail_fault`; the slowest searched unit's wall projects past the watchdog (its
  wall ÷ its turns × 250 > 10 h); **the pooled exclusion share exceeds 0.14**; **any distillation field or capture is
  missing**; the mean searched game costs more than **12.4 CPU-min** (2× the model's 6.2).
- **What follows a hold:** the orchestrator decides from the hold's cause alone: continue (a flag, if any, is then read
  at the record), or stop and re-register (a new budget or builder is a new preset and a new registration; the
  checkpoint's units are then discarded unread). **The size never changes.**

**The watchdog** (S′-LIVE's, unchanged: `main.x35.kakuna.UNIT_SEARCH_S`, 10 h of wall a searched unit; it is on the
play path, so it is not touched). A timer firing is a harness defect, never an outcome: a post-treatment drop, imputed
and capped. **Risk: negligible.** At the bucket means S′-LIVE's longest searched games (≈ 245 search-path decisions,
189 of them past turn 64) project to 0.72 h (H), 1.08 h (F), 1.00 h (P72) at K = 6 CPU; no S′-LIVE game exceeds 10 h
even at 3× the means (`inputs/cost_model_inc.json`). The longest stage-1b sequence (162 decisions) built its worlds in
2,098 CPU-s. A unit at K = 14 holds one logical CPU, so its wall ≈ its CPU.

**The peer-drop and HOLD policy** (the orchestrator's standing decision of 2026-11-05, S′LIVE-OP-6 / OP-7; the brief).
- A unit exhausted by **`peer_long_tail_fault`** (Kakuna's "Force resetting due to long-tail error", printed when it
  races our turn-250 forfeit: SOP H5 / H21, `main.anchors.session.PEER_FAULT_LINE`) is a post-treatment drop, imputed
  at its worst case and counted against the 0.5 % cap. **It does not hold the lane.**
- **The lane is HELD** (`main.reads hold --state STATE --lane anchor --by sprime_honest_operator --why "<cause>"`) on:
  1. an exhaustion of **any other error class** (root-caused from its class in the unit's attempt logs, never its
     result);
  2. a panel's post-treatment drops reaching **3** (the VOID cap is 9);
  3. **more than 2** exhaustions of any class in any 6 h. Kept as the standing rule; at K = 14 the lane runs ≈ 2.3× as
     many units an hour as at K = 6, so at the same per-unit fault rate it trips ≈ 2.3× as often. A hold is only a
     pause (FINDING S′HP-10).
- The operator checks these at every watcher wake (§14: `main.reads status`'s `exhausted` per label, and each new
  exhausted unit's class); the supervisor does not apply them by itself.

**Built, unchanged:** the forfeit-win guard (a game ended by a peer drop or a forfeit other than our own turn cap never
counts); the off-loop search (the peer's keepalive is held, S-U A9); the 250-turn cap, symmetric in both arms.

## 7. The leaks: what this read closes, and what it does not

- **Closed by construction** (design §1–§2, §9.1): every channel of S′-LIVE's inventory. L1 (the true seed in the
  prefix), L2 (persisted dice at the root), L3 (the true opponent team as the scaffold), L4 (the opponent's private
  commands), L5 (keying by the truth), L6 (exclusions that depend on the truth). The incremental builder's carry at a
  decision is a function of the carry at the previous checkpoint and of its own I, the first built cold, so it is a
  function of I alone (design §9.1).
- **Closed by test — the leak test (INV) status.** `honest_inv_test.py` and `honest_inc_test.py` (slow) PASSED at E6
  on stage 1b's code (design §9.3–§9.4, stamp `port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`): for one case per
  channel, an alternative truth D′ that ships the same I leaves `r0s4_honest_inc`'s record identical **at every
  decision of the case's whole sequence from turn 10** (sequence mode: L1 7 decisions, L2 3, L3 2, L4 9, L5 2, L6 24),
  each truth on its own searcher; `r0s4_rolls`' record moves on every channel (the test's teeth); the control (another
  I) moves it; a sequence re-run, a decision rebuilt alone and a decision seeded with its carry reproduce every record.
  INV runs on a DIGEST network, so the HP fix (a P\* forward change) cannot touch it; with R3 (the pin's builder files
  byte-identical to stage 1b's) the pass carries to the pin.
- **What INV does not prove.** It is one constructed case per channel, not every decision. The identity guard covers
  only that the live search is the offline one (carry included), not that the offline one is leak-free on states INV
  never tried. A channel the inventory missed is not ruled out (FINDING S′H-1 stands).
- **Not leaks, kept:** the biases of §8 (functions of I, worlds uniformly weighted) and the leaf's opponent model (P\*,
  not Kakuna: S-U A2). All are part of S′h as defined.

## 8. The builder's own FINDINGs: what each means for the verdict

S′h is `r0s4_honest_inc@10` AS BUILT; every item below is a property of that search, a function of I. None is a leak,
none breaks the pairing (one result per deal, both arms on the same deal), and none can make the rule mis-measure
S′h's effect. Each limits what the verdict SAYS, and each is reported.

| finding | what it is | threatens the verdict? | why, and what the read does |
|---|---|---|---|
| **S′H-12** (correlated dice per battle) | the warm chain judges every candidate on ONE set of chain dice per battle (C22), and a carried world keeps its dice path (its hidden counters, the opponent's exact HP) for up to 4 decisions, longer under an overage | **No** | It makes a battle's successive world sets correlated, which changes what S′h is (its strength), not whether the read measures it. The unit of analysis is the deal (one paired result per game), so correlation between decisions inside a game cannot bias the interval. The measured cost: the incremental worlds move the pick no more than re-drawing `r0s4_honest`'s own worlds (63 % agreement against its 63 % floor, stage 1b). A NULL closes "search as built", never "a leak-honest search with independent worlds" (§1). |
| **S′H-13** (a segment no world matches excludes the rest of a battle) | a rare joint dice event (stage 1b: a Beat Up of three strikes, a crit on the third, a wake-up) defeats every carried, escalated, chain and cold world; every later decision of that battle is then an exclusion (π plays) | **No, for validity; yes, it dilutes the treatment** | Exclusions are part of S′h (D8) and a function of I, never a selection of deals: every deal counts on both arms. The dilution is concentrated in long F and P72 games (64+: 15.0 % and 20.5 % of decisions in stage 1b; 18 of 138 F battles and 7 of 133 P72 battles held an exclusion) and is inside the projection (U 3.4 %). The exclusion bound flags a live share above both twice it and its own error (§6); the builder report gives the stuck share and the battles with a stuck stretch per panel (§12). A stuck battle costs a few CPU-s a decision (C28), so it threatens neither the watchdog nor the wall. On a NULL with `excess_exclusions` set, the record says the arm searched materially less than registered. |
| **S′H-15** (the correlation bound is soft) | C17 draws max(2, 8 − survivors) fresh worlds, so a battle whose survivors are all old keeps them: 8.3 % of stage 1b's searched decisions held a world older than 3 decisions (548 at age 4, 98 at 5, a tail to 36) | **No** | The same reasoning as S′H-12: a property of the search, a function of I, inside the agreement already measured. **The fix is recorded and NOT built** (the brief): count the fresh worlds needed against the survivors still within the bound (8 − the young ones), so old worlds retire as soon as the chain yields replacements, at ≈ 1 fresh world's cost each (≈ 0.5–2 CPU-s late). It would be a new preset and a new registration, and a run of it after this read's outcome (a NULL above all) must say it was chosen after that outcome, as §10's third arm must. The record reports the overage share per panel (`honest.inc.overage`, §12). |
| **The first decision's cold build** | each battle's first searched decision (turn ≥ 10) is `r0s4_honest`'s own cold build at budget B3 (C25), up to ≈ 300 CPU-s | **No** | Its CPU is inside stage 1b's 10–19 bucket mean, so inside the model; at K = 14 one worker holds the decision while the peer idles behind the held keepalive (A9), as S′-LIVE's 30-s exclusions did. A cold build that fails is retried in full at the next 3 decisions, then on the doublings (C28): those exclusions are inside the projection. The identity re-score rebuilds it first for every sampled decision (its cost is in §11). |
| S′H-10, S′H-11 (fresh dice cannot reach every decision; the per-segment rejection's and the chain's common-dice weighting) | | No | Stage 1's findings, unchanged: biases and limits of S′h, functions of I, worlds uniformly weighted. |

## 9. Power (normal approximation, at both SEs)

Two SEs for G_U, both scaled to n: S′-LIVE's REALISED bootstrap SE (0.974 pp at 2,698 pairs, × √(2,698 / n)), and the
CONSERVATIVE iid SE √(d / n) at S′-LIVE's realised discordance d = 0.341 (FINDING S′HP-9: the realised one is 12 %
below the iid value with no structural cause, so planning rests on the conservative one). P(GO) = Φ(δ / SE − 1.96);
P(NULL) = Φ((3 − 1.96·SE − δ) / SE). Each cell is realised [conservative].

| deals | SE (pp) | P(GO) at S′-LIVE's +4.60 | **P(GO) at half, +2.30** | P(GO) at +1 | P(NULL) at 0 | max P(STOP) |
|---|---|---|---|---|---|---|
| 3,840 | 0.82 [0.94] | ≈ 1.00 [0.998] | 0.80 [0.69] | 0.23 [0.18] | 0.96 [0.89] | 0.10 [0.29] |
| 4,500 | 0.75 [0.87] | ≈ 1.00 [1.00] | 0.86 [0.75] | 0.26 [0.21] | 0.98 [0.93] | 0.02 [0.19] |
| **5,100 (registered)** | **0.71 [0.82]** | **≈ 1.00 [1.00]** | **0.90 [0.80]** | 0.29 [0.23] | 0.99 [0.96] | **0 [0.10]** |

- **5,100 (85 unit seeds) is the smallest whole-seed size reaching 80 % at half S′-LIVE's effect on the conservative
  SE** (≈ 5,060 needed). §3 gives the argument from stage 1b's numbers.
- **The classes at 5,100.** Conservative SE: GO-HONEST iff Ĝ_U ≳ +1.60 pp, NULL iff Ĝ_U ≲ +1.40 pp, STOP between
  (P(STOP) ≤ 0.10, its maximum near +1.5 pp). Realised SE: GO iff ≳ +1.39, NULL iff ≲ +1.61, so the two regions
  overlap and STOP is ≈ unreachable; a GO in the overlap is flagged `go_below_margin`.

  | true G_U (conservative SE) | 0 | +1.0 | +1.5 | +2.0 | +2.3 | +3.0 | +4.6 |
  |---|---|---|---|---|---|---|---|
  | P(GO-HONEST) | 0.025 | 0.23 | 0.45 | 0.69 | 0.80 | 0.96 | ≈ 1 |
  | P(NULL) | 0.96 | 0.69 | 0.45 | 0.23 | 0.14 | **0.025** | ≈ 0 |

  The nominal error rates: 2.5 % false GO-HONEST at 0, 2.5 % false NULL at +3 pp.
- **What can shift d** (§3): not the departure rate (≈ 99 % of S′-LIVE's searched games already departed); GUESS
  d ≈ 0.34–0.35 (SE ×1.00–1.01).
- **GOAL-B's meter** at this n: the searched arm's score SE ≈ 0.70 pp, so LB > 0.50 needs a point of ≈ 0.514.
- **The record reports** the achieved d, the achieved SE and the power they imply; none changes n.

## 10. The third arm (P\* + `r0s4_rolls` on the same deals): NOT played

A third arm would measure the leak's share directly, paired by deal: rolls − honest.

| option | added CPU-h (S′-LIVE's rolls unit, 8.06 CPU-min at K = 6) | the leak share's SE | power at a leak share of +2.30 pp |
|---|---|---|---|
| **none: §12's cross-read difference** (S′-LIVE's G_U − this read's, unpaired) | 0 | **1.20 pp** (realised) | 0.48 |
| a third of the deals (1,700) | ≈ 228 | 1.41–1.53 pp | 0.32–0.37 |
| half the deals (2,550) | ≈ 343 | 1.15–1.25 pp | 0.45–0.51 |
| all 5,100 deals | ≈ 685 (**+119 %** of the read; ≈ +2.1 days of the whole box) | 0.82–0.89 pp | 0.74–0.80 |

(√(d / n) at a rolls-vs-honest discordance of 0.34–0.40.) The rolls arm costs more than the honest arm it would be
compared with. A sub-sample is dominated (below ≈ 2,500 deals its paired SE is no better than the free cross-read
difference); the full arm would more than double the read for a quantity this read decides nothing on: the goal-(b)
question is the honest arm against raw. **No third arm.** If a later decision needs the paired leak share, a new
registration may play `r0s4_rolls` on these deals and must state that it was chosen after this read's outcome.

## 11. Cost and wall time (ESTIMATE; cost never re-sizes the read)

**The model** (`inputs/cost_model_inc.py` → `cost_model_inc.json`): stage 1b's per-decision CPU by panel and turn
bucket (the builder over every decision, plus the targets' forward and playouts) × S′-LIVE's search-path decisions a
game in each bucket, × S′-LIVE's K = 6 / K = 2 ratio κ by bucket (1.55–1.68), plus S′-LIVE's per-game overhead.

| panel | search-path decisions a game | projected exclusions | CPU-min a searched game, K = 2 / K = 6 |
|---|---|---|---|
| H | 36.0 | 0.29 % | 2.91 / 4.58 |
| F | 42.8 | 5.75 % | 5.45 / 8.67 |
| P72 | 36.3 | 4.02 % | 3.34 / 5.29 |
| **U** | 38.4 | **3.35 %** | **3.90 / 6.18** |

**The read:**

| item | CPU-h |
|---|---|
| 5,100 searched games (K = 6 / K = 14 CPU) | 525.4 |
| 5,100 raw one-game units (S′-LIVE: 0.60 CPU-min a unit at K = 6) | 51.1 |
| the identity re-score, 600 decisions, REBUILD (one process, a quiet box; mean 97 CPU-s a decision: F 170, H 53, P72 69; max ≈ 1,600; ESTIMATE, FINDING S′HP-7) | 16.2 |
| the checkpoint's re-score (≈ 140 decisions, rebuild) | ≈ 4 |
| **total** | **≈ 597** |

**Wall, by the box's state** (S′-LIVE life 4's whole-unit rate: 5.80 CPU-h a wall hour at K = 6, 0.97 a worker):

| shape | when | K, cores | the read's lanes (577 CPU-h) |
|---|---|---|---|
| **BOX** (registered) | no training run live, no GPU lease | **K = 14 − the K of any other read lane on the box, on cores 0–15** | **≈ 43 h at K = 14** (≈ 34 h at stage 1b's SMT-probe figure, UNVERIFIED for whole units) |
| AGENT | a training run live and PINNED to 0-4, 8-12 (X65-F1's ruling) | K = 6 on 5-7, 13-15 | ≈ 115 h (99 h × the pinned trainer's 1.153) |
| QUIET | an UNPINNED training run live | K = 2, default slots | S′-LIVE's life 3 rate, 3–5 deals an hour: weeks; in practice the lane is held |

- **The ETA on the BOX shape at K = 14** from the first unit: the checkpoint (120 units) ≈ 0.5 h, its reading and
  re-score ≈ 0.5 h, the bulk ≈ 42 h, the registered re-score ≈ 2 h (`--jobs 14`: ≈ 16 CPU-h × κ ≈ 1.6 ÷ 14, its
  longest decision ≈ 45 min), the record minutes: **≈ 45 h, ≈ 1.9 days**. Each other lane's unit on the box adds
  ≈ 1/14 of the bulk (K = 12 beside X60-T's look 1: ≈ 50 h).
- **K = 14 on 16 logical CPUs** (8 physical cores, both SMT siblings busy on 7) leaves two for the supervisor and the
  sessions. A unit is our process and its Kakuna peer, which alternate, so ≈ 1 busy CPU a unit. Memory: the smoke's
  units peaked at ≈ 2.5 GB each (searched process and peer, `mem_cap`), ≈ 35 GB at 14 against the heavy slice's 64 GB
  and the box's 89 GB (each unit capped at 12 GB; FINDING S′HP-6); a re-score worker ≈ 0.8 GB (two peaked at 1.66 GB),
  ≈ 12 GB at `--jobs 14`, run under a 24-GB cap.
- **Another read's lane on the box** (X60-T's look 1 runs at K = 2 tonight): the life script subtracts every other live
  supervisor's K from 14 (no more than 14 read units on the box) and refuses the BOX shape below K = 2; the
  orchestrator's queue decides which lane waits.
- **What the BOX shape costs the box, and how it gives way.** The read's supervisor runs with `GEN3AI_HEAVY_SLOTS=K`
  (`main.reads` clamps K to the slot count at its start). While no run is live and no lease is held, no unit takes a
  slot at all. The moment slots are wanted (a run started or a lease granted without the hold of §14), every NEW unit
  sets the BOX shape aside (`kakuna.unit_shape`): it runs on the agent cores and does NOT inherit the raised slot
  count, so it queues for the box's usual two slots behind the other agents' jobs; only the units already in flight
  (≤ K, each ≤ ≈ 1 h) keep their cores (FINDING S′HP-2).

## 12. Reported, never decided

- G per panel with its interval, from the same replicates; the three intervals (paired, GO-imputed, NULL-imputed).
- Each arm's pooled-U score over the paired games (`score_u`); GOAL-B's imputed score and LB (`goal_b`).
- The discordance share (b + c; never b − c), the achieved d and SE and the power they imply.
- Pre- and post-treatment drops by panel and cause; one-sided deals; timer kills; every exhausted unit's error class.
- **Per panel:** the deviation rate, the exclusion share against its bound (`exclusions`), the searched CPU a game.
- **The builder report** (`inputs/inc_report.py` over the counted searched units' decision logs, outcome-blind): per
  panel the share of search-path decisions in each `inc.mode` (cold, warm, stuck, recold), the battles with a stuck
  stretch (S′H-13), the share of searched decisions holding an overage world and the age distribution (S′H-15), the
  warm chain's fates, and the cold first decisions' CPU.
- **The cross-read difference** (S′-LIVE's G_U − this read's G_U, on different deals): the two paired point
  estimates, and an interval from the two SEs in quadrature (√(0.974² + SE²)), labelled UNPAIRED and SELECTED (§2). It
  is the leak's share only if the two reads' deal samples are exchangeable and S′h differs from S′ only by the leaks;
  it also differs by its builder's budget, its exclusions (the flag), its correlated worlds (§8) and the HP fix (§2).
- **The identity guard:** decisions sampled and re-scored, by panel, turn bucket and unit-seed range; the checkpoints
  rebuilt (`rebuilt`); the re-scoring checkout; any killed attempt whose torn last line was dropped (`torn_tails`).
- **The distillation report** (`distill`): fields lacking, units without a capture, per panel.

No outcome is printed while the read runs. The operator reads statuses, G-PREFIX, the identity guard, the exclusion
share, the builder report, the distillation report and `search-cost` only.

## 13. Predictions (GUESSES, before any game)

- **G_U ≈ +2.5 pp (−0.5, +5.5).**
  - The prior's top is S′-LIVE's +4.60 [+2.66, +6.48]: S′h is S′'s search with honest worlds.
  - It is shrunk for the leaks: dlive's leak-exposed departures read +2.6 pp a departure against −0.06 pp for the
    leak-clean ones (both intervals include 0); S′-LIVE's speed pins touched 23.5 % of decisions and its counters 6.9 %.
  - It is kept from 0 by stage 1's agreement: where both searched, `r0s4_honest` picked `r0s4_rolls`' action in 67 % of
    the floor's states against 72 % for `r0s4_rolls` re-drawn under another key; and stage 1b's: the incremental
    builder picks `r0s4_honest`'s action as often as `r0s4_honest` picks its own (63 %).
- **The classes:** P(GO-HONEST) ≈ 0.6; P(NULL) ≈ 0.3; P(STOP) ≈ 0.1. P(GOAL-B-REACHED) ≈ 0.35.
- **By panel:** no ordering predicted. **d ≈ 0.35.**
- **Exclusions (live, per panel):** H ≈ 0.5 %, F ≈ 6 %, P72 ≈ 4 %; `excess_exclusions` not set.
- **Cost:** the mean searched game ≈ 6–8 CPU-min of whole-unit CPU.
- **Guards:** G-PREFIX 0 mismatches; the identity guard PASS, 600 re-scored; post-treatment drops < 0.5 % a panel
  (S′-LIVE: 0.11 % at most); every distillation field present.

## 14. Operation

From the MAIN checkout's registration commit, a clean detached worktree at the pin plays. Every supervisor life is a
transient `systemd-run --user` service whose script checks the commit, the engine and the box's state before it
`exec`s the supervisor (S′-LIVE lives 3–4), so no agent session's restart can kill it (S′LIVE-OP-3). Never under
`timeout`.

**Once, at the pin (after the registration commit).** The worktree already exists (made 2026-11-06 for R1–R5,
clean at the pin); skip `worktree add` and `bootstrap.sh` if it is still there and `git status` is clean.

```bash
P=e46e3ddf82b7ee842a9104d59e72b41c36398c56
WT=/home/goodlad/dev/gen3ai-reads/sprime_honest_inc_${P:0:9}
git -C /home/goodlad/dev/gen3ai worktree add --detach "$WT" "$P"
cd "$WT" && nice -n 19 ./scripts/bootstrap.sh --skip-env --with-rust --no-check
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES=""
SPEC=/home/goodlad/dev/gen3ai/designs/research_state/measurements/sprime_honest_play_2026-11-06/read_spec.json
STATE=/home/goodlad/gen3ai_archive/sprime_honest_play_2026-11-06
E6='port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130'
"$PY" -c "from main.anchors.server import engine_stamp as e; s=e(); print(s); assert s=='$E6'"
"$PY" -m main.reads check-spec --spec "$SPEC"             # pin $P; play_path_differs_at_head must be 0
"$PY" -m main.reads plan --spec "$SPEC" --state "$STATE"   # 10,200 units; plays nothing
mkdir -p "$STATE/ops" "$STATE/identity"
```

**A life's script** (`$STATE/ops/life<n>.sh`; its first two lines are the life's own):

```bash
#!/bin/bash
VERB=(resume --by sprime_honest_operator --why "life <n>: <the reason>")   # life 0 (the checkpoint): VERB=(run --max-units 120)
SHAPE=box                                                                   # box | agent | quiet (§11)
set -u
P=e46e3ddf82b7ee842a9104d59e72b41c36398c56
WT=/home/goodlad/dev/gen3ai-reads/sprime_honest_inc_${P:0:9}
STATE=/home/goodlad/gen3ai_archive/sprime_honest_play_2026-11-06
E6='port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130'
cd "$WT" || exit 90
[ "$(git rev-parse HEAD)" = "$P" ] || { echo "[life] the checkout is not the pin: refusing"; exit 94; }
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
       OMP_NUM_THREADS=1 PATH=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin:/usr/local/bin:/usr/bin:/bin
case "$SHAPE" in
  box)
    w=$("$PY" -c "from utils.heavy_slots import wanted; print(wanted() or '')") || exit 95
    [ -z "$w" ] || { echo "[life] the box is not free ($w): refusing the box shape"; exit 96; }
    # every OTHER live read supervisor's K (X60-T's look 1, say) is subtracted: <= 14 read units on the box
    OTHER=$("$PY" -c "from main.reads import supervisors as SV; from utils import procfs
me = SV.canonical('$STATE'); k = 0
for pid, ln, st in SV._scan('/proc', ()):
    if st != me:
        a = [t for t in procfs.read_text(pid, 'cmdline', '/proc').split(chr(0)) if t]
        k += int(a[a.index('--concurrency') + 1]) if '--concurrency' in a else 1
print(k)") || exit 97
    K=$((14 - OTHER)); CPUS=0-15
    [ "$K" -ge 2 ] || { echo "[life] other lanes hold $OTHER of 14: refusing the box shape"; exit 98; }
    export GEN3AI_HEAVY_SLOTS=$K GEN3AI_ANCHOR_CORES=0-15 ;;
  agent) K=6; CPUS=5-7,13-15; export GEN3AI_HEAVY_SLOTS=6 ;;
  quiet) K=2; CPUS=5-7,13-15 ;;
  *) echo "[life] unknown shape $SHAPE"; exit 93 ;;
esac
s=$(nice -n 19 taskset -c "$CPUS" "$PY" -c "from main.anchors.server import engine_stamp as e; print(e())") || exit 91
echo "[life] engine $s; shape $SHAPE, K $K on $CPUS"
[ "$s" = "$E6" ] || { echo "[life] ENGINE NOT E6: refusing"; exit 92; }
exec nice -n 19 taskset -c "$CPUS" "$PY" -m main.reads "${VERB[@]}" --state "$STATE" --lane anchor --hours 72 \
    --concurrency "$K"
```

```bash
# launch life <n> (never under `timeout`; the service outlives any agent session)
systemd-run --user --unit "sprime-honest-inc-life<n>" --collect -p StandardOutput=append:"$STATE/ops/life<n>.log" \
    -p StandardError=append:"$STATE/ops/life<n>.log" bash "$STATE/ops/life<n>.sh"
PID=$("$PY" -m main.reads pid --state "$STATE" --lane anchor --wait 180)
timeout 3300 bash -c "while [ -d /proc/$PID ]; do sleep 30; done"     # ONE bounded watcher; re-arm on a timer wake

# at every wake: the lane, the exhaustions, the box
"$PY" -m main.reads status --state "$STATE"                           # holds, progress, exhausted per label
"$PY" -c "from utils.heavy_slots import wanted; print(wanted())"     # None = the box is free (SHAPE=box may stand)
"$PY" -c "from utils.procfs import live_training_runs as r
for d, p in r().items(): print(d, [l for l in open(f'/proc/{p}/status') if l.startswith('Cpus_allowed_list')])"   # each live run's cores

# at the checkpoint (life 0 exits after 120 units): cost, exclusions, builder, distillation, the re-score
"$PY" -m main.reads search-cost --state "$STATE" > "$STATE/ops/checkpoint_cost.json"
"$PY" -c "import json,collections as C; d=json.load(open('$STATE/ops/checkpoint_cost.json')); a=C.defaultdict(lambda:[0,0])
for g in d['per_game']:
    p=g['cell'].split('__')[1]; a[p][0]+=g['excluded']; a[p][1]+=g['searched']+g['excluded']
E=sum(e for e,n in a.values()); N=sum(n for e,n in a.values())
print({p:(e,n,round(e/n,4)) for p,(e,n) in a.items()}, 'pooled', E, N, round(E/N,4))"   # panels REPORTED; pooled > 0.14 holds
"$PY" /home/goodlad/dev/gen3ai/designs/research_state/measurements/sprime_honest_play_2026-11-06/inputs/inc_report.py \
    "$STATE/anchor/u/pstar_s/cells" > "$STATE/ops/checkpoint_inc_report.json"
RESCORE=(-m main.value_search.live_identity rescore --cells "$STATE/anchor/u/pstar_s/cells"
         --panels heldout72_hpfix,forum72,p72 --model "$STATE/ckpts/pstar_s/checkpoint_280000170_steps.zip"
         --model-sha256 814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed --engine "$E6" --carry rebuild)
# the box only while it is free (no run, no lease), less every other live read lane's K (as the life script); else 5b
OTHER=$("$PY" -c "from main.reads import supervisors as SV; from utils import procfs
k = 0
for pid, ln, st in SV._scan('/proc', ()):
    a = [t for t in procfs.read_text(pid, 'cmdline', '/proc').split(chr(0)) if t]
    k += int(a[a.index('--concurrency') + 1]) if '--concurrency' in a else 1
print(k)")
if [ -z "$("$PY" -c "from utils.heavy_slots import wanted; print(wanted() or '')")" ] && [ $((14 - OTHER)) -ge 3 ]
then RS=(0-15 $((14 - OTHER))); else RS=(5-7,13-15 3); fi
nice -n 19 taskset -c "${RS[0]}" scripts/ops/mem_cap.sh 24 "$PY" "${RESCORE[@]}" --jobs "${RS[1]}" --max 600 --min 1 \
    --out "$STATE/identity/checkpoint.json"
"$PY" -c "import json; from main.reads.rules_searched_play import distill_cells as d
print(json.dumps(d('$STATE/anchor/u/pstar_s/cells', honest=True), indent=1))"   # decision logs + capture dirs only

# when every unit is counted or exhausted: the registered re-score, then the record
# (RS re-read the same way first)
nice -n 19 taskset -c "${RS[0]}" scripts/ops/mem_cap.sh 24 "$PY" "${RESCORE[@]}" --jobs "${RS[1]}" --max 600 --min 300 \
    --out "$STATE/identity/rescore.json"
"$PY" -m main.reads record --state "$STATE"
```

- **The flags are the CLI's.** Check them with `--help` at the pin before the first unit; a refusal is never worked
  around. Use the model path `plan` reports.
- **The checkpoint reads no result.** `search-cost`, `inc_report.py` and `distill_cells` read decision logs, capture
  dirs, summaries' status and units' CPU; `decide` and `record` are NEVER run before the lane ends.
- **Watching.** One bounded watcher on the supervisor's own PID (`main.reads pid`, never `$!`, never a name match), via
  `[ -d /proc/<PID> ]`; on a timer wake one `status`, the HOLD checks of §6 and the box check, then re-arm.
- **STOPPING (any reason, a training run that must resume included).** `"$PY" -m main.reads hold --state "$STATE"
  --lane anchor --by <who> --why "<reason>"`: the supervisor starts no new unit, lets the ones in flight finish (a
  searched unit ≈ 5–15 min of wall, the longest ≈ 1 h; a raw one minutes) and exits; the watcher wakes when its PID
  is gone. **A training run starts only after that** (the orchestrator's queue). Units killed instead (an emergency:
  `systemctl --user stop` of the read's own `gen3ai-capped-x35kakuna-*` scopes, by name from `systemctl --user
  list-units`) are not failed attempts: with no `end` event, `resume` sets each cell aside as `.partial.<epoch>` and
  replays it whole (S′LIVE-OP-2 / OP-3). A stop costs wall time, never validity. **The safety net:** a unit launched
  while the heavy slots are wanted (a run live, a lease held) sets the BOX shape aside (`kakuna.unit_shape`): the agent
  cores, and the box's own two slots (the life's raised count is not inherited), so a run started without the hold
  meets the units already in flight (≤ K, each ≤ ≈ 1 h) and at most two new ones; the operator holds at the next
  wake (≤ 55 min).
- **RESUMING.** The next life, `VERB=(resume …)`, with the shape the box allows: `box` with no run live and no lease
  (the script refuses `box` otherwise); `agent` beside a run PINNED to 0-4, 8-12 (every live trainer's
  `Cpus_allowed_list` within `0-4,8-12`); `quiet` beside an unpinned run. K and the cores are scheduling only: the same
  units, seeds and records (`anchor_pool_test`).
- **Surviving a logout** (`Linger=no` on this box, FINDING S′LIVE-OP-4). A `systemd-run --user` service survives any
  agent session's restart but NOT the stop of `user@1000`, which systemd does when the user's LAST login session
  closes. If a life dies anyway (the watcher wakes with `/proc/<PID>` gone and `status` shows no live supervisor), the
  operator starts the next life with `resume`; a cut-off unit is replayed whole. The fix is the owner's
  (`loginctl enable-linger goodlad`); no agent changes it (FINDING S′HP-3).
- **The order:** the orchestrator's queue. On the BOX shape the read takes the box for ≈ 1.9 days (§11).

## 15. Measured before registration (outcome-blind; outside the read), and the registration checks

**Stage 1b** ([`../sprime_honest_inc_stage1b_2026-11-06/README.md`](../sprime_honest_inc_stage1b_2026-11-06/README.md),
at `f7da935ab`, E6): the incremental builder over 407 battles' whole sequences (9,758 decisions, 444 targets): the
per-bucket cost and exclusions of §11; agreement 63 % with `r0s4_honest` (its own floor 63 %); the leak test in sequence
mode identical on every channel; `r0s4_honest` (tooth 24 / 24) and `r0s4_rolls` (600 / 600) unchanged at E6.

**The cost model** ([`inputs/cost_model_inc.py`](inputs/cost_model_inc.py),
[`inputs/cost_model_inc.json`](inputs/cost_model_inc.json)): §11. Its S′-LIVE inputs are `cost_model.json`'s
(`inputs/cost_model.py`, written for the draft on `r0s4_honest`: its own per-decision figures are superseded).

**Identity references from before the HP fix are history, never this read's evidence** (the coordinator,
2026-11-06). P\*'s outputs changed at `83272a8f9`, so every identity record made before it (the draft's
`inputs/smoke_rescore.json` and `inputs/stage1_tooth.json` on `r0s4_honest`, stage 1b's tooth and its 600 / 600
`r0s4_rolls` re-score, the pre-fix smoke's own re-scores below) is re-derived at the pin or not cited as identity
evidence. The read's guard is post-fix by construction (live play and the offline re-score both at the pin); its
reference before the read is the smoke AT THE PIN.

**The pre-fix smoke** (R4's input only) ([`inputs/smoke_inc_spec.json`](inputs/smoke_inc_spec.json); state
`~/gen3ai_archive/sprime_honest_inc_smoke_2026-11-06/`; home72 seeds 200–201, game 0, both arms; 2026-11-06 at
`21b3140c1`, an earlier local form of the S′H-14 unit, before the HP fix: its play path differs from the shipped unit's
only in `kakuna.py`'s unit shape; K = 2 on the agent cores beside a live training run). Outcome-blind:
- **Play.** 4 units, all counted; G-PREFIX 2 / 2 identical. The searched games ran 40 and 64 turns: 95 search-path
  decisions, 87 searched, 8 excluded (game 201: one stuck stretch, 6 decisions in `stuck` mode), departures 37.9 %.
  Searched-unit CPU 2.2 and 10.9 CPU-min (`search-cost`); each unit (searched process and peer) peaked at ≈ 2.5 GB.
- **The builder** (`inputs/inc_report.py`): 2 cold first decisions at 5.5 CPU-s (max 7.6); the warm chain fit 73,
  refit 12, failed 2, skipped 6; 2 of 87 searched decisions held a world past the bound (age 4).
- **The identity guard, both carry modes**: 3 sampled decisions, one of them an EXCLUDED decision (t47) inside game
  201's stuck stretch. REBUILD (`--jobs 2`, spawned workers): PASS 3 / 3, 70 checkpoints rebuilt, 355 CPU-s, peak
  1.66 GB for both workers. SEEDED: PASS 3 / 3, none rebuilt, 215 CPU-s (t47's full attempt alone 211).
- **The rule end to end** (`main.reads record` on the smoke's own spec, `searched_play` on home72, not a result):
  identity PASS with no problems (carry `rebuild` as registered), every distillation field present (87 searched and
  8 excluded records, 0 lacking, both captures), exclusions 8 / 95 under the smoke's 0.2 bound, no VOID reason.

**The smoke AT THE PIN** ([`inputs/smoke_inc_pin_spec.json`](inputs/smoke_inc_pin_spec.json), a PINNED spec; state
`~/gen3ai_archive/sprime_honest_inc_smoke_pin_2026-11-06/`; the same two home72 deals, both arms, played at the
candidate pin `74a6db17e` (S′H-14 on `83272a8f9`; the shipped pin differs from it in one test file: the same play
path and re-score code), re-scored in both carry modes at the pin, and its record run through the rule, whose
`_rescore_at_pin` check then runs for real). Outcome-blind:
- **Play.** 4 units, all counted; G-PREFIX 2 / 2 identical; 95 search-path decisions, 87 searched, 8 excluded (game
  201's stuck stretch), departures 37.9 %; searched-unit CPU 2.4 and 10.2 CPU-min; cold first decisions 4.3 CPU-s
  (max 5.3); the builder's modes, chain fates and ages as the pre-fix smoke's.
- **The identity guard at the pin `e46e3ddf8`, both carry modes**: REBUILD PASS 3 / 3, 0 mismatches, 70 checkpoints
  rebuilt, 281 CPU-s; SEEDED PASS 3 / 3, 186 CPU-s; each records its checkout, the pin, clean; no torn tail.
- **The rule end to end at the pin** (`main.reads record`, `searched_play` on home72, not a result): identity PASS
  with no problems (the carry as registered, and `_rescore_at_pin` accepting a re-score at `e46e3ddf8` for a spec
  pinned to `74a6db17e`: the same play path and re-score code), every distillation field present (0 lacking, both
  captures), exclusions 8 / 95 under the smoke's 0.2 bound, no VOID reason. **This is the read's identity reference.**

**The registration checks, at the pin, in order** (each passes or the registration stops and is reported):
- **R0.** The H panel (FINDING S′HP-1): **PASSED** (the orchestrator, 2026-11-06). The owner's "no human replay data"
  constraint is the owner's rule for rented or cloud machines, over-broadened in the brief; on this box H
  (`heldout72_hpfix`) is the standing evaluation panel (`designs/research_state/heldout_panels.json`) and was in
  S′-LIVE's U, which this read replicates.
- **R1.** The pin holds the HP fix and S′H-14 (`git merge-base --is-ancestor`), and its worktree is clean. At the
  pin `e46e3ddf8`: **PASSED** (holds `83272a8f9`; config v165, `OP_SEMANTICS` `gen3_our_hp_exact_v1`; the pin's
  worktree `/home/goodlad/dev/gen3ai-reads/sprime_honest_inc_e46e3ddf8` clean).
- **R2.** `engine_stamp()` at the pin is E6: **PASSED**.
- **R3a.** `compare("f7da935ab", "335355e6f")` = `honest_inc.py` only, and that file's two versions have equal ASTs with
  docstrings stripped: **PASSED 2026-11-06** (§4).
- **R3b.** `main.reads.playpath.compare("335355e6f", PIN)`: the differing files are the HP fix's under
  `src/agents/model/` and `src/main/x35/kakuna.py` only (§4): **PASSED** (exactly the 9 HP-fix files and
  `kakuna.py`; the candidate `74a6db17e` the smoke played at has the same play path, `[]`).
- **R4.** The smoke's re-score at the pin, REBUILD mode: 0 mismatches on `searched`, `why`, `search_key`, `honest`;
  mismatches on `played`, `q`, `log_pi`, `root_v` counted and reported as the HP fix's. A `root_guard` mismatch from
  the search's own forward check (the HP fix moving π's argmax off the recorded `pi_argmax`) is the HP fix's too, and
  that decision's builder fields are then UNVERIFIED (named). Any builder-field mismatch STOPS the registration. At the
  candidate (rebuild, `--jobs 2`): PASS 3 / 3, 0 mismatches on every field (no network field moved on these three),
  70 checkpoints rebuilt, 296 CPU-s. **At the pin `e46e3ddf8`: PASS 3 / 3, 0 mismatches on every field**, 70 rebuilt,
  296 CPU-s, checkout the pin, clean. No HP-fix difference to count on these three.
- **R5.** The spec carries `"pin": "e46e3ddf8"` (`check-spec` accepts a spec with no pin, so this is read, not
  assumed); `main.reads check-spec` at the pin: OK, `pin` = `e46e3ddf82b7…`, `is_head`, `play_path_differs_at_head` 0,
  `sha256_final` true, spec sha256 `9222393a14c79ff52e2a2bd315864626a9f6ea68ffeea709352c40e79d618afa`; `plan` into a
  scratch state at the pin: 5,100 units a label (10,200), unit seeds 700–784, `pstar_s` all `r0s4_honest_inc@10`,
  both models `814823c514fb…`, its `code.json` the pin, clean; the scratch state deleted. **PASSED.**

## Registration

- Pin `e46e3ddf8`. `check-spec`: `play_path_differs_at_head` 0. The spec's sha256 is
  `9222393a14c79ff52e2a2bd315864626a9f6ea68ffeea709352c40e79d618afa` (with its pin; a change to the spec moves it).
- Engine E6. Preset `r0s4_honest_inc@10`. Deals: U unit seeds 700–784, 5,100. Identity: REBUILD, the first 600.
- State `~/gen3ai_archive/sprime_honest_play_2026-11-06/`.
- Committed with [`read_spec.json`](read_spec.json) before any game.
- Written by the S′-HONEST play designer (orchestrator brief, 2026-11-06), ported from the unregistered draft for
  `r0s4_honest@10` (local branch `sprime-honest-prereg`, `a9c9f6499`). One fresh independent review (§ Review) and its
  verification were folded in before registration; the coordinator accepted the 5,100 deals and resolved R0.

## Review (before registration)

**REVIEW A** (one fresh, independent opus-high review, 2026-11-06, given the code, the data and the docs; a statistics
lens and a leak / engineering lens). Verdict: one BLOCKER and five MAJOR findings, no leak path, no validity flaw in the
rule. Every number it recomputed held (the §9 power at both SEs, the cut-points, §10's SE, the bounds' arithmetic, the
projections, the cost per game, the totals, the rebuild estimate, the sampled count, the watchdog); it agreed with §8
that none of S′H-12, S′H-13, S′H-15 or the cold first build threatens validity; it found the carry a function of I, the
re-score's store cleared per state in every worker, the HP fix outside the builder's import closure, every leak path
closed, the rule backward compatible with S′-LIVE's record, and every §14 flag real. Dispositions:

| # | finding | disposition |
|---|---|---|
| B1 | R3 as written fails: `335355e6f` changed `honest_inc.py` (docstring and a comment) after stage 1b's `f7da935ab`; §4 said its play path was kept | **FIXED.** §4 and §15 R3a / R3b: the reference is `335355e6f`, and its `honest_inc.py` change is shown docstring-only (equal ASTs, PASSED) |
| M1 | the checkpoint's per-panel exclusion hold fires on noise (P ≈ 0.13 F, 0.25 P72 at 20 games), and the projection's own error (F [1.4, 13.0] %) can set the read's flag by itself | **FIXED.** `inputs/excl_bounds.py`: the bound is the larger of 2× the projection and its 95 % upper end (F 0.12 → 0.13; H 0.05, P72 0.09 unchanged), and the flag's meaning reworded; at the checkpoint the panels are reported and only the POOLED share above its 99th percentile (0.14) holds (§6) |
| M2 | the power rests on a realised SE 12 % below the iid value, unexplained | **FIXED.** Not structural (team draws are with replacement); §3 / §9 plan on the conservative SE and register **5,100 deals** (85 seeds), affordable only because of stage 1b's cost, +10.5 h of the box; §9 gives both SEs |
| M3 | the BOX shape breaks rule 5b: another lane's K is not subtracted, and the raised slot count leaks to every unit, so a run started mid-life meets up to 14 new heavy units | **FIXED.** The life script subtracts every other live supervisor's K and refuses below 2; `kakuna.unit_shape` (in S′H-14's unit) drops the raised `GEN3AI_HEAVY_SLOTS` from a unit launched while slots are wanted, so it queues for the box's two; §11 and §14 restated |
| M4 | §6 stated the smoke's result before it existed; no sampled decision followed a recovered stuck stretch | **FIXED.** §15 holds the measured smoke (both modes, 3 / 3); §6 says what it did not cover and that the checkpoint's ≈ 140-decision re-score is the first wide coverage |
| M5 | nothing checks the re-score ran at the pin | **FIXED in code** (S′H-14's unit): `live_identity` records its checkout (`inputs.code`); `searched_play` VOIDs a pinned leak-honest read whose re-score checkout is missing, dirty or not the pin's play path (§5, §6) |
| m1 | a torn last line in a killed attempt's `.partial` log would stall the read (exit 2 forever) | **FIXED in code**: that one torn, newline-less last line is dropped and named (`torn_partial_tails`); anything else still refuses (§6) |
| m2 | R4 misses a `root_guard` from the forward check; 3 decisions carry little weight | **FIXED.** R4 registers that case as the HP fix's with its builder fields UNVERIFIED; §4 says R3b carries the weight |
| m3 | §2 understated the HP fix and guessed where the changelog measured | **FIXED.** §2 cites the changelog's P\* measurement (0 top-action changes in 16,785 rows) |
| m4 | R5 passes a spec with no pin | **FIXED.** R5 requires `pin` = PIN and `sha256_final` |
| m5 | the 6-h exhaustion rule was set at K = 6 | **KEPT** as the standing rule (the brief names it); §6 states it trips ≈ 2.3× as often at K = 14, and a hold is only a pause (FINDING S′HP-10) |
| m6 | "the same search" overclaims | **FIXED.** §2: indistinguishable by pick agreement at that n, not equal strength |
| m7 | §3 compared departure rates from different sets | **FIXED.** Stage 1b's 38.6 % against S′-LIVE's live 36.6 % |
| m8 | the re-score commands take the box with no check | **FIXED.** §14 picks the box (`--jobs 14`) only while `wanted()` is empty, else `--jobs 3` on the agent cores |
| m9 | GOAL-B's Wilson interval treats games as independent | **STATED** in §5: a claim about these panels' teams |
| m10 | a later run of S′H-15's fix must say it was chosen after the outcome | **STATED** in §8 |

**REVIEW A, its verification of the fixes** (the same reviewer, the code at the shipped unit and this PREREG): every
finding FIXED or properly disposed; the 5,100 argument correct (n ≥ 5,059 needed; 84 seeds fall short); the §14
script correct as bash (its `OTHER` read X60-T's look 1 as 2, so K = 12; a `--concurrency=N` spelling fails closed).
Four new MINOR items, all fixed: N1 §9's 4,500 row (max P(STOP) 0.02 at the realised SE, not 0); N2 the rule now also
requires the re-score's off-path code (`live_identity.py`, `sign_gate.py`) byte-equal to the pin's; N3 the torn-tail
tolerance now covers an uncounted unit's last attempt, which is never renamed `.partial`; N4 §14's re-score shape
subtracts the other lanes' K as the life script does.

## FINDINGs

- **S′HP-1 (the H panel and "no human replay data": RESOLVED, H stays).** H's 72 teams were extracted from Metamon's
  ladder-replay archive (`hl_05_26`), hidden sets filled from usage; no human decision enters the read. The
  orchestrator ruled (2026-11-06) that the constraint is the owner's rule for rented or cloud machines, over-broadened
  in the brief; on this box H is the standing evaluation panel and was in S′-LIVE's U (R0).
- **S′HP-2 (the BOX shape and the box's other heavy jobs).** `main.reads` clamps K to the heavy-slot count, so the BOX
  life raises `GEN3AI_HEAVY_SLOTS` to K. With no run live and no lease no unit takes a slot; once slots are wanted, new
  units drop the raised count and queue for the usual two (`kakuna.unit_shape`), but the ≤ K units in flight keep
  their cores until they finish. Accepted for the read's ≈ 1.9 days (the owner gives it the box).
- **S′HP-3 (a logout stops the lane).** `Linger=no` (S′LIVE-OP-4): stopping `user@1000` stops every life. A stop costs
  wall time, never validity (§14); the fix is the owner's.
- **S′HP-4 (the unit core set was hard-coded).** `main.x35.kakuna.CORES` pinned every unit to `5-7,13-15`, so a K = 14
  supervisor would have crowded 14 units onto 6 logical CPUs whatever its own `taskset`. S′H-14's unit added
  `kakuna.unit_shape`. `kakuna.py` is on the play path, so the pin's play path differs from stage 1b's there (R3b);
  it changes scheduling only.
- **S′HP-5 (the HP fix is in the read but not in the evidence).** Stage 1b, the leak test's runs, the smoke and S′-LIVE
  ran before it. The builder is untouched (R3b, R4), but S′h's values and P\*'s raw play move on the rows it fixes
  (0 top-action changes in its own 16,785-row measurement): the cross-read difference against S′-LIVE carries it.
- **S′HP-6 (the BOX shape's memory at 14 units is measured on two).** ≈ 2.5 GB a unit on the smoke, ≈ 35 GB at 14
  against the 64-GB heavy slice. An OOM inside the slice fails an attempt (exit 86) and the unit retries; a unit that
  exhausts with no summary is §5's fail-closed path.
- **S′HP-7 (the re-score's REBUILD cost is an ESTIMATE).** ≈ 16 CPU-h from stage 1b's bucket means over S′-LIVE's game
  structure (the smoke's 3: 355 CPU-s); the checkpoint's re-score measures it on ≈ 140 live decisions first.
- **S′HP-8 (another read's lane shares the box).** X60-T's look-1 anchor lane runs at K = 2 tonight; the smoke took
  heavy slots beside it for ≈ 15 min. The BOX shape subtracts any such lane's K (§14).
- **S′HP-9 (S′-LIVE's realised SE is 12 % below its iid SE).** 0.974 against 1.118 pp (REVIEW A M2), with team draws
  WITH replacement, so no stratification explains it; likely chance (≈ 2 SE of the SE estimate). The read plans on the
  conservative SE (5,100 deals) and reports its own achieved SE.
- **S′HP-11 (the training session's standing GPU lease blocks the BOX shape).** `utils.heavy_slots.wanted()` keys rule
  5b on a held GPU lease as well as a live run, and the training session holds a standing lease (owner 'Training Oct',
  to 2026-11-10). With it held and no run live, the life script refuses `box` and `kakuna.unit_shape` sets the box
  aside for every new unit, so the read takes the whole box only once that lease is released; until then it runs the
  AGENT shape (K = 6), and a later life moves it to the BOX shape (hold, drain, resume).
- **S′HP-10 (the exhaustion hold at K = 14).** "More than 2 exhaustions in 6 h" was set beside K = 6; at K = 14 it
  trips ≈ 2.3× as often at the same per-unit fault rate. Kept as the standing rule; a hold is a pause.
