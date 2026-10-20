# L3: ADVERSARIAL REVIEW, with amendments to stage 1c (2026-10-20)

This reviews the L3 read (pre-registered `4d149799`; read `9f2a283d`; [`README.md`](README.md),
[`PREREGISTRATION.md`](PREREGISTRATION.md), [`l3_run.py`](l3_run.py), [`l3_read.py`](l3_read.py),
`src/main/value_search/opp_likelihood.py`, the template in `src/main/value_search/worlds.py`). It also reviews stage 1c's
pre-registration ([`../search_1c_2026-10-20/PREREGISTRATION.md`](../search_1c_2026-10-20/PREREGISTRATION.md),
`c9bdc928`), before that unit's read. The review is docs-only: code and rows were read, nothing was re-run. The
amendments in §3 went to the orchestrator by message before this file shipped.

## Verdict on L3: STANDS-WITH-CAVEATS

The read is clean. I found no leak in the code: the opponent model sees nothing of the TRUE world that an honest world
cannot supply. The gates are real, and the registered rule was applied as written. But the headline does not
transfer as stated:

- **"The opponent's own past choices pick out the true world"** holds for THIS bank: pool teams, and half of the
  states against greedy copies of our own network.
- **"0.249 is the range the dose curve says search needs"** compares two different quantities and should not steer
  anything (F3).

The claim steers a research path, so under the plateau rule it stays **PROVISIONAL**. It was read on one pre-plateau
anchor (225M; X26 plateaued near 280M) and on one pool-opponent bank. Stage 1c measures the right thing (nearness on
the binding facts), so L3's caveats cost little IF 1c's amendments land.

## 1. The leak audit (the brief's first attack): no leak found

| path checked | finding |
|---|---|
| Template-filled moves scored? | **No.** `prepare_root` turns every opponent move the searched side never saw that mon use into a `("MOVE", k, slot)` tuple, and every switch to a never-seen mon into a `("SLOT", i)` tuple. `opponent_steps` marks every tuple step `unobserved` (excluded). A revealed Hidden Power is a tuple too, so it is excluded as well (conservative). Tokens are NAME-based (`move <id>`), so a scored token means the same move in every world. |
| An unrevealed true move scored by accident (e.g. an `attribution_miss` kept as the truth's token)? | Guarded. `_opp_inputs` raises unless the scored token is exactly one LEGAL token in every world. A world lacking a true unrevealed move would raise, and with 8 worlds per state some would. **0 defects across 283 states** is strong indirect evidence that no hidden move was scored. |
| Post-root information? | **No.** `truth_log` appends the post-root commands, but `_replay` replays `log_commands(lg, root.at)` only. The root-round choice, when the opponent chose first, is `pending` (excluded, 1 case). |
| The opponent's rows carry our side's view? | No. The rows are read from the opponent's seat (`trackers[root.oi]`). Our team is the truth in every world and the command log is shared, so the opponent's view of us is the same across worlds. |
| The truth replays on natural dice, worlds on forced damage rolls | Not a discriminating channel. Our side's rows match the truth's exactly in every world (the guard). What differs is the opponent's own exact HP and stats, which are legitimate consequences of the world's hidden spread. |
| The shared true seed and hidden counters (world residuals) | Shared by the truth and every world. They help worlds pass the gate; they cannot favour the truth under L. Declared. |
| Encoder drift since the pin | G4: 283 / 283 states byte-identical at `9c033489`. |

## 2. Findings, ranked by severity

**F1 (HIGH): the snapshot stratum is a near-oracle, and it carries the pooled WORKS.**
- The pooled AUC 0.692 is the equal-weight average of the snapshot stratum (0.796) and the bot stratum (0.588).
- The snapshot opponents are SENTINELS: earlier X26 snapshots, **GREEDY by default** (`designs/training/eval_rules.md`).
  So every observed action is the ARGMAX of a network close to the 225M scorer, computed on exactly the truth's rows.
  Under this model the truth's rows are the inputs that produced the action. A world that changes any hidden fact can
  move that argmax.
- Alone, the bot stratum is **WEAK** (lower bound 0.536 < 0.555). By L3's own rule for WEAK, it would NOT have built
  1c: its `lift_moves` is −0.005 [−0.019, +0.007], not > 0. Its exact-moves lift is even clearly negative.
- The pooled primary was registered in advance, so the verdict is procedurally valid. But the pooled number describes
  no real population. A ladder human is neither a greedy copy of our network nor a scripted bot. The bot figure (or
  something below it) is the more defensible prior. That is UNVERIFIED: no off-pool or human bank exists.

**F2 (HIGH): pool familiarity is not ruled out by the familiarity control.**
- The control (AUC_conf 0.481) shows the network is not SURER on the truth's rows. It does not show the network
  PREDICTS BETTER on them for reasons that vanish off-pool.
- X26 played these exact pool teams for about 225M steps, on both sides (sentinels draw the trainee's own teams). Its
  policy on a pool team's rows encodes TEAM-SPECIFIC learned play, which the earlier snapshot shares. On an honest
  world the opponent's own team is an off-pool Smogon draw, and those habits do not apply.
- That mechanism separates the truth from honest worlds through "pool-ness", and it is also exactly what AUC_regret
  measures. Against a ladder team none of it exists.
- **Kill test (not run):** score a pool DECOY in place of the truth. A decoy is a pool team consistent with the
  searched side's log that is NOT the true team; stage 1's C_pool worlds on these states are such decoys when they
  miss. If decoys also beat honest worlds, part of the AUC is pool-ness, not truth.
- The off-pool replication (amendment A4) settles it for search.

**F3 (HIGH): p_truth = 0.249 is not the dose curve's p, and should not be quoted as "the range search needs".**
- The dose curve's p is the weight search puts on the true world INSIDE the worlds it plays. L3's p_truth is the
  truth's weight when it is inserted among 8 distant honest worlds. No honest search ever holds the truth: the
  exact-world mass is ≤ 0.0025 per draw.
- p_truth also scales with the candidate count. Against 128 proposals the same likelihood gaps would leave the truth
  far less than a quarter, unless it dominates them.
- The quantity search needs is the exact-world mass after resampling (1c's q_SM), not p_truth.
- This sentence is repeated in `design_world_sampler.md` §4 (f), `search_ledger.md` §1 and `UNDERSTANDING.md`
  ("Among nine candidates it would give the truth a quarter of the weight, the range the dose curve asks for"). It
  should be withdrawn or labelled COUNTERFACTUAL there. Not fixed here, per the brief.

**F4 (MEDIUM): AUC vs 8 honest worlds measures discrimination of the WHOLE truth, not nearness on the binding
facts.**
- A world differs from the truth in every hidden fact: species and moves, but also items and spreads. The opponent's
  own exact HP and stats are on its rows. Part of the AUC may come from the non-binding classes (item, spread), which
  search does not bind on.
- That is consistent with S4 (re-weighting moves the binding facts ≈ 0).
- The decision-relevant L3 metric was S4: does L order honest PROPOSALS by nearness? It was NOT DETECTED pooled.
- **Decomposition test (not run):** score the truth with ONE class replaced by a world's (its unrevealed moves only,
  its never-seen species only, its spread only). That says which facts L can see.
- The WORKS → build-1c consequence is still reasonable, because 1c's primary (ΔH) is truth-anchored on the binding
  facts.

**F5 (MEDIUM): one pre-plateau anchor, one pool bank, and 1c designed on the same states.** The 225M anchor predates
X26's plateau (~280M). Stage 1c's design (N, the corpus pairing, the tempered descriptive) was chosen after reading L3
on the SAME 287 states. Neither is a defect. Together they are why any 1c positive must replicate off-bank (A4).

**F6 (LOW): the S5 dose-response is weak** (Spearman +0.13; the 5–9 bin is below the 1–4 bin). A few early choices
carry most of the signal. That fits F1's picture: the snapshot's argmax on the lead / early turns identifies the
team's own habits. It is weaker evidence of an accumulating inference from play.

**F7 (LOW, process, already self-reported):** README finding 6, a blanket `pgrep`/`kill` of 14 timers, broke rule 2.
No run was touched.

**What holds:**
- The gates (G1 to G4) are real.
- The tie band is applied deterministically.
- The bootstrap is battle-clustered.
- The read script's sha matches its registration.
- The truth goes through the same deduplicated forward as the worlds.
- The README already names hazards 1 and 4 as findings.
- The direction of the conclusion, "a judge with nothing near the truth to pick", is the right reading.

## 3. Amendments to stage 1c's PREREGISTRATION (sent to the orchestrator before this shipped)

All are computable from rows the 1c driver already stores (`near4`, `log_w`, `ll`, `facts` per proposal). None needs
a rebuild. They must be registered (dated, the original text unedited) BEFORE the interim read or any meter is
printed.

**A1 (CRITICAL, leak): a STRICT read of every corpus arm.**
- `run_1c.py` builds C_ladder with `strict=False`. The bank's truths are pool teams, and `design_world_sampler.md` F3
  requires any corpus read on pool opponents to drop near copies of the TRUE team.
- L concentrates its weight on a correct, familiar near copy (F2). So a few corpus near copies can produce NEARER,
  DOSE and even SEARCH-PAYS that a ladder opponent could not.
- Re-read C128L, M256L, C64L, M128L, C128d and C8 with every `near4` proposal's weight set to 0 and the rest
  renormalised. A state whose set is all near4 becomes a STRICT exclusion, counted.
- The corpus arm's phase-A VERDICT is the STRICT read, with the non-strict one beside it.
- Report three numbers: the share of states with at least one near4 proposal, the weight L puts on them (the leak
  mass), and ΔH on the states with none.
- Phase B resamples a corpus arm from its STRICT weights.

**A2 (HIGH): attribution.**
- ΔH vs F8 mixes N, the dice weight and L (and the corpus, for C). Read the claim "the opponent's choices move the
  worlds nearer" only on **F128L − F128d** and **C128L − C128d** (STRICT): co-primary for attribution, same NEARER rule.
- Phase B: when F128L is NEARER, also play **F128d**; when C128L is NEARER, also play **C128d**.
- This is the brief's control (N proposals with L vs N without). A UNIFORM 8 of N exchangeable proposals is
  distributed as F8, so F8 already is the uniform-N control. The missing control is the dice-weighted large set.

**A3 (HIGH): a PERMUTED-L placebo per primary arm.**
- Within each state, permute the L values across the proposals with a fixed sha256 key
  (`search_1c_placebo_v1:<arm>:<id>`), then weight by exp(log_w + L_perm). Its sharpness and ESS match the real arm;
  it carries no information. Zero cost.
- Placebo − the dice arm must NOT read NEARER. If it does, STOP the read (meter or leak).

**A4 (HIGH): generalisation.**
- The registered replication banks (E2's competent-opponent states, the maturity plateau bank) are POOL-opponent banks
  too. Require ONE OFF-POOL opponent bank (held-out or `hl_05_26` teams, design F4) AND the plateau checkpoint
  (≥ 280M), per the plateau rule.
- Until both exist, any NEARER or SEARCH-PAYS is labelled **PROVISIONAL-POOL** and unlocks only cheap, reversible work.
- Every verdict line states the BOT stratum's verdict beside the pooled one. A result carried by the snapshot stratum
  alone is labelled **SELF-PLAY-ONLY** (F1).

**A5 (MEDIUM): multiplicity.** The unit-level claim "honest search pays" over up to 3 played primary arms (5 with A2)
uses Holm-Bonferroni lower bounds. Per-arm 95 % intervals stay descriptive.

**A6 (MEDIUM): power.**
- Phase B's NULL needs the LEAF-K8 interval inside ±0.01. The between-arm half-width is about ±0.021 at 287 (§7), and
  wider on a truncated prefix, so NULL is very likely unreachable.
- Register now that **NOT-DETECTED means "no gain of about +0.03 or more (search-sized) detected"**: it neither
  reopens the route nor licenses a re-run on the same bank. State the n played.

**A7 (LOW): descriptive.**
- Phase B IMP split by ESS < 2 vs ≥ 2. A resample that collapses onto one world is search on a single guess.
- M_hit and U_hit per arm beside H, because the corpus raises the species share by construction.

**On the rest of 1c's design:**
- **Deterministic (rule 8):** yes. The bands are ±0.0005 around 0 and ±0.0095 around ±0.01; seeds and keys are fixed;
  the resample is systematic on a sha256 uniform; G5 recomputes it.
- **Phase-A power:** fine. The meters are deterministic per state, and F8 is nested in P_F, so the contrast is paired.
- **The NEARER bar is low (lo > +0.0005):** acceptable only because it is a trigger for cheap playouts, not a claim.
- **M256L (hazard 2) has no proper mixture weight:** keep it descriptive-in-spirit, whatever its row says.
- **Could it read WORKS from a leak?** Phase A: yes, through corpus near copies of pool teams (A1). Phase B: the
  same, because search over a near-copy world is near-oracle search. F worlds cannot leak this way.
