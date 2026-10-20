# What the published bots do that we don't: five architecture and recipe candidates, costed and ranked

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision, build or read that differs from
this doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-19): DESIGN; B-GROW BUILT.** Nothing here is registered or launched. **B-GROW** (§4) is built
(`--grow trunk_layers=N`, `gen3_grow_fork_v1`, config v159; `TASK_BACKLOG` T64), so X47 / X47b need only their GO and
a pin at or after it. B-KFFN and B-INT are not built; their BUILD SPECS (2026-10-19) are [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md) (`TASK_BACKLOG` T66 / T67). **X47 is STAGED (2026-10-19) as the X35 kit arm `toy_x47`** in batch 1's toy block (interleaved with its toy C0, the orchestrator's ruling C2; the toy arms run K9(b) `warn`, C3), its argv dry-run validated on a stand-in T\*: [`BATCH1_LAUNCH_SHEET.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md). Each candidate has its cheapest A/B,
its decision rule, its risks and its build, written so the orchestrator can adopt it into an X35 batch. Launches need
the orchestrator's GO, and a design block is not a launch command.

**Trigger.** The Showdown-bot survey
([`../research_state/literature/showdown_bots_landscape_2026-10-18.md`](../research_state/literature/showdown_bots_landscape_2026-10-18.md))
and the orchestrator's comparison found five ideas that the top bots of October 2026 use and we don't. We already
share their core: entity tokens, a pointer action head, a small transformer, PPO self-play against past selves, a
Rust simulator and one-turn search. Event history and pair history are ours alone ([`../ARCHITECTURE.md`](../ARCHITECTURE.md) §2.3).

**Where this sits.** The owner's plan (2026-10-16, binding; [`design_compute_efficiency.md`](design_compute_efficiency.md)
header) is:
1. production to its plateau (X26, P\* frozen at 280,000,170);
2. the toy to its plateau (X34, running; 141M on 2026-10-19);
3. plateau-breakers on both (X35);
4. then see whether the architecture matters there.

This doc treats each candidate as one more X35 arm wherever it can be one. That is the cheapest form, because an arm
shares its batch's control C0.

**Vocabulary.**
- **The toy** is T-ENT, "entity tokens + attention only": production's static per-Pokémon tokens, TWO attention
  rounds, the pointer head and the win-probability critic. It has no damage operator, beliefs, intent head, edges,
  seats or event history ([`design_x26_readiness.md`](design_x26_readiness.md) §3.5).
- **P\* / T\*:** the production and toy plateau checkpoints. **C0:** X35's shared plain-continuation control.
- **R1:** X35's primary readout, defined in [`design_compute_efficiency.md`](design_compute_efficiency.md) §5.6.4. It is
  Δ_pop, the b-ON minus b-OFF population score against a frozen reference population. The interval is Welch's over
  seed finals at 97.5 % per look. The bar is ±2.0 pp, with the classes GAIN / NULL / LOSS / UNDECIDED.
- **Upcycling / growth:** adding parameters to a trained network so that its function is UNCHANGED at the moment
  they are added. Examples are a new layer whose output starts at zero, or new expert weights copied from the shared
  weights.
- **ρ:** an arm's wall time per step divided by C0's.

---

## 0. One screen

- **Two ideas already exist in our code, in a form the bots' write-ups did not describe.**
  - **The opponent-action head exists.** Production's flat intent head predicts the opponent's next action, but it is
    trained **DETACHED**: its loss cannot reach the shared trunk. The bots' lever is therefore ONE switch,
    `opp_intent_grad_mode: shaping`, which today is a config-only flag (§2.3).
  - **Set-level moveset mutation is X41** (D3 in [`design_diversity_teams_opponents.md`](design_diversity_teams_opponents.md)
    §4.5(a)). It needs no new row (§2.5).
- **Imitation from human replays was already designed and then PUNTED by the owner on 2026-08-18.** The design is
  [`../ai_v11/design_human_replay_objectives.md`](../ai_v11/design_human_replay_objectives.md). Its measured census:
  - only **16.7 %** of rating ≥ 1500 decisions show the acting side's whole team with all four moves;
  - own items are known on **3.9 %** of own mons, and EV spreads on **none**;
  - our observation always holds our own full sets.

  A BC (behaviour-cloning) warm start trains on exactly those broken inputs. Re-opening it is the owner's call (§2.4, Q1).
- **The ranked plan** (§3):
  1. **First new candidate after X34 plateaus: X47, DEPTH GROWTH ON THE TOY.** T\* gets two more identity-init rounds
     (2 → 4). This is the largest structural gap to the bots: they use about 8 layers, the toy 2, production 3, and
     production's third round was never screened on its own. It costs ≈ 17 GPU-h for 3 seeds and needs one build,
     B-GROW (✅ BUILT 2026-10-19: `--grow trunk_layers=4`).
  2. **X48: per-token-kind feed-forward weights on PRODUCTION**, upcycled at P\*, ≈ 15 GPU-h. It cannot run on the
     toy, which has none of the token kinds the idea separates.
  3. **X47b: production depth 3 → 4**, only if X47 reads GAIN. ≈ 16 GPU-h.
  4. **X49: intent SHAPING on production**, ≈ 15 GPU-h. The prior is weak.
  5. **X50, human data:** the zero-GPU scoping read (S0) is DONE (2026-10-20). It is FEASIBLE on volume, but no
     decision has a complete own side. The GPU A/B waits for the owner's Q1.
  - **Redundant or dominated:**
    - set mutation (X41 already covers it);
    - fresh-from-scratch depth and expert A/Bs (the growth forks dominate them);
    - experts on the toy, and an intent head on the toy (both architecture-bound);
    - 8-layer depth (dominated by depth 4 until depth 4 reads GAIN);
    - a BC warm start before the cheaper human-label route (ai_v11 rung 1).
- **Against the queue:** X45(b)'s 0.5 GPU-h read and X35's staged arms (A1, C0, D1, A2, A3) stay AHEAD of every
  candidate here. X44 (≈ 10.4 GPU-h, ×1.4–1.8 per fresh run) also ranks ahead. X14 stays deferred behind X44.
- **Timing hazard (F-BI-1):** an arm shares C0 only at its batch's pin. B-GROW must land before the toy batch's pin.
  If it lands later, X47 needs its own toy C0, at +12.6 GPU-h. **B-GROW landed 2026-10-19**: the toy batch's pin must be
  at or after its commit.

---

## 1. The shared frame

1. **Growth forks, not fresh runs.** The question at a plateau is whether the idea BREAKS it. Three forms are
   function-preserving when the fork starts:
   - an identity-init round (its output projections start at exactly zero; `trunk_depth.IdentityInitRound`);
   - upcycled per-kind experts (each starts as a copy of the shared feed-forward);
   - a gradient-routing switch (the network's function does not change at all).

   So any change in b-ON against C0 comes from TRAINING with the lever, never from a perturbed start. A fresh A/B
   would be pre-plateau and PROVISIONAL (owner rule 2026-10-11), and its arms would pay for their own control.
2. **Equal compute.** A capacity lever changes the cost per step (ρ ≠ 1). Every arm runs X35's fixed length (toy L_T
   70M, production L_P 20M), so it can share C0.
   - **PRIMARY: equal effective learn GPU time.** b-ON is read at its node s_eq. s_eq is the largest 1M checkpoint
     whose Σ(`collect_ms` + `update_wall_s`) since the fork is ≤ the mean Σ of C0's seeds at their final
     ([`design_compute_efficiency.md`](design_compute_efficiency.md) §5.4). If a node's Σ lies within 1e-9 of the
     bound, the next lower node is taken (rule 8).
   - **SECONDARY: matched steps,** finals against finals.
   - When ρ ≤ 1.02 the two reads coincide, and only the final is read.
3. **The readout is X35's R1**, on the arm's own architecture and against that architecture's frozen R. The guards
   are the same:
   - G-F, the head-to-head;
   - G-C, the collapse battery;
   - G-K, Kakuna, on production. On the toy it is reported if it can be read: whether `main.anchors` loads the toy is
     UNVERIFIED (X37c).

   Looks come at 3 and 5 seeds. Seeds: toy 3601–3605, production 3501–3505, matched to C0.
4. **The verdict classes** (pre-registrable as written):
   - **ADOPT-CANDIDATE:** primary R1 GAIN and every guard passes;
   - **PER-STEP ONLY:** primary NULL or UNDECIDED, secondary GAIN. Reported to the owner as a throughput-dependent
     result (it depends on X44 / X45 / X14); no adoption;
   - **TRADE:** R1 qualifies but a guard fails;
   - **NULL / LOSS / UNDECIDED** after look 2.

   Nothing is adopted by the rule alone: an ADOPT-CANDIDATE goes to the owner.

   **For X47 (AM-X47-2, ACCEPTED by the orchestrator 2026-10-20, before any read; launch sheet §0):**
   - **X47b RUNS** on GAIN / ADOPT-CANDIDATE, and on PER-STEP ONLY. A PER-STEP ONLY is called "throughput-dependent"
     only when the per-seed gap between the final and s_eq has a 95 % interval wholly above 0. Otherwise it is
     "BOUNDARY" (AM-X47-5).
   - **UNDECIDED with Δ̂ > 0:** X47b keeps its rank-9 conditional slot, unlicensed but not dropped. The report says "no
     effect ≥ 4.6 pp detected" (σ 1.07 pp, 80 %), never "depth does not help". This class is final only at look 2.
   - **UNDECIDED with Δ̂ ≤ 0 (AM-X47b-4, ruled 2026-10-20 before any read):** X47b LEAVES its conditional slot and is
     deprioritised in the backlog. The report says "no effect ≥ X pp detected; point estimate ≤ 0", where X is the
     CURRENT look's detectable effect (AM-X47c-1, computed from the registered power function by
     `readout.r1_detectable_effect`): 4.6 pp after look 2, ≈ 6.8 pp when it ends at look 1.
   - **NULL:** X47b drops in rank, not out.
   - **LOSS:** X47b is not run on this evidence.
   - **TRADE:** X47b runs only on the orchestrator's ruling.
   - A NULL or LOSS is a verdict on THIS growth recipe, not on depth (review F-XR-11).
   - A GAIN whose out-of-lineage member (`pstar_x26`) has Δ̂ ≤ 0 goes to the owner labelled "in-lineage" (AM-X47-7).
5. **The toy only RANKS until X35's transfer calibration passes (§5.2 there).** A toy NULL never drops the
   production arm before then. A candidate that READS machinery the toy lacks skips the toy (§5.2 item 4 there).
6. **Throughput basis** (MEASURED, [`design_compute_efficiency.md`](design_compute_efficiency.md) §4.4 and §5.6.1):
   - **toy:** ≈ 4,800 steps/s, ≈ 5.8 GPU-h per 100M. A 70M fork is ≈ 4.2 GPU-h per seed;
   - **production:** ≈ 1,334–1,370 steps/s. A 20M fork is ≈ 4.4 GPU-h per seed, and an arm of 3 seeds plus guards
     is ≈ 13 + 1.4 GPU-h.

   Every ρ below is ESTIMATED and is measured on the arm's first 1M.

---

## 2. The candidates

### 2.1 C-DEPTH: a deeper trunk (X47 toy, X47b production)

- **What the bots do (self-reported).** Wolfe_Alt6767 uses 8 layers at width 256, with 7.5M parameters. mikumiku37
  uses about 8.7M parameters, depth unstated. **UNVERIFIED** beyond the survey's reading of the forum posts.
- **What we have.** The toy has 2 rounds and production has 3, both at d_model 128 with FFN (feed-forward) width 256.
  `--trunk-layers N` accepts N ∈ {2, 3, 4} (`trunk_depth.TRUNK_LAYERS_CHOICES`). Each extra round is an
  `IdentityInitRound`: a pre-LN block whose `out_proj` and `linear2` start at zero, so the deeper trunk at
  initialisation IS the shallower one, bit for bit ([`../ARCHITECTURE.md`](../ARCHITECTURE.md) §2.1 step 9).
  Production's third round costs +15.5 % of the GPU update (MEASURED). It has never been screened on its own
  ([`design_compute_efficiency.md`](design_compute_efficiency.md) §3, F8).
- **Mechanism.** Every round lets every token read every other token once more. Gen 3 decisions chain facts: "their
  Skarmory sets Spikes, so my Starmie's Rapid Spin matters, so keeping Starmie healthy matters, so I don't trade it
  into Tyranitar." Two rounds can compose only two hops of that chain.
- **Hypothesis H-DEPTH.** The toy's plateau is partly a COMPOSITION limit. Two more rounds, added at T\*, raise its
  population score at equal effective learn time. A GAIN would also tell R-T2 that the toy's ceiling is
  capacity-bound.
- **The ONE lever.** `trunk_layers` 2 → 4 at the fork, as identity-init rounds (+264,960 parameters).
- **The cheapest A/B.** **Toy, fork at T\*, 3 seeds (3601–3603), 70M each, against the toy C0 at the same pin;
  look 2 at 3604–3605 (with toy C0 3604–3605) iff look 1 reads CONTINUE (AM-4, orchestrator 2026-10-19).**
  The read is `main.x35.x47` (B-X47-READ, built 2026-10-19): s_eq by the §1.2 node rule from Σ(`collect_ms` +
  `update_wall_s`) over each fork's OWN TensorBoard rows, restart-aware; the §1.4 classes; constructed-case tests.
  - ρ ≈ **1.13, MEASURED** (2026-10-20, `main.x35.x47 seq` on X47 s3601's first 6.1M steps against all of toy C0
    s3601: collect 1.065, update 1.202). The FLOP estimate was 1.3–1.4 (+5.2 MFLOP forward per round-row against
    the toy's 17.4 total, §5.3 there). The toy's collect is mostly the CPU simulator, so it dilutes the GPU's ratio.
  - Cost ≈ 3 × 4.3 × 1.13 ≈ **14.7 GPU-h**, plus the R1 read (≈ 0.3 GPU-h in a lease window, or ≈ 15 CPU
    lane-hours for its ≈ 63k games, [`X47_READOUT_REVIEW.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/X47_READOUT_REVIEW.md) F-XR-6).
  - The primary read lands at s_eq ≈ 62M (node ≈ 352M), so it nearly coincides with the matched-step secondary.
  - **Production (X47b)** is P\* + 1 round (3 → 4), 3 × 4.4 × ≈ 1.09 + 1.4 ≈ **16 GPU-h**. It is CONDITIONAL on X47
    reading GAIN or PER-STEP ONLY. Before calibration a toy NULL only lowers its rank.
- **The decision rule.** §1.4, primary at equal time, with X47's own lines (AM-X47-2) and its amendments AM-X47-1 to
  -12 (launch sheet §0; the read plan is its §4.5). The registered DESCRIPTOR is the extra rounds' weight norms over
  training: `python -m main.x35.x47 norms`, reported as a ratio to the base rounds, "engaged" at ≥ 0.01.
  - **As registered, the descriptor is weak** (F-AMX-4). On the live s3601 the ratio is already 0.012 at the first 1M
    node, so under AdamW the line reads "engaged" almost at once.
  - A functional descriptor is ADDED (AM-X47b-3), reported only: `python -m main.x35.x47 policy-diff`, the mean KL on
    `bank_x26` between the node and the same weights with the extra rounds' `out_proj` / `linear2` zeroed.
  - G-C pools the seed pairs of finals by a crossed cluster bootstrap and reads the toy's own truth (AM-X47b-1 / -2).
- **Risks:**
  - more compute per step, which is why the primary is at equal time;
  - pre-LN extra rounds on top of post-LN base rounds is the mixed stack production already runs, so there is no new
    risk class;
  - the optimizer state of the new parameters (§4, B-GROW: they start with NO AdamW state, the parent's carries);
  - a deeper trunk makes the CPU meters slower (T41 territory).
- **Code.** The module exists. A fresh `--trunk-layers 4` toy argv already PARSES at HEAD: `checkargs` exit 0,
  "27 of 39 ARCH-surface keys differ … ✓ this command still launches", on the X34 staging argv with depth 4. The fork
  form is REFUSED at child start: `ModelVersionError: trunk_layers mismatch` (`model_version/compat.py`). It needs
  **B-GROW** (§4). Depth 8 needs `TRUNK_LAYERS_CHOICES` widened (≈ 0.25 agent-day); it is a dose arm, run only after
  depth 4 reads GAIN.

### 2.2 C-KFFN: per-token-kind feed-forward weights, the "position mixture of experts" (X48, production)

- **What the bots do.** Jaxcalibur describes an 8.5M "position-MoE" transformer (survey §3, from its own page). The
  brief's reading is that Pokémon, move and event tokens get different MLP weights inside one transformer. The
  routing details are **UNVERIFIED** here. Related published forms, cited from memory and **UNVERIFIED**: modality
  experts (VLMo / BEiT-3, Bao et al. 2021–2022) and sparse upcycling (Komatsuzaki et al. 2022).
- **What we have.** One FFN per round is shared by all 64 tokens. They are of five KINDS:
  - 12 Pokémon and OTHER_species;
  - 3 board tokens;
  - 4 of our move seats (E3);
  - 12 threat seats (E4 / E5, incl. OTHER_move);
  - 32 event seats.

  A learned token-type embedding tells them apart at the input (`team_transformer.token_type_emb`, 9 rows). Each
  kind sits at a FIXED position range under the seat contract (§2.3 there).
- **Mechanism.** Attention mixes; the FFN transforms each token on its own. A shared FFN must spend its 256 hidden
  units on five very different kinds: an event record, a Pokémon's state, a move's damage, a threat. Untying it gives
  each kind its own transform at the SAME FLOPs per token.
- **Hypothesis H-KFFN.** At production's plateau the shared FFN is a capacity bottleneck between kinds. Untying it
  raises the population score. ρ ≈ 1, so equal time and matched steps coincide.
- **The ONE lever.** `trunk_ffn_by_kind` off → on: per-kind `linear1` / `linear2` in every round, routed by fixed
  position ranges. Our and their Pokémon SHARE an expert (the bots' kind is "Pokémon"; it keeps the side-symmetric
  design). +4 × 65,920 parameters per round × 3 rounds ≈ +0.79M (2.61M → 3.40M). At the fork each expert is a COPY of
  the round's shared FFN, so the function is unchanged up to fp32 reassociation (§4).
- **The cheapest A/B.** **Production, fork at P\*, 3 seeds, 20M, against the production C0.** ρ ≈ 1.00–1.05 ESTIMATED:
  five smaller matmuls instead of one. Cost ≈ **15 GPU-h**.
- **Why not the toy.** The toy has no move, threat or event tokens. Its only kinds are Pokémon and 3 board tokens, so
  the faithful partition unties 3 of 15 tokens and is nearly vacuous. A toy split by SIDE (ours / theirs / board) is a
  DIFFERENT lever (it breaks the side symmetry) and is rejected here. This is an architecture-bound breaker (§1.5).
- **The decision rule.** §1.4. If it reads ADOPT-CANDIDATE, an ATTRIBUTION arm follows: a wider shared FFN with equal
  parameters. It separates "untying" from "more parameters".
- **Risks:**
  - more parameters per update;
  - the per-kind slicing adds small kernels (CPU meters, GPU utilisation already 6–8 %, §1.6 there);
  - a change to the extractor's state_dict means a config-version bump;
  - an expert that sees only 3 tokens (board) may overfit;
  - upcycled experts that never diverge give a cheap NULL. The descriptor is the experts' pairwise weight distance
    over training.
- **Code.** **B-KFFN**, SPECIFIED in [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md) §1 (≈ 1.5 agent-days, opus-high). Five experts per round, with the shared
  FFN kept as the Pokémon expert, routed by six build-time seat-role segments (OTHER_species is not beside the team
  tokens, and the E5 tail is typed HISTORY, so never by type id). A CLI-tier field and a config bump. The fork copies
  the shared FFN into each expert, and the gate is bit-identical at one 64-row batch (MEASURED achievable). It needs
  **B-GROW**, made field-generic first (that doc's §3).

### 2.3 C-INTENT: an auxiliary head that predicts the opponent's next action (X49, production)

- **What the bots do.** Jaxcalibur has auxiliary opponent-action and opponent-set heads (survey §5.4). Nessie123 has
  auxiliary heads for weather, terrain, damage and moves. The literature on auxiliary tasks shaping representations
  (UNREAL, Jaderberg et al. 2016) and on opponent-modelling auxiliaries (He et al. 2016) is cited from memory,
  **UNVERIFIED**.
- **What we have.** **The head exists.** Production's flat intent head (`agents/model/flat_intent.py`, T2) scores the
  opponent's next action over:
  - their active's K move seats;
  - OTHER_move;
  - a switch to each slot;
  - OTHER_species.

  It is trained on intent labels at coefficient 0.05, but its trunk read is **stop-grad**
  (`opp_intent_grad_mode: "detached"`, [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §3). The belief heads, by
  contrast, run `shaping`. The flag was demoted to CONFIG_ONLY on 2026-08-23: no run in 107 had ever typed it
  (`flag_registry.py`). The `shaping` arm is still constructible.

  What else exists:
  - the "fight detector" `grad/opp_intent_policy_cosine` is logged (`instrumented_ppo/ppo.py`), and reads ≈ 0 by
    construction under `detached`;
  - the head's quality is MEASURED: intent log loss −0.151 nats from X5, and it under-reacts ("swings 58 % as far as
    the truth", [`design_compute_efficiency.md`](design_compute_efficiency.md) §2).
- **Mechanism.** Under `shaping` the intent loss also trains the trunk: the shared tokens must encode what the opponent
  is about to do.
- **Hypothesis H-INTENT.** Making the trunk predict the opponent's next action raises the population score at P\*. A
  second hope, read only as a descriptor: a sharper intent head is a better opponent model for search.
- **The prior is WEAK.** Belief heads already shape the trunk, and the told-the-truth belief knock-ins read NULL at
  15M ([`design_compute_efficiency.md`](design_compute_efficiency.md) §2). The sampler's candidate (f), "the opponent's
  own choices as evidence", needs π_opp(action | the opponent's view in world w). That is the POLICY seen from the
  opponent's seat, not this head, which predicts from OUR view
  ([`design_world_sampler.md`](design_world_sampler.md) §(f)). So the search half of the idea does not need this
  training lever.
- **The ONE lever.** `opp_intent_grad_mode` detached → shaping at the fork. The function is unchanged; only gradients
  move.
- **The cheapest A/B.** **Production, fork at P\*, 3 seeds, 20M, against the production C0.** ρ ≈ 1.00–1.03 ESTIMATED:
  one more backward path through the trunk. Cost ≈ **15 GPU-h**. It skips the toy, which has no intent head; building
  one there would be a different lever.
- **The decision rule.** §1.4. The registered descriptors:
  - the intent log loss on held-out C0 rows;
  - the under-reaction slope;
  - `grad/opp_intent_policy_cosine`.

  A persistently negative cosine together with R1 LOSS is read as "the aux task FIGHTS the policy" (reported, not a
  separate verdict).
- **Risks:**
  - the aux gradient competes with the policy (the detector exists);
  - self-play intent labels model OUR population's opponents, which deepens pool-specific memorisation (the H3 channel
    of the diversity doc). The G-K Kakuna guard on the held-out panel covers it.
- **Code.** **B-INT**, SPECIFIED in [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md) §2 (≈ 0.5 agent-day, opus-medium). `--grow opp_intent_grad_mode=shaping`,
  with the flag RE-PROMOTED to the CLI tier. That is required: a CONFIG_ONLY field is frozen at `detached` in the
  build, so without it the arm would be a silent no-op. The fork-time gate adds a GRADIENT check (the parent's intent
  CE reaches no trunk parameter, and the child's does). MEASURED: `shaping` adds exactly one trunk route, the intent CE
  at `opp_intent_coef`, because the policy's trunk gradient is bit-identical under both modes (move resolution reads
  α / β stop-grad).

### 2.4 C-HUMAN: an imitation warm start from human replays (X50)

- **What the bots do (self-reported).** Wolfe_Alt6767 uses an imitation warm start, then PPO. RotomDotExe uses
  self-play, then supervised training on about 6k human replays to model OPPONENTS. Metamon uses offline RL on
  reconstructed human replays (arXiv 2504.04395, peer-reviewed). The VGC bots play doubles with team preview, so
  their own-side information problem differs from gen 3's. How they handled it is **UNVERIFIED**.
- **What we have.**
  - **The corpus:** our own download of public gen3ou replays, `<main>/replays/showdown/gen3ou/`, gitignored.
    376,410 logs (4.0 GB, 109 date folders, 2026-05-18 → 2026-09-23), counted 2026-10-19.
  - **The reader:** `main.live.replay_scan` reads each replay from BOTH seats through the Rust reader and encodes every
    turn.
  - **The previous design:** [`../ai_v11/design_human_replay_objectives.md`](../ai_v11/design_human_replay_objectives.md),
    with a four-rung objective ladder ordered by robustness to the reconstruction's breakage, and a Phase-0 census
    (2026-08-18, 263,159 logs then). **The owner PUNTED it on 2026-08-18:** "the partial-information nuance is exactly
    why this needs more refinement". It is re-opened only by owner decision.
  - **No reader of ours turns a replay into training rows today.** The Python BC log reader was deleted with poke-env
    (`designs/deleted_flags.md`: "to be rebuilt on the Rust reader when ai_v11 opens").
- **What blocks a warm start (MEASURED by S0, 2026-10-20, the Rust reader, the whole corpus; at rating ≥ 1500 there
  are 2.73M decisions; [`x50_replay_census_2026-10-20`](../research_state/measurements/x50_replay_census_2026-10-20/README.md)):**
  - all six own Pokémon AND all four of the acting Pokémon's moves (tier A) are known on **18.0 %** of decisions,
    which is **489,973**;
  - **every** own Pokémon at 4 moves: **0.03 %** of decisions. A two-ability species' ability is known on 0.29 % of
    own Pokémon, and the EV spread on **0 %**;
  - the own item is known on **69.7 %** of own Pokémon. ai_v11's 3.9 % was the deleted Python reader's number,
    overturned (F-X50-2);
  - our observation always contains our full sets (the static tokens read species, set and actual stats).

  So every BC input on our own side must be COMPLETED. Under the owner's rule the completion must be Smogon-derived
  (`hp_repair` plus Smogon spreads). That is the corpus-artifact class of TECH_DEBT's P1 row (HP Dark, defaulted
  EVs / natures). At play time the network reads the truth instead, so the BC phase trains on a distribution it never
  plays.
- **Is it a prior?** The human's clicked action is a label from a ladder replay, and the standing rule sanctions it
  (ai_v11 D2: "doubly sanctioned"). The COMPLETED own-side sets are READ by the network, so they must trace to Smogon.
  The brief's framing ("training data, not priors") holds for the labels, not for the completed inputs.
- **Mechanism.** A BC init starts PPO from human-like play: fewer steps to competence, and possibly a different
  basin.
- **Hypothesis H-BC.** At matched steps a BC-initialised toy is ahead early, and the gap persists to 50M. Separately,
  ai_v11's rung 1 hypothesis: human opponent actions as labels give a HUMAN opponent model, the one the sampler doc
  calls "the right one for the ladder" (its (f)).
- **The ONE lever.** The initial weights: BC on human decisions, against random init. The PPO recipe is identical,
  with no KL to the BC policy (that would be a second lever).
- **The cheapest A/B, staged:**
  - **S0 (0 GPU, CPU at rule 5b, ≈ 1 agent-day, sonnet-high):** re-run the ai_v11 census on the Rust reader over the
    whole 376k corpus. Measure:
    - the parse / refusal rate;
    - the per-DECISION mask-hazard rate (ai_v11 P0.4, never measured);
    - tier-A counts by rating floor;
    - the share of the acting side's set that is revealed BY GAME END. Hindsight completion is TRUTH for that game:
      the player knew their own set.

    S0 decides feasibility: ai_v11's floor is ≥ 50k admitted tier-A decisions at ≥ 1500.

    **S0 DONE 2026-10-20 ([`x50_replay_census_2026-10-20`](../research_state/measurements/x50_replay_census_2026-10-20/README.md)): FEASIBLE by the floor.**
    - Reader: 0 refusals and 0 encoder failures over 752,820 seat reads (one classified spectator case).
    - Volume: tier A at ≥ 1500 is 489,973, about 10x the floor; 458,354 of them are also hazard-clean.
    - Mask hazard per turn decision: 2.5 % (a proxy that undercounts).
    - Loss enrichment of tier A: only 1.04.
    - The catch is not volume: no decision has a COMPLETE own side (above). The intent-label yield is in the
      "Dominated" bullet below.
  - **S1 (only on S0 FEASIBLE and the owner's Q1 = re-open), toy, fresh, 3 + 3 seeds to 50M:** BC-init against
    random-init, at one pin.
    - Cost: ≈ 6 × 2.9 + the BC phase (≈ 1 GPU-h ESTIMATED: ≈ 0.26–1.5M rows × a few epochs at the toy learner's
      ≈ 100k rows/s) ≈ **18.5 GPU-h**.
    - The toy is the right testbed: it reads no belief heads, so the completion artifact reaches it only through our
      own static tokens.
- **The decision rule (S1).**
  - PRIMARY: the mirrored 3 × 3 cross at matched steps 50M, seeds as units (T40's statistic, or `cross_stat` on the
    square), at the ±2.0 pp bar.
  - SECONDARY: the same cross at equal wall, INCLUDING the BC phase's GPU time and its CPU encode wall.
  - DESCRIPTORS: the KL between the trainee and the BC policy over steps (does PPO wash the init out?), and the
    held-out Kakuna panel.
  - Every read is PROVISIONAL: it is pre-plateau.
- **Risks:**
  - GIGO in the completed own-side sets;
  - human errors imprinted (median ≈ 1,560 at the ≥ 1500 floor);
  - PPO forgetting the init within a few M steps (AlphaStar needed a KL to its supervised policy, from memory,
    **UNVERIFIED**);
  - **the Kakuna anchor loses independence**: Kakuna learned from the same human population (F-BI-4);
  - `data/` is frozen while X26 is live, so any derived dataset lives in the archive, not `data/`.
- **Code.** S0: BUILT (`census.py` beside its README: `main.live.replay_scan` + `core_events --present-stream` + a
  text timeline). S1: B-BC, the large build, opus-high ≈ 5–8 agent-days:
  - a per-side REQUEST SYNTHESISER: the reader decides only at a `|request|`, and a spectator replay has none
    (S0 F-X50-5). Rung 1 needs it too;
  - POV rows with Smogon-completed own sets. The Rust reader asserts a fabricated own spread as known on a
    spectator read (S0 F-X50-4: IVs 31, EVs 0, serious), and the rows must override it;
  - action labels mapped to the 11-dim space, with ai_v11's mask-hazard filter;
  - a BC trainer on the pointer head that writes a standard checkpoint with its lineage;
  - a declared "initialise a FRESH run from this checkpoint" path. Today `--model` means resume or fork, with the
    recipe restored.
- **Dominated for the search half** by ai_v11's rung 1: the human OPPONENT's actions as intent labels, offline. It
  needs no policy change and it is cheaper than BC. It is the better first re-opened rung if the owner re-opens
  ai_v11. **S0's label yield at ≥ 1500 (2.44M viewer turn decisions):**
  - **86.0 %** carry the opponent's observable choice (2.10M labels), and 44.8 % (1.10M) carry an
    already-revealed target that alpha / beta address deterministically;
  - **69.4 %** survive the production label rules (approximated on the text, UNVERIFIED against the labeller);
  - Hidden Power (4.3 %) is unlabelable: the replay never states its type;
  - ai_v11's "92 %" counted paired turns, not usable labels.

### 2.5 C-MUTATE: set-level moveset mutation in self-play (no new row; X41)

- **What the bots do.** Nessie123 plays half of its self-play games with mutated sets (self-reported).
- **What we have.** This is **X41 / D3 (a)** ([`design_diversity_teams_opponents.md`](design_diversity_teams_opponents.md)
  §4.5): per episode, with probability q, resample k Pokémon's moves, item, EVs and nature from Smogon's per-species
  usage. Its A/B, rule, cost (≈ 13 GPU-h + a 1–2 agent-day build) and risks are written there. It ranks below X38
  (D1, REAL human sets, launch-ready) and runs if D1 reads TRADE with a large in-pool cost.
- **What this doc adds.** Nessie's 50 % share is one more data point for q, which the X41 row now cites. The lever is
  architecture-agnostic, so it COULD be calibrated on the toy. That is not proposed: D3's primary is the held-out
  Kakuna read, and the toy's Kakuna path is UNVERIFIED (X37c).
- **Verdict: REDUNDANT** with X41, and dominated by X38 unless X38 reads TRADE.

---

## 3. The ranking

Rank is expected information per GPU-hour toward the owner's plan, with built arms above unbuilt arms of equal promise
and conditions respected. X35's staged order is the orchestrator's (§5.6.2 there). This table only slots the
candidates into it.

| rank | id | what | GPU-h | built? | why here |
|---|---|---|---|---|---|
| 1 | X45(b) | the grouped-forward GPU read | ≈ 0.5 | **BUILT** | makes every later arm cheaper; adopted on a parity gate, no learning A/B |
| 2 | X35 batch 1 (production) | A1, C0, D1, A2 | ≈ 13–14.6 each | **BUILT / staged** | targets the MEASURED plateau mechanism (a slow climb with a small real cycle) |
| 3 | X44 | fewer PPO passes, matched wall (pilot) | ≈ 10.4 | flags | ×1.4–1.8 on every future fresh run. A fresh-run recipe question, so independent of the plateau forks |
| 4 | X35 toy calibration | A1 + A2 + the toy C0 at T\* | ≈ 3 × 12.6 | **BUILT** | without it the toy may only rank |
| **5** | **X47** | **toy depth 2 → 4, growth at T\*** | **≈ 17** | **B-GROW** | **FIRST new candidate after X34 plateaus.** It joins the toy batch and shares its C0. It is the biggest structural gap to the bots, and production's own depth is an unscreened high cost |
| 6 | X35-A3 | main exploiters | ≈ 18 | **BUILT** | as staged |
| 7 | X35-A4 | trailing-magnet KL | ≈ 14.8 | T56 | as staged; targets the measured treadmill |
| 8 | **X48** | **per-kind FFN, production, upcycled at P\*** | **≈ 15** | **B-KFFN + B-GROW** | zero extra FLOPs; production-only |
| 9 | **X47b** | **production depth 3 → 4** | **≈ 16** | **B-GROW** | CONDITIONAL on X47 GAIN or PER-STEP ONLY; an UNDECIDED with Δ̂ ≤ 0 takes it out of this slot (AM-X47b-4) |
| 10 | **X49** | **intent SHAPING, production** | **≈ 15** | **B-INT + B-GROW** | weak prior (§2.3) |
| 11 | X35-A5 / A6 | Nash archive / league exploiters | ≈ 14.4 / 18 | T57 / flags | conditional, as staged |
| 12 | X14 | overlap collection with the update | build-heavy | NO | deferred until X44 reads (its gain is larger after X44) |
| **13** | **X50** | **human data: S0 DONE 2026-10-20 (FEASIBLE on volume: 489,973 tier A at ≥ 1500; no complete own side; 2.10M intent labels); S1 BC toy A/B** | **0 / ≈ 18.5** | **S0 BUILT / B-BC** | GPU part gated on owner Q1 (S0 FEASIBLE). Recommended: rung 1 first (§5) |
| — | X41 (C-MUTATE) | set mutation | ≈ 13 | build | REDUNDANT here; its own conditional row |

**Dominated or redundant:**
- **Fresh-from-scratch depth or expert A/Bs.** The growth forks dominate them: they share C0, carry no pre-plateau
  PROVISIONAL tag and need no extra control. A fresh depth-4 toy PARSES at HEAD today, so the fresh form stays
  available if B-GROW slips. It would cost ≈ 20 GPU-h for 3 + 3 seeds to 50M and pay for its own control.
- **8 layers** (the bots' depth). Dominated by depth 4 until depth 4 reads GAIN; it is then a dose arm.
- **Experts on the toy.** Nearly vacuous (§2.2).
- **An intent head on the toy.** A different lever from the bots' (§2.3).
- **A BC warm start before ai_v11's rung 1** (§2.4).
- **Set mutation.** X41 (§2.5).

**The first experiment and its cost.** **X47**: T\* + 2 identity-init rounds, seeds 3601–3603, 70M each, ABBA with the
toy C0, in the toy batch at one pin; look 2 (3604–3605 with toy C0 3604–3605) only on a look-1 CONTINUE (AM-4).
- ≈ 17 GPU-h, plus ≈ 0.3 GPU-h for the read; look 2 ≈ 20 GPU-h more (2 × 5.8 + 2 × 4.3).
- Its prerequisite is B-GROW (✅ BUILT 2026-10-19), landed before the toy batch's pin.
- The first R1 look lands ≈ 17 GPU-h after its first seed launches.

---

## 4. The builds (proposed; filed in `TASK_BACKLOG` when the orchestrator adopts each)

| build | what | effort | needed by |
|---|---|---|---|
| **B-GROW** ✅ **BUILT 2026-10-19** (T64; as built: `--grow trunk_layers=N` only — the other two fields are declared and refused until B-KFFN / B-INT; the parent loads strictly at its own architecture and the growth is applied in place; the gate runs on fresh CPU builds; the seeded pool grows too; `designs/model/versioning.md` "Declared growth at a fork") | **Declared growth at a fork.** A typed `--grow <field>=<value>` list, limited to `trunk_layers`, `trunk_ffn_by_kind` and `opp_intent_grad_mode`, lets exactly those fields differ from the parent's `model_config.json` in `check_compatible`. Parent parameters load STRICTLY. New parameters are initialised function-preservingly: identity rounds as today; experts as copies of the shared FFN. **At fork time, the child's fp32 forward must match the parent's on real rows** (bit-identical for rounds and the grad switch; ≤ 1e-6 for experts, or a typed refusal; SUPERSEDED for experts by B-KFFN's spec: bit-identical at one 64-row batch, MEASURED achievable). New parameters' Adam moments start at zero, and the existing ones carry over (how a fork carries the optimizer state today is UNVERIFIED, F-BI-3). A `grow` block in `metadata.json`'s lineage. Tests that fail on revert | opus-high, ≈ 1.5–2 agent-days | X47, X47b, X48, X49 |
| B-KFFN (T66) | per-kind FFN experts (§2.2), its flag and config version; **SPEC: [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md) §1**, incl. the field-generic growth refactor (§3 there) | opus-high, ≈ 1.5 agent-days (+0.25 for the refactor) | X48 |
| B-INT (T67) | `--grow opp_intent_grad_mode=shaping`: the flag back on the CLI tier, a growth row and a gradient gate; **SPEC: [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md) §2** | opus-medium, ≈ 0.5 agent-day (after B-KFFN's refactor) | X49 |
| B-DEPTH8 | widen `TRUNK_LAYERS_CHOICES` | ≈ 0.25 agent-day | an 8-layer dose arm, only after X47 GAIN |
| S0 ✅ **BUILT + RUN 2026-10-20** ([`x50_replay_census_2026-10-20`](../research_state/measurements/x50_replay_census_2026-10-20/README.md)) | the replay census on the Rust reader (§2.4) | sonnet-high, ≈ 1 agent-day, CPU | X50 |
| B-BC | the BC pipeline (§2.4) | opus-high, ≈ 5–8 agent-days | X50 S1, only after the owner's Q1 |

---

## 5. Owner question

**Q1. Re-open ai_v11 (human replays), which you punted on 2026-08-18?** The bots' imitation warm starts are new
evidence, but they come from doubles with team preview, where the acting side's missing information differs. My
recommendation: run **S0** (zero GPU) now either way. Then re-open **rung 1 first**: human opponent actions as intent
labels, offline, feeding search's opponent model. Re-open **rung 3**, the BC warm start, only if S0 clears ai_v11's
50k floor and rung 1 reads something. Until you answer, X50 stops at S0.

**S0's answer (2026-10-20): re-open narrowly, rung 1 only; keep the BC warm start closed for now.**
- The label side is clean and dense. At ≥ 1500, 86 % of the opponent's turn actions are observable choices: 2.1M
  labels, half of them on targets the head addresses deterministically. The opponent half of the input is in its
  native distribution, and a turn-by-turn read leaks nothing as long as the opponent's set is never completed from
  the future.
- Behaviour cloning fails on a different axis than volume. Tier A clears the 50k floor about 10x, but a decision with
  the WHOLE own side known does not exist (every mon at 4 moves: 0.03 %; EVs never). So every BC row is a
  completion, and S1 would measure the completion as much as the human play.
- The next rung-1 step is still zero GPU: ai_v11's P0.2 / P0.3 offline probe. Score the current intent head on
  held-out human decisions against the Smogon prior. It needs the request synthesiser and an own-spread override
  first (S0 F-X50-5 / F-X50-4), the same core B-BC needs. Train on human labels only if the probe reads something.

---

## FINDINGS (standing rule 7)

- **F-BI-1 (HAZARD, scheduling).** An arm shares C0 only at its batch's pin. If B-GROW lands after the toy batch's pin,
  X47 needs its own toy C0: +12.6 GPU-h, a 74 % cost increase. Dispatch B-GROW before T\* is called.
- **F-BI-2 (FIXED 2026-10-19).** `main.checkargs` and `main.launcher --dry-run` now judge a `--model` argv's
  architecture against the checkpoint's `model_config.json` with the trainer's own version build and
  `check_compatible` (`main.train.fork_arch`), and refuse a mismatch FATAL_CONFIG with the child's own reason (a toy
  fork typing `--trunk-layers 4` reads `trunk_layers mismatch: saved=2, current=4`). It is ADVISORY when the child is
  pinned to another commit, so a growth fork should be dry-run at a pin that IS the checked tree. A growth argv
  passes only through a declared option: `fork_arch.DECLARED_GROWTH` is B-GROW's seam, and since B-GROW
  (2026-10-19) its row is `--grow` (dest `grow`; the fields are `agents.model.growth.GROWABLE`'s BUILT ones): a
  `--grow` argv is judged by `ModelVersion.check_compatible_grown`, the very gate the trainer's load runs
  (`load_model_snapshot(grow=…)`), so the dry run and the child agree. Detail: `designs/launcher/launch_guards.md`.
- **F-BI-3 (RESOLVED 2026-10-19, B-GROW).** A fork carries the parent's WHOLE policy AdamW state: SB3 loads it and
  `checkpoint_state._validate_or_reset_optimizer_state` realigns every entry by NAME, param group and LR included
  (`--fork-lr` pins the LR). A growth rebuilds the optimizer over the grown policy, carries each parent parameter's
  state object unchanged and gives the new parameters none (zero moments and their own step at their first update).
- **F-BI-4 (a reading hazard for X50).** Kakuna (Metamon) learned from human ladder replays. A BC-initialised model
  shares its training population, so the Kakuna anchor stops being an independent opponent for that arm. The
  in-house cross is the primary for that reason.
- **F-BI-5 (RESOLVED 2026-10-20 by S0: §2.4 now quotes S0's numbers; ai_v11's own-item figure was overturned).**
  The census numbers in §2.4 were ai_v11's (2026-08-18, the Python reader, 263k logs, ≈ 30k
  decisions at ≥ 1500). The corpus has since grown to 376k, and the reader is now Rust. S0 re-measures them; they
  are not re-verified here.
- **F-BI-6 (UNVERIFIED literature, from memory):** VLMo / BEiT-3 modality experts; sparse upcycling (Komatsuzaki et al.
  2022); Net2Net-style function-preserving growth (Chen et al. 2015); UNREAL auxiliary tasks (Jaderberg et al. 2016);
  opponent-modelling auxiliaries (He et al. 2016); AlphaStar's supervised init and its KL. Every bot's architecture
  figure is the author's self-report, read through the survey's page summariser (survey F5).
- **F-BI-7 (ESTIMATED costs).** Every ρ is an estimate from counted FLOPs. The toy's depth-4 ρ ≈ 1.3–1.4 could be off
  by the GPU's poor utilisation of small kernels (§1.4 there). X47's cost scales with it, and its first 1M measures
  it.
- **F-BI-8 (precedence).** X47 overlaps X36's rung 1 ("toy + depth 3" at equal time, fresh, DEFERRED). X47 answers the
  plateau question; X36 still owns attribution of a CEILING, if R-T2 reads one.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-19 | **This doc CREATED** (design worker, orchestrator brief after the bot survey) | Five bot ideas costed as X35-frame arms. **Growth forks** at the plateau checkpoints share X35's C0, with equal effective learn time as the primary for capacity levers. Ranked: X47 toy depth growth FIRST after X34 plateaus (≈ 17 GPU-h, B-GROW), then X48 per-kind FFN on production, X47b (conditional), X49 intent shaping, and X50 human data (S0 now, the GPU part on the owner's Q1). C-MUTATE folded into X41. All of them after X45(b), X35 batch 1, X44 and the toy calibration | fresh-from-scratch A/Bs (pre-plateau, own controls); experts on the toy (no move / event tokens); a toy intent head (a different lever); 8 layers before 4; a BC warm start before ai_v11's rung 1 (the owner punted ai_v11 on 2026-08-18, and its census shows 16.7 % tier-A, 3.9 % own items known); a new set-mutation row (X41 exists) | `trunk_depth.py`, `model_version/compat.py`, `flag_registry.py` (`opp_intent_grad_mode`), `team_transformer.py`; `checkargs` exit 0 on a fresh depth-4 toy argv at HEAD; the fork refusal path (F-BI-2); `../ai_v11/design_human_replay_objectives.md` §5; the survey |
| 2026-10-19 | **F-BI-2 fixed** (build worker) | The dry-run tools check a `--model` argv's architecture with the trainer's own code; growth only through `fork_arch.DECLARED_GROWTH` | a second comparator in the launcher; refusing a pinned-other-commit fork (that commit's check is the authority) | `src/main/launcher/fork_arch_dry_run_test.py` (refusals fail on revert) |
| 2026-10-19 | **B-GROW BUILT** (build worker; `TASK_BACKLOG` T64) | `--grow trunk_layers=N`, fork-only: strict parent load at its own architecture, then in-place identity-round growth (a fresh depth-N build's modules), the optimizer rule above (F-BI-3 RESOLVED), the fork-time gate on fresh CPU builds (bit-identical logits + values on 64 real rows, else FATAL_CONFIG), the seeded pool grown the same way, the `grow` lineage stamp and `model_config.json` `growth` (config v159); `checkargs` / `--dry-run` resolve and print a `--grow` argv | non-strict loading of the parent into the child architecture (would need an exception to the strict-load mechanism); the gate on the learner's own device (a GPU need not reproduce itself bit for bit); building `trunk_ffn_by_kind` / `opp_intent_grad_mode` growth now (B-KFFN / B-INT, out of scope: declared and refused) | `growth_test.py` (toy 2 → 4 after one AdamW step: max \|Δ\| 0.0, 264,960 new parameters; compiled child = compiled parent 0.0, 2.4e-7 from eager), `grow_fork_test.py`, a CPU `--debug` parent + grow-fork smoke |
| 2026-10-19 | **X47 STAGED in X35 batch 1's toy block (launch-staging worker)** | [`BATCH1_LAUNCH_SHEET.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md): X47 = the toy C0 argv + `--grow trunk_layers=4`, run `rb_x35t_x47d4_s<seed>`, seeds 3601–3603, +70M; `checkargs` exit 0 and the dry-run's `grow : DECLARED trunk_layers 2 -> 4` at `803a13e3` on the 200M STAND-IN for T\*; recommended order interleaved with the toy C0 (toy A2 last, choice C2) | appending X47 after toy A2 (not ABBA with its C0); a kit arm now (out of scope: the launch is by hand until `main.x35.kit` gains `toy_x47`, F-LS-1) | `batch1_validation_803a13e3/` |
| 2026-10-19 | **X47 is a KIT ARM; interleaved with its toy C0 (launch-kit worker; orchestrator rulings C2 / C3)** | `main.x35.kit` has `toy_x47` (the toy C0 argv + `--grow trunk_layers=4`, seeds 3601–3603, `rb_x35t_x47d4_s<seed>`), so `launch_x35.sh` checks and launches it and the disk check counts it (F-LS-1 closed); the batch-1 order puts it beside its toy C0 (s3601: C0 then X47; s3602: X47 then C0; s3603: C0 then X47), toy A2 last; the toy arms, X47 included, run K9(b) `warn` under X34's ceilings | launching X47 by hand; X47 after toy A2 | `src/main/x35/kit_test.py`, `BATCH1_LAUNCH_SHEET.md` §4.2 / §5, `batch1_validation_9c3bf3c4/` |
| 2026-10-19 | **B-KFFN / B-INT SPECIFIED (design worker)** | Build specs in [`design_growth_builds_kffn_int.md`](design_growth_builds_kffn_int.md): B-KFFN ≈ 1.5 agent-days (five per-kind experts, the shared FFN kept as the Pokémon expert, seat-role segments, bit-identical copy at the fork), B-INT ≈ 0.5 agent-day (the CLI-tier re-promotion is REQUIRED, plus a gradient gate), and a field-generic growth refactor first; filed as `TASK_BACKLOG` T66 / T67. B-INT's estimate rose from 0.25 because the CONFIG_ONLY freeze makes the tier change mandatory, and B-KFFN's from 1 because of the growth row, the pool and `pin_worker` | B-INT as a CLI-only change (it would be a silent no-op at a fork); token-type routing; a tolerance gate for the experts | CPU measurements in that doc (fresh production build at `b79b66bb`, trained `rb_es_end2_s2005`) |
| 2026-10-19 | **AM-4: X47 look 2 REGISTERED and its readout BUILT** (X35 amendment worker; the orchestrator's ruling on the batch-1 review F-RV-4) | X47 seeds 3601–3605 (look 2 = 3604–3605, against toy C0 3604–3605, launched only on a look-1 CONTINUE; `main.x35.batch` enforces it); `main.x35.x47` (B-X47-READ): Σ(`collect_ms` / 1000 + `update_wall_s`) summed over each tag's own rows (`update_wall_s` is dumped with the NEXT rollout's row), the fork's own events only, restart-aware by `time_to_step.py`'s resume rule; s_eq = the largest node with Σ ≤ the C0 mean final Σ (rule 8: within 1e-9 ⇒ the next lower node); ρ ≤ 1.02 reads the finals; the §1.4 classes with CONTINUE when the primary is look-1 CONTINUE; G-F and G-C required, G-K reported; the update-only s_eq and ρ split REPORTED only (F-RV-9's fix is not ruled) | X47 at 3 seeds as a RANKING read (a +5 pp depth effect reads GAIN ≈ 1 time in 3 at σ 1.2 pp); a row-wise collect / update pairing (mis-attributes the lagged update row) | `src/main/x35/x47_test.py`; `X35_BATCH1_REVIEW.md` F-RV-4 / F-RV-9 |
| 2026-10-19 | **X47's argv is now on the REAL T\* (launch-staging worker)** | X34 plateaued; T\* = the 290M node (sha256 `cfa7f7be…a22965e7`), so X47's five lines (seeds 3601–3605) read `--model …/checkpoint_290000087_steps.zip --steps 360000087 --grow trunk_layers=4` and all five PASS `checkargs` and the dry-run's `grow : DECLARED trunk_layers 2 -> 4` at `03b983be`; the toy C0 they share a control with froze its R on the same T\* | X47 on the 200M stand-in (retired) | `../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md` §4.2–§4.3, `batch1_validation_03b983be/` |
| 2026-10-20 | **X47's ρ MEASURED; its readout REVIEWED before the first read (adversarial review worker)** | ρ ≈ 1.13 (collect 1.065, update 1.202) on X47 s3601's first 6.1M steps, so s_eq ≈ 352M and the cost ≈ 4.9 GPU-h per seed; the review's verdict is SOUND-WITH-AMENDMENTS (AM-X47-1 to AM-X47-11, for the orchestrator to rule): the equal-time point's measured biases total ≈ 3M steps or less; in-learn eval IS inside `collect_ms` (≈ 3.8 % of Σ, the docstring's claim is false, ≈ 0.25M steps of bias); the guards G-C and G-F have never run on a toy; the likely final class (UNDECIDED) licenses nothing until it is ruled | the FLOP estimate ρ 1.3–1.4; s_eq ≈ 50–55M | [`X47_READOUT_REVIEW.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/X47_READOUT_REVIEW.md) §1–§3 |
| 2026-10-20 | **AM-X47 RULED and registered before any X47 read (orchestrator; X47 amendment worker)** | All eleven of the review's amendments were ACCEPTED, AM-X47-2 as proposed (UNDECIDED with Δ̂ > 0 keeps X47b's slot; the report says "no effect ≥ 4.6 pp detected"). One was ADDED: X47 arms are not thinned until the read (AM-X47-12). Built: `seq` reads only finished, K9(b)-PASS, unthinned runs and names the files; the eval share; the PER-STEP ONLY gap; the per-member Δ and the "in-lineage" label; look 2's moved nodes; the norm reader. The pre-read found that h2h plays a grown toy, that G-F and G-C read on toys, and that G-C's seed-pooled interval has no code (F-AMX-1) | UNDECIDED read as "depth does not help"; subtracting eval from Σ; a flip rule (F-RV-9) | `../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md` §0, §4.5; `src/main/x35/x47_test.py` |
| 2026-10-20 | **AM-X47b RULED and registered before any X47 read (orchestrator; G-C pooling worker)** | G-C pools the seed pairs of finals (each arm final with its same-seed control final) by a crossed cluster bootstrap over seed pairs and bank battles; the readers refuse a bare interval. The toy's B2 reads a truth re-branched under T\*, production keeps X26@150M's. The norm descriptor stays reported, and a policy-difference descriptor is added, reported only. A final UNDECIDED with Δ̂ ≤ 0 deprioritises X47b | a single-pair G-C; production's truth on the toy; an argmax or weight-norm descriptor deciding anything | `../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md` §0 AM-X47b |
| 2026-10-20 | **AM-X47c-1: the UNDECIDED line states the current look's detectable effect (orchestrator; X35 follow-up worker)** | A final UNDECIDED with Δ̂ ≤ 0 can end at look 1, where R1 detects ≈ 6.8 pp at 80 %, not look 2's 4.6; the line now prints the look's own figure from the registered power function (σ 1.07 pp, 200,000 common-random-number draws, bisection) | the constant 4.6 at every look (it overstated look 1's power) | `../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md` §0 AM-X47c; `src/main/x35/readout.py`, `x47.py`, `x47_test.py` |
| 2026-10-20 | **X50 S0 RUN (census worker)** | The census on the Rust reader over all 376,410 logs ([`x50_replay_census_2026-10-20`](../research_state/measurements/x50_replay_census_2026-10-20/README.md)): FEASIBLE by ai_v11's floor (489,973 tier A at ≥ 1500), but no decision has a complete own side; intent-label yield 86.0 % (choice) / 69.4 % (production rules) of opponent turn actions at ≥ 1500; §2.4, §3, §4 and §5 now carry it. Recommendation on Q1: re-open rung 1 only | treating "FEASIBLE on volume" as a GO for S1 (the completion problem is not a volume problem); quoting ai_v11's 3.9 % own items (a deleted-reader artifact); the production intent labeller on replays (it folds at decisions, and a replay has none: F-X50-5) | `census.py` / `census_report.py` / `shards/` / `results/census.md` there; HEAD `dfeb3d92d` |
