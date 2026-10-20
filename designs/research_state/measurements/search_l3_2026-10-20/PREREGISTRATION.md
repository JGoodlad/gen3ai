# L3: does the opponent's own behaviour pick out the true world? — PREREGISTRATION (2026-10-20)

**Written and committed BEFORE any likelihood is computed on a registered state.** Spec: the literature review
[`hidden_info_search_review_2026-10-20.md`](../../literature/hidden_info_search_review_2026-10-20.md) §3 and §10 (L3),
[`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §4 (f) and
[`search_ledger.md`](../../../endstate/search_ledger.md) §7 Q15. Nothing below changes after the read; a deviation is
reported as a FINDING in the README, never edited here. The read script [`l3_read.py`](l3_read.py) is part of this
registration (its sha256 is in [`l3_registration.json`](l3_registration.json)).

**What was touched before registration (no likelihood of any registered state, no AUC of any state).**
- The module (`src/main/value_search/opp_likelihood.py`) and its tests ran on the committed test bank (`BANK_V1`)
  with a MODEL-FREE forward (a fixed random projection of the row).
- The driver's per-state function ran as an engineering smoke on 10 states of the maturity interim bank that are
  NOT LEAN and NOT among the 287. It printed only the build time, the step counts, the per-world move match of the
  honest worlds (a builder property) and G4's flag (`True` on the 5 states where it was computed); 1 of the 10 was
  a build exclusion.
- The read script's selfcheck ran on synthetic states.
- The bank's strata and the 287 ids were read (stage 1's registered set).

Abbreviations: F = the fully honest world builder (every hidden fact drawn from Smogon usage, kept only if it replays
what our side saw); AUC = area under the ROC curve, here the probability that the truth's likelihood beats a random
honest world's; CI = a battle-clustered percentile bootstrap interval (2,000 draws, seed 20261020, states weighted equally).

## 0. The question

Honest search does not beat the policy: its worlds are almost never right about the opponent's team (an exact world
has mass ≤ 0.0025 under every candidate sampler; the seen mons' unrevealed MOVES are the class every sampler lacks,
`design_world_sampler.md` §6.1). An unused move leaves no protocol line, but the choices the opponent made INSTEAD of it
do. That inference from play is the best-replicated hidden-information lever in the literature (Skat, bridge, Hanabi;
the review §3). **Does the TRUE world out-rank today's honest worlds under the likelihood of the opponent's own
observed choices, with our policy network seated as the opponent?** If it does, re-weighting (or resampling a larger
proposal set) by that likelihood is a sampler lever worth a playout stage (stage 1c). If it does not, candidate (f)
is dropped for our network as the opponent model at this maturity.

## 1. The bank

The world sampler's STAGE-1 bank, unchanged: the maturity battery's interim LEAN states at X26 225M that F could
build at every anchor, **287 states** (`world_sampler_stage1_2026-10-19/s1_common.py` `common_ids()`, sha256
`cf8d8c6f…dfb`; read-only from `~/gen3ai_archive/hidden_info_maturity`), in stage 1's registered processing order.
Each state carries its opponent STRATUM from `states_mat_x26.json`: **bot** (a scripted eval opponent) or **snapshot**
(one of X26's own earlier snapshots). The searched side is X26; its opponent's past decisions are the evidence.

No other bank is read (E2's 15M pins and Tier 1 need other checkpoints and encoders; a replication there is a follow-up,
not part of this unit).

## 2. The candidate worlds

Per state, **the truth plus J = 8 honest worlds**:
- **Honest worlds:** today's builder, `WorldSampler(reveal="off", dice="rolls", max_tries=60)` with F2's legality rule ON
  (its default), the studies' call `sample(log, at, side, 8, key=<decision id>, screen_cache={})`, at the driver's
  commit. A world replays the damage rolls its gate accepted (its log's `"rolls"`). A state the builder cannot fill is
  an EXCLUSION (counted, named), never a defect.
- **The truth:** the true team rendered through the SAME template every world is rendered from
  (`opp_likelihood.truth_log`), replayed on its own rolls.

## 3. The likelihood (`main.value_search.opp_likelihood`)

**Definition.** For a world w: log L(w) = Σ_t log π(a_t | o_t^opp(w)), over the opponent's past decisions t before the
root:
- π is the masked softmax of the network's 11 policy logits, temperature 1, computed in float64 from the float32 logits.
- o_t^opp(w) is the **OPPONENT's own observation row** at its t-th decision, in world w's prefix replayed through
  `core_events --trackers --obs` at the driver's commit. The core encodes each side from its own seat: the row holds
  the opponent's own team as world w has it (its moves, item, spread), and our team only as the opponent saw it. The
  opponent model sees what the opponent saw, never our side's view.
- a_t is the action it took (the log's token, mapped through the decision's legal tokens).

**The opponent model.** X26's **225M** checkpoint (`checkpoint_225000152_steps.zip`, sha256 `bc93a321…19ce`, the
maturity archive's registered copy), loaded on CPU at the driver's commit (`policy_spectrum.reader.load_checkpoint`,
`qhat.logits_and_values`, 1 torch thread).

**Which decisions count** (one list per state, the same for the truth and every world):
- **scored**: every opponent choice whose result the searched side saw, kept as the truth's token in every world.
- **unobserved** (EXCLUDED, counted): a choice the world template rewrote to a SLOT, i.e. a move the searched side
  never saw that mon use, or a switch to a mon it never saw. Its token is world-specific and its slot is the truth's,
  so scoring it would leak. It is treated as carrying no evidence.
- **pending** (EXCLUDED, counted): the opponent's choice for the ROOT round when it chose first. The searched side has
  not seen it. It is pending iff the replay without it still emits the searched side's root request.
- A decision with ONE legal action contributes exactly 0. It is counted as scored but not informative; informative
  = ≥ 2 legal actions.

**Numerics.** All rows of a state (the truth and the 8 worlds) go through ONE deduplicated forward, so identical rows
score identically in every world. The per-step log-probabilities are summed in step order. Two log-likelihoods within
**1e-6 nats** are a TIE (rule 8: the rounding band is excluded, never broken by chance).

**The familiarity control.** conf(w) = Σ_t log max_a π(a | o_t^opp(w)) over the same steps: the network's confidence in
its OWN top choice. It reads no observed action.

## 4. The metrics

Per state s, Δ_sj = log L(truth) − log L(world j), and A_s = (1/8) Σ_j [1(Δ_sj > 1e-6) + ½·1(|Δ_sj| ≤ 1e-6)].

**PRIMARY: AUC = the mean of A_s over the OK states** (pooled, both strata), with its CI.

**SECONDARY** (each with its CI, pooled and per stratum):
- **S1.** top1: the share of states where the truth beats all 8 worlds clearly. Chance is ≤ 1/9 when the truth is
  exchangeable with the worlds.
- **S2.** p_truth: the truth's self-normalised likelihood weight among {truth} ∪ the 8 worlds; uniform = 1/9. It is
  the weight a likelihood-weighted search would put on the truth IF the proposal set held it. That is the dose
  curve's unit, `search_ledger.md` §5.1.
- **S3.** AUC_conf: the same AUC with conf in place of log L (the familiarity control). AUC_regret: the AUC with
  log L − conf (the observed action's log-probability relative to the network's top choice).
- **S4.** The reweighting lift, on the honest worlds only (the truth is not among them; this is what stage 1c would
  do). Per state, Σ_j w_j x_j − mean_j x_j with w = softmax(log L) over the 8 worlds. The x are:
  - **lift_moves**: the share of the seen mons' UNREVEALED true moves the world holds (Hidden Power untyped);
  - **lift_moves_exact**: 1 iff every seen mon's unrevealed set is exact;
  - **lift_species**: the share of the true never-seen species the world holds.

  A state without such facts is skipped for that x.
- **S5.** AUC by the truth's informative-step count (0, 1–4, 5–9, ≥ 10), and Spearman(A_s, informative steps): the
  review predicts the screen grows with the number of opponent decisions.
- **S6.** The informative share: the states where any |Δ_sj| > 1e-6.

## 5. The decision rule (deterministic; rule 8: ±0.005 around each boundary is not clear)

On the PRIMARY pooled AUC's 95 % CI [lo, hi], in this order:

| verdict | rule | meaning |
|---|---|---|
| **WORKS** | lo > 0.555 | the opponent's choices separate the truth from honest worlds MATERIALLY |
| **WEAK** | lo > 0.505 | a detectable but small separation |
| **REVERSED** | hi < 0.495 | the truth ranks BELOW honest worlds: the opponent model is anti-informative |
| **NO-EVIDENCE** | hi < 0.545 | no material separation; a small one is not excluded only below 0.55 |
| **NOT-DETECTED** | otherwise | inconclusive at this n |

**Registered consequences.**
- **WORKS:** stage 1c is worth building on this population, reweighting or resampling a larger honest proposal set by
  the likelihood, with playouts. The review's L3 decision ("a clear AUC > 0.5 ⇒ build 1c") is met.
- **WEAK:** 1c is built only if S4's lift_moves is also clearly > 0 pooled (its lower bound > 0). Otherwise the signal
  is recorded and 1c is not built on it alone.
- **NO-EVIDENCE / REVERSED / NOT-DETECTED:** candidate (f) with our network as the opponent model is dropped at this
  maturity. A behaviour-cloned opponent model is a different candidate.

The per-stratum verdicts (same rule) are SECONDARY. The review named the BOT stratum for its decision. This
registration makes the pooled read primary, because on the bot stratum our network is a mis-specified opponent model
by construction. Both strata are reported with their verdicts.

## 6. GIGO gates (a failure STOPS the read; reported, never patched)

- **G1.** The rows' ids equal the 287 registered ids (sha256 of the sorted concatenation).
- **G2.** At most 2 DEFECT rows. A defect is any guard of the module raising:
  - a world's searched-side rows differ from the truth's;
  - the opponent's tracker choice differs from the log's token;
  - a scored token is not exactly one legal token;
  - a replay fails.

  Exclusions (the builder's honest `WorldError`) are counted, not defects.
- **G3.** Determinism: 10 fixed states (sha256 order, salt `search_l3_recheck_v1:`) re-scored in a fresh process
  reproduce the stored log L and conf exactly.
- **G4** (descriptive, not a stop). The checkpoint was trained on the encoder at X26's pin `9c033489`; the run
  encodes at HEAD. Commits since the pin touch the Hidden-Power belief of a Transformed target (`0cef9320`) and a
  board reading (Imprison). Per state, the truth's opponent rows at the scored steps are compared byte for byte at
  the pin's `core_events`. If any differ, the primary is also read on the identical states only, reported beside it.

## 7. Hazards declared in advance (each reported as a FINDING with the read)

1. **Optimistic on the snapshot stratum.** There the opponent was an earlier X26 snapshot, so the opponent model is
   nearly the opponent. Against a ladder human it is not (`design_world_sampler.md` §4 (f)).
2. **Mis-specified on the bot stratum.** Scripted bots do not choose like the network.
3. **Familiarity.** The truth is a training-pool team the network has played against and with. It may be SURER on
   the truth's rows for reasons unrelated to what the opponent did. S3's AUC_conf measures that confound; a ladder
   opponent's team is not a pool team.
4. **Ranking, not reweighting.** The 8 honest worlds never hold the truth, so the AUC says whether the likelihood
   POINTS toward the truth. S4 says whether it moves the honest worlds' mass toward the truth's facts. Neither is a
   playout read (stage 1c).
5. **The world residuals** (`design_world_sampler.md` §3.2): the truth's dice except the damage rolls, and the hidden
   counters. They are shared by the truth and every world, so they move both alike.
6. **One model, one bank, one anchor** (X26 225M, 287 states). No replication is part of this unit.

## 8. Cost and run

About 10 CPU-s per state (measured on the smoke), ≈ 1 CPU-h with G3 and G4. It runs in ONE heavy slot (rule 5b:
`nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh --slot`), CPU only (`CUDA_VISIBLE_DEVICES=""`). Rows go to
`~/gen3ai_archive/search_l3/`. The read writes `result_l3.json` into this folder.
