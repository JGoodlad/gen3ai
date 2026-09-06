# Fold displacement — where a fold actually MOVES the weights, and whether that predicts its off-slice damage

**Status: PRE-REGISTERED 2026-09-05, before any displacement, projection, KL or cosine was
computed.** Everything below the `## PRE-REGISTRATION` heading was written first; results are
appended under `## RESULTS` and the pre-registration is not edited afterwards.

Predecessor: [`../sharing_kernel/README.md`](../sharing_kernel/README.md). It measured a
score-function gradient kernel at ONE parameter point (the parent's) and returned **NOT DETECTED**
on both its predictions — the taught/untaught split is not a direction the kernel distinguishes in
either era. Its two *norm* findings are what this probe follows up:

> the current architecture puts **85% of its policy-gradient norm in the encoders (51.8%) + team
> transformer (33.6%) and 0.66% in the pointer head** … If a fold's collateral is to be localised,
> that norm distribution says the place to instrument is the encoder/transformer trunk, not the
> action head. Testing the kernel account properly would mean (a) the KL / full-distribution kernel
> rather than the argmax score function, and (b) the kernel measured *along* a fold rather than at
> its start.

This probe does (a) and takes the first step of (b): it measures the **actual displacement a fold
produced**, per parameter group, and asks whether its first-order projection onto untaught states
predicts the measured off-slice movement.

---

## PRE-REGISTRATION

### P1 — per-group displacement

For each arm and depth, `Δθ_g = θ_arm,g − θ_parent,g` over the **policy's** named parameters,
reported two ways:

* **relative displacement** `|Δθ_g| / |θ_g|` (θ_g the PARENT's group weights), and
* **share of squared displacement** `|Δθ_g|² / |Δθ|²`.

**Prediction P1.** The **FUNDED** arms (`TCFUNDA`, `TCFUNDB` — the half of the teacher-content 2×2
that robbed the parent on untaught teams) move the **encoder + team-transformer** groups MORE than
the **UNFUNDED** arms (`TCUNFA`, `TCUNFB` — parent-neutral) at matched depth; the loss-off control
**C1** moves least of all nine arms; the **pointer head** is a small share of `|Δθ|²` everywhere.

### P2 — does the first-order projection predict off-slice movement?

On each **untaught** state `s′` (off-slice by construction — the 8 untaught teams are disjoint from
every teacher slice), with all gradients taken at the **PARENT's** parameters:

* `g_a(s′) = ∇_θ log π_parent(a|s′)` for every LEGAL action `a`;
* the **linear projection** `u_a = ⟨g_a, Δθ⟩`, and `δ(s′) = u_{a*}` at the argmax action `a*`
  (the quantity the task names);
* the **first-order KL** `KL₁(s′) = ½ Δθᵀ F(s′) Δθ = ½ Σ_a p_a u_a²`, where `F` is the
  per-state Fisher of the masked categorical over legal actions and `p_a` the parent's masked
  probabilities. (For a categorical `Σ_a p_a g_a = 0`, so `F = Σ_a p_a g_a g_aᵀ` exactly — no
  centering term is dropped.)
* the **ACTUAL** `KL(parent‖arm)(s′)`, by two forward passes, using `masked_kl_rows` **imported**
  from `agents.training.instrumented_ppo.distill_anchor` — never reimplemented; the identical
  statistic the live anchor monitor and `offline_collateral_kl.py` use.

**Group decomposition.** `u_a` is exactly additive over groups (`u_a = Σ_g u_a^{(g)}` with
`u_a^{(g)} = ⟨g_a^{(g)}, Δθ^{(g)}⟩`), so the quadratic decomposes exactly as
`KL₁ = Σ_g [½ Σ_a p_a u_a u_a^{(g)}]`. That additive share is the primary group decomposition. The
**group-alone** quadratic `½ Σ_a p_a (u_a^{(g)})²` is reported beside it as the "if only this group
had moved" figure; it does not sum to `KL₁` and is labelled as such.

**Prediction P2.**
1. The first-order projection **ranks the arms in the same order as the measured off-slice KL** in
   `../../reuse_batch_2026-09-03/offline_collateral_kl/` (canonical seeded clustered column):
   `R4DOSE12 (0.3062) < R4DOSE6 (0.3502) < C1 (0.3702) < B2 (0.3938) < R4DOSE3 (0.4416)`.
2. The per-group decomposition of `δ` / `KL₁` shows the **TRUNK** (encoders + team transformer),
   **not the action head**, carrying the off-slice movement.

**Registered as a way to be wrong.** A 4.45M-step displacement is not small, so the first-order
expansion may simply be a bad model of the realised KL. The **per-state Pearson and Spearman
correlation between `KL₁` and the actual KL** is therefore reported for every arm, so the reader
sees how linear the map is before reading any ranking off it. If those correlations are low, the
ranking claim in P2.1 is reported as unsupported *by construction*, not spun.

### P3 — direction, not magnitude

Per group, the cosine `⟨Δθ_A, Δθ_B⟩ / (|Δθ_A||Δθ_B|)` between arms.

**Prediction P3.** Funded-vs-unfunded cosine is **HIGH in the encoders** (both halves saw the same
states — same 16 teams, same dose, same pin for leg B) and **LOWER where the teacher content
differs**. Crucially, the **replicate-pair cosines** (`TCFUNDA·TCFUNDB` and `TCUNFA·TCUNFB`) are the
**FLOOR** any funded-vs-unfunded difference must clear: two arms differing only in seed already
disagree by some amount, and a funded-vs-unfunded cosine inside that band is **WITHIN FLOOR**, never
a direction.

### Arms, depths and checkpoints

Parent for every arm: **`ai_v9_59_R2ACTION_0827/final_model.zip`**, fork step **28,115,184** — read
from each arm's own `metadata.json` → `lineage.fork_parent`, not assumed. All ten runs carry
`arch_signature = gen3_critic_route_wave_v1`.

The 2×2 at three depths. The two interior depths are the checkpoints **common to all four arms**
(each arm has its own restart cadence, so the later checkpoint steps differ between arms; these two
do not):

| depth | step | Δ from parent |
|---|---|---|
| p1M | `checkpoint_29115216_steps.zip` | +1.000M |
| mid | `checkpoint_30115248_steps.zip` | +2.000M |
| end | `final_model.zip` | (recorded at load) |

Reuse-batch arms at **END** (`final_model.zip`): `ai_v9_150_R4DOSE12_0901`,
`ai_v9_151_R4DOSE6_0901`, `ai_v9_152_R4DOSE3_0901`, `ai_v9_140_B2_0901`, `ai_v9_141_C1_0901` — the
five run dirs read from `offline_collateral_kl.py::ARMS`.

**Key alignment is ASSERTED, not assumed**: every arm's policy `state_dict` must match the parent's
key-for-key and shape-for-shape, or the script exits.

**Buffers are reported SEPARATELY and never inside a group** — PopArt / running-normalizer buffers
are not parameters and their "displacement" is a statistic of the data stream, not of the update.

### States

`../sharing_kernel/states_gen.npz` reused verbatim: **456** parent-piloted greedy states, **24 teams
× 19**, ≥2 legal actions everywhere, 16 taught / 8 untaught. The 8 untaught teams are exactly the
`rev3_untaught_pulldown_selection.json` set that `offline_collateral_kl.py` scores — verified by
content sha. Taught and untaught are reported **separately** throughout.

### Parameter groups

`GROUP_RULES` / `group_of()` **imported from `../sharing_kernel/kernel.py`**, not copied: the same
six groups (`encoders`, `team_transformer`, `action_head`, `projection_mlp`, `belief_op`, `critic`),
the same first-match-wins prefix table, and the same hard failure on an ungrouped parameter.

⚠️ **The `critic` group is NOT off this probe's path, unlike the predecessor's.** There, `critic`
received exactly zero score-function gradient and was a check. Here a fold's *displacement* moves
critic weights like any other — P1 reports it. It stays out of the `KL₁` decomposition only because
`log π` does not depend on it, which is a fact about the projection, not about the fold.

### Inference

Team is the cluster unit. **Cluster bootstrap over the 8 untaught teams** (and separately over the
16 taught), 20,000 draws, **one fixed resampling index set shared by every arm** so an arm-vs-arm
difference is paired on the same team draws — the convention `offline_collateral_kl.py` uses.
Vocabulary: **SIGNIFICANT / WITHIN FLOOR / NOT DETECTED**; a CI that straddles zero is NOT DETECTED
and never a direction. Seed 20260905.

P1 and P3 are properties of two weight vectors — they have **no sampling distribution over states**
and get **no CI**. They are reported as exact numbers and nothing else.

### Caveats registered in advance

1. **First order at 4.45M steps may be poor.** Registered above with its own reported diagnostic.
2. **Adam makes `Δθ` direction ≠ gradient direction.** Every arm runs AdamW; the per-coordinate
   second-moment rescaling means a large `Δθ` in a group can reflect small, consistent gradients
   there as easily as large ones. Nothing in P1 or P3 licenses a statement about gradient magnitude.
3. **`Δθ` is an integrated path, not a step.** The arms restarted several times; `θ_end − θ_parent`
   is the net chord of a trajectory that may have travelled much further.
4. **The reuse-batch arms and the 2×2 arms are not one experiment** — different pins, different
   pools, different teacher sets, different launch batches. Any cross-batch comparison is
   descriptive. The 2×2's four arms ARE matched to each other; the reuse batch's five are matched to
   each other.
5. **Weight-space distance is not function-space distance.** Two arms can be far apart in `θ` and
   identical in behaviour (permutation/scale symmetries) — the whole reason P2 recomputes an ACTUAL
   forward KL rather than trusting `|Δθ|`.
6. **Greedy piloting.** The state batch is the parent's greedy trajectory (inherited from the
   predecessor for comparability), not its training rollout distribution, and not the stochastic
   distribution `offline_collateral_kl.py` piloted. So my actual-KL levels are **not** on the same
   scale as that artifact's; only the ORDERING is comparable, and that is how P2.1 is written.
7. **Loader.** `load_foreign_opponent` (env=None, inference-only, arch-family check) rather than
   `load_model_snapshot`, which requires a live `VecEnv` and enforces trainee-config compatibility.
   It is the standing offline loader for this program — both the predecessor and
   `offline_collateral_kl.py` use it.

---

## RESULTS

Run 2026-09-05 on an idle box (load average 1.16–1.6 on 16 cores, no training run). CPU only,
`nice -n 10`, BLAS threads pinned to 1 by both scripts; **4 torch threads** for the batched
gradient pass and the group projections, as permitted. Wall cost: `deltas.py` 22 s (18 model loads),
`project.py` 88 s (3064 action-gradients), `tables.py`/`ordering_perm.py`/`pooled_2x2.py` < 5 s.

**Reproducible, exactly.** Both stages were re-run end to end into a second scratch directory:
`displacement.json` (2652 numeric leaves) and `projection.json` (1450 numeric leaves) came back with
**max |diff| = 0.000e+00**, and `deltas.npy` was **bit-identical**.

### Artifacts

| file | what |
|---|---|
| `deltas.py` | P1 + P3 — loads parent + 17 arm-depths, per-group displacement, the cosine matrices |
| `project.py` | P2 — parent per-legal-action gradients, `KL₁`, the exact group decomposition, actual `KL(parent‖arm)`, cluster bootstrap |
| `tables.py` | the three README tables + the two floor tests → `results_table.txt`, `verdict_stats.json` |
| `ordering_perm.py` | exact 120-permutation p for the arm-ordering agreement → `ordering_perm.json` |
| `pooled_2x2.py` | the pooled funded-vs-unfunded contrast against its own replicate floor → `pooled_2x2.json` |
| `displacement.json` | per-arm per-group displacement, buffer diffs, full 17×17 cosine matrices per group |
| `projection.json` | per-arm `KL₁` / actual KL / correlations / group decomposition / contrasts, per slice |
| `results_table.txt` | everything above, printed |

### Exact checkpoints used

All ten runs carry `arch_signature = gen3_critic_route_wave_v1`; every arm's `metadata.json` names
`ai_v9_59_R2ACTION_0827` as `lineage.fork_parent` at `fork_step = 28,115,184`. **Key alignment
asserted and passed for all 17**: 281 parameter tensors, 3,147,887 parameters, identical names and
shapes, no exception raised.

| arm | run | file | step | Δ steps |
|---|---|---|---|---|
| parent | `ai_v9_59_R2ACTION_0827` | `final_model.zip` | 28,115,184 | — |
| TCFUNDA / B, TCUNFA / B @p1M | the four 2×2 runs | `checkpoints/checkpoint_29115216_steps.zip` | 29,115,216 | +1,000,032 |
| … @mid | " | `checkpoints/checkpoint_30115248_steps.zip` | 30,115,248 | +2,000,064 |
| … @end | " | `final_model.zip` | **32,637,168** | +4,521,984 |
| R4DOSE12 / 6 / 3, B2, C1 @end | the five reuse runs | `final_model.zip` | **32,637,168** | +4,521,984 |

**FINDING (provenance correction).** The brief gave the end step as 32,567,760. Every one of the
nine arms' `final_model.zip` reports `num_timesteps = 32,637,168` — all nine identical. The figure
above is what the checkpoints say.

**FINDING (provenance gap).** `num_timesteps` appears **nowhere in `metadata.json`** — not at the
top level, not in `lineage`. A run's step count can only be read by opening the `.zip`, which the
JSON-only offline tools (`main.lineage`, `main.sidecar_audit`, `main.dose`) cannot do. Worth a
top-level key.

Doses, from `python -m main.dose` (needed to read Table 1 honestly): the four 2×2 arms are all
`--fork-lr-freeze`d at **4.557e-08 (2.12× v8)** — genuinely matched. The dose ladder is
1.139e-08 / 2.279e-08 / 4.557e-08 (0.53× / 1.06× / 2.12×). **B2 and C1 are NOT dose-matched**: their
KL controllers ran free and their medians differ — B2 **6.845e-08 (3.19×)**, C1 **8.215e-08
(3.83×)**, i.e. C1 ran at **1.20× B2's dose**. Every B2-vs-C1 sentence below carries that.

### States

`../sharing_kernel/states_gen.npz` verbatim: 456 states, 24 teams × 19. **FINDING (brief
correction):** the split is **304 taught / 152 untaught rows** (16 taught teams, 8 untaught), not
the "150 / 150" the brief described. The 8 untaught team shas match
`rev3_untaught_pulldown_selection.json` exactly.

Mean legal actions 6.72; **3064** per-action gradients taken at the parent. The exact group
decomposition of `KL₁` closes to **7.1e-15**.

---

### TABLE 1 — per-group displacement

Parameter counts: `encoders` 223,159 (7.1%) · `team_transformer` 284,592 (9.0%) ·
`projection_mlp` 1,329,536 (42.2%) · `belief_op` 512,267 (16.3%) · `action_head` 55,683 (1.8%) ·
`critic` 742,650 (23.6%). Parent `|θ| = 231.847`.

**1a. Relative displacement `|Δθ_g| / |θ_g|` (×1000)** — the size-corrected view.

| arm | step | \|Δθ\| | rel | encoders | team_tr | proj_mlp | belief | **action_head** | critic |
|---|---|---|---|---|---|---|---|---|---|
| TCFUNDA@p1M | 29,115,216 | 3.266 | 0.01409 | 5.20 | 19.43 | 29.29 | 25.35 | **38.91** | 9.76 |
| TCFUNDA@mid | 30,115,248 | 4.481 | 0.01933 | 7.08 | 26.02 | 40.36 | 35.01 | **53.61** | 13.36 |
| TCFUNDA@end | 32,637,168 | 6.720 | 0.02898 | 10.82 | 39.17 | 58.48 | 55.20 | **76.32** | 19.57 |
| TCFUNDB@p1M | | 3.264 | 0.01408 | 5.13 | 19.34 | 29.55 | 25.18 | **39.60** | 9.45 |
| TCFUNDB@mid | | 4.463 | 0.01925 | 7.04 | 25.82 | 40.42 | 34.76 | **54.19** | 12.93 |
| TCFUNDB@end | | 6.757 | 0.02914 | 10.83 | 39.45 | 58.44 | 55.79 | **75.46** | 20.39 |
| TCUNFA@p1M | | 3.148 | 0.01358 | 5.00 | 18.62 | 28.04 | 24.91 | **35.84** | 9.40 |
| TCUNFA@mid | | 4.313 | 0.01860 | 6.80 | 24.86 | 38.62 | 34.15 | **49.49** | 13.12 |
| TCUNFA@end | | 6.511 | 0.02808 | 10.35 | 37.65 | 55.31 | 55.01 | **70.09** | 20.70 |
| TCUNFB@p1M | | 3.153 | 0.01360 | 5.01 | 18.80 | 28.16 | 24.96 | **35.62** | 8.93 |
| TCUNFB@mid | | 4.319 | 0.01863 | 6.83 | 25.01 | 38.63 | 34.38 | **48.71** | 12.73 |
| TCUNFB@end | | 6.508 | 0.02807 | 10.38 | 37.79 | 55.43 | 55.26 | **69.25** | 19.42 |
| R4DOSE12@end | | 3.976 | 0.01715 | 6.74 | 23.86 | 35.77 | 29.46 | **51.22** | 12.23 |
| R4DOSE6@end | | 5.160 | 0.02226 | 8.47 | 30.35 | 45.57 | 40.11 | **59.96** | 17.17 |
| R4DOSE3@end | | 6.544 | 0.02823 | 10.33 | 36.21 | 58.39 | 51.60 | **73.06** | 23.10 |
| B2@end | | **8.075** | 0.03483 | 12.44 | 45.24 | 69.90 | 67.25 | **86.04** | 27.51 |
| C1@end (loss OFF) | | **7.995** | 0.03449 | 12.52 | 45.12 | 63.18 | 72.07 | **78.22** | 29.77 |

**1b. Share of `|Δθ|²` (%)** — read against the parameter counts, because it largely IS them.

| arm | encoders | team_tr | proj_mlp | belief | action_head | critic |
|---|---|---|---|---|---|---|
| TCFUNDA@end | 9.0 | 12.5 | 40.6 | 30.9 | **2.6** | 4.4 |
| TCFUNDB@end | 8.9 | 12.5 | 40.1 | 31.2 | **2.5** | 4.7 |
| TCUNFA@end | 8.8 | 12.3 | 38.7 | 32.7 | **2.4** | 5.2 |
| TCUNFB@end | 8.8 | 12.4 | 38.9 | 33.0 | **2.3** | 4.6 |
| R4DOSE12@end | 10.0 | 13.2 | 43.4 | 25.2 | **3.4** | 4.9 |
| R4DOSE6@end | 9.4 | 12.7 | 41.8 | 27.7 | **2.7** | 5.7 |
| R4DOSE3@end | 8.7 | 11.2 | 42.7 | 28.5 | **2.5** | 6.4 |
| B2@end | 8.2 | 11.5 | 40.2 | 31.8 | **2.3** | 6.0 |
| C1@end | 8.5 | 11.7 | 33.5 | 37.2 | **1.9** | 7.1 |
| *(parameter share)* | *7.1* | *9.0* | *42.2* | *16.3* | *1.8* | *23.6* |

The only two groups whose displacement share departs materially from their parameter share are
**`belief_op` (16.3% of params, 25–37% of the displacement — over-moved ~2×)** and **`critic`
(23.6% of params, 4–7% of the displacement — under-moved ~4×)**.

**1c. Buffers, reported separately.** Across all 17 arms every constant data table
(`damage_op.*`, the priors, the index maps — 80 buffers) is **byte-identical to the parent**. The
only buffers that moved are PopArt's three running-normalizer scalars, e.g. at end depth
`popart.sigma` moved 0.52 / 0.24 / 0.69 / 0.23 on the 2×2 and 1.09–2.45 on the reuse arms.

**1d. The P1 floor test at END depth** — funded mean vs unfunded mean, against the two replicate
gaps (arms differing only in seed), on relative displacement ×1000:

| group | funded | unfunded | gap | repl. F | repl. U | verdict |
|---|---|---|---|---|---|---|
| ALL | 29.064 | 28.077 | **+0.987** | 0.160 | 0.010 | CLEARS FLOOR |
| encoders | 10.828 | 10.362 | **+0.466** | 0.010 | 0.028 | CLEARS FLOOR |
| team_transformer | 39.310 | 37.717 | **+1.593** | 0.274 | 0.144 | CLEARS FLOOR |
| projection_mlp | 58.463 | 55.366 | **+3.097** | 0.039 | 0.122 | CLEARS FLOOR |
| belief_op | 55.494 | 55.133 | +0.361 | 0.582 | 0.248 | WITHIN FLOOR |
| action_head | 75.886 | 69.670 | **+6.215** | 0.862 | 0.837 | CLEARS FLOOR |
| critic | 19.975 | 20.058 | −0.083 | 0.819 | 1.288 | WITHIN FLOOR |

**Post-hoc, not pre-registered: displacement grows as the SQUARE ROOT of steps.** The four 2×2 arms
are dose-frozen, so their three depths are a clean growth curve, and all four fit
`|Δθ| ∝ t^0.48` (exponents **0.479 / 0.483 / 0.482 / 0.481**). Across the nine end-depth arms,
`|Δθ| ∝ dose^0.37` (9 points, two batches — descriptive only). A √t law is the signature of a
largely **incoherent** walk, and it is what makes Table 3 legible.

---

### TABLE 2 — first-order projection vs actual off-slice KL

**2a. Untaught slice (152 states / 8 teams).** `KL1/act` is the first-order over-prediction factor;
`r` / `ρ` are the per-state Pearson / Spearman between `KL₁` and the actual KL.

| arm | actual KL | cluster CI95 | KL₁ | KL1/act | r | ρ |
|---|---|---|---|---|---|---|
| TCFUNDA@p1M | 0.27961 | [+0.2238,+0.3295] | 0.32751 | 1.171 | +0.628 | +0.786 |
| TCFUNDA@mid | 0.37417 | [+0.3054,+0.4445] | 0.48056 | 1.284 | +0.504 | +0.765 |
| TCFUNDA@end | 0.42960 | [+0.3403,+0.5376] | 0.75187 | **1.750** | +0.257 | +0.612 |
| TCFUNDB@p1M | 0.22028 | [+0.1707,+0.2720] | 0.29824 | 1.354 | +0.555 | +0.681 |
| TCFUNDB@mid | 0.31871 | [+0.2349,+0.4049] | 0.44459 | 1.395 | +0.446 | +0.691 |
| TCFUNDB@end | 0.39891 | [+0.2848,+0.5456] | 0.71992 | **1.805** | +0.386 | +0.690 |
| TCUNFA@p1M | 0.24425 | [+0.1836,+0.3081] | 0.38533 | 1.578 | +0.374 | +0.719 |
| TCUNFA@mid | 0.27939 | [+0.2095,+0.3595] | 0.43472 | 1.556 | +0.397 | +0.686 |
| TCUNFA@end | 0.38648 | [+0.2883,+0.4960] | 0.62334 | 1.613 | +0.461 | +0.676 |
| TCUNFB@p1M | 0.23559 | [+0.1931,+0.2717] | 0.36572 | 1.552 | +0.445 | +0.761 |
| TCUNFB@mid | 0.31273 | [+0.2456,+0.4116] | 0.47446 | 1.517 | +0.473 | +0.703 |
| TCUNFB@end | 0.36565 | [+0.2887,+0.4528] | 0.53171 | 1.454 | +0.388 | +0.659 |
| R4DOSE12@end | 0.39373 | [+0.3060,+0.4906] | 0.53765 | 1.366 | +0.460 | +0.715 |
| R4DOSE6@end | 0.41966 | [+0.3016,+0.5576] | 0.49961 | 1.191 | +0.466 | +0.712 |
| R4DOSE3@end | **0.55746** | [+0.4165,+0.7354] | 0.82141 | 1.473 | +0.398 | +0.593 |
| B2@end | 0.48734 | [+0.4277,+0.5435] | **0.85550** | 1.755 | +0.259 | +0.628 |
| C1@end | 0.43094 | [+0.3592,+0.5018] | 0.59633 | 1.384 | +0.339 | +0.516 |

(The taught slice, 304 states / 16 teams, is in `results_table.txt`; the same pattern, with the
over-prediction factor reaching **2.06** for B2.)

**2b. The ordering ladder** against the published clustered off-slice KL
(`R4DOSE12 0.3062 · R4DOSE6 0.3502 · C1 0.3702 · B2 0.3938 · R4DOSE3 0.4416`), with the **exact**
one-sided p over all 120 relabellings of five arms:

| predictor (untaught) | values in arm order `[D12, D6, D3, B2, C1]` | ρ | exact p |
|---|---|---|---|
| `\|Δθ\|` (weight-space distance) | 3.976 · 5.160 · 6.544 · 8.075 · 7.995 | +0.700 | 0.117 (14/120) |
| **`KL₁`** (first-order) | 0.5376 · 0.4996 · 0.8214 · 0.8555 · 0.5963 | **+0.800** | **0.067** (8/120) |
| **actual `KL(parent‖arm)`** | 0.3937 · 0.4197 · 0.5575 · 0.4873 · 0.4309 | **+1.000** | **0.0083** (1/120) |

Taught slice: 0.700 / 0.600 / 0.900 (p 0.117 / 0.175 / 0.042).

**2c. The group decomposition of `KL₁`** — exact additive shares (%), untaught, end depth
(`[bracket]` = the group-alone quadratic, which does **not** sum):

| arm | encoders | team_tr | proj_mlp | belief | **action_head** | critic |
|---|---|---|---|---|---|---|
| TCFUNDA@end | 31.4 [24.4] | 28.6 [21.3] | 33.0 [28.6] | 6.0 [3.8] | **1.0 [1.8]** | **0.0 [0.0]** |
| TCFUNDB@end | 35.2 [30.1] | 31.7 [24.1] | 28.6 [26.1] | 4.0 [3.6] | **0.5 [2.0]** | **0.0** |
| TCUNFA@end | 28.4 [25.4] | 31.7 [24.4] | 31.5 [28.0] | 4.2 [4.6] | **4.3 [1.9]** | **0.0** |
| TCUNFB@end | 25.2 [25.3] | 35.1 [30.6] | 34.7 [32.7] | 1.2 [5.2] | **3.8 [2.1]** | **0.0** |
| R4DOSE12@end | 27.0 [24.7] | 33.6 [22.6] | 28.1 [29.0] | 7.6 [4.9] | **3.7 [2.5]** | **0.0** |
| R4DOSE6@end | 31.2 [31.5] | 34.5 [28.7] | 26.2 [29.2] | 2.6 [6.1] | **5.6 [2.0]** | **0.0** |
| R4DOSE3@end | 21.7 [22.6] | 33.2 [23.5] | 35.2 [31.9] | 6.4 [3.5] | **3.5 [1.3]** | **0.0** |
| B2@end | 21.9 [24.2] | 33.0 [30.0] | 39.3 [39.3] | 1.8 [6.7] | **4.1 [1.4]** | **0.0** |
| C1@end | 29.5 [28.6] | 38.5 [35.3] | 20.0 [20.5] | 9.8 [7.5] | **2.2 [2.0]** | **0.0** |

Across every arm and depth, **`encoders + team_transformer` carry 51–74%** of the first-order
off-slice KL, the **`projection_mlp` a further 11–39%**, and the **pointer head 0.5–5.6%**. The
critic is exactly zero, as it must be — `log π` does not read it.

**2d. Paired contrasts** (cluster bootstrap over the 8 untaught teams, one shared index set):

| contrast | actual Δ | CI95 | verdict |
|---|---|---|---|
| `R4DOSE3 − R4DOSE12` | +0.1637 | [+0.0721, +0.2732] | **SEPARATES** |
| `TCFUNDA − TCUNFA` | +0.0431 | [−0.0202, +0.0942] | NOT DETECTED |
| `TCFUNDB − TCUNFB` | +0.0333 | [−0.0248, +0.1023] | NOT DETECTED |
| *(replicate)* `TCFUNDA − TCFUNDB` | +0.0307 | [−0.0198, +0.0778] | NOT DETECTED |
| *(replicate)* `TCUNFA − TCUNFB` | +0.0208 | [−0.0490, +0.1195] | NOT DETECTED |
| `C1 − B2` | −0.0564 | [−0.1494, +0.0563] | NOT DETECTED |
| `B2 − R4DOSE3` | −0.0701 | [−0.1996, +0.0241] | NOT DETECTED |

Pooling the two legs (`pooled_2x2.json`): funded 0.41425 vs unfunded 0.37607, gap **+0.0382
[−0.0067, +0.0928]** — and the funded replicate gap is **+0.0307**, the unfunded **+0.0208**. The
funded-vs-unfunded off-slice displacement is **the same size as the seed-to-seed difference**, and
its CI straddles zero: **NOT DETECTED**. (Taught: +0.0725 [−0.0085, +0.1522], floor +0.0269 — also
NOT DETECTED, though one leg alone, `TCFUNDB − TCUNFB` = +0.0841 [+0.0088, +0.1649], separates.
One of two legs separating while the pooled contrast does not is not a result.)

**2e. `C1 − B2` reproduces the reuse batch's own non-result.** That artifact measured
−0.0245 [−0.0841, +0.0267]; this probe, on a different state distribution, measures
−0.0564 [−0.1494, +0.0563]. Both straddle zero, both in the same direction. The meter is silent
about C1-vs-B2 in both places.

---

### TABLE 3 — cosine between displacements

| pair | ALL | encoders | team_tr | proj_mlp | belief | action_head | critic |
|---|---|---|---|---|---|---|---|
| **REPLICATE floor** funded A · funded B | **0.5630** | 0.7343 | 0.7341 | 0.2747 | 0.7993 | 0.5507 | 0.7058 |
| **REPLICATE floor** unfund A · unfund B | **0.5742** | 0.7315 | 0.7381 | 0.2646 | 0.8163 | 0.5385 | 0.7216 |
| cross: funded A · unfunded A | 0.5259 | 0.6860 | 0.6816 | 0.2175 | 0.7814 | 0.4770 | 0.7113 |
| cross: funded B · unfunded B | 0.5310 | 0.6823 | 0.6859 | 0.2205 | 0.7892 | 0.4720 | 0.7113 |
| cross: funded A · unfunded B | 0.5261 | 0.6861 | 0.6847 | 0.2201 | 0.7813 | 0.4741 | 0.6914 |
| cross: funded B · unfunded A | 0.5298 | 0.6872 | 0.6844 | 0.2188 | 0.7856 | 0.4781 | 0.7108 |
| reuse: B2 · C1 (loss OFF) | 0.4286 | 0.5444 | 0.5469 | 0.0738 | 0.7064 | **0.1202** | 0.7168 |
| reuse: dose 2.12× · dose 0.53× | 0.5140 | 0.6763 | 0.6619 | 0.3141 | 0.6774 | 0.5594 | 0.6712 |
| CROSS-BATCH (not one experiment) TCFUNDA · B2 | 0.3106 | 0.3738 | 0.3539 | 0.1048 | 0.4946 | 0.2958 | 0.6102 |

Depth rotation, `cos(arm@p1M, arm@end)`: **0.615 / 0.624 / 0.615 / 0.622** on ALL parameters — a
fold's own direction turns by ~52° over its own run.

**The exact pairing permutation.** Four arms admit exactly **three** splits into two pairs, so the
smallest attainable p is **1/3**. The observed (replicate) pairing has the highest within-pair mean
cosine in **every** group — but that is rank 1 of 3:

| group | within (obs) | cross (obs) | gap | rank | exact p |
|---|---|---|---|---|---|
| ALL | 0.5686 | 0.5282 | +0.0404 | 1/3 | 0.333 |
| encoders | 0.7329 | 0.6854 | +0.0475 | 1/3 | 0.333 |
| team_transformer | 0.7361 | 0.6842 | +0.0519 | 1/3 | 0.333 |
| projection_mlp | 0.2696 | 0.2192 | +0.0504 | 1/3 | 0.333 |
| belief_op | 0.8078 | 0.7844 | +0.0235 | 1/3 | 0.333 |
| action_head | 0.5446 | 0.4753 | +0.0693 | 1/3 | 0.333 |
| critic | 0.7137 | 0.7062 | +0.0075 | 1/3 | 0.333 |

---

## VERDICT

### P1 — funded arms move the trunk more; C1 moves least; the head is a small share

**SPLIT: two-thirds confirmed, one-third REFUTED, and the framing is wrong on the axis that
matters.**

* **Funded > unfunded at matched depth and matched dose: CONFIRMED, clearing the replicate floor**
  in `encoders` (+4.5% relative), `team_transformer` (+4.2%), `projection_mlp` (+5.6%),
  `action_head` (+8.9%) and ALL (+3.5%); WITHIN FLOOR in `belief_op` and `critic`. The four arms are
  genuinely dose-matched (all `--fork-lr-freeze` at 4.557e-08), so this is a clean within-experiment
  contrast. But note the size: the funded/unfunded *content* difference buys a **3.5%** larger step
  in a fold that moved 2.9% of `|θ|`.
* **"C1, the loss-off control, moves least": REFUTED, and not marginally.** C1 has the
  **second-largest displacement of all seventeen** (`|Δθ| = 7.995`, against B2's 8.075 and the whole
  2×2's 6.51–6.76). With its distillation loss switched off entirely, C1 travelled **1.23× as far as
  a funded fold**. Its dose was 1.20× B2's, which explains essentially all of it — and that is the
  point: **displacement magnitude is set by the dose, not by the distillation term.** PPO does the
  moving.
* **"The pointer head is a small share everywhere": CONFIRMED as stated (1.9–3.4% of `|Δθ|²`) —
  and the statement is measuring the wrong thing.** The head is 1.8% of the parameters, so a 2.5%
  share is *nothing happening*. On the size-corrected measure the **action head is the group that
  moves MOST**, `|Δθ_g|/|θ_g|` ≈ 0.076 at end depth, against the **encoders' 0.011 — the group that
  moves LEAST, by a factor of seven**. The predecessor found the encoders carry 52% of the
  policy-gradient norm and the pointer head 0.66%. Those two facts together are the real finding
  here: **the group that receives the most gradient moves the least, and the group that receives
  almost none moves the most.** Under AdamW that is exactly what per-coordinate second-moment
  normalisation does — it divides the gradient out — so gradient-norm share and displacement share
  are close to unrelated quantities, and any argument that reasons from one to the other is
  unsound. This probe was set up on the predecessor's recommendation to "instrument the trunk
  because that is where the norm is"; the displacement says the norm was never the right guide.

### P2.1 — the first-order projection ranks the arms like the measured off-slice KL

**NOT SUPPORTED — but the ACTUAL recomputation reproduces the published ordering exactly, and that
is the more valuable result.**

`KL₁` gets ρ = **+0.800** (exact p = 0.067, 8 of 120 relabellings) — directionally right, but it
swaps two adjacent pairs (`R4DOSE12`/`R4DOSE6`, and `B2`/`R4DOSE3`), and 0.067 does not clear.
Raw `|Δθ|` does worse still: ρ = +0.700, p = 0.117.

The **actual forward `KL(parent‖arm)`** reproduces the published clustered ordering **exactly**:
ρ = **+1.000**, exact p = **0.0083** (1 of 120), on the untaught slice — and ρ = +0.900 on taught.
That is a genuine independent replication of `offline_collateral_kl.py`'s ordering: different
piloting (greedy, not stochastic), different opponent, different seeds, different state count.
`offline_collateral_kl`'s ordering is a property of the arms, not of its draw.

The ladder is the clean statement: **`|Δθ|` 0.70 → `KL₁` 0.80 → actual 1.00**. Each layer of
modelling buys ordering fidelity, and only the full forward pass gets there. A first-order
surrogate is **not** a substitute for recomputing the KL.

**The registered "way to be wrong" fired, and it explains the gap.** Per-state linearity is
moderate and **degrades monotonically with displacement**: at +1M steps Pearson +0.37…+0.63 /
Spearman +0.68…+0.79, at +4.5M steps Pearson **+0.26…+0.46** / Spearman +0.52…+0.69. The error is a
systematic **over-prediction** growing with `|Δθ|` — `KL₁/actual` rises 1.17 → 1.75 (TCFUNDA),
1.35 → 1.81 (TCFUNDB), and reaches **2.06** for B2 on the taught slice. A quadratic form
extrapolated over a 4.5M-step chord over-states the divergence by up to 2×, and it over-states it
*most* for the arms that moved furthest — which is precisely why it mis-ranks B2 (largest `|Δθ|`,
largest `KL₁`, only 4th-largest actual damage) and R4DOSE3 (mid `|Δθ|`, largest actual damage).

At the **arm** level the surrogate is nonetheless a decent meter: across all 17 arm-depths,
`KL₁` vs actual is Pearson +0.906 / Spearman +0.926 on untaught (`|Δθ|` alone: +0.813 / +0.877).
Most of that is depth, though — three depths of four arms span a wide `|Δθ|` range — so it
demonstrates the surrogate tracks *magnitude*, not that it discriminates *arms at matched depth*,
where the paired contrasts above show it does not.

### P2.2 — the group decomposition puts the off-slice movement in the TRUNK, not the head

**CONFIRMED, and this is the strongest result in the probe.**

`encoders + team_transformer` carry **51–74%** of the first-order off-slice KL in every arm at every
depth; adding `projection_mlp` takes it to **~90%**. The **pointer head carries 0.5–5.6%**, and the
critic exactly 0.0% (the decomposition closes to 7.1e-15, so this is arithmetic, not an estimate).

This reproduces the predecessor's norm-share finding in an independent statistic — and upgrades it
from *a hypothetical gradient step at the parent* to *the displacement that actually happened*.
Whatever transmits a fold's collateral to untaught states, it is not the action-scoring layer.

Note the reconciliation with P1, because the two look contradictory and are not: the action head
moves the *most* relative to its own weights and contributes the *least* to off-slice KL. It is a
small, heavily-moved output map sitting on a trunk whose representation the untaught states share.
The head's motion is largely *along directions the untaught states do not excite*; the trunk's much
smaller relative motion is *shared* by construction.

### P3 — direction: high cosine in the encoders, lower where teacher content differs

**NOT DETECTED for the localisation claim; and the headline is the floor itself.**

* **The replicate floor is enormous.** Two arms differing only in random seed — same 16 teams, same
  teacher content, same frozen dose, same pin, same pool — share a displacement cosine of only
  **0.563 / 0.574** on all parameters. Under a shared-drift model (`d + n₁` vs `d + n₂` with
  independent `n`), that puts ~**75%** of the displacement length in a reproducible direction and
  ~**66%** in seed noise. Together with the measured **`|Δθ| ∝ t^0.48`** growth, the picture is
  coherent and worth stating plainly: **a fold's weight displacement is substantially a random
  walk.**
* **Funded vs unfunded is a ~7% perturbation on top of that.** All four cross cosines (0.526–0.531)
  sit below both replicate cosines, in **all seven** groupings — but with four arms there are only
  three pairings, so the exact permutation p is **1/3 = 0.333** everywhere. The point estimate is
  consistent; the design cannot resolve it. **NOT DETECTED.**
* **The localisation prediction fails.** Predicted: high in encoders, lower where teacher content
  differs. Observed ordering of replicate cosine is `belief_op` 0.81 > `team_transformer` 0.736 ≈
  `encoders` 0.733 > `critic` 0.714 > `action_head` 0.545 > **`projection_mlp` 0.270**; and the
  funded-vs-unfunded drop is roughly **uniform** across groups (−0.024 to −0.069), not concentrated
  anywhere. Encoders are high, but so is nearly everything; the content difference is not localised.
* **`projection_mlp` is the anomaly worth following.** It holds the **largest share of `|Δθ|²`
  (~40%)** and has by far the **lowest replicate cosine (0.270)** — two seeds' displacements there
  are nearly orthogonal. Most of a fold's displacement mass sits in its least reproducible group.
* **Turning the loss OFF changes direction far more than changing teacher content does.**
  `B2 · C1` = 0.429 on ALL and **0.120 in the action head**, against a funded-vs-unfunded 0.53 /
  0.475. The distillation term barely changes how *far* a fold goes (C1 travelled 0.99× B2's
  distance at 1.20× the dose) but substantially changes *which way* — and it does so most in the
  action head, the group that contributes least to off-slice KL. That is a suggestive pairing for
  the ledger's "the LOSS channel carries both the gift and the leak", but B2 and C1 are not
  dose-matched, so it is a lead and not a measurement.

### One-paragraph summary

A fold's weight displacement is dominated by its **dose**, not by its distillation term: the
loss-off control C1 travelled 1.23× as far as either half of the dose-matched 2×2, and
`|Δθ| ∝ t^0.48` — a random-walk law — with a seed-to-seed direction cosine of only 0.57. Against
that floor, the funded-vs-unfunded teacher-content difference is a real but tiny +3.5% in step
length (clearing the replicate floor in the trunk and the head, WITHIN FLOOR in belief and critic)
and an unresolvable −0.04 in direction (exact p = 1/3, the design's floor). The predecessor's
localisation finding **survives and strengthens**: 51–74% of the first-order off-slice KL sits in
`encoders + team_transformer`, ~90% with the projection MLP, and **0.5–5.6% in the pointer head**.
But the probe also **breaks the predecessor's inferential bridge**: gradient-norm share does not
predict displacement — under AdamW the encoders receive 52% of the policy-gradient norm and move
*least* of any group (0.011 relative), while the pointer head receives 0.66% and moves *most*
(0.076). Finally, the first-order surrogate is **not** a substitute for the real thing: it ranks the
arms at ρ = 0.80 (p = 0.067) and over-predicts by up to 2× in a way that grows with displacement,
whereas simply recomputing the actual forward KL reproduces the published off-slice ordering
**exactly** (ρ = 1.000, exact p = 0.0083) on an entirely different state distribution — which is
itself a clean independent validation of `offline_collateral_kl.py`.

---

## CAVEATS — what this measurement cannot say

1. **First order is measurably poor at this displacement, and the probe says by how much.**
   `KL₁/actual` runs 1.17–1.81 (2.06 worst) and the per-state Pearson falls from ~0.6 at +1M steps
   to ~0.3 at +4.5M. Nothing that depends on `KL₁` as a *level* should be believed; its usable
   content is ordering, and even there it is beaten by the forward pass.
2. **AdamW breaks the gradient→displacement inference in BOTH directions.** Registered in advance,
   and it turned out to be the load-bearing caveat: the per-coordinate second moment divides the
   gradient out, so P1's relative-displacement ranking is close to the *inverse* of the
   predecessor's gradient-norm ranking. No sentence here licenses a claim about gradient magnitude,
   and none about optimizer-step direction either — `Δθ` is the integral of *preconditioned* steps.
3. **`Δθ` is a chord, not a path.** Each arm restarted several times; `θ_end − θ_parent` is the net
   displacement of a trajectory that certainly travelled further. The √t growth is a statement about
   the chord.
4. **P3's design floor is 1/3.** Two replicate pairs is the minimum that defines a floor at all and
   it cannot produce a conventional p-value. A funded/unfunded direction claim needs ≥3 seeds per
   condition.
5. **The two batches are not one experiment.** The reuse arms (B2/C1/R4DOSE*) and the 2×2 differ in
   pin, pool, teacher set and launch batch; B2 and C1 additionally ran with a free KL controller and
   are **not** dose-matched to each other (1.20× apart) or to the 2×2. Every cross-batch cosine and
   comparison above is labelled descriptive.
6. **Greedy piloting, and a different state distribution from the artifact it is compared to.** The
   actual-KL *levels* here (0.37–0.56 untaught) are not on the same scale as
   `offline_collateral_kl`'s (0.31–0.44) and were never expected to be. Only the ORDERING was
   compared, and only the ordering should be quoted.
7. **Eight untaught teams.** Every cluster CI is over 8 clusters. That is enough to resolve the dose
   axis (`R4DOSE3 − R4DOSE12` separates in both meters) and not enough to resolve funded-vs-unfunded
   or C1-vs-B2 — the same resolution limit the reuse batch recorded.
8. **Composition of the two team sets is not matched** (inherited from the predecessor: the untaught
   8 are balance/stall-heavy and cannot contain hyper_offense). Taught and untaught are reported
   separately and never differenced.
9. **Weight-space distance is not function-space distance.** Table 1 and Table 3 are facts about
   `θ`; only Table 2's actual-KL column is a fact about behaviour. The gap between them is visible
   in the data — B2 has the largest `|Δθ|` and only the 4th-largest off-slice KL.

## Where a follow-up should look

Two things this probe makes cheap and one it makes necessary.

* **The necessary one: a replicate-seeded direction experiment.** P3's 1/3 floor is a design limit,
  not a resolution limit. Three seeds per condition would give 10 pairings and a p as low as 0.1;
  four would give 35. Given that the seed-to-seed cosine is 0.57, *any* future claim that two folds
  "went in different directions" needs this control.
* **`projection_mlp` is the unexamined mass.** 42% of the parameters, ~40% of the displacement,
  ~30% of the first-order off-slice KL, and a replicate cosine of **0.27**. It is simultaneously the
  biggest and the least reproducible thing a fold moves, and nothing in this program has probed it.
* **The cheap one: `KL₁` per-group is now a live instrument.** `project.py` computes an exact
  additive per-group attribution of off-slice divergence for any (parent, arm) pair in ~90 s with no
  battles. Pointed at a fold *in progress* — the predecessor's item (b), "the kernel measured along
  a fold rather than at its start" — it would say which group is leaking, while there is still time
  to anchor it.
