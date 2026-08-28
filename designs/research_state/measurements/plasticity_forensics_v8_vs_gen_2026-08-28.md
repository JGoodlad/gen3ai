# Plasticity forensics — v8-era distill loop vs the gen-era R2 fleet (2026-08-28)

**Question.** The v8-era generalist converted its exploiter forks into +69 anchored ELO; the
current-era R2 fleet converted its supply at ~13% (+1.6pp pooled). The *plasticity account* says
the difference is the PARENT: a converged parent's forks specialize without renovating shared
machinery, so their deltas are near-additive and distill back cheaply, while a plastic parent's
forks rewire the trunk and their content is expensive to transfer.

**Answer: the plasticity account is NOT supported.** Two of its four pre-registered predictions
are refuted *in the opposite direction*, and the other two are mixed. What the data do show is a
**supply** problem rather than a **transfer** problem: the R2 forks never became behaviourally
distinct at all — their policy moved no further from the parent than a no-fork control's did, and
moved by the same amount on their pinned teams as on held-out teams.

All numbers here are read-only over `models/`, CPU-only, era-pinned. Every model was loaded and
forwarded under the git commit its run recorded.

---

## Verdict table

| | Prediction | Verdict | Deciding numbers |
|---|---|---|---|
| **P1** | v8 parent markedly LESS trainable than rev-1 final | **REFUTED (opposite)** | Lyle capacity_ratio on `pi_features`: v8 parent **1.154**, rev-1 **0.948**. The *converged* parent shows no capacity loss; the *plastic* one shows mild loss. Cross-check: canonical `main.capacity` gives rev-1 **0.9483** vs my **0.9480**. |
| **P2** | fork-vs-parent feature drift SMALLER for v8, esp. trunk/encoder vs head | **MIXED** | CKA distance at matched fork-length. Supports on the trunk: `projection` **0.0146 (v8) → 0.0491 (v9), 3.4×**; `pi_features` 0.0296→0.0507. Contradicts on the value head: `mlp.value_net` **0.0985 (v8) > 0.0653 (v9)**. Ties at the two earliest phases: `pokemon_encoder` 0.0014 vs 0.0017, `team_transformer` 0.0094 vs 0.0101. |
| **P3** | v8 deltas in late/head; R2 deltas spread into trunk/encoders | **REFUTED (opposite)** | Share of total ‖ΔW‖², matched length. Trunk: **v8 0.47 vs v9 0.28**. Aux/belief heads: **v8 0.10 vs v9 0.46**. Encoders tied (0.061 vs 0.072). v8 is the era that puts its delta in the shared trunk (G2+G5 = 0.76 of Δ²); R2 puts it in aux heads. |
| **P4** | fork-vs-parent KL/agreement low for v8 (esp. off-slice), high for R2 | **MIXED** | Off-slice KL supports: v8 **0.190** vs v9 **0.309** (1.63×). Top-1 agreement contradicts: v8 **0.657** vs v9 **0.730** — the v8 teachers changed their chosen action *more often*. On the as-distilled artifacts v8 is further on both (KL 0.395, top-1 0.557). |

**Headline mechanism the data support instead.** The R2 forks' policy shift is (a) the same
magnitude as a genuine no-fork continuation's (‖shift‖ 231–257 vs control **259**), (b)
**undifferentiated between their pinned teams and held-out teams** (top-1 agreement 0.69–0.77
on-slice vs 0.71–0.74 off-slice), and (c) no more mutually aligned than it is aligned with that
control (inter-fork cos **+0.383**, fork-vs-control cos 0.29–0.44). The v8 teachers, by contrast,
show exactly the specialization signature the R2 fleet lacks: their argmax agreement with the
parent **drops on their own slice** (0.417–0.519) relative to off-slice (0.543–0.577).

---

## Provenance — what was compared

Both triples were resolved by reading `metadata.json` `original_command` chains.

| Era | Parent | Forks | Fold product | Control |
|---|---|---|---|---|
| **v8** | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` (**277,178,472** steps) | `ai_v8_06_semistall_3team`, `ai_v8_09_pool10`, `ai_v8_13_defensive10` | `ai_v8_14_distill3_0725` | **MISSING** (no no-fork continuation of ai_v8_04 exists) |
| **gen (v9)** | `ai_v9_29_rev1_0823/final_model.zip` (**25,067,760** steps) | `ai_v9_53..57_R2F5a..e` | `ai_v9_59_R2ACTION_0827` | `ai_v9_58_R2CTRL_0827` — verified a genuine plain continuation (`--exploiter` ABSENT, `--trainee-teams` ABSENT, `--distill-coef 0.0`) |

**Step-matching.** The v8 teachers ran 7.4–18.7M steps past the fork; the R2 fleet ran **3.047M**.
Comparing final-vs-final would confound era with fork-training length, so every v8 fork was also
read at the checkpoint nearest the R2 fleet's length — `checkpoint_280748576` / `280728930` /
`280716057`, i.e. Δ = 3.13–3.15M steps vs the R2 fleet's Δ3.05M (a 1.03–1.04× mismatch). **All
cross-era comparisons below are at matched length**; final-checkpoint rows are shown separately
and labelled.

**Era code.** Current code cannot load v8 checkpoints (obs 2992 vs 2501, different module set).
Forwards were run in two detached worktrees pinned to the runs' own commits —
`b13b30b289c5eaba136a930a4ab63451e209fbe5` (v8) and `77f922e742e2e1c6441bcaa4ec20f97c85a0e8c9`
(gen) — with `PYTHONPATH` pointed at the era `src/` so it beats the editable install. Feature
matrices were dumped as `.npy` under era code and all statistics computed afterwards from
current-code pure-NumPy estimators (`agents.model.capacity_probes`,
`agents.training.rank_metrics.effective_rank`).

**State sets.** Each era's own `eval_traces`, from the PARENT's trace step nearest the fork point
(`ai_v8_04/step_276000000`; `ai_v9_29/step_24000000`), pooled across all opponents and battles,
seeded subsample **n = 3000** per era. These are ~99% off-slice by construction (the parents
rarely drew the pinned teams: 0–26 of 3000). On-slice sets (n=1500/fork) come from each fork's own
last trace step, which is dominated by its pins (2 distinct teams for each R2 fork; 1 for each v8
teacher).

**Acid test — the input reconstruction is exact.** The traces record only `obs`; the Dict
observation has ~15 further auxiliary channels. Those were filled with "unknown" defaults, and the
reconstruction was then validated by forwarding the *snapshot the trace itself shipped* and
comparing to the recorded logits:

| Era | max abs Δ logits | corr | top-1 agreement |
|---|---|---|---|
| v8 | 1.07e-05 | 0.9999999999996 | **1.000** |
| gen | 1.91e-05 | 0.9999999999996 | **1.000** |

Float32 noise. The auxiliary channels do not affect the forward, so every feature/logit matrix
below is faithful.

---

## Phase A — weight-delta location (scores P3)

Per-group ‖ΔW‖_F / ‖W_parent‖_F and share of total squared delta, deduping SB3's three saved
copies of the shared extractor (`features_extractor` / `pi_` / `vf_`). Group map derived from the
actual key prefixes of both eras: **G1** embeddings + pokemon_encoder + entity_seats +
history_events + zarch_encoder + edge_bias + damage_op + assembler + status/outgoing/refine/prefuse
projections · **G2** projection + team_transformer + pre_proj_norm + film · **G3** cls_pool +
value_entity_pool + value_projection + value_pre_norm · **G4** all belief/aux/intent/cf/win/
value_dist heads · **G5** mlp_extractor · **G6** action_net (v8) / pointer_head (v9) + value_net.

### Share of total ‖ΔW‖², at matched fork length

| group | v8 semistall3 | v8 pool10 | v8 defensive10 | **v8 mean** | gen F5a–e | **gen mean** | param share v8 / gen |
|---|---|---|---|---|---|---|---|
| G1 encoders | 0.064 | 0.061 | 0.061 | **0.062** | 0.069–0.079 | **0.072** | 0.057 / 0.070 |
| **G2 trunk** | 0.450 | 0.471 | 0.473 | **0.465** | 0.263–0.291 | **0.278** | 0.328 / 0.283 |
| G3 pools/value-trunk | 0.075 | 0.074 | 0.069 | 0.073 | 0.053–0.060 | 0.056 | 0.245 / 0.098 |
| **G4 aux/belief heads** | 0.117 | 0.098 | 0.097 | **0.104** | 0.439–0.490 | **0.462** | 0.069 / 0.197 |
| G5 mlp extractor | 0.284 | 0.290 | 0.295 | 0.290 | 0.107–0.121 | 0.113 | 0.299 / 0.334 |
| G6 action/value head | 0.011 | 0.006 | 0.005 | 0.007 | 0.018–0.019 | 0.018 | 0.002 / 0.018 |
| **global ‖Δ‖/‖W‖** | 0.0210* | 0.0210 | 0.0209 | **0.021** | 0.0253–0.0268 | **0.026** | |

\* semistall3 read at Δ3.57M. Global relative movement is 1.23× larger for the R2 fleet at a
3–4% *shorter* fork length — the fleet does move more overall.

**But the LOCATION is the opposite of P3.** v8 puts **76%** of its squared delta into the shared
trunk + mlp (G2+G5); the R2 fleet puts **39%** there and **46%** into aux/belief heads. The
encoders are essentially untouched in both eras (6–7%). P3 predicted v8 in the head and R2 in the
trunk/encoders; measured, v8 is the trunk-renovating era.

**Caveat, and it is load-bearing:** G4 holds 6.9% of v8's parameters but 19.7% of the gen-era's,
and the gen-era aux heads include recently-enabled substrate heads that were still actively
learning. The R2CTRL control shows a G4 share of **0.345** — i.e. most of that aux-head movement
is ordinary continued training, not anything the fork did. Correcting for it does not rescue P3
(it removes mass from the R2 column's G4 without adding any to its trunk).

### Delta geometry between sibling forks (weight level)

Mean pairwise cosine between sibling fork delta vectors (reference 0 = orthogonal; k-independent):

| group | v8 (k=3) | gen (k=5) |
|---|---|---|
| ALL | +0.195 | +0.224 |
| G1 encoders | +0.302 | +0.256 |
| G2 trunk | +0.133 | +0.104 |
| G4 aux heads | +0.563 | +0.343 |
| G5 mlp | +0.100 | +0.042 |
| **G6 action/value head** | **+0.891** | **+0.130** |

The G6 row looks decisive and **should not be read as one**: v8's G6 is a 5.6k-parameter linear
`action_net`, the gen-era's is a 55k-parameter structured `pointer_head`. That is an
era-asymmetric comparison of different objects. The era-robust version of the same question is
measured at the *function* level in Phase C, and it comes out **flat** (+0.375 vs +0.383) — so the
weight-level G6 contrast is an artifact of the head redesign, not a finding.

**The ctrl-subtracted variant of this table is discarded as an artifact.** Subtracting the R2CTRL
delta made every group's inter-fork cosine collapse onto ≈+0.73 uniformly, because
‖ctrl‖ (0.0370) exceeds ‖fork‖ (0.0253) while cos(fork,ctrl) is only +0.109 — so the subtraction
dominates every vector and manufactures the agreement. Reported here only so it is not re-derived.

---

## Phase B — feature geometry (scores P2)

Linear CKA distance (1 − CKA) between fork and parent representations on the **same 3000 states**,
at matched fork length. Hooks fired three times per batch (`extract_features` /`get_distribution` /
`predict_values`); the repeats were verified byte-identical (`max_repeat_diff = 0.000e+00`) before
deduping.

| phase | v8 mean | gen mean | gen / v8 |
|---|---|---|---|
| `pokemon_encoder` | 0.0014 | 0.0017 | 1.21 |
| `team_transformer` | 0.0094 | 0.0101 | 1.07 |
| **`projection`** | **0.0146** | **0.0491** | **3.36** |
| `cls_pool` | 0.0486 | 0.0699 | 1.44 |
| `pi_features` | 0.0296 | 0.0507 | 1.71 |
| `vf_features` | 0.0661 | 0.0640 | 0.97 |
| `mlp.policy_net` | 0.0796 | 0.0904 | 1.14 |
| **`mlp.value_net`** | **0.0985** | **0.0653** | **0.66** |

P2 holds on the policy-side trunk (the `projection` tap drifts 3.4× more for the R2 fleet, and
`pi_features` 1.7×) and fails on the value head (v8 drifts 1.5× more). The two earliest phases are
**tied and near zero in both eras** — neither era's forks renovate the encoders, which is the part
of P2's "especially trunk/encoder" framing that simply does not happen anywhere.

### Effective rank (participation ratio, 512-wide taps)

| tap | v8 parent | v8 forks | gen parent | gen forks |
|---|---|---|---|---|
| `projection` | 30.08 | 30.20 | 13.52 | 13.43 |
| `pi_features` | 50.24 | 49.53 | 20.59 | 20.15 |
| `vf_features` | 3.85 | 4.09 | 3.24 | 3.44 |

Forking changes rank by ≤0.5 in either era — **no rank collapse from specialization**. The
gen-era running at roughly half the v8 participation ratio at equal width is a generation
difference (different obs, arch and 25M vs 277M steps), not a fork effect, and is not evidence for
or against any of P1–P4.

---

## Phase C — function drift (scores P4)

KL(parent‖fork) in nats and top-1 agreement, off-slice shared states, n=3000. v8 logits are
recorded unmasked so the unmasked column is the era-symmetric one; the masked column is shown for
the gen era to confirm masking does not change the verdict (it does not — it moves both eras the
same way).

| model | KL (masked) | KL (unmasked) | top-1 (masked) | V corr |
|---|---|---|---|---|
| **matched length** | | | | |
| v8 semistall3 | 0.1857 | 0.1857 | 0.6597 | 0.771 |
| v8 pool10 | 0.1995 | 0.1995 | 0.6793 | 0.954 |
| v8 defensive10 | 0.1862 | 0.1862 | 0.6327 | 0.952 |
| **v8 mean** | **0.190** | 0.190 | **0.657** | |
| gen F5a–F5e | 0.269–0.349 | 0.366–0.464 | 0.709–0.742 | 0.934–0.956 |
| **gen mean** | **0.309** | 0.399 | **0.730** | |
| gen **R2CTRL** (no fork) | **0.3245** | 0.4557 | **0.7300** | 0.953 |
| **as-distilled finals** | | | | |
| v8 semistall3 / pool10 / defensive10 | 0.333 / 0.447 / 0.405 | — | 0.577 / 0.543 / 0.552 | |
| **v8 final mean** | **0.395** | | **0.557** | |
| v8 product (ai_v8_14) | 0.291 | | 0.601 | 0.803 |
| gen product (R2ACTION) | 0.415 | 0.557 | 0.756 | 0.945 |

**The control row is the most important line in this table.** R2CTRL — a run with no exploiter and
no pinned teams — sits at KL 0.3245 / top-1 0.730, i.e. **inside the range of the forks it is meant
to be a control for** (0.269–0.349 / 0.709–0.742). Three million steps of ordinary continued
training moves the policy exactly as far from rev-1 as forking onto two pinned teams does.

### Is a fork doing anything the control is not?

Cosine between each fork's mean-centred logit shift and the control's, on the same states:

| fork | ‖shift‖ | cos(fork, ctrl) | residual fraction |
|---|---|---|---|
| R2CTRL | 258.76 | 1.000 | — |
| F5a | 246.85 | 0.299 | 0.954 |
| F5b | 232.53 | 0.289 | 0.957 |
| F5c | 230.66 | 0.318 | 0.948 |
| F5d | 256.69 | 0.435 | 0.900 |
| F5e | 236.60 | 0.400 | 0.917 |
| R2ACTION (product) | 296.27 | 0.111 | 0.994 |

So the forks *are* moving in partly their own directions (90–96% of each shift is orthogonal to
the control's) — but **no further than the control moves**, and their mutual agreement (+0.383,
below) is no higher than their agreement with a run that never had an exploiter. There is no
distinctive shared "exploit direction" in the fleet's supply.

### Inter-fork agreement of the policy shift — the era-robust additivity test

| | v8 (k=3) | gen (k=5) |
|---|---|---|
| mean pairwise cos of (fork − parent) logit shift | **+0.375** | **+0.383** |
| range | 0.358–0.406 | 0.276–0.506 |

**Flat.** At the function level the two eras' sibling forks agree with each other to the same
degree. Whatever explains the transfer-efficiency gap, it is *not* that v8's teachers pointed in a
more common direction. (This is the measurement that overturns the weight-level G6 reading above,
and it is the one to trust — it compares behaviour, not two differently-shaped head modules.)

### On-slice vs off-slice — where the eras genuinely differ

| fork | on-slice KL | off-slice KL | **on-slice top-1** | **off-slice top-1** |
|---|---|---|---|---|
| v8 semistall3 | 0.390 | 0.333 | **0.519** | 0.577 |
| v8 pool10 | 0.384 | 0.447 | **0.496** | 0.543 |
| v8 defensive10 | 0.361 | 0.405 | **0.417** | 0.552 |
| gen F5a | 0.342 | 0.330 | 0.768 | 0.741 |
| gen F5b | 0.434 | 0.298 | 0.692 | 0.741 |
| gen F5c | 0.369 | 0.269 | 0.751 | 0.742 |
| gen F5d | 0.369 | 0.349 | 0.725 | 0.709 |
| gen F5e | 0.322 | 0.300 | 0.736 | 0.717 |

Every v8 teacher changes its chosen action **more** on its own pinned teams than off them
(agreement 0.417–0.519 vs 0.543–0.577). **No R2 fork does** (0.692–0.768 vs 0.709–0.742 — the sign
is inconsistent and the gap is within noise). The R2 fleet's measured +12pp on-slice / −8pp
off-slice is therefore being produced by a policy change that is *not* selective for the teams it
was trained on: an undifferentiated shift that happens to pay on two teams and cost elsewhere.
That is a team-LOCAL adjustment with no team-identity gating, and it is consistent with this
project's existing finding that count dominates conditioning.

*Caveat:* on-slice states come from each fork's own trace distribution while off-slice states come
from the parent's, so the on/off contrast is confounded with a state-distribution shift. The
contrast is nevertheless drawn identically in both eras, and it is the era **difference** that is
being read.

---

## Phase D — trainability (scores P1)

Lyle-style capacity probe as this project defines it: ridge OOF fit of frozen features to 8 fixed
random targets `tanh(signed_log(obs) @ N(0,1)/√D)`, 5 folds; `capacity_ratio = r²_trained /
r²_fresh`, where the fresh arm is a same-config extractor built through **each era's own
production init path** (SB3 orthogonal init, then `restore_identity_init()` where it exists — it
exists in gen-era code and does not in v8-era code, which is the correct era-faithful behaviour and
matches the known "SB3 ortho-init clobbered every zero-init pre-2026-08-01" record). Two fresh
seeds, averaged. Shared per-era state set, n=3000.

| tap | model | r² trained | r² fresh | **capacity_ratio** |
|---|---|---|---|---|
| `pi_features` | **v8 parent** | 0.4071 | 0.3527 | **1.154** |
| | v8 semistall3 / pool10 / defensive10 | 0.413 / 0.414 / 0.411 | " | 1.170 / 1.173 / 1.164 |
| | v8 product | 0.3804 | " | 1.079 |
| | **gen parent (rev-1)** | 0.4076 | 0.4300 | **0.948** |
| | gen F5a–F5e | 0.409–0.414 | " | 0.951–0.963 |
| | gen product | 0.4192 | " | 0.975 |

A ratio below 1 is Lyle capacity loss. **The converged 277M-step v8 parent shows none (1.154); the
plastic 25M-step rev-1 shows mild loss (0.948).** P1 predicted the reverse, and predicted it
*markedly*. The forks track their parents in both eras (v8 1.164–1.173, gen 0.951–0.963), so this
is a property of the parent, not of forking.

**Cross-validation.** The project's own `python -m main.capacity` on rev-1, using its own
independent state sampler, returns `pi_features` capacity_ratio **0.9483** against my **0.9480** —
agreement to 3 decimals through a separate code path. Its full v9 table:

| run | role_tokens | team_tokens | value_pooled | pi_features | vf_features |
|---|---|---|---|---|---|
| rev-1 parent | 0.9940 | 0.9631 | 0.9399 | 0.9483 | 0.8739 |
| R2F5a | 0.9707 | 0.9848 | 0.9769 | 1.3631 | 0.8808 |
| R2ACTION | 1.0134 | 0.9865 | 0.9605 | 1.1014 | 0.7403 |

⚠ The canonical battery re-samples states **per run** from that run's own eval_traces, so its
fresh arm moves between rows (r²_fresh 0.4669 / 0.3319 / 0.3933) and its cross-run column is not a
clean comparison — which is why the shared-state-set numbers above are primary and this table is a
cross-check on the rev-1 cell only.

⚠ **The `vf_features` tap is UNRELIABLE in the gen era and its P1 numbers are withheld.** Gen-era
`vf_features` is near-singular — condition number ~1e16 with 20–25 dead (constant) columns, against
v8's ~1.3e4 — and the ridge accordingly returned nonsense for two forks (r² = −15.99, −3.41).
That is a property of the tap's conditioning, not a trainability reading. `pi_features` is
well-conditioned in both eras (cond 3.5e3 gen / 9.6e2 v8) and is the tap P1 is scored on.

---

## MISSING cells

| Cell | Reason |
|---|---|
| **v8 no-fork control** | No plain continuation of `ai_v8_04` was ever run. The gen era's R2CTRL is the single most informative row in Phase C and has **no v8 counterpart**, so "was the v8 fork delta also indistinguishable from ordinary continued training?" is unanswerable from the archive. This is the biggest gap in the comparison and the one that would most change the conclusion. |
| **v8-era canonical `main.capacity`** | `capacity_probes.py` / `main/capacity.py` postdate `b13b30b`. Worked around by running the identical estimator (current pure-NumPy `random_targets`/`ridge_oof`/`r2_columns`) over era-code feature dumps, validated against the canonical tool on the rev-1 cell (0.9480 vs 0.9483). Not a gap in the measurement, only in the tooling path. |
| **Full-network gradient-descent Lyle probe** | Not run. P1 is scored on the *linear-readout-on-frozen-features* proxy, which is what this project's canonical instrument implements. A few-hundred-step torch probe would test a strictly stronger notion of trainability and is not guaranteed to agree. |
| **gen-era on-slice states from the parent's distribution** | rev-1's traces contain 0–17 decisions per fork's pinned teams out of 3000, so the on-slice half of P4 had to come from the forks' own traces. |
| **P2 parent-probe transfer** | Optional in the brief; not run (CKA + rank answered P2 and the ridge machinery was spent on P1). |

## Caveats

- **Era asymmetry is irreducible.** The two eras differ in obs (2992 vs 2501), module set (v8 has
  FiLM/zarch; gen has the event window, intent/cf/pair-outcome heads and a pointer head), parent
  maturity (277M vs 25M steps), fork count (3 vs 5) and pinned-team count (3–10 vs 2). Every
  cross-era statistic reported is dimensionless (a ratio, a cosine, a CKA, a share) for this
  reason, but no dimensionless statistic makes those differences vanish.
- **Fork length matched to 1.03–1.04×, not exactly.** v8 Δ3.13–3.15M vs gen Δ3.05M. Final-length
  rows are labelled and never mixed into a matched comparison.
- **Group mapping is a judgement.** G1–G6 were assigned by module-name prefix (mapping listed in
  full above); modules unique to one era necessarily land in one column only, which is why the
  param-share column is printed beside every Δ² share.
- **v8 off-slice contamination.** Three of ten `pool_cluster_564` team files parse to <4 species
  (nicknamed/itemless lines) and two pairs are duplicates, so a handful of v8 states labelled
  off-slice may be on-slice. At ≤26/3000 the effect is negligible.
- **Trainability targets are era-specific** by construction (they are random functions of the raw
  obs, whose dimension differs). The ratio to a same-era fresh net is the comparable quantity; the
  raw r² columns are not comparable across eras and the fresh arms do differ (0.353 vs 0.430).
- **Logit masking.** v8 traces record unmasked logits and carry no `action_mask`; masks were
  reconstructed as all-legal for v8. The unmasked column is therefore the era-symmetric metric, and
  the gen-era masked/unmasked pair is shown to demonstrate the choice does not flip any verdict.

## Reproduction

Scripts under `tmp/` in this worktree (untracked — `tmp/` is gitignored):
`plast_phaseA.py` (weight deltas), `plast_phaseA2.py`
(delta geometry), `plast_build_states.py` / `plast_build_onslice.py` (state sets),
`plast_forward.py` + `plast_fresh.py` (era-code forwards — run with
`PYTHONPATH=/tmp/plasticity_{v8,v9}era/src`), `plast_analyze.py` (Phases B/C/D).
Every raw number is in the sibling `.json`.
