# The capacity-eval battery — saturation as a regular reading

**What it is.** `python -m main.capacity <run_dir_or_checkpoint.zip>` — an offline, read-only
battery over one checkpoint. It emits a printed table and a strict-JSON artifact
(`<run>/capacity_battery.json` by default) designed to be **differenced against another
generation at matched step**. Measured 2026-08-23 on an idle box: **7.0 s** at the default 3000
states, ~2 s of which is the two extractor passes.

**Why it exists.** The flywheel era piles distilled skills into ONE fixed-capacity trunk.
Conditioning was closed by two independent nulls (`project_lut_conditioning_ceiling_result`,
`project_code_rank_ceiling`), so there is no FiLM and no LoRA to grow into — the network we have
is the network the whole flywheel runs on. Saturation therefore needs to be **visible before a
long fruitless hunt**, not diagnosed after one. This is a tripwire, not a verdict.

---

## 🚨 The validity discipline — read this before quoting any number

This project once derived a "conditioning headroom" claim from a **low participation ratio**
(`PR(K_ū)=17`). The reading was a **noise artifact**; the lever built on it was refuted, twice, on
two orthogonal manipulations. The minted lesson:

> **Gate a lever on "does this quantity PREDICT performance?", never on "is it low?"**

Three rules follow, and they are non-negotiable:

1. **No kill/build decision from any number in this artifact, alone.** An alarm licenses an
   *investigation*, never a change.
2. **A single generation's reading is uncalibrated.** Every metric here is meaningful only as a
   **difference** between two artifacts at matched step, on the same battery version.
3. **Every alarm needs PAIRED BEHAVIOURAL evidence.** The `validity` block is shipped *inside*
   every artifact so a reader six months from now cannot get the number without the caveat.

---

## The four probes

### (a) Representation effective rank

Participation ratio (`pr`) and `srank99`, centered, at five taps in pipeline order:
`role_tokens` (post-encoder) → `team_tokens` (post-transformer) → `value_pooled` →
`pi_features` / `vf_features`. Token taps are ranked over the `[N·12, D]` token population, which
is the same definition the **live** `rank/trunk_*` training metric uses — the two are
differenceable because the estimator is literally the same function
(`agents.training.rank_metrics.effective_rank`).

⚠️ **`srank99` is the VARIANCE-percentile spelling** (`n99`: leading dims holding 99% of the
variance), the one already in production. It is *not* Kumar et al.'s singular-value-threshold
srank. The two are named alike in the literature and are different numbers.

Reported alongside a **fresh** arm: the same-config extractor, randomly initialised through the
production path (SB3 orthogonal init **then** `restore_identity_init()` — a bare constructor is
not the network training starts from; see `src/agents/model/CLAUDE.md` → *Identity-at-init is NOT
free*). The fresh column is a scale reference, not a target.

| | |
|---|---|
| **movement means** | A FALLING PR / srank99 at matched step across generations = the representation collapsing into fewer directions — the classic capacity-loss precursor. |
| **confirm with** | Anchored ELO at matched snapshot COUNT (`ladder.json`, end-of-run — never a mid-run `eval/elo`), plus a per-skill retention read on the distilled behaviours. Rank falling while ELO and retention hold is a representation getting more EFFICIENT. |
| **no verdict** | `PR(K_ū)=17` was "low", was noise, and its lever was refuted. |

### (b) Trainability / capacity loss (Lyle et al.)

*Understanding and Preventing Capacity Loss in RL.* Fit a cross-validated linear (ridge) head from
the **frozen** features to **K=8 fixed random target functions of the obs**, and compare against
the identical probe on the fresh arm.

`capacity_ratio = r2_trained / r2_fresh`. `nmse = 1 − r2` (normalized MSE against the target's own
variance), so the target scale cancels.

**The target family is deliberately batch-statistic-free:**
`t_k(x) = tanh( signed_log(x) · w_k / sqrt(D) )`, `w_k ~ N(0,1)` from `seed`. Two properties are
load-bearing:

* **No z-scoring.** A z-scored target would be silently re-defined by every generation's own eval
  traces — same recorded `seed`, different function, unreadable difference. Pinned by
  `test_random_targets_use_no_batch_statistics`.
* **The `signed_log` squash is not cosmetic.** Raw obs mixes ~600-valued embedded IDs with 0-1
  fractions; unsquashed, the projection is ~entirely the ID columns and `tanh` saturates into a
  near-binary target no head can fit gradually. Pinned by
  `test_random_targets_are_not_saturated_by_wide_scale_columns`.

| | |
|---|---|
| **movement means** | A ratio falling well below 1 **and continuing to fall** generation over generation = the trained features have lost the ability to express new functions. |
| **confirm with** | A trainability test with STAKES: a fresh exploiter fork's learning curve on a NEW skill, from this generation's base vs an earlier generation's base at matched step. That is the behavioural claim; this ratio is only its cheap proxy. |
| **no verdict** | A low ratio licenses running the exploiter-fork test. Nothing else. The trained net is not *supposed* to match a random net at random targets — only the TREND is readable. |

### (c) Probe decodability

OOF linear probes (ridge; AUC for the two binary facts) from each tap to ground-truth facts read
**out of the obs vector itself** through `Gen3ObservationEncoder.get_layout()` — no hardcoded
index anywhere. Eight facts: our/opp active HP, our/opp alive count, our/opp spikes,
`clock_elapsed`, `turns_since_progress`. A degenerate target (zero variance, or a positive rate
outside [0.02, 0.98] on this sample) is **skipped with a printed reason**, never silently fitted.

These are **deliberately easy**. A high r²/AUC is the expected reading and says nothing.

| | |
|---|---|
| **movement means** | DRIFT: an established fact becoming less decodable at matched step = capacity being reallocated away from it. |
| **confirm with** | **Decodable ≠ used**, and this repo has the measurement: the belief-latent role-geometry probe found SPECIES geometry strongly decodable and the move-id table not at all, and neither predicted whether the head HELPED. A decodability drop needs an ablation or an intervention showing the policy's behaviour actually depends on the dropped fact. |
| **no verdict** | — |

### (d) Parameter census

Per-phase `n_params` / `param_share` / `l2_norm` / `rms` / `zero_frac`, from the parameters alone.
**`rms` is the comparable column** — a raw norm grows with width, so two phases of different size
are not differenceable on it.

| | |
|---|---|
| **movement means** | A phase whose `rms` is ~unchanged from init while every sibling has grown is not learning; `zero_frac ≈ 1.0` on a zero-init route means it never left identity. |
| **confirm with** | Weight norm is not usage. `critic_route_audit` / `edge_ablation_audit` and the per-edge-family liveness metrics price a route in |dV| / argmax-flip terms. This census only says where to point them. |
| **no verdict** | — |

---

## Reading two artifacts

1. **Match the step**, not the wall clock or the snapshot index.
2. **Match `battery_version`.** It is bumped whenever an estimator's DEFINITION changes; two
   artifacts at different versions measured different things and must not be differenced.
3. **Match `seed`.** It seeds the state subsample, the random targets and the CV folds together.
4. **Verify the sampling coverage, do not trust it.** `meta.sampling` carries per-step and
   per-opponent SAMPLED-ROW counts from the shared stratified sampler
   (`agents.model.audit_states.collect_states` — round-robin over step dirs × opponents, then a
   seeded row subsample). The sampler exists because the old sorted-glob drew every state from one
   lexically-first step dir while labelling itself a pool average.
5. **`meta.arch_signature` must match on both sides.** The battery loads through
   `load_model_snapshot` → `check_compatible`, so an arch-drifted run exits 1 with the mismatch
   NAMED and writes no artifact. `current_version` is built from the saved config's own arch
   TOGGLES, which makes the check exactly the drift question: *given this run's flags, does
   today's code build the same architecture?*

---

## Baseline — gen-17 `ai_v9_26_baitent_probe_0823` (leg B), 2026-08-23

`legB_final_model.zip`, step 36,175,872, `arch gen3_critic_route_wave_v1`, 3000 states from 167
traces (5 step dirs × 14 opponents), seed 0, battery v1, 7.0 s. **This row is a REFERENCE POINT,
not a finding** — nothing below is a verdict on anything.

```
(a) tap             dim       PR  PR/dim  srank99  PR_fresh  sr99_fresh
    role_tokens     128    14.59   0.114       86     13.31          85
    team_tokens     128    16.16   0.126      105      5.14          83
    value_pooled    128     2.47   0.019       32      5.54          54
    pi_features     512    16.01   0.031      331      9.34         190
    vf_features     512     3.05   0.006       91      6.58         125

(b) tap             r2_trained  r2_fresh   nmse_tr  nmse_fresh   ratio
    role_tokens         0.4287    0.4230    0.5713      0.5770   1.014
    team_tokens         0.4859    0.4919    0.5141      0.5081   0.988
    value_pooled        0.4556    0.4828    0.5444      0.5172   0.944
    pi_features         0.4621    0.5008    0.5379      0.4992   0.923
    vf_features         0.4445    0.4913    0.5555      0.5087   0.905

(c) fact                 role_tok      team_tok      value_pool     pi_feat       vf_feat
    our_active_hp      0.435|0.309   0.613|0.536   0.613|0.534   0.950|0.629   0.598|0.511
    opp_active_hp      0.396|0.310   0.609|0.213   0.603|0.233   0.832|0.264   0.531|0.262
    our_alive          0.990|0.970   0.973|0.951   0.972|0.956   0.965|0.974   0.981|0.964
    opp_alive          0.952|0.952   0.912|0.928   0.873|0.937   0.856|0.956   0.883|0.949
    our_spikes (AUC)   1.000|1.000   0.995|0.982   0.984|0.976   0.978|0.998   0.981|0.984
    opp_spikes (AUC)   1.000|1.000   0.983|0.979   0.974|0.975   0.979|0.997   0.973|0.979
    clock_elapsed      0.970|0.967   0.928|0.920   0.905|0.900   0.914|0.947   0.912|0.916
    turns_since_prog   0.253|0.336   0.292|0.280   0.275|0.273   0.466|0.887   0.341|0.292

(d) 1,973,350 params — projection 30.6% (rms 0.046) · team_transformer 14.4% (0.113) ·
    cls_pool 10.2% (0.094) · hidden_opp_belief 10.1% (0.100) · pokemon_encoder 8.2% (0.099).
    Only two phases carry meaningful zeros: value_projection 0.8%, move_belief 3.8%.
```

Three things a reader will notice, stated as **observations with their confounds**, not as
findings:

* **`value_pooled` PR = 2.47 of 128, and it is BELOW its own fresh arm (5.54).** This is the
  critic's known low-rank regime — the same one the plasticity audit measured at ~7× below the
  policy, and concluded was the **steady state of a scalar objective** rather than stiffness
  (`project_plasticity_null`: the critic RE-EXPANDS; resets and ReDo bought nothing). It is
  therefore *expected here*, and this artifact is not new evidence about it. The reason it is
  still worth carrying: that note also says **richer targets should raise it, and this probe is
  the meter** — so if the flywheel's distillation targets ever do their job, this is the cell
  that should move.
* **The trainability ratio has a monotone shape along the pipeline** (role 1.014 → team 0.988 →
  value_pooled 0.944 → pi 0.923 → vf 0.905): the deeper the tap, the more specialised. That is
  what a trained network is *supposed* to look like at one point in time. **It carries no
  information about capacity loss until there is a second generation to difference it against.**
* **`turns_since_progress` at `pi_features` reads 0.466 trained vs 0.887 fresh** — the one cell
  where the random network beats the trained one by a wide margin, and the most interesting thing
  in the table. It is *a priori* the shape of capacity reallocation. It is also *a priori* the
  shape of a fact the policy correctly stopped caring about. Rule (3) applies: an ablation or
  intervention decides, not this number.

---

## Honest limitations of v1

* **Token taps are MEAN-POOLED for probes (b) and (c).** The rank probe (a) sees the full
  `[N·12, D]` token population, but a state-level target needs a state-level row, and the mean is
  the cheapest one that keeps the tap comparable across generations. It discards per-mon structure,
  so a role/team-token number here is a **lower bound** on what those tokens carry.
* **The fresh arm is ONE seed.** It is not an ensemble, so its column carries a sampling error the
  artifact does not quantify. Read `capacity_ratio` trends, not single-generation values.
* **The ridge is LINEAR.** Capacity loss that shows only under a nonlinear head is invisible here.
  Lyle et al.'s stronger form fine-tunes an MLP; that is a v2 lever, deliberately not built,
  because a nonlinear probe's own optimisation becomes a second explanation for every movement.
* **The state sample is the run's OWN eval traces.** Two generations' samples come from different
  battles against a shifting opponent pool, so a decodability difference partly reflects a
  DISTRIBUTION difference. The random-target family is immune to this by construction (rule above);
  the fact probes are not. A shared fixed state set across generations is the honest fix and is not
  built.
* **`meta.sampling` records counts, not filenames.** The selection is exactly reproducible from
  `(patterns, max_states, seed, sampler version)`, but the artifact does not enumerate which traces
  were read.
* **`opp_alive` counts obs slots with `hp > 0`, including UNREVEALED opponents.** That is what the
  obs says, not the true remaining party. Deliberate — the probe measures what the network can
  recover from its input.

---

## Where the code lives

| file | holds |
|---|---|
| `src/agents/model/capacity_probes.py` | the engine — every estimator, pure NumPy, unit-testable without a checkpoint |
| `src/main/capacity.py` | the CLI — checkpoint resolution, model load + drift diagnosis, the table, the artifact |
| `src/agents/model/capacity_probes_test.py` | the math pins (30 tests, 0.6 s) |
| `src/main/capacity_test.py` | resolution / render / strict-JSON / validity-block pins (14 tests, 0.5 s) |

Reused rather than rebuilt: `agents.training.rank_metrics.effective_rank` (one rank definition
shared with the live `rank/*` training metric) and `agents.model.audit_states.collect_states` (the
stratified sampler the route/edge audits already use). `ridge_oof` is new and does not duplicate
the prober's `fit_probe`: that helper is single-target and re-solves per (fold, λ), while the
battery fits ~16 targets × 5 taps × 2 arms and shares ONE economy SVD per fold across every target
and every penalty. Same estimator, same OOF-selected penalty; different factorisation.
