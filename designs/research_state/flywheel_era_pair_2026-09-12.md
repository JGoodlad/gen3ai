# THE FLYWHEEL-ERA PAIR — registration (2026-09-12)

> **Status: REGISTERED, NOT LAUNCHED.** Both argvs are built, validated by execution and frozen
> below. The Training Run session launches them verbatim from the files named in §3. Nothing in
> this document has seen a number from either arm — it is written before the first step.
>
> **Standing rule 1** (pre-register both branches, the bar and the comparator, before any number
> exists) is what this file is for. **Standing rule 11**: every arm code carries its human
> description at every use.

---

## 1. Purpose — the question, in one sentence

**At full length, on one pin, one dose, one seed-slot, one eval regime and one architecture: does a
win-probability critic (`--critic winprob` — V(s) *is* the terminal win indicator's BCE head) beat
the era's hand-shaped distributional critic (`--critic shaped` — V(s) is the distributional E[Z]
over a PBRS-shaped return, PopArt-normalised), on strength, on the critic's own conditioning rows,
and on the untaught meter?**

The owner's pivot of 2026-09-12 wound down search and named PPO effectiveness as the object of
study; the asymmetric critic (`--value-true-team`, the privileged ceiling probe) stays OFF in both
arms. A full-length pair is the instrument the 10M ladder could not be: fifteen 10M arms and seven
levers produced **zero confirmed detections on a decision row** [ledger 2026-09-12 · *THE CRITIC
LADDER READ ON STRENGTH*; *VERDICT · strata_b FAILS*], and the one lever that did confirm
(`--vf-coef 1.5`, `vf15_b`) moved the row DOWNWARD. Length is the axis the ladder never bought.

Two arms:

| code | run name | human description |
|---|---|---|
| **arm S** | `ai_v13_01_flywheel_shaped` | **the era's SHAPED configuration** — the registered baseline `v9_long_baseline`'s reward and critic settings, transplanted onto the current pin and the current ladder ecology |
| **arm W** | `ai_v13_02_flywheel_winprob` | **the win-prob critic** — `ai_v12_11_ladder_ctrl10M`'s argv (the era's control), at full length |

**The third leg, free:** `ai_v12_02_winprob_critic` — the existing 75M win-prob run at
`--ent-coef 0.02`. It is *not* a matched control (different pin, different entropy coefficient,
mixed eval regime) and is read as a descriptor only; see §8.4.

### 1.1 🚨 Why arm S is NOT `ctrl10M`'s argv with `--critic` flipped

Two one-flag-flip shaped controls were run on 2026-09-12 and **both are VOID**:

* `ai_v12_26_ladder_ctrl10M_shaped` — reached 10,027,008 and **NEVER CROSSED** into self-play
  (`pool_snapshot_count` 0 at all five cycles, `bots8` ends 0.5312, below the 0.55 gate).
* `ai_v12_27_ladder_ctrl10M_shaped_dense` — also **NEVER CROSSED** (bots8 0.531 at 10M), and the
  campaign's **first G7 breach** (worst 1.522 = 121.7 % of bar).

The era's own shaped fresh run, `ai_v9_29_rev1_0823` (= the registry baseline `v9_long_baseline`),
**crossed at 2,000,016 — its first eval cycle** — and promoted from 2M on. So the shaped head *as
the era ran it* does not learn slowly; what learns slowly is `ctrl10M`'s argv with the mode flipped:
terminal-indicator ON, draw-penalty 0, victory-value 1.0, hand shaping OFF, PopArt OFF, no
distributional head, gamma 1.0, the win-prob head kept as an auxiliary at coefficient 1.0.
[ledger 2026-09-12 · *READ · the shaped-critic family at the LADDER argv is a curriculum failure*]

**Therefore arm S reproduces the ERA's reward composition**, not a flag flip. §2.2 is the proof
that it does, key by key.

---

## 2. Provenance — how each arm was built

Nothing here was hand-copied. Every value was resolved through a registry name, then through the
run's own recorded `metadata.json`, then through the current parser (rule 9: quote a baseline from
the RESOLVED file, never from an argv).

### 2.1 The chain

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.baselines                      # v9_long_baseline = ai_v9_29_rev1_0823@25,067,760
python -m main.lineage models/ai_v9_29_rev1_0823          # role=fresh, root, no fork parent
# then metadata.json's original_command (IMMUTABLE block) and model_config.json (config v101)
```

* `v9_long_baseline` → `ai_v9_29_rev1_0823`, `final_model.zip`, commit `d78aa810`, config v101,
  `arch_signature gen3_critic_route_wave_v1`, 25,067,760 steps,
  sha256 `2f17fbab…1025717`. Purpose on file: *"the gen-era FRESH from-zero 25M run, anchored
  ladder 2098 at 12 nodes — the long-run reference every gen-era arm is read against."*
* Arm W's base → `ai_v12_11_ladder_ctrl10M`'s `original_command` (pin `f3502568`), the ladder
  family's control argv.

Arm S = **arm W's argv with the reward-and-critic block replaced by the era's**. Arm W = **arm W's
base argv with the four matched-variable edits**. Both were produced by a script operating on the
two recorded `original_command` strings; no token was retyped.

### 2.2 🚨 The proof that arm S reproduces the era's composition

Arm S's argv was resolved through the launch path's own functions
(`build_parser` → `desugar_umbrella_flags` → `resolve_critic_mode` → `_resolve`'s shaped
`gamma` default) and compared key-by-key against `ai_v9_29_rev1_0823/model_config.json`:

| key | era (v101, recorded) | arm S (resolved, HEAD) | |
|---|---|---|---|
| `use_popart` | True | True | ✓ |
| `win_prob_mode` | shaping | shaping | ✓ |
| `win_prob_coef` | 0.05 | 0.05 | ✓ |
| `value_dist_mode` | shaping | shaping | ✓ |
| `value_dist_bins` | 51 | 51 | ✓ |
| `value_dist_vmin` | −12.0 | −12.0 | ✓ |
| `value_dist_vmax` | 12.0 | 12.0 | ✓ |
| `value_dist_coef` | 1.0 | 1.0 | ✓ |
| `value_from_dist` | True | True | ✓ |
| `value_tail_weight` | 0.3 | 0.3 | ✓ |
| `draw_penalty` | −35.0 | −35.0 | ✓ |
| `vf_coef` | 0.5 | 0.5 | ✓ |
| `no_progress_penalty` | 0.15 | 0.15 | ✓ |
| `mat_alive_weight` | 1.25 | 1.25 | ✓ |
| `self_ko_hp_penalty` | 0.0 | 0.0 | ✓ |
| `switch_bias_weight` | 0.0 | 0.0 | ✓ |
| `bias_additivity` | 1.0 | 1.0 | ✓ |
| `belief_grad_mode` | shaping | shaping | ✓ |
| `edge_bias_families` | `d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r` | *identical* | ✓ |

**Reward/critic keys recorded at v101 that differ: ZERO.**

Five fields the era ran but **v101's schema did not record** — their era value is the then-default,
and arm S states each one EXPLICITLY rather than relying on a default that has since moved:

| key | era value, and why | arm S |
|---|---|---|
| `critic` | `shaped` — `--critic` did not exist at `d78aa810`; `cbcb0bfb` introduced it as a MODE beside *the unchanged `shaped` default* (config v109) | `--critic shaped` (typed) |
| `hand_shaping` | True — the flag existed and defaults ON; the era argv did **not** pass `--no-hand-shaping` | `--hand-shaping` (typed) |
| `terminal_indicator` | False — argparse default, "and every generation to date" | `--no-terminal-indicator` (typed) |
| `victory_value` | 30.0 — argparse default, `= reward_weights.VICTORY_VALUE` | `--victory-value 30.0` (typed) |
| `gamma` | 0.9999 — hardcoded at the era pin; today `_resolve("gamma", reward_weights.PBRS_GAMMA)` under `shaped` | left UNSET, resolves to 0.9999 — *deliberately*, see §4 |

Confirmed live in the arm-S smoke banner (§7.3):
`🎯 [CRITIC] shaped — V(s) = the distributional E[Z] in raw shaped-return units (PopArt ON), gamma=0.9999; the win-prob head is an auxiliary at --win-prob-coef 0.05.`

### 2.3 Flag drift between the era pin (`d78aa810`) and HEAD (`6eb9c776`)

Tested by **execution**, not by reading a changelog: the era's recorded `original_command`, 229
tokens, was replayed through `python -m main.checkargs --argv "…"` on the current parser.

```
checked 132 flags from --argv
  accepted by the trainer parser : 131
  launcher-owned (not forwarded) : 1
  unrecognized                   : 0
  refused combinations           : 1  ✗  --distill-team-bias > 0 requires --distill-teacher
  ARCH SURFACE vs designs/production_config.json  [production_config@360f8378dd90]
    ✓ every ARCH-surface key matches the production mirror
```

**Result: not one flag of the era's reward or critic block was deleted, renamed, or changed arity.**
`designs/deleted_flags.md` lists no reward-family flag. Resolution, flag by flag:

| era flag | what changed between `d78aa810` and `6eb9c776` | resolved as |
|---|---|---|
| *(no `--critic`)* | `cbcb0bfb` added `--critic {shaped,winprob}`, default `shaped`. `resolve_critic_mode`: *"Under `shaped` nothing beyond the mode itself is assigned, so a run that does not type the flag is byte-identical."* | **typed `--critic shaped`** — self-documenting, byte-identical to the era's silence |
| `--gamma` | was a hardcoded 0.9999; `cbcb0bfb` made it a FLAG whose shaped default reads `reward_weights.PBRS_GAMMA` (the same constant PBRS uses, so the two cannot drift) | **left unset** ⇒ 0.9999, era-identical |
| `--use-popart` | unchanged; now carries a `combination_checks` refusal requiring an explicit `--clip-range-vf none` | satisfied — the era argv already carried it, and so does arm S |
| `--value-dist-*`, `--value-from-dist`, `--value-tail-weight` | unchanged; now REFUSED under `--critic winprob` (nine such refusals). Irrelevant under `shaped` | carried verbatim |
| `--win-prob-mode`, `--win-prob-coef` | unchanged under `shaped` (auxiliary readout). `--win-prob-coef` is now refused under `winprob` (the BCE *is* the value loss, weighted by `--vf-coef`) | carried verbatim on S; absent on W |
| `--victory-value`, `--draw-penalty`, `--terminal-indicator`, `--hand-shaping` | unchanged defaults (30.0 / −35.0 / False / True). `resolve_critic_mode` deliberately does **not** imply them — they have concrete argparse defaults, so "left alone" and "typed the default" are indistinguishable, and an implication would silently overwrite an operator's choice | **all four typed explicitly on both arms** |
| `--distill-team-bias 0.4` | its GUARD is new: `--distill-team-bias > 0` now requires `--distill-teacher`. The era argv would be REFUSED today | **not transplanted** — it is not a reward/critic flag, and the era ran it as a silent no-op (`--distill-coef 0.0`, no teacher). Noted, not carried |
| `--cf-*`, `--capacity-telemetry` | still present | **not transplanted** — telemetry, not reward; carrying them would be a confound |

🚨 **The one era property arm S CANNOT reproduce, by design:** the era ran under the *stochastic*
eval-sentinel regime (pre-2026-09-07). Arm S runs greedy+symmetric like arm W. That is a matched
variable of the pair, and it makes `win_rate_vs_pool` / `eval/elo` **not comparable to the era run**
(rule of evidence 15, §3.2 rule 5). `ladder.json` is unaffected. See §8.4.

---

## 3. The two argvs, verbatim

Both are also on disk, one line each, ready for `python -m main.launcher $(cat …)`:

```
/home/goodlad/.claude/jobs/9ab51de6/tmp/flywheel_pair/argv_S_flywheel_shaped.txt    (247 tokens)
/home/goodlad/.claude/jobs/9ab51de6/tmp/flywheel_pair/argv_W_flywheel_winprob.txt   (231 tokens)
```

> The duplicated `--device cuda`, `--log-level periodic` and `--self-play` tokens are inherited
> from `ai_v12_11_ladder_ctrl10M`'s recorded command and are **deliberately left in place**:
> argparse takes the last, and de-duplicating by hand is exactly the transcription risk this
> registration exists to avoid. Both arms carry the identical duplicates.

### 3.1 Arm S — `ai_v13_01_flywheel_shaped` (the era's shaped configuration)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.launcher \
  --device cuda --log-level periodic --belief-grad-mode shaping --beta-setvalued-coef 0.05 \
  --bias-additivity 1.0 --clip-range 0.15 --clip-range-vf none --compile-opponents \
  --compile-opponents-strict --compile-trainer --consequence-topk 6 --damage-candidate-k 0 \
  --damage-matrices both --damage-op --damage-outgoing --damage-topk 6 \
  --defensive-entropy-anneal-frac 0.0 --defensive-entropy-boost 1.0 --device cuda \
  --distill-coef 0.0 --distill-value-coef 0.0 --distill-value-feat-coef 0.0 \
  --edge-bias-families d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r --ent-coef 0.05 \
  --entity-tail-seats --entity-topk-seats 6 --eval-battles 100 --eval-concurrency-per-worker 1 \
  --eval-device cpu --eval-shard-games 25 --eval-workers 5 --exploiter-bot-fraction 0.5 \
  --exploiter-temp-anneal-frac 0.2 --exploiter-temp-end 1.0 --exploiter-temp-mode fixed \
  --exploiter-temp-ratchet-factor 0.9 --exploiter-temp-ratchet-games 500 \
  --exploiter-temp-ratchet-wr 0.55 --hp-belief-mode composed --hp-type-belief-coef 0.05 \
  --intent-move-cell --keep-crashes 10 --keep-eval-snapshots 10 --keep-eval-trace-steps 20 \
  --keep-stalls 50 --log-level periodic --lr 0.0003 --mat-alive-weight 1.25 --min-lr 1e-05 \
  --move-belief-coef 0.05 --move-belief-latent-coef 0.05 --move-belief-mode both \
  --move-candidate-floor 0.02 --move-latent --move-prior-fusion --n-envs 48 --n-epochs 10 \
  --n-sentinels 5 --n-steps 2048 --no-progress-penalty 0.15 --opd-beta 1.0 --opd-coef 0.0 \
  --opp-belief-aux-coef 0.05 --opp-belief-cls-k 6 --opp-belief-moves-weight 1.0 \
  --opp-intent-coef 0.05 --search-teacher-batch-size 256 --search-teacher-beta 1.0 \
  --search-teacher-buffer-size 20000 --search-teacher-coef 0.0 --search-teacher-value-coef 0.0 \
  --self-ko-hp-penalty 0.0 --self-play --self-play-temp 1.0 --self-play-use-cpu \
  --snapshot-ladder-games 100 --species-prior-fusion --spread-belief --spread-belief-coef 0.05 \
  --spread-belief-nature --stable-opponent-mastered-wr 0.8 --stable-opponent-selfplay-share 0.2 \
  --stable-opponent-temp 1.0 --switch-bias-weight 0.0 --t0-species-prior \
  --teacher-confirm-rollouts 8 --teacher-gen-battles 12 --teacher-refresh-steps 500000 \
  --teacher-search-budget 200 --teacher-search-freq 0 --teacher-search-workers 3 \
  --team-block-episodes 1 --team-pfsp off --team-pfsp-cap 3.0 --team-pfsp-floor 0.05 \
  --unified-damage both --unified-moves both --use-bridge rust --value-entity-pool \
  --value-threat-inject --warmstart-battles 200 --warmstart-bc-steps 4000 --weight-decay 1e-05 \
  --win-prob-mode shaping --history-events --item-belief --intent-threshold --intent-conditional \
  --op-drop-renders --op-believed-lean --value-entity-pool-full --pair-outcome-cell \
  --pair-outcome-switch --switch-branch-cell --conditional-threat-cell \
  --critic shaped --no-terminal-indicator --victory-value 30.0 --draw-penalty -35.0 \
  --vf-coef 0.5 --self-play --intent-label-bot-weight 0.25 --batch-size 2048 \
  --grad-accum-steps 32 --restart-interval-hours 3 --steps 75000000 --seed 1001 \
  --eval-sentinel-greedy --no-value-true-team --hand-shaping --use-popart \
  --value-dist-mode shaping --value-dist-bins 51 --value-dist-vmin -12.0 --value-dist-vmax 12.0 \
  --value-dist-coef 1.0 --value-from-dist --value-tail-weight 0.3 --win-prob-coef 0.05 \
  --pin-commit 6eb9c776940ed6040ddb90acfbc28b038457fc42 --run-name ai_v13_01_flywheel_shaped
```

### 3.2 Arm W — `ai_v13_02_flywheel_winprob` (the win-prob critic)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.launcher \
  --device cuda --log-level periodic --belief-grad-mode shaping --beta-setvalued-coef 0.05 \
  --bias-additivity 1.0 --clip-range 0.15 --clip-range-vf none --compile-opponents \
  --compile-opponents-strict --compile-trainer --consequence-topk 6 --damage-candidate-k 0 \
  --damage-matrices both --damage-op --damage-outgoing --damage-topk 6 \
  --defensive-entropy-anneal-frac 0.0 --defensive-entropy-boost 1.0 --device cuda \
  --distill-coef 0.0 --distill-value-coef 0.0 --distill-value-feat-coef 0.0 \
  --edge-bias-families d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r --ent-coef 0.05 \
  --entity-tail-seats --entity-topk-seats 6 --eval-battles 100 --eval-concurrency-per-worker 1 \
  --eval-device cpu --eval-shard-games 25 --eval-workers 5 --exploiter-bot-fraction 0.5 \
  --exploiter-temp-anneal-frac 0.2 --exploiter-temp-end 1.0 --exploiter-temp-mode fixed \
  --exploiter-temp-ratchet-factor 0.9 --exploiter-temp-ratchet-games 500 \
  --exploiter-temp-ratchet-wr 0.55 --hp-belief-mode composed --hp-type-belief-coef 0.05 \
  --intent-move-cell --keep-crashes 10 --keep-eval-snapshots 10 --keep-eval-trace-steps 20 \
  --keep-stalls 50 --log-level periodic --lr 0.0003 --mat-alive-weight 1.25 --min-lr 1e-05 \
  --move-belief-coef 0.05 --move-belief-latent-coef 0.05 --move-belief-mode both \
  --move-candidate-floor 0.02 --move-latent --move-prior-fusion --n-envs 48 --n-epochs 10 \
  --n-sentinels 5 --n-steps 2048 --no-progress-penalty 0.15 --opd-beta 1.0 --opd-coef 0.0 \
  --opp-belief-aux-coef 0.05 --opp-belief-cls-k 6 --opp-belief-moves-weight 1.0 \
  --opp-intent-coef 0.05 --search-teacher-batch-size 256 --search-teacher-beta 1.0 \
  --search-teacher-buffer-size 20000 --search-teacher-coef 0.0 --search-teacher-value-coef 0.0 \
  --self-ko-hp-penalty 0.0 --self-play --self-play-temp 1.0 --self-play-use-cpu \
  --snapshot-ladder-games 100 --species-prior-fusion --spread-belief --spread-belief-coef 0.05 \
  --spread-belief-nature --stable-opponent-mastered-wr 0.8 --stable-opponent-selfplay-share 0.2 \
  --stable-opponent-temp 1.0 --switch-bias-weight 0.0 --t0-species-prior \
  --teacher-confirm-rollouts 8 --teacher-gen-battles 12 --teacher-refresh-steps 500000 \
  --teacher-search-budget 200 --teacher-search-freq 0 --teacher-search-workers 3 \
  --team-block-episodes 1 --team-pfsp off --team-pfsp-cap 3.0 --team-pfsp-floor 0.05 \
  --unified-damage both --unified-moves both --use-bridge rust --value-entity-pool \
  --value-threat-inject --warmstart-battles 200 --warmstart-bc-steps 4000 --weight-decay 1e-05 \
  --win-prob-mode shaping --history-events --item-belief --intent-threshold --intent-conditional \
  --op-drop-renders --op-believed-lean --value-entity-pool-full --pair-outcome-cell \
  --pair-outcome-switch --switch-branch-cell --conditional-threat-cell \
  --critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0 --draw-penalty 0 \
  --vf-coef 0.5 --self-play --intent-label-bot-weight 0.25 --batch-size 2048 \
  --grad-accum-steps 32 --restart-interval-hours 3 --steps 75000000 --seed 1001 \
  --eval-sentinel-greedy --no-value-true-team \
  --pin-commit 6eb9c776940ed6040ddb90acfbc28b038457fc42 --run-name ai_v13_02_flywheel_winprob
```

---

## 4. The resolved-config diff — every differing key, labelled

Both argvs were parsed and resolved through the launch path's own functions
(`build_parser` → the explicitness snapshot → `desugar_umbrella_flags` → `resolve_critic_mode` →
`resolve_eval_sentinel_regime`), then diffed.

**283 resolved keys. 16 differ. 15 are the treatment or forced by it; 1 is the run's identity.
ZERO confounds.**

| key | arm S | arm W | label | why |
|---|---|---|---|---|
| `critic` | `shaped` | `winprob` | **TREATMENT** | this *is* the experiment |
| `hand_shaping` | True | False | **TREATMENT** | the reward composition the critic objective implies. The two smokes print the difference in one line: arm S `1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)`, arm W `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)` |
| `terminal_indicator` | False | True | **FORCED** | a `[0,1]` critic needs the return to be `V·1{win}`; `combination_checks` REQUIRES `--terminal-indicator` under `winprob` and the era ran it OFF |
| `victory_value` | 30.0 | 1.0 | **FORCED** | under `winprob` the return must be the win *indicator*, so V = P(win) exactly; 30.0 is the era's scale that `MAT_HP_WEIGHT`/`MAT_ALIVE_WEIGHT` are calibrated against |
| `draw_penalty` | −35.0 | 0.0 | **FORCED** | with `terminal_indicator` ON the draw ordering is *inapplicable, not merely inert* — a `[0,1]` critic cannot represent "a timeout is worse than a loss". −35.0 is the era's validated value |
| `gamma` (resolved) | 0.9999 | 1.0 | **FORCED** | `resolve_critic_mode` implies 1.0 under `winprob` — an IDENTITY, not a tuning (at γ=1 with a terminal-only reward V(s) is *exactly* P(win\|s)). Under `shaped`, `_resolve("gamma", reward_weights.PBRS_GAMMA)` = 0.9999, which PBRS's own γ must equal for the potentials to be policy-invariant. **Neither value is choosable given the mode.** Left unset on both arms so the implication is visible in the banner rather than asserted in an argv |
| `use_popart` | True | False | **FORCED** | `resolve_critic_mode` implies False under `winprob`, and `winprob_critic_refuses_popart` refuses an explicit True: a bounded stationary Bernoulli payoff has no scale to track, and `_denorm` would take V out of `[0,1]` |
| `value_dist_mode` | `shaping` | *(unset ⇒ none)* | **FORCED** | `winprob_critic_refuses_value_dist`: a 51-atom categorical over a two-valued return *is* a Bernoulli with 50 redundant DoF — a SECOND critic |
| `value_dist_bins` | 51 | — | **FORCED** | with the above |
| `value_dist_vmin` | −12.0 | — | **FORCED** | with the above |
| `value_dist_vmax` | 12.0 | — | **FORCED** | with the above |
| `value_dist_coef` | 1.0 | — | **FORCED** | with the above |
| `value_from_dist` | True | — | **FORCED** | `winprob_critic_refuses_value_from_dist`: both flags name WHICH readout is the critic and they name different ones |
| `value_tail_weight` | 0.3 | 0.0 | **FORCED** | the per-sample tail weighting of the distributional value loss; no distributional loss exists on W |
| `win_prob_coef` | 0.05 | *(unset)* | **FORCED** | under `shaped` the win-prob head is an AUXILIARY readout at 0.05 (the era's dose); under `winprob` its BCE *is* the value loss and is weighted by `--vf-coef` — `winprob_critic_refuses_win_prob_coef` refuses an explicit value |
| `run_name` | `ai_v13_01_flywheel_shaped` | `ai_v13_02_flywheel_winprob` | **IDENTITY** | not a config |

### 4.1 The matched variables, stated

| variable | value, both arms | note |
|---|---|---|
| `--pin-commit` | `6eb9c776940ed6040ddb90acfbc28b038457fc42` | current `main` HEAD at registration time. All four 2026-09-12 landings are on it. `--dry-run` echoes its subject line |
| `--steps` | 75,000,000 | |
| `--seed` | 1001 | the ladder family's replicate slot |
| `--ent-coef` | 0.05 | `ent05` PASSED: H_end 1.086, inside v8's 1.07–1.11 plateau, +0.29 nats = 3.9× the floor above the widest control, slope still POSITIVE at 10M [ledger 2026-09-12 · *VERDICT · ent05 PASSES*] |
| `--clip-range` | 0.15 | our value; the 0.10-vs-0.15 leg remains the one untested cross-era regime axis and is deliberately NOT varied here |
| `--n-envs` / `--batch-size` / `--n-steps` / `--n-epochs` / `--lr` / `--grad-accum-steps` | 48 / 2048 / 2048 / 10 / 3e-4 / 32 | ⇒ identical dose, §5 |
| `--vf-coef` | 0.5 | ⚠️ see hazard H4 — 0.5 multiplies a BCE on W and an MSE-over-shaped-return on S; the number transfers, the *meaning* does not |
| eval regime | `--eval-sentinel-greedy` **typed on both** | current default (`EVAL_SENTINEL_GREEDY_DEFAULT = True`); typed so the regime is in the argv, not inferred. `promote_threshold` follows to 0.55 |
| `--value-true-team` | `--no-value-true-team` **typed on both** | the privileged critic stays OFF (owner, 2026-09-12) |
| `--restart-interval-hours` | 3 | |
| `--device` | cuda | |
| `--use-bridge` | rust (default) | serverless; no Showdown server involved |

### 4.2 `--clip-range-vf` — the explicit recommendation

**Recommendation: `--clip-range-vf none`, typed on BOTH arms.** It already is (inherited from both
source argvs); this records *why* it is the right setting and not an accident.

Read at the source (`src/agents/training/instrumented_ppo/ppo.py:385`):

```python
elif critic_winprob or self.clip_range_vf is None:
    # No clipping. Under `winprob` the scalar MSE is a DIAGNOSTIC (its term is
    # dropped below), so clipping would only make `train/value_loss` read as a
    # clipped quantity in probability units.
```

* **Arm W (`winprob`)** — `clip_range_vf` is **inert by construction**: the branch short-circuits on
  `critic_winprob`, so the value is never read. `none` and `0.5` produce identical training. Typing
  `none` states that fact instead of leaving a live-looking knob in the argv.
* **Arm S (`shaped` + PopArt)** — the PopArt branch takes precedence (value loss in normalized
  space) and vf-clipping is **mutually exclusive** with it. `combination_checks`'
  `popart_needs_explicit_clip_off` REFUSES the launch unless `--clip-range-vf none` is typed:
  *"PopArt would clip in un-normalized units and cripple the critic."* The flag's default is 0.5,
  so silence here is a refusal, not a default.
* **Why the same value on both is not a coincidence worth hiding:** it is the only setting that is
  simultaneously *required* on S and *honest* on W. Any other choice would either refuse arm S or
  put a dead knob in arm W's recorded config for a future reader to mis-read.

---

## 5. The dose

```
updates_per_env_step = n_epochs / (batch_size × grad_accum_steps) = 10 / (2048 × 32) = 1.5259e-4
dose_rate            = lr × updates_per_env_step = 3e-4 × 1.5259e-4 = 4.578e-8
```

**Identical on both arms by declaration** — the four inputs are the same tokens in both argvs.
This follows the ladder family's own precedent (ledger 2026-09-11 · *AMENDMENT to the strata dose
rule*): doses are equal **by declaration**, because the realized `lr_median` is set by the KL
controller's annealing and is not a variable either argv controls.

For scale, `python -m main.dose` on the runs already on disk (reference `ai_v8_14_distill3_0725`,
2.145e-08):

| run | eff. batch | epochs | lr_median | updates/step | dose_rate | vs ref |
|---|---|---|---|---|---|---|
| `ai_v12_11_ladder_ctrl10M` | 65,536 | 10 | 3.96e-4 | 1.526e-4 | **6.042e-08** | 2.82× |
| `ai_v12_28_ladder_ent05` | 65,536 | 10 | 3.60e-4 | 1.526e-4 | **5.493e-08** | 2.56× |
| `ai_v9_29_rev1_0823` (era) | 16,384 | 10 | 1.21e-4 | 6.104e-4 | **7.359e-08** | 3.43× |

Both arms launch into the 65,536 / 1.526e-4 row. 🚨 **The era run is NOT dose-matched to arm S** —
it ran `--grad-accum-steps 8` (4× the updates per env step) and annealed to a 3× lower LR. Arm S is
the era's *reward composition* on the ladder family's *dose*, and that is stated rather than
smoothed over: any arm-S-vs-era comparison carries a dose confound. The pair's internal
comparison does not.

---

## 6. What the pair does NOT vary (and why each is deliberate)

* **Architecture / obs.** Both carry the identical 39-key ARCH surface, verified clean against
  `designs/production_config.json` on both argvs (§7.1). `family=critic` rows are excluded from the
  surface *by their own declaration* — they are "the readouts an experiment deliberately VARIES" —
  which is exactly why a critic-mode pair can have a clean surface diff.
* **Opponent ecology.** Same 8-bot pool, same `--self-play`, same `--n-sentinels 5`, same
  `--team-pfsp off`, same `--stable-opponent-*` block, same team distribution.
* **Clip range.** 0.15 on both. The cross-era 0.10-vs-0.15 leg stays untested; varying it here would
  reintroduce the collinearity `ent05` just broke.
* **`--value-true-team`.** OFF on both, typed.
* **Seed.** One slot (1001), one arm each. See §9 — this is the pair's binding limitation.

---

## 7. Validation by EXECUTION (standing rule 7)

Every command below was run from the **main checkout** at HEAD `6eb9c776`. No training was
launched; the launcher was used only with `--dry-run`; `train_rl_agent.py` only with `--debug` and
a `--run-dir` under `/home/goodlad/.claude/jobs/9ab51de6/tmp/flywheel_pair/`. Nothing was written
under `models/`.

### 7.1 `python -m main.checkargs --argv "$(cat …)"` — both arms

**Arm S:**
```
checked 141 flags from --argv
  parser                         : the CURRENT tree's — --pin-commit in the argv names HEAD (6eb9c776)
  accepted by the trainer parser : 139
  launcher-owned (not forwarded) : 2
  unrecognized                   : 0
  ARCH SURFACE vs designs/production_config.json  [production_config@360f8378dd90]
    ✓ every ARCH-surface key matches the production mirror (39 of 51 registry toggles are the ARCH
      surface; 9 critic readouts + 3 non-structural rows are excluded by their own declaration)
  ✓ this command still launches                                                      [exit 0]
```

**Arm W:**
```
checked 132 flags from --argv
  parser                         : the CURRENT tree's — --pin-commit in the argv names HEAD (6eb9c776)
  accepted by the trainer parser : 130
  launcher-owned (not forwarded) : 2
  unrecognized                   : 0
  ARCH SURFACE vs designs/production_config.json  [production_config@360f8378dd90]
    ✓ every ARCH-surface key matches the production mirror
  ✓ this command still launches                                                      [exit 0]
```

**0 unrecognized, 0 refused combinations, ARCH SURFACE clean — on both.**

### 7.2 `python -m main.launcher --dry-run …` — both arms

**Arm S** (arm W identical but for the run dir and its own name):
```
── launcher --dry-run — resolving only, nothing will be created ────────────
  role        : FRESH
  run dir     : models/ai_v13_01_flywheel_shaped   [would be created]
  pin         : 6eb9c776940ed6040ddb90acfbc28b038457fc42  (source: pin_commit)
                ↳ research: ent05 PASSES — --ent-coef 0.05 reproduces v8's entropy regime at 10M …
  steps       : --steps 75,000,000 (fresh run, from 0)
  interpreter : /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
  transport   : in-process bridge [rust] (no Showdown server)
  restarts    : every 3.0h, grace 20.0 min, max 3 crash restart(s), nice 10
  effective   : the argv IS the config (no --model, nothing to inherit)
      --grad-accum-steps     32           (from the argv)
  ARCH SURFACE vs designs/production_config.json  [production_config@360f8378dd90]
    ✓ every ARCH-surface key matches the production mirror
  ✓ DRY RUN — this command would launch. Nothing was created or modified.            [exit 0]
```

`role: FRESH` on both (no `--model`, so the empty-self-play-pool fork hazard does not apply).
The `[CRITIC]` and `[Reward] composition` banners are **child-only** and therefore do not appear in
a dry run — they are in §7.3.

### 7.3 The `--debug --steps 10000` smokes — the resolved banners

**Arm S** — `Training complete`, 42 episodes, no FATAL, no traceback:
```
⚖️  [EVAL REGIME] eval sentinels GREEDY + symmetric teams (--eval-sentinel-greedy, source=argv);
    promote_threshold=0.55 (source=default)
[Opponents] training pool = 8 bots (heuristic, heuristic2, staller, staller_v2, aggressive,
    aggressive_v2, setup_sweep, setup_sweep_v2)
[Reward] composition: 1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)
🎯 [CRITIC] shaped — V(s) = the distributional E[Z] in raw shaped-return units (PopArt ON),
    gamma=0.9999; the win-prob head is an auxiliary at --win-prob-coef 0.05.
[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))
…
Training complete. Model saved to …/flywheel_pair/smoke_S/final_model
```

**Arm W** — `Training complete`, no FATAL, no traceback:
```
⚖️  [EVAL REGIME] eval sentinels GREEDY + symmetric teams (--eval-sentinel-greedy, source=argv);
    promote_threshold=0.55 (source=default)
[Reward] composition: 1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)
🎯 [CRITIC] winprob — V(s) = sigmoid(win-prob logit) in [0,1]; the value loss IS that head's BCE
    against the terminal outcome, weighted by --vf-coef 0.5 (a BCE, NOT the shaped-return MSE 0.5
    was tuned for). Reward = the TERMINAL WIN INDICATOR alone; gamma=1; PopArt OFF;
    win_prob_mode='shaping'. At victory_value 1.0 and gamma 1.0, V(s) == P(win|s) exactly.
    ⚠️ A [0,1] critic cannot express 'a timeout is worse than a loss' — stall rate and mean episode
    length are PRIMARY endpoints on this arm (--arm-no-progress-tax is the contingency,
    currently OFF).
[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))
…
Training complete. Model saved to …/flywheel_pair/smoke_W/final_model
```

**The two composition lines are the treatment, stated by the code itself:**
`1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)` against
`1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)`.

### 7.4 🚨 Smoke deviations, declared

The smoke argv is the launch argv minus `--restart-interval-hours` / `--pin-commit` / `--run-name`
(launcher-owned or superseded), with `--steps 10000 --run-dir <tmp> --debug` appended, **plus three
forced deviations**:

1. **`--device cpu` appended.** `--debug` defaults to CPU *only when the argv is silent*; both argvs
   carry an explicit `--device cuda`, which wins. The first attempt therefore ran the smoke on the
   GPU beside the live `vf025` arm and died with `torch.OutOfMemoryError` (122 MiB short of an
   11.63 GiB card holding a 5.93 GiB training process). **The live arm was unharmed** — the OOM
   fell in the smoke's own process, and `ai_v12_29_ladder_vf025` was verified still writing
   TensorBoard events 9 s later. Subsequent smokes ran with `--device cpu` *and*
   `CUDA_VISIBLE_DEVICES=""`. **Standing lesson: `--debug` does not fence the GPU when the argv
   names a device.**
2. **`--no-compile-trainer --no-compile-opponents`.** `--compile-trainer` FATALs on CPU
   (`[CompileTrainer] FATAL: --compile-trainer requires CUDA, but the model is on 'cpu'`).
3. **`--compile-opponents-strict` dropped** (it has no `--no-` form).

🚨 **The smoke exercises a STRICTLY SMALLER surface than the real launch, and these three
deviations widen that gap.** `--debug` is one `DummyVecEnv` with no forkserver; it bypasses the
**forkserver preload**, `--compile-trainer`, `--compile-opponents` and the warm-start layer — and
the three deviations above remove the compile layer explicitly. On 2026-09-09 a `--value-true-team`
arm passed exactly this smoke and died two minutes into its real launch, in the preload.
**The first two minutes of each real launch remain the ONLY test of that layer.** Watch for the
`[CompileTrainer]` / `[CompileOpponents]` lines and the first `🏁 Episode Finished`.

### 7.5 Smoke results

| arm | outcome | steps reached | `ep_len_mean` | resolved critic |
|---|---|---|---|---|
| **S** `ai_v13_01_flywheel_shaped` | ✅ **`Training complete`** — no FATAL, no traceback | 10,240 | 30.1 | `shaped`, PopArt ON, γ = 0.9999, win-prob head auxiliary at 0.05 |
| **W** `ai_v13_02_flywheel_winprob` | ✅ **`Training complete`** — no FATAL, no traceback | 10,240 | 34.7 | `winprob`, PopArt OFF, γ = 1, V = σ(logit) ∈ [0,1] |

Both reached `[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))`, built their
envs, collected rollouts, completed PPO updates and saved a final model.

⚠️ **Neither `ep_len_mean` nor the `🏁 Episode Finished` line count is a measurement here.** The
episode line is throttled by `--log-level periodic` (42 lines on S against 9 on W at the identical
10,240 steps, while `ep_len_mean` differs by only 4.6 turns), and 10k steps is an essentially
untrained policy on one env. The row is recorded so a future reader does not mistake the log-line
counts for a difference. The real ep-len read is §8.4b, on the arms themselves.

Throughput on CPU with compile off was ~10 fps, so 10,240 steps took ~16 min each — expected for
the full production surface on one env, not a signal.

---

## 8. THE REGISTERED READ — fixed here, before either arm starts

> Written before any number from either arm exists. Rule 1. Both branches and the bar are stated;
> §9 says what will and will not be claimed.

### 8.1 Strength — the primary endpoint

**Instrument:** `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29)
[§3.2 rule 1]. Read through `agents.training.snapshot_ladder.fit_ladder(run_dir, first_n=N,
steps=COMMON, write=False)` — `first_n` forces `write=False`, so nothing under `models/` is touched.

**Matched snapshot COUNT, not matched step** [§3.2 rule 3]. 🚨 Two corrections are mandatory and
both have bitten before:

1. **Refit on the COMMON step set.** If the two arms cross into self-play at different steps they
   rate different early nodes, and a naive `first_n=N` compares arm A's node *k* against arm B's
   node *k+1* — matched count, un-matched steps, with nothing in the tree warning about it
   (hazard 1 of the 2026-09-12 strength read). The self-play crossing is a **per-arm lottery**
   (standing rule 15, 2026-09-10 amendment): six of eight 10M arms crossed at ~4.13M, two at
   ~2.16M. **Read each arm's crossing from its TensorBoard (`main.ops.tb_read.selfplay_crossing_step`,
   first `*_pool` scalar step) before fitting anything.**
2. **The newest node is systematically inflated** [§3.2 rule 2]. Both arms finish at the same step,
   so the inflation is common-mode and cancels in a pairwise Δ to first order. **Registered
   cross-check: the second-newest node of the same fit**, exactly as the 2026-09-12 table did.

**How many nodes the read needs — computed, not assumed.** Refitting the existing 75M win-prob run
`ai_v12_02_winprob_critic` at increasing `first_n` gives the instrument's own resolution curve
(`se(Δ) = √(se_arm² + se_ctrl²)`, CI95 = 1.96·se(Δ)):

| nodes | newest node | se | se(Δ) | CI95 ± | frozen pairs |
|---|---|---|---|---|---|
| 4 | 42.0M | 15.70 | **22.20** | 43.5 | 6 |
| 6 | 46.0M | 13.60 | 19.23 | 37.7 | 15 |
| 8 | 50.0M | 12.50 | 17.68 | 34.6 | 28 |
| 10 | 54.0M | 11.60 | 16.40 | 32.2 | 45 |
| 12 | 58.0M | 10.70 | 15.13 | 29.7 | 66 |
| 16 | 66.0M | 9.30 | 13.15 | 25.8 | 120 |
| **20** | **74.0M** | **8.50** | **12.02** | **23.6** | **190** |

The `n=4` row reproduces the 2026-09-12 finding exactly (se(Δ) 21–24 ⇒ CI95 ±43–48, *"the ladder at
four nodes cannot resolve its own floor"*).

🚨 **REGISTERED: the read is at n = 20 nodes**, the full ladder a 75M run retains (`ai_v12_02` rates
36M → 74M at 2M spacing), refit on the two arms' COMMON step set. At n = 20, **se(Δ) = 12.0, CI95
±23.6**, so under the family's own detection rule (|Δ| past the floor AND Δ's own CI excluding the
floor) **the smallest claimable difference is |Δ| > 45.0 + 23.6 ≈ 69 Elo.** Anything smaller is
reported as NOT DETECTED, never as equivalence (see §9.2). If the two arms retain different node
counts, the read is at `min(n_S, n_W)` and the table above gives that row's resolution — **n ≥ 12
is the floor below which the read should not be reported at all**, since below it CI95 exceeds the
floor and the delta-CI clause can never fire.

**The floor.** 45.0 Elo — the max pairwise |Δ| over the three same-argv 10M controls
(`ctrl10M` / `_b` / `_c`: 45.0 / 35.3 / 9.7), reproduced from an independent refit on 2026-09-12.
🚨 It is the **max**, never the mean [rule 3]. 🚨 **It belongs to a different DEPTH** (10M, four
nodes) than the read (75M, twenty nodes) and is imported as the only floor in hand — §9.1.

### 8.2 The within-run late slope

Elo per M steps over each arm's last third, fit on its own ladder nodes, with the node-to-node
residual spread as its own uncertainty. **Registered because it is the one strength statistic that
does not need a cross-run floor.** Read on both arms and reported side by side; the "still learning
at 277M" property that separates v8's line from ours is a slope statement, and 75M is the first
length at which we can ask it of a current-architecture arm. Not an endpoint, and **no bar is set**:
whichever way it comes out, it is reported as a description.

### 8.3 Policy entropy H

`train/entropy_loss` (nats; the action space is unchanged since 2026-05-22), median-20, window
pinned before the number is read, per standing rule 15 (never across the self-play crossing).

**Prediction, registered:** both arms sit near v8's plateau (1.07–1.11), because `ent05` PASSED on
this architecture, obs, reward and ecology (H_end 1.0861 at 10M, slope +0.025 nats/M and still
rising) and 0.05 is typed on both arms here.

🚨 **If arm S's H differs from arm W's at matched steps, that is a FINDING, not a bar.** The
entropy coefficient is matched; entropy is downstream of the objective, so a difference localises
*how the reward composition shapes exploration* and is one of the few places the pair can say
something the ladder could not. No PASS/FAIL attaches to it. The replicate floor in hand is **0.074
nats** (three seeds, `entropy_forensics_v8_vs_ours_2026-09-12`), measured at 10M — a difference
below it is not read.

### 8.4 The critic rows

**Decision row: `cond.opp_class_auc.t4_10`** — opponent-CLASS (pool vs bot) AUC of V over turns
4–10, the row on which the opponent is actually observable (turn 1 reads 0.502/0.508, i.e. nothing;
turns 4–10 read 0.82 from `value_pooled` and 0.700–0.723 from the online head). Instrument:
`critic_read` v6, two offline 800-game full-capture draws per checkpoint, cluster-bootstrapped over
TEAMS [rule 10], reweighted by `eval_manifest.json`'s per-opponent capture rates [rule 17 — a tree
without a manifest is SELECTION UNKNOWN and is refused].

🚨 **ARM S MUST BE READ BOTH WAYS, AND BOTH MUST BE LABELLED.** On every win-prob arm read to date,
`V` and the win-prob head are the *same tensor*, and `conditioning_meters` builds its `V` column
from the npz's **`win_probs`**, reporting `max_abs_values_minus_winprobs` as a QC scalar that has
always been ~0. **On arm S they are different readouts**, and the QC scalar will be LARGE by
construction — *that is not a defect*. So:

| registered row | column | what it is on arm S | comparable to |
|---|---|---|---|
| **S-V** | `values` | **the actual critic** — distributional E[Z] in shaped-return units | arm W's V. **This is the critic-vs-critic contrast and the treatment's own row.** |
| **S-WP** | `win_probs` | **the auxiliary win-prob head** at `--win-prob-coef 0.05` — a diagnostic, not the value function | arm W's V, as a *head-vs-head* contrast: same readout family, but a critic on one side and a diagnostic on the other |

Reporting either one unlabelled is the failure this row is registered against — the default v6 path
returns **S-WP**, i.e. the auxiliary head, which is the one a reader would most easily mistake for
arm S's critic. **Engineering item before the read, not before the launch:** `critic_read`/
`conditioning_meters` needs a column selector so S-V can be computed without editing the tool; the
`values` column is already carried in the npz.

**Supporting critic rows, same draws, both arms:** `gate.resolution.all` and `gate.resolution.bot`
(outcome resolution — the row on which `vf15_b` and `strata_b` both showed the dissociation:
*V resolves outcomes as well as the controls while discriminating opponent class worse*),
`cond.opp_class_auc.t1_3` (the weaker sibling, ~⅓ the effect size), and the frame QC set
(own-team → class leak, frozen-forward max |V_fwd − V_rec|, battles, teams, capture 1.0 per
opponent, snapshot md5). **A frame whose own-team → class leak sits outside the established
0.492–0.498 band is not read.**

**Floors:** the v6 two-pair floors in hand are **0.0220** (first draw) and **0.0245**
(`hp800b_floor_v6.json`, second draw — this supersedes the single-pair 0.031 the
`repr_class_decode_*` reads used). They are **10M, four-node-era, control-vs-control** floors and
are imported for want of anything at 75M — §9.1.

### 8.4b Stall rate and mean episode length — PRIMARY on arm W, by the banner's own instruction

Arm W's `[CRITIC]` banner does not merely describe the mode; it names an endpoint:

> ⚠️ *A `[0,1]` critic cannot express 'a timeout is worse than a loss' — stall rate and mean episode
> length are PRIMARY endpoints on this arm (`--arm-no-progress-tax` is the contingency, currently
> OFF).*

**Registered accordingly, on both arms:** `rollout/ep_len_mean` and the timeout/stall fraction,
read on each arm's own span and never across the self-play crossing (rule 15).

* **Rule 12 binds: a timeout is never a semantic outcome, and neither is a draw.** They go in
  separate buckets from wins and losses; **a run above 25 % timeouts is INCONCLUSIVE**, not
  reported.
* This is also a G7 input. The shaped family's only G7 breach to date
  (`ctrl10M_shaped_dense`, worst 1.522) came with `rollout/ep_len_mean` climbing 22.3 → 35.2,
  peak 41.5, *with* `no_progress_tax` live — so arm S, which carries the same single BIAS term,
  is the arm to watch on this row, and arm W is the arm the mode's own warning is about. **Both
  are watched; neither has a bar** beyond rule 12's 25 % inconclusive threshold and the standing
  G7 operational condition of §9.3.

### 8.5 The untaught meter

At run end, both arms, `python -m main.untaught_meter` against the registry-named fixed opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip`) and the
registry-named `untaught_meter_config`, so the level is comparable to every banked one.

⚠️ Neither arm is architecture-identical to the meter's config, so **read `--config auto` as well
and report both**; a level that only exists under one config resolution is not a level. Reported as
a descriptor beside strength, not as an endpoint: the meter's own axes and floors are established
at ~1M-step fold depths, not at 75M fresh.

### 8.6 The cross-run comparisons, with every confound named

| comparison | what it can say | confounds, named |
|---|---|---|
| **arm S vs arm W** (the pair) | the treatment. Matched pin, dose, length, seed slot, eval regime, architecture, ecology | **none on the resolved config** (§4: 16 differing keys, 15 treatment-or-forced). The confound is n=1 per arm — §9 |
| **arm W vs `ai_v12_02_winprob_critic`** (the 0.02 leg, 75M win-prob) | the entropy-coefficient axis at full length, for free | pin (`f971caf2` vs HEAD), `--ent-coef` (0.02 vs 0.05), and **the eval regime boundary**: `ai_v12_02` records no `eval_sentinel_greedy` at all (pre-v112 schema) and spans 2026-09-07. 🚨 **`win_rate_vs_pool` and `eval/elo` are NOT comparable across it** — the asymmetric regime was worth **+8.9 pp [+7.0, +10.7]** to the trainee. **`ladder.json` and every bot edge ARE unaffected** (the ladder always played greedy-vs-greedy with symmetric builders and since `3e6875a5` drops eval-sentinel edges from its fit entirely). So this comparison is **ladder-only** |
| **either arm vs `ai_v8_03_zarch_control_0718`** (the v8 line) | the cross-era question | 🚨 the heaviest confound set in the programme: different architecture, obs dim, action space era, reward composition, clip range (0.10 vs 0.15), ecology, and **step counts that are not commensurable across architectures with different sample efficiency**. `ai_v8_03`'s ladder rates **10 nodes, 150.9M → 248.0M** — a different depth AND a different count from our 20-at-75M. Both bot-anchored, both far above the anchors, so **§3.2 rule 4 applies: a direct match is required before any gap is quoted as a difference.** Without one this is reported as a *shape* comparison (slope, entropy trajectory) and never as an Elo difference |
| **arm S vs `ai_v9_29_rev1_0823`** (the era run itself) | "did the era's composition survive the transplant?" | pin, dose (`--grad-accum-steps` 8 vs 32, lr_median 3×), `--ent-coef` (0.02 vs 0.05), eval regime (stochastic vs greedy), length (25M vs 75M), **12 ladder nodes vs 20**. A sanity check on the transplant, not a measurement |

---

## 9. Bars — and what will NOT be claimed

### 9.1 🚨 There is no run-to-run floor at 75M, and there will not be one

Every floor quoted in §8 — 45.0 Elo on strength, 0.0220/0.0245 on the decision row, 0.074 nats on
entropy — was measured on **10M arms**. Standing rule 3: *a floor belongs to the REGIME and to the
DEPTH*, and *"a floor from few draws at one depth is close to uninformative — three separate
readings in this programme were retracted for it."*

**One seed per arm at 75M. Two arms. There is no replicate at this depth and none is affordable**
(~35 GPU-h each). The honest position:

* The **only floor proxy available within the pair** is each arm's **node-to-node spread on its own
  ladder** — the adjacent-|Δ| distribution over its 20 nodes. That bounds *within-run* wobble; it
  does **not** bound run-to-run variance, and the two are known to differ by a large factor
  (`famine_comparator`'s entry says so explicitly: the adjacent-node spread of 172/186 was rejected
  as a floor because it is dominated by early LEARNING, not noise).
* Rule 19: identify which variance component dominates, then replicate at THAT level. Strength and
  the conditioning decision row are **run-level** rows. The remedy is a full arm each, which the
  pair does not have.
* Rule 22 is therefore binding: **a single-arm detection is a CANDIDATE until its own seed replicate
  agrees.** On strength, the two same-pin seed-replicate pairs in hand differ by **83 and 58 Elo**
  (CIs clear of zero) against a control-pair difference of **10** — *a lever arm's variance is not
  the control's*, and the understatement runs a factor of 6–8.

### 9.2 What will be claimed, and what will not

| if the read shows… | claim |
|---|---|
| \|ΔElo\| > ~69 at n=20, same sign on the second-newest-node cross-check | **CANDIDATE — a direction at full length, n=1 per arm.** Named as a candidate, never as a family verdict. The named next increment is a seed replicate of whichever arm leads |
| \|ΔElo\| ≤ ~69 | **NOT DETECTED.** 🚨 **Never "equivalent".** Rule 6: equivalence needs the DELTA's own CI *inside* the bar; here CI95 ±23.6 sits inside a 45.0 floor imported from another depth, so the equivalence clause is unavailable and a bar against a point estimate would be vacuous |
| a decision-row Δ clearing ±0.022/0.0245 on both draws, both readings labelled | **CANDIDATE at run level.** Rule 22: the campaign's only confirmed detection to date (`vf15_b`) needed two seeds. This pair has one |
| arm S's and arm W's H differ at matched steps | **FINDING** (§8.3), reported with its 0.074 floor, no verdict attached |
| anything at all vs the v8 line | **SHAPE ONLY** unless a direct match is played (§3.2 rule 4) |

🚨 **Not claimable from this pair under any outcome:** that either critic is better *in general*;
that the result transfers to a fold, a distillation, or the ladder; that a 75M difference implies a
277M one. And **never** the circular "⅔ of losses are team-draw, so they are unwinnable" — those
losses are HEADROOM.

### 9.3 The kill / no-kill position

Neither arm has a kill rule. Both are registered to **run to 75,000,000 or to a crash budget
exhaustion**, whichever comes first. The reason is on the record twice this week: the two void
shaped controls were let run and *the dynamics were the yield*, and a two-eval-cycle projection of a
crossing was made, was wrong in the optimistic direction, and should have been declined rather than
qualified. **No cross-projection of arm S's crossing from its first two eval cycles will be made.**
The one condition that ends an arm early is an operational one — a G7 breach sustained across two
consecutive cycles with the anti-stall tax active — and it ends it as an **OPS** event, reported as
such, not as a verdict on the critic.

---

## 9b. STATUS 2026-09-14 — arm S complete, arm W live

Arm S (`1750680d`) crossed at 4,128,768, G7 under bar for 37 cycles, 20 ladder nodes, `final_model.zip` at 75,005,952. Its standalone reads are in `measurements/flywheel_armS_reads_2026-09-14/` (`04e13eae`): Elo 2036.6 (se 8.9) at 20 nodes; late slope +1.14 ± 0.44 Elo/M; H flat at 1.03; decision row 0.768 (critic) / 0.767 (auxiliary head); untaught 54.5 pp; anchors 0.630 (SmallRL greedy) / 0.450 (Foul Play @1.25M visits). **Two amendments to the read: (1) the 0.02 leg `ai_v12_02` is REFIT under the current recipe (1984.2), never quoted from its committed file (2057.3, stale recipe — +73 Elo); (2) the registered critic control for arm S is refused by `critic_read` (regime + game-count mismatch) — critic rows are LEVELS, read at 74,000,016 (no eval cycle at the final step); arm W's are read at its last evaluated step and the distance from the end checked. `--v-column {win_probs,values}` exists for the both-ways read; calibration and `gate.*` rows are defined only on the probability column.** Arm W launched 13:07 PT 2026-09-14; the pair read is S vs W on these rows.

## 9c. VERDICT 2026-09-16 — NOT DETECTED on every endpoint; three pre-designated findings lean shaped

`measurements/flywheel_pair_read_2026-09-15/` (`986ed5f5`). Strength Δ(S − W) +17.5 [−6.9, +41.9] at 20 refit nodes (claimable ≈ 69) and +6.7 [−20.2, +33.6] on the 16-shared-step refit — NOT DETECTED. Anchors all NOT DETECTED. Critic guard +0.025 inside the floor. Findings (§8.3 / §8.5, not endpoints): arm S entropy FLAT 71M vs arm W decaying −0.097 nats; untaught +8.31 pp to S (8/8 teams); arm W critic ECE 0.069 vs 0.018 at equal resolution; realized dose 1.44× in S's favour (H-F). One seed per arm; rule 22 binds; the named next increment is a seed replicate of arm W if a family claim on the findings is ever needed. The era proceeds on the win-prob critic.

## 10. Sequencing and ETA

**The GPU is single.** Order: `vf025` (live) → **arm S** → **arm W**.

| | | |
|---|---|---|
| `ai_v12_29_ladder_vf025` (live, 10M) | 4,800,000 steps at 20:38 PT, ~1.95 M steps/h | banks **≈ 23:30 PT 2026-09-12** |
| **arm S** `ai_v13_01_flywheel_shaped` | 75M | **≈ 31–38 GPU-h** ⇒ banks **≈ 2026-09-14, 07:00–14:00 PT** |
| **arm W** `ai_v13_02_flywheel_winprob` | 75M | **≈ 31–38 GPU-h** ⇒ banks **≈ 2026-09-15 15:00 – 2026-09-16 04:00 PT** |

**Why arm S first:** it is the arm carrying the transplant, so it is the one whose first two minutes
can fail in a way the smoke cannot see (§7.4), and the one whose first eval cycle answers the
question the two void controls raised — *does the era's composition still cross?* Arm W is a
lengthened re-run of an argv that has completed fifteen times; its risk is the lowest in the pair.

**Where the rate comes from** (measured from checkpoint mtimes, this box):

| run | span | rate |
|---|---|---|
| `ai_v12_02_winprob_critic` (75M, the only 75M precedent) | 2.4M → 73.1M in 35.44 h | **1.995 M/h** ⇒ 75M ≈ 37.6 h |
| `ai_v12_11_ladder_ctrl10M` | 2.4M → 9.97M in 3.51 h | 2.158 M/h ⇒ 34.8 h |
| `ai_v12_28_ladder_ent05` | 2.4M → 7.2M in 2.07 h | 2.314 M/h ⇒ 32.4 h |
| `ai_v12_25_ladder_vf15_b` | 2.4M → 7.2M in 1.98 h | 2.425 M/h ⇒ 30.9 h |

The 10M arms give 31–35 h; the only 75M precedent realized 37.6 h, because **eval and ladder cost
grow with the snapshot pool** and a 10M arm never pays the late-run share. **Point estimate ~35 h
per arm; ~70 h for the pair.**

⚠️ **Arm S's rate is the less certain of the two.** It carries PopArt, a 51-atom distributional
head and seven live PBRS potentials that arm W does not. The two shaped 10M controls ran *faster*
(3.11 / 3.35 M/h) — but only because **they never crossed into self-play**, and bot opponents are
far cheaper than running a snapshot policy. That number must not be used as arm S's rate: if arm S
crosses as the era's run did (2,000,016, first cycle), it pays the self-play cost from 2M on.

---

## 11. Hazards

**H1 — "it launches" and "it is the experiment" are independent checks.** Both were asked. The ARCH
SURFACE is clean on both argvs (§7.1), which is the check the 2026-09-06 incident existed to add
(31 silently-reverted architecture keys, 24.4M steps, ~7 GPU-h). 🚨 **The command blocks in §3.1 and
§3.2 are a DESIGN DOCUMENT'S command blocks — the exact artifact class that caused that incident.
Launch from the argv FILES named in §3, not by copying out of this markdown.**

**H2 — a bare run directory means the run's LAST snapshot.** Neither arm names a teacher, a parent,
an exploiter source or a `--win-prob-pbrs-source`, so the rule does not bite here. It will bite the
moment either arm is used as a source; name the `.zip` or `@step` then.

**H3 — an argv is not a config, but these two are.** Both are FRESH (`role: FRESH`, no `--model`),
so nothing is inherited and the argv *is* the config. On the launcher's 3-hourly **restarts**,
`--lr`, `--batch-size`, `--n-steps` and `--gamma` become INERT (SB3 restores the checkpoint's own
values). That is correct and matched across the pair. `--ent-coef` is **live** on a resume
(`model_build.py:543`), unlike `--lr` — which is why 0.05 holds for the whole run.

**H4 — `--vf-coef 0.5` does not mean the same thing on the two arms.** On W it multiplies a **BCE**
over a win indicator; on S it multiplies a PopArt-normalised **MSE** over a shaped return. The
combination check says so in as many words: *"`--vf-coef` now multiplies a BCE rather than an MSE
over a shaped return, so 0.5 does not transfer between the two loss families; re-tune it on this
arm."* **It is deliberately NOT re-tuned**, because re-tuning one arm and not the other would add a
free parameter to the treatment. It is listed here so the pair's Δ is never read as if the
value-loss scale were matched — it is matched in *number*, not in *units*. `--vf-coef` is also fixed
for a run's lifetime (a differing value on resume is FATAL).

**H5 — the conditioning instrument reads the wrong column on arm S by default.** §8.4. The v6 path's
`V` is the npz's `win_probs`; on arm S that is the *auxiliary* head, not the critic.
`max_abs_values_minus_winprobs` will be large and is **not** a defect.

**H6 — the self-play crossing is a per-arm lottery** and the two arms may cross at different steps.
Read each from TensorBoard; never assume a common step; any statistic struck in the 2M–4M band is
crossing-conditional (rule 15).

**H7 — a fresh worktree pays for a cargo build on its first rust test.** The launcher creates an
isolated worktree at the pinned commit; the rust `sim_bridge` binary is published to children from
the main checkout's `target/release` (confirmed in the smoke banner). Do not build into main's
`target/` from a worktree; set `POKESIM_SIM_BRIDGE_BIN` if in doubt.

**H8 — `--debug` does not fence the GPU.** §7.4. An explicit `--device cuda` in the argv beats
`--debug`'s CPU default, and a smoke run that way competes with the live arm for VRAM. It cost one
smoke attempt here (the live arm was unharmed and verified so). Append `--device cpu` and
`CUDA_VISIBLE_DEVICES=""` to any smoke of a production argv.

**H9 — a clean completion does not guarantee `final_model.zip`.** `ctrl10M_shaped`'s
restart-boundary child found training already complete and exited without writing the canonical
name; `latest.txt` read `final_model_interrupted.zip`. **Every offline trigger on these two arms
tests `latest.txt`, never a fixed filename.**

**H10 — instruments must not read an artifact another process is still writing** (rule 16). Both
arms will be read while the *other* is live. A ladder mid-update returns a correctly-formatted
verdict about the previous node; the live-run instruments are `main.ops.*` / `scripts/ops/` and
refuse when a precondition of the live read is unmet. Offline meters go on banked runs only.

**H11 — the era transplant is not dose-matched to the era.** §5. Arm S is the era's *reward
composition* on the ladder family's *dose*. Any arm-S-vs-era number carries that confound.

---

## 12. Ledger paragraph — ready to append (do NOT edit `ledger.md` from this document)

> ### 2026-09-12 · REGISTRATION · THE FLYWHEEL-ERA PAIR — two 75M arms, `--critic shaped` (the ERA's configuration, transplanted) vs `--critic winprob`, matched on pin / dose / length / seed / eval regime / architecture; the resolved-config diff is **16 keys of 283, fifteen of them the treatment or forced by it and one the run name — ZERO confounds**; and the strength read is computed to need **20 ladder nodes** (se(Δ) 12.0, CI95 ±23.6 ⇒ smallest claimable |Δ| ≈ 69 Elo)
>
> `designs/research_state/flywheel_era_pair_2026-09-12.md`, written before either arm started.
> **Owner direction 2026-09-12 (*PIVOT*):** search wound down, PPO effectiveness is the object,
> `--value-true-team` stays OFF, a full-length pair approved. **The arms.**
> `ai_v13_01_flywheel_shaped` (arm S — the era's shaped configuration) and
> `ai_v13_02_flywheel_winprob` (arm W — the win-prob critic), both `--pin-commit 6eb9c776`
> (main HEAD; all four 2026-09-12 landings are on it), `--steps 75000000`, `--seed 1001`,
> `--ent-coef 0.05`, `--clip-range 0.15`, `--clip-range-vf none`, `--eval-sentinel-greedy` and
> `--no-value-true-team` typed on both, `--restart-interval-hours 3`, dose **4.578e-8 by
> declaration** (`3e-4 × 10 / (2048 × 32)`, the four inputs identical tokens in both argvs).
> **Arm S is NOT `ctrl10M` with the mode flipped** — that configuration was run twice on 2026-09-12
> and both arms are VOID (never crossed; `ctrl10M_shaped_dense` also breached G7 at 1.522) while the
> era's own shaped fresh run crossed at 2,000,016. Arm S therefore carries the registered baseline
> `v9_long_baseline`'s (`ai_v9_29_rev1_0823`, commit `d78aa810`, config v101) reward and critic
> block, resolved from its `model_config.json` and **verified key-by-key: all 17 reward/critic keys
> the v101 schema records match EXACTLY** (PopArt ON, `value_dist shaping/51/±12/coef 1.0`,
> `value_from_dist`, `value_tail_weight 0.3`, `win_prob_mode shaping` at coef 0.05,
> `draw_penalty −35.0`, the whole BIAS class), with the five fields v101 did not record
> (`critic`, `hand_shaping`, `terminal_indicator`, `victory_value`, `gamma`) typed explicitly at
> their era values rather than left to defaults that have since moved. **Flag drift, tested by
> EXECUTION:** the era's 229-token recorded argv replays through the current parser with
> **0 unrecognized** and a clean ARCH surface; not one reward-family flag was deleted, renamed or
> changed arity; the single refusal is `--distill-team-bias 0.4` without a teacher (a new guard on a
> flag the era ran as a silent no-op, and not transplanted). **Validation:** `checkargs` exit 0 on
> both (141 / 132 flags, 0 unrecognized, 0 refused, ARCH SURFACE clean against
> `production_config@360f8378dd90`); `--dry-run` `role: FRESH` on both; `--debug --steps 10000`
> smokes of both from the main checkout into tmp, arm S reaching `Training complete` with the banner
> `[CRITIC] shaped — V(s) = the distributional E[Z] in raw shaped-return units (PopArt ON),
> gamma=0.9999; the win-prob head is an auxiliary at --win-prob-coef 0.05` and composition
> `1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)` — **both arms reached `Training complete` at
> 10,240 steps with no FATAL and no traceback**, arm W's composition reading
> `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)`, the treatment stated by the code
> itself. **REGISTERED READ, fixed before launch:**
> strength at matched snapshot COUNT on `snapshot_ladder/ladder.json` refit over the COMMON step
> set at **n = 20** — derived by refitting `ai_v12_02_winprob_critic`'s ladder at increasing
> `first_n`, which reproduces the four-node se(Δ) 22.2 / CI95 ±43.5 exactly and falls to
> **se(Δ) 12.0 / CI95 ±23.6 at twenty**, so the smallest claimable |Δ| is 45.0 + 23.6 ≈ **69 Elo**
> and n ≥ 12 is the floor below which the read is not reported; the second-newest node as the
> inflation cross-check; the within-run late slope (no bar); policy entropy H against v8's 1.07–1.11
> plateau and the 0.074 replicate floor, **a difference between the arms being a FINDING, not a
> bar**; `cond.opp_class_auc.t4_10` on two offline 800-game draws, 🚨 **read BOTH ways on arm S and
> both labelled** — `values` (the actual shaped critic) and `win_probs` (the auxiliary head at 0.05)
> are different tensors there, the v6 tool defaults to the latter, and
> `max_abs_values_minus_winprobs` will be large BY CONSTRUCTION and is not a defect; **stall rate and
> `rollout/ep_len_mean` as PRIMARY endpoints**, promoted by arm W's own `[CRITIC]` banner (a `[0,1]`
> critic cannot express "a timeout is worse than a loss"; `--arm-no-progress-tax` is the contingency
> and stays OFF), read under rule 12 — a timeout is never a semantic outcome and an arm above 25 %
> is INCONCLUSIVE; the untaught meter at run end under both the registry config and `--config auto`.
> **Bars, stated honestly:**
> there is **no run-to-run floor at 75M and none is affordable** — one seed per arm — so every floor
> quoted (45.0 Elo, 0.0220/0.0245, 0.074 nats) is imported from 10M four-node depth, the only
> within-pair proxy is each arm's node-to-node spread (which bounds within-run wobble, not run-to-run
> variance), and **rule 22 binds: any Δ is a CANDIDATE, never a family verdict** — the two same-pin
> seed pairs in hand differ by 83 and 58 Elo against a control pair's 10. A null is reported
> **NOT DETECTED, never "equivalent"** (rule 6: the delta's own CI must sit inside the bar, and
> ±23.6 inside an imported 45.0 makes that clause unavailable). **Comparisons and their confounds:**
> vs `ai_v12_02_winprob_critic` (the 0.02 leg) is **ladder-only** — it spans the 2026-09-07
> opponent-regime boundary, so `win_rate_vs_pool` / `eval/elo` are not comparable across it (+8.9 pp
> [+7.0, +10.7] to the trainee under the old asymmetry) while `ladder.json` and every bot edge are
> unaffected; vs the v8 line (`ai_v8_03_zarch_control_0718`, 10 nodes over 150.9M–248.0M) is
> **SHAPE ONLY** absent a direct match (rule 4). **Sequencing:** single GPU, `vf025` → arm S → arm W;
> ~31–38 GPU-h each (the only 75M precedent realized 1.995 M steps/h ⇒ 37.6 h; the 10M arms 2.16–2.43
> M/h ⇒ 31–35 h; arm S's rate is the less certain, and the two void shaped controls' 3.1–3.3 M/h must
> NOT be used for it because they never crossed into self-play). Arm S first — it carries the
> transplant, and its first eval cycle answers whether the era's composition still crosses.
> **No cross-projection of that crossing from two eval cycles will be made.** Hazards recorded:
> a design-doc command block is not a launch command (launch from the argv files);
> `--vf-coef 0.5` multiplies a BCE on W and a PopArt-normalised MSE on S and is deliberately NOT
> re-tuned; `--debug` does **not** fence the GPU when the argv names `--device cuda` (one smoke
> attempt OOMed beside the live `vf025` arm — the live arm was verified unharmed, still writing TB
> events 9 s later — and every subsequent smoke ran `--device cpu` with `CUDA_VISIBLE_DEVICES=""`);
> the smoke bypasses the forkserver preload and the compile layer, so **the first two minutes of each
> real launch remain the only test of it**; triggers test `latest.txt`, never a fixed filename.
> Tag: **REGISTRATION · flywheel-era pair · era composition VERIFIED key-by-key · 16/283 keys differ,
> zero confounds · read at 20 nodes, ~69 Elo resolution · n=1 per arm ⇒ CANDIDATE at best**.

## 9d. STATUS 2026-09-20 — the continuation control REFUTES the frozen-parent comparator on this side; fold-2 on the era-1 recipe is WITHDRAWN

`wcont_control_read_2026-09-20/` (arm W +12M, no teachers, frozen dose 4.272e-9): +15.50 pp [+12.00, +18.88] on the untaught 8, 8/8 teams, 4.2× the floor; both per-slice cells cleared; level with teacher t1 on t1's own team unseen; 5 promotions vs the fold path's 0. The era-1 fold sits −9.75 pp / −0.165 (Big-5) BELOW it, outside the floor, with the gap present at +3M. Three levers moved together (loss, team bias 0.4, two-specialist pool) — the carrier is not identified. **Next GPU arm: the three-lever split** (fold-1's argv, `--distill-coef 0.0`, nothing else changed, +12M to 87,097,344); the second control seed queued behind it. Every future fold on this side is read against a CONTINUATION at matched steps and dose, never against the frozen parent. [ledger 2026-09-20 · *THE CONTINUATION CONTROL*]
