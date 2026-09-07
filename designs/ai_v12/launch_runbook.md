# ai_v12 — CLEAN-WORLD LAUNCH RUNBOOK

> **[STATE]** The four commands below are **verified launchable**: every one passes
> `python -m main.checkargs --argv "…"` at exit 0, and every one has been run as a real
> `--debug --steps 8000` CPU smoke whose reward banner, round-trip test and `train/pbrs_*`
> scalars are recorded in §5. This document is the thing the training session executes from.
>
> It is a RUNBOOK, not a registration. The arms, the endpoints and the escalation rules are
> already registered — ledger `e22bd08` (the three-arm ladder), `627ab58` (no launch bias;
> earned escalation), `cfbc9bf` (draw = −1; the coefficient spelling), `2d38a4a` (PopArt
> retirement), `db9bb5c` (the V_shaped-constancy prediction), `4d22ae4` (the pure-sparse control
> and the 5M pre-test), `132d198` (wave A's landed flag surface). Nothing here adds an arm.

---

## 1. What is being launched, and what each pairwise difference means

Three generation-scale arms, identical in every respect except **where the potential comes
from**, all at terminal `{win +1, loss −1, draw −1}`:

| arm | potential φ | what it tests |
|---|---|---|
| **SPARSE** | none | the famine claim — can this recipe learn from outcome-only signal at all? |
| **SELF-φ** | the run's OWN live win-prob head | self-bootstrapping shaping; exact invariance per rollout, approximate across |
| **FROZEN-φ** | a mature prior-generation head, frozen | ancestral scaffolding; **exact** invariance |

- `SELF − SPARSE` = the value of self-shaping (and the direct test of the demoted bootstrap hypothesis)
- `FROZEN − SELF` = the value of maturity + exact-vs-approximate invariance
- `FROZEN − SPARSE` = the total worth of outcome-grounded shaping

> ### ⚠️ THE LADDER ABOVE IS THE **SHAPED**-CRITIC ladder — under `--critic winprob` it has TWO rungs
>
> `gen3_winprob_critic_mode_v1` and `gen3_frozen_phi_actor_only_v1` (2026-09-06) between them
> re-shape this table for the win-prob critic, and the middle rung stops existing there:
>
> | rung | under `--critic shaped` (this document) | under `--critic winprob` |
> |---|---|---|
> | SPARSE | `$CLEAN` alone | the live `ai_v12_01_winprob_critic` arm |
> | SELF-φ | `--win-prob-pbrs-coef $COEF` | **REFUSED** — with `V ≡ φ`, `γφ(s′) − φ(s)` IS the TD residual GAE already turns into the advantage, so route 1 would add the advantage to the reward and then take the advantage of that (`design_winprob_only_critic.md` §3.7) |
> | FROZEN-φ | `--win-prob-pbrs-coef $COEF --win-prob-pbrs-source $PHI_SRC` | **`--win-prob-pbrs-frozen <run\|zip>`** — one flag, NO coefficient (φ is already in the value currency, so it is exactly 1.0 and is printed at startup), and the shaping is **ACTOR-ONLY**: it reaches the advantages and nothing else |
>
> **So `FROZEN − SELF` is not measurable on that critic** and the arm reads `FROZEN − SPARSE`
> alone — which is worth saying out loud, because it is the contrast this document's §1 calls "the
> total worth of outcome-grounded shaping" and it no longer decomposes.
>
> **Everything in §4 below still applies to the winprob FROZEN arm, with the metric names moved**:
> `train/pbrs_phi_mean` → `pbrs/frozen_phi_mean`, `train/pbrs_episode_dose` →
> `pbrs/frozen_phi_episode_dose`, and the `🧊` banner is `[FrozenPhi]` rather than `[WinProbPBRS]`.
> §4.2's constancy check is unchanged and is still the cheapest live proof that the frozen source is
> the thing being read. The `signal/adv_shaped_minus_unshaped_mean` series is NEW there and has no
> counterpart here: it is the telescoping term the policy gradient gained, and at λ = 1 it is exactly
> `−1.0 ×` the φ mean. The winprob arm's own command is `design_winprob_only_critic.md` §5.4.

The incumbent comparison is **free** — existing rev-1/rev-3-class 25M checkpoints via h2h +
anchored ELO. It is an imperfect control (era-config differences) and is a reference, not an arm.

Ahead of all three, a **paired 5M pre-test** (SPARSE vs the incumbent SHAPED recipe, same seed,
same team draws) sizes the full runs in hours of GPU rather than in generations: crater / crawl /
keep-pace.

---

## 2. THE ARGVS

Every arm shares one block. Paste this block once; it is exactly what `checkargs` validated and
exactly what the smokes ran. **Keeping the shared part shared is not tidiness — it is the
experiment**: the arms must differ in one thing, and a hand-retyped 110-flag command cannot be
audited for that. `src/main/launch_runbook_test.py` parses these blocks OUT OF THIS FILE and runs
them through the live parser, so a flag deleted anywhere in the tree fails a test with this
document named, rather than a launch two days later.

**Provenance of `$ARCH` / `$TRAIN`: the live gen-15 run's own recorded command**
(`ai_v9_72_R3SELF_0828`, `config_version` 107 = current HEAD), with exactly these removed —
the run-specific (`--run-name`, `--run-dir`, `--steps`, `--model`), the fold-specific
(`--distill-teacher` and its `--distill-*` companions — these arms are FRESH, with no teachers),
the already-default no-ops (`--opd-coef 0.0`, the search-teacher zeros, the entropy boosts at
1.0), the reward-side flags made inert by `--no-hand-shaping` (`--mat-alive-weight`,
`--no-progress-penalty`, `--bias-additivity`, `--self-ko-hp-penalty`, `--switch-bias-weight` —
all retained in `$SHAPED`, where they are the incumbent), and **`--use-popart`** (§6.2).

```bash
export PYTHONPATH=$PYTHONPATH:src
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

# ── the ARCH surface — identical across every arm (the gen-15 production architecture) ─────────
ARCH="--unified-moves both --unified-damage both --damage-op --damage-outgoing --damage-matrices both --damage-topk 6 --damage-candidate-k 0 \
--move-latent --move-prior-fusion --move-belief-mode both --move-belief-coef 0.05 --move-belief-latent-coef 0.05 --move-candidate-floor 0.02 \
--species-prior-fusion --t0-species-prior \
--opp-belief-cls-k 6 --opp-belief-aux-coef 0.05 --opp-belief-moves-weight 1.0 \
--spread-belief --spread-belief-nature --spread-belief-coef 0.05 \
--hp-belief-mode composed --hp-type-belief-coef 0.05 --item-belief \
--belief-grad-mode shaping \
--entity-topk-seats 6 --entity-tail-seats --consequence-topk 6 \
--edge-bias-families d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r \
--history-events --op-drop-renders --op-believed-lean \
--value-entity-pool --value-entity-pool-full --value-threat-inject --value-from-dist \
--opp-intent-coef 0.05 --intent-move-cell --intent-threshold --intent-conditional --intent-label-bot-weight 0.25 --beta-setvalued-coef 0.05 \
--pair-outcome-cell --pair-outcome-switch --switch-branch-cell --conditional-threat-cell \
--cf-records --cf-records-keep 4096 --cf-twin-heads --cf-twin-coef 0.1 --cf-shadow-critic --cf-shadow-coef 0.1 --cf-evidential --cf-evidential-coef 0.05 \
--capacity-telemetry --rank-tripwire warn"

# ── the TRAINING surface — identical across every arm. NOTE: no --use-popart (see §6.2). ───────
TRAIN="--device cuda --log-level periodic \
--n-envs 48 --n-steps 2048 --batch-size 2048 --grad-accum-steps 2 --n-epochs 10 \
--lr 0.0003 --min-lr 1e-05 --ent-coef 0.02 --clip-range 0.15 --clip-range-vf none \
--vf-coef 0.5 --weight-decay 1e-05 --value-tail-weight 0.3 \
--use-bridge rust --compile-opponents --compile-opponents-strict --compile-trainer \
--self-play --self-play-temp 1.0 --self-play-use-cpu --n-sentinels 5 \
--stable-opponent-mastered-wr 0.8 --stable-opponent-selfplay-share 0.2 --stable-opponent-temp 1.0 \
--team-block-episodes 64 --team-pfsp off \
--eval-battles 100 --eval-workers 5 --eval-shard-games 25 --eval-device cpu --eval-concurrency-per-worker 1 \
--snapshot-ladder-games 100 --checkpoint-every-steps 150000 \
--keep-crashes 10 --keep-eval-snapshots 10 --keep-eval-trace-steps 20 --keep-stalls 50 \
--warmstart-battles 200 --warmstart-bc-steps 4000 --seed 42"

# ── the CLEAN-WORLD reward, and the critic support RE-SIZED to it (§6.3 — do not skip) ─────────
CLEAN="--no-hand-shaping --victory-value 1.0 --draw-penalty -1.0 \
--win-prob-mode read_only --win-prob-coef 0.05 \
--value-dist-mode shaping --value-dist-bins 51 --value-dist-vmin -2.0 --value-dist-vmax 2.0 --value-dist-coef 1.0"

# ── the INCUMBENT reward — the 5M pre-test's control arm ONLY ──────────────────────────────────
SHAPED="--all-shaping-pbrs --mat-alive-weight 1.25 --no-progress-penalty 0.15 --bias-additivity 1.0 \
--draw-penalty -35.0 --self-ko-hp-penalty 0.0 --switch-bias-weight 0.0 \
--win-prob-mode shaping --win-prob-coef 0.05 \
--value-dist-mode shaping --value-dist-bins 51 --value-dist-vmin -12.0 --value-dist-vmax 12.0 --value-dist-coef 1.0 \
--use-popart"

# the frozen potential. ABSOLUTE, deliberately — the path resolves against the CHILD's cwd, §6.5
PHI_SRC="/home/goodlad/dev/gen3ai/models/ai_v9_70_R3ACTION_0828/final_model.zip"
COEF="0.3"                                                # the ladder's top rung — §6.1
```

### 2.1 ARM 1 — SPARSE (the famine test)

```bash
$PY -m main.launcher --restart-interval-hours 3 \
  --run-name cw1_sparse --steps 25000000 $TRAIN $ARCH $CLEAN
```

### 2.2 ARM 2 — SELF-φ

```bash
$PY -m main.launcher --restart-interval-hours 3 \
  --run-name cw2_self_phi --steps 25000000 $TRAIN $ARCH $CLEAN \
  --win-prob-pbrs-coef $COEF
```

### 2.3 ARM 3 — FROZEN-φ

```bash
$PY -m main.launcher --restart-interval-hours 3 \
  --run-name cw3_frozen_phi --steps 25000000 $TRAIN $ARCH $CLEAN \
  --win-prob-pbrs-coef $COEF --win-prob-pbrs-source $PHI_SRC
```

### 2.4 THE 5M PRE-TEST — a PAIR, same seed, run first

```bash
$PY -m main.launcher --restart-interval-hours 3 \
  --run-name pt_sparse --steps 5000000 $TRAIN $ARCH $CLEAN

$PY -m main.launcher --restart-interval-hours 3 \
  --run-name pt_shaped --steps 5000000 $TRAIN $ARCH $SHAPED
```

⚠️ **The pre-test is a RECIPE comparison, not a reward-isolating one.** `$SHAPED` carries
`--use-popart` and a ±30-sized critic support because that is the incumbent's *validated* recipe;
dropping PopArt from a ±30 dense stream is a known-bad configuration (value gradient swamps the
trunk) and would make the control a straw man. The consequence, stated so nobody reads past it:
the pre-test's contrast is `{±1, no hand shaping, no PopArt, support ±2}` vs
`{±30, 8 PBRS + 1 BIAS, PopArt, support ±12}` — **four differences, deliberately bundled**,
because the question it answers is "does the clean recipe learn at a comparable rate", not
"which of these four did it". If the owner wants the reward isolated instead, drop `--use-popart`
and re-size `--value-dist-vmin/-vmax` in `$SHAPED` too — and expect a degraded control.

**Its endpoint is OUTCOME-UNIT, never reward.** The two arms optimize different reward scales, so
`rollout/ep_rew_mean` is not comparable between them by construction. Read
`eval/win_rate_vs_bots`, `eval/elo`, and episode length at matched STEP.

### 2.5 Verification of record

```
$ python -m main.checkargs --argv "<each of the five argvs above>"
  cw1_sparse      110 flags, 0 unrecognized, ✓ this command still launches
  cw2_self_phi    111 flags, 0 unrecognized, ✓
  cw3_frozen_phi  112 flags, 0 unrecognized, ✓
  pt_sparse       110 flags, 0 unrecognized, ✓
  pt_shaped       115 flags, 0 unrecognized, ✓
```

`checkargs` also checks the `flag_registry` `requires` graph, so a dependency crash inside
`Gen3FeaturesExtractor.__init__` (later and dearer than an argparse error) is excluded too. The
counts above are over the FORWARDED flags; `--restart-interval-hours` is in `checkargs`'
`LAUNCHER_ONLY` set and is classified rather than rejected if you paste the whole launcher line.

---

## 3. PRE-FLIGHT

1. **Re-run `checkargs` on the day.** It is 2 seconds and it is the only thing that catches a
   flag deleted between this document and the launch.
2. **`git log -1`** — record the hash the arms launch on. The launcher pins a resumed run to its
   checkpoint's commit; a fresh run pins nothing.
3. **Confirm the frozen-φ source loads** before committing 25M steps to it (§6.5). The FROZEN-φ
   smoke in §5 is that check; re-run it if `$PHI_SRC` changes.
4. **The GPU is single-tenant.** Three 25M arms are sequential, not concurrent, unless the box
   has been re-provisioned.
5. **Long-run SOP applies** — watcher + chain + hourly off-minute repair cron, set up at launch.

---

## 4. WHAT A HEALTHY FIRST HOUR LOOKS LIKE

### 4.1 Startup banners — read these before walking away

| line | SPARSE | SELF-φ | FROZEN-φ |
|---|---|---|---|
| `[Reward] composition:` | `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)` | same | same |
| `[ModelVersion] Round-trip smoke test PASSED` | required | required | required |
| `🧊 [WinProbPBRS] frozen φ from …` | absent | absent | **required**, and it prints the source's `arch_signature` + `config_version` |
| `[Reward] ⚠️ ORDERING` | must NOT appear | must not | must not |
| `[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE` | must NOT appear | must not | must not |
| `[Reward] ⚠️ TERMINAL SCALE` (draw magnitude) | must NOT appear | must not | must not |

Any `⚠️` from those last two rows means the terminal magnitude and something sized against it have
drifted apart — **stop and fix, do not train through it**.

### 4.2 TensorBoard, first ~3 iterations

**Every arm:**

- **`train/value_loss` — ⚠️ THE V-LOSS-SCALE TRIPWIRE (`2d38a4a`). This is the scalar the runbook
  says to watch.** Under `--value-from-dist` the scalar MSE term is dropped from the loss but is
  still *computed and logged* as the E[Z]-mean-vs-return diagnostic, so it means the same thing
  either way: mean squared error of the critic's mean against the realized return. With PopArt
  retired it is in **raw return units**, hence directly interpretable for the first time — a critic
  that predicts nothing but the unconditional mean scores ≈ `Var(return) ≈ 1.0` on a ±1 outcome at
  a ~50% win rate. **Above ~1.0 and not falling ⇒ the critic is worse than a constant.** It should
  fall well below 1 within the first few iterations and `train/explained_variance` should climb
  above 0. Measured in the SPARSE smoke (§5): `0.337 → 0.243 → 0.222` with EV `−0.19 → 0.14`.
  (On the incumbent `$SHAPED` arm the same scalar is in PopArt-normalized units and this reading
  does not apply.) Companion in support units: `value_dist/mean_abs_err`.
- `popart/sigma`, `popart/mu`, `popart/value_weight_norm` — **must be ABSENT.** Their absence is
  the confirmation that PopArt is retired; their presence means `--use-popart` was inherited or
  re-typed.
- `train/clip_fraction_vf` — with `--clip-range-vf none` there is no value clipping.
- `value_dist/pit_mean` ≈ 0.5 ⟺ calibrated; `value_dist/mean_abs_err` and `value_dist/std` are now
  in ±1 units, so `mean_abs_err` ≳ 1 is a critic that cannot tell a win from a loss.
- `train/scaffolding_gauge` / `_rho` / `_n` — the shaped-vs-game value gap. On SPARSE the two
  readouts should agree closely from early on (there is no scaffolding to disagree about); a
  persistent gap there is a finding about the critic, not about shaping.
- `rollout/ep_len_mean` and the stall rate — **the primary safety endpoint**, from turn one.
- `🏁 Episode Finished | Reward: ±1.00` in the log — the ±1 terminal, visible per episode.

**The φ arms additionally (`train/pbrs_*`), and this is the sanity gate the arm must pass before
it is believed:**

| scalar | what it is | expected at `coef 0.3` |
|---|---|---|
| `train/pbrs_episode_dose` | **THE sizing meter** — a complete episode's discounted shaping budget as a fraction of one win. By the telescoping identity = `coef·E[φ(s₀)]/V` | **`0.3 × pbrs_phi_mean`**, so ≈ 0.15 at a calibrated φ≈0.5 and higher while the head is optimistic. Measured in the SELF-φ smoke (§5): **0.187 at `phi_mean` 0.680** — i.e. the shaping is worth ~19% of a win per game |
| `train/pbrs_episode_dose_n` | how many complete episodes that averaged | non-zero within the first iteration or two |
| `train/pbrs_terminal_share` | per-step \|shaping\| ÷ \|terminal\|, always defined | a small positive number, stable across rollouts |
| `train/pbrs_reward_share` | the legacy dense-stream meter — see the ⚠️ below | **NaN on any rollout with no episode end** (R1's F3 — never `0.0`), **and length-driven where it IS defined. Do not size on it.** The SELF-φ smoke measured it at **0.425** against a true dose of 0.187 — a 2.3× disagreement |
| `train/pbrs_shaping_mean` | ≈ 0 by construction (the term telescopes) | near zero |
| `train/pbrs_phi_mean` | the potential's mean level | ≈ 0.5 early |

🚨 **`pbrs_reward_share` is structurally uninformative on this arm and is no longer the number to
read.** Its denominator is the UNSHAPED stream's own mean \|reward\|, and `--no-hand-shaping`
makes that stream terminal-only: exactly 0 on any rollout with no episode end, and
"±1 ÷ episode length" otherwise — which moves the meter with the episode length rather than with
the coefficient. **R1's F3 fixed the worse half** (it read `0.0` — "negligible" — for the one case
where the shaping is 100% of the reward; it now reads **NaN**), but NaN is honest, not a size.
Probe N §7.5 named the rest before the arm was buildable; `pbrs_episode_dose` and
`pbrs_terminal_share` are the built companions and their denominator is the run's own
`--victory-value`, a constant.

**If `pbrs_episode_dose` reads ≪ 0.05, the arm is homeopathic and is measuring nothing** — stop,
do not let it run a generation and then read the null as a verdict on the lever. That is precisely
the failure §7.2 of `design_winprob_behavior_coupling.md` was written to prevent.

🎯 **The dose's TRAJECTORY separates the two φ arms, and it is the cheapest live check that the
frozen source is actually being read. This was MEASURED in the smokes, not predicted.** Same seed,
same code, three iterations:

| | `pbrs_phi_mean` | `pbrs_episode_dose` |
|---|---|---|
| **SELF-φ** (a head being trained) | `0.680 → 0.347 → 0.216` | `0.187 → 0.104 → 0.087` |
| **FROZEN-φ** (a fixed function of state) | `0.403 → 0.391 → 0.391` | **`0.231 → 0.234 → 0.228`** |

That is §2.4's "exact invariance per rollout, approximate across" showing up as a number. **A
FROZEN-φ arm whose dose wanders like SELF-φ's means the frozen source is not being read** — check
for the `🧊` banner. And SELF-φ's collapse is not a defect: it is the caveat the frozen arm exists
to remove, which is what makes `FROZEN − SELF` a named quantity rather than a hope.

### 4.3 FROZEN-φ only — the constancy prediction (`db9bb5c`)

PBRS with a *good* potential drives `V_shaped` toward a **constant**: all evaluative content
migrates into the reward stream and advantages go fully local. That is the design working as
intended, and it is a **checkable prediction**, so it is a cheap sanity row for this arm rather
than an alarm:

```bash
python -m main.scaffolding_gauge models/cw3_frozen_phi --constancy
```

- `v_std` / `dispersion` **falling over checkpoints** ⇒ the theory holding.
- ⚠️ `train/scaffolding_rho` will **fall** as this happens, and that is *ambiguous by
  construction* — "the heads diverged" and "V ran out of variance to rank with" look identical in
  ρ. Read ρ **beside** the constancy row, never alone.
- `within_frac → 0` is the FAILURE mode the raw `v_std` cannot distinguish: V has become a
  per-battle constant, i.e. a matchup lookup rather than a position evaluator.
- The outcome-readable quantity in a shaped arm is **`V_shaped + coef·φ`**, not `V`. The shaped
  critic's informative content is the RESIDUAL — where this generation disagrees with the frozen
  ancestor.

Nothing in this row applies to SPARSE (no potential) and it is weaker on SELF-φ (a co-evolving
potential is not the fixed φ the result assumes).

---

## 5. THE SMOKES — evidence, not assertion

Every arm was run as a real `--debug --steps 8000` job on CPU with the serverless rust bridge.
The smokes differ from the launch argvs only in ways orthogonal to what they check: `--debug`
(1 env, CPU, no eval), no `--device cuda`, no `--compile-*` (`--compile-trainer` REFUSES a
non-cuda device by design), no `--self-play` / `--warmstart-*` / eval sizing (nothing eval runs
under `--debug`), `--n-epochs 2`, and `--batch-size 512` so the single-env 2048-row rollout
divides evenly by batch × grad-accum.

### 5.1 SPARSE — `--steps 8000`, exit 0, `Training complete`

| check | result |
|---|---|
| `[Reward] composition:` | ✅ `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)` |
| round-trip | ✅ `[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: (1, 512))` |
| **zero shaping keys** | ✅ the string `pbrs` appears **0 times** in the entire 3038-line log; no `train/pbrs_*` scalar exists in the tfevents |
| PopArt retired | ✅ no `popart/mu`, `popart/sigma`, `popart/value_weight_norm` |
| `[Reward] ⚠️ ORDERING` / `⚠️ TERMINAL SCALE` / `⚠️ VALUE-DIST SUPPORT` | ✅ none appears |
| terminal magnitude | ✅ `🏁 Episode Finished \| Reward: -1.00 \| Status: LOSS`; per-turn `[REWARD] … Base: +0.0000` throughout |
| `train/value_loss` (the tripwire) | ✅ `0.337 → 0.243 → 0.222`, well under the ≈1.0 predict-the-mean level |
| `train/explained_variance` | ✅ `−0.195 → 0.051 → 0.140`, rising |
| `train/scaffolding_gauge` | ✅ present (`0.507 → 0.420 → 0.318`), `scaffolding_n` 2048 |
| recorded `model_config.json` | ✅ `victory_value 1.0 · draw_penalty −1.0 · hand_shaping false · use_popart false · win_prob_mode read_only · win_prob_pbrs_coef 0.0 · value_dist_vmin/vmax −2.0/2.0 · config_version 107` |

The `pt_sparse` pre-test arm is this configuration at `--steps 5000000`; the smoke covers both.

### 5.2 SELF-φ — `--steps 8000`, `--win-prob-pbrs-coef 0.3`

| check | result |
|---|---|
| `[Reward] composition:` | ✅ `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)` |
| round-trip | ✅ PASSED |
| `train/pbrs_*` present | ✅ all seven scalars |
| **`pbrs_episode_dose`** | ✅ `0.187 → 0.104 → 0.087` over `63 → 46 → 48` complete episodes (`pbrs_episode_dose_n`) |
| **the telescoping identity, LIVE** | ✅ `pbrs_phi_mean` `0.680 → 0.347 → 0.216`, and `coef·φ = 0.3·φ` = `0.204 / 0.104 / 0.065` against a measured dose of `0.187 / 0.104 / 0.087`. The identity holds on real episodes, not only in the unit test (the residual is φ at episode STARTS vs the buffer mean) |
| `pbrs_terminal_share` | ✅ `0.01306 → 0.00727 → 0.00502` = `shaping_absmean` ÷ `victory_value` 1.0 |
| `pbrs_shaping_mean` | ✅ `−0.0058 → −0.0024 → −0.0021`, ≈0 as the telescoping requires |
| ⚠️ `pbrs_reward_share` | `0.425 → 0.317 → 0.210` — defined here (episodes DO end inside a 2048-step single-env rollout) and **2.3× / 3.1× / 2.4× the true dose**, drifting with a different quantity. The companion is not decoration |
| PopArt retired | ✅ no `popart/*` |
| `train/value_loss` | ✅ `0.445 → 0.266 → 0.234`, falling, under the ≈1.0 level |
| `train/scaffolding_*` | ✅ gauge `0.502 → 0.542 → 0.242`, `n` 2048 |

### 5.3 FROZEN-φ — `--steps 8000`, `--win-prob-pbrs-coef 0.3 --win-prob-pbrs-source …`

**Run twice, and the first run is evidence too.**

**(a) THE NEGATIVE PATH, unintentionally measured.** The first attempt used the relative
`models/ai_v9_70_R3ACTION_0828/final_model.zip` from a git worktree, where `models/` does not
exist. Result:

```
[WinProbPBRS] FATAL: could not load --win-prob-pbrs-source models/ai_v9_70_R3ACTION_0828/final_model.zip:
  --stable-opponents: no model .zip found for '…' (expected a run dir with best_model/best_model.zip,
  a direct .zip, or a run dir + @step).
```
exit code **3 = `FATAL_CONFIG`** — a loud, named refusal at startup, exactly the designed
behaviour (never a crash-restart loop, and never a silent fall back to live-φ). It is also the
reason `$PHI_SRC` is an absolute path (§6.5).

**(b) THE REAL RUN**, with the absolute path:

| check | result |
|---|---|
| `[Reward] composition:` | ✅ `1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)` |
| **the frozen source LOADS** | ✅ `🧊 [WinProbPBRS] frozen φ from /home/goodlad/dev/gen3ai/models/ai_v9_70_R3ACTION_0828/final_model.zip on cpu (arch_signature=gen3_critic_route_wave_v1, config_version=107) — the LIVE win-prob head is now a diagnostic only` |
| round-trip | ✅ PASSED, *after* the frozen model was attached |
| `train/pbrs_*` | ✅ `episode_dose` **0.231 → 0.234 → 0.228** over `65 → 58 → 57` episodes · `phi_mean` `0.403 → 0.391 → 0.391` · `terminal_share` 0.0156 · `shaping_mean` −0.0074 |
| 🎯 **the potential is FIXED, and the meter shows it** | ✅ FROZEN-φ's dose is **flat** (`0.231 → 0.234 → 0.228`, φ `0.403 → 0.391 → 0.391`) where SELF-φ's collapses (`0.187 → 0.104 → 0.087`, φ `0.680 → 0.347 → 0.216`) — same seed, same code, three iterations. **§2.4's "exact invariance per rollout, approximate across" is now a MEASUREMENT**, and it is the cheapest possible live check that the frozen source is actually the thing being read |
| **the source is genuinely being read** | ✅ `phi_mean` **0.403** here vs **0.680** on SELF-φ at the same iteration of the same seed — a different network, and the difference has the right SIGN: the frozen MATURE head reads episode STARTS high (implied φ(s₀) ≈ 0.77 from the dose) and the losing mid-game states this random policy produces low, while the untrained live head does the opposite |
| ⚠️ `pbrs_reward_share` | 0.491 against a true dose of 0.231 — **2.1×**, the same disagreement as SELF-φ |
| PopArt retired | ✅ no `popart/*` |
| `train/value_loss` | ✅ `0.340 → 0.253 → 0.216`, falling |
| `_excluded_save_params` | ✅ the saved checkpoint's pickled `data` carries `win_prob_pbrs_coef`, `win_prob_pbrs_terminal_scale` and `_pbrs_metrics` — and **no `_winprob_phi_source`**: the frozen foreign model is not embedded in our checkpoint |

### 5.4 SHAPED — the 5M pre-test's INCUMBENT control arm

| check | result |
|---|---|
| `[Reward] composition:` | ✅ **`1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)`** — the validated ai_v8 composition, i.e. the control really is the incumbent |
| recorded `model_config.json` | ✅ `victory_value 30.0 · draw_penalty −35.0 · hand_shaping true · all_shaping_pbrs true · use_popart true · win_prob_mode shaping · value_dist_vmin/vmax −12/12` |
| **PopArt PRESENT** | ✅ `popart/mu −5.06 → −6.13 · popart/sigma 5.16 → 5.54 · popart/value_weight_norm 0.194 → 0.181` — which is what makes their ABSENCE on the clean arms a signal rather than an assumption |
| `⚠️ VALUE-DIST SUPPORT` | ✅ silent, as required: with PopArt on the support is in σ units and the guard's comparison is meaningless |
| `[Reward] ⚠️ ORDERING` | ✅ silent (`draw_penalty −35 ≤ −victory_value −30`) |
| `train/value_loss` | **`4.71 → 3.15 → 2.03` — in PopArt-NORMALIZED units.** Not comparable to the clean arms' `~0.2–0.4`; this is exactly why the §4.2 tripwire reading is scoped to the PopArt-off arms |
| no `train/pbrs_*` | ✅ this arm carries no PBRS coefficient |
| reward scale, visible | ✅ `rollout/ep_rew_mean` **−40.4 … −40.9** here vs **−0.94** on the clean arms — the ±30 + dense-shaping stream against the ±1 terminal-only one |

⚠️ **Honest provenance for this arm only.** It ran to completion (`exit 0`; 4 rollouts and 3 train
steps in its tfevents), but its stdout redirect ended up empty, so the composition line is not
quoted from a captured banner — it is computed from this run's own recorded `model_config.json`
through `format_reward_composition`, the same pure function the startup banner prints. Every other
row is read from its live tfevents. The three EXPERIMENTAL arms' banners in §5.1-5.3 are quoted
from their captured logs.

---

## 6. SIZING NOTES — raised, and where not solved, said so

### 6.1 ⚠️ THE COEFFICIENT LADDER, RE-STATED FOR A ±1 TERMINAL — with the arithmetic

`design_winprob_behavior_coupling.md` §7.2 re-sized E1's ladder from `{0, 0.1, 0.3}` to
`{0, 3, 9}` because the first draft "assumed a terminal reward of order 1, and the live scale is
`VICTORY_VALUE = 30`". **The clean world makes the terminal exactly ±1, so that correction has to
be applied in reverse — and getting the direction wrong here is the same class of error.**

**The doc's rule, and the invariant it is defined by.** The per-episode discounted shaping sum
telescopes to exactly `−coef·φ(s₀)`, and `φ ∈ [0,1]`, so the shaping's whole per-episode budget is
bounded by `coef`. The ladder's unit is therefore that budget expressed as a fraction of the
terminal magnitude — which is the doc's own language: *"coef 9 caps the per-episode total at 9,
~30% of the terminal"*.

```
dose  ≡  (per-episode shaping budget)/|terminal|  =  coef/V

  at V = 30 :  coef {0, 3, 9}          ⇒  dose {0, 0.10, 0.30}      (the doc's ladder)
  at V = 1  :  dose {0, 0.10, 0.30}    ⇒  coef = dose x V = {0, 0.1, 0.3}
```

**⇒ THE LADDER AT `--victory-value 1.0` IS `--win-prob-pbrs-coef ∈ {0, 0.1, 0.3}`.**

(That `coef/V` is the **bound** — the doc sizes with `φ ≤ 1`. The **realized** dose is
`coef·E[φ(s₀)]/V`, so at a calibrated φ ≈ 0.5 the top rung delivers ≈ 0.15 of a win per episode,
which is what `train/pbrs_episode_dose` reads back. Keep the two apart: the ladder is defined on
the bound, the meter reports the realization.)

The first draft's numbers come back, and for exactly the reason they were wrong at 30: they were
always *fractions of the terminal*, and the terminal changed. The error was never in the digits —
it was in leaving the unit implicit.

**The second anchor, and where the ledger's `2c` spelling comes from.** The textbook potential is
`φ* = V*(s) = E[terminal | s]`. At a ±1 terminal with a calibrated head that is `2p − 1`. Probe N
§6.5 rules that the [−1,+1] mapping is spelled as a **coefficient on φ = p**, never as a `2p − 1`
potential — the terminal convention `φ(s′) := 0` is correct for a [0,1] potential and wrong for a
[−1,+1] one, and the `−1` offset pays a per-step `+1e-4·coef` bonus for LONGER episodes, the wrong
sign in an arm with no anti-stall term. So at ±1:

```
full-strength, myopia-inducing shaping  =  --win-prob-pbrs-coef 2.0  on  phi = p
the ladder {0, 0.1, 0.3}                =  {0, 5%, 15%} of that
```

🚨 **`--win-prob-pbrs-coef <2c>` in `132d198`'s argv is a SPELLING rule, not an instruction to
double the ladder.** The `2` is the [0,1]→[−1,+1] factor. Read as "type twice your intended
fraction" it launches at `{0, 0.2, 0.6}` — 20% and 60% of a win, **2× the ai_v12 dose**. Stated in
the ledger's own `2c` form the ladder is `c ∈ {0, 0.05, 0.15}` and `2c = {0, 0.1, 0.3}` — the same
three numbers on the command line.

⚠️ **One inconsistency inside the ai_v12 doc, flagged so nobody re-derives from it.** §7.2 says
"`coef = VICTORY_VALUE` makes `coef·φ(s)` an estimate of the expected terminal reward". At V = 30
the calibrated expected terminal is `60p − 30`, so the V\*-matched coefficient is **60, not 30** —
the doc conflates "the terminal magnitude" with "the V\*-matched coefficient", which differ by
exactly 2. It does **not** move the ladder, because the ladder is defined by its fractions of the
terminal magnitude, not by that claim.

**Why the two φ arms run at ONE coefficient, and why it is the top rung (`0.3`).** They must share
a coefficient or `SELF − SPARSE` and `FROZEN − SELF` confound *source* with *dose*. `0.3` because
the clean world has deleted every other dense term, so a homeopathic arm collapses onto SPARSE and
the ladder measures nothing — the doc's own "deliberately aggressive rather than safe" rung is the
right place to sit when the null is "shaping does nothing". If the owner prefers the middle rung,
`0.1` is the registered alternative; **decide at registration, never mid-run.**

### 6.2 POPART RETIREMENT — no flag needs building; the answer is OMISSION

**`--use-popart` is already opt-in.** It is `action=BoolFlag, default=None`, and
`main/train/config.py` resolves an unpassed value with `_resolve("use_popart", False)` — for a
**fresh** run (no `--model`) that is **`False`**. Production carries PopArt only because every
production command types `--use-popart` explicitly (the live gen-15 argv does). So:

> **PopArt is retired by leaving `--use-popart` off the command line. There is no gap and no new
> flag.** `$TRAIN` above deliberately does not contain it.

Two mechanical facts that go with it:

- It is **version-checked and cannot be toggled on a resume** (`model_version/compat.py` — a
  mismatch breaks the value head's `state_dict`). Correct for a fresh generation, and it means a
  clean arm can never silently acquire PopArt on a restart.
- **`--clip-range-vf none` composes with PopArt OFF.** The guard in `config.py` fires only when
  `use_popart and clip_range_vf is not None`; `none` on its own is simply accepted. Verified by
  the smokes, all of which carry `--clip-range-vf none` and no `--use-popart`.

**Keep `--clip-range-vf none` anyway.** Its default is `0.5` in **raw value units**: on the ±30
incumbent that was 1.7% of the outcome range (tight), on a ±1 stream it is 50% (loose but binding
on early jumps). It has never been sized for either, because production always ran PopArt + none.
Passing `none` keeps the clean arms' effective behaviour identical to the incumbent's — no value
clipping — instead of silently introducing a constraint nobody chose.

### 6.3 🚨 THE OTHER CONSTANT SIZED AGAINST ±30 — the distributional critic's SUPPORT

**This was a live launch-blocker and is now fixed in `$CLEAN` and guarded in code.**
`--value-dist-vmin/-vmax` are a FIXED atom support in the **same space as the value target**, and
that target is the PopArt-normalized return when PopArt is on and the **raw** return when it is
off (`instrumented_ppo/ppo.py` — `popart.normalize(returns) if popart is not None else returns`).
Retiring PopArt therefore moves the support into raw reward units, where the production
`[−12, +12]` spans **12× the entire ±1 outcome range**. HL-Gauss smooths with `σ = 0.75·Δ`, so at
51 bins Δ = 0.48 and σ = 0.36 against a 2-wide return range: the whole win/loss axis lives inside
~4 of 51 bins. Under `--value-from-dist` **that head IS the critic**, so this is not a degraded
diagnostic — it is a critic that cannot resolve a win from a draw.

🔴 **THIS IS R1's F1, AND ITS LAUNCH RULE LIVES HERE BY REGISTRATION** (ledger `4f00d07`):
*"clean-world + `--value-from-dist` + no-PopArt requires the dist support resized to ±1 — the guard
warns, nothing stops the run."* R1 found it by adversarial audit and this runbook found it by
sizing the arms; the two agree, and the launch rule is the runbook's to carry.

- **Fix, in `$CLEAN`:** `--value-dist-vmin -2.0 --value-dist-vmax 2.0` (Δ = 0.08, σ = 0.06),
  which brackets the ±(1 + coef) return range with headroom.
- **Guard:** `_terminal_scale_guards` in `main/train/config.py` prints
  `[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE` when the dist head is on, PopArt is OFF, and
  the support either fails to bracket `max(victory_value, |draw_penalty|)` or quantizes it into too
  few atoms — naming the bin width and the atom count the outcome axis is left with. Silent under
  PopArt (where the units are σ and the comparison is meaningless) and silent on the sized `$CLEAN`
  support. It **WARNS, never refuses**: a wide support can be a deliberate choice, and a command
  that works today must not become a `FATAL_CONFIG`. **So the operator is the enforcement** — read
  the startup banners (§4.1) before walking away.
- **The same function carries R1's F2**, which this runbook would otherwise have walked into:
  `--victory-value 1.0` with an INHERITED `--draw-penalty -35.0` passes the ordering guard (a draw
  IS worse than a loss) while making the timeout 35× a clean loss — "1 TERMINAL" that is really a
  stall-avoidance objective. `$CLEAN` pairs `1.0` with `-1.0` explicitly for this reason, and
  `launch_runbook_test.py` asserts the pairing.

### 6.4 `--ent-coef` is NOT rescaled — this is deliberate

`0.02` was tuned against a ±30 dense stream, and the instinct to divide it by 30 is **wrong**. SB3
normalizes advantages **per minibatch** (`if self.normalize_advantage`), which forces the
policy-gradient term's scale independent of the reward scale — so `ent-coef` competes against a
normalized surrogate either way. Probe N §7.3.

What *does* change is the advantage **signal-to-noise before** normalization: sparser rewards ⇒
noisier advantages ⇒ the normalized gradient is relatively noisier. **If anything the clean arm
may want MORE exploration, not less.** This is an open sizing question carried into the run, not a
mechanical rescale, and it is not a mid-run knob: pick it at launch.

### 6.5 The frozen potential's identity must be PINNED

A clean-world run is uninterpretable if the identity of its φ is not recorded. `--win-prob-pbrs-source`
lands in `metadata.json` via `cli_args`, and `model_build` **prints** the resolved zip, its
`arch_signature` and its `config_version` at startup — copy that line into the arm's record.

🚨 **USE AN ABSOLUTE PATH, and `$PHI_SRC` above is one.** The path is resolved against the
CHILD PROCESS'S CWD, and the launcher's `Popen` passes no `cwd=` — so the child inherits whatever
directory the operator launched from, while its `PYTHONPATH` points at the pinned worktree. From
the main checkout a relative `models/…` works (that is why the live run's `--distill-teacher`
does); **from any worktree it does not, because `models/` exists only in the main checkout.**
This was measured, not reasoned: the first FROZEN-φ smoke, run from a worktree with a relative
path, exited `FATAL_CONFIG` at startup (§5.3). Worktree isolation moves the *code*, not the cwd.

`$PHI_SRC` is set to the rev-3 fold's cheap-arm final (`ai_v9_70_R3ACTION_0828`). The
requirement on a source is that it shares our `arch_signature` (an obs-FAMILY check, so a
prior-generation φ is viable) and that its win-prob head is mature. Alternatives at the same
signature: `ai_v9_71_R3ACTIONHI_0828`, or `ai_v9_72_R3SELF_0828`'s final once that run lands.
**Whichever is chosen, all three arms' comparability depends on FROZEN-φ's source never changing
mid-experiment.**

### 6.6 Two smaller sizing notes

- **`--vf-coef 0.5` is resume-immutable and FATAL on mismatch** (`check_vf_coef`). It must be
  right at launch. It scales a value loss that is now in raw ±1 units rather than
  PopArt-normalized ones — carried unchanged, deliberately, so the arms differ from the incumbent
  in the reward composition rather than in the loss weighting, but flagged as untuned.
- **`--value-tail-weight 0.3`** blends in the CVaR of the worst value misses. It is relative
  (a fraction of the same per-sample squared errors) and therefore scale-free; carried unchanged.

---

## 7. ENDPOINTS

| endpoint | rank | instrument |
|---|---|---|
| **stall rate / cap-terminations, mean game length** | **PRIMARY (safety)** | `rollout/ep_len_mean`, the `[STALL LOGGED]` count, `<run>/stalls/` |
| anchored ELO at matched snapshot COUNT | primary (strength) | `python -m main.elo <run>`, and `<run>/snapshot_ladder/ladder.json` at run END |
| per-team piloted win rate, paired draws | primary (strength) | the matched-extraction harness |
| whiff / re-click / loop census | behavioural | `python -m main.prober.query loops` (carries its gen-15 baseline) |
| switch rate | behavioural | `prober.query`, and the stall canary's companion |
| scaffolding gauge + the constancy row | mechanism | `train/scaffolding_*`; `python -m main.scaffolding_gauge <run> [--constancy]` |
| exploitability | generation-level | `python -m main.exploitability` over admission artifacts |

🚨 **Reading an ELO has three rules.** The headline is `<run>/snapshot_ladder/ladder.json`
(dense, ±10), not `eval/elo` (±29); a rating is only final once the run is (BT re-solves every
node on every add and the newest is systematically inflated — gen-10's 12M fell 2089 → 2021 over
12 refits); and a cross-run comparison must be at matched snapshot **COUNT**, not matched step.
**Never narrate a mid-run ELO or delta.**

⚠️ **The h2h dilution note.** The incumbent reference comes free from existing 25M checkpoints, but
it is a *reference*, not a control: era-config differences (reward composition, PopArt, critic
support, arch generation) all move together. A head-to-head win rate against a differently-trained
incumbent measures the difference between two whole recipes and cannot attribute it. A purity
fourth arm is registered as an option **only if the three-arm verdict is close**.

---

## 8. ESCALATION RULES — pre-registered, and none of them fire at launch

1. **NO anti-stall bias at launch** (`627ab58`). With draw = loss, stalling is *weakly dominated*
   — any line with ε win probability beats it — so a bias is only needed if the model lands in a
   can't-win-won't-lose local optimum, which is an EMPIRICAL condition. *"The bias would have to be
   earned after we find the model refuses to win."*
2. **The bias enters ONLY if the stall-rate primary endpoint fires** — cap-terminations materially
   above the incumbent's rate at matched strength, with the exact trigger frozen at arm
   registration — and it enters as a REGISTERED change, never a mid-run patch.
3. **HEAD FIX BEFORE BIAS.** If probe O's residual stall-tail over-confidence is what drives it,
   the mitigation is a head repair (stall-tail labels from the factory — outcome-unit, theoretically
   clean), because FROZEN-φ pays positive reward while marching toward a cap exactly where the head
   historically over-reads. The repair stays inside the clean worldview; a bias term leaves it.
4. **No mid-run coefficient changes.** The φ arms' comparability is the experiment. A dose that
   reads homeopathic at the first-rollout gate (§4.2) is a RELAUNCH, not an adjustment.
5. **The F1/F2b no-progress-clock fixes stay OFF** (`--progress-decision-tense`,
   `--progress-switch-freeze`). They are retrain-class and they repair a BIAS term that
   `--no-hand-shaping` has already zeroed — they are irrelevant to these arms and would only
   change the `turns_since_progress` obs scalar. They belong to the incumbent-lineage question.

---

## 9. NAMED GAPS

| # | gap | severity | owner / disposition |
|---|---|---|---|
| G0 | **`--win-prob-pbrs-source` resolves against the CHILD'S CWD.** A relative `models/…` works from the main checkout and FATALs from a worktree. Measured, not reasoned (§5.3). | **HIGH but CLOSED here** — `$PHI_SRC` is absolute, and a wrong path is a clean `FATAL_CONFIG` at startup rather than a silent live-φ fallback | mitigated in the runbook; nothing to build. |
| G1 | **`--compile-trainer` × a frozen φ source is UNVERIFIED on real CUDA.** The frozen model is a separate policy object loaded eagerly and never compiled, and `compile_trainer_extractor` refuses a non-cuda device — so no CPU tier can reach the composition. Carried from wave A's own honesty note. | **HIGH** — it is on the FROZEN-φ launch path and the failure would be at startup | training session: launch FROZEN-φ with a 1-minute `--steps 20000` GPU pre-flight before the 25M command. If it fails, `--no-compile-trainer` costs ~38% of steps/hour and is a valid fallback. |
| G2 | **A multi-cycle self-play run with a frozen φ source has never run end-to-end.** Every leg is gated; the composition (frozen source × pool promotion × eval workers × launcher restart) is not. Restart is the specific risk: `--win-prob-pbrs-source` is inherited on a flagless resume, but a launcher restart re-loads the source from its path. | MEDIUM | training session: after the first 3h restart, confirm the `🧊 [WinProbPBRS]` line reappears in the child log. |
| G3 | **`--vf-coef 0.5` is untuned for a raw ±1 value loss** and is resume-immutable/FATAL on mismatch, so it cannot be revisited without a fresh run. | MEDIUM | owner: accept as carried-unchanged (the deliberate choice), or re-size before launch. Watch `train/value_loss` (§4.2). |
| G4 | **`--ent-coef` sizing is open** (§6.4) — not rescaled, correctly, but the advantage SNR genuinely worsens on a sparse stream and the direction of the desirable change is *up*. | MEDIUM | owner: a decision at launch. Deliberately not solved here. |
| G5 | **The pre-test bundles four differences** (§2.4). It answers "does the clean recipe keep pace", not "which ingredient". | LOW — by design, stated | owner: accept, or run the reward-isolating variant at a degraded control. |
| G6 | **`train/pbrs_reward_share` remains in the metric set** and still reads as a plausible number on a partly-dense stream. A future reader can still size on it by mistake. | LOW | mitigated: it is now OMITTED rather than zeroed when undefined, and both the module docstring and this runbook name `pbrs_episode_dose` as the meter. |
| G7 | **The critic-support hazard (§6.3) is a WARNING, not a refusal.** A launch that ignores the banner still trains. | LOW | deliberate: a support choice can be legitimately unusual, and the incumbent PopArt path must stay silent. The banner names the bin count the outcome axis is left with, which is the actionable number. |
| G8 | **`--victory-value` changes nothing about `MAT_HP_WEIGHT` / `MAT_ALIVE_WEIGHT`**, which are calibrated against the 30 scale. Inert here (Φ_mat is off under `--no-hand-shaping`), but a *partial* clean world — ±1 terminal with hand shaping ON — would be silently mis-scaled. | LOW (latent) | not built: no arm asks for it. Named so a future half-clean arm does not walk into it. |

---

## 10. PROVENANCE

Ledger: `4d22ae4` · `cfbc9bf` · `e22bd08` · `627ab58` · `2d38a4a` · `db9bb5c` · `132d198`.
Probe N's record: `designs/research_state/measurements/no_progress_tax_review_2026-08-29.md`
(§5 the config enumeration, §6 the frozen-source spec + the affine detail, §7 the sizing flags).
Design: `designs/ai_v12/design_winprob_behavior_coupling.md` (§2.4 the drifting-φ caveat, §6 E1,
§7.2 the coefficient correction this document reverses for ±1).
Code: `agents/training/winprob_pbrs.py` · `agents/training/reward_manager.py` (the composition
census) · `main/train/{parser,config,model_build}.py`.
