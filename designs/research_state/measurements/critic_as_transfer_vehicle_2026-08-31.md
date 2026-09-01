# M5 — IS THE CRITIC THE OFF-SLICE VEHICLE?

**2026-08-31 · owner-ordered, one of the six probes aimed at the (i)/(ii) fork the rev-4 scorecard
opened** (ledger: *either v8's gift came from something no gen-era arm varies, or the fold metrics
do not measure what produced it*).

Artifacts beside this file: `critic_as_transfer_vehicle_2026-08-31.json` (every number) ·
`critic_as_transfer_vehicle_probe.py` (phases `collect | analyze | report`; resumable) ·
`critic_as_transfer_vehicle_2026-08-31_tables.md` (the rendered tables, emitted by `report`, from
which every table below is copied verbatim).

---

## The question, and why it is not the same as "did the fold work"

Distilled behaviour can only generalise off-slice if the value function prices it correctly in
contexts the teacher never visited. Three measurements motivate suspecting the critic specifically:

* the critic pathway runs ~7× LOWER effective rank than the policy pathway at every checkpoint
  (steady state, not decay — the plasticity audit);
* the critic is the **main casualty** of action-level distillation — off-slice `|ΔV|` of 4–9 on a
  ±12 scale within a handful of steps (`distillability_index_gen_2026-08-28.md` §5.6);
* the win-prob head is aggregate-calibrated but **per-state blind** — per-state |error| 0.278
  against an aggregate bias of +0.036, AUC 0.679 against ground truth's 0.970
  (`exploiter_fingerprint_truthcheck_2026-08-31.md`).

So: **after a fold, on states drawn from teams it never taught, what actually changed — the
policy's action distribution or the critic's valuation, and which of the two ORDERS with the
measured per-team gift?**

## Method

**Two eras, each a (parent → fold) pair plus a fixed reference opponent that is an ancestor of
both and equal to neither.**

| era | parent | fold | fixed reference opponent | measured untaught outcome of the fold |
|---|---|---|---|---|
| **v8** | `ai_v8_04_distill_4teacher_0722` `final_model_interrupted.zip` | `ai_v8_14_distill3_0725` | `ai_v8_03_zarch_control_0718` | **GIFTED +5.42pp** [+3.44, +7.42] (probe P) |
| **gen** | `ai_v9_59_R2ACTION_0827` `final_model.zip` | `ai_v9_70_R3ACTION_0828` | `ai_v9_29_rev1_0823` | **null −0.75pp** [−4.56, +3.00] (probe Q) |

**States are GENERATED, not read from eval traces, and the reason is a mask.** The recorded
`states.npz` carries no `action_mask` and its `logits` are already-normalised log-probs, so the
`logits > -1e8` recovery returns ALL-LEGAL — the documented vacuous-guard trap that put 38.4%
phantom legality into a year of flip/KL audits. This probe plays its own battles and takes the
mask straight out of `embed_battle`, where it is the server-authoritative one.

**One pass, three networks, identical inputs.** For each probe team the actor pilots that ONE
pinned team against the reference opponent (greedy both sides, in-process bridge, no server), and
at every decision with ≥2 legal actions the **parent**, the **fold** and a **control** checkpoint
are all scored on the *identical* observation and the *identical* mask, inside the same process.
Nothing is re-derived offline, so a state cannot drift between arms.

**The meters.**

| axis | meter | what it is blind to |
|---|---|---|
| POLICY change | masked `KL(fold‖parent)`, total variation, argmax agreement | — |
| CRITIC change | Spearman ρ(V) pooled **and within-battle**, mean \|Δz(V)\| after per-arm z-scoring, ρ of the within-battle TD sequence, win-prob level shift | — |
| CRITIC quality | AUC(V, outcome), AUC(win-prob, outcome), and the **Murphy decomposition** of the win-prob Brier score into RELIABILITY (level error) and RESOLUTION (discriminating content) | — |

The Murphy split is what answers the mission's crux directly: `Brier = reliability − resolution +
uncertainty`, so *reliability* is exactly "is the level right" and *resolution* is exactly "does it
separate states that end differently". AUC is a second, purely rank-based resolution meter that
shares no arithmetic with it.

**The critic's ranking over ACTIONS is substituted, and the substitution is named.** The critic is
a `V`, not a `Q`; a genuine per-action ranking needs one-ply materialisation of every legal
successor (the prober's `lookahead`), which is seconds per action per state and does not exist on
the v8-era tree at all. What is reported instead is the ranking the critic induces **over states**,
computed *within battle* (the pooled version is dominated by the global who-is-winning axis that
both critics get right), plus the rank correlation of the within-battle **TD sequence** — the
credit the critic assigns to the transitions actually taken. That is decision-relevant but it is
not the action ranking, and no claim here should be read as if it were.

**PopArt is why |ΔV| is z-scored.** The two arms carry different PopArt shift/scale, so a raw
`|ΔV|` conflates an affine re-scaling with a change of shape. Each arm's V is z-scored within the
cell before differencing; what survives is the part of the critic's landscape that moved.

### The matched-noise control — why "the policy moved more" would otherwise be meaningless

KL (nats) and rank de-correlation (dimensionless) are different units; "the policy moved more than
the critic" is not a statement one can make by comparing them. Every fold change is therefore also
computed for a **parent ← earlier-checkpoint-of-the-parent's-own-run** pair, i.e. ordinary training
with no fold at all, and reported as a RATIO. The control is what makes the comparison scale-free.

| era | control checkpoint | control span | fold span | matched? |
|---|---|---|---|---|
| v8 | `ai_v8_04/checkpoints/checkpoint_269716291_steps.zip` | 7.46M steps (269.72M → 277.18M) | ~14.8M | **half** — the control understates ordinary movement, so every v8 ratio is an OVER-estimate of how special the fold was |
| gen | `ai_v9_59/snapshots/snapshot_000024000000.zip` | 4.07M steps (24.00M → 28.07M) | ~4.55M | yes |

### Labels: many states with 1-draw labels, not few states with tight-MC ones

The mission offered the truthcheck's tight-MC method and asked which was chosen. **Chosen:
realized battle outcomes, at scale.** A battle's outcome is one draw from the true value
distribution at every state that battle contains, so AUC, the Murphy reliability/resolution split
and the reliability curve are all consistently estimable from single draws with battle-clustered
inference; what R=40 MC buys is the resolution of an *individual anchor*, which is what the
truthcheck needed (it was classifying single states into a boundary band) and this probe does not.
The budget therefore bought states rather than rollouts. The cost of the choice is stated in
Limits: the label is the *actor's* continuation, so the arm that generated the states is judged
on-policy and the other off-policy — which is why both state sets were collected.

### Design

Battles are played by BOTH arms as the state generator (`--actor parent` and `--actor fold`) so
the on-policy/off-policy asymmetry can be read rather than assumed. `n = 12` paired battles per
(team, actor); the opponent's team draw and the sim seeds come from one `random.Random(20260831)`
sequence, identical across teams and actors.

| cell | teams | classes |
|---|---|---|
| `v8/parent` | 22 | 16 untaught + 6 taught (probe P's own probe set, resolved from its published per-team shas against the era pool) |
| `v8/fold` | 16 | untaught only |
| `gen/parent` | 14 | 8 untaught (probe Q's pre-registered selection) + 6 taught (one per rev-3 teacher cluster F6a–F6f, read from each arm's recorded `--trainee-teams`) |
| `gen/fold` | 8 | untaught only |

**The v8 arm runs in an era-pinned worktree** (`/tmp/probeP_v8era` @ `b13b30b2`) on the **node**
bridge — the era's rust bridge predates the seedless-seed fix `bc00d4d` and would replay one dice
stream. The gen arm runs on the current tree on rust. Both arms of a comparison always live in one
era and one tree; nothing is compared across trees.

**ACID.** Every collection process refuses to start unless the parent and fold state-dicts differ
(`L2 > 1e-3`): measured **238.923** (v8) and **51.835** (gen). A mis-resolved path that loads one
zip twice would otherwise read as a perfect null.

