# Negative transfer and the shared function — why you cannot teach eight teams without touching the other 711

> **What this is.** A durable explainer for one concept cluster: why a distillation fold on a
> handful of taught teams moves the policy on every untaught team, why that movement is one
> displacement judged from two places (the GIFT and the LEAK), what sets its sign, and whether the
> entity architecture changes any of it. Written to teach, intuitive first, then technical, no code.
> Grounded in the gen-era fold campaign (2026-08-27 → 2026-09-05), the v8 reproduction, and the
> flat-head → pointer-head architecture boundary at v51.

---

## TL;DR

- **There is one network, and a "team" is not a compartment in it.** A team is a region of the
  input space processed by the same encoders, the same transformer trunk and the same pointer head.
  A fold that says "on these eight teams, match the teacher" moves the shared weights, and every
  untaught team that leans on the same features moves with them.
- **Gift and leak are one displacement, not two effects.** The fold produces a single change of the
  shared function. Seen from a team it helps, we call it a gift; from a team it hurts, a leak. The
  loss-off control (C1, the fold with the distillation term switched off) showed neither, which is
  exactly what this picture predicts: no pull, no movement.
- **Content sets the direction; dose sets the magnitude.** The taught-to-untaught coupling is a
  kernel whose *sign* comes from whether the teacher's direction aligns with what untaught teams
  needed. That is why funded vs unfunded teachers separated the untaught outcome and K=3 / 6 / 12
  did not, and why a scalar divergence to the parent cannot tell a gift from a robbery.
- **Negative transfer and generalisation are the same mechanism with opposite sign.** v8's +4.64 on
  teams it was never taught is the same kernel that dug the gen-era's three-to-four point hole.
- **Every gen-era fold digs a hole then recovers** if the teachers are near the parent, and does not
  recover if they are funded. The mechanism story (dense supervised pull first, noisy RL restoring
  force later) is an interpretation; the recovery gap is a measurement.
- **The architecture changed across the v8 boundary in two ways that matter here**: v8 had the flat
  positional action head and a two-round physics-in-the-loop refine; the gen era has the
  pointer-native head and no loop. The sharing argument says entity structure *amplifies* transfer
  in both directions, so it cannot flip the sign by itself. The candidate that could is that
  **closed-form physics leaves a teacher only team-specific content to teach**, and team-specific
  content is what leaks. That hypothesis is pre-registered and under test (§4).

---

## 1. Intuitive level

### The soundboard

Tighten one string on a piano and the tension changes across the whole soundboard. Whether the
neighbouring strings go sharp or flat is not something you chose. It is a property of how the board
is built.

That is a fold. The distillation loss says "on states from these eight teams, produce the teacher's
move distribution". Gradient descent finds the *cheapest* weight change that does so, and cheapest
is measured in weight space, not in "how many other teams does this disturb". The features the
taught states lean on get moved. Every untaught state that leans on the same features moves with
them. In a shared trunk that is nearly every state.

### Why "gift" and "leak" are the same thing

It is tempting to picture a fold as doing two things: teaching the taught teams (good) and
disturbing the others (bad). It does one thing. It displaces the shared function. A team on which
the displacement happens to help calls it a gift; a team on which it hurts calls it a leak. They are
not separable at the update, only at the meter.

The cleanest evidence is the control arm with the loss switched off, C1. Same teachers in the
pool, same sampling, same dose, coefficient zero. It showed neither the on-slice gain nor the
untaught hole. No pull on the string, no movement on the board.

### The hole, then the recovery

Every frozen gen-era fold measured so far drops three to four points on untaught teams by about one
million steps in. With teachers near the parent (the unfunded R5F parents) it climbs back to
parent-neutral by the end. With funded teachers (their R5FUND forks, trained harder on their own
teams) it does not, and the difference in recovery between the two is roughly five points.

Here is a mechanism, held loosely because it is an interpretation and not a measurement. The
distillation term is a dense, low-noise, supervised gradient: every taught state pushes from the
first update. The reinforcement signal on untaught teams is sparse and noisy and needs whole
episodes to notice that behaviour regressed. So the supervised pull wins early and the untaught side
drops. Then RL pushes back, a restoring force. Near-parent teachers ask for a small displacement, so
the restoring force wins. Funded teachers ask for a large one and keep asking, so at the same dose
it cannot. A clean test would be a fold with PPO switched off entirely: if the hole never recovers,
the restoring force is real. It has not been run.

### Why v8 is the anomaly

v8's fold showed +4.64 on untaught teams at 1.09 million steps, before any restoring force could
have acted, and the gift then grew to a peak near +9.7 before decaying. It is a sign-flipped
transient: a hump where we get a hole. Under the soundboard picture that means v8's teachers'
direction was aligned with off-slice improvement from the very first update, or the 277-million-step
parent had a trunk geometry where the taught-to-untaught coupling ran positive. The gen-era data
cannot separate those. The replication in flight (three short arms to +1.09M on the era code) asks
the prior question first: does the no-hole fact even reproduce.

### Update, 2026-09-06: the anomaly dissolved

Three cells on v8's own line, era code and era meter, all forked from v8's parent at the same step and
stopped at the same depth of about 1.08M steps, with three arms each:

| cell | loss | teacher-team bias | teachers in the pool | untaught gain vs frozen parent |
|---|---|---|---|---|
| phase 1, the full recipe | on | 40% | yes | +4.56 [+1.14, +7.81] |
| cell 1 | off | off (the era ties bias to the loss) | yes | +4.92 [+1.63, +8.04] |
| cell 2, plain continuation | off | off | no | +3.45 [+0.46, +6.48] |

Paired on the same teams, the full recipe minus the plain continuation is −1.11 [−3.12, +0.91], inside
the replicate floor. **v8's recipe is equivalent to training its parent on with nothing added.** The gift
was the parent still learning, measured against a frozen copy of itself. Re-based on the continuation,
v8's celebrated +4.64 is about +1.2 and not significant. Two consequences. First, an untaught delta
quoted against a frozen parent overstates a fold by whatever the continuation would have gained, so
every such number needs a continuation control at matched depth; on a mature parent that is about 3.5
points. Second, the gift-versus-leak contrast between the eras is now a contrast between what the two
*parents* do when trained on, not between what the two folds do. Whether our 28M-step parent also gains
under a plain continuation is the open question (cell G5); if it does not, the difference is maturity,
and if it does, every hole in the gen-era record is deeper than recorded and every "neutral" fold forwent
progress. The mechanism in §1 and §2 stands unchanged; what changed is the baseline it is measured against.

## 2. Technical level

### The kernel

Write the parameters as one vector θ. The fold minimises the PPO loss plus a coefficient times the
Kullback–Leibler (KL) divergence from the teacher's action distribution, evaluated on taught states
only. The KL term contributes a gradient vector in θ. What one step along it does to an untaught
state s′ is, to first order, the inner product of that vector with the Jacobian of s′'s output
with respect to θ. Summed over the taught batch this is the **neural tangent kernel** between the
taught set and s′: ⟨∇θ f(s), ∇θ f(s′)⟩. In a trunk where every token passes through one shared
encoder, one shared transformer and one shared scorer, that kernel is large almost everywhere.

Three consequences follow, and the campaign measured all three.

- **Direction is content, magnitude is dose.** Dose (learning rate × epochs ÷ effective batch)
  scales the length of the displacement. Its direction is the teacher's. The sign of the kernel's
  effect on an untaught team is a property of direction. So halving or doubling dose (K=12 / 6 / 3,
  0.53× to 2.12× v8's) left the untaught outcome unchanged, while swapping funded for unfunded
  teachers changed it.
- **A scalar divergence is a norm, not a direction.** The offline collateral KL against the parent
  measures how far the off-slice outputs moved. Two folds with equal KL can be one that gifted and
  one that robbed. The dose arms separated cleanly by KL; the loss-on and loss-off arms did not,
  because the displacement *length* was similar and what differed was where it pointed. The meter
  that decides anything is win rate on untaught teams, measured directly.
- **Generalisation is the positive branch.** There is no separate machinery for "helpful spillover"
  and "harmful spillover". The kernel is the same object; alignment of the teacher's direction with
  the untaught teams' need sets which branch you are on.

### What the remedies buy and cost

- **Anchor or trust region on off-slice displacement** (`--distill-anchor-target-kl` and its
  monitor). Bounds the leak by bounding the displacement seen from untaught states. It bounds the
  gift by the same amount, because they are one displacement. It trades the chance of a v8 for
  protection against a robbery.
- **Gradient projection** (`--distill-anchor-mode grad_project`). Removes the component of the
  distillation gradient that changes off-slice behaviour and leaves PPO's gradient free. Cleaner in
  principle, since it separates the two *sources* at the update. If gift and leak share a subspace,
  projection kills both, and the subspace is estimated from a finite sample of off-slice states.
- **Rehearsal with the parent as teacher on untaught teams.** The continual-learning classic. It
  makes the parent the ceiling on every untaught team, which forfeits the thing we want.
- **Teacher selection by direction.** The fleet-geometry measurement found six teachers pointing six
  different ways. If a gift requires alignment between teacher direction and untaught need, the
  lever is *which* teachers, not how hard. This is the only remedy that could reproduce a gift
  rather than merely prevent a robbery, and it is the one we understand least.

## 3. Does the entity architecture change any of this?

### What actually changed across the v8 boundary

Checked from git and the runs' `model_config.json`, not from memory. At v8's commit (b13b30b2) the
trunk was identical to today's: two transformer layers, width 128, four heads, twelve mon tokens
plus a global token. Two things differed:

- **v8 had the flat action head.** Eleven logits came from SB3's stock linear layer over the policy
  latent. The logit for "move slot 2" had its own weight column regardless of which move sat in slot
  2 on this team. The pointer-native head (`gen3_pointer_native_v1`, v51) landed 2026-08-03, after
  v8's fold; today move k is scored from move k's own token plus its physics cells and switch j from
  mon j's token, with one shared scorer per entity family.
- **v8 had a loop.** Its config carries `damage_refine_rounds: 2` with the outgoing and status
  refine flags on: a lean damage kernel re-ran *between* the two transformer layers from the current
  move belief and injected a per-mon threat summary back onto the tokens, both directions. That is a
  small physics-in-the-loop recurrent trunk. It was deleted at v50 because it was about a fifth of
  the per-forward CPU and the trunk-enrichment audit found it near-inert three times running
  (ledger K9/K10).

So v8 gifted on a flatter head with a loop, and the gen era robs on a more structured head without
one. Everything else changed too (a 277M-step parent vs 28M, a different pool, the frame deletion,
fifteen edge families, the move seats), so this is one correlation across one boundary.

### The sharing argument, and which way it points

How much a taught-state update moves an untaught state is the kernel, and the kernel is set by how
much of the network the two share. Three candidate stories, sorted by what they do to sharing:

**"The flat model had to infer structure, so cross-team effects were easier."** This runs backwards.
In the flat head, team identity is not slot identity, so the last layer had to carry "which mon is
in slot 3" implicitly in the latent, and two teams shared a head update only if their latents landed
near each other. The pointer head factors the score as a function of the entity's token and the
context with one scorer for all six slots. A lesson written as "with this Skarmory token in this
context, Spikes scores high" is now read by every team carrying Skarmory. That is *more* sharing at
the head. More sharing makes cross-team effects larger in both directions; it does not choose the
sign. On this argument alone the entity head should make v8-style gifts easier and robberies easier
too, and it cannot explain a sign flip.

**"Rigid structure should help."** It does, for the generalist learning on its own. Permutation
equivariance is structural now where the flat model had to learn it from every slot ordering. Cold
start is uniform over legal actions by construction. Type effectiveness, damage rolls, speed order,
status landing, trapping and chip are computed, not learned, and delivered as per-action cells and
as fifteen families of attention bias. Faster early learning is what fresh generations show.

**The version that could flip the sign.** Split what a specialist teacher knows into *general game
skill* and *team-specific tactics*. General skill gifts, because it is true on every team.
Team-specific tactics leak, because they are true on eight teams and wrong on some of the other
711. The more general skill the architecture supplies in closed form, the less of it a teacher has
left to teach, so whatever remains of the teacher's advantage over the parent is more team-specific
*by construction*. Rigidity helps the generalist and eats the gift at the same time. v8's model
computed damage too, so this is a matter of degree, but the edge families, the move seats and the
pointer cells all landed after v8, and each moved a piece of general knowledge from learned to
supplied.

### What the pre-registered probes test (2026-09-05)

Both are offline, CPU-only, on existing checkpoints; results land under
`designs/research_state/measurements/arch_transfer_2026-09-05/`.

- **Content locality.** KL(teacher ‖ parent) on states from the teacher's own teams versus untaught
  teams, per teacher, cluster-bootstrapped over teams, against a matched-noise floor from two
  adjacent parent checkpoints. Prediction: gen-era teachers are *more local* than v8's three, and
  within the gen era the funded teachers that robbed are more local than the unfunded ones that were
  neutral. The within-era half is the decisive one because it has no architecture confound. If
  locality is similar across eras or reversed, the architecture story is dead.
- **Sharing kernel.** Cosine between per-state score-function gradients for taught versus untaught
  states, decomposed by parameter group (encoders, transformer, action head, projections), on the
  pointer-head parent and the flat-head v8 parent, with a permutation null over team labels.
  Prediction: a higher cross-team ratio on the pointer head, concentrated in the head group. This
  tests the sharing argument's premise, not the sign.

**Sharing-kernel result (2026-09-05, `arch_transfer_2026-09-05/sharing_kernel/`): NOT DETECTED, and
the premise failed.** The gen-era cross/within ratio was higher than v8's in direction only
(delta +0.28, paired CI [−0.27, +0.95], permutation p 0.19), the action head was the *one* group where
the difference ran the other way, and the pointer head carries **0.66% of the policy-gradient norm**
against the flat head's 6.2%. In the current model 85% of that norm sits in the encoders (52%) and
the team transformer (34%). Within either era the taught/untaught split is not a direction the kernel
distinguishes at all (cross cosines 0.003–0.011, every ratio within 1.6 null SDs of 1). So "the
pointer head shares more" is dead as stated: whatever couples taught to untaught teams lives in the
trunk, in both eras, and a score-function kernel at one parameter point is the wrong instrument for a
KL loss applied along a fold. The follow-up (`fold_displacement/`) measures the actual fold
displacement per parameter group and its first-order projection onto untaught states.

**Content-locality result (2026-09-05, `arch_transfer_2026-09-05/content_locality/`): REFUTED, sign
REVERSED.** v8's three teachers are LOCAL: sibling-control locality R = 1.45 [1.27, 1.67], each teacher
diverging from v8's parent about 45% more on its own taught teams than its sibling teachers do on the
same states. Our gen-era teachers are essentially GLOBAL (unfunded 1.07 [0.98, 1.16], funded
1.10 [1.00, 1.20]); the cross-era difference is SIGNIFICANT. Within the gen era, locality does NOT
separate the funded (robbing) teachers from the unfunded (neutral) ones; what separates them is
magnitude: funded teachers sit farther from the parent *everywhere*, taught +0.13 and untaught +0.10,
both SIGNIFICANT. So "closed-form physics leaves teachers only team-specific content" is dead as
stated, and the surviving reading is the opposite of the pre-registration: the teachers that gifted
differed from their parent mostly where they specialised, and the teachers that robbed differed
everywhere. If teacher divergence predicts robbery it is *how far*, not *where*. A matched-noise
floor (two adjacent parent checkpoints) was what made this readable: raw taught/untaught KL ratios
read the team sets (gen floor 1.07–1.16, v8 floor 0.69–0.81), so the headline is the sibling control,
same team, same states, own teacher versus siblings. Consequence for the v8 line: v8's teachers
carried little off-slice content for the distillation term to inject, which raises the prior that
v8's early gift came through ecology or plain continued learning (phase-2 cells 1 and 2) or through
generalisation of on-slice content.

**Correction to the locality result (teacher-distance probe, same day):** the locality probe loaded each
teacher's `final_model` file, but a fold resolves a teacher through the opponent-pool resolver whose first
rung is `best_model/best_model.zip`, and every teacher run differs between the two. Re-measured on the
files a fold actually uses, both conclusions *strengthen* (funded minus unfunded off-slice +0.098 → +0.142;
v8's teachers 5.2× → 4.5× their floor), but every level and ratio the locality artifact prints is on
networks no fold used. **Teacher-distance result** (`arch_transfer_2026-09-05/teacher_distance/`): across
seventeen folds that collapse to five distinct teacher sets, mean off-slice teacher distance orders the
sets' untaught outcomes (Spearman −0.90, CI excluding zero) but the slope spans zero and two folds at the
identical distance differ by 8 points, so distance says something about a *set* and nothing about a *fold*.
The ordering is confounded with teacher training budget (the two are as correlated with the outcome as
each other), and half of a gen-era teacher's distance is inherited from the fork origin before it trains a
step. In floor units, the one currency both eras share, v8's teachers sat at 4.5× and gifted; our farthest
sat at 15× and robbed hardest.

**Locality, corrected (`arch_transfer_2026-09-05/content_locality_v2/`, the fold's own checkpoint resolver and
two references).** On the files a fold actually loads, v8's three teachers have sibling-control locality
1.83 [1.53, 2.17], and our sixteen sit at 1.07 [0.94, 1.20] (unfunded) and 1.11 [1.00, 1.21] (funded) against
the fold parent. Measured against their *true* fork origin instead, ours rise only to 1.25 [1.03, 1.47] and
1.20 [1.10, 1.30]: the fork-origin offset explains about a quarter of the unfunded gap and an eighth of the
funded one, and v8's number rose too under the corrected files, so the gap does not close. Gen-era
exploiters are genuinely more global than v8's even from their own origin, which independently reproduces
the exploiter-drift probe's flat on/off ratio of about 1.25 on different states and a different statistic.
Within the gen era, locality still does not separate the robbing teachers from the neutral ones. One caveat
the reader must carry: the v8 headline rests on the thinnest cell, a three-team teacher whose off-slice
divergence sits 1.6× above the floor, so the direction is robust and the magnitude less so.

**FiLM result (z-swap probe, `arch_transfer_2026-09-05/zswap/`): REFUTED.** v8's whole line carried a
32-dimensional team code: a DeepSets mean over our six mons' static facts, entering the network at exactly
one line as a FiLM on both heads after projection and never touching the trunk. Substituting the parent's
code into a v8 teacher removes 3.8% [2.1, 5.5] of the teacher's on-slice divergence against a 20% rail, and
the locality ratio does not move at all (−0.001 [−0.038, +0.032]). The mechanism was engaged, not idle: FiLM
is a co-equal term in the heads and the exploiters grew it, yet it did not localise anything. Deleting all
team-conditional modulation accounts for about a fifth of the excess locality, through the generator
weights rather than the code; the rest is shared trunk. So "add a per-team code to the gen era" should not
be funded on the locality result. Same probe, on the resolved teacher files: v8's sibling-control locality
is 1.83 [1.53, 2.18], stronger than the original 1.45.

**Fold-displacement result (2026-09-05, `arch_transfer_2026-09-05/fold_displacement/`).** Three things
survive. (1) **Displacement magnitude is set by dose, not by the distillation term**: the loss-off control
C1 moved the weights as far as the fold B2 did (|Δθ| 7.99 vs 8.08), and every dose-frozen arm grew as
|Δθ| ∝ t^0.48, so a fold's displacement is substantially a random walk (two arms differing only in seed
share cosine 0.56). (2) **The off-slice divergence is carried by the trunk**: encoders plus team
transformer account for 51–74% of the first-order off-slice KL (about 90% with the projections), the
pointer head 0.5–5.6%, the critic exactly zero. (3) **Recomputed KL(parent‖arm) reproduces the offline
collateral-KL ordering exactly** (Spearman 1.0, p 1/120) on different pilots and seeds, an independent
validation of that artifact; the first-order surrogate over-predicts by 1.2–1.8× and worsens with depth.
One correction to the kernel probe's reasoning: under Adam, gradient-norm share does not predict
displacement (the pointer head moves *most* per parameter, the encoders *least*, a factor of 7 the
opposite way from their norm shares), so "the trunk is where the norm is" was not a valid step even
though the conclusion holds on the KL decomposition. And a null that matters: funded-minus-unfunded
off-slice KL is +0.038 [−0.007, +0.093], the same size as seed noise, so the robbing arms are NOT farther
from the parent off-slice than the neutral ones. A scalar divergence is a norm, not a direction, and
this is the measurement of it.

**Exploiter-drift result (2026-09-05, `arch_transfer_2026-09-05/exploiter_drift/`), and the fact that
reframes the locality finding.** Our sixteen teachers were not forked from the fold parent. All eight R5F
exploiters fork from the rev-1 generalist at 25.07M steps, and the fold parent (R2ACTION) is itself a fold of
that same checkpoint at the same step, so parent and teachers are *siblings*: two 3M-step walks from one
origin, already 0.53 nats apart everywhere before any exploiter training began. v8's exploiters, by contrast,
were forked from v8's parent itself. That offset is exactly the "global" divergence the locality probe
measured, and it inflates the sibling-control denominator toward 1. Three measured facts sit on top of it:
(1) an exploiter's off-slice divergence from its origin grows as t^0.80 [0.76, 0.86], directed rather than
diffusive, while its parameters walk as t^0.55; (2) the on/off ratio is flat from the first checkpoint
(1.22 at 150k steps, 1.26 at 5M), so an exploiter is global from the start, not because it trained past
specialisation; the "trains past specialisation" mechanism is dead; (3) the exchange rate is about 0.02 nats
of off-slice drift per point of on-slice win-rate gain, flat across the budget. The one non-flat thing is the
distance to the *parent*: flat for the first 1.2M steps of exploiter training (the displacement is transverse
to the parent), then rising into the bracket where folds rob. Two cheap causal tests follow: a fold whose
teachers are the same exploiters stopped at 1.2M (checkpoints exist), and a fold whose teachers are forked
from the parent itself, which is v8's recipe.

What is *not* testable: swapping the head inside the era. Phase 2 of the v8 replication can swap
teachers, ecology and hyperparameters inside the era code, but the architecture boundary cannot be
crossed by a fork, so the head hypothesis lives or dies on the two probes above.

## 4. Where this lives in our architecture and record

- The fold loss and its instruments: `src/agents/training/instrumented_ppo/` (`distill_anchor.py`
  for the off-slice trust region and collateral meters, `distill_grad_project.py` for the
  source-separated sibling, `distill_stop_callback.py` for the plateau-and-rise stop rule).
- The flags: `--distill-teacher`, `--distill-coef`, `--distill-anchor-target-kl`,
  `--distill-anchor-mode grad_project`, `--fork-lr` / `--fork-lr-freeze` (the dose pin). Read a run's
  dose with `python -m main.dose <run>`.
- The offline off-slice displacement column and its reproducibility rule (concurrency=1, five seeds):
  `designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/`.
- The 2×2 teacher-content batch (funded vs unfunded, two frozen replicate pairs):
  `designs/research_state/measurements/teacher_content_2x2_2026-09-04/` and the ledger entries of
  2026-09-04/05.
- The v8 gift curve: `designs/research_state/measurements/v8_gift_timing_2026-09-01.json`; the era
  meter `v8_gift_timing_probe.py`; the phase-1 replication pre-registration in the ledger
  (commit 006a2886).
- The architecture boundary: `designs/CHANGELOG.md` v50 (prefuse, refine-loop deletion) and v51
  (pointer-native head); `designs/ARCHITECTURE.md` §2–§3 for the current chain and head inputs.
- Floors and vocabulary (SIGNIFICANT / WITHIN FLOOR / NOT DETECTED; every floor is OPERATIONAL and
  includes self-play pool divergence): the ledger entries of 2026-09-04 and the memory
  `project_untaught_meter_axes`.

## Synthesis

A distillation fold does one thing: it displaces a function every team shares. Whether a given team
experiences that as a gift or a leak is decided by the direction of the displacement, which is the
teacher's content, and not by its length, which is the dose. The gen-era campaign measured exactly
this pattern: content moves the untaught side, dose does not, a scalar divergence cannot tell the
two apart, and the loss-off control moves nothing. Every gen-era fold digs an early hole; near-parent
teachers let RL fill it back in and funded teachers do not. v8 is the anomaly because its
displacement pointed the right way from the first update, and the architecture is a live suspect
for why ours do not: the pointer head shares more, which amplifies transfer without choosing its
sign, and closed-form physics may leave teachers with only the team-specific content that leaks.
Two pre-registered offline probes decide whether that story has data behind it.

## See also

- [[distillation_flywheel_lessons]] — the campaign this note abstracts from: the fold as GIFT − LEAK,
  the dose cell, the v8 hump, the replication plan.
- [[continual_learning_and_forgetting]] — the same mechanism in the stability/plasticity vocabulary,
  and the interference-vs-capacity-vs-drift diagnostic.
- [[entity_tokens_biases_pointers]] — the pointer head, edge biases and weight sharing this note
  says amplify transfer; its Part 4 addendum on depth and looped transformers.
- [[shortcut_learning_and_feature_delivery]] — why computed physics reaches the head as cells and
  edges, the "supplied vs learned" split §3 leans on.
- [[on_policy_self_distillation]] — the search-as-teacher variant of the same loss, where the
  teacher's direction is the policy's own improvement.
- Memories: `project_v8_gift_is_a_transient_hump`, `project_fold_transfer_is_local`,
  `project_negative_transfer_verdict`, `project_teacher_fleet_geometry`,
  `project_arch_transfer_validation`.
