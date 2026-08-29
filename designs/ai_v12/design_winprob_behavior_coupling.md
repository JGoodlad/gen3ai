# design — WIN-PROB → BEHAVIOR COUPLING: three routes from a barometer to a coach

> **[STATE 2026-08-29]** Opened as the first ai_v12 document. **Route 1 is BUILT and OFF
> (`--win-prob-pbrs-coef 0.0` = byte-identical). Routes 2+3 are BUILT and OFF
> (`--search-teacher-mode crater` = the existing behavior, byte-identical).** No arm has been
> run; **nothing here is sanctioned to run** until its era registers arms with pre-registered
> predictions. §6's experiment ladder is the registrable form; it is not a registration.
>
> Era: **ai_v12**. `ai_v11` is reserved for the human-replay chapter and is untouched by this.
> Everything built here is a **train-loop knob** — never version-locked, never consulted by
> `check_compatible`, recorded for provenance in the `td_aux_coef` class.
>
> Provenance for the framing: ledger `b070d6e` (representation-vs-reward shaping unconflated;
> the barometer/coach distinction), `1984dc7` (the three-route taxonomy), `85aadd4` (probe L
> registration), `6b43618` (this build's dispatch).

---

## 1. The problem: the head is a BAROMETER, not a COACH

`WinProbHead` (`src/agents/model/aux_value_heads.py`) reads `value_pooled` and emits one logit;
`sigmoid` of it is P(win | state). It is supervised by the Monte-Carlo episode outcome
(win = 1 / loss = 0) propagated undiscounted to every step of the episode
(`WinProbLabelCallback`, `src/agents/training/win_prob_callback.py`). The production base runs
`--win-prob-mode shaping` at `--win-prob-coef 0.05`.

**That live "shaping" mode carries ZERO behavioral force, and the word has misled readers.** It
is *representation* shaping: the BCE gradient is allowed to reach the shared trunk (rather than
being stop-gradded as under `read_only`), so outcome-predictive features get a subsidy in the
trunk. There is no gradient path anywhere from *predicting wins* to *choosing winning actions*.
The head is a SIDE readout by construction — its logit is stashed at
`features_extractor.last_win_prob_logits` and is never concatenated into `pi` or `vf`, precisely
so the privileged future-outcome label cannot leak into the acting path. The policy is free to
ignore the subsidised features entirely, and V compresses to its own target regardless (the
measured ~7× critic-vs-policy rank gap is that steady state).

There is a second, deeper reason the head cannot coach on its own, and it survives any dose:

> **Its labels are self-referential.** They are outcomes *under the current policy*. A habitual
> whiff that still wins 55% of the time teaches the head "55%" — never "the whiff was the
> mistake". Action-level badness requires a **counterfactual contrast** that a state-level
> outcome label structurally lacks.

This is the whole reason the standing puzzle — "win-prob shaping has been live for generations
and the bait loops persist" — was never a dose mystery. The live mode was never pointed at
behavior.

**What manufactures the missing contrast is the ONE-PLY SUCCESSOR READ**: evaluate each legal
action's successor state and read φ there. That is what search already does at inference
(`src/main/search_dividend/`), and it is the raw material every route below consumes in a
different way.

---

## 2. The three routes

| | route 1 — PBRS | route 2 — ranking distillation | route 3 — confirmed overrules |
|---|---|---|---|
| **level** | reward | target | inference |
| **edits** | the RETURN stream | the POLICY DISTRIBUTION | the ACTION PLAYED |
| **works through** | the RL credit machinery (GAE → advantage → policy gradient) | supervised CE at chosen states, bypassing credit entirely | nothing — it acts, it does not train |
| **status** | **UNSANCTIONED** — needs its own registration | bait-sanctioned *shape* (distillation-shaped levers are the only family that fit the bait verdict) | **VALIDATED mechanism**, measured net-zero dividend as an inference lever |
| **built** | `--win-prob-pbrs-coef` | `--search-teacher-mode winprob_oneply` | the confirm filter inside that mode |

"One's value, one's policy." The three asymmetries below are the reason to build all three
rather than pick one.

### 2.1 SUPPRESS vs PRESCRIBE — complementary blindness

PBRS can **punish a whiff without knowing the alternative**. The potential drop φ(s′) − φ(s) is
a scalar attached to the transition; the policy-gradient step suppresses the action that
produced it, and softmax renormalization redistributes the freed mass across the remaining
legal actions *according to the policy's own current preferences*. Nothing has to know what the
right move was.

Distillation can **prescribe the alternative without carrying why**. A cross-entropy target on
action A\* moves probability onto A\* directly. Nothing has to know what was wrong with the
action played, or that anything was wrong at all.

These are not two spellings of one lever. They fail differently: PBRS is useless when *every*
legal action is bad (suppressing all of them changes nothing after renormalization), and
distillation is useless when the identified A\* is wrong (it teaches the wrong point-decision
with full confidence). A whiff-suppression program that only has one of them has a hole shaped
exactly like the other.

### 2.2 GENERALIZATION — a shape vs a point

PBRS teaches the lesson **through the value pathway**. The shaping term enters the advantage,
which enters the policy gradient, but it also enters the *return target the critic is fit to* —
so the lesson is absorbed into a function approximator over states and transfers to whiffs
never seen. The lesson learned is "transitions that drop my win probability are expensive",
which is a shape.

Distillation teaches **point-decisions**. Generalization is whatever the network's interpolation
gives you between the taught states and the untaught ones — real, but incidental and unbounded
in neither direction. This is the same profile the exploiter→generalist flywheel measures every
revolution (per-team gain × coverage fraction × retention), and it is why route 2 alone should
never be expected to move a general meter.

### 2.3 RISK — one route has a shield and the other does not

**Route 1 is protected by the potential-based-shaping invariance theorem** (Ng, Harada &
Russell 1999). For any potential function φ over states, replacing r(s, a, s′) with
r + γφ(s′) − φ(s) leaves the **optimal policy set unchanged**: the shaping telescopes over any
trajectory to γ^T φ(s_T) − φ(s_0), a term that depends only on the endpoints, so it adds the
same constant to every policy's return from a given start state. A **miscalibrated φ therefore
costs learning SPEED, not CORRECTNESS.** That is an unusually strong shield for a research
lever, and it is most of the reason route 1 is worth building despite being unsanctioned.

**Route 2 has no shield at all.** A distillation target is a supervised label; if it is wrong,
the policy is trained to be wrong, and there is no theorem that rescues it. Worse, the specific
bias it would import is *known and measured* — see §3.

### ⚠️ 2.4 THE LEARNED-DRIFTING-φ CAVEAT (required section, not a footnote)

**The invariance theorem assumes φ is a FIXED function of state. Ours is not.** φ = σ(win-prob
logit) is a *learned head inside the network being trained*, and it moves every optimizer step.
Three consequences, none of which are fatal but all of which must be named in any arm design:

1. **Exact invariance degrades to approximate invariance.** The telescoping argument holds
   exactly only within a window where φ is constant. Our φ is constant *within a rollout*
   (PPO freezes the policy during collection, and route 1's φ is read once per rollout with the
   collection-time weights — see §4.2), so the shaping is exactly telescoping **per rollout**.
   Across rollouts, φ drifts, and the constant added to each start state's return changes. The
   theorem's guarantee therefore applies to each rollout's contribution but not to the
   trajectory of the whole optimization.

2. **The drift is not adversarial, and its direction is known.** φ is trained toward the true
   outcome probability under the current policy. As the policy improves, φ tracks it. This is
   the "virtuous loop" reading (better head → better shaping → better policy → better outcome
   data), and it is also the failure mode's mechanism: φ is a moving target, and a moving
   potential can in principle inject a persistent bias that the fixed-φ theorem excludes.
   **The bound is: any bias is at most the drift in φ over the horizon of one credit-assignment
   window.** Early in a run — φ untrained, drifting fast — the shaping is closest to a
   free-form reward hack. Late in a run — φ converged — it is closest to the theorem.

3. **The operational consequence: this lever wants to be turned on LATE, or annealed.** An arm
   that enables PBRS from step 0 of a fresh run is testing the worst case for the shield. The
   arms in §6 therefore start from a mature base. The **scaffolding gauge** registered in
   `596608e` (the divergence between the V-implied outcome and the win-prob head across states)
   is the natural instrument for "has φ settled enough for this" and, later, for annealing the
   coefficient toward the pure game.

**Additionally: our φ is READ, not re-derived.** It is the head's own detached output, so it
inherits the head's known defect (the G0 bias map: the scalar head's problem is **RESOLUTION,
not offset** — population-mean gaps of 0.05–0.07 against a true within-decile spread of
0.11–0.36). A low-resolution φ is a *blurry* potential: it does not point the wrong way on
average, it fails to distinguish states it should. Under the theorem that is exactly the
harmless failure — a φ that is constant over a set of states contributes nothing over that set
and cannot mislead within it. **A blurry potential is a weak potential, not a wrong one.** This
is the single most reassuring fact about route 1, and it is a direct consequence of the
resolution-not-offset diagnosis.

### ⚠️ 2.5 THE WINNER'S-CURSE DISCIPLINE (a REQUIREMENT on route 2, not a caveat)

Route 2's targets come from ranking successor states by φ. The instrument doing the ranking is
**biased, and its bias is differential** — exactly the quantity that matters here. This is not
a hypothetical: it is the mechanism that produced the defensive-search iter-2 result.

The evidence, in order:

- **Defensive search iter 1** (`defensive_search_first_cell_2026-08-29.md`): 400 side-swapped
  paired mirror games. Win 0.4937 [0.4448, 0.5427] vs honest_1s 0.2929 — **search stopped
  losing**, at 1.8% overrule rate, budget-limited at the elimination floor.
- **Defensive search iter 2** (`defensive_search_iter2_2026-08-29.md`): 1600 games / 800 pairs,
  the allocator un-throttled exactly as specified. Separated-of-raced 0.157 → 0.4542 (95% of
  probe I's ceiling), **overrules 1.8% → 5.82% (13× more evidence-certified overrules)**, and
  the win rate landed on **0.5003 [0.4803, 0.5203] — the point estimate IS the null.**
- **The mechanism**: CRN pairing removes dice noise *and the shared offset*, so what the racing
  procedure statistically CERTIFIES is the leaf's residual **differential** bias
  (RMS 0.122 — larger than most true gaps) as much as it is signal. **Statistical separation of
  a biased reader is not correctness.**

Route 2 imports that reader wholesale. Therefore:

> **REQUIREMENT.** A `winprob_oneply` target is admissible only if the preferred action's margin
> clears the **measured** differential noise floor AND the preference survives **paired-rollout
> confirmation** — rollouts to a terminal, which contain the opponent response that the one-ply
> leaf structurally lacks. An arm that distils un-confirmed one-ply preferences is not a cheaper
> version of this program; it is the iter-2 failure with a gradient attached.

The counter-evidence that keeps route 2 alive rather than killing it: **probe K** re-judged
iter-2's 3,531 recorded overrules under opponent-marginalized ground truth and found
**+0.0474 [+0.0216, +0.0730] per-decision gain — REAL**. The overrules were *right*; the
per-decision → per-episode transfer is what failed (+4.7pp × ~2.2 overrules/game bought
+0.0003). That is a strong argument for route 2 specifically: **a per-decision gain that does
not transfer through play might still transfer through TRAINING**, because a training target
changes the policy everywhere the network generalizes, not only at the 2.2 decisions per game
where the searcher happened to intervene. Route 2 is the natural response to probe K's finding,
not a repeat of iter 2.

### 2.6 Route 3 is not defense-in-depth — it is route 2's SUPPLY

The amendment banked in `1984dc7`: **confirmed overrules are route 2's highest-quality training
targets.** This is the AlphaZero loop in miniature — search manufactures the curriculum, the
network absorbs it, and the improved network makes the next search better. The distinction from
AlphaZero is that our search is *sparse and expensive*, so the curriculum is a trickle of
high-confidence corrections rather than a full-game visit-count distribution. That is precisely
what the existing ExIt/AWR seam is shaped for (`Correction` records in a ring buffer, sampled
into an advantage-weighted CE), which is why routes 2 and 3 are built as **one mode on that
seam** rather than as a new pipeline.

---

## 3. The pipeline of record: 3 filters → 2 transplants → 1 repairs credit

```
   decision states (sampled from fresh trainee-vs-opponent battles)
        │
        │  ROUTE 3 — the FILTERS
        ├── contested gate:  n_legal ≥ 2  AND  |P(win|s) − 0.5| < band        (the H rule)
        ├── one-ply read:    φ(s′_a) for every legal a  (reconstruction + reroll, impl=rust)
        ├── margin gate:     φ(s′_{A*}) − φ(s′_{played}) ≥ noise floor
        └── confirmation:    N paired rollouts to a terminal, CRN-anchored;
                             A* must still win  ⇒  a CONFIRMED OVERRULE
        │
        │  ROUTE 2 — the TRANSPLANT
        ├── each confirmed overrule becomes a `Correction`
        │      (obs, action_mask, better_action=A*, advantage=confirmed Δwin, ...)
        └── the existing AWR CE folds it into the policy at --search-teacher-coef
        │
        │  ROUTE 1 — the CREDIT REPAIR  (independent; no shared plumbing)
        └── every transition in every rollout gets  coef·(γ·φ(s′) − φ(s))  added to its reward
               before GAE, so the RL machinery itself learns the shape of the lesson
```

Route 1 is deliberately NOT downstream of routes 2/3. It is dense (every transition, every
rollout, free) where routes 2/3 are sparse and expensive (a few thousand confirmed corrections
per run, each costing N rollouts). They are complements in cost profile as well as in mechanism.

---

## 4. What was built

### 4.1 Flag surface

**Route 1 — PBRS reward shaping**

| flag | type | default | class |
|---|---|---|---|
| `--win-prob-pbrs-coef` | float ≥ 0 | `0.0` (OFF, byte-identical) | train-loop knob, `td_aux_coef` class: recorded for provenance + flagless-resume read-back, never version-locked |

Requires `--win-prob-mode read_only|shaping` (there is no head to read under `none`) — a
config-time `parser.error`, not a runtime crash. γ is **the run's own `--gamma`**; there is no
separate PBRS discount, because a shaping discount that differs from the return discount breaks
the telescoping identity that is the entire shield.

TensorBoard: `train/pbrs_shaping_mean`, `train/pbrs_shaping_absmean`, `train/pbrs_phi_mean`,
`train/pbrs_reward_share` — enough for a live run to show the term's magnitude *relative to the
reward stream it is perturbing*, which is the number that decides whether a coefficient is
sane.

**Routes 2+3 — the win-prob teacher mode**

| flag | type | default | class |
|---|---|---|---|
| `--search-teacher-mode` | `crater` \| `winprob_oneply` | `crater` (existing behavior) | operational; re-pass on resume like `--search-teacher` |
| `--winprob-teacher-band` | float | `0.15` | operational — the contested gate half-width |
| `--winprob-teacher-margin` | float | `0.02` | operational — the one-ply Δφ noise floor |
| `--teacher-confirm-rollouts` | int | `8` (**existing flag, reused**) | operational — route 3's paired-rollout confirmation count; `0` disables confirmation (route 2 alone, the undisciplined control arm of E2) |

**Naming note.** The dispatch named a `--search-teacher-confirm`; the seam already owns
`--teacher-confirm-rollouts` with the same meaning, already threaded into
`SearchTeacherCallback(confirm_rollouts=…)`. Adding a second spelling for one number is how a
flag surface rots, so the existing one is reused. The `--winprob-teacher-*` prefix marks the two
knobs that exist **only** in the new mode.

### 4.2 Route 1 implementation — where the shaping actually happens, and why there

**The constraint that determines the design: env workers have no model.** `Gen3Env` runs in a
`SubprocVecEnv` worker with no policy weights, so φ cannot be computed where rewards are
produced. The shaping is therefore **trainer-side buffer augmentation, applied after collection
and before GAE.**

The insertion point is `InstrumentedMaskablePPO.collect_rollouts`
(`src/agents/training/instrumented_ppo/ppo.py`), which already exists as an override to
dispatch between the stock and async collectors. Both collectors finish by computing GAE
(`rollout_buffer.compute_returns_and_advantage(last_values, dones)`) and both leave
`model._last_obs` = s_T and `model._last_episode_starts` = the terminal dones. So the override
can, when the coefficient is non-zero:

1. **Read φ for the whole buffer in one batched, `no_grad` forward** over
   `rollout_buffer.observations` — chunked, on the training device, using the *collection-time*
   weights (`train()` has not run yet). This is transport-agnostic: **it is identical on the
   sync and async paths**, which is why it was preferred over a per-step callback capture.
2. **Read φ(s_T)** from `model._last_obs` in the same style (one extra small forward).
3. **Build φ_next** with the terminal convention: for row *t*, φ_next = φ at row *t+1* unless
   row *t+1* starts a new episode (`episode_starts[t+1] == 1`), in which case row *t*
   terminated its episode and **φ_next := 0**. For the final row, φ_next = φ(s_T) where the
   episode continued and 0 where it ended.
4. **Add** `coef · (γ·φ_next − φ)` to `rollout_buffer.rewards` in place.
5. **Re-run** `compute_returns_and_advantage(last_values, dones)` with the same arguments the
   collector used, so returns and advantages reflect the shaped stream.

> **Why the batched re-forward rather than reusing the collection-time stashes.** The stock
> collector's per-step `last_win_prob_logits` stash *is* available in a callback's `_on_step`,
> and that is how `WinProbLabelCallback` captures terminals. But the async collector forwards a
> *wave* of envs at a time and its callback locals cannot recover the env→row mapping (the same
> reason the win-target capture had to be inlined into the async collector). One batched
> re-forward gives both paths the identical, obviously-correct quantity for a cost of roughly
> 1/`n_epochs` of one training pass. **The async-rollout case is COVERED, not documented around.**

**φ is detached, structurally.** The forward runs under `torch.no_grad()` and the result is
converted to `numpy` before it touches the buffer. The buffer's `rewards` array is numpy; there
is no tensor, no graph, and no possible gradient path from the policy loss back through the
potential. A dedicated test asserts this by constructing the shaping through the live code path
and failing if any produced quantity carries `requires_grad`.

**Interactions, verified:**

- **PopArt** reads `self.rollout_buffer.returns` at the top of `train()`, which is *after*
  `collect_rollouts` returns. The shaping therefore lands in **raw reward space** and PopArt
  normalizes the shaped returns — the correct order, and the only one that keeps the value loss
  in the units of the stream being optimized (the foundations ruling, `596608e` §1).
- **GAE** is recomputed, not patched. Advantages, returns and the `signal/adv_*` diagnostics all
  see the shaped stream consistently.
- **`--grad-accum-steps`** is a `train()`-loop knob and is untouched: the buffer it reads is
  simply the shaped one.
- **Timeout bootstrap** (`TimeLimit.truncated`) is orthogonal — SB3 folds the bootstrap value
  into the *reward* before the buffer sees it, and the shaping is added on top. §4.3 states the
  convention this implies.

### 4.3 The terminal and truncation conventions

**Terminal (a real episode end — win/loss/draw).** φ(s_terminal) := 0 by convention, so the
final transition of an episode carries shaping `−coef·φ(s_{T−1})`. This is the standard
episodic-PBRS convention and it is what makes the per-episode sum telescope to exactly
`−coef·φ(s_0)` (γ-weighted): a constant per start state, which is the invariance theorem's
whole content. **The unit test asserts this identity numerically on a synthetic buffer.**

**Truncation (a time limit / buffer boundary, not a real end).** These are *not* the same case
and conflating them is the classic PBRS bug: a truncated trajectory whose φ is forced to 0 gets
a large spurious negative reward for the crime of the rollout ending. Two sub-cases:

- **Buffer-boundary truncation** (the episode is still running when the rollout ends): handled
  by the bootstrap — the last row's φ_next is φ(s_T) from `model._last_obs`, *not* 0. The
  episode's shaping simply continues into the next rollout. Correct and free.
- **`TimeLimit.truncated`** (the env's own 250-turn deadline): SB3 marks these `done=True`, so
  the buffer sees an episode boundary and the convention above assigns φ_next = 0. **This is a
  known, bounded approximation and it is stated rather than hidden.** In this project a
  250-turn timeout is a *real* game outcome (the forfeit deadline; `StallConfig.threshold`
  imports `MAX_TURNS`), scored as a loss/draw by the reward manager — so φ_next = 0 is
  arguably the *correct* reading here rather than an approximation at all. The test pins the
  behavior either way, so a future change to the timeout's semantics fails loudly.

### 4.4 Routes 2+3 implementation — one mode on the existing seam

The dispatch's instruction was explicit: *if the existing seam's generation runs in workers or
offline processes, follow its architecture — do NOT invent a new pipeline.* It does, and this
does.

**What was reused, unchanged:**

- `agents/training/teacher/buffer.py` — the `Correction` record and the `CorrectionBuffer` ring.
  A `winprob_oneply` target is a `Correction` with the same fields and the same meaning
  (`better_action` = A\*, `advantage` = the confirmed win-probability gain), so **the consumer
  did not change at all**.
- `instrumented_ppo/distill_terms.py::_searchteacher_loss` — the advantage-weighted masked CE.
  Unchanged; it never knew where its corrections came from.
- `instrumented_ppo/ppo.py`'s sampling block and `--search-teacher-coef` /
  `--search-teacher-beta` / `--search-teacher-batch-size`. Unchanged.
- `agents/training/teacher/callback.py::SearchTeacherCallback` — the worker pool, the persistent
  mode, the re-freeze cadence, the crash/respawn logic, the shard ingest. Unchanged except for
  threading the mode through to the worker.
- `agents/training/teacher/generate.py` — fresh loss-trace generation for the persistent worker.
  Unchanged.

**What is new** (`agents/training/teacher/winprob_oneply.py`): the *selection and production*
half only — the contested gate, the one-ply φ ranking over legal actions, the margin gate, and
the assembly of a `Correction` from a confirmed preference. It sits beside `selection.py` /
`produce.py` as a peer, and the mode string chooses between them.

**What was REUSED from `src/main/search_dividend/`, and what was not.** The defensive-search
machinery is the conceptual source of the confirm discipline, and §2.5 is built on its
measurements. Its *pure* half is genuinely importable and is imported:

- **`defensive.gate(n_legal, win_prob, cfg)` and `DefensiveConfig` are used directly** for the
  contested gate. This is the H rule as shipped and measured (`wp_margin` default 0.15,
  `FORCED_N_LEGAL = 1`), and re-typing the same two clauses into the teacher would have created
  two definitions of "contested" that could drift apart while both looked right. The teacher's
  `--winprob-teacher-band` is exactly `DefensiveConfig.wp_margin`.
- **`defensive.verdict(...)` / `resolve_action(...)` are NOT used**: they answer "which action
  do I PLAY", and the teacher answers "is this a target". Same inputs, different question.
- **`racing.Racer` and the budget/deadline machinery are NOT used.** They are the *allocator* —
  they exist to spend a per-decision wall-clock budget inside a battle in flight, racing arms
  with CRN dice until one separates. The teacher works offline from a recorded reconstruction in
  a worker with no clock to race against, and its confirmation is a fixed-N paired rollout, not
  an anytime elimination. Importing the racer here would drag a live-battle dependency into a
  path that has none, to get a stopping rule the path cannot use.
- **The paired-rollout confirmation is expressed through `ProbeSession.replay_counterfactual`**,
  the same offline primitive the existing `produce_correction` reaches for its Wilson gate — not
  through `playoff.PlayoffRunner`, which needs a live `SearchEngine` and a shared `Deadline`.

The net duplication is the paired-margin arithmetic (a handful of lines), and it is recorded
here so the next reader does not "fix" it into an import that would carry a live-battle
dependency with it.

---

## 5. What is NOT built here (deliberately)

- **No annealing schedule for the PBRS coefficient.** §2.4 argues the lever wants to be annealed
  as φ matures, and the scaffolding gauge is the natural trigger — but an unmeasured schedule is
  a second free parameter on an unsanctioned lever. E1 runs a flat ladder first.
- **No epistemic-uncertainty gate.** The empowerment program's item (2) — checkpoint-disagreement
  spread as the racer's uncertainty input — would be the principled replacement for a fixed
  `--winprob-teacher-margin`. It is registered there, not here.
- **No contrastive re-training of the head.** The empowerment program's item (1) (marginalized,
  sibling-differenced labels from the R1 v2 factory) attacks the *reader's* bias directly and is
  the deeper fix for §2.5. It is a separate build.
- **No `--win-prob-pbrs-coef` on the eval/inference path.** Shaping is a training-time reward
  edit; nothing about eval changes.

---

## 6. Experiments — the registrable ladder

**These are ladder sketches, not registrations.** Each states its endpoints, its predictions and
its kill condition at the level where those are decidable; the exact argv, base checkpoint, n
and seed are set at registration time by the era that runs them. Nothing below is authorized to
consume a generation slot.

**Standing endpoints for every arm.**

- **The whiff/loop census** — `python -m main.prober.query loops`, which reads whiff / re-click /
  loop rates off the RAW protocol (never the rendered timeline) against the gen-15 baseline it
  carries. This is the *behavioral* endpoint: it is the thing all three routes exist to move.
- **The standard piloting meter** — per-team piloted win rate on the matched-extraction harness,
  paired draws, seniority as a separate term (`feedback_matched_extraction_row` discipline).
- **`grad/` shares and the rank watch** — the crystallization history (value_cls / FitNets) says
  any new gradient into the trunk gets a rank tripwire.
- **`train/pbrs_*`** for route 1 arms, to confirm the term's magnitude matches its coefficient.

### E1 — the PBRS coefficient ladder (route 1 alone)

Three arms on a **fixed mature base**: `--win-prob-pbrs-coef ∈ {0.0, 0.1, 0.3}`, everything else
identical, matched steps, matched team draws.

*Why start from a mature base and not fresh:* §2.4 item 3 — an untrained φ drifts fastest, which
is the worst case for the shield. Starting mature tests the lever, not the shield's boundary.

*Why this coefficient range:* the shaping term is in raw reward units and telescopes to
`−coef·φ(s_0)`, i.e. at coef 0.1 the total per-episode shaping is bounded by 0.1 reward units
against a terminal reward of order 1. 0.3 is the point at which the shaping is a third of the
outcome signal — a deliberate upper arm chosen to be visibly aggressive rather than safe.

**Registered predictions.** (i) Whiff rate falls monotonically in the coefficient — this is the
suppress mechanism's most direct prediction and the one route 1 is *for*. (ii) Piloting is flat
or slightly up; PBRS is advantage-invariant in the fixed-φ limit, so a *large* piloting change
in either direction is evidence about the drift caveat, not about the lever. (iii)
`train/pbrs_reward_share` tracks the coefficient linearly (a sanity check, not a finding).

**Kill conditions.** Whiff rate flat across the whole ladder ⇒ route 1 does not carry force at
any dose we would run, and the lever is dead (not "under-dosed" — 0.3 is already a third of the
outcome signal). Piloting *down* at 0.1 with CIs excluding zero ⇒ the learned-φ drift is
injecting real bias and the shield does not hold in practice; the lever is dead and §2.4 gets
its empirical entry.

### E2 — the winner's-curse demonstration (route 2, disciplined vs not)

Two arms, both `--search-teacher-mode winprob_oneply`, differing in **one flag**:

- **disciplined**: `--teacher-confirm-rollouts 8` (the default; margin gate + paired-rollout
  confirmation)
- **undisciplined control**: `--teacher-confirm-rollouts 0` (margin gate only — statistically
  separated one-ply preferences, distilled without confirmation)

This is the arm that turns §2.5 from an argument into a measurement. It is also the only arm in
this ladder whose *expected* result is that one side is harmful.

**Registered predictions.** The undisciplined arm produces **far more** targets (the confirm
filter is the expensive one — expect a large multiple) and **worse** outcomes: flat-to-negative
piloting, and specifically a *rise* in some whiff family as the head's differential bias is
transplanted into the policy. The disciplined arm produces a trickle and moves piloting up on
the covered slice. The **difference between the arms** is the finding, not either level.

**Kill conditions.** Both arms flat ⇒ route 2's targets carry no force at achievable volume, and
the honest reading is a *volume* problem (probe K's +4.7pp per decision × a few thousand
decisions is a small curriculum) rather than a quality one — which points at the label factory,
not at more confirmation. Disciplined ≈ undisciplined and both positive ⇒ the confirm filter is
buying nothing and the winner's curse does not survive contact with training; §2.5 is
over-stated and the cheap arm is the arm.

### E3 — the combined arm

`--win-prob-pbrs-coef <E1's winner>` **and** `--search-teacher-mode winprob_oneply` at E2's
disciplined settings, against each alone.

*Why this is not just "both on".* §2.1 predicts **complementary blindness**, which is a specific,
falsifiable claim: the combined arm should beat the *sum* of the two individual effects on the
whiff census, because PBRS handles the states where no good alternative exists (which route 2
cannot label) and distillation handles the states where the right move is identifiable but the
value pathway is too diffuse to find it. If the effects merely add, they are two spellings of
one lever and one of them should be retired.

**Registered prediction.** Super-additive on the whiff census; at most additive on piloting
(coverage still bounds route 2's contribution there).

**Kill condition.** Sub-additive — i.e. the combination is *worse* than the better single arm ⇒
the two routes interfere (the most likely mechanism: PBRS suppresses the very action route 2 is
prescribing at a state where they disagree), and the program ships one route, not two.

### E4 — the 3→2 pipeline arm at full strictness

The pipeline of record end to end: confirmed-overrule targets only, with the margin floor raised
to the *measured* differential-bias RMS (0.122 from iter 2) rather than the default 0.02.

*Why it is a separate arm from E2's disciplined side:* E2's margin is a working default; E4 asks
whether the *right* floor is the measured leaf-bias RMS. At that floor the target volume
collapses (0.122 is larger than most true gaps — that is exactly iter-2's finding), so this arm
is a deliberate **quality-over-volume** extreme and its comparison against E2-disciplined is the
volume/quality trade curve.

**Registered prediction.** Fewer targets by roughly an order of magnitude; per-target effect
larger; net effect on piloting *smaller* than E2-disciplined. If that is what happens, the trade
curve says "loosen", and the program's next move is the label factory rather than the filter.

**Kill condition.** E4 ≥ E2-disciplined on piloting ⇒ target *quality* is the binding constraint
and every subsequent arm should tighten rather than loosen; the confirm mechanism becomes the
program's centre of gravity.

### Sequencing

E1 and E2 are independent and can run in either order or concurrently (route 1 shares no
plumbing with routes 2/3). E3 requires both. E4 requires E2. **All four are gated behind an era
that has a generation slot for them; none is registered by this document.**

---

## 7. Probe L — does the head already know?

> **STATUS: PENDING.** Probe L (`85aadd4`) was dispatched concurrently with this build and its
> record — `designs/research_state/measurements/whiff_head_knowledge_2026-08-29.md` — had **not
> landed on `main` as of this document's last check (2026-08-29)**. This section is the
> structure it will fill; the predictions below are quoted from the registration and are **not
> results**.

**The question.** On recorded battles, join the prober's model-free bait-loop/whiff census
(immune-move clicks against pivots, read off the raw protocol) with the model's own reads at
those same decisions. Two sub-questions:

1. Did **α** (the opponent-intent head) predict the pivot?
2. Did the **one-ply win-prob ranking prefer a non-whiff action AT DECISION TIME** — not merely
   confess the probability drop afterwards?

**Registered predictions** (verbatim from `85aadd4`, per the bait verdict where credit is
CONVICTED and punishment is null): *the head KNOWS* — **≥ 60% of immune-whiff decisions have the
win-prob ranking preferring an alternative at decision time with real margin**, and **α flags
the pivot on a majority**.

**The decision rule, and what it does to THIS document.**

| probe L outcome | consequence for ai_v12 |
|---|---|
| **head knows + policy ignores** (the predicted case) | Every route above is live. The sanctioned lever family is **distillation-shaped** — route 2 is directly indicated, and a shaping-dose ladder above 0.05 is licensed. E1/E2/E3 proceed as written. |
| **head does NOT know** | The gap is **obs/coverage** (the incoming-damage under-read family), and *no* amount of coupling helps until the head is fixed. Routes 1–3 are all premature: route 1's φ would be blurry exactly where it matters, and route 2's targets would be noise. **The whole document parks** behind an obs/coverage fix, and the built code stays OFF as pre-positioned machinery. |

**This is the document's single largest external dependency.** The code is built either way —
that was the point of building it ahead of its era — but §6's ladder is only worth a generation
slot in the first row.

*(When probe L lands, replace this section with its measured cross-tab: n decisions, the
head-prefers-alternative fraction with its CI, the margin distribution against the 0.122
differential-bias RMS, and α's pivot-flag rate. Keep the registered predictions above the
results, per the pre-registration discipline.)*

---

## 8. What would kill this document

- **Probe L returns head-doesn't-know** (§7) — parks everything until the obs/coverage fix.
- **E1's kill condition on the whiff census** — route 1 carries no force at any dose we would
  run.
- **E1's piloting-down-at-0.1 condition** — the learned-φ drift injects real bias; §2.4's caveat
  becomes a verdict.
- **E2 both-arms-flat** — route 2's achievable target volume is too small to matter, and the
  program's centre moves to the label factory.
- **E3 sub-additive** — the two routes interfere and only one survives.
- **A head-level fix lands first** (contrastive marginalized labels, empowerment item 1) and
  removes the differential bias — this does not kill the document but it **invalidates §2.5's
  calibration**: the noise floor, `--winprob-teacher-margin`'s default, and E4's whole premise
  are all keyed to a bias that would no longer exist. Re-measure before running E4.

---

## 9. Era framing

**This is ai_v12 material.** `ai_v11` is reserved for the human-replay chapter
(`design_human_replay_objectives.md`, itself punted) and is not touched here. The active
programme at the time of writing is the exploiter–generalist flywheel; this document consumes no
part of it.

Everything built is **OFF by default and byte-identical when off**, and everything is a
**train-loop knob** — no `ARCH_SIGNATURE` bump, no `MODEL_CONFIG_VERSION` bump, no
`check_compatible` participation, recorded for provenance and flagless-resume read-back in the
`td_aux_coef` class. A run that does not type these flags is bit-identical to one built before
them.

**Nothing here runs until its era registers arms.**
