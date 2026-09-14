# Metamon at a MATCHED REGIME — pre-registration, 2026-09-14

**Committed BEFORE any game of this measurement was played.** Nothing below was written with a
number from this series in hand. The only numbers quoted are from the 2026-09-14 de-risk
(`../metamon_derisk_2026-09-14/`), from `designs/training/eval_and_rating.md`, and from counting
files in Metamon's `competitive` team set.

---

## 1. Why this measurement exists

The de-risk read our `ai_v12_02_winprob_critic` final snapshot at **0.742 vs `SmallRL`** and
**0.583 vs `SyntheticRLV2`**, and both numbers were taken in a **MIXED regime**: our side greedy
(`play.py --temperature 0`, our measurement convention), Metamon's side sampling at its own eval
default (`action_temperature = 1.0`), and **both sides drawing from OUR 719-team pool**. Each half
of that asymmetry is a project's own convention, and each is worth something.

We already know what the sampling half is worth *inside our own project*: on the same frozen pair,
a greedy player against a temperature-1.0 opponent reads **+8.9 pp [+7.0, +10.7]** better than the
same pair played greedy-vs-greedy (`designs/training/eval_and_rating.md`; `gen3_eval_sentinel_
greedy_default_v1`; UNDERSTANDING §3.2 rule 5). If that transferred, our de-risk numbers would be
inflated by most of the margin they claim.

So: a pre-registered 2×2 per Metamon model.

## 2. The design

| axis | levels |
|---|---|
| **sampling regime** (BOTH sides at the same setting) | **greedy** — us `--temperature 0`, Metamon `Agent.get_actions(sample=False)` (argmax) · **t1.0** — us `--temperature 1.0`, Metamon `action_temperature=1.0` (its shipped default) |
| **team set** (BOTH sides draw from it) | **home** — our 719-team pool, the de-risk's nickname-free export (`gen3ai_pool`) · **away** — Metamon's own `competitive` gen3ou set, **20 teams** |
| **model** | `SmallRL` (ckpt 40, 13.9M) · `SyntheticRLV2` (ckpt 48, 200.9M) |

**100 games per cell**, 4 cells per model, **800 games total**. Our side is
`models/ai_v12_02_winprob_critic/final_model.zip` (75,005,952 steps) on CPU throughout; the
registry name `production` is STALE and does not load at HEAD, exactly as the de-risk recorded.

**Why `competitive` is the away set.** Metamon's README says of it: *"This is the set used for
human ladder evaluations in the paper"*, and its own cross-generation strength tables are headed
"Competitive TeamSet". It is therefore the set its paper policies were evaluated on for gen3ou. It
is also the set its authors say Metamon **has overfit to**. That is the point of an away game: it
is their ground, chosen by them, with the disadvantage stated up front.

**Role is balanced WITHIN every cell.** Games 1–50 of a cell have our side as the challenger,
games 51–100 have Metamon's side as the challenger. Showdown makes the challenger p1, and p1/p2 is
not a priori neutral (speed ties, ordering). Balancing inside the cell keeps role orthogonal to
both factors and to the primary row, rather than confounding it with a cell.

**Team draws are seeded and logged per game.** A private RNG per side, seeded from
`base_seed = 20260914` as `seed = base_seed + 1000*cell_index + (0 ours / 1 theirs)`; each side's
per-game draw and the file/`team_sha` it resolved to is written into `games.jsonl`. Ours through
`$GEN3AI_TEAM_SEED` (`gen3_team_draw_rng_v1`, the supported seam); Metamon's by giving `TeamSet.
yield_team` a private `random.Random` over its **sorted** file list, because its shipped
implementation draws from the process-global `random`.

**Everything else is held fixed** and is the de-risk's harness unchanged: our own pinned Showdown
submodule (`deps/pokemon-showdown` @ `e0551883f`) on a port in 9300–9399, CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice -n 10`, `--forfeit-turn-limit` at its default 250,
`--concurrency 1`, `VanillaAttention` on the Metamon side (hazard H2), the nickname-free pool
export (hazard H1), the module-global port rebind (H4) and `PYTHONUNBUFFERED=1` (H5).

## 3. How the Metamon regime is SET, and how it is VERIFIED

**Set.** `amago`'s rollout loop calls `policy.get_actions(..., sample=self.sample_actions_val)`, and
`get_actions` with `sample=False` on a discrete policy takes `torch.argmax(action_dists.probs)` —
exact greedy. `sample_actions_val` is an `Experiment` field that Metamon's
`make_placeholder_experiment` does not pass, so we set it on the built experiment. We do **NOT**
try to reach greedy by driving `action_temperature` toward 0: `MetamonDiscrete.forward` computes
`vec / self.temperature` (a zero divides by zero) **and then clips the probabilities to
`[0.001, 0.99]` before renormalising** — so even at a tiny temperature the argmax action is played
only ~99.2% of the time on a 9-way action space. A temperature knob cannot express greedy here.
That clipping is a pre-registered hazard, not a post-hoc excuse.

**Verified, not assumed.** Two instruments, both written before the run:

1. The `sample` keyword actually received by `Agent.get_actions` is recorded on every decision. It
   must be `False` in every greedy cell and `True` in every t1.0 cell.
2. The actor's probability vector is captured on every decision and compared to the action the
   policy emitted. **`argmax_match_rate` must be 1.000 in a greedy cell.** The same instrument in
   the t1.0 cells is the POSITIVE CONTROL: it must read materially below 1.000 there, or the
   instrument has no power and neither reading means anything.

Our own side is verified the same way: the observer records `RLPlayer.stochastic` and its
temperature, and the greedy cells must report `stochastic=False`.

## 4. The pre-registered predictions

Every one of these is a claim that can be wrong, stated with a direction and a bound.

### P1 — the PRIMARY row · `SyntheticRLV2`, greedy-vs-greedy, away teams

**Prediction: the bar is NOT cleared.** Point prediction **0.42–0.52**, and at n = 100 a Wilson
95 % lower bound above 0.50 needs p̂ ≳ 0.60, which we do not expect to see.

Reasoning: the de-risk's 0.583 was bought on two home advantages at once. Strip the sampling
asymmetry (worth up to ~9 pp to the greedy side if it transfers at all) and strip our team-pool
advantage, and the honest centre is near even — which is what a 200M-parameter offline-RL policy
trained on a far larger corpus than ours *should* be.

**The bar, stated once:** *our model is BETTER than `SyntheticRLV2`* iff the **Wilson 95 % lower
bound on this one cell is > 0.50**. Anything else is **NOT DETECTED**, never "equal" — an
equivalence claim would need the DELTA's own CI inside a stated bar, and no such bar is registered
here (UNDERSTANDING §7 rule 6).

### P2 — `SmallRL`, greedy-vs-greedy, away teams

**Prediction: the bar IS cleared.** Point prediction **0.60–0.72**. `SmallRL` is 13.9M parameters
against our 3.06M, but the de-risk's 16 pp ordering between the two Metamon policies is larger than
the ~9 pp the regime can plausibly cost us, so the margin should survive both strips.

### P3 — the TEMPERATURE effect (greedy − t1.0, both sides matched)

**Prediction: SMALL and NOT DETECTED — |Δ| ≤ 0.05 with a CI covering zero.**

This is the prediction the task asks for explicitly, and the honest answer is that **the +8.9 pp
should NOT transfer**. That figure is an **ASYMMETRY term**: what the greedy side gains *because the
opponent is sampling*. Moving BOTH sides together removes the asymmetry from both halves at once,
and what is left is only the second-order difference in how much each policy loses by sampling from
its own distribution rather than committing. There is no reason for those two losses to differ by
much.

**If it is nonzero, we predict the sign favours us in the greedy cells,** by ≤ 0.08. Our policy is
selected, evaluated and reported greedy throughout (`--temperature 0` is "the measurement setting"
in `play.py`'s own help text), whereas Metamon's shipped eval default is T = 1.0 sampling and its
self-play corpus was generated by sampling — so Metamon's policy is the better-matched of the two
to its own stochastic regime, and ours is the better-matched to determinism.

### P4 — the TEAM-SET effect (home − away)

**Prediction: POSITIVE, and the LARGEST of the three effects — +0.05 to +0.20.** Both directions
push the same way: our pool is our training distribution (719 teams the model has played millions of
games on), and `competitive` is a 20-team set its authors state Metamon has overfit to. A team-set
effect that comes out near zero would be a genuine surprise and would say our conditioning on our
own pool is weaker than we think.

### P5 — matched regime vs the de-risk's mixed read

**Prediction: every matched-regime cell reads at or below its de-risk comparator.** Specifically,
the home-greedy cells should sit **0 to 9 pp below** 0.742 (`SmallRL`) and 0.583 (`SyntheticRLV2`) —
the de-risk's own conditions but with Metamon's side no longer sampling. A matched cell reading
*above* the mixed read would falsify the transfer of the +8.9 pp entirely and is worth reporting
loudly.

### P6 — game shape, and the failure mode greedy invites

**Prediction: mean turns in the greedy cells ≥ mean turns in the t1.0 cells, and if the 250-turn
forfeit limit fires anywhere in these 800 games it fires in a greedy cell.** Two deterministic
policies on a small team set can lock into a repeating stall (PP-stall loops, Leftovers recovery
races) that neither will break, and a deterministic policy has no coin to flip its way out. The
de-risk saw 0 firings in 180 mixed-regime games with a 178-turn maximum, so this is not idle: the
away cells draw both sides from 20 teams, so repeated team PAIRS are common and a locked line
repeats with them.

Otherwise: **0 protocol-parse failures, 0 timeouts, 0 tracebacks, 0 illegal-action fallbacks on our
side**, as the de-risk measured. Any of these going non-zero is a finding.

## 5. The recommendation rule, registered in advance

Which regime our **recurring** Metamon baseline should use is decided by this rule, not by which
number comes out flattering:

* **Default to greedy-vs-greedy.** It is the protocol every other strength number in this project
  is taken under (`ladder.json` is greedy-vs-greedy with symmetric builders; `--temperature 0` is
  `play.py`'s documented measurement setting), and a baseline whose regime differs from the rest of
  the ladder cannot be read next to it. Determinism also shrinks the variance of a fixed-n read.
* **Switch to t1.0-vs-t1.0 if EITHER** (a) the greedy cells fire the forfeit limit, or produce ties
  or 250-turn games, at a rate that costs more than ~2 % of games, **or** (b) the greedy cells show
  materially fewer distinct game trajectories than the t1.0 cells at the same team draws — i.e.
  determinism has collapsed the effective sample size.
* **Report BOTH regimes on a milestone read** regardless of which becomes the recurring one, because
  a deterministic policy in a simultaneous-move hidden-information game is exploitable in principle
  and the sampling cell is the cheap standing check on that.

## 6. What this measurement CANNOT say

* Nothing about Metamon on **Metamon's** Showdown pin. Both clients play on ours (13 commits behind
  its bundled submodule); comparing against numbers Metamon produced on its own server is out of
  scope, as it was in the de-risk.
* Nothing about **exploitation**. A deterministic policy is exploitable only by an opponent that
  adapts; neither side adapts here. Metamon's transformer conditions on the sequence *within one
  battle* (its hidden state is reset on `done`, so it carries nothing across battles), and our model
  carries nothing at all beyond the observation. Neither can mine a fixed opponent over 100 games.
  Whatever these cells say about greedy play, it is not evidence that greedy is safe against an
  adapting opponent.
* Nothing about **which model is stronger in general**. 100 games per cell resolves ~±10 pp at best;
  a cell that lands near even will stay "NOT DETECTED" no matter how the point estimate reads.
