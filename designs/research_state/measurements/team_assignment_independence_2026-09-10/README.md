# DOES A TRAINING ROLLOUT ASSIGN THE TRAINEE'S TEAM INDEPENDENTLY OF THE OPPONENT?

The [N-curve](../winprob_refit_ncurve_2026-09-10/README.md) §9 named this as **the single largest
open question** it left behind, and §12 recorded it as a live counter-hypothesis: *"whether a
TRAINING rollout assigns the trainee's team independently of the opponent is NOT established."*

It matters because of the N-curve's §3 frame finding. In the probe read's EVAL frames the
trainee's **own team alone** predicted the opponent's class at AUC **0.856** (CTRL) and **0.877**
(A) — only **37 of 180** and **47 of 216** distinct trainee teams ever faced a sentinel there —
while on a matched-team frame the same decode reads **0.508**. If a TRAINING rollout has the same
property, the online head has a turn-1 tell that a matched-team eval frame denies it, and every
turn-1 meter on the critic-ladder campaign carries an own-team mediation term that has never been
priced.

Run 2026-09-10/11, offline, CPU only. No GPU, no Showdown server, no launcher, no model forward,
nothing written under `models/`. Reproduce with
`python3 measure.py --workers 48 --episodes 4000 --perm 400`.

---

## VERDICT

**INDEPENDENT — by construction in the code, and bounded at 192,000 simulated episodes per
configuration in the measurement.** Across 4 self-play mixes and 4 further global seeds at the
worst of them, the own-team → opponent-class decode never leaves **|AUC − 0.5| ≤ 0.0042**, against
**0.856 / 0.877** in the probe read's eval frames. The N-curve's counter-hypothesis is
**ELIMINATED** — not because a test failed to reject, but because the effect is bounded two orders
of magnitude below the confound it was raised about. The matched-team frame is the right model of
a training rollout, and the turn-1 mixture "defect" is confirmed as **not a defect**.

**1 — By construction.** The trainee's team and the opponent are drawn by **different objects,
from different RNG streams, at different call sites**, and nothing reads either to choose the
other. `MatchupSpec` holds the two sides as two independent `TeamSource`s and says so as design
intent (`src/agents/training/matchup_spec.py:13-15`, built at `:203-204`). The opponent is chosen
at **`src/agents/training/wrappers.py:495`** (`_select_episode_opponent`) off the wrapper's own
**private** `random.Random(worker_idx)` (`wrappers.py:221`, seeded from `env_factory.py:239`). The
trainee's team is drawn at **`src/utils/teambuilder.py:208`** (`Gen3Teambuilder._draw_team`, called
from `yield_team` at `:192`, reached through `poke_env/player/player.py:873-877`) off the
process-global `random` module (`teambuilder.py:65-69`). Nothing in `_select_episode_opponent` /
`_pick_challenge_opponent` / `_pick_stable` / `_pick_floor_opponent` reads any team state; the only
consumers of the "which team did I just yield" handle `_last_pool_idx` are the two **outcome
recorders** called from `step()` AFTER the episode (`teambuilder.py:233-235`, `:281`, via
`wrappers.py:632-633`).

**2 — Measured, and it reads the null everywhere.** Own-team → opponent-class AUC, by the N-curve's
own leave-one-out estimator, at four self-play mixes × 192,000 episodes each:

| self-play fraction | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|
| **own-team LOO → opp-class AUC** | **0.5021** | **0.5008** | **0.5015** | **0.4945** |
| its permutation null (95%) | [0.4945, 0.5036] | [0.4951, 0.5028] | [0.4943, 0.5041] | [0.4931, 0.5052] |
| distinct teams drawn | 717 | 717 | 717 | 717 |
| **teams that faced the pool class** | **717 / 717** | **717 / 717** | **717 / 717** | **717 / 717** |
| teams that faced the bot class | 717 / 717 | 717 / 717 | 717 / 717 | 717 / 717 |

**All 717 drawn teams face both opponent classes** — against the probe read's 37 of 180. A
rollout's frame is the matched-team frame.

⚠️ **Three of sixteen intervals are grazed, none of them twice, and the BOUND is the result** —
§4. At n = 192,000 the permutation null is ±0.004 wide, so "inside the null" is a claim about the
fourth decimal of an AUC and is the wrong thing to lean on. The claim that carries weight is that
**no configuration and no seed ever put the decode further than 0.0042 from chance**, and a
dependence that small cannot produce a 0.86 decode.

**3 — The one edge that DOES exist runs the other way, and is harmless.** `_apply_opponent_team`
(`wrappers.py:527-535`) pins a specialist stable/exploiter opponent to **its own** recorded team.
That couples **opponent → opponent's team**, never opponent → trainee's team. The ladder arm
measured here (`ai_v12_11_ladder_ctrl10M`) names no `--stable-opponents` and no `--exploiter`, so
the edge is not even live on it.

**4 — 🚨 The one real channel by which a dependence COULD have existed, named and measured.** Both
teambuilders default to the `random` **MODULE** as their stream, so the trainee's draw and the
opponent's draw are consecutive calls off **one** Mersenne state — and `random.choice` consumes a
number of words that depends on the sequence LENGTH. A specialist opponent piloting a 1-6 team pin
therefore advances the shared stream by a different amount than the flat 719-team pool does, which
means the **opponent's class can shift the PHASE of the stream the next episode's trainee draw
comes off**. The measurement reproduces that interleaving rather than assuming it away
(`measure.py::simulate` draws the opponent side after the trainee side, every episode, off the
shared module stream). It is an interleaving artefact and not a dependence, and the numbers above
say so.

---

## 1. What was replayed, and in what order

The draw is not one function — it is four call sites in three modules, and the measurement runs
them in the order `reset()` does:

```
wrappers.py:537-547        reset() -> _select_episode_opponent() -> _apply_opponent_team()
                                   -> super().reset()
poke_env/environment/env.py:673-675   -> agent1.battle_against(agent2)
poke_env/player/player.py:759 / :597  -> get_next_team() on EACH side
poke_env/player/player.py:873-877     -> self._team.yield_team()
utils/teambuilder.py:192 -> :208      -> yield_team() -> _draw_team()
```

The opponent half runs the **real** `MaskableAgentWrapper` against a stub env — the same harness
`src/agents/training/wrappers_test.py:13-33` uses, which is what makes this a CPU measurement and
not a training run. The team half runs the **real** `Gen3Teambuilder` over the **real** 719-team
pool (`TeamLoader`), built through the **real** `MatchupSpec.from_args`.

**The configuration is the production ladder arm's**, read from
`models/ai_v12_11_ladder_ctrl10M/metadata.json`'s `original_command`: `--self-play`, `--team-pfsp
off`, `--team-block-episodes 1`, 9 scripted bots, no `--trainee-team(s)` pin, no
`--stable-opponents`, no `--exploiter`. Trainee teams are therefore `default_biased` — the full
719-team pool with a **10% tilt** toward the 72 curated sample teams — and the opponent side is the
flat 719-team pool.

`self_play_fraction` is a **global scalar** pushed after every eval (`snapshot_pool.py:93-125`,
`wrappers.py:223-228`), a function of the measured win rate vs bots and of **nothing team-side**,
so it is swept rather than fixed: 0.0 / 0.25 / 0.5 / 0.75 / 0.9. **`0.0` is reported DEGENERATE and
not as a result** — at that mix every episode is a bot episode, the class label has one level, and
a class AUC is undefined. A statistic undefined on every permutation is refused in code
(`measure.py::permutation_null`) rather than emitted as a NaN that reads like a null.

48 workers (the arm's own `--n-envs 48`), 4,000 episodes each = **192,000 episodes per
configuration**; a pool snapshot identity is held for 500 episodes because `_ensure_pool_model`
re-samples **once per generation**, not per episode (`wrappers.py:410-430`).

---

## 2. The three registered statistics

### (i) own team → opponent CLASS

| fraction | LOO-AUC | null (95%) | MI(team; class) bits | its null (95%) |
|---|---|---|---|---|
| 0.25 | 0.5021 | [0.4945, 0.5036] | 0.002784 | [0.002436, 0.002974] |
| 0.50 | 0.5008 | [0.4951, 0.5028] | 0.002896 | [0.002459, 0.002985] |
| 0.75 | 0.5015 | [0.4943, 0.5041] | 0.002904 | [0.002437, 0.002960] |
| 0.90 | 0.4945 | [0.4931, 0.5052] | 0.002490 | [0.002470, 0.003005] |

🚨 **The plug-in mutual information of a 717-category variable against a 2-category one is bounded
away from zero at any finite n** — every value above is ~0.0027 bits and every null is ~0.0027
bits. Quoting the MI against **zero** would "detect" independence's own sampling noise; it is read
against the permutation null and against nothing else. That is the N-curve's own standing rule (§9)
applied to its own successor.

### (ii) own team → opponent IDENTITY

| fraction | distinct opponents | H(opponent) bits | MI(team; opponent id) | its null (95%) |
|---|---|---|---|---|
| 0.25 | 393 | 5.3327 | 0.681145 | [0.680148, 0.683535] |
| 0.50 | 393 | 6.8805 | 0.927480 | [0.926170, 0.930736] |
| 0.75 | 393 | 8.0393 | **1.065212** | [1.056503, **1.062502**] |
| 0.90 | 393 | 8.5112 | 1.107139 | [1.106917, 1.114295] |

⚠️ **One cell is marginally outside its null and is reported rather than rounded away**: at
fraction 0.75 the MI sits **+0.0027 bits above** the null's 97.5th percentile, on an entropy of
8.04 bits — a **0.03%** excursion. §4 replicates it.

### (iii) does the TEAM distribution differ by opponent class?

| fraction | χ² (dof 716) | χ²/dof | null χ² (95%) | min expected | KS D on team index | KS p |
|---|---|---|---|---|---|---|
| 0.25 | 741.8 | 1.036 | [645.9, 790.4] | 46.5 | 0.00428 | 0.5945 |
| 0.50 | 769.0 | 1.074 | [636.8, 784.6] | 98.9 | 0.00323 | 0.7586 |
| 0.75 | 771.8 | 1.078 | [643.4, 781.9] | 49.6 | 0.00713 | 0.0731 |
| 0.90 | 659.7 | 0.921 | [653.2, 798.2] | 20.1 | 0.00674 | 0.4773 |

Every χ² is inside its own Monte-Carlo null. The **bias-draw share** — the fraction of episodes
taking the 10% tilt toward the 72 sample teams — is the same on both sides of the class split at
every mix (0.1011 / 0.1004, 0.1006 / 0.1010, 0.1009 / 0.0989, 0.1003 / 0.1024), which is the
sharpest single check that the opponent does not reach the team branch: the bias coin is the one
place where the team draw's own control flow forks.

---

## 3. 🚨 WHICH REGISTERED ROWS A NON-INDEPENDENT ASSIGNMENT WOULD HAVE TOUCHED

This is the part that decides how much the result is worth, and it is **two registered rows plus a
floor** — not a lean.

| what | why a team↔opponent dependence would have moved it | status now |
|---|---|---|
| **`cond.own_team_r2.t1`** — the critic ladder's **registered decision row** for arm 8, one of the two rows a ladder arm is judged on | It decodes the trainee's own-team leave-one-battle-out win rate from `V` at turn 1. If teams were assigned partly by opponent, a team's LOO win rate would carry the opponent's strength, and the row would read the OPPONENT through the team — the exact mediation the N-curve found in the probe read's frames on the class axis. The arm-vs-control **delta** would inherit it whenever the two arms' team↔opponent couplings differed | **CLEAN.** The coupling is zero in the training rollout, so the row measures own-team information and nothing else |
| **the identity-bias replicate floor, 0.07–0.10** (rule 19, run-level family) | The floor is built from control-vs-control reads. A team↔opponent coupling that varied run to run (it would — each worker's opponent stream is `random.Random(idx)` while the team stream is OS-entropy-seeded) would inflate the run-level spread, i.e. the floor itself, and every arm would then be judged against a bar partly made of this artefact | **CLEAN.** No coupling to inflate it; the 0.07–0.10 floor is not carrying a mediation term |
| every **turn-1** conditioning row (`cond.spread_ratio.t1_3`, `cond.opp_class_auc.t1`) | The N-curve's §9 conditional: *"if it does not, the online head has a turn-1 tell the eval frame denies it, and every turn-1 meter on this campaign needs the own-team mediation reference run beside it."* | **NO EXTRA REFERENCE NEEDED.** The antecedent is false |

**What this does NOT clear.** The eval-vs-training **population** gap survives untouched — these
are training rollouts against a self-play pool plus 9 bots, and a `critic_read` frame is eval
battles against a pinned 12-opponent panel. That remains the N-curve's principal limit. And it
says nothing about the probe read's own frames, which really were starved: that was the **capture
quota**, not the pairing rule (§5).

---

## 4. The marginal cells, replicated — and the bound that is the actual result

Four fractions × four statistics is sixteen 95% intervals, so ~0.8 excursions are expected under
the null before any data is seen. **Three appeared**, all at or near the worst fraction, so that
fraction was re-run at **three fresh global seeds** rather than argued away
([`replicates.json`](replicates.json)):

| self-play 0.75, global seed | 20260910 (base) | 101 | 202 | 303 |
|---|---|---|---|---|
| own-team LOO → class AUC | 0.5015 | 0.4959 | **0.5042** | 0.5023 |
| its permutation null (95%) | [0.4943, 0.5041] | [0.4956, 0.5032] | [0.4947, **0.5034**] | [0.4948, 0.5033] |
| MI(team; opponent id) bits | **1.065212** | 1.058981 | 1.059435 | 1.057617 |
| its null (95%) | [1.0565, **1.0625**] | [1.0564, 1.0622] | [1.0559, 1.0622] | [1.0567, 1.0629] |
| χ² (dof 716) | 771.8 | 657.7 | **822.5** | 745.0 |
| its null (95%) | [643.4, 781.9] | [651.2, 800.3] | [648.1, **788.2**] | [647.4, 782.4] |
| KS p on team index | 0.0731 | 0.2374 | 0.1796 | 0.2677 |

**No excursion replicates.** The MI is outside only at the base seed, the χ² and the AUC only at
seed 202, and the AUC excursions point in **opposite directions** across seeds (0.4959 and 0.5042).
Three of sixteen at the base and three of sixteen across the replicate block, never the same
statistic twice, is what an independent draw scattered against sixteen 95% intervals looks like —
and it is what the code path predicts, the two streams being a private `random.Random(worker_idx)`
and the process-global `random` module with no call site reading across them.

🚨 **So the reading is the BOUND, not the verdict of any one interval.** At n = 192,000 a
permutation null is ±0.004 wide; "inside" and "outside" are then statements about the fourth
decimal of an AUC, and either would be a weak thing to rest a campaign on. What the measurement
actually establishes is a **ceiling on the effect**: over 8 (fraction, seed) cells and 1.5 million
simulated episodes, the largest own-team → opponent-class decode observed is **0.5042** and the
smallest **0.4945**, i.e. **|AUC − 0.5| ≤ 0.0042**. The confound this was raised to rule out reads
**0.856** and **0.877**. Nothing at 0.004 produces that.

⚠️ **What the four seeds do and do not re-randomise.** Each worker's OPPONENT stream is
`random.Random(worker_idx)` (`wrappers.py:221`) — deterministic in the worker index and therefore
**identical across all four seeds**; only the global team stream is re-seeded. That is the right
design for this question, because the quantity under test is the PAIRING and the pairing is fully
re-randomised. It does mean the four cells are not independent replicates of the opponent sequence
itself, and a defect that lived in `random.Random(idx)`'s own sequence would be invisible to all
four alike.

---

## 5. The EVAL side — the pairing rule is exonerated, the CAPTURE QUOTA is convicted

The N-curve asked which mechanism produced a frame in which only 37 of 180 teams ever faced a
sentinel. **It is the capture quota, and the pairing rule could not have done it.**

**The pairing rule is one i.i.d. draw from ONE builder, identical for every opponent.**
`src/main/eval_worker.py:296-303` builds a single `trainee_tb` for the whole worker and
`:221-223` hands that same object to every shard unit whatever the opponent is; the draw then
fires per battle through the same `poke_env` path as training. `_build_trainee_tb`
(`eval_worker.py:65-77`) is the run's pin if any, else the same `bias_prob=0.1` builder training
uses. There is no per-opponent team list, no fixed assignment, no stratification. And the
arithmetic refuses it: 100 games × 5 sentinels = **500 sentinel battles per cycle**, and
coupon-collector over 180 teams gives **≈169 distinct teams in one cycle**, essentially all 180
within two. A games-derived frame cannot show 37.

**The quota fits exactly.** Traces are capped **per opponent** at 10 losses / 5 wins / 5 draws
(`eval_callback.py:170-189`), per shard after `per_shard` rounding (`:230-246`), and the battle is
still played and counted when the bucket is full — only the capture is dropped (`:1188-1210`). At
~190 traces per cycle over 9 bots + 5 sentinels the sentinel side receives roughly **65–140 traces
per cycle**, each carrying exactly one trainee team, so the distinct-team count is hard-bounded by
the trace count: 180·(1−e^(−70/180)) ≈ **58**, and lower once the loss quota concentrates captures.
**37 of 180 sits inside that envelope.** The teams played those battles; the battles were never
written to disk.

A secondary asymmetry falls out of the same mechanism and is worth knowing: the roster is **9 bots
against 5 sentinels** and the quota is per opponent, so a trace-derived frame *always* shows more
distinct teams having faced bots than sentinels, with no behavioural meaning whatever.

**The fix is already built and is what this campaign's offline frames use**: `main.ops.
eval_trace_gen` captures **everything** by default (`eval_trace_gen.py:437-460`), which is why the
hp800 frames read 602 of 602 teams facing every opponent.

---

## 6. Files

| file | what |
|---|---|
| [`measure.py`](measure.py) | the replay + the four statistics + the permutation nulls |
| [`results.json`](results.json) | the five configurations, every number above |
| [`replicates.json`](replicates.json) | fraction 0.75 at three fresh global seeds |
