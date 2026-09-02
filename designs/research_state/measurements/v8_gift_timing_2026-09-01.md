# v8 GIFT TIMING — does the untaught gain ACCRUE over the fold's PPO time?

**Status: PRE-REGISTERED 2026-09-01 19:38:54 PDT (2026-09-02T02:38:54Z), before the first
measurement cell.** Results are appended below only after the battles run.

---

## 0. PRE-REGISTRATION (frozen — nothing below this block may be edited after the first cell)

### The question

Probe P (`v8_redistribution_pfsp_2026-08-30.md`) measured that v8's fold —
`ai_v8_14_distill3_0725`, forked from `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip`
at **277,583,267** steps and run to **292,623,779** (≈ **+15.04M** fold steps), PFSP 2.5, teachers
in pool — **GIFTED +5.42pp [+3.44, +7.42], z=4.83, 14/16 teams positive** to 16 pool teams it never
taught. That is an ENDPOINT measurement. It says nothing about *when* during the fold the gift
appeared.

**Owner's hypothesis (2026-09-01):** the gift is the off-slice PPO signal RE-OPTIMISING the
distillation leak — the leak is a *perturbation*, PPO is the *selector* — which needs PPO TIME.
Gen-era folds ran 4.5M steps at 3-6× the dose and robbed; v8 ran ~15M and gifted.

**Prediction under the hypothesis:** the untaught gain ACCRUES over the fold — small early, most of
it late.

**Competing account** (the leak is the direct content, or the gift arrives with the teachers'
content): the gain is present EARLY (first ~1-2M fold steps) and flat after.

### The registered bars — scored against INTERVALS, never points

> **P1.** *At least half of the final untaught gain is ABSENT at the first retained checkpoint
> (+1.09M fold steps).* Operationally: **PASS** iff the 95% cluster-bootstrap interval on
> `gain(+1.09M)` lies entirely **below** `0.5 × gain(final)` (point estimate of the final gain,
> taken from this probe's own `foldfinal` cell). **FAIL** iff that interval lies entirely **above**
> it. **UNDECIDED** iff the interval straddles the half-gain line.

> **P2.** *The untaught gain at the LAST checkpoint exceeds the gain at the MIDPOINT checkpoint by
> more than the paired CI half-width* — i.e. the gift is still rising in the second half.
> Operationally: **PASS** iff the 95% cluster-bootstrap interval on the paired, per-team difference
> `gain(final) − gain(MID)` lies entirely **above zero**. **FAIL** iff it lies entirely below zero.
> **UNDECIDED** iff it straddles zero.
>
> **MID is fixed here, before data, as `c283636` = checkpoint 283,635,665 (+6.05M)** — the scored
> cell nearest the fold's half-way point (+7.52M) among the six on the registered grid. The
> unscored retained checkpoint 284,632,327 (+7.05M) is nearer still, but the grid was registered
> as +1/+3/+6/+9/+12M and is not being re-chosen to suit the test. `c287136` (+9.55M) is reported
> as a **sensitivity**, labelled as such, never as the verdict.

A bar on a point estimate is a vacuous guard (ledger 2026-09-01). Every verdict above is an
interval verdict, and **UNDECIDED is a real outcome** that will be reported as one.

### The instrument — REUSED, not rebuilt

| | |
|---|---|
| script | `v8_gift_timing_probe.py` (this directory) |
| cell constants | copied **VERBATIM** from `v8_fold_behavioral_fingerprint_probe.py::V8_SELECTION`, which copied them from probe P's own `/tmp/probeP/selection.json` |
| untaught team set | probe P's 16 pre-registered untaught probe teams (sha10s in `v8_gift_timing_inputs/selection.json`) — archetype-stratified: 4 balance, 3 hyper_offense, 3 offense, 3 semi_stall, 3 stall |
| opponent teams | probe P's 8 fixed opponent teams, all 8 used |
| fixed reference opponent | `ai_v8_03_zarch_control_0718/final_model_interrupted.zip` — an ancestor of every arm and equal to none (parent-as-opponent would make the parent arm a self-mirror) |
| CRN | `random.Random(f"{team_sha}:{opp_sha}")` → 4-int sim seeds, probe P's construction verbatim. **Game index i is the same battle for every arm**, and this probe's battles are a CRN **prefix subsample** of probe P's own 30-game cells |
| regime | greedy both sides (`stochastic=False`), pinned single team per side, node bridge, no server, CPU, `nice -n 15`, 1 battle in flight per process, `GEN3AI_TIMEOUT_SCALE=12` |
| era pin | `b13b30b289c5eaba136a930a4ab63451e209fbe5` — a private copy of the era checkout, `PYTHONPATH=<era>/src` |
| n | **16 games per (team, opp) cell** = 128 battles per team per arm = 2,048 battles per arm. Fixed here, pre-data, from the compute budget; probe P used 30 |

**The five reproducibility seeds do not exist at this commit.** `$GEN3AI_{PLAYER,TEAM,POLICY,POOL,
STALLER}_SEED` landed 2026-08-30, months after `b13b30b`; a grep over the era tree finds **zero**
references, so probe P could not have set them either. Determinism comes from the same three
things it came from for probe P and M4: no policy draw (`stochastic=False`), no team draw (one
pinned team per side), no dice draw (an explicit 4-int sim seed per battle). This is recorded
rather than asserted.

### The cells

Fork point 277,583,267. `ai_v8_14_distill3_0725/checkpoints/` retains **14** checkpoints (not 28 —
`models/` retention thinned the run); the full retained list is in
`v8_gift_timing_inputs/checkpoints.json`. Six arms are scored plus the parent:

| arm | file | step | fold Δsteps |
|---|---|---:|---:|
| `parent` | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` | 277,583,267 | 0 (the fork source; probe P's baseline arm) |
| `c278672` | `checkpoints/checkpoint_278671945_steps.zip` | 278,671,945 | **+1.089M** (FIRST retained) |
| `c280656` | `checkpoints/checkpoint_280656375_steps.zip` | 280,656,375 | +3.073M |
| `c283636` | `checkpoints/checkpoint_283635665_steps.zip` | 283,635,665 | +6.052M (**MID**, P2) |
| `c287136` | `checkpoints/checkpoint_287136098_steps.zip` | 287,136,098 | +9.553M |
| `c290116` | `checkpoints/checkpoint_290115536_steps.zip` | 290,115,536 | +12.532M |
| `foldfinal` | `final_model_interrupted.zip` | 292,623,779 | **+15.041M** (LAST; probe P's fold arm) |

**Why `foldfinal` and not the 292,100,648 checkpoint.** The last checkpoint is +14.52M, but the
run's `final_model_interrupted.zip` is +15.04M — 0.52M further on, and
`‖ckpt − final‖₂ = 17.15` against the fold's total travel of 238.9, so they are **not** the same
weights. Probe P's fold arm is the final; scoring the checkpoint instead would put this curve's
endpoint on a different model from the headline it is timing. Both ends of this curve are
therefore probe P's own two zips, byte-for-byte.

### Statistics, fixed before data

- **Per-team win rate** = wins / finished over that team's 8 opponents × 16 games (128 battles).
- **Gain** = arm − parent, on the **same battles** (CRN), reported as the **equal-weight mean over
  the 16 teams** — probe P's convention, so the numbers are on probe P's scale.
- **Cluster bootstrap over TEAMS**, 8,000 resamples (probe P's count), 95% percentile interval.
  Teams are the unit the claim generalises over.
- A **binomial (pooled, battle-level) interval** is reported beside it and is explicitly the
  *weaker* one — it ignores team clustering, which the Simpson-trap rule says is the wrong unit.
- **P2's difference interval is paired at the team level**: resample teams, and within a resample
  take `gain_final(team) − gain_mid(team)` on that team's shared battles.
- A cell whose `finished < requested` is reported; win rate uses `finished` as the denominator and
  any shortfall is printed rather than absorbed.

### What this probe CANNOT say

1. **One fold, one lineage.** Six points on v8_14's trajectory. A time course here is not a claim
   about folds in general, and cannot be, at n=1 fold.
2. **Step is confounded with everything else that moves with step** — the pool contents, the PFSP
   weights, the LR schedule, the distillation dose already delivered. "It accrues" would mean
   *something that grows over the fold* produced it, not specifically PPO re-optimisation.
3. **The reference opponent is fixed and single.** The gain is against `ai_v8_03`, not against the
   world.
4. **A non-monotone curve is not evidence for either account** and will be reported as such.

---

## 1. THE HEADLINE

**Neither prediction survives. The untaught gift is a TRANSIENT, not a ramp and not a step: it
roughly DOUBLES over the first ~3M fold steps, plateaus near +8 to +10pp for the next ~9M, and then
DECAYS monotonically over the last ~2.5M back to the +5pp probe P recorded.**

```
untaught gain vs the parent, pp, equal-weight mean over 16 teams

+10 |                    ●9.03                    ●9.67
    |         ●8.06                  ●8.06              ●8.25
 +8 |                                                        ●7.03
    |
 +6 |
    | ●4.64                                                        ●4.98
 +4 |
    +--+-------+---------+---------+---------+------+------+------+---->
     +1.1M   +3.1M     +6.1M     +9.6M    +12.5M +13.5M +14.5M  +15.0M   fold steps
```

| bar | verdict | interval |
|---|---|---|
| **P1** — ≥ half the final gain absent at +1.089M | **UNDECIDED** | gain(first) = +4.64pp, CI **[+2.20, +7.47]**, straddling the half-gain line at +2.49pp. The sharper paired form (`gain(first) − ½·gain(final)`) is **+2.15pp [−0.20, +4.66]** — also UNDECIDED, but with the whole interval bar 0.2pp of it on the "already present" side |
| **P2** — still rising in the second half | **FAIL** | `gain(final) − gain(MID=+6.05M)` = **−4.05pp, CI [−8.01, −0.49]**, wholly below zero. Sensitivity at +9.55M: **−3.08pp [−5.86, −0.20]**, also wholly below zero |

**The owner's hypothesis is refuted on its own second half.** The prediction was *small early, most
of it late*. What happened is *most of it by +3M, and the last quarter of the fold gives a third of
it back*. If the gift were the off-slice PPO signal grinding a distillation perturbation into
generalisable policy, more PPO time would not have removed nearly half of it.

**But the flat-from-the-start competitor loses too.** The gain at +1.089M is +4.64pp and at +3.073M
is +8.06pp — the paired difference between those two arms is **+3.42pp**, and 14/16 teams are
already positive at the first checkpoint. So something *does* build over the first ~2-3M steps; it
simply finishes building long before the fold does, and then erodes.

**The consequence that matters most for the ledger: probe P's +5.42pp is a reading taken on the
decayed tail.** The fold's actual peak externality on untaught teams was **+9.67pp [+6.79, +12.50]**
at +12.53M — roughly double the number the scorecard carries. `endpoint − peak = −4.69pp
[−8.11, −1.51]` (paired over teams; the peak arm is *selected*, so read this as descriptive, not as
a registered test). **Stopping v8's fold ~2.5M steps earlier would, on this meter, have kept about
twice the gift.**

---

## 2. THE CURVE

**Meter stamp — every number in §2 and §3 carries it:** `regime` greedy both sides
(`stochastic=False`), node bridge, no server, CPU, `nice -n 15`, 1 battle in flight per process,
`GEN3AI_TIMEOUT_SCALE=12`, era pin `b13b30b289c5eaba136a930a4ab63451e209fbe5` · `opponent`
`ai_v8_03_zarch_control_0718/final_model_interrupted.zip` (fixed for every arm) · `team set` probe
P's 16 untaught probe teams × probe P's 8 fixed opponent teams · `n` 16 CRN games per cell = 128
battles per team per arm · **18,432 battles over 9 arms, 1,152 cells, ZERO errors and ZERO short
cells.**

| arm | fold Δ | battles | pooled WR | gain vs parent (mean over teams) | cluster-boot 95% | pooled binomial 95% | z | teams + |
|---|---:|---:|---:|---:|---|---|---:|---:|
| `parent` | +0.000M | 2048 | 0.3877 | — | — | — | — | — |
| `c278672` | +1.089M | 2048 | 0.4341 | **+4.64pp** | [+2.20, +7.47] | [+1.63, +7.65] | +3.02 | 14/16 |
| `c280656` | +3.073M | 2048 | 0.4683 | **+8.06pp** | [+4.88, +11.28] | [+5.04, +11.08] | +5.23 | 14/16 |
| `c283636` | +6.052M | 2048 | 0.4780 | **+9.03pp** | [+5.62, +12.60] | [+6.01, +12.05] | +5.86 | 15/16 |
| `c287136` | +9.553M | 2048 | 0.4683 | **+8.06pp** | [+4.74, +11.18] | [+5.04, +11.08] | +5.23 | 14/16 |
| `c290116` | +12.532M | 2048 | 0.4844 | **+9.67pp** | [+6.79, +12.50] | [+6.64, +12.69] | +6.27 | 15/16 |
| `c291106` *(follow-up)* | +13.523M | 2048 | 0.4702 | **+8.25pp** | [+4.15, +11.82] | [+5.23, +11.27] | +5.35 | 14/16 |
| `c292101` *(follow-up)* | +14.517M | 2048 | 0.4580 | **+7.03pp** | [+3.27, +10.35] | [+4.01, +10.05] | +4.57 | 13/16 |
| `foldfinal` | +15.041M | 2048 | 0.4375 | **+4.98pp** | [+2.15, +7.62] | [+1.97, +7.99] | +3.24 | 13/16 |

The **cluster bootstrap over teams is the primary interval**; the pooled binomial is printed beside
it because a reader will ask, and it is the weaker of the two — it treats 2,048 battles as 2,048
independent observations when the claim generalises over 16 teams.

**The two follow-up arms are NOT part of the registered grid.** They were added after the six
registered cells came back humped, for one purpose: to establish whether the fall from +12.53M to
the endpoint is a *progressive decline* or an artifact of `final_model_interrupted.zip` being an
interrupt save rather than a periodic checkpoint. The answer is **progressive** — 9.67 → 8.25 →
7.03 → 4.98 over four consecutive arms, monotone, with `c292101` an ordinary periodic checkpoint.
The decline is a property of the last ~2.5M fold steps, not of the file the run happened to end on.

### 2.1 The instrument checked itself against probe P, three ways

The endpoint arm and the baseline arm here are probe P's **own two zips**, so this probe's
endpoint is a direct replication of the headline it is timing — on 16 CRN games per cell instead
of probe P's 30, i.e. a **prefix subsample of probe P's own battles**, not a fresh draw.

| check | probe P | here | verdict |
|---|---|---|---|
| parent's untaught pooled WR vs the fixed reference | 0.3828 | **0.3877** | reproduces |
| fold − parent on untaught teams | **+5.42pp [+3.44, +7.42]** | **+4.98pp [+2.15, +7.62]** | reproduces; each point sits inside the other's interval |
| `‖parent − reference‖₂` (ACID, parameter space) | 53.3 (M4's reading) | **53.3345** | the era-pinned load is the same load |

Every arm additionally passed the ACID gate at load (arm and reference distinct networks,
3,512,397 parameters each) — a mis-resolved path that loaded one zip twice would read as a perfect
null, so distinctness is a gate rather than a nicety.

---

## 3. PER-TEAM ROWS

Gains in pp against the parent, on the same battles. Positive = that arm beat the fork source on
that team.

| team | arch | parent WR | `c278672` | `c280656` | `c283636` | `c287136` | `c290116` | `c291106` | `c292101` | `foldfinal` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `048182d1e9` | stall | 0.203 | +3.1 | +7.8 | +6.2 | +7.8 | +3.1 | +15.6 | +10.2 | +8.6 |
| `32f549483f` | offense | 0.328 | −4.7 | +7.0 | +6.2 | +6.2 | +11.7 | +3.9 | +9.4 | +2.3 |
| `593d7fb8a8` | offense | 0.422 | +4.7 | +14.8 | +6.2 | +16.4 | +12.5 | +4.7 | +10.9 | +7.0 |
| `7163ad9387` | semi_stall | 0.375 | +9.4 | +14.1 | +12.5 | +10.9 | +14.1 | +16.4 | +16.4 | +13.3 |
| `7c2cb5cec1` | hyper_offense | 0.414 | +11.7 | +10.2 | +7.8 | +7.8 | +8.6 | +7.0 | +14.8 | +8.6 |
| `89fcef3b53` | offense | 0.523 | +1.6 | +21.1 | +25.0 | +18.8 | +22.7 | +12.5 | +12.5 | +6.2 |
| `9292a21833` | hyper_offense | 0.523 | +2.3 | −3.9 | −7.0 | −7.0 | −3.9 | −10.2 | −9.4 | −2.3 |
| `a577a735b7` | balance | 0.453 | +7.0 | −0.8 | +8.6 | +7.8 | +12.5 | +8.6 | +0.0 | +1.6 |
| `a6b630e6b4` | balance | 0.328 | +3.1 | +5.5 | +5.5 | +3.9 | +6.2 | +14.8 | +1.6 | +7.8 |
| `b26ed9c8e1` | semi_stall | 0.266 | +2.3 | +14.1 | +14.1 | +11.7 | +14.1 | +17.2 | +8.6 | +3.9 |
| `c84f2b64a2` | stall | 0.297 | +1.6 | +7.8 | +5.5 | +3.9 | +9.4 | +5.5 | +10.9 | +8.6 |
| `c90e782cad` | balance | 0.352 | +19.5 | +14.1 | +18.8 | +16.4 | +15.6 | +18.8 | +8.6 | +10.9 |
| `d0a4d2bcb8` | balance | 0.391 | +7.8 | +8.6 | +12.5 | +3.9 | +8.6 | +4.7 | +8.6 | −4.7 |
| `dd460484fc` | semi_stall | 0.477 | +2.3 | +2.3 | +0.8 | +14.1 | +3.9 | +1.6 | +14.1 | +3.9 |
| `eaa88395e7` | hyper_offense | 0.477 | +2.3 | +0.8 | +12.5 | +6.2 | +7.8 | +14.1 | +1.6 | +10.9 |
| `f593373169` | stall | 0.375 | +0.0 | +5.5 | +9.4 | +0.0 | +7.8 | −3.1 | −6.2 | −7.0 |

Two teams carry most of the shape's amplitude — `89fcef3b53` (offense) runs +1.6 → +25.0 → +6.2 and
`b26ed9c8e1` (semi_stall) +2.3 → +17.2 → +3.9 — but the hump is not theirs alone: **14 of 16 teams
peak at an interior arm rather than at either end**, which is why the pooled curve is humped and the
cluster interval on the endpoint-minus-peak contrast clears zero.

One team, `9292a21833` (hyper_offense), is **negative at every arm past +1.1M** and reaches −10.2pp.
It is the single untaught team the fold consistently *robbed*, and the robbery deepens exactly where
the pooled gift peaks — a reminder that "the fold gifted untaught teams" is a statement about a mean
over 16 teams, not about every team.

---

## 4. SCORING THE BARS

### P1 — "≥ half the final gain is absent at the first retained checkpoint": **UNDECIDED**

- `gain(+1.089M)` = **+4.64pp**, cluster CI **[+2.20, +7.47]**.
- The half-gain line is `0.5 × gain(final)` = **+2.49pp**.
- The interval **straddles** it (by 0.29pp at the low end), so the registered rule returns
  UNDECIDED. It does not return PASS, and the point estimate is on the FAIL side: 93% of the final
  gain is already present at the first checkpoint.
- **Paired supplementary** (sharper — the registered rule bootstraps `gain(first)` alone and treats
  the half-gain as a constant, discarding the fact that both arms played the same battles):
  `gain(first) − ½·gain(final)` = **+2.15pp, CI [−0.20, +4.66]**. Still UNDECIDED, and still with
  almost the whole interval on the "already present" side. **Reported beside the registered
  verdict, never in place of it.**

**The registered denominator turned out to be the contentious choice, and this is worth stating
plainly.** P1 was written against "the final gain" because the pre-registration assumed a monotone
curve, where *final* and *most it ever reached* are the same number. They are not. Re-asked against
the PEAK — `gain(first) − ½·gain(+12.53M)` = **−0.20pp, CI [−2.98, +2.54]** — the answer is
UNDECIDED too, but now centred almost exactly on the half line: at the first checkpoint the fold
holds **48% of the gift it would eventually reach**. That version is **POST-HOC** (the denominator
was chosen after seeing the shape, and the peak arm is itself selected as a maximum), so it settles
nothing; it is recorded because a reader who only saw the registered framing would take away
"the gift was there from the start", and against the peak that is half true at most.

### P2 — "still rising in the second half": **FAIL**

- `gain(final) − gain(MID = +6.05M)` = **−4.05pp**, paired cluster CI **[−8.01, −0.49]** — wholly
  below zero.
- Sensitivity at `c287136` (+9.55M): **−3.08pp, CI [−5.86, −0.20]** — wholly below zero, same sign.
- Both intervals clear zero only narrowly at the upper end, so the strength of the claim is
  "detectably falling", not "collapsing".

The fold's second half does not add untaught gain. It **removes** it.

---

## 5. WHAT THIS DOES AND DOES NOT LICENSE

**Does:**

1. **The accrual mechanism, as stated, is not what produced v8's gift.** More PPO time does not
   monotonically buy more untaught externality on this fold; past ~+12.5M it costs it.
2. **The scorecard's +5.42pp under-states v8's untaught externality by about half.** Any
   cross-generation comparison that uses v8's endpoint as "what a good fold delivers" is comparing
   against a decayed reading. Where a rev-N fold is scored at *its* endpoint, the comparison is at
   least consistent — but "v8 gifted +5.4pp" and "v8's fold reached +9.7pp" are both true and mean
   different things.
3. **A cheap, testable operational lever exists and is NOT yet tested:** *stop the fold at the
   untaught peak.* On this meter the peak sat at +12.5M of a +15.0M fold. That is a claim about one
   fold measured post-hoc; it needs a pre-registered replication on a different fold before it is a
   recipe.

**Does not:**

1. **n = 1 fold.** Nine points on `ai_v8_14`'s trajectory. Nothing here says gen-era folds have the
   same shape, and the natural next probe is exactly that — the rev-2/rev-3 folds' own untaught
   curves, where the endpoint readings are −7.1pp and −0.75pp.
2. **Step is confounded with everything that moves with step.** The pool grew, the PFSP weights
   moved, the LR schedule ran, and the distillation dose accumulated. "Something that peaks near
   +12.5M and then reverses" is what was measured; *which* of those it is was not.
3. **The reference opponent is fixed and single** (`ai_v8_03`). The curve is a gain against one
   ancestor, not against the world; a different reference could scale or reshape it.
4. **The decline's mechanism is untouched.** Whether the last 2.5M steps over-fit the taught slice,
   drifted the policy off the gift's behavioural direction (M4's `switch|losing_matchup` /
   `switch|early` axes), or simply regressed toward the pool is not measurable from win rates. The
   M4 fingerprint instrument re-run at `c290116` vs `foldfinal` would answer it, on identical
   boards, for the cost of one more dual-scored pass.

---

## 6. ARTIFACTS

| file | holds |
|---|---|
| `v8_gift_timing_2026-09-01.json` | every number above — per-arm aggregates, all 144 per-team rows, both bars, the shape block |
| `v8_gift_timing_2026-09-01_tables.md` | the two tables, regenerated from the JSON |
| `v8_gift_timing_2026-09-01_cells.jsonl.gz` | **all 1,152 cells** — per-cell win counts and the per-game outcome vector, so every aggregate above is re-derivable battle-by-battle without replaying anything |
| `v8_gift_timing_probe.py` | the battle pass (probe P's cells, verbatim; resume-safe) |
| `v8_gift_timing_analyze.py` | the analysis — pure arithmetic over recorded outcomes, no model and no battle |
| `v8_gift_timing_inputs/checkpoints.json` | the 14 retained checkpoints + the final, each with its `num_timesteps` read from the zip and its fold delta |
| `v8_gift_timing_inputs/selection.json` | probe P's team selection, with byte-equality against the fingerprint probe's copy **asserted by the script that wrote it** |
| `v8_gift_timing_inputs/seeds.json` | the environment, the regime, and the note on the five inert seed vars |

