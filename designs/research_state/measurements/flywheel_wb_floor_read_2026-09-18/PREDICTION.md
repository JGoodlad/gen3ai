# PRE-REGISTRATION — the 75M RUN-LEVEL FLOOR read (W_b vs W), 2026-09-18

**Committed BEFORE any W_b number in this directory exists.** Standing rule 1: both branches, the
bar, and the comparator, written down first. What is already known to me at the time of writing is
listed in §0 and is explicitly excluded from the "blind" claims — a prediction made after reading
the number is not a prediction.

---

## 0. What this read is, and what is already known

`ai_v13_04_flywheel_winprob_b` (**W_b**, `--seed 1002`) is the **token-exact seed replicate** of
`ai_v13_02_flywheel_winprob` (**arm W**, `--seed 1001`): the argv multiset differs by `1001 → 1002`
plus the run name, same pin `6eb9c776`, same declared dose, same `--ent-coef 0.05`, same eval
regime declared `source=argv`, both COMPLETE at 75,005,952
[ledger 2026-09-18 · *`ai_v13_04_flywheel_winprob_b` COMPLETE*, `32273e1a`].

**W_b exists to measure ONE thing: the 75M RUN-LEVEL FLOOR.** Until it landed, every row of the
flywheel pair read [`flywheel_pair_read_2026-09-15/`, `986ed5f5`] was floored by a bar imported
from a **10M four-node** depth, and registration §9.1 said in advance that no 75M floor existed and
none was affordable. This read produces W_b's run-end numbers **by exactly the code paths that
produced W's and S's** — same scripts, same harness, same seeds, same cells, same ports — and then
takes `|W − W_b|` per row as that row's floor.

🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19 and 22). There is no CI on
`|W − W_b|` itself: it is a single realisation of run-to-run variance, and the true floor may be
larger or smaller. A `|W − W_b|` that comes out small is **not** licence to call a smaller `|S − W|`
real — it is one draw from a distribution whose spread is unknown. Every verdict below is therefore
labelled **"at n = 2"**.

### What I already know, and must not pretend to have predicted

The completion ledger entry banked five live-frame / operational rows from the run's own TensorBoard
and child log. **These are known to me and my predictions on them are NOT blind**, and are marked
so wherever they appear:

| banked row | W | W_b | Δ |
|---|---|---|---|
| crossing | 4,128,768 | 4,128,768 | 0 |
| pool snapshots | 20 | 20 | 0 |
| `bots8` final | 0.9100 | 0.9212 | +0.0112 |
| H_end (med-20, from the completion read) | 1.0658 | 1.0468 | −0.0190 |
| G7 worst ratio | 1.131 | 1.160 | +0.029 |
| `critic_ece` (LIVE, last) | 0.0276 | 0.0149 | −0.0127 |
| `critic_skill` (LIVE, last) | 0.2539 | 0.2946 | +0.0407 |

🚨 **Rule 20 binds on the critic block above: a LIVE-frame lean is not evidence about the OFFLINE
population in either direction.** The offline `gate.*` family this read produces is a different
instrument on a different frame (full-capture, 9,600 battles, two seeded draws) and its floor may
not resemble the live one at all. The live rows are quoted as context and are never used as a bar.

Everything else below — the ladder refits, the node-by-node curves, the late slopes, the offline
critic rows, the untaught meter, the external anchors, the eval episode-length series — is **blind
at the time of writing.**

---

## 1. The rows, and how each is produced

Sign convention: **Δ = S − W** on the pair's own rows (as in the pair read), and the floor is
reported as **|W − W_b|**, unsigned, with the signed `W_b − W` printed beside it.

| # | row | instrument, reused verbatim |
|---|---|---|
| 1 | **ladder Elo** at 20 refit nodes, newest + second-newest | `fit_ladder(run_dir, first_n=20, write=False)` under the CURRENT recipe on BOTH sides (rule 24), plus the committed-vs-refit check and the recipe stamp `eval_sentinel_edges_dropped` |
| 1b | **node-by-node curves** W / W_b / S, and the COMMON-STEP refit | `fit_ladder(steps=COMMON)`; report the shared-step count and refuse to report below n = 12 |
| 2 | **late slopes**, newest node dropped | the pair read's `pair_strength_read.py`, unchanged |
| 3 | **H trajectory + H_end** | `main.ops.tb_read` on `train/entropy_loss`, sign-corrected, median-20, each arm's own post-crossing span |
| 4 | **`cond.opp_class_auc.t4_10`** + the gate/ECE family | `main.ops.critic_read` W_b (ARM) vs W (CONTROL), both win-prob critics — **a like-for-like pair, both columns the actual value function**, unlike the pair read's ROW A (a CRITIC against a DIAGNOSTIC). Two offline 800-game full-capture draws per arm, **seeds 20260910 / 20260911**, at each run's last evaluated step, matched distance from the end |
| 5 | **untaught meter** | `main.untaught_meter` against the registry name `untaught_meter_opponent`, seed 0, concurrency 1, both config resolutions, W_b and W in ONE invocation |
| 6 | **Metamon `SmallRL` greedy** — home (the pair's own `run_cell.sh`, team seeds `BASE_OURS 20260914` / `BASE_THEIRS 20270914`, port 9450) and away (`python -m main.anchors`, tier B) | 100 games each, two role-balanced half-cells, regime verified per decision |
| 7 | **Foul Play** @ `--search-time-ms 1000`, 80 games over the same 8 pinned pool teams, port 9417 | with the **realized visits/decision** recorded per session — rule 23, a WIDTH meter |
| 8 | **the G7 excursion at ~30M**, both arms | the eval episode-length series, read as a DESCRIPTOR |

Everything CPU-only (`CUDA_VISIBLE_DEVICES=""`), `nice`, `OMP_NUM_THREADS=1` for peers,
`POKESIM_SIM_BRIDGE_BIN` pinned; ports 9400–9499 plus `main.anchors`'s own 9500-range server; every
`--out` under the job tmp, never the cwd; nothing written under `models/`; **:8000 and :8001 are
never touched**, and the live `ai_v13_05_exploit_big5starmie` arm is not touched.

---

## 2. 🚨 THE DECISION RULE, fixed here

For each of the **three pre-designated S-vs-W findings** and for **strength**:

> A finding is **"OUTSIDE THE 75M FLOOR"** if and only if
> **(a)** `|S − W| > |W − W_b|` **AND**
> **(b)** the `S − W` CI **excludes the point** `|W − W_b|`.
> Otherwise it is **"WITHIN FLOOR at n = 2"**.

Three clauses bind on top of it and none may be dropped:

* **Never "equivalent"** (rule 6). WITHIN FLOOR is *not* equivalence: equivalence needs the DELTA's
  own CI inside a bar, and a one-pair floor is a point, not a bar.
* **No CI on the floor** (rule 19). `|W − W_b|` is a single realisation. A finding that clears it at
  n = 2 is a CANDIDATE that has survived ONE replicate, never a family verdict.
* **The realized-dose gap still confounds S.** Arm S took **1.44×** arm W's realized dose
  (`lr_median` 4.32e-4 vs 3.00e-4, dose_rate 6.592e-08 vs 4.578e-08 — hazard H-F). W and W_b are
  matched on the treatment AND (blind) presumably close on realized dose, so the floor is measured
  on a pair that does NOT carry that gap while the finding it is applied to DOES. **Clearing the
  floor therefore does not clear the dose confound**, and no row may be read as if the optimiser had
  been held fixed.

---

## 3. MY PREDICTIONS — branch by branch, with the prior each reads against

### 3.1 Strength — PREDICT: **WITHIN FLOOR**

**Prior:** rule 22's own evidence — *"the two same-pin seed-replicate pairs in hand differ by 83 and
58 Elo (CIs clear of zero) against a control-pair difference of 10."* Those were 10M arms, so the
transfer to 75M is itself an assumption; against it, the pair read's within-run adjacent-node spread
on arm W ran to **56.9** at its largest single swing.

**I predict `|W − W_b|` at the newest refit node exceeds `|S − W| = 17.5`**, i.e. strength reads
WITHIN FLOOR at n = 2, reinforcing the pair read's NOT DETECTED rather than overturning it. I put
this at ~70/30. The branch that would surprise me: `|W − W_b| < 10`, which would say run-to-run Elo
variance at 75M/20 nodes is far tighter than at 10M/4 nodes — a genuinely new fact about the
instrument, and one that would make the +17.5 a live candidate rather than noise. **Either way the
+17.5 cannot become a claim here**, because the 45.0-Elo imported floor and the ±24.4 CI are
unchanged by this read: (b) would still have to hold.

### 3.2 Entropy — PREDICT: the **TRAJECTORY finding SURVIVES**; the H_end LEVEL was never a finding

**NOT BLIND on H_end**: the ledger banks 1.0658 (W) vs 1.0468 (W_b), **Δ −0.0190 nats**. I therefore
predict nothing about it and simply record that this is a quarter of the imported 0.074-nat
three-seed floor — the first 75M evidence that the 10M floor is not wildly wrong for this row.

**BLIND on the trajectory**, which is the actual finding (§8.3 of the registration; arm S flat at
−3e-5 ± 5e-5 nats/M over 71M steps, arm W decaying at −1.36e-3 ± 6e-5, t = −23.6). **I predict W_b
DECAYS like W** — same objective, same coefficient — with a post-crossing slope within ~30 % of W's,
so `|slope_W − slope_W_b|` lands far below `|slope_S − slope_W| ≈ 1.33e-3` and the FLAT-vs-DECAYING
contrast reads **OUTSIDE THE 75M FLOOR**. ~85/15. The branch that would kill it: W_b coming out flat
or decaying at a third of W's rate, which would say the trajectory is a seed draw and the pair
read's cleanest localisation is not one.

### 3.3 The untaught meter — PREDICT: **genuinely uncertain, lean OUTSIDE**

**Prior:** the meter's floors are all **fold** floors at ~1M depth (1.19 / 1.66 / 4.27 pp) and
neither arm is a fold, so this read produces the first RUN-level floor the meter has ever had at any
depth. The pair's Δ is **+8.31 pp [+5.69, +11.19], 8 of 8 teams.**

**I predict `|W − W_b|` lands between 2 and 6 pp** — above the tightest fold floor, below the pair's
own 8.31 — so the finding reads OUTSIDE THE FLOOR on clause (a) and the verdict turns on clause (b),
whether the [+5.69, +11.19] CI excludes the floor point. I put OUTSIDE at ~55/45, which is close
enough to a coin flip that I am registering it as **the row this read is most likely to overturn.**
If `|W − W_b| > 5.69` the finding falls to WITHIN FLOOR outright, and the largest and cleanest
movement in the whole pair read becomes a seed.

### 3.4 Calibration / ECE — PREDICT: **OUTSIDE, but the caveat survives untouched**

**NOT BLIND in the live frame**: `critic_ece` moved 0.0276 → 0.0149 between the seeds, −46 % of its
own value. 🚨 **Rule 20 forbids reading that as evidence about the offline population**, and the
offline instrument here is a different frame with ~50× the battles.

**BLIND on the offline `gate.ece.all`.** The pair's Δ is **0.0688 (W) vs 0.0178 (S) = 0.0510**, and
the eval-draw spreads on that row were tiny (0.0008–0.0028) against a 0.0245 imported floor. **I
predict the offline `|W − W_b|` on `gate.ece.all` lands in 0.005–0.020**, i.e. materially below
0.0510, so the ECE finding reads **OUTSIDE THE 75M FLOOR** at ~70/30. 🚨 **And clearing the floor
changes NOTHING about the finding's standing caveat:** the pair's ECE row is **a CRITIC against a
DIAGNOSTIC** (arm S's auxiliary win-prob head at coef 0.05 vs arm W's value function), and no floor
can convert it into a critic-vs-critic statement. The W-vs-W_b pair, by contrast, is
**critic-vs-critic on both sides** — which is exactly why it is a clean floor for the row and not a
substitute for the missing comparison.

### 3.5 The registered critic guard `cond.opp_class_auc.t4_10` — no branch is pre-designated

The pair read returned NOT DETECTED on ROW A (+0.0254 against a 0.02451 imported floor) and NOT
CONFIRMED on ROW B. It is reported here with its new floor for completeness; **no verdict flips on
it either way**, because a row that did not clear an imported floor cannot be rescued by a measured
one. **I predict `|W − W_b|` on this row lands at or above the 0.0245 import** — my reason being
that the pair read's own per-arm eval-DRAW spread was already 0.0104–0.0177 on this row, and the
run-level component sits above the draw-level one (rule 19).

### 3.6 The anchors — PREDICT: **no verdict moves**

Metamon `SmallRL` at n = 100 resolves ~±10 pp at best and the SOP's own three-seed RUN-level floor
on the away cell is **0.110** [SOP amendment 2026-09-16]. The pair's anchor deltas were −0.020 /
+0.020 / +0.080, all NOT DETECTED. **I predict every anchor row stays NOT DETECTED and that
`|W − W_b|` on the away cell lands inside 0.110**, corroborating the SOP's floor at 75M rather than
moving anything. Foul Play is a **WIDTH meter** (rule 23) and its campaigns are not width-matched;
**I predict W_b's realized visits/decision differ from W's 1.153 M by more than the win-rate gap can
survive**, so that row is reported as a width comparison and **no model claim is taken from it under
any outcome.**

### 3.7 The G7 excursion — a DESCRIPTOR, and it is already banked as reproduced

**NOT BLIND**: the completion entry states the excursion reproduced (W 1.047 → 1.087 → 1.111 at
~30M, back to 0.989 by 54M; W_b worst 1.160 at ~30M, 0.993 at the end). What is blind is the
**eval episode-length series** underneath it, which is what §3 of this read actually produces: the
step range, the magnitude, and whether the two arms' bumps sit on top of each other. **I predict the
same step window (≈26–34M) and a comparable magnitude on both**, reported as a descriptor with no
bar — a shape, never a verdict.

---

## 4. What will NOT be claimed, under any outcome

* **That the 75M floor is now KNOWN.** One pair bounds it. Two runs is n = 2.
* **That a row clearing the floor is a family verdict.** It is a candidate that has survived one
  replicate — and on the three findings it is a candidate that *still carries the 1.44× realized-dose
  confound*, which W_b does not measure.
* **That WITHIN FLOOR means the two critics are equivalent.** Rule 6.
* **That the ECE finding is critic-vs-critic.** It is not, and no floor makes it so.
* **Anything about the critic objective in general, at 277M, in a fold, or after a distillation.**
* **Any Foul Play difference attributed to a model** while the widths are unmatched (rule 23).
* And never the circular *"⅔ of losses are team-draw, so they are unwinnable"* — those losses are
  HEADROOM.

---

*Nothing in `ledger.md`, `UNDERSTANDING.md` or `flywheel_era_pair_2026-09-12.md` is edited from this
directory. The ledger paragraph is drafted as the last section of `README.md`, ready to append.*
