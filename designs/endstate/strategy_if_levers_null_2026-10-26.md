# If D3, N1-P and X59-S all come back without a GAIN: the plan for goal (b)

**Status: DRAFT (2026-10-26), explicit-only. Nothing is built, registered or launched.** It is a contingency plan
for the MODAL outcome of the three reads in flight. Goal (b), its meter and every read it cites are
[`design_beat_metamon_path.md`](design_beat_metamon_path.md)'s. The levers N1–N5 are the survey's
([`sota_survey_2026-10-26.md`](../research_state/literature/sota_survey_2026-10-26.md) §5). Search is
[`search_ledger.md`](search_ledger.md) and [`design_search_next.md`](design_search_next.md). The from-scratch breadth
run is [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md). The piloting-gap analysis being written in
parallel was NOT read here (it had not landed); §4's tree routes on its outcome classes, not its numbers.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = X26's plateau
(`rb_x26_s2601` @280M); C0 = X35's plain-continuation control; U = the three unfamiliar panels pooled (goal (b)'s
meter); F = `forum72`; S = in-house strength; GPU-h = hours of our local card; lane-h = one CPU lane for an hour;
**GUESS** = an unmeasured effect; **ESTIMATE** = a cost from measured rates; **MY JUDGEMENT** = an inference no read
states.

---

## 0. One screen

- **The scenario is the likeliest one.** The registered priors give N1-P a GAIN with 0.20 and D3 with 0.15; X59-S's
  own prediction for Δ_data is −1 pp (−6 to +4), so HUMAN-DATA-BETTER is perhaps 0.15 (GUESS). If they are independent,
  **all three miss a GAIN with probability ≈ 0.58**.
- **What "null" would then mean.** Every data-side lever tried at the plateau reads ≤ ≈ 2 pp: breadth at 10³ and
  10⁴ teams, set mutation, practice, belief fixes (U +1.5 / −1.5 pp, n.d.), a fresher or older pool, and 100M more
  steps (X26c FLAT). Four explanations remain, and the next work should separate them rather than add a fifth lever of
  the same kind:
  1. **plasticity**: the plateau's representation, formed on 719 teams, no longer bends;
  2. **capacity**: 2.03M parameters cannot hold team-conditional piloting broadly;
  3. **resolution**: the levers DO help by +1–2 pp each, and a ±4 pp read cannot see it;
  4. **the meter**: Kakuna trained on 10⁴–10⁵ teams that likely overlap our "unfamiliar" panels.
- **The top three by expected pp per GPU-h** (§3), each after a zero-GPU gate:
  1. **N5, an offline breadth student** on X59-S's machinery: +1.5 pp (−1 to +5), ≈ 8 GPU-h, ≈ 2–4 agent-days and
     ≈ 75–115 lane-h. Highest per GPU-h, widest uncertainty, biggest build.
  2. **N2, a training-only mask on the own-team set channel**: +1.5 (0 to +3), ≈ 12.3 GPU-h at look 1 and ≈ 1.5
     agent-days, only if the exact-set sensitivity probe does not kill it.
  3. **A breadth STACK fork**: the union of the forum, set-mutation and procedural corpora at the plateau. +1.0 (−1.5
     to +3.5), ≈ 12.3 GPU-h and ≈ 0.5–1 agent-day, only if ≥ 2 of D1 / D3 / N1-P end UNDECIDED with Δ̂ > 0 (D1
     already has, at +2.10).
- **The decision tree** (§4) spends a zero-GPU week first: the N2 and N3 probes, a plasticity probe and search's E2
  bridge. It then routes to the plasticity branch (from-scratch breadth) or the capacity branch (width), with the
  stack and N2 forks running meanwhile. **It escalates to the owner at ≈ 90 GPU-h without a GAIN.**
- **The only bet whose expected value alone reaches the ≈ 4.3 pp gap** is the scale bet: a wider network trained from
  scratch on 10⁴ teams to its plateau, two seeds, ≈ 150–250 GPU-h. It is last per GPU-h. It is the owner's call, and
  realistic only with a second card.

---

## 1. What each in-flight read leaves open if it misses

| read | what a miss looks like | what it rules out | what it does NOT rule out |
|---|---|---|---|
| **D3 look 2** (X41, set mutation; look 1 CONTINUE: H +1.27 [−2.63, +5.16], P −0.24 NON-INFERIOR) | UNDECIDED (registered 0.67) or NULL (0.15) | an exact-set-identification fix worth ≥ ≈ 3 pp at 0.3 share, 20M, at the plateau | a +1–2 pp effect (and P costs nothing, so it is a free stack member) |
| **N1-P** (12,000 procedural teams at the plateau; trained, read pending) | NULL (0.30) or UNDECIDED (0.47) | "N1 at the plateau", at this share and 20M (its PREREG §6) | N1 from step 0 (plasticity), a human-team corpus, the toy screen |
| **X59-S** (three matched students, training now) | NOT-RESOLVED on Δ_data; UNCLEAR on Δ_labels | nothing alone: its UNCLEAR routes to **H-TEAMS first with the D1 contrast**; only SELF-PLAY-DATA-BETTER, with Δ_labels not HUMAN-LABELS-BETTER and Δ_states' LB ≤ 0, closes human data at this volume | the human prior at a larger volume; offline RL (FR-3: on the same human data offline RL beats imitation by ≈ 16 pp) |

Already closed for goal (b), and not re-opened by this plan:
- **The belief channel.** The prior mix moved no Kakuna games at λ = 0.25 or 1 (`belief_prior_mix_2026-10-26/`),
  consistent with X37a putting the gap in PILOTING, not FACING.
- **More plain training** (X26c FLAT, Δ +0.08 [−3.90, +4.06]); the archive and SPRT pools (NULL); exploiters and
  X49 (they serve facing or exploitability); a different opponent (PastaMittens ≈ Kakuna); training against Kakuna
  (REJECTED: it trains on the meter).
- **"Human data alone beats us".** P\* beats every human-only Metamon model (U 0.631–0.790); the human prior is
  only a candidate as an ADDITION to self-play.

---

## 2. The four explanations, and the cheapest read that separates them

| explanation | predicts | the cheapest separating read | cost |
|---|---|---|---|
| **Plasticity** | the plateau resists new data that a young network would absorb | **the plasticity probe**: on one frozen bank of unfamiliar-team states, X26 @25M, @100M and P\* (all banked): the fraction of near-dormant trunk units, the effective rank of the trunk's features, and how fast a short CPU fine-tune on held-out-team states lowers the loss from each checkpoint. Loss of plasticity predicts all three worsening with steps (MY JUDGEMENT, from the continual-learning literature on dormant units and plasticity loss; not re-fetched in this unit) | ≈ 1 agent-day, ≈ 5–10 CPU-h, 0 GPU |
| **Capacity** | a wider network pilots unseen teams better at equal data | the N4 toy screen (fresh toy at 2× width vs the toy, matched wall clock); a build first, because `D_MODEL` is a constant in `src/agents/model/arch_constants.py` | ≈ 1.5–2 agent-days + ≈ 13–15 GPU-h |
| **Resolution** | each lever is +1–2 pp; combined they add up | **the stack fork** (§3 rank 3), read once on F and U | ≈ 12.3 GPU-h |
| **The meter** | Kakuna is familiar with our "unfamiliar" panels | `forum216b` (sealed, X54) and the reverse cell; the panel-validity read found no panel shown Kakuna-familiar, and dropping H LOWERS P\*'s score | already read for the panels; the open part needs downloads that were not made |

**One fact cuts against capacity, weakly.** The 1.42M toy T\* drops home72 → F half as far as P\* on the logit scale
(0.14 vs 0.28, NOT DETECTED; X51). A smaller network that memorises less is what capacity-as-memorisation predicts,
not capacity-as-generalisation. Procgen predicts the reverse. Neither reading is supported at that resolution.

**Resolution is the explanation most likely to be true and least likely to be tested by another single lever**
(MY JUDGEMENT). D1 read +2.10, D3 +1.27, X52 unpractised +1.3, the λ = 0.25 belief mix +1.50: four point estimates
between +1 and +2 pp, none resolvable by a read whose interval is ≈ ±4 pp. A fifth lever read alone will look the
same. The stack fork reads their sum.

---

## 3. The levers, ranked by expected pp per GPU-h

"Expected pp" is the gain on U against Kakuna over P\* (or over the matched control), **conditional on all three
misses** (each GUESS is lower than the survey's unconditional one where the misses bear on it). "GPU-h" is to the
lever's first decisive read. A GPU-free lever is ranked by CPU and agent-days instead, below the table.

| rank | lever | expected pp (GUESS, 80 %) | GPU-h to the read | CPU lane-h | agent-days | pp per 10 GPU-h | its gate |
|---|---|---|---:|---:|---:|---:|---|
| **1** | **N5: offline breadth student.** A CPU self-play corpus by P\* and its archive on ≥ 10⁴ teams; a student from P\* trained by advantage-filtered BC plus a critic (Metamon's recipe) on it; breadth costs CPU battles, not on-policy GPU steps | **+1.5 (−1, +5)** | ≈ 8 (5–10) | ≈ 75–115 (corpus ≈ 60–100 + reads ≈ 15) | 2–4 (X59-S's student trainer exists; the corpus writer, the filter and the critic are new) | ≈ 1.9 | X59-S's students trained cleanly (its VOID guards); a CPU measurement of P\*'s forward rate on the corpus box (UNVERIFIED) |
| **2** | **N2: set-channel noise.** In training only, each own mon's set fields are masked or replaced by a usage draw with probability p; play is noise-free | **+1.5 (0, +3)** | ≈ 12.3 (look 1; ≈ 20.5 final) | ≈ 44 | ≈ 1.5 (probe 0.5 + flag 1) | ≈ 1.2 | the **exact-set sensitivity probe** (zero GPU): pool-team KL ≥ 1.5× unfamiliar-team KL. KILL if the ratio's interval covers 1.0 and excludes 1.5. A D3 miss lowers it (D3 attacks the same identification) |
| **3** | **The breadth stack.** One union corpus of `w_forum_v1@ABC`, `d3_setmut_v1` and `n1_proc_v1` (re-checked for panel exclusion), at the plateau, D1's frame and rule | **+1.0 (−1.5, +3.5)**, sub-additive | ≈ 12.3 | ≈ 44 | ≈ 0.5–1 (a union corpus and its pin: `--team-mix` takes one corpus) | ≈ 0.8 | ≥ 2 of D1 / D3 / N1-P final UNDECIDED with Δ̂ > 0. P's cost may compound (D1's −1.6 pp): P is co-primary |
| 4 | **H-TEAMS (X60).** Human ladder teams in the pool, the human-data toggle; X59-S's UNCLEAR route | +1.0 (−1, +4) | ≈ 12–20 | ≈ 44 | ≈ 2–3 (X60's four fixes; a team corpus) | ≈ 0.6 | X59-S not closing human data (§1); X60 READY; LOCAL only (no human data on a rented box) |
| 5 | **N3: a regret-scored team curriculum** (PLR over teams, ACCEL edits) | +0.5 (0, +3) | ≈ 12.3 | ≈ 44 | ≈ 2–2.5 | ≈ 0.4 | the zero-GPU probe: Spearman ρ of a team's mean positive advantage on P\*'s traces against its Kakuna piloting deficit; KILL if ρ < 0.2 with its interval covering 0. Its value grows only on top of a breadth corpus |
| 6 | **Fresh breadth without N1-P's GO** (X53-N1 Stage A as designed) | +1.5 (−1, +5) | ≈ 39 | ≈ 54 | ≈ 1 | ≈ 0.4 | N1-P UNDECIDED with Δ̂ > 0 **and** the plasticity probe shows loss |
| 7 | **N4: width from scratch.** The toy screen first, then production | toy: a screen; production +1.5 (−1, +4) | toy ≈ 13–15; production ≈ 40–60 + a fresh control (CF, shared with X53) | ≈ 30–54 | ≈ 1.5–2 (a width build) | ≈ 0.3 | the plasticity probe shows NO loss (the capacity branch) |
| 8 | **The scale bet:** N4 × N1 from scratch to the plateau, two seeds | **+3 (−1, +8)** | ≈ 150–250 | ≈ 150 | ≈ 3 | ≈ 0.15 | the N4 toy screen GO; a second card; the owner's word |

**GPU-free levers**, ranked by CPU and agent-days:
- **Search, E2 → E1 → E4** (`design_search_next.md` §3). At P\*, L1's anchored search reads +0.0001 [−0.0097, +0.0101]
  per decision; E1-0's real-bot referee reads UNDECIDED. But +4.6 pp in games needs only ≈ +0.0018 per decision
  (the performance-difference bridge), below every interval measured. **E2 (the bridge) is the cheapest open
  question in this tree:** ≈ 0.5 agent-day and 25–35 lane-h. E4, a live player against Kakuna, is gated on E1–E3
  and costs ≈ 300–470 lane-h. Expected for goal (b): +0.5 (−1, +2.5) (GUESS).
- **An ensemble of our checkpoints at play:** +0.5 (0, +2), ≈ 1 agent-day and ≈ 10 lane-h. Its members trained on
  the same teams, so it cannot fix piloting breadth.

**Not ranked**, with the reason:
- **Test-time adaptation** (practise the handed team for ≈ 0.6 GPU-h): it is a deployment REGIME, never the
  generalist's number. It stays an owner option at the exit (§5).
- **A strategy latent (AlphaStar's z):** no Pokémon evidence, a multi-day build.
- **High-temperature collection:** it needs an off-policy correction in PPO. N5's corpus reaches it.

**Honest uncertainty.**
- Every expected pp above is a GUESS, and the intervals overlap almost completely: the ranking separates costs
  better than it separates effects.
- No row's point estimate alone reaches +4.3 pp except the scale bet's. The plausible route to goal (b) is a sum of
  two or three +1–2 pp gains, read together.
- The probability that any one lever of rank 1–7 reads GAIN on its own registered rule is ≈ 0.10–0.25 each (GUESS,
  in line with the D1 / D3 / N1-P priors).

---

## 4. The decision tree

```
T0  the three reads land (D3 look 2 · N1-P look 1 [+ look 2] · X59-S)
 ├─ any GAIN ─────────► its own licence (N1-P ⇒ X53-N1; D3 ⇒ adoption path; X59-S ⇒ X60). This doc does not apply.
 └─ none ─────────────► T1
T1  THE ZERO-GPU WEEK (≈ 2.5–3 agent-days, ≈ 40–60 lane-h, run in parallel; the GPU keeps its queue)
      a. N2's exact-set sensitivity probe
      b. N3's PLR-signal probe
      c. the plasticity probe (§2)
      d. search's E2 bridge read
      e. the piloting-gap analysis (the parallel unit), if it has landed
      f. the stack's eligibility: count D1 / D3 / N1-P finals UNDECIDED with Δ̂ > 0
T2  THE CHEAP FORKS (≈ 12.3 GPU-h each at look 1, D1's frame against the shared C0)
      f ≥ 2 ──────────► the STACK fork (rank 3)
      a not KILL ─────► the N2 flag, then its fork (rank 2)
      X59-S UNCLEAR ──► H-TEAMS, once X60 is READY (rank 4; local)
      b passes ───────► N3 waits for a breadth corpus that has read positive
T3  THE BRANCH ON T1c
      plasticity LOSS ──► fresh breadth (X53-N1 Stage A, ≈ 39 GPU-h; rank 6)
      NO loss ──────────► the N4 toy screen (≈ 13–15 GPU-h; rank 7); GO ⇒ production width with a fresh control
      UNCLEAR ──────────► both, the toy screen first (cheaper)
T4  N5 (rank 1) as soon as its build lands. It is GPU-light, so it runs beside T2 / T3 and never waits for them
T5  E1 → E4 (search), only if E2 shows the bridge holds and an arm predicts ≥ +4.6 pp. CPU only
T6  THE EXIT: ≈ 90 GPU-h spent in T2–T4 with no GAIN (≈ 4 days of our card; ≈ 2 with a second card), or every
    branch exhausted ⇒ escalate to the owner (§5)
```

**Routing on the piloting-gap analysis** (by outcome class; the analysis owns its numbers):
- the gap concentrates in identifiable decision classes (switching, set-up, item-dependent lines) ⇒ N3 with a
  targeted score, and a targeted D3-style corpus for those mons, move ahead of N2;
- it concentrates in team identity (familiar sets read as a signature) ⇒ N2 moves to the top of T2;
- it is spread evenly over decisions ⇒ breadth and capacity (T3) carry the hope, and the stack moves first;
- it localises to the critic (value errors on unfamiliar teams, not policy errors) ⇒ the amortisation question in
  `design_beat_metamon_path.md` §4 is re-opened before any T2 fork.

**Why this order.**
- T1 costs no GPU and can kill N2 and N3 outright, and point T3 one way.
- T2's forks run in the X35 frame, whose control C0 is banked and shared, so each costs only its own arm.
- T3 is the expensive split, and it waits for the probe that chooses the side.
- N5 is first per GPU-h but last to be ready, so it runs beside the forks rather than ahead of them.

---

## 5. The exit: what the owner is asked if the tree ends without a GAIN

1. **Re-scope goal (b)?** The deployment-shaped headline L (we bring our teams, Kakuna brings H) already reads 0.630
   [0.573, 0.686], and goal (a) is met. Goal (b) is the harder, team-neutral question (`design_beat_metamon_path.md`
   §1).
2. **Adopt test-time adaptation as the deployed regime?** Practise the team you are handed (≈ 0.6 GPU-h per team);
   reported as a regime, never as the generalist's number.
3. **Fund the scale bet** (≈ 150–250 GPU-h, rank 8) with a second or bought card?
4. **Or return the GPU** to the other standing goals (search, the AlphaStar-style league), leaving goal (b) open
   with the evidence above.

---

## 6. The second GPU in this plan

It is not a lever; it is a rate. The cloud plan prices a rented 5090 at ≈ $0.38 per local GPU-h of work, at ≈ 1.65×
our card ([`../ops/cloud_gpu_plan.md`](../ops/cloud_gpu_plan.md) §3).
- **T2's forks** go on the cloud only behind the cloud control C0v (≈ 12 GPU-h, ≈ $5, the card rule §1.2). The
  first cloud fork pays for it.
- **Fresh pairs** (T3's X53-N1 or a wide arm) bring their own control and upload no checkpoint, so they are the
  natural cloud workload (the X53-N1 doc §9).
- **H-TEAMS and every human-data arm stay local.**
- With two cards the tree's GPU half takes ≈ 2 days instead of ≈ 4. The CPU reads (≈ 44 lane-h a fork look) then
  become the bottleneck on rule 5b's three agent cores, as the cloud plan's FINDING says.

---

## 7. FINDINGS (standing rule 7)

1. **The scenario's probability assumes the three reads are independent.** D3 and N1-P share the same frame,
   control and meter, so their misses are positively correlated, and ≈ 0.58 is if anything low.
2. **The piloting-gap analysis was not read** (it had not landed). §4 routes on its outcome classes only.
3. **The plasticity probe is new and unbuilt.** Its literature basis is MY JUDGEMENT from memory, not re-fetched or
   verified in this unit, and none of its three measures has a registered bar yet.
4. **N5's corpus cost rests on an UNVERIFIED CPU forward rate** for P\* at corpus scale. X59-S's measured
   ≈ 2.6 ms per forward is the nearest figure.
5. **A union corpus for the stack is a new registered corpus**, because `--team-mix` takes one
   (`agents.training.matchup_spec.parse_team_mix`). Its panel exclusion must be re-derived on the union, not inherited.
6. **Every expected pp here is a GUESS**, and each read's resolution (≈ ±4 pp) is wider than most of them.
7. This unit ran no game and used no GPU.

---

## 8. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT)** (research-strategy worker, orchestrator brief) | a contingency plan for D3, N1-P and X59-S all missing a GAIN: four explanations (plasticity, capacity, resolution, the meter) and the cheapest read separating each; a ranking by expected pp per GPU-h (N5, N2, the stack first; the scale bet last but the only one whose expected value alone reaches the gap); a tree that spends a zero-GPU week first (N2 / N3 / plasticity probes, search's E2), then cheap forks, then a branch on plasticity, with an owner escalation at ≈ 90 GPU-h | another single lever of D1's kind read alone (a ±4 pp read cannot see +1–2 pp); more plain training (X26c FLAT); belief fixes (the prior mix moved no games); FACING levers; the scale bet first (≈ 150–250 GPU-h before the cheap separations); test-time adaptation as goal (b)'s number (a regime) | `design_beat_metamon_path.md` §0–§6; `sota_survey_2026-10-26.md` §5; `belief_prior_mix_2026-10-26/README.md`; `human_only_vs_pstar_2026-10-25/`; `x59s_matched_students_2026-10-25/PREREG.md` §5–§6; the D3 and N1-P PREREGs; `design_search_next.md` §3; `cloud_gpu_plan.md` |
