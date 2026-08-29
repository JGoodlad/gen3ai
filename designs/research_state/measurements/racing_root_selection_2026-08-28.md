# RACING root-action selection vs the fixed grid — 2026-08-28

**Registered prediction (ledger `f5b7da5`): ≥2× budget reduction at matched decision quality.**

**Verdict: the middle branch — a smaller but REAL gain.** On the primary axis (the per-decision
*deadline* a search must be granted) the measured reduction is **1.08× / 1.41× / 1.47×** at 80% /
90% / 95% agreement with a large-budget gold argmax. On the *spend* axis (what the decision
actually consumed, since a race hands the remaining clock back) it is **1.09× / 1.56× / 1.87×**, and
at the top of the swept range racing reaches **1.000** gold agreement for **2.40× less spend** than
the grid needs to reach **0.989**. So ≥2× is real at the high-quality end and on the spend axis, and
is not reached on the deadline axis at any swept quality level. The registered 2× is **not met as
stated**; the lever is positive and worth keeping.

Two findings landed that the prediction did not ask for and that matter more than the ratio:

1. **The game's decisions are BIMODAL, not gradual.** Under the shipped rule, **52.2%** of root
   decisions never separate within 32 paired samples at all; of the 47.8% that do, the **median is
   exactly the minimum-samples floor** (5 of 32) and 47 of 86 separate on the first round they are
   allowed to. There is almost no middle. Racing saves nothing on the majority half — that mass is
   where a *time manager* should cap the decision rather than where a better allocator should try
   harder.
2. **The fixed grid at the production budget is badly under-resolved, and that reframes the whole
   search-dividend battery.** At a **1 s** cell — the battery's own default — the grid agrees with
   its own large-budget argmax on only **86.1%** of decisions. At 0.5 s, **80.6%**. Roughly one in
   seven "searched" decisions in every battery cell run to date is not the decision that search
   would make with more samples. A null dividend measured through that much allocator noise is
   weaker evidence than it looked.

Data: `racing_root_selection_2026-08-28.json` (the shipped `seq` rule) and
`racing_root_selection_2026-08-28_zrule.json` (the registered `z` rule).
Code: `src/main/search_dividend/racing.py`, `ab_racing.py`.

---

## What was measured

180 recorded `move_selection` decisions with ≥3 legal actions, drawn from 23 battles of
`models/ai_v9_66_R3F6d_0828/eval_traces/step_30000000` (current architecture, `arch_signature =
gen3_critic_route_wave_v1`), turns 1–48, **7.21 legal actions on average** (range 4–9). Every
decision scored on the **win-prob** head, `honest` arm (belief-determinized worlds), `m_opp = 4`,
depth 1.

**The design is a BANK-AND-REPLAY, and that is what makes it a paired comparison rather than a
matched one.** Phase 1 draws 32 CRN-paired rounds per decision (one determinized world + one freshly
minted dice seed, α-marginalized inside) and records the per-action value vector each round produced
— allocator-blind, every action on every round. Phase 2 replays BOTH allocators over that one bank:
the grid consumes rounds `0..n-1` scoring everything, the racer consumes the same rounds in the same
order but only its live entries. At every budget point the two arms see byte-identical samples, so
the difference between them is allocation and nothing else. A whole budget sweep then costs no sim
time, which is why the parameter sweep below was affordable.

**Cost is priced in two units and the report carries both.** `budget_s` is the deadline that would
have to be reserved; `spend_s` is what was consumed. They coincide for the grid, which always spends
its whole budget, and diverge for racing, which stops the moment the field collapses. The constants
are **measured during banking, not assumed**: `open_root` = **1.9 ms** and one action-slot =
**14.5 ms**. The near-free open is a rust-search-driver fact and it matters — it is the term that
would otherwise punish racing for running more, cheaper rounds.

### The gold reference, and its own noise

Gold = the grid's argmax over all 32 rounds. **Doubling check: 11 of 180 argmaxes (6.1%) move
between 16 and 32 rounds.** The registration asked for <2% and it is not met — 32 paired samples is
not enough to pin a gen-3 root argmax. That bounds every number below: an allocator that reproduced
the *true* argmax perfectly would still score somewhere near 0.94–0.97 against this gold, so
agreements in the high nineties are at the gold's own resolution and the arm-vs-arm **difference**
is the reportable quantity, not the absolute level. Closing it would need ~4× the bank (the sampling
error falls as 1/√n), which is a 1.5 h run rather than the 22 min this one took — worth doing before
any decision rests on the absolute agreement rate.

---

## The accuracy-vs-budget table

`seq` rule (the shipped default), δ=0.05, floor 5. `spend` is seconds actually consumed; `arms` is
action-slot evaluations; `reg` is mean win-prob regret against gold; `nd` is how many of the 180
decisions disagreed with gold.

| budget s | grid agree | grid spend | grid arms | grid reg | race agree | race spend | race arms | race reg | race nd |
|---|---|---|---|---|---|---|---|---|---|
| 0.103 | 0.782 | 0.086 | 5.8 | 0.00895 | 0.782 | 0.086 | 5.8 | 0.00895 | 19 |
| 0.142 | 0.700 | 0.111 | 7.5 | 0.00566 | 0.700 | 0.111 | 7.5 | 0.00566 | 54 |
| 0.194 | 0.706 | 0.136 | 9.2 | 0.00488 | 0.706 | 0.136 | 9.2 | 0.00488 | 53 |
| 0.266 | 0.794 | 0.228 | 15.4 | 0.00253 | 0.794 | 0.228 | 15.4 | 0.00253 | 37 |
| 0.364 | 0.794 | 0.320 | 21.6 | 0.00210 | 0.794 | 0.317 | 21.5 | 0.00210 | 37 |
| 0.499 | 0.806 | 0.437 | 29.6 | 0.00229 | 0.817 | 0.433 | 29.3 | 0.00211 | 33 |
| 0.684 | 0.872 | 0.630 | 42.7 | 0.00145 | 0.878 | 0.613 | 41.4 | 0.00138 | 22 |
| **0.937** | **0.861** | 0.888 | 60.2 | 0.00144 | **0.922** | 0.772 | 51.9 | 0.00047 | 14 |
| 1.284 | 0.928 | 1.214 | 82.3 | 0.00078 | 0.950 | 0.977 | 65.4 | 0.00021 | 9 |
| 1.760 | 0.944 | 1.701 | 115.3 | 0.00013 | 0.967 | 1.150 | 76.8 | 0.00007 | 6 |
| 2.411 | 0.972 | 2.342 | 158.7 | 0.00004 | 0.994 | 1.227 | 82.0 | 0.00000 | 1 |
| **3.304** | **0.989** | **2.993** | **202.9** | 0.00005 | **1.000** | **1.247** | **83.3** | 0.00000 | 0 |

**Budget-to-reach, and the reduction ratio — the headline:**

| target agreement | grid | racing (`seq`) | **reduction** | racing (`z`) | reduction |
|---|---|---|---|---|---|
| 80% — deadline | 0.432 s | 0.398 s | **1.08×** | 0.379 s | 1.14× |
| 90% — deadline | 1.139 s | 0.810 s | **1.41×** | 0.825 s | 1.38× |
| 95% — deadline | 1.891 s | 1.284 s | **1.47×** | never | — |
| 80% — spend | 0.378 s | 0.346 s | 1.09× | 0.312 s | 1.21× |
| 90% — spend | 1.079 s | 0.693 s | 1.56× | 0.473 s | **2.28×** |
| 95% — spend | 1.830 s | 0.977 s | **1.87×** | never | — |
| top of range | 0.989 @ 2.99 s | **1.000** @ 1.25 s | **2.40×** (arms 2.43×) | 0.933 @ 0.58 s | — |

Three things this table says that the single ratio does not:

* **Below ~0.4 s the two arms are IDENTICAL.** The floor has not been reached, so nothing has been
  eliminated and racing *is* the grid. Racing cannot lose at small budgets and cannot win there
  either — a useful safety property, and the reason there is no risk in defaulting a small cell to it.
* **The 0.142–0.194 s rows are NON-MONOTONE** (0.782 → 0.700 → 0.706). At one or two rounds the
  argmax is close to a coin flip and the third round changes it; the dip is the sampling noise of a
  1-round estimate, not a defect of either allocator, and it is the same for both by construction.
* **Regret separates the arms harder than agreement does.** At 0.937 s the agreement gap is 6.1 pp
  but the mean regret gap is **3×** (0.00144 vs 0.00047): racing's remaining disagreements are on
  near-ties that cost almost nothing, while the grid's include genuinely wrong picks. An agreement
  rate alone cannot tell an allocator that errs on ties from one that errs on decisive turns.

---

## Where the wins come from: samples-to-separation

Rounds at which the field collapsed to a single action, over all 180 decisions:

| rule | never | median | mean | at the floor | distribution |
|---|---|---|---|---|---|
| `seq` (floor 5) | **94 / 180 = 52.2%** | 5 | 9.35 | 47 | 5:47 · 6:7 · 7:4 · 8:2 · 9:2 · 10:1 · 11:2 · 13:3 · 14:3 · 15:2 · 17:2 · 18:1 · 20:1 · 23:2 · 25:1 · 28:2 · 29:1 · 30:1 · 32:2 |
| `z` (floor 3) | 30 / 180 = 16.7% | 3 | 6.03 | 82 | 3:82 · 4:19 · 5:8 · 6:8 · 7:3 · 8:3 · 9:4 · 10:1 · 11:2 · 12:1 · 14:1 · 15:2 · 16:4 · 17:2 · 19:5 · 21:1 · 23:1 · 29:2 · 30:1 |

**The distribution is U-shaped and the middle is nearly empty.** Under either rule the modal
outcome is "separates on the very first round it is permitted to" and the second mode is "never".
Between round 6 and round 32 the two rules together account for ~50 of 360 decision-races. The
practical consequence is that **the floor is the binding parameter, not the threshold** — raising
the floor from 3 to 5 moved the never-rate from 16.7% to 52.2% and the ceiling from 0.933 to 1.000,
while changing `z` from 2.0 to 3.0 at a fixed floor moved the ratios by ~0.03×.

**52% never-separate is the number a time manager should act on.** Those decisions consume the whole
deadline for a result the search itself cannot distinguish from a tie; capping them and returning the
clock to the decisions that *do* separate is a strictly better use of a fixed per-game budget than
either allocator. That is the follow-on this probe found and did not build.

---

## Why the DEFAULT rule was changed from the registered `z` to `seq`

`z` (a one-sided normal test at each look) was the registered rule and it **loses**. Its agreement
with gold **ceilings at 0.933**: it stops so early — median 3 rounds, 46% of separations on the very
first permitted round — that the decisions it gets wrong are settled and abandoned before any
evidence could overturn them, so no amount of extra budget helps. It is visibly flat from 1.28 s
onward (0.928 → 0.922 → 0.933 → 0.933) while the grid climbs past it.

`seq` inflates the elimination radius by a union bound over every look and every comparison. On the
same bank it reaches **1.000**, beats `z` at every quality level from 90% up, and is the only rule
that ever gets to 95% at all. It is also the only one whose stated error rate is a *guarantee*
rather than a per-look figure — see below.

**A default that structurally cannot reach the quality bar is the wrong default however cheap it
is.** `z` stays selectable (`--racing-rule z`) because it is right when a wrong elimination is cheap:
on the *spend* axis at 90% it is the best number in the whole study (2.28×).

### The `seq` floor is enforced by the rule, not left to the caller

The union-bound radius is exact in the *true* standard deviation and plugs in the *sample* one,
which is biased low on a handful of points — so the nominal δ is not delivered at a small floor.
Measured over 600 synthetic races (four arms, true gap 0.02 against a per-round sd of 0.10,
δ = 0.05), the rate at which the true best arm is eliminated:

| floor | 3 | 4 | **5** | 6 | 8 |
|---|---|---|---|---|---|
| false-drop rate | 0.080 | 0.030 | **0.0083** | 0.0050 | 0.0033 |
| power to separate a real gap (0.30) | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |

The floor is **free in power** and 5 is where the measurement first clears δ with margin, so `seq`
raises its own floor to 5 (`SEQ_MIN_SAMPLES`) regardless of what the caller asked for. An error
target that silently depends on another parameter is not an error target.

---

## The parameter sweep (free, on the same bank)

| config | 80%× | 90%× | 95%× | ceiling | never | median | within-race saving |
|---|---|---|---|---|---|---|---|
| `z` z=2.0 floor=3 *(registered)* | 1.14 | 1.38 | — | 0.933 | 0.167 | 3 | 0.239 |
| `z` z=2.0 floor=5 | 1.05 | 1.22 | 1.07 | 0.967 | 0.194 | 5 | 0.215 |
| `z` z=2.0 floor=8 | 1.00 | 1.08 | 1.37 | 0.994 | 0.217 | 8 | 0.179 |
| `z` z=3.0 floor=3 | 1.12 | 1.32 | 0.78 | 0.950 | 0.311 | 4 | 0.317 |
| `z` z=3.0 floor=5 | 1.05 | 1.41 | 1.47 | 0.989 | 0.367 | 5 | 0.290 |
| `z` z=3.0 floor=8 | 1.00 | 1.10 | 1.50 | 1.000 | 0.389 | 8 | 0.253 |
| **`seq` floor=5 (SHIPPED)** | **1.08** | **1.41** | **1.47** | **1.000** | 0.522 | 5 | 0.330 |
| `seq` floor=8 | 1.00 | 1.08 | 1.50 | 1.000 | 0.550 | 8 | 0.294 |
| `z` z=1.5 floor=3 | 1.14 | 1.29 | — | 0.917 | 0.100 | 3 | 0.193 |

(Deadline-axis ratios. Grid ceiling on the same sweep: 0.989.) The frontier is flat — every
configuration that reaches a 1.000 ceiling lands at 1.47–1.50× at 95%, and no setting anywhere in
the grid reaches 2× on the deadline axis. **That flatness is itself the result**: the ceiling is set
by how often gen-3 root decisions are separable at all, not by how the elimination is tuned.

---

## What this does NOT show

* **No live-play result.** Every number here is agreement with a *self-referential* gold — the same
  critic, the same estimator, more samples. It says racing reproduces the grid's own verdict more
  cheaply; it says nothing about whether either verdict wins games. The search-dividend battery is
  the instrument for that and was not re-run.
* **Depth 1 only.** A racing round is depth 1 by construction and the two mechanisms are not
  composed in this build — a first round allowed to deepen would consult the remaining clock and
  swallow the budget the race needs. Whether iterative deepening on a *narrowed* beam beats width is
  a separate and probably more interesting question.
* **One arm, one checkpoint, one opponent tier.** `honest`, `ai_v9_66` at 30 M, the bot roster's
  traces. The oracle arm has one world by construction, so its rounds are dice re-draws and its
  between-round variance is structurally smaller; racing there is a different measurement.
* **The gold carries 6.1% of its own noise** (above). Every absolute agreement number is bounded by
  that; the arm-vs-arm difference is not.

---

## Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
# phase 1 — ~22 min, CPU, one core (the bank is written once)
python -m main.search_dividend.ab_racing models/ai_v9_66_R3F6d_0828 \
  --traces models/ai_v9_66_R3F6d_0828/eval_traces/step_30000000 \
  --decisions 180 --rounds 32 --max-opp 4 --per-battle 3 --seed 7 \
  --bank tmp/racing_bank.json

# phase 2 — seconds, and re-cuttable at any parameter without re-measuring
python -m main.search_dividend.ab_racing --replay tmp/racing_bank.json \
  --out designs/research_state/measurements/racing_root_selection_2026-08-28.json
python -m main.search_dividend.ab_racing --replay tmp/racing_bank.json --racing-rule z \
  --out designs/research_state/measurements/racing_root_selection_2026-08-28_zrule.json
```

Live use (OFF by default; `grid` is byte-identical to the registered battery):

```bash
python -m main.search_dividend <ckpt> --arm honest --root-strategy racing \
  --budget 1 --opponents self --games 30 --out tmp/sd_racing.jsonl
```
